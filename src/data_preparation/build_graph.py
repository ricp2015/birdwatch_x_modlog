"""Build monthly author snapshots and a reply graph from Reddit JSONL files.

Each user file is sorted and replayed independently. Snapshots contain cumulative
values through the active month and must be joined downstream with an as-of lookup.
Contribution scores are collection-time values, not historical scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json as _std_json
import math
from pathlib import Path
import shutil
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it


try:
    import orjson

    def _loads(s):
        return orjson.loads(s)
except ImportError:

    def _loads(s):
        return _std_json.loads(s)


DEFAULT_INPUT_DIR = Path("data/raw/user_contributions/user_contributions")
DEFAULT_AUTHOR_INDEX = Path("data/interim/reddit/author_index.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/interim/reddit/user_temporal_features")

# Keep these fields aligned with build_author_index.py.
FIELD_ID = "id"
FIELD_NAME = "name"
FIELD_BODY = "body"
FIELD_TITLE = "title"
FIELD_PARENT = "parent_id"
FIELD_SUBREDDIT = "subreddit"
FIELD_CREATED_UTC = "created_utc"
FIELD_SCORE = "score"

USERS_PER_PART = 200
SQLITE_IN_CHUNK = 900


def infer_type_and_fullname(record: dict) -> Tuple[Optional[str], Optional[str]]:
    """Return the contribution type and canonical Reddit fullname."""
    name = record.get(FIELD_NAME)
    if isinstance(name, str) and name.startswith("t3_"):
        return "post", name
    if isinstance(name, str) and name.startswith("t1_"):
        return "comment", name

    bare_id = record.get(FIELD_ID)
    if not bare_id:
        return None, None
    bare_id = str(bare_id)
    if FIELD_BODY in record and FIELD_PARENT in record:
        fullname = bare_id if bare_id.startswith("t1_") else f"t1_{bare_id}"
        return "comment", fullname
    if FIELD_TITLE in record:
        fullname = bare_id if bare_id.startswith("t3_") else f"t3_{bare_id}"
        return "post", fullname
    return None, None


def year_month(ts: float) -> str:
    """Convert a Unix timestamp to a UTC year-month value."""
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def load_and_sort_user_contributions(path: Path) -> List[dict]:
    """Load and chronologically sort one user's valid contributions."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = _loads(line)
            except Exception:
                continue
            ctype, fullname = infer_type_and_fullname(record)
            if ctype is None or fullname is None:
                continue
            ts = record.get(FIELD_CREATED_UTC)
            if ts is None:
                continue
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ts):
                continue
            try:
                year_month(ts)
            except (OverflowError, OSError, ValueError):
                continue
            try:
                score = float(record.get(FIELD_SCORE, 0) or 0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            rows.append(
                {
                    "type": ctype,
                    "fullname": fullname,
                    "subreddit": record.get(FIELD_SUBREDDIT) or "__unknown__",
                    "created_utc": ts,
                    "score": score,
                    "parent_id": record.get(FIELD_PARENT),
                }
            )
    rows.sort(key=lambda r: r["created_utc"])
    return rows


def batch_lookup_authors(
    parent_ids: Set[str], author_index: sqlite3.Connection, chunk_size: int = SQLITE_IN_CHUNK
) -> Dict[str, str]:
    """Resolve parent authors in batches to avoid one query per comment."""
    result: Dict[str, str] = {}
    ids = list(parent_ids)
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        cursor = author_index.execute(
            f"SELECT thing_id, author FROM author_index WHERE thing_id IN ({placeholders})",
            chunk,
        )
        for thing_id, author in cursor:
            result[thing_id] = author
    return result


GLOBAL_SCHEMA = pa.schema(
    [
        ("username", pa.string()),
        ("year_month", pa.string()),
        ("cum_n_posts", pa.int64()),
        ("cum_n_comments", pa.int64()),
        ("cum_n_contributions", pa.int64()),
        ("cum_karma_sum", pa.float64()),
        ("cum_n_distinct_subreddits", pa.int64()),
        ("cum_n_replies_made", pa.int64()),
    ]
)

SUBREDDIT_SCHEMA = pa.schema(
    [
        ("username", pa.string()),
        ("subreddit", pa.string()),
        ("year_month", pa.string()),
        ("cum_n_posts_in_sub", pa.int64()),
        ("cum_n_comments_in_sub", pa.int64()),
        ("cum_karma_sum_in_sub", pa.float64()),
        ("cum_n_months_active_in_sub", pa.int64()),
    ]
)

REPLY_SCHEMA = pa.schema(
    [
        ("from_user", pa.string()),
        ("to_user", pa.string()),
        ("thing_id", pa.string()),
        ("parent_id", pa.string()),
        ("subreddit", pa.string()),
        ("created_utc", pa.float64()),
    ]
)


class PartitionedParquetWriter:
    """Write buffered rows as independently readable Parquet parts."""

    def __init__(self, out_dir: Path, schema: pa.Schema, start_part: int = 0):
        self.out_dir = out_dir
        self.schema = schema
        self.part_idx = start_part
        self.buffer: List[dict] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_rows(self, rows: List[dict]) -> None:
        self.buffer.extend(rows)

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        part_path = self.out_dir / f"part_{self.part_idx:06d}.parquet"
        pq.write_table(table, part_path)
        self.part_idx += 1
        self.buffer = []

    def close(self) -> None:
        self.flush()


def _next_part_index(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    existing = sorted(out_dir.glob("part_*.parquet"))
    if not existing:
        return 0
    last = existing[-1].stem
    return int(last.split("_")[-1]) + 1


def process_one_user(
    username: str,
    contributions: List[dict],
    author_lookup: Dict[str, str],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Replay one user and return global, community, and reply rows."""
    global_rows: List[dict] = []
    subreddit_rows: List[dict] = []
    reply_rows: List[dict] = []

    cum_n_posts = 0
    cum_n_comments = 0
    cum_karma_sum = 0.0
    cum_n_replies_made = 0
    subs_seen: Set[str] = set()
    sub_state: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n_posts": 0, "n_comments": 0, "karma_sum": 0.0, "n_months_active": 0}
    )

    current_month: Optional[str] = None
    subs_touched_this_month: Set[str] = set()

    def flush_month(month: Optional[str]) -> None:
        """Store cumulative state for the completed active month."""
        if month is None:
            return
        global_rows.append(
            {
                "username": username,
                "year_month": month,
                "cum_n_posts": cum_n_posts,
                "cum_n_comments": cum_n_comments,
                "cum_n_contributions": cum_n_posts + cum_n_comments,
                "cum_karma_sum": cum_karma_sum,
                "cum_n_distinct_subreddits": len(subs_seen),
                "cum_n_replies_made": cum_n_replies_made,
            }
        )
        for sub in subs_touched_this_month:
            st = sub_state[sub]
            subreddit_rows.append(
                {
                    "username": username,
                    "subreddit": sub,
                    "year_month": month,
                    "cum_n_posts_in_sub": int(st["n_posts"]),
                    "cum_n_comments_in_sub": int(st["n_comments"]),
                    "cum_karma_sum_in_sub": st["karma_sum"],
                    "cum_n_months_active_in_sub": int(st["n_months_active"]),
                }
            )

    for contrib in contributions:
        month = year_month(contrib["created_utc"])
        if current_month is None:
            current_month = month
        elif month != current_month:
            flush_month(current_month)
            subs_touched_this_month = set()
            current_month = month

        sub = contrib["subreddit"]
        score = float(contrib["score"] or 0)

        if contrib["type"] == "post":
            cum_n_posts += 1
            sub_state[sub]["n_posts"] += 1
        else:
            cum_n_comments += 1
            sub_state[sub]["n_comments"] += 1

        cum_karma_sum += score
        sub_state[sub]["karma_sum"] += score

        if sub not in subs_touched_this_month:
            subs_touched_this_month.add(sub)
            sub_state[sub]["n_months_active"] += 1
        subs_seen.add(sub)

        # Keep replies to other sampled users only.
        if contrib["type"] == "comment" and contrib["parent_id"]:
            to_user = author_lookup.get(contrib["parent_id"])
            if to_user is not None and to_user != username:
                cum_n_replies_made += 1
                reply_rows.append(
                    {
                        "from_user": username,
                        "to_user": to_user,
                        "thing_id": contrib["fullname"],
                        "parent_id": contrib["parent_id"],
                        "subreddit": sub,
                        "created_utc": contrib["created_utc"],
                    }
                )

    flush_month(current_month)
    return global_rows, subreddit_rows, reply_rows


def run(input_dir: Path, author_index_path: Path, output_dir: Path, resume: bool = True) -> None:
    """Build or resume all temporal feature datasets."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not author_index_path.is_file():
        raise FileNotFoundError(f"Author index not found: {author_index_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    global_dir = output_dir / "user_global_snapshots"
    sub_dir = output_dir / "user_subreddit_snapshots"
    reply_dir = output_dir / "reply_graph"
    progress_path = output_dir / "_progress_users_done.txt"

    if not resume:
        for path in (global_dir, sub_dir, reply_dir):
            if path.exists():
                shutil.rmtree(path)
        progress_path.unlink(missing_ok=True)
    elif progress_path.exists() and not all(
        path.is_dir() for path in (global_dir, sub_dir, reply_dir)
    ):
        raise RuntimeError(f"Incomplete resume state in {output_dir}; run again with --no-resume")
    elif not progress_path.exists() and any(
        next(path.glob("part_*.parquet"), None) is not None
        for path in (global_dir, sub_dir, reply_dir)
        if path.is_dir()
    ):
        raise RuntimeError(
            f"Parquet parts exist without a checkpoint in {output_dir}; run again with --no-resume"
        )

    author_index = sqlite3.connect(author_index_path)
    author_index.execute("PRAGMA query_only = ON")
    table_exists = author_index.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='author_index'"
    ).fetchone()
    if table_exists is None:
        author_index.close()
        raise ValueError(f"Missing author_index table in {author_index_path}")

    global_writer = PartitionedParquetWriter(
        global_dir, GLOBAL_SCHEMA, _next_part_index(global_dir)
    )
    sub_writer = PartitionedParquetWriter(sub_dir, SUBREDDIT_SCHEMA, _next_part_index(sub_dir))
    reply_writer = PartitionedParquetWriter(reply_dir, REPLY_SCHEMA, _next_part_index(reply_dir))

    done_users: Set[str] = set()
    if resume and progress_path.exists():
        done_users = set(progress_path.read_text(encoding="utf-8").splitlines())
        print(f"Resume: skipping {len(done_users)} processed users.")

    all_files = sorted(input_dir.glob("*.jsonl"))
    todo = [p for p in all_files if p.stem not in done_users]
    if not all_files:
        author_index.close()
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")
    print(f"Users: {len(all_files)} | pending: {len(todo)}")

    pending_users: List[str] = []
    with open(progress_path, "a", encoding="utf-8") as progress_fh:

        def flush_part() -> None:
            """Write buffered datasets before recording their completed users."""
            global_writer.flush()
            sub_writer.flush()
            reply_writer.flush()
            if pending_users:
                progress_fh.write("\n".join(pending_users) + "\n")
                progress_fh.flush()
                pending_users.clear()

        for i, path in enumerate(tqdm(todo, desc="Replaying contributions")):
            username = path.stem
            contributions = load_and_sort_user_contributions(path)
            if contributions:
                parent_ids_needed = {
                    c["parent_id"]
                    for c in contributions
                    if c["type"] == "comment" and c["parent_id"]
                }
                author_lookup = (
                    batch_lookup_authors(parent_ids_needed, author_index)
                    if parent_ids_needed
                    else {}
                )

                g_rows, s_rows, r_rows = process_one_user(username, contributions, author_lookup)
                global_writer.add_rows(g_rows)
                sub_writer.add_rows(s_rows)
                reply_writer.add_rows(r_rows)

            pending_users.append(username)

            if (i + 1) % USERS_PER_PART == 0:
                flush_part()

        flush_part()

    global_writer.close()
    sub_writer.close()
    reply_writer.close()
    author_index.close()

    print(f"\nCompleted. Output: {output_dir}")
    print(f"  user_global_snapshots/     ({global_writer.part_idx} parts)")
    print(f"  user_subreddit_snapshots/  ({sub_writer.part_idx} parts)")
    print(f"  reply_graph/               ({reply_writer.part_idx} parts)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing user JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--author-index",
        type=Path,
        default=DEFAULT_AUTHOR_INDEX,
        help=f"Author index path (default: {DEFAULT_AUTHOR_INDEX})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parquet output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--no-resume", action="store_true", help="Delete existing parts and rebuild from the start"
    )
    args = ap.parse_args()

    run(
        input_dir=args.input_dir,
        author_index_path=args.author_index,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
