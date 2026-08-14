"""Build a SQLite index from Reddit contribution IDs to sampled authors.

Each JSONL filename identifies its author. The index therefore covers only users in
the collected sample; replies to other users remain unresolved by design.
"""

from __future__ import annotations

import argparse
import json as _std_json
from pathlib import Path
import sqlite3
from typing import Iterator, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it


# orjson is optional but significantly faster for large JSONL collections.
try:
    import orjson

    def _loads(s):
        return orjson.loads(s)
except ImportError:

    def _loads(s):
        return _std_json.loads(s)


DEFAULT_INPUT_DIR = Path("data/raw/user_contributions/user_contributions")
DEFAULT_OUTPUT_DB = Path("data/interim/reddit/author_index.sqlite")

FIELD_ID = "id"
FIELD_NAME = "name"
FIELD_BODY = "body"
FIELD_TITLE = "title"
FIELD_PARENT = "parent_id"

BATCH_SIZE = 50_000


def infer_fullname(record: dict) -> Optional[str]:
    """Return the canonical Reddit fullname, or None when its type is unknown."""
    name = record.get(FIELD_NAME)
    if isinstance(name, str) and (name.startswith("t1_") or name.startswith("t3_")):
        return name

    bare_id = record.get(FIELD_ID)
    if not bare_id:
        return None
    bare_id = str(bare_id)
    if bare_id.startswith("t1_") or bare_id.startswith("t3_"):
        return bare_id

    # Comments have body and parent_id fields; posts have a title.
    if FIELD_BODY in record and FIELD_PARENT in record:
        return f"t1_{bare_id}"
    if FIELD_TITLE in record:
        return f"t3_{bare_id}"
    return None


def iter_user_files(input_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Yield each sampled username and its JSONL path."""
    for path in sorted(input_dir.glob("*.jsonl")):
        yield path.stem, path


def peek_schema(input_dir: Path, n_files: int = 2, n_lines: int = 3) -> None:
    """Print a small schema sample without building the index."""
    print("Schema preview")
    for i, (username, path) in enumerate(iter_user_files(input_dir)):
        if i >= n_files:
            break
        print(f"\nFile: {path.name} (author from filename: {username})")
        with open(path, "r", encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= n_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _loads(line)
                except Exception as e:
                    print(f"  Line {j}: [ERROR] Invalid JSON: {e}")
                    continue
                print(f"  Line {j}: keys={sorted(record.keys())}")
                inferred = infer_fullname(record)
                status = "[OK]" if inferred else "[WARNING] Unrecognized record"
                print(f"    -> inferred fullname: {inferred} {status}")


def build_index(input_dir: Path, output_db: Path, resume: bool = True) -> None:
    """Build or resume the on-disk author index."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    done_path = output_db.with_suffix(".done_users.txt")
    if not resume:
        for path in (
            output_db,
            Path(f"{output_db}-wal"),
            Path(f"{output_db}-shm"),
        ):
            path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)

    conn = sqlite3.connect(output_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS author_index (
            thing_id TEXT PRIMARY KEY,
            author   TEXT NOT NULL
        )
    """)
    conn.commit()

    done_users = set()
    if resume and done_path.exists():
        done_users = set(done_path.read_text(encoding="utf-8").splitlines())
        print(f"Resume: skipping {len(done_users)} indexed users.")

    all_files = list(iter_user_files(input_dir))
    todo = [(u, p) for u, p in all_files if u not in done_users]
    if not all_files:
        conn.close()
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")
    print(f"Files: {len(all_files)} | pending: {len(todo)}")

    n_rows_total = 0
    n_skipped_total = 0
    batch = []

    with open(done_path, "a", encoding="utf-8") as done_fh:
        for username, path in tqdm(todo, desc="Building author index"):
            n_skipped = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = _loads(line)
                    except Exception:
                        n_skipped += 1
                        continue
                    fullname = infer_fullname(record)
                    if fullname is None:
                        n_skipped += 1
                        continue
                    batch.append((fullname, username))
                    if len(batch) >= BATCH_SIZE:
                        conn.executemany(
                            "INSERT OR REPLACE INTO author_index (thing_id, author) VALUES (?, ?)",
                            batch,
                        )
                        conn.commit()
                        n_rows_total += len(batch)
                        batch = []
            # Commit the remainder before checkpointing this user.
            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO author_index (thing_id, author) VALUES (?, ?)",
                    batch,
                )
                conn.commit()
                n_rows_total += len(batch)
                batch = []
            n_skipped_total += n_skipped
            done_fh.write(username + "\n")
            done_fh.flush()

    conn.execute("PRAGMA synchronous = NORMAL")
    conn.close()
    print(f"\nIndexed in this run: {n_rows_total:,} contributions")
    print(f"Skipped rows: {n_skipped_total:,}")
    print(f"Saved to: {output_db}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing user JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--output-db",
        type=Path,
        default=DEFAULT_OUTPUT_DB,
        help=f"SQLite output path (default: {DEFAULT_OUTPUT_DB})",
    )
    ap.add_argument(
        "--peek", action="store_true", help="Print sample records without building the index"
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete the existing index and checkpoint before rebuilding",
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    if args.peek:
        peek_schema(input_dir)
        return

    build_index(input_dir, args.output_db, resume=not args.no_resume)


if __name__ == "__main__":
    main()
