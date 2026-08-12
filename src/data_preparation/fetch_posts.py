"""
Fetches two things needed for the expert-finding pipeline:

  1. title + body for every post already in the vote dataset (identified by item_id, e.g. "t3_dxlv9b").
  Output: data/interim/reddit/post_texts.parquet

  2. USER HISTORY - up to N_USER_DOCS (default 150) posts + comments per user, used to build embeddings.
  Output: data/interim/reddit/user_documents.parquet    data/interim/reddit/user_history_summary.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Paths
STEP1_DIR   = Path("data/interim/reddit")
OUTPUT_DIR  = Path("data/interim/reddit")
SPLITS_DIR   = STEP1_DIR / "splits"
ORIGINAL_CSV = Path("data/processed/final_intersection_dataset.csv")  # full pre-filter dataset

# API settings  (mirrors step1 fetcher)
ARCTIC_SHIFT_BASE   = "https://arctic-shift.photon-reddit.com"
API_TIMEOUT_SEC     = 30
API_RETRIES         = 3
API_BACKOFF_SEC     = 2.0
API_SLEEP_SEC       = 0.35          # polite delay between requests

# Collection limits
POST_CHUNK_SIZE  = 25               # IDs per bulk-post request (safe limit)
N_USER_DOCS      = 150              # total posts+comments per user (param)
N_USER_POSTS     = 75               # half posts ...
N_USER_COMMENTS  = 75              # ... half comments  (adjusted if N_USER_DOCS changes)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Shared helpers  (same pattern as step1)

def _chunk(lst: List[Any], size: int):
    """Yield fixed-size chunks from a sequence."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _safe_request(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = API_TIMEOUT_SEC,
) -> Any:
    """Run request with retry and error handling."""
    url = f"{ARCTIC_SHIFT_BASE}{path}"
    params = {k: v for k, v in (params or {}).items() if v is not None}
    last_exc: Optional[Exception] = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                log.warning("4xx for %s - skipping (%s)", url, exc)
                return None
            last_exc = exc
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < API_RETRIES:
            wait = API_BACKOFF_SEC * attempt
            log.debug("Retry %d/%d in %.1fs for %s", attempt, API_RETRIES, wait, url)
            time.sleep(wait)
    log.warning("All retries failed for %s (%s)", url, last_exc)
    return None


def _normalize_payload(payload: Any) -> List[Dict]:
    """Normalize payload into a consistent schema."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "children"):
            if key in payload:
                val = payload[key]
                if isinstance(val, list):
                    out = []
                    for item in val:
                        if isinstance(item, dict) and "data" in item and isinstance(item["data"], dict):
                            out.append(item["data"])
                        elif isinstance(item, dict):
                            out.append(item)
                    return out
                if isinstance(val, dict):
                    return [val]
        return [payload]
    return []


def _safe_text(value: Any) -> Optional[str]:
    """Run text with retry and error handling."""
    if value is None:
        return None
    s = str(value).strip()
    # Reddit marks removed/deleted posts with these placeholders
    if s.lower() in {"[removed]", "[deleted]", ""}:
        return None
    return s


def _combine_post_text(item: Dict) -> Optional[str]:
    """Join a post title and body into one text field."""
    title    = _safe_text(item.get("title"))
    selftext = _safe_text(item.get("selftext"))
    if title and selftext:
        return f"{title}\n\n{selftext}"
    return title or selftext


def _strip_prefix(item_id: str) -> str:
    """Remove prefix from the supplied value."""
    return item_id.split("_", 1)[-1] if "_" in item_id else item_id


# 1.  POST TEXTS  (fetch by item_id from the vote dataset)

def _load_all_item_ids(input_csv: Optional[Path] = None) -> List[str]:
    """Load all item ids from its configured source."""
    for csv_path in [p for p in [input_csv, ORIGINAL_CSV] if p is not None]:
        if csv_path.exists():
            ids = (
                pd.read_csv(csv_path, usecols=["item_id"], low_memory=False)
                ["item_id"].dropna().unique().tolist()
            )
            log.info(
                "Loaded item_ids from %s: %d unique posts (full pre-filter dataset)",
                csv_path, len(ids),
            )
            return ids

    # fallback: density-filtered parquet
    filtered_path = STEP1_DIR / "filtered_votes.parquet"
    if filtered_path.exists():
        ids = pd.read_parquet(filtered_path, columns=["item_id"])["item_id"].unique().tolist()
        log.warning(
            "Falling back to filtered_votes.parquet (%d posts). "
            "Pass --input-csv to use the full ~73k dataset instead.", len(ids),
        )
        return ids

    # absolute last resort: split files
    log.warning("No CSV found - reading from split parquets only (~5k posts).")
    ids_set: set[str] = set()
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}_votes.parquet"
        if p.exists():
            ids_set.update(pd.read_parquet(p, columns=["item_id"])["item_id"].tolist())
    log.info("Total unique item_ids from splits: %d", len(ids_set))
    return list(ids_set)


def _load_done_post_ids(out_path: Path) -> set[str]:
    """Load done post ids from its configured source."""
    if out_path.exists():
        done = set(pd.read_parquet(out_path, columns=["item_id"])["item_id"].tolist())
        log.info("Resuming post-text fetch: %d already done", len(done))
        return done
    return set()


def _fetch_post_chunk(bare_ids: List[str]) -> List[Dict]:
    """Fetch post chunk from the remote API."""
    for id_list in (
        ",".join(bare_ids),                          # bare:  dxlv9b,abc123
        ",".join(f"t3_{i}" for i in bare_ids),      # full:  t3_dxlv9b,t3_abc123
    ):
        payload = _safe_request(
            "/api/posts/ids",                        # correct bulk-by-id endpoint
            params={"ids": id_list},                 # no 'limit' - IDs are explicit
        )
        items = _normalize_payload(payload)
        if items:
            return items
    return []


def fetch_post_texts(output_dir: Path, input_csv: Optional[Path] = None) -> pd.DataFrame:
    """Fetch post texts from the remote API."""
    out_path = output_dir / "post_texts.parquet"
    all_ids  = _load_all_item_ids(input_csv)
    done_ids = _load_done_post_ids(out_path)
    todo     = [i for i in all_ids if i not in done_ids]

    log.info("Post-text fetch: %d remaining / %d total", len(todo), len(all_ids))
    if not todo:
        log.info("Nothing to fetch - loading existing file.")
        return pd.read_parquet(out_path)

    rows: List[Dict] = []
    bare_todo = [_strip_prefix(i) for i in todo]
    # keep a map bare_id -> full item_id for the output
    bare_to_full = {_strip_prefix(i): i for i in todo}

    # helper: strip t3_ prefix from whatever the API returns in 'id'
    def _bare(raw_id: str) -> str:
        """Remove the Reddit type prefix from an item ID."""
        return raw_id.split("_", 1)[-1] if "_" in raw_id else raw_id

    chunks = list(_chunk(bare_todo, POST_CHUNK_SIZE))
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Fetching post texts")):
        items = _fetch_post_chunk(chunk)
        fetched_bare = {_bare(str(it.get("id", ""))) for it in items}

        for item in items:
            bare_id = _bare(str(item.get("id", "")))
            full_id = bare_to_full.get(bare_id, f"t3_{bare_id}")
            text    = _combine_post_text(item)
            # skip posts with no usable text (image posts, link posts, removed)
            if text is None:
                continue
            rows.append({
                "item_id":    full_id,
                "subreddit":  item.get("subreddit"),
                "author":     item.get("author"),
                "created_utc": item.get("created_utc"),
                "title":      _safe_text(item.get("title")),
                "selftext":   _safe_text(item.get("selftext")),
                "text":       text,
                "score":      item.get("score", np.nan),
                "url":        item.get("url"),
                "is_self":    item.get("is_self", None),
            })

        # posts missing from the API response -> record as null so we don't re-fetch
        for bare_id in chunk:
            if bare_id not in fetched_bare:
                full_id = bare_to_full.get(bare_id, f"t3_{bare_id}")
                rows.append({
                    "item_id": full_id,
                    "subreddit": None, "author": None, "created_utc": None,
                    "title": None, "selftext": None, "text": None,
                    "score": np.nan, "url": None, "is_self": None,
                })

        time.sleep(API_SLEEP_SEC)

        # incremental checkpoint every 200 chunks
        if rows and (chunk_idx + 1) % 200 == 0:
            _checkpoint_post_texts(out_path, done_ids, rows)

    # final save
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_parquet(out_path, index=False)

    n_with_text = new_df["text"].notna().sum()
    log.info(
        "Post-text fetch complete: %d / %d posts have usable text (%.1f%%)",
        n_with_text, len(new_df), 100 * n_with_text / max(len(new_df), 1),
    )
    return new_df


def _checkpoint_post_texts(out_path: Path, done_ids: set, rows: List[Dict]) -> None:
    """Write post texts checkpoint to disk."""
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["item_id"])
    new_df.to_parquet(out_path, index=False)
    log.info("  Checkpoint: %d post-text rows saved", len(new_df))


# 2.  USER HISTORY  (posts + comments per user, for Segnale B embeddings)

def _load_done_usernames(output_dir: Path) -> set[str]:
    """Load done usernames from its configured source."""
    summary_path = output_dir / "user_history_summary.parquet"
    if summary_path.exists():
        done = set(pd.read_parquet(summary_path, columns=["username"])["username"].tolist())
        log.info("Resuming user-history fetch: %d users already done", len(done))
        return done
    return set()


def _load_all_usernames(input_csv: Optional[Path] = None) -> List[str]:
    """Load all usernames from its configured source."""
    for csv_path in [p for p in [input_csv, ORIGINAL_CSV] if p is not None]:
        if csv_path.exists():
            users = (
                pd.read_csv(csv_path, usecols=["username"], low_memory=False)
                ["username"].dropna().unique().tolist()
            )
            log.info(
                "Loaded usernames from %s: %d unique users (full pre-filter dataset)",
                csv_path, len(users),
            )
            return users

    # fallback: filtered parquet
    filtered_path = STEP1_DIR / "filtered_votes.parquet"
    if filtered_path.exists():
        users = pd.read_parquet(filtered_path, columns=["username"])["username"].unique().tolist()
        log.warning(
            "Falling back to filtered_votes.parquet (%d users). "
            "Pass --input-csv to use the full dataset instead.", len(users),
        )
        return users

    # last resort: splits
    log.warning("No CSV found - reading usernames from split parquets only.")
    users_set: set[str] = set()
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}_votes.parquet"
        if p.exists():
            users_set.update(pd.read_parquet(p, columns=["username"])["username"].tolist())
    log.info("Total unique users from splits: %d", len(users_set))
    return list(users_set)


def _collect_one_user(
    uname: str,
    n_posts: int,
    n_comments: int,
    after: Optional[str],
    before: Optional[str],
) -> Tuple[List[Dict], Dict]:
    """Collect one user from the available records."""
    docs: List[Dict] = []

    # posts
    posts_payload = _safe_request(
        "/api/posts/search",
        params={
            "author": uname,
            "after":  after,
            "before": before,
            "limit":  n_posts,
            "sort":   "desc",       # most recent first
        },
    )
    posts = _normalize_payload(posts_payload)
    for item in posts:
        text = _combine_post_text(item)
        if text is None:
            continue
        docs.append({
            "username":    uname,
            "source":      "post",
            "thing_id":    item.get("id") or item.get("name"),
            "subreddit":   item.get("subreddit"),
            "created_utc": item.get("created_utc"),
            "text":        text,
            "score":       item.get("score", np.nan),
        })

    time.sleep(API_SLEEP_SEC)

    # comments
    comments_payload = _safe_request(
        "/api/comments/search",
        params={
            "author": uname,
            "after":  after,
            "before": before,
            "limit":  n_comments,
            "sort":   "desc",
        },
    )
    comments = _normalize_payload(comments_payload)
    for item in comments:
        text = _safe_text(item.get("body"))
        if text is None:
            continue
        docs.append({
            "username":    uname,
            "source":      "comment",
            "thing_id":    item.get("id") or item.get("name"),
            "subreddit":   item.get("subreddit"),
            "created_utc": item.get("created_utc"),
            "text":        text,
            "score":       item.get("score", np.nan),
        })

    summary = {
        "username":           uname,
        "n_posts_fetched":    len(posts),
        "n_comments_fetched": len(comments),
        "n_docs_with_text":   len(docs),
        "posts_truncated":    len(posts)  >= n_posts,
        "comments_truncated": len(comments) >= n_comments,
    }
    return docs, summary


def fetch_user_history(
    output_dir: Path,
    n_user_docs: int,
    after: Optional[str] = None,
    before: Optional[str] = None,
    input_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch user history from the remote API."""
    docs_path    = output_dir / "user_documents.parquet"
    summary_path = output_dir / "user_history_summary.parquet"

    all_users  = _load_all_usernames(input_csv)
    done_users = _load_done_usernames(output_dir)
    todo       = [u for u in all_users if u not in done_users]

    log.info("User-history fetch: %d remaining / %d total", len(todo), len(all_users))

    # split the budget evenly between posts and comments
    n_posts    = n_user_docs // 2
    n_comments = n_user_docs - n_posts

    new_docs:     List[Dict] = []
    new_summaries: List[Dict] = []

    for idx, uname in enumerate(tqdm(todo, desc="User history")):
        docs, summary = _collect_one_user(
            uname, n_posts=n_posts, n_comments=n_comments,
            after=after, before=before,
        )
        new_docs.extend(docs)
        new_summaries.append(summary)
        time.sleep(API_SLEEP_SEC)

        # checkpoint every 500 users so a crash doesn't lose everything
        if (idx + 1) % 500 == 0:
            _checkpoint_user_history(docs_path, summary_path, new_docs, new_summaries)
            new_docs, new_summaries = [], []   # reset buffer after save
            log.info("  Checkpoint at user %d / %d", idx + 1, len(todo))

    # final save of the remaining buffer
    _checkpoint_user_history(docs_path, summary_path, new_docs, new_summaries)

    docs_df    = pd.read_parquet(docs_path)    if docs_path.exists()    else pd.DataFrame()
    summary_df = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()

    log.info(
        "User-history fetch complete: %d docs | %d users",
        len(docs_df), len(summary_df),
    )
    return docs_df, summary_df


def _checkpoint_user_history(
    docs_path:    Path,
    summary_path: Path,
    new_docs:     List[Dict],
    new_summaries: List[Dict],
) -> None:
    """Write user history checkpoint to disk."""
    for path, rows in [(docs_path, new_docs), (summary_path, new_summaries)]:
        if not rows:
            continue
        new_df = pd.DataFrame(rows)
        if path.exists():
            existing = pd.read_parquet(path)
            new_df   = pd.concat([existing, new_df], ignore_index=True)
        # deduplicate on natural keys so re-runs are idempotent
        dedup_col = "username" if "thing_id" not in new_df.columns else None
        if dedup_col is None:
            new_df = new_df.drop_duplicates(subset=["username", "thing_id"])
        else:
            new_df = new_df.drop_duplicates(subset=[dedup_col])
        new_df.to_parquet(path, index=False)


# 3.  Infer time window from the vote dataset

def _infer_time_window() -> Tuple[Optional[str], Optional[str]]:
    """Infer time window from the available timestamps."""
    dfs = []
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}_votes.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p, columns=["timestamp"]))
    if not dfs:
        return None, None
    ts = pd.concat(dfs)["timestamp"]
    # timestamps may be numeric (unix) or already datetime
    if pd.api.types.is_numeric_dtype(ts):
        ts = pd.to_datetime(ts, unit="s", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    after  = ts.min().date().isoformat()
    before = ts.max().date().isoformat()
    log.info("Dataset time window: %s -> %s", after, before)
    return after, before


# Entry point

def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser(description="Fetch data for expert-finding pipeline")
    parser.add_argument("--posts-only",  action="store_true", help="Only fetch post texts")
    parser.add_argument("--users-only",  action="store_true", help="Only fetch user history")
    parser.add_argument(
        "--n-user-docs", type=int, default=N_USER_DOCS,
        help=f"Max posts+comments per user (default {N_USER_DOCS})",
    )
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="Path to the original intersection CSV (default: data/processed/final_intersection_dataset.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    after, before = _infer_time_window()

    do_posts = not args.users_only
    do_users = not args.posts_only

    if do_posts:
        log.info("=" * 50)
        log.info("FETCH 1/2 - Post texts")
        log.info("=" * 50)
        post_df = fetch_post_texts(output_dir, input_csv=args.input_csv)
        n_ok = post_df["text"].notna().sum()
        log.info("Posts with text: %d / %d", n_ok, len(post_df))

    if do_users:
        log.info("=" * 50)
        log.info("FETCH 2/2 - User history (n_docs=%d)", args.n_user_docs)
        log.info("=" * 50)
        docs_df, summary_df = fetch_user_history(
            output_dir,
            n_user_docs=args.n_user_docs,
            after=after,
            before=before,
            input_csv=args.input_csv,
        )
        log.info(
            "User docs: %d total | %d users covered",
            len(docs_df), len(summary_df),
        )

    log.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
