"""Fetch resumable Reddit text, user documents, and scores from Arctic Shift."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.utils.tabular import read_table
from tqdm import tqdm

INPUT_CSV = Path("data/processed/final_intersection_dataset.csv")
OUTPUT_DIR = Path("data/interim/reddit/auxiliary")
FILTERED_VOTES = Path("data/interim/reddit/datasets/filtered_votes.parquet")
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"

API_TIMEOUT_SEC = 30
API_RETRIES = 3
API_BACKOFF_SEC = 2.0
API_SLEEP_SEC = 0.35
POST_CHUNK_SIZE = 25
SCORE_CHUNK_SIZE = 20
N_USER_DOCS = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _chunk(values: List[Any], size: int) -> Iterable[List[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _normalize_payload(payload: Any) -> List[Dict]:
    """Normalize common Arctic Shift response envelopes to records."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "children"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, dict):
                return [value]
            if isinstance(value, list):
                normalized = []
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("data"), dict):
                        normalized.append(item["data"])
                    elif isinstance(item, dict):
                        normalized.append(item)
                return normalized
        return [payload]
    return []


class ArcticShiftClient:
    """Shared Arctic Shift session with retry, backoff, normalization, and pacing."""

    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE,
        timeout: int = API_TIMEOUT_SEC,
        retries: int = API_RETRIES,
        backoff: float = API_BACKOFF_SEC,
        sleep_seconds: float = API_SLEEP_SEC,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        # Shared across sections in one `--section all` run so post texts and
        # Reddit scores never fetch the same item twice.
        self._post_cache: Dict[str, Dict] = {}
        self._missing_post_ids: set[str] = set()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """GET and normalize one endpoint, returning [] after terminal failure."""
        url = f"{self.base_url}{path}"
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        last_error: Optional[Exception] = None
        try:
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.session.get(url, params=clean_params, timeout=self.timeout)
                    response.raise_for_status()
                    return _normalize_payload(response.json())
                except requests.exceptions.HTTPError as exc:
                    last_error = exc
                    if exc.response is not None and 400 <= exc.response.status_code < 500:
                        log.warning("HTTP %s for %s", exc.response.status_code, url)
                        return []
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
            log.warning("All retries failed for %s (%s)", url, last_error)
            return []
        finally:
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

    def fetch_posts_by_ids(self, item_ids: List[str]) -> List[Dict]:
        """Fetch a post chunk, accepting both bare and t3-prefixed endpoint forms."""
        requested = [_strip_prefix(item_id) for item_id in item_ids]
        unresolved = [
            item_id
            for item_id in requested
            if item_id not in self._post_cache and item_id not in self._missing_post_ids
        ]
        for use_prefix in (False, True):
            if not unresolved:
                break
            values = [f"t3_{item_id}" if use_prefix else item_id for item_id in unresolved]
            items = self.get("/api/posts/ids", params={"ids": ",".join(values)})
            for item in items:
                canonical = _canonical_post_id(item)
                if canonical is not None:
                    self._post_cache[_strip_prefix(canonical)] = item
            unresolved = [item_id for item_id in unresolved if item_id not in self._post_cache]
        self._missing_post_ids.update(unresolved)
        return [self._post_cache[item_id] for item_id in requested if item_id in self._post_cache]


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "[removed]", "[deleted]"} else text


def _combine_post_text(item: Dict) -> Optional[str]:
    title = _safe_text(item.get("title"))
    body = _safe_text(item.get("selftext"))
    return f"{title}\n\n{body}" if title and body else title or body


def _strip_prefix(item_id: str) -> str:
    text = str(item_id)
    return text.split("_", 1)[1] if text.startswith(("t1_", "t3_")) else text


def _canonical_post_id(item: Dict, fallback: Optional[str] = None) -> Optional[str]:
    name = item.get("name")
    if isinstance(name, str) and name.startswith("t3_"):
        return name
    raw_id = item.get("id") or fallback
    if raw_id is None:
        return None
    raw_id = str(raw_id)
    return raw_id if raw_id.startswith("t3_") else f"t3_{_strip_prefix(raw_id)}"


def _load_ids(input_csv: Path, column: str) -> List[str]:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    values = read_table(input_csv, columns=[column])[column]
    return values.dropna().astype(str).drop_duplicates().tolist()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Replace one checkpoint only after a complete temporary Parquet write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _merge_checkpoint(
    path: Path,
    rows: List[Dict],
    deduplicate_on: List[str],
) -> pd.DataFrame:
    """Merge buffered rows with an existing checkpoint idempotently."""
    new = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        new = pd.concat([existing, new], ignore_index=True) if not new.empty else existing
    if not new.empty:
        new = new.drop_duplicates(subset=deduplicate_on, keep="last")
        _atomic_parquet(new, path)
    return new


def _post_text_row(item: Dict, fallback_id: Optional[str] = None) -> Dict:
    return {
        "item_id": _canonical_post_id(item, fallback_id),
        "subreddit": item.get("subreddit"),
        "author": item.get("author"),
        "created_utc": item.get("created_utc"),
        "title": _safe_text(item.get("title")),
        "selftext": _safe_text(item.get("selftext")),
        "text": _combine_post_text(item),
        "score": item.get("score", np.nan),
        "url": item.get("url"),
        "is_self": item.get("is_self"),
    }


def fetch_post_texts(
    client: ArcticShiftClient,
    input_csv: Path,
    output_dir: Path,
    checkpoint_chunks: int = 200,
) -> dict:
    """Fetch and checkpoint canonical post text rows for every vote item."""
    output_path = output_dir / "post_texts.parquet"
    all_ids = _load_ids(input_csv, "item_id")
    done = (
        set(pd.read_parquet(output_path, columns=["item_id"])["item_id"].dropna())
        if output_path.exists()
        else set()
    )
    todo = [item_id for item_id in all_ids if item_id not in done]
    rows: List[Dict] = []
    chunks = list(_chunk(todo, POST_CHUNK_SIZE))
    for chunk_index, item_chunk in enumerate(tqdm(chunks, desc="Post texts")):
        items = client.fetch_posts_by_ids(item_chunk)
        by_bare_id = {_strip_prefix(_canonical_post_id(item) or ""): item for item in items}
        for item_id in item_chunk:
            item = by_bare_id.get(_strip_prefix(item_id))
            rows.append(
                _post_text_row(item, item_id) if item is not None else _post_text_row({}, item_id)
            )
        if rows and (chunk_index + 1) % checkpoint_chunks == 0:
            _merge_checkpoint(output_path, rows, ["item_id"])
            rows.clear()
    frame = _merge_checkpoint(output_path, rows, ["item_id"])
    return {
        "section": "post-texts",
        "output": str(output_path),
        "n_requested": len(all_ids),
        "n_rows": len(frame),
        "n_with_text": int(frame["text"].notna().sum()) if not frame.empty else 0,
    }


def _infer_time_window(input_csv: Path) -> Tuple[Optional[str], Optional[str]]:
    timestamps = read_table(input_csv, columns=["timestamp"])["timestamp"]
    numeric = pd.to_numeric(timestamps, errors="coerce")
    parsed = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None, None
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def _collect_user_documents(
    client: ArcticShiftClient,
    username: str,
    n_posts: int,
    n_comments: int,
    after: Optional[str],
    before: Optional[str],
) -> Tuple[List[Dict], Dict]:
    posts = client.get(
        "/api/posts/search",
        params={
            "author": username,
            "after": after,
            "before": before,
            "limit": n_posts,
            "sort": "desc",
        },
    )
    comments = client.get(
        "/api/comments/search",
        params={
            "author": username,
            "after": after,
            "before": before,
            "limit": n_comments,
            "sort": "desc",
        },
    )
    documents: List[Dict] = []
    for item in posts:
        text = _combine_post_text(item)
        if text is not None:
            documents.append(
                {
                    "username": username,
                    "source": "post",
                    "thing_id": item.get("name") or item.get("id"),
                    "subreddit": item.get("subreddit"),
                    "created_utc": item.get("created_utc"),
                    "text": text,
                    "score": item.get("score", np.nan),
                }
            )
    for item in comments:
        text = _safe_text(item.get("body"))
        if text is not None:
            documents.append(
                {
                    "username": username,
                    "source": "comment",
                    "thing_id": item.get("name") or item.get("id"),
                    "subreddit": item.get("subreddit"),
                    "created_utc": item.get("created_utc"),
                    "text": text,
                    "score": item.get("score", np.nan),
                }
            )
    return documents, {
        "username": username,
        "n_posts_fetched": len(posts),
        "n_comments_fetched": len(comments),
        "n_docs_with_text": len(documents),
        "posts_truncated": len(posts) >= n_posts,
        "comments_truncated": len(comments) >= n_comments,
    }


def fetch_user_documents(
    client: ArcticShiftClient,
    input_csv: Path,
    output_dir: Path,
    n_user_docs: int = N_USER_DOCS,
    after: Optional[str] = None,
    before: Optional[str] = None,
    checkpoint_users: int = 500,
) -> dict:
    """Fetch canonical root-level user documents and per-user coverage summaries."""
    documents_path = output_dir / "user_documents.parquet"
    summary_path = output_dir / "user_history_summary.parquet"
    usernames = _load_ids(input_csv, "username")
    done = (
        set(pd.read_parquet(summary_path, columns=["username"])["username"].dropna())
        if summary_path.exists()
        else set()
    )
    todo = [username for username in usernames if username not in done]
    if after is None and before is None:
        after, before = _infer_time_window(input_csv)
    n_posts = n_user_docs // 2
    n_comments = n_user_docs - n_posts
    document_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for user_index, username in enumerate(tqdm(todo, desc="User documents")):
        documents, summary = _collect_user_documents(
            client, username, n_posts, n_comments, after, before
        )
        document_rows.extend(documents)
        summary_rows.append(summary)
        if (user_index + 1) % checkpoint_users == 0:
            _merge_checkpoint(documents_path, document_rows, ["username", "thing_id"])
            _merge_checkpoint(summary_path, summary_rows, ["username"])
            document_rows.clear()
            summary_rows.clear()
    documents = _merge_checkpoint(documents_path, document_rows, ["username", "thing_id"])
    summaries = _merge_checkpoint(summary_path, summary_rows, ["username"])
    return {
        "section": "user-documents",
        "outputs": [str(documents_path), str(summary_path)],
        "n_requested_users": len(usernames),
        "n_users": len(summaries),
        "n_documents": len(documents),
        "after": after,
        "before": before,
    }


def _score_row(item: Dict, fallback_id: Optional[str] = None) -> Dict:
    created_utc = item.get("created_utc")
    try:
        created_dt = pd.to_datetime(float(created_utc), unit="s", utc=True)
    except (TypeError, ValueError, OverflowError):
        created_dt = pd.NaT
    title = _safe_text(item.get("title"))
    body = _safe_text(item.get("selftext"))
    return {
        "item_id": _canonical_post_id(item, fallback_id),
        "author": item.get("author"),
        "subreddit": item.get("subreddit"),
        "title": title,
        "selftext": body,
        "full_text": f"{title}\n\n{body}" if title and body else title or body,
        "url": item.get("url"),
        "domain": item.get("domain"),
        "score": item.get("score", np.nan),
        "upvote_ratio": item.get("upvote_ratio", np.nan),
        "num_comments": item.get("num_comments", np.nan),
        "created_utc": created_utc,
        "created_dt": created_dt,
        "removed_by_category": item.get("removed_by_category"),
        "removal_reason": item.get("removal_reason"),
        "banned_by": item.get("banned_by"),
        "locked": item.get("locked", False),
        "over_18": item.get("over_18", False),
        "is_self": item.get("is_self"),
        "link_flair_text": item.get("link_flair_text"),
        "author_flair_text": item.get("author_flair_text"),
        "permalink": item.get("permalink"),
    }


def fetch_reddit_scores(
    client: ArcticShiftClient,
    votes_path: Path,
    output_dir: Path,
    checkpoint_chunks: int = 200,
) -> dict:
    """Fetch Reddit score metadata for each unique moderated post."""
    if not votes_path.exists():
        raise FileNotFoundError(votes_path)
    votes = read_table(votes_path, columns=["item_id", "community", "label"])
    moderated = votes.drop_duplicates("item_id").reset_index(drop=True)
    output_path = output_dir / "moderated_posts_scores.parquet"
    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    done = set(existing["item_id"].dropna()) if not existing.empty else set()
    todo = moderated[~moderated["item_id"].isin(done)].copy()
    rows: List[Dict] = []
    chunks = list(_chunk(todo["item_id"].tolist(), SCORE_CHUNK_SIZE))
    for chunk_index, item_chunk in enumerate(tqdm(chunks, desc="Reddit scores")):
        items = client.fetch_posts_by_ids(item_chunk)
        by_bare_id = {_strip_prefix(_canonical_post_id(item) or ""): item for item in items}
        for item_id in item_chunk:
            item = by_bare_id.get(_strip_prefix(item_id))
            rows.append(_score_row(item or {}, item_id))
        if rows and (chunk_index + 1) % checkpoint_chunks == 0:
            fetched = pd.DataFrame(rows)
            fetched = todo[["item_id", "community", "label"]].merge(
                fetched, on="item_id", how="inner"
            )
            _merge_checkpoint(output_path, fetched.to_dict("records"), ["item_id"])
            rows.clear()
    if rows:
        fetched = pd.DataFrame(rows)
        fetched = todo[["item_id", "community", "label"]].merge(fetched, on="item_id", how="inner")
        result = _merge_checkpoint(output_path, fetched.to_dict("records"), ["item_id"])
    else:
        result = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    return {
        "section": "reddit-scores",
        "output": str(output_path),
        "n_requested": len(moderated),
        "n_rows": len(result),
        "n_with_score": int(result["score"].notna().sum()) if not result.empty else 0,
    }


def _write_manifest(
    output_dir: Path, section: str, reports: List[dict], client: ArcticShiftClient
) -> None:
    manifest = {
        "pipeline": "fetch_reddit_auxiliary_data",
        "section": section,
        "source": client.base_url,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retry_policy": {
            "timeout_seconds": client.timeout,
            "retries": client.retries,
            "backoff_seconds": client.backoff,
            "request_sleep_seconds": client.sleep_seconds,
        },
        "reports": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fetch_reddit_auxiliary_data.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["post-texts", "user-documents", "reddit-scores", "all"],
        default="all",
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--votes", type=Path, default=FILTERED_VOTES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-user-docs", type=int, default=N_USER_DOCS)
    parser.add_argument("--after", default=None)
    parser.add_argument("--before", default=None)
    parser.add_argument("--base-url", default=ARCTIC_SHIFT_BASE)
    parser.add_argument("--request-sleep", type=float, default=API_SLEEP_SEC)
    args = parser.parse_args()

    client = ArcticShiftClient(base_url=args.base_url, sleep_seconds=args.request_sleep)
    reports = []
    if args.section in {"post-texts", "all"}:
        reports.append(fetch_post_texts(client, args.input_csv, args.output_dir))
    if args.section in {"user-documents", "all"}:
        reports.append(
            fetch_user_documents(
                client,
                args.input_csv,
                args.output_dir,
                n_user_docs=args.n_user_docs,
                after=args.after,
                before=args.before,
            )
        )
    if args.section in {"reddit-scores", "all"}:
        reports.append(fetch_reddit_scores(client, args.votes, args.output_dir))
    _write_manifest(args.output_dir, args.section, reports, client)
    for report in reports:
        log.info("%s", report)


if __name__ == "__main__":
    main()
