from pathlib import Path
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

INPUT_PATH = Path("data/processed/final_intersection_dataset.csv")
OUTPUT_DIR = Path("results/step1/reddit/")

# birdwatch thresholds
MIN_VOTES_PER_POST = 5
MIN_VOTES_PER_USER = 10
# chronological split ratios (sum to 1)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# API docs: https://github.com/ArthurHeitmann/arctic_shift/tree/master/api
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"
API_TIMEOUT_SEC = 30
API_TIMEOUT_SLOW_SEC = 20
API_RETRIES = 3
API_BACKOFF_SEC = 2.0
# history collection controls
USER_DOC_LIMIT = 250
USER_POST_LIMIT = 100
USER_COMMENT_LIMIT = 100
USER_INTERACTION_LIMIT = 100
USER_SUBREDDIT_LIMIT = 100
USER_FLAIR_LIMIT = 100
# optional temporal window for history collection. If None, the study period from the vote dataset is used
HISTORY_AFTER = None
HISTORY_BEFORE = None
# sleep for the API.
API_SLEEP_SEC = 0.35

## Helper functions

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# yield successive sub-lists of length `size` from `lst`
def _chunk(lst: List[Any], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

# convert a date-like object to ISO date string
def _parse_date(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    if isinstance(x, datetime):
        return x.date().isoformat()
    return str(x)

# GET a JSON endpoint with basic retry/backoff handling
def _safe_request(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = API_TIMEOUT_SEC) -> Any:
    url = f"{ARCTIC_SHIFT_BASE}{path}"
    params = {k: v for k, v in (params or {}).items() if v is not None}
    last_exc: Optional[Exception] = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                log.warning("Request failed for %s (%s)", url, exc)
                return None
            last_exc = exc
            if attempt < API_RETRIES:
                time.sleep(API_BACKOFF_SEC * attempt)
            else:
                log.warning("Request failed for %s (%s)", url, exc)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < API_RETRIES:
                time.sleep(API_BACKOFF_SEC * attempt)
            else:
                log.warning("Request failed for %s (%s)", url, exc)
    return None

# convert common Arctic Shift response shapes into a list of dicts
def _normalize_payload(payload: Any) -> List[Dict]:
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

# sanitize text
def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None

def _combine_post_text(item: Dict) -> Optional[str]:
    title = _safe_text(item.get("title"))
    selftext = _safe_text(item.get("selftext"))
    if title and selftext:
        return f"{title}\n\n{selftext}"
    return title or selftext

# convert a Unix creation timestamp to account age in days
def _compute_tenure(created_utc: Any) -> float:
    if created_utc is None:
        return np.nan
    return (time.time() - float(created_utc)) / 86400

# 1. Load and clean vote dataset
# load the pre-intersected dataset from CSV and apply basic cleaning
def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df["vote"] = pd.to_numeric(df["vote"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])].copy()
    df["vote"] = df["vote"].astype("int8")
    df["label"] = df["label"].astype("int8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    core_cols = ["username", "community", "item_id", "timestamp", "vote", "label"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    log.info("Dropped %d rows with nulls in core columns", before - len(df))
    before = len(df)
    df = df.drop_duplicates(subset=["username", "item_id"])
    log.info("Dropped %d duplicate (username, item_id) pairs", before - len(df))
    df = df.reset_index(drop=True)
    log.info(
        "Dataset loaded: %d votes | %d unique users | %d unique posts | %d communities",
        len(df),
        df["username"].nunique(),
        df["item_id"].nunique(),
        df["community"].nunique(),
    )
    return df

# 2. Density filtering
# iteratively prune the vote matrix until both density thresholds are satisfied (Birdwatch)
def apply_density_filter(
    df: pd.DataFrame,
    min_votes_per_post: int = MIN_VOTES_PER_POST,
    min_votes_per_user: int = MIN_VOTES_PER_USER,
) -> pd.DataFrame:
    log.info(
        "Applying density filter: >=%d votes/post, >=%d votes/user",
        min_votes_per_post,
        min_votes_per_user,
    )
    filtered = df.copy()
    for iteration in range(1, 100):
        prev_len = len(filtered)
        valid_posts = filtered["item_id"].value_counts()
        valid_posts = valid_posts[valid_posts >= min_votes_per_post].index
        filtered = filtered[filtered["item_id"].isin(valid_posts)]
        valid_users = filtered["username"].value_counts()
        valid_users = valid_users[valid_users >= min_votes_per_user].index
        filtered = filtered[filtered["username"].isin(valid_users)]
        log.info("  Iteration %d: %d -> %d votes", iteration, prev_len, len(filtered))
        if len(filtered) == prev_len:
            break

    log.info(
        "After density filter: %d votes | %d users | %d posts | %d communities",
        len(filtered),
        filtered["username"].nunique(),
        filtered["item_id"].nunique(),
        filtered["community"].nunique(),
    )
    return filtered.reset_index(drop=True)

# 3. Arctic Shift metadata
# query the Arctic Shift API for user-level metadata. Skips users if already fetched
def fetch_user_metadata(usernames: list[str], output_dir: Optional[Path] = None) -> pd.DataFrame:
    existing = pd.DataFrame()
    if output_dir is not None:
        meta_path = output_dir / "users.parquet"
        if meta_path.exists():
            existing = pd.read_parquet(meta_path)
            done = set(existing["username"])
            usernames = [u for u in usernames if u not in done]
            log.info(
                "Resuming metadata: %d already done, %d remaining",
                len(done), len(usernames),
            )
    log.info("Fetching Arctic Shift metadata for %d users...", len(usernames))
    records: list[dict] = []
    for uname in tqdm(usernames, desc="Arctic Shift metadata"):
        records.append(_query_single_user_metadata(uname))
        time.sleep(API_SLEEP_SEC)
    df_new = pd.DataFrame(records) if records else pd.DataFrame()
    df = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    if not df.empty:
        n_found = df["karma"].notna().sum()
        log.info(
            "Metadata retrieved for %d / %d users (%.1f%%)",
            n_found,
            len(df),
            100 * n_found / len(df),
        )
    return df

# fetch aggregate metadata for user via /api/users/search
def _query_single_user_metadata(uname: str) -> dict:
    try:
        payload = _safe_request("/api/users/search", params={"author": uname, "limit": 1})
        items = _normalize_payload(payload)
        info = next(
            (x for x in items if str(x.get("author", "")).lower() == uname.lower()),
            {},
        )
        meta = info.get("_meta", {})
        return {
            "username":               uname,
            "karma":                  meta.get("total_karma",         np.nan),
            "post_karma":             meta.get("post_karma",          np.nan),
            "comment_karma":          meta.get("comment_karma",       np.nan),
            "num_posts":              meta.get("num_posts",           np.nan),
            "num_comments":           meta.get("num_comments",        np.nan),
            "tenure_days":            _compute_tenure(meta.get("earliest_comment_at")),
            "earliest_post_at":       meta.get("earliest_post_at",    np.nan),
            "earliest_comment_at":    meta.get("earliest_comment_at", np.nan),
            "last_post_at":           meta.get("last_post_at",        np.nan),
            "last_comment_at":        meta.get("last_comment_at",     np.nan),
            # filled by compute_user_features(), not by the API
            "active_communities":     np.nan,
            "subreddit_entropy":      np.nan,
            "n_interaction_partners": np.nan,
            "total_interactions":     np.nan,
        }
    except Exception as exc:
        log.warning("Metadata fetch failed for %s (%s)", uname, exc)
        return {
            "username": uname,
            "karma": np.nan, "post_karma": np.nan, "comment_karma": np.nan,
            "num_posts": np.nan, "num_comments": np.nan, "tenure_days": np.nan,
            "earliest_post_at": np.nan, "earliest_comment_at": np.nan,
            "last_post_at": np.nan, "last_comment_at": np.nan,
            "active_communities": np.nan, "subreddit_entropy": np.nan,
            "n_interaction_partners": np.nan, "total_interactions": np.nan,
        }

# left-join user metadata onto the vote dataframe
def merge_user_metadata(df: pd.DataFrame, user_meta: pd.DataFrame) -> pd.DataFrame:
    enriched = df.merge(user_meta, on="username", how="left")
    missing = enriched["karma"].isna().sum()
    log.info(
        "Votes without metadata: %d / %d (%.1f%%)",
        missing,
        len(enriched),
        100 * missing / len(enriched),
    )
    return enriched

# 4. Arctic Shift user history
# collect Reddit history for later profiling/embedding
def fetch_user_history(
    usernames: List[str],
    after: Optional[str] = HISTORY_AFTER,
    before: Optional[str] = HISTORY_BEFORE,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    _history_files = {
        "documents":          "user_documents.parquet",
        "subreddit_activity": "user_subreddit_activity.parquet",
        "interactions":       "user_interactions.parquet",
        "flairs":             "user_flairs.parquet",
        "summary":            "user_history_summary.parquet",
    }
    existing: Dict[str, pd.DataFrame] = {k: pd.DataFrame() for k in _history_files}
    if output_dir is not None:
        history_dir = output_dir / "history"
        summary_path = history_dir / "user_history_summary.parquet"
        if summary_path.exists():
            existing["summary"] = pd.read_parquet(summary_path)
            done = set(existing["summary"]["username"])
            usernames = [u for u in usernames if u not in done]
            log.info(
                "Resuming history: %d already done, %d remaining",
                len(done), len(usernames),
            )
            for key, fname in _history_files.items():
                p = history_dir / fname
                if p.exists():
                    existing[key] = pd.read_parquet(p)
    log.info("Collecting user history for %d users...", len(usernames))
    doc_rows: list[dict] = []
    subreddit_rows: list[dict] = []
    interaction_rows: list[dict] = []
    flair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for uname in tqdm(usernames, desc="Arctic Shift history"):
        u_docs, u_subs, u_inter, u_flairs, u_summary = _collect_one_user_history(
            uname=uname,
            after=after,
            before=before,
        )
        doc_rows.extend(u_docs)
        subreddit_rows.extend(u_subs)
        interaction_rows.extend(u_inter)
        flair_rows.extend(u_flairs)
        summary_rows.append(u_summary)
        time.sleep(API_SLEEP_SEC)

    def _concat(existing_df: pd.DataFrame, new_rows: list) -> pd.DataFrame:
        new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
        if existing_df.empty:
            return new_df
        if new_df.empty:
            return existing_df
        return pd.concat([existing_df, new_df], ignore_index=True)

    documents         = _concat(existing["documents"],         doc_rows)
    subreddit_activity = _concat(existing["subreddit_activity"], subreddit_rows)
    interactions      = _concat(existing["interactions"],      interaction_rows)
    flairs            = _concat(existing["flairs"],            flair_rows)
    summary           = _concat(existing["summary"],           summary_rows)
    log.info(
        "History collected: %d docs | %d subreddit rows | %d interaction rows | %d flair rows",
        len(documents),
        len(subreddit_activity),
        len(interactions),
        len(flairs),
    )
    return {
        "documents": documents,
        "subreddit_activity": subreddit_activity,
        "interactions": interactions,
        "flairs": flairs,
        "summary": summary,
    }

# collect post/comment for one user
def _collect_one_user_history(
    uname: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Dict]:
    docs: list[dict] = []
    subreddit_rows: list[dict] = []
    interaction_rows: list[dict] = []
    flair_rows: list[dict] = []
    posts_payload = _safe_request(
        "/api/posts/search",
        params={
            "author": uname,
            "after": after,
            "before": before,
            "limit": USER_POST_LIMIT,
            "sort": "asc",
        },
    )
    posts = _normalize_payload(posts_payload)
    truncated_posts = len(posts) >= USER_POST_LIMIT
    for item in posts:
        docs.append(
            {
                "username": uname,
                "source": "post",
                "thing_id": item.get("id") or item.get("name"),
                "subreddit": item.get("subreddit"),
                "created_utc": item.get("created_utc"),
                "title": _safe_text(item.get("title")),
                "text": _combine_post_text(item),
                "score": item.get("score", np.nan),
                "num_comments": item.get("num_comments", np.nan),
                "link_id": item.get("link_id"),
                "parent_id": item.get("parent_id"),
                "url": item.get("url"),
            }
        )
    comments_payload = _safe_request(
        "/api/comments/search",
        params={
            "author": uname,
            "after": after,
            "before": before,
            "limit": USER_COMMENT_LIMIT,   # capped to 100
            "sort": "asc",
        },
    )
    comments = _normalize_payload(comments_payload)
    truncated_comments = len(comments) >= USER_COMMENT_LIMIT
    for item in comments:
        docs.append(
            {
                "username": uname,
                "source": "comment",
                "thing_id": item.get("id") or item.get("name"),
                "subreddit": item.get("subreddit"),
                "created_utc": item.get("created_utc"),
                "title": None,
                "text": _safe_text(item.get("body")),
                "score": item.get("score", np.nan),
                "num_comments": np.nan,
                "link_id": item.get("link_id"),
                "parent_id": item.get("parent_id"),
                "url": None,
            }
        )
    subs_payload = _safe_request(
        "/api/users/interactions/subreddits",
        params={
            "author": uname,
            "after": after,
            "before": before,
            "limit": USER_SUBREDDIT_LIMIT,
        },
    )
    subs = _normalize_payload(subs_payload)
    for item in subs:
        subreddit_rows.append(
            {
                "username": uname,
                "subreddit": item.get("subreddit"),
                "count": item.get("count", np.nan),
                "weighted_count": item.get("weighted_count", np.nan),
                "after": after,
                "before": before,
            }
        )
    try:
        inter_payload = _safe_request(
            "/api/users/interactions/users",
            params={
                "author": uname,
                "after": after,
                "before": before,
                "limit": USER_INTERACTION_LIMIT,
            },
            timeout=API_TIMEOUT_SLOW_SEC,
        )
        inter = _normalize_payload(inter_payload)
        for item in inter:
            interaction_rows.append(
                {
                    "username": uname,
                    "other_user": item.get("author") or item.get("other_author") or item.get("target"),
                    "count": int(item["count"]) if item.get("count") is not None else np.nan,
                    "after": after,
                    "before": before,
                }
            )
    except Exception as exc:
        log.warning("interactions/users skipped for %s (%s)", uname, exc)
    try:
        flairs_payload = _safe_request(
            "/api/users/aggregate_flairs",
            params={"author": uname},
        )
        flairs_data = (flairs_payload or {}).get("data", {}) if isinstance(flairs_payload, dict) else {}
        for subreddit, flair_dict in flairs_data.items():
            if isinstance(flair_dict, dict):
                for flair_text, count in flair_dict.items():
                    flair_rows.append(
                        {
                            "username": uname,
                            "subreddit": subreddit,
                            "author_flair_text": flair_text,
                            "count": int(count) if count is not None else np.nan,
                        }
                    )
    except Exception as exc:
        log.warning("aggregate_flairs skipped for %s (%s)", uname, exc)
    summary = {
        "username": uname,
        "n_documents": len(docs),
        "n_posts": len(posts),
        "n_comments": len(comments),
        "n_subreddit_rows": len(subreddit_rows),
        "n_interaction_rows": len(interaction_rows),
        "n_flair_rows": len(flair_rows),
        "posts_truncated": truncated_posts,
        "comments_truncated": truncated_comments,
    }
    return docs, subreddit_rows, interaction_rows, flair_rows, summary

# 5. Density / history alignment helpers
def infer_history_window(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if HISTORY_AFTER is not None or HISTORY_BEFORE is not None:
        return HISTORY_AFTER, HISTORY_BEFORE
    after = df["timestamp"].min().date().isoformat()
    before = df["timestamp"].max().date().isoformat()
    return after, before

# 6. Chronological split
# split the vote dataframe chronologically into train/val/test sets based on posts
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Split ratios must sum to 1.0"

    post_times = (
        df.groupby("item_id")["timestamp"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"timestamp": "post_time"})
    )
    n = len(post_times)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    post_times["split"] = "test"
    post_times.iloc[:train_end, post_times.columns.get_loc("split")] = "train"
    post_times.iloc[train_end:val_end, post_times.columns.get_loc("split")] = "val"
    df_split = df.merge(post_times[["item_id", "split"]], on="item_id", how="left")
    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        subset = df_split[df_split["split"] == name].drop(columns="split").copy()
        splits[name] = subset.reset_index(drop=True)
        if len(subset) > 0:
            log.info(
                "Split '%s': %d votes | %d users | %d posts | %s -> %s",
                name,
                len(subset),
                subset["username"].nunique(),
                subset["item_id"].nunique(),
                subset["timestamp"].min().date(),
                subset["timestamp"].max().date(),
            )
        else:
            log.info("Split '%s': empty", name)

    return splits

# 7. Derived user features
# shannon entropy (bits) of a discrete distribution given raw counts. Returns NaN if the array is empty or all zeros
def _shannon_entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

# compute entropy of the user's monthly activity distribution
def _fetch_temporal_entropy(
    uname: str,
    after: Optional[str],
    before: Optional[str],
) -> float:
    monthly: dict[str, int] = {}
    for content_type in ("posts", "comments"):
        try:
            payload = _safe_request(
                f"/api/{content_type}/search/aggregate",
                params={
                    "author":    uname,
                    "aggregate": "created_utc",
                    "frequency": "month",
                    "after":     after,
                    "before":    before,
                },
            )
            for item in _normalize_payload(payload):
                bucket = str(item.get("created_utc", ""))
                count  = item.get("count", 0)
                try:
                    monthly[bucket] = monthly.get(bucket, 0) + int(count)
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            log.warning("temporal aggregate failed for %s/%s (%s)", uname, content_type, exc)
    if not monthly:
        return np.nan
    return _shannon_entropy(np.array(list(monthly.values()), dtype=float))

# populate derived feature columns in users.parquet
def compute_user_features(
    output_dir: Path = OUTPUT_DIR,
    compute_temporal: bool = True,
) -> pd.DataFrame:
    meta_path = output_dir / "users.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"users.parquet not found at {meta_path}")

    users = pd.read_parquet(meta_path)
    history_dir = output_dir / "history"

    # 1. Subreddit-based features (no API)
    need_sub = users["active_communities"].isna().any()
    if need_sub:
        sub_path = history_dir / "user_subreddit_activity.parquet"
        if sub_path.exists():
            subs = pd.read_parquet(sub_path)

            agg = subs.groupby("username").agg(
                active_communities=("subreddit", "nunique"),
                _counts=("count", list),
            ).reset_index()

            agg["subreddit_entropy"] = agg["_counts"].apply(
                lambda lst: _shannon_entropy(np.array([x for x in lst if pd.notna(x)], dtype=float))
            )
            agg = agg.drop(columns=["_counts"])

            users = users.drop(columns=["active_communities", "subreddit_entropy"], errors="ignore")
            users = users.merge(agg, on="username", how="left")
            log.info(
                "active_communities filled: %d / %d users",
                users["active_communities"].notna().sum(), len(users),
            )
        else:
            log.warning("user_subreddit_activity.parquet not found — skipping subreddit features")

    # 2. Interaction-based features (no API)
    need_inter = users["n_interaction_partners"].isna().any()
    if need_inter:
        inter_path = history_dir / "user_interactions.parquet"
        if inter_path.exists():
            inter = pd.read_parquet(inter_path)

            agg_inter = inter.groupby("username").agg(
                n_interaction_partners=("other_user", "nunique"),
                total_interactions=("count", "sum"),
            ).reset_index()

            users = users.drop(columns=["n_interaction_partners", "total_interactions"], errors="ignore")
            users = users.merge(agg_inter, on="username", how="left")
            log.info(
                "interaction features filled: %d / %d users",
                users["n_interaction_partners"].notna().sum(), len(users),
            )
        else:
            log.warning("user_interactions.parquet not found — skipping interaction features")

    # 3. Temporal entropy (lightweight aggregate API)
    if compute_temporal:
        if "temporal_entropy" not in users.columns:
            users["temporal_entropy"] = np.nan

        # infer study window from earliest/latest timestamps already in users
        after_ts  = users["earliest_comment_at"].min()
        before_ts = users["last_comment_at"].max()
        after  = pd.to_datetime(after_ts,  unit="s", utc=True).date().isoformat() if pd.notna(after_ts)  else None
        before = pd.to_datetime(before_ts, unit="s", utc=True).date().isoformat() if pd.notna(before_ts) else None

        todo = users.loc[users["temporal_entropy"].isna(), "username"].tolist()
        log.info("Computing temporal entropy for %d users (API)...", len(todo))

        for uname in tqdm(todo, desc="temporal entropy"):
            users.loc[users["username"] == uname, "temporal_entropy"] = (
                _fetch_temporal_entropy(uname, after=after, before=before)
            )
            time.sleep(API_SLEEP_SEC)

        log.info(
            "temporal_entropy filled: %d / %d users",
            users["temporal_entropy"].notna().sum(), len(users),
        )
    # 4. Save users.parquet
    users.to_parquet(meta_path, index=False)
    log.info("Enriched users.parquet saved to %s", meta_path)
    return users

# 8. Saving
# write all processed artefacts to Parquet files under ``output_dir``
def save_outputs(
    filtered_df: pd.DataFrame,
    user_meta: pd.DataFrame,
    history_tables: dict[str, pd.DataFrame],
    splits: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    history_dir = output_dir / "history"
    splits_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)

    filtered_df.to_parquet(output_dir / "filtered_votes.parquet", index=False)
    user_meta.to_parquet(output_dir / "users.parquet", index=False)

    history_tables["documents"].to_parquet(history_dir / "user_documents.parquet", index=False)
    history_tables["subreddit_activity"].to_parquet(history_dir / "user_subreddit_activity.parquet", index=False)
    history_tables["interactions"].to_parquet(history_dir / "user_interactions.parquet", index=False)
    history_tables["flairs"].to_parquet(history_dir / "user_flairs.parquet", index=False)
    history_tables["summary"].to_parquet(history_dir / "user_history_summary.parquet", index=False)

    for split_name, split_df in splits.items():
        split_df.to_parquet(splits_dir / f"{split_name}_votes.parquet", index=False)

    log.info("All outputs saved to %s", output_dir)

def print_summary(df: pd.DataFrame) -> None:
    log.info("--- Dataset summary ---")
    label_dist = df.drop_duplicates("item_id")["label"].value_counts()
    log.info("Post label distribution:\n%s", label_dist.to_string())
    vote_dist = df["vote"].value_counts()
    log.info("Vote distribution:\n%s", vote_dist.to_string())
    top_communities = df.drop_duplicates("item_id")["community"].value_counts().head(5)
    log.info("Top 5 communities by post count:\n%s", top_communities.to_string())

# Main function
def run_step1(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """
    1. Load and clean dataset.
    2. Apply density filter.
    3. Collect user metadata for remaining users.
    4. Collect user history tables for the same users.
    5. Merge metadata.
    6. Print summary.
    7. Chronological split.
    8. Save.
    """
    log.info("=" * 50)
    log.info("STEP 1 - DATA PREPARATION (Reddit)")
    log.info("=" * 50)
    df = load_dataset(input_path)
    df = apply_density_filter(df)
    usernames = df["username"].unique().tolist()
    history_after, history_before = infer_history_window(df)
    user_meta = fetch_user_metadata(usernames, output_dir=output_dir)
    history_tables = fetch_user_history(usernames, after=history_after, before=history_before, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history").mkdir(exist_ok=True)
    user_meta.to_parquet(output_dir / "users.parquet", index=False)
    history_dir = output_dir / "history"
    history_tables["documents"].to_parquet(history_dir / "user_documents.parquet", index=False)
    history_tables["subreddit_activity"].to_parquet(history_dir / "user_subreddit_activity.parquet", index=False)
    history_tables["interactions"].to_parquet(history_dir / "user_interactions.parquet", index=False)
    history_tables["flairs"].to_parquet(history_dir / "user_flairs.parquet", index=False)
    history_tables["summary"].to_parquet(history_dir / "user_history_summary.parquet", index=False)
    user_meta = compute_user_features(output_dir=output_dir, compute_temporal=True)
    df = merge_user_metadata(df, user_meta)
    print_summary(df)
    splits = chronological_split(df)
    save_outputs(df, user_meta, history_tables, splits, output_dir)
    log.info("Step 1 complete.")
    return {
        "filtered_df": df,
        "user_meta": user_meta,
        "history_tables": history_tables,
        "splits": splits,
    }

if __name__ == "__main__":
    run_step1(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)