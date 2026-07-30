from pathlib import Path
import json
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
# split ratios (sum to 1) -- now RANDOM, not chronological
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# fixed seed so the random split is reproducible across runs
RANDOM_SEED = 42
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

# ── per-method filter thresholds (mirrors constants in each method's source) ─
# VAR community filter
_VAR_MIN_SUB_USERS = 40
# TFR skill filter: min votes per (user, community) to have any skill entry
_TFR_MIN_SKILL_VOTES = 5
# SEF user reliability filter
_SEF_MIN_USER_VOTES = 10
# windowed-fold training proportions and fixed test fraction
_WINDOW_SIZES    = [0.2, 0.4, 0.6, 0.8, 1.0]
_WINDOW_TEST_FRAC = 0.20
_WINDOW_VAL_FRAC  = 0.20   # NEW: fixed validation fraction, held constant across all window sizes


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
            "limit": USER_COMMENT_LIMIT,
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

# 6. Random split (was: chronological_split)
# split the vote dataframe RANDOMLY (item-level, fixed seed) into train/val/test sets
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """
    NOTE: kept the name `chronological_split` for call-site compatibility
    (run_step1 calls this function), but the split is now a random,
    seeded shuffle of item_id, not an ordering by post_time.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Split ratios must sum to 1.0"

    item_ids = df["item_id"].unique().tolist()   # plain Python list -> safe for rng.shuffle
    rng = np.random.RandomState(random_state)
    shuffled_ids = item_ids.copy()
    rng.shuffle(shuffled_ids)

    assignment = pd.DataFrame({"item_id": shuffled_ids})
    n = len(assignment)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    assignment["split"] = "test"
    assignment.iloc[:train_end, assignment.columns.get_loc("split")] = "train"
    assignment.iloc[train_end:val_end, assignment.columns.get_loc("split")] = "val"

    df_split = df.merge(assignment[["item_id", "split"]], on="item_id", how="left")
    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        subset = df_split[df_split["split"] == name].drop(columns="split").copy()
        splits[name] = subset.reset_index(drop=True)
        if len(subset) > 0:
            log.info(
                "Split '%s' (random, seed=%d): %d votes | %d users | %d posts | %s .. %s",
                name,
                random_state,
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


# ===========================================================================
# 9. FILTER IDENTIFICATION  — which posts/users each method would exclude
# ===========================================================================

def _iterative_density_filter(
    data: pd.DataFrame,
    min_post_votes: int,
    min_user_votes: int,
) -> Tuple[set, set]:
    """Return (valid_item_ids, valid_usernames) after iterative density pruning."""
    d = data.copy()
    for _ in range(100):
        prev = len(d)
        d = d[d["item_id"].isin(
            d["item_id"].value_counts()[lambda s: s >= min_post_votes].index)]
        d = d[d["username"].isin(
            d["username"].value_counts()[lambda s: s >= min_user_votes].index)]
        if len(d) == prev:
            break
    return set(d["item_id"].unique()), set(d["username"].unique())


def identify_method_filters(
    df: pd.DataFrame,
    external_scores_path: Optional[Path] = None,
    post_texts_path: Optional[Path]      = None,
) -> Dict[str, Dict]:
    """
    For each method, compute which posts and users its internal filters would
    exclude.  Returns a dict keyed by method name; each value contains:

      valid_item_ids  – set[str]  posts that survive this method's filters
      valid_usernames – set[str]  users that survive this method's filters
      n_items_kept / n_items_dropped
      n_users_kept / n_users_dropped
      reason          – plain-text description of the filter logic

    These sets are used downstream by build_intersection_dataset().
    """
    all_items = set(df["item_id"].unique())
    all_users = set(df["username"].unique())
    report: Dict[str, Dict] = {}

    # ── BL1-BL3 / CN  (Birdwatch iterative density filter) ──────────────────
    items_bl, users_bl = _iterative_density_filter(df, MIN_VOTES_PER_POST, MIN_VOTES_PER_USER)
    report["BL_CN"] = {
        "valid_item_ids":  items_bl,
        "valid_usernames": users_bl,
        "n_items_kept":    len(items_bl),
        "n_items_dropped": len(all_items) - len(items_bl),
        "n_users_kept":    len(users_bl),
        "n_users_dropped": len(all_users) - len(users_bl),
        "reason": (
            f"Iterative density filter: post needs >={MIN_VOTES_PER_POST} votes, "
            f"user needs >={MIN_VOTES_PER_USER} votes. Applied until convergence."
        ),
    }

    # ── BL4 / BL5  (same density filter + external Reddit score required) ────
    if external_scores_path is not None and external_scores_path.exists():
        ext       = pd.read_parquet(external_scores_path)[["item_id"]]
        items_ext = items_bl & set(ext["item_id"].unique())
        missing_ext = len(items_bl) - len(items_ext)
    else:
        items_ext   = items_bl
        missing_ext = 0
        log.warning("identify_method_filters: external scores not found at %s — "
                    "BL4/BL5 item filter approximated as BL_CN", external_scores_path)
    report["BL4_BL5"] = {
        "valid_item_ids":  items_ext,
        "valid_usernames": users_bl,
        "n_items_kept":    len(items_ext),
        "n_items_dropped": len(all_items) - len(items_ext),
        "n_users_kept":    len(users_bl),
        "n_users_dropped": len(all_users) - len(users_bl),
        "reason": (
            f"BL_CN density filter + post must appear in external scores parquet "
            f"(Arctic Shift Reddit score available). {missing_ext} posts excluded "
            f"for missing external score."
        ),
    }

    # ── TFR  (no post-level filter; user needs >= _TFR_MIN_SKILL_VOTES votes
    #          in at least one (community) to have a usable skill entry) ───────
    votes_per_user_comm = (
        df.groupby(["username", "community"])["item_id"]
        .count()
        .reset_index(name="n_votes")
    )
    users_with_skill = set(
        votes_per_user_comm[
            votes_per_user_comm["n_votes"] >= _TFR_MIN_SKILL_VOTES
        ]["username"].unique()
    )
    report["TFR"] = {
        "valid_item_ids":  all_items,       # TFR uses all posts (pre-density CSV)
        "valid_usernames": users_with_skill,
        "n_items_kept":    len(all_items),
        "n_items_dropped": 0,
        "n_users_kept":    len(users_with_skill),
        "n_users_dropped": len(all_users) - len(users_with_skill),
        "reason": (
            f"No post-level filter (TFR reads the raw pre-density CSV). "
            f"A user contributes to skill only if they cast >={_TFR_MIN_SKILL_VOTES} "
            f"votes in at least one (community) bucket. Posts with zero skilled voters "
            f"are silently dropped at prediction time (~15% coverage loss at runtime)."
        ),
    }

    # ── SEF  (post needs non-null text; user needs >= _SEF_MIN_USER_VOTES votes)
    if post_texts_path is not None and post_texts_path.exists():
        pt        = pd.read_parquet(post_texts_path)[["item_id", "text"]]
        items_sef = set(pt[pt["text"].notna()]["item_id"].unique())
    else:
        items_sef = all_items
        log.warning("identify_method_filters: post_texts not found at %s — "
                    "SEF item filter not applied", post_texts_path)
    users_sef = set(
        df[df["username"].isin(
            df["username"].value_counts()[lambda s: s >= _SEF_MIN_USER_VOTES].index
        )]["username"].unique()
    )
    report["SEF"] = {
        "valid_item_ids":  items_sef,
        "valid_usernames": users_sef,
        "n_items_kept":    len(items_sef),
        "n_items_dropped": len(all_items) - len(items_sef),
        "n_users_kept":    len(users_sef),
        "n_users_dropped": len(all_users) - len(users_sef),
        "reason": (
            f"Post must have non-null text in post_texts.parquet (required for FAISS "
            f"embedding). User must have >={_SEF_MIN_USER_VOTES} votes globally to "
            f"enter prec_df (reliability lookup table)."
        ),
    }

    # ── VAR  (own iterative density min_user=5 + community needs >=40 users) ─
    items_var_tmp, users_var_tmp = _iterative_density_filter(df, MIN_VOTES_PER_POST, 5)
    df_var = df[
        df["item_id"].isin(items_var_tmp) &
        df["username"].isin(users_var_tmp)
    ]
    valid_communities = (
        df_var.groupby("community")["username"]
        .nunique()[lambda s: s >= _VAR_MIN_SUB_USERS]
        .index
    )
    df_var    = df_var[df_var["community"].isin(valid_communities)]
    items_var = set(df_var["item_id"].unique())
    users_var = set(df_var["username"].unique())
    report["VAR"] = {
        "valid_item_ids":  items_var,
        "valid_usernames": users_var,
        "n_items_kept":    len(items_var),
        "n_items_dropped": len(all_items) - len(items_var),
        "n_users_kept":    len(users_var),
        "n_users_dropped": len(all_users) - len(users_var),
        "reason": (
            f"Iterative density filter (min 5 votes/post, 5 votes/user) then "
            f"community must have >={_VAR_MIN_SUB_USERS} unique users. "
            f"The community-size filter is the most restrictive across all methods."
        ),
    }

    # ── print summary table ──────────────────────────────────────────────────
    log.info("=== Per-method filter summary ===")
    log.info("%-10s  %10s  %13s  %10s  %13s",
             "Method", "items_kept", "items_dropped", "users_kept", "users_dropped")
    for name, r in report.items():
        log.info("  %-10s  %10d  %13d  %10d  %13d",
                 name,
                 r["n_items_kept"],   r["n_items_dropped"],
                 r["n_users_kept"],   r["n_users_dropped"])

    return report


# ===========================================================================
# 10. INTERSECTION DATASET
# ===========================================================================

def build_intersection_dataset(
    df: pd.DataFrame,
    filter_report: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Keep only posts and users that survive ALL method-specific filters, then
    re-apply the BL/CN density filter to guarantee the minimum-votes
    invariant still holds after the intersection shrinks the matrix.

    The resulting dataset is the most conservative common ground: every
    method can run on it without any further internal filtering.
    """
    item_intersection: Optional[set] = None
    user_intersection: Optional[set] = None

    for r in filter_report.values():
        if item_intersection is None:
            item_intersection = r["valid_item_ids"].copy()
            user_intersection = r["valid_usernames"].copy()
        else:
            item_intersection &= r["valid_item_ids"]
            user_intersection &= r["valid_usernames"]

    df_inter = df[
        df["item_id"].isin(item_intersection) &
        df["username"].isin(user_intersection)
    ].copy()

    # re-apply density filter to the reduced matrix
    prev = len(df_inter)
    for _ in range(100):
        df_inter = df_inter[df_inter["item_id"].isin(
            df_inter["item_id"].value_counts()[lambda s: s >= MIN_VOTES_PER_POST].index)]
        df_inter = df_inter[df_inter["username"].isin(
            df_inter["username"].value_counts()[lambda s: s >= MIN_VOTES_PER_USER].index)]
        if len(df_inter) == prev:
            break
        prev = len(df_inter)

    log.info(
        "Intersection dataset: %d votes | %d posts | %d users | %d communities",
        len(df_inter),
        df_inter["item_id"].nunique(),
        df_inter["username"].nunique(),
        df_inter["community"].nunique(),
    )
    log.info(
        "  vs full post-filter: %d votes | %d posts | %d users",
        len(df), df["item_id"].nunique(), df["username"].nunique(),
    )
    return df_inter.reset_index(drop=True)


# ===========================================================================
# 11. TWO MAIN SPLITS  (full and intersection)
# ===========================================================================

def _chrono_split_to_disk(
    df: pd.DataFrame,
    out_dir: Path,
    label: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> None:
    """
    Randomly split df (item-level, fixed seed) and write train/val/test
    parquets to out_dir.

    NOTE: function name kept as `_chrono_split_to_disk` for call-site
    compatibility (produce_splits calls this), but items are now assigned
    to train/val/test via a seeded random shuffle, not by first-vote
    timestamp order. No post straddles two splits (split is per item_id).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    item_ids = df["item_id"].unique().tolist()   # plain Python list -> safe for rng.shuffle
    rng = np.random.RandomState(random_state)
    shuffled_ids = item_ids.copy()
    rng.shuffle(shuffled_ids)

    assignment = pd.DataFrame({"item_id": shuffled_ids})
    n         = len(assignment)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    assignment["split"] = "test"
    assignment.iloc[:train_end,  assignment.columns.get_loc("split")] = "train"
    assignment.iloc[train_end:val_end, assignment.columns.get_loc("split")] = "val"

    df_split = df.merge(assignment[["item_id", "split"]], on="item_id", how="left")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        subset = df_split[df_split["split"] == name].drop(columns="split").copy()
        subset.reset_index(drop=True).to_parquet(out_dir / f"{name}_votes.parquet", index=False)
        log.info(
            "  [%s] %s (random, seed=%d): %d votes | %d posts | %s .. %s",
            label, name, random_state, len(subset), subset["item_id"].nunique(),
            subset["timestamp"].min().date(), subset["timestamp"].max().date(),
        )


def produce_splits(
    df_full:         pd.DataFrame,
    df_intersection: pd.DataFrame,
    output_dir:      Path,
) -> None:
    """
    Produce and save two independent train/val/test splits:

      splits_full/          — TRULY UNFILTERED dataset: every vote that
                              survives only the minimal cleaning in
                              load_dataset() (valid vote/label, deduped
                              (username,item_id) pairs) — NOT the density
                              filter (min votes/post, min votes/user). Every
                              post is included, even ones with a single
                              voter; methods are expected to fall back
                              (e.g. net-vote) when they have no real signal
                              for an item, instead of silently dropping it.
                              This is the "how much do performances degrade
                              when nothing gets filtered out" population.

      splits_intersection/  — only posts/users surviving EVERY method's
                              specific filter (built on top of the
                              density-filtered dataset); guarantees no
                              method needs to do further filtering at
                              evaluation time.

    Both splits use the same random, seeded item-level assignment logic
    as chronological_split() (name kept, logic is now random — see there).

    NOTE: df_full here is the RAW (pre-density-filter) dataframe passed in
    by run_step1 — see the "full (unfiltered)" section there. It is NOT the
    same `df` used for splits/ or splits_intersection/.
    """
    log.info("=== Producing splits_full (unfiltered — every post, fallback expected) ===")
    _chrono_split_to_disk(df_full,         output_dir / "splits_full",         "full")

    log.info("=== Producing splits_intersection ===")
    _chrono_split_to_disk(df_intersection, output_dir / "splits_intersection",  "intersection")

    log.info("Both split sets saved under %s", output_dir)


# ===========================================================================
# 12. WINDOWED FOLDS  (data-sufficiency / learning-curve analysis)
# ===========================================================================

def produce_windowed_folds(
    df: pd.DataFrame,
    output_dir: Path,
    window_sizes: List[float] = _WINDOW_SIZES,
    test_frac:    float       = _WINDOW_TEST_FRAC,
    val_frac:     float       = _WINDOW_VAL_FRAC,
    tag: str = "full",
    random_state: int = RANDOM_SEED,
) -> None:
    """
    Data-sufficiency analysis: one fold per training-window size.

    Motivation
    ----------
    Different methods need different amounts of data to reach acceptable
    performance. This analysis answers:

      "If only a random w% subset of the TRAINABLE pool were available for
       training, how well would method X perform, holding validation and
       test completely fixed?"

    A method that saturates at w=0.4 is usable with 40% of the trainable
    pool; one that keeps improving up to w=1.0 needs all of it.

    Structure  (fixed 2024 redesign — see note below)
    ---------------------------------------------------
    Items are randomly shuffled once (fixed seed).

    Fixed test set        = a random test_frac of items — IDENTICAL across
                             every window size w.
    Fixed validation set  = a random val_frac of items, disjoint from test —
                             ALSO IDENTICAL across every window size w.
    Trainable pool        = the remaining (1 - test_frac - val_frac) items,
                             in a fixed random order.

    For each w ∈ window_sizes:
      training set = first floor(w × |trainable_pool|) items from the pool.
      validation set and test set are the SAME data for every w — only the
      amount of training data changes.

    NOTE ON A PREVIOUS BUG: an earlier version of this function derived the
    validation set as "whatever remains of the pool after taking the first
    w fraction as train" — so validation shrank as training grew, and
    became EMPTY at w=1.0 (train = entire pool). That conflated two
    different things (training-set size vs. validation-set size) and made
    downstream comparisons across w not isolate the effect of training data
    alone. This version fixes that: validation and test are carved out
    ONCE, before iterating over w, and never change.

    Saved under output_dir/windowed_folds_{tag}/:
      w{w*100:03d}/train_votes.parquet  ← size varies with w
      w{w*100:03d}/val_votes.parquet    ← IDENTICAL for every w
      w{w*100:03d}/test_votes.parquet   ← IDENTICAL for every w
      manifest.json                     ← sizes and random seed per window
    """
    out_root = output_dir / f"windowed_folds_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    assert test_frac + val_frac < 1.0, "test_frac + val_frac must leave room for a trainable pool"

    item_ids = df["item_id"].unique().tolist()   # plain Python list -> safe for rng.shuffle
    rng = np.random.RandomState(random_state)
    shuffled_ids = item_ids.copy()
    rng.shuffle(shuffled_ids)

    n_items = len(shuffled_ids)
    n_test  = int(n_items * test_frac)
    n_val   = int(n_items * val_frac)

    # fixed test/val slices, carved out ONCE — never touched inside the w loop
    test_ids = set(shuffled_ids[: n_test]) if n_test > 0 else set()
    val_ids  = set(shuffled_ids[n_test: n_test + n_val]) if n_val > 0 else set()
    trainable_pool = shuffled_ids[n_test + n_val:]   # fixed random order
    n_trainable = len(trainable_pool)

    test_votes = df[df["item_id"].isin(test_ids)].copy()
    val_votes  = df[df["item_id"].isin(val_ids)].copy()

    manifest = {
        "tag":                   tag,
        "split_type":            "random",
        "random_seed":           random_state,
        "n_items_total":         n_items,
        "n_items_test":          len(test_ids),
        "n_items_val":           len(val_ids),
        "n_items_trainable_pool": n_trainable,
        "test_frac":             test_frac,
        "val_frac":              val_frac,
        "n_test_votes":          len(test_votes),
        "n_val_votes":           len(val_votes),
        "windows":               [],
    }

    log.info("=== Windowed folds (%s, random, seed=%d) ===", tag, random_state)
    log.info("  total items: %d | test (fixed): %d | val (fixed): %d | trainable pool: %d",
              n_items, len(test_ids), len(val_ids), n_trainable)

    for w in window_sizes:
        train_cut = int(n_trainable * w)
        train_ids = set(trainable_pool[:train_cut])

        train_votes = df[df["item_id"].isin(train_ids)].copy()

        folder = out_root / f"w{int(w * 100):03d}"
        folder.mkdir(exist_ok=True)
        train_votes.to_parquet(folder / "train_votes.parquet", index=False)
        val_votes.to_parquet(  folder / "val_votes.parquet",   index=False)
        test_votes.to_parquet( folder / "test_votes.parquet",  index=False)

        entry = {
            "w":             w,
            "folder":        folder.name,
            "n_train_items": len(train_ids),
            "n_val_items":   len(val_ids),    # constant across windows, by design
            "n_test_items":  len(test_ids),   # constant across windows, by design
            "n_train_votes": len(train_votes),
            "n_val_votes":   len(val_votes),
            "n_test_votes":  len(test_votes),
        }
        manifest["windows"].append(entry)

        log.info(
            "  w=%3.0f%%  train=%5d posts (random, from trainable pool)  val=%5d (fixed)  test=%5d (fixed)",
            w * 100, len(train_ids), len(val_ids), len(test_ids),
        )

    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info("  Saved → %s", out_root)


# ===========================================================================
# 13. FILTER REPORT  — JSON-serialisable summary
# ===========================================================================

def save_filter_report(
    filter_report: Dict[str, Dict],
    df_full:       pd.DataFrame,
    df_inter:      pd.DataFrame,
    output_dir:    Path,
) -> None:
    """Persist a JSON with per-method filter stats and the intersection summary."""
    serialisable: Dict[str, Any] = {}
    for name, r in filter_report.items():
        serialisable[name] = {
            k: v for k, v in r.items()
            if k not in ("valid_item_ids", "valid_usernames")   # sets are not JSON-serialisable
        }

    serialisable["__intersection__"] = {
        "n_votes":         len(df_inter),
        "n_items_kept":    df_inter["item_id"].nunique(),
        "n_items_dropped": df_full["item_id"].nunique() - df_inter["item_id"].nunique(),
        "n_users_kept":    df_inter["username"].nunique(),
        "n_users_dropped": df_full["username"].nunique() - df_inter["username"].nunique(),
        "n_communities":   df_inter["community"].nunique(),
        "reason": (
            "Intersection of all method-specific valid_item_ids and valid_usernames, "
            "followed by a re-applied BL/CN density filter."
        ),
    }

    path = output_dir / "filter_report.json"
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    log.info("Filter report saved → %s", path)


# ===========================================================================
# Main function
# ===========================================================================

def run_step1(
    input_path:           Path          = INPUT_PATH,
    output_dir:           Path          = OUTPUT_DIR,
    external_scores_path: Optional[Path] = None,
    post_texts_path:      Optional[Path] = None,
) -> dict[str, Any]:
    """
    Full data-preparation pipeline.

    Original steps (unchanged, split logic now random instead of chronological)
    ----------------------------------------------------------------------------
    1. Load and clean dataset.
    2. Apply density filter.
    3. Collect user metadata (Arctic Shift API).
    4. Collect user history tables (Arctic Shift API).
    5. Merge metadata onto votes.
    6. Print summary.
    7. Random split (seeded) → splits/  (legacy, kept for backwards-compatibility).
    8. Save all artefacts.

    New steps
    ---------
    9.  Identify per-method filters → filter_report.json
    10. Build intersection dataset (posts/users surviving ALL method filters).
    11. Produce splits_full/ and splits_intersection/  (random, seeded, 70/15/15).
        splits_full/ is built from the RAW, UNFILTERED population (only the
        minimal load_dataset() cleaning — no density filter): every post is
        included, even ones with a single voter. splits_intersection/ is
        built from the density-filtered intersection of every method's
        specific requirements. Evaluation code is expected to fall back
        (e.g. net-vote) on splits_full items it has no real signal for,
        instead of silently dropping them — this measures how much
        performance degrades when nothing gets filtered out upstream.
    12. Produce windowed_folds_full/ and windowed_folds_intersection/
        (data-sufficiency analysis, 5 window sizes, fixed random val/test).
        Both windowed_folds_full/ and windowed_folds_intersection/ use the
        density-filtered populations (df / df_inter respectively) — same as
        before. windowed_folds_full/ was briefly switched to the raw
        unfiltered population, then reverted back on request, to isolate
        the effect of unfiltering on splits_full alone first.

    Parameters
    ----------
    external_scores_path : path to moderated_posts_scores.parquet
        Used to identify which posts have an external Reddit score (BL4/BL5).
        If None or missing, BL4/BL5 filter falls back to the BL_CN filter.
    post_texts_path : path to post_texts.parquet
        Used to identify which posts have non-null text (SEF embedding filter).
        If None or missing, the SEF item filter is not applied.
    """
    log.info("=" * 50)
    log.info("STEP 1 - DATA PREPARATION (Reddit)")
    log.info("=" * 50)

    # ── original pipeline ────────────────────────────────────────────────────
    df_raw = load_dataset(input_path)   # minimal cleaning ONLY (valid vote/label,
                                          # deduped (username,item_id)) — NO density
                                          # filter. Kept aside to build the truly
                                          # unfiltered "full" splits later.
    df = apply_density_filter(df_raw)
    usernames = df["username"].unique().tolist()
    history_after, history_before = infer_history_window(df)

    user_meta = fetch_user_metadata(usernames, output_dir=output_dir)
    history_tables = fetch_user_history(
        usernames, after=history_after, before=history_before, output_dir=output_dir
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    history_dir.mkdir(exist_ok=True)

    user_meta.to_parquet(output_dir / "users.parquet", index=False)
    history_tables["documents"].to_parquet(history_dir / "user_documents.parquet",         index=False)
    history_tables["subreddit_activity"].to_parquet(history_dir / "user_subreddit_activity.parquet", index=False)
    history_tables["interactions"].to_parquet(history_dir / "user_interactions.parquet",   index=False)
    history_tables["flairs"].to_parquet(history_dir / "user_flairs.parquet",               index=False)
    history_tables["summary"].to_parquet(history_dir / "user_history_summary.parquet",     index=False)

    user_meta = compute_user_features(output_dir=output_dir, compute_temporal=True)
    df = merge_user_metadata(df, user_meta)
    print_summary(df)

    # same metadata merge applied to the RAW (unfiltered) population — users
    # who didn't pass the density filter simply get NaN metadata (left join),
    # exactly the kind of gap the per-method net-vote fallback is meant to
    # cover at evaluation time.
    df_raw = merge_user_metadata(df_raw, user_meta)

    splits = chronological_split(df)   # now random, seeded — see function docstring
    save_outputs(df, user_meta, history_tables, splits, output_dir)   # writes legacy splits/

    # ── new: multi-split and windowed folds ──────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 1 — multi-split & windowed folds")
    log.info("=" * 50)

    # resolve default paths if caller did not supply them
    if external_scores_path is None:
        external_scores_path = output_dir / "moderated_posts_scores.parquet"
    if post_texts_path is None:
        post_texts_path = Path("results/step3_expert/post_texts.parquet")

    filter_report = identify_method_filters(
        df,
        external_scores_path=external_scores_path,
        post_texts_path=post_texts_path,
    )
    df_inter = build_intersection_dataset(df, filter_report)

    save_filter_report(filter_report, df, df_inter, output_dir)

    # NOTE: df_raw (unfiltered) goes into splits_full only, for now — see
    # discussion with the user: windowed_folds_full/* is reverted back to
    # the density-filtered df, to isolate what changes when JUST splits_full
    # becomes unfiltered before extending this further.
    produce_splits(df_raw, df_inter, output_dir)

    produce_windowed_folds(df,       output_dir, tag="full")
    produce_windowed_folds(df_inter, output_dir, tag="intersection")

    log.info("Step 1 complete.")
    return {
        "filtered_df":        df,
        "raw_unfiltered_df":  df_raw,
        "intersection_df":    df_inter,
        "user_meta":          user_meta,
        "history_tables":     history_tables,
        "splits":             splits,
        "filter_report":      filter_report,
    }


if __name__ == "__main__":
    run_step1(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)