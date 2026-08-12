from pathlib import Path
import time
import logging
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from typing import Any, Optional, List, Dict

VOTES_PATH     = Path("data/interim/reddit/filtered_votes.parquet")
OUTPUT_DIR     = Path("data/interim/reddit")
OUTPUT_PARQUET = OUTPUT_DIR / "moderated_posts_scores.parquet"
OUTPUT_CSV     = OUTPUT_DIR / "moderated_posts_scores.csv"

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"
API_TIMEOUT_SEC   = 30
API_RETRIES       = 3
API_BACKOFF_SEC   = 2.0
API_SLEEP_SEC     = 0.35
CHUNK_SIZE        = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _chunk(lst, size):
    """Yield fixed-size chunks from a sequence."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _safe_request(path: str, params: Optional[Dict] = None) -> Any:
    """Run request with retry and error handling."""
    url = f"{ARCTIC_SHIFT_BASE}{path}"
    params = {k: v for k, v in (params or {}).items() if v is not None}
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT_SEC)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                log.warning("HTTP %s per %s", exc.response.status_code, url)
                return None
            if attempt < API_RETRIES:
                time.sleep(API_BACKOFF_SEC * attempt)
        except requests.RequestException as exc:
            if attempt < API_RETRIES:
                time.sleep(API_BACKOFF_SEC * attempt)
            else:
                log.warning("Richiesta fallita: %s", exc)
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
        return [payload]
    return []


def _parse_post(item: Dict) -> Dict:
    """Parse post from the supplied input."""
    created_utc = item.get("created_utc")
    title    = str(item.get("title", "")).strip() or None
    selftext = str(item.get("selftext", "")).strip() or None
    full_text = f"{title}\n\n{selftext}" if title and selftext else (title or selftext)
    return {
        "item_id": item.get("name") or item.get("id"),
        "author":                item.get("author"),
        "subreddit":             item.get("subreddit"),
        "title":                 title,
        "selftext":              selftext,
        "full_text":             full_text,
        "url":                   item.get("url"),
        "domain":                item.get("domain"),
        "score":                 item.get("score",            np.nan),
        "upvote_ratio":          item.get("upvote_ratio",     np.nan),
        "num_comments":          item.get("num_comments",     np.nan),
        "created_utc":           created_utc,
        "created_dt":            pd.to_datetime(float(created_utc), unit="s", utc=True) if created_utc else pd.NaT,
        "removed_by_category":   item.get("removed_by_category"),
        "removal_reason":        item.get("removal_reason"),
        "banned_by":             item.get("banned_by"),
        "locked":                item.get("locked", False),
        "over_18":               item.get("over_18", False),
        "is_self":               item.get("is_self"),
        "link_flair_text":       item.get("link_flair_text"),
        "author_flair_text":     item.get("author_flair_text"),
        "permalink":             item.get("permalink"),
    }


def fetch_posts(item_ids: List[str]) -> pd.DataFrame:
    """Fetch posts from the remote API."""
    records = []
    for chunk in tqdm(list(_chunk(item_ids, CHUNK_SIZE)), desc="Fetch post da Arctic Shift"):
        payload = _safe_request("/api/posts/ids", params={"ids": ",".join(chunk)})
        items = _normalize_payload(payload)
        records.extend([_parse_post(it) for it in items])
        time.sleep(API_SLEEP_SEC)
    return pd.DataFrame(records) if records else pd.DataFrame()


def run():
    """Run the configured analysis workflow."""
    # 1. Carica i post moderati
    votes = pd.read_parquet(VOTES_PATH)
    moderated = (
        votes[["item_id", "community", "label"]]
        .drop_duplicates(subset="item_id")
        .reset_index(drop=True)
    )
    log.info("Post moderati unici: %d", len(moderated))

    # Skip item IDs already fetched.
    existing = pd.DataFrame()
    if OUTPUT_PARQUET.exists():
        existing = pd.read_parquet(OUTPUT_PARQUET)
        done = set(existing["item_id"].dropna())
        moderated = moderated[~moderated["item_id"].isin(done)]
        log.info("Resume: %d present, %d remaining", len(done), len(moderated))

    # 3. Fetch da Arctic Shift
    if not moderated.empty:
        fetched = fetch_posts(moderated["item_id"].tolist())
        fetched = moderated.merge(fetched, on="item_id", how="left")
        result = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
    else:
        result = existing

    log.info("Post con score disponibile: %d / %d", result["score"].notna().sum(), len(result))
    print(result[["item_id", "subreddit", "score"]].to_string())
    print(("Score == 0: %d | Score != 0: %d | NaN: %d",
         (result["score"] == 0).sum(),
         (result["score"] != 0).sum(),
         result["score"].isna().sum()))
    # 4. Salva
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PARQUET, index=False)
    result.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved to:\n  %s\n  %s", OUTPUT_PARQUET, OUTPUT_CSV)

    return result


if __name__ == "__main__":
    run()
