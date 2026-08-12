"""
Scrapes subreddit rules using Reddit's public JSON endpoint.


Input: CSV or parquet with a column named `community` or `subreddit`.

Output:
data/raw/subreddit_rules.parquet     one row per rule
data/raw/subreddit_rules.jsonl       same, for easy manual inspection
data/raw/subreddit_rules_errors.csv  subreddits that failed (banned/private/404)

Output schema: 
subreddit    str   e.g. "AskReddit"
rule_index   int   0-based position in the subreddit rule list
short_name   str   rule title
description  str   full description (may be empty)
kind         str   "link" | "comment" | "all"
has_rules    bool  False for subreddits that exist but have no rules
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DELAY = 2.0   # seconds between requests; don't go below 1.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; subreddit_rules_scraper/1.0; "
        "academic research; no personal data collected)"
    )
}


# Load subreddit list from dataset

def load_subreddit_list(votes_path: Path) -> List[str]:
    """Load subreddit list from its configured source."""
    p = Path(votes_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    col = next((c for c in df.columns if c.lower() in ("community", "subreddit")), None)
    if col is None:
        raise ValueError(
            f"No 'community'/'subreddit' column found. Columns: {list(df.columns)}"
        )

    raw = df[col].dropna().astype(str).unique().tolist()

    # normalise: strip leading slashes and "r/" prefix
    cleaned = []
    for s in raw:
        s = s.strip().lstrip("/")
        if s.lower().startswith("r/"):
            s = s[2:]
        if s:
            cleaned.append(s)

    cleaned = sorted(set(cleaned))
    log.info("Found %d unique subreddits in dataset", len(cleaned))
    return cleaned


# Fetch rules for a single subreddit (no auth)

def fetch_rules(subreddit: str, session: requests.Session) -> List[Dict]:
    """Fetch rules from the remote API."""
    url = f"https://www.reddit.com/r/{subreddit}/about/rules.json"
    resp = session.get(url, headers=HEADERS, timeout=15)

    if resp.status_code == 404:
        raise ValueError(f"Not found (404): r/{subreddit}")
    if resp.status_code == 403:
        raise ValueError(f"Private or banned (403): r/{subreddit}")
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429) - increase --delay")
    resp.raise_for_status()

    rules_raw = resp.json().get("rules", [])

    rows = []
    for idx, rule in enumerate(rules_raw):
        rows.append({
            "subreddit":   subreddit,
            "rule_index":  idx,
            "short_name":  (rule.get("short_name") or "").strip(),
            "description": (rule.get("description") or "").strip(),
            "kind":        (rule.get("kind") or "all").strip(),
        })
    return rows


# Main scraping loop with resume support

def scrape_all(
    subreddits: List[str],
    out_dir:    Path,
    delay:      float = DEFAULT_DELAY,
    resume:     bool  = True,
) -> pd.DataFrame:
    """Collect all from the remote source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path  = out_dir / "subreddit_rules.jsonl"
    errors_path = out_dir / "subreddit_rules_errors.csv"

    # resume: find already-processed subreddits
    done: set[str] = set()
    if resume and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["subreddit"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info("Resume mode: %d subreddits already in JSONL", len(done))

    prior_errors: List[Dict] = []
    if resume and errors_path.exists():
        prior_errors = pd.read_csv(errors_path).to_dict("records")
        done |= {e["subreddit"] for e in prior_errors}

    todo = [s for s in subreddits if s not in done]
    log.info("%d subreddits left to scrape", len(todo))

    errors: List[Dict] = list(prior_errors)
    session = requests.Session()

    with open(jsonl_path, "a", encoding="utf-8") as fh:
        for sub in tqdm(todo, desc="Scraping rules", unit="sub"):
            try:
                rows = fetch_rules(sub, session)

                # subreddit exists but has no rules -> write sentinel row
                if not rows:
                    rows = [{
                        "subreddit":   sub,
                        "rule_index":  -1,
                        "short_name":  "",
                        "description": "",
                        "kind":        "none",
                    }]

                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()

            except RuntimeError:
                # rate-limit: back off 30 s and retry once
                log.warning("Rate limited - sleeping 30 s then retrying r/%s", sub)
                time.sleep(30)
                try:
                    rows = fetch_rules(sub, session)
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                except Exception as exc2:
                    log.warning("Retry failed r/%s: %s", sub, exc2)
                    errors.append({"subreddit": sub, "error": str(exc2)})

            except Exception as exc:
                log.warning("Failed r/%-30s %s", sub, exc)
                errors.append({"subreddit": sub, "error": str(exc)})

            time.sleep(delay)

    # save errors
    if errors:
        pd.DataFrame(errors).drop_duplicates("subreddit").to_csv(errors_path, index=False)
        log.info("%d failures logged to %s", len(errors), errors_path)

    # consolidate JSONL -> parquet
    all_rows: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["subreddit", "rule_index", "short_name", "description", "kind"]
    )
    df["has_rules"] = df["rule_index"] >= 0

    parquet_path = out_dir / "subreddit_rules.parquet"
    df.to_parquet(parquet_path, index=False)

    n_ok = df.loc[df["has_rules"], "subreddit"].nunique()
    log.info(
        "Saved %d rules across %d subreddits -> %s",
        int(df["has_rules"].sum()), n_ok, parquet_path,
    )
    return df


# CLI

def parse_args() -> argparse.Namespace:
    """Parse args from the supplied input."""
    p = argparse.ArgumentParser(
        description="Scrape subreddit rules - no API credentials needed"
    )
    p.add_argument(
        "--votes",
        default="data/processed/final_intersection_dataset.csv",
        help="CSV/parquet with a 'community' or 'subreddit' column",
    )
    p.add_argument(
        "--out_dir", default="data/raw",
        help="Output directory (default: data/raw)",
    )
    p.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between requests (default: {DEFAULT_DELAY}). Min: 1.0",
    )
    p.add_argument(
        "--no_resume", action="store_true",
        help="Scrape from scratch, ignoring any existing output",
    )
    return p.parse_args()


def main() -> None:
    """Run the command-line workflow."""
    args = parse_args()
    subs = load_subreddit_list(Path(args.votes))
    df   = scrape_all(
        subreddits = subs,
        out_dir    = Path(args.out_dir),
        delay      = args.delay,
        resume     = not args.no_resume,
    )
    print("\nPreview (first 10 rules with content):")
    preview = df[df["has_rules"] & (df["short_name"] != "")].head(10)
    print(preview[["subreddit", "rule_index", "short_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
