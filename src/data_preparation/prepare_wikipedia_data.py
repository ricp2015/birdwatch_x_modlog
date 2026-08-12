from pathlib import Path
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = Path("data/interim/wikipedia/standard")
# density thresholds (Birdwatch)
MIN_VOTES_PER_DEBATE = 5
MIN_VOTES_PER_USER   = 10
# chronological split ratios (sum to 1)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# vote-label -> binary vote mapping, only keep/delete variants are retained
KEEP_LABELS   = {"keep", "keep speedy"}
DELETE_LABELS = {"delete", "delete speedy"}
# outcome-label -> binary ground truth mapping
KEEP_OUTCOMES   = {"keep", "keep speedy"}
DELETE_OUTCOMES = {"delete", "delete speedy"}

# early-votes mode: if not None, only the first MAX_VOTES_PER_DEBATE votes
# (ordered by timestamp) are kept per debate before density filtering.
# This breaks the tautological correlation between vote majority and outcome,
# turning CN into an early-detection task: can bridging patterns in early votes
# predict the final admin decision before consensus forms?
# Recommended values to sweep: 5, 10, 20, 50 (None = all votes, baseline).
MAX_VOTES_PER_DEBATE: Optional[int] = None

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# helper functions
# convert a Wikipedia account creation date to a Unix timestamp
def _parse_signup(signup: Any) -> Optional[float]:
    """Parse signup from the supplied input."""
    if signup is None or (isinstance(signup, float) and np.isnan(signup)):
        return np.nan
    try:
        dt = pd.to_datetime(signup, utc=True)
        return dt.timestamp()
    except Exception:
        return np.nan

# convert a Unix creation timestamp to account age in days
def _compute_tenure(created_ts: Any) -> float:
    """Compute tenure from the supplied data."""
    if created_ts is None or (isinstance(created_ts, float) and np.isnan(created_ts)):
        return np.nan
    return (time.time() - float(created_ts)) / 86400


def _shannon_entropy(counts: np.ndarray) -> float:
    """Calculate Shannon entropy for observed category counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

# 1. Load ConvoKit corpus and build base dataframes
def load_corpus() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load corpus from its configured source."""
    try:
        from convokit import Corpus, download
    except ImportError as exc:
        raise ImportError(
            "convokit is required. Install it with: pip install convokit"
        ) from exc
    log.info("Loading Wikipedia AfD corpus via ConvoKit...")
    corpus = Corpus(filename=download("wiki-articles-for-deletion-corpus"))

    speaker_records = []
    for uid, speaker in corpus.speakers.items():
        meta = speaker.meta or {}
        speaker_records.append(
            {
                "speaker_id": uid,
                "name":       speaker.id,
                "editcount":  meta.get("editcount"),
                "signup":     meta.get("signup"),
                "gender":     meta.get("gender"),
            }
        )
    df_speakers = pd.DataFrame(speaker_records)

    utt_records = []
    for utt in corpus.iter_utterances():
        meta = utt.meta or {}
        utt_records.append(
            {
                "utt_id":          utt.id,
                "speaker_id":      utt.speaker.id if utt.speaker else None,
                "conversation_id": utt.conversation_id,
                "timestamp":       utt.timestamp,
                "text":            utt.text,
                "type":            meta.get("type"),
                "vote_label":      meta.get("label"),      # renamed: vote text label
                "raw_label":       meta.get("raw_label"),
                "citations":       meta.get("citations"),
            }
        )
    df_utts = pd.DataFrame(utt_records)
    if not df_utts.empty and "timestamp" in df_utts.columns:
        df_utts["datetime"] = pd.to_datetime(df_utts["timestamp"], unit="s", utc=True, errors="coerce")
        df_utts["year"]     = df_utts["datetime"].dt.year.astype("Int32")
        df_utts["n_citations"] = df_utts["citations"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    conv_records = []
    for conv in corpus.iter_conversations():
        meta = conv.meta or {}
        conv_records.append(
            {
                "conversation_id":           conv.id,
                "outcome_label":             meta.get("outcome_label"),
                "outcome_raw_label":         meta.get("outcome_raw_label"),
                "outcome_decision_maker_id": meta.get("outcome_decision_maker_id"),
                "outcome_timestamp":         meta.get("outcome_timestamp"),
                "outcome_rationale":         meta.get("outcome_rationale"),
            }
        )
    df_convs = pd.DataFrame(conv_records)
    if not df_convs.empty and "outcome_timestamp" in df_convs.columns:
        df_convs["outcome_dt"]   = pd.to_datetime(
            df_convs["outcome_timestamp"], unit="s", utc=True, errors="coerce"
        )
        df_convs["outcome_year"] = df_convs["outcome_dt"].dt.year

    log.info(
        "Corpus loaded: %d speakers | %d utterances | %d debates",
        len(df_speakers),
        len(df_utts),
        len(df_convs),
    )
    return df_speakers, df_utts, df_convs

# 2. Build vote matrix
# construct the user x debate vote dataframe from the raw utterances
def build_vote_matrix(
    df_utts: pd.DataFrame,
    df_convs: pd.DataFrame,
) -> pd.DataFrame:
    """Build vote matrix from the supplied data."""
    log.info("Building vote matrix...")
    votes = df_utts[df_utts["type"] == "vote"].copy()
    vote_label_lower = votes["vote_label"].str.lower().str.strip()
    keep_mask   = vote_label_lower.isin(KEEP_LABELS)
    delete_mask = vote_label_lower.isin(DELETE_LABELS)
    votes = votes[keep_mask | delete_mask].copy()
    votes["vote"] = np.where(vote_label_lower[votes.index].isin(KEEP_LABELS), 1, -1)
    votes["vote"] = votes["vote"].astype("int8")
    log.info("  Vote utterances after binary filter: %d", len(votes))

    # build binary_convs by working entirely on a self-contained normalised column
    # to avoid any index-misalignment risk when selecting outcome labels
    df_convs = df_convs.copy()
    df_convs["outcome_lower"] = df_convs["outcome_label"].str.lower().str.strip()
    binary_convs = df_convs[
        df_convs["outcome_lower"].isin(KEEP_OUTCOMES | DELETE_OUTCOMES)
    ].copy()
    binary_convs["label"] = np.where(
        binary_convs["outcome_lower"].isin(KEEP_OUTCOMES), 1, -1
    ).astype("int8")
    log.info("  Debates with binary outcome: %d / %d", len(binary_convs), len(df_convs))

    votes = votes.merge(
        binary_convs[["conversation_id", "label"]],
        on="conversation_id",
        how="inner",
    )

    votes = votes.rename(columns={
        "speaker_id":      "username",
        "conversation_id": "item_id",
    })

    # parse timestamp once; field comes in as Unix seconds from ConvoKit
    votes["timestamp"] = pd.to_datetime(
        votes["timestamp"],
        unit="s",
        utc=True,
        errors="coerce"
    )
    log.info(
        "Timestamp year distribution BEFORE cleaning:\n%s",
        votes["timestamp"].dt.year.value_counts().sort_index().head(10).to_string()
    )

    # keep only valid time range
    min_year = 2005
    invalid_time_mask = (
        votes["timestamp"].isna() |
        (votes["timestamp"].dt.year < min_year)
    )
    n_invalid = invalid_time_mask.sum()
    if n_invalid > 0:
        log.info("Dropping %d votes with timestamp < %d or invalid", n_invalid, min_year)
    votes = votes.loc[~invalid_time_mask].copy()

    core_cols = ["username", "item_id", "vote", "label", "timestamp", "utt_id"]
    votes = votes[[c for c in core_cols if c in votes.columns]].copy()
    required = ["username", "item_id", "vote", "label", "timestamp"]
    missing  = [c for c in required if c not in votes.columns]
    if missing:
        raise ValueError(f"Missing columns before dropna: {missing}")

    before = len(votes)
    votes  = votes.dropna(subset=required)
    log.info("Dropped %d rows with nulls in core columns", before - len(votes))

    before = len(votes)
    votes  = (
        votes.sort_values("timestamp")
        .drop_duplicates(subset=["username", "item_id"], keep="last")
    )
    log.info("Dropped %d duplicate (username, item_id) pairs", before - len(votes))
    votes = votes.reset_index(drop=True)
    log.info(
        "Vote matrix: %d votes | %d unique users | %d unique debates",
        len(votes),
        votes["username"].nunique(),
        votes["item_id"].nunique(),
    )

    # alignment diagnostic
    alignment = votes.groupby("label")["vote"].mean()
    log.info("Vote-label alignment (avg vote per label):\n%s", alignment.to_string())
    if alignment.get(1, 0) < alignment.get(-1, 0):
        log.warning(
            "Polarity check FAILED: avg vote for label=+1 (%.4f) < label=-1 (%.4f)",
            alignment.get(1, 0), alignment.get(-1, 0),
        )

    return votes


# 2b. Early-votes truncation
def truncate_to_first_k_votes(
    df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Limit to first k votes to the requested size."""
    log.info("Truncating to first %d votes per debate...", k)
    before = len(df)
    df_sorted = df.sort_values(["item_id", "timestamp"])
    df_truncated = (
        df_sorted
        .groupby("item_id", sort=False)
        .head(k)
        .reset_index(drop=True)
    )
    log.info(
        "After truncation: %d -> %d votes | %d debates | %d users",
        before,
        len(df_truncated),
        df_truncated["item_id"].nunique(),
        df_truncated["username"].nunique(),
    )
    # report how much the correlation drops relative to full data
    full_corr = float(np.corrcoef(
        df["vote"].astype(float), df["label"].astype(float)
    )[0, 1])
    trunc_corr = float(np.corrcoef(
        df_truncated["vote"].astype(float), df_truncated["label"].astype(float)
    )[0, 1])
    log.info(
        "Vote-label Pearson correlation | full votes: %.4f -> first-%d votes: %.4f",
        full_corr, k, trunc_corr,
    )
    return df_truncated


# 3. Density filtering
# iteratively prune the vote matrix until both density thresholds are satisfied (Birdwatch paper)
def apply_density_filter(
    df: pd.DataFrame,
    min_votes_per_debate: int = MIN_VOTES_PER_DEBATE,
    min_votes_per_user:   int = MIN_VOTES_PER_USER,
) -> pd.DataFrame:
    """Apply density filter to the supplied data."""

    log.info(
        "Applying density filter: >=%d votes/debate, >=%d votes/user",
        min_votes_per_debate,
        min_votes_per_user,
    )
    filtered = df.copy()
    for iteration in range(1, 100):
        prev_len = len(filtered)
        valid_debates = filtered["item_id"].value_counts()
        valid_debates = valid_debates[valid_debates >= min_votes_per_debate].index
        filtered = filtered[filtered["item_id"].isin(valid_debates)]
        valid_users = filtered["username"].value_counts()
        valid_users = valid_users[valid_users >= min_votes_per_user].index
        filtered = filtered[filtered["username"].isin(valid_users)]
        log.info("  Iteration %d: %d -> %d votes", iteration, prev_len, len(filtered))
        if len(filtered) == prev_len:
            break

    log.info(
        "After density filter: %d votes | %d users | %d debates",
        len(filtered),
        filtered["username"].nunique(),
        filtered["item_id"].nunique(),
    )

    # post-filter diagnostics
    corr = float(np.corrcoef(filtered["vote"].astype(float), filtered["label"].astype(float))[0, 1])
    log.info("Vote-label Pearson correlation (post-filter): %.4f", corr)
    alignment = filtered.groupby("label")["vote"].mean()
    log.info("Avg vote per label (post-filter):\n%s", alignment.to_string())

    return filtered.reset_index(drop=True)

# 4. User metadata
def build_user_metadata(
    usernames: List[str],
    df_speakers: pd.DataFrame,
    df_utts: pd.DataFrame,
) -> pd.DataFrame:
    """Build user metadata from the supplied data."""
    log.info("Building user metadata...")

    username_set = set(usernames)
    vote_utts = df_utts[
        (df_utts["type"] == "vote") &
        (df_utts["speaker_id"].isin(username_set))
    ].copy()
    vote_utts["label_lower"] = vote_utts["vote_label"].str.lower().str.strip()
    agg = vote_utts.groupby("speaker_id").agg(
        n_votes=("utt_id", "count"),
        n_citations=("n_citations", "sum"),
        active_debates=("conversation_id", "nunique"),
    )
    delete_mask = vote_utts["label_lower"].isin(DELETE_LABELS)
    delete_rate = vote_utts.assign(delete=delete_mask).groupby("speaker_id")["delete"].mean()
    agg["delete_rate"] = delete_rate
    agg["citation_rate"] = agg["n_citations"] / agg["n_votes"]
    vote_utts["month"] = vote_utts["datetime"].dt.to_period("M")
    monthly_counts = vote_utts.groupby(["speaker_id", "month"]).size()

    def entropy_from_series(s):
        """Calculate entropy from a pandas series."""
        return _shannon_entropy(s.values.astype(float))
    
    temporal_entropy = monthly_counts.groupby("speaker_id").apply(entropy_from_series)
    agg["temporal_entropy"] = temporal_entropy

    def policy_entropy_fn(subdf):
        """Calculate policy entropy for one user group."""
        counts = {}
        for cit_list in subdf["citations"].dropna():
            if isinstance(cit_list, list):
                for p in cit_list:
                    counts[p] = counts.get(p, 0) + 1
        if not counts:
            return np.nan
        return _shannon_entropy(np.array(list(counts.values()), dtype=float))
    policy_entropy = vote_utts.groupby("speaker_id").apply(policy_entropy_fn)
    agg["policy_entropy"] = policy_entropy
    speaker_map = df_speakers.set_index("speaker_id")
    meta = speaker_map.reindex(usernames)[["editcount", "signup", "gender"]]
    meta["signup_ts"] = meta["signup"].apply(_parse_signup)
    meta["tenure_days"] = meta["signup_ts"].apply(_compute_tenure)
    df_meta = (
        meta.join(agg, how="left")
        .reset_index()
        .rename(columns={"speaker_id": "username"})
    )
    log.info("User metadata built: %d users", len(df_meta))
    return df_meta

# left-join user metadata onto the vote dataframe
def merge_user_metadata(df: pd.DataFrame, user_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge user metadata into the working data."""
    enriched = df.merge(user_meta, on="username", how="left")
    missing = enriched["editcount"].isna().sum()
    log.info(
        "Votes without metadata (no editcount): %d / %d (%.1f%%)",
        missing,
        len(enriched),
        100 * missing / len(enriched),
    )
    return enriched

# 5. Chronological split
# split the vote dataframe chronologically into train/val/test sets, partitioned by first-vote time of each debate
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
) -> Dict[str, pd.DataFrame]:
    """Create item-level train, validation, and test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Split ratios must sum to 1.0"

    debate_times = (
        df.groupby("item_id")["timestamp"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"timestamp": "debate_time"})
    )

    n         = len(debate_times)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    debate_times["split"] = "test"
    debate_times.iloc[:train_end, debate_times.columns.get_loc("split")] = "train"
    debate_times.iloc[train_end:val_end, debate_times.columns.get_loc("split")] = "val"
    df_split = df.merge(debate_times[["item_id", "split"]], on="item_id", how="left")
    splits: Dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        subset = df_split[df_split["split"] == name].drop(columns="split").copy()
        splits[name] = subset.reset_index(drop=True)
        if len(subset) > 0:
            # timestamp is already a tz-aware datetime - call .date() directly
            log.info(
                "Split '%s': %d votes | %d users | %d debates | %s -> %s",
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

# 6. Summary
def print_summary(df: pd.DataFrame) -> None:
    """Print summary to the console."""
    log.info("--- Dataset summary ---")
    label_dist = df.drop_duplicates("item_id")["label"].value_counts()
    log.info("Debate outcome distribution (post-filter):\n%s", label_dist.to_string())
    vote_dist = df["vote"].value_counts()
    log.info("Vote distribution:\n%s", vote_dist.to_string())
    # timestamp is already a tz-aware datetime - no re-parsing needed
    year_dist = df["timestamp"].dt.year.value_counts().sort_index()
    log.info("Votes by year:\n%s", year_dist.to_string())

# 7. Saving
def save_outputs(
    filtered_df: pd.DataFrame,
    user_meta:   pd.DataFrame,
    splits:      Dict[str, pd.DataFrame],
    output_dir:  Path = OUTPUT_DIR,
    splits_dir:  Optional[Path] = None,
) -> None:
    """Write outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = splits_dir or Path("data/splits/wikipedia/standard/random")
    splits_dir.mkdir(exist_ok=True)

    filtered_df.to_parquet(output_dir / "filtered_votes.parquet", index=False)
    user_meta.to_parquet(  output_dir / "users.parquet",          index=False)

    for split_name, split_df in splits.items():
        split_df.to_parquet(splits_dir / f"{split_name}_votes.parquet", index=False)

    log.info("All outputs saved to %s", output_dir)


# Main function
def prepare_dataset(
    output_dir:           Path         = OUTPUT_DIR,
    splits_dir:           Optional[Path] = None,
    max_votes_per_debate: Optional[int] = MAX_VOTES_PER_DEBATE,
) -> Dict[str, Any]:
    """Prepare dataset for downstream use."""
    log.info("=" * 50)
    log.info(
        "STEP 1 - DATA PREPARATION  (Wikipedia AfD)  max_votes=%s",
        str(max_votes_per_debate) if max_votes_per_debate else "all",
    )
    log.info("=" * 50)
    df_speakers, df_utts, df_convs = load_corpus()
    df = build_vote_matrix(df_utts, df_convs)

    if max_votes_per_debate is not None:
        df = truncate_to_first_k_votes(df, max_votes_per_debate)

    df = apply_density_filter(df)
    usernames = df["username"].unique().tolist()
    user_meta = build_user_metadata(usernames, df_speakers, df_utts)
    df = merge_user_metadata(df, user_meta)
    print_summary(df)
    splits = chronological_split(df)
    save_outputs(df, user_meta, splits, output_dir, splits_dir)
    log.info("Step 1 complete.")
    return {
        "filtered_df": df,
        "user_meta":   user_meta,
        "splits":      splits,
    }

if __name__ == "__main__":
    prepare_dataset(
        output_dir=Path("data/interim/wikipedia/standard"),
        splits_dir=Path("data/splits/wikipedia/standard/random"),
        max_votes_per_debate=None,
    )

    # early-detection sweep
    for k in [5, 10, 20, 50]:
        prepare_dataset(
            output_dir=Path(f"data/interim/wikipedia/early/p{k:02d}"),
            splits_dir=Path(f"data/splits/wikipedia/early/p{k:02d}/random"),
            max_votes_per_debate=k,
        )
