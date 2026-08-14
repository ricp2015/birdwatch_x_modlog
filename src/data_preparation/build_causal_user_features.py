"""Join temporal Reddit user features to votes without using future activity.

Monthly contribution snapshots are considered available only from the first day
of the following month. Reply events use a strict event timestamp < vote timestamp
lookup. Collection-time contribution scores are excluded by default because they
are not historical observations; enable them only for an explicit ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable


DEFAULT_VOTES = Path("data/processed/final_intersection_dataset.csv")
DEFAULT_FEATURE_ROOT = Path("data/interim/reddit/user_temporal_features")
DEFAULT_METADATA = Path("data/processed/user_metadata.csv")
DEFAULT_OUTPUT = Path("data/interim/reddit/causal_user_vote_features.parquet")

KEY_COLUMNS = ["username", "item_id", "timestamp"]


def _read_parquet_parts(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a directory of Parquet parts or a single Parquet file."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def _user_key(values: pd.Series) -> pd.Series:
    """Build a case-insensitive Reddit username join key."""
    # Use plain Python strings rather than pandas StringDtype. Recent pandas
    # versions distinguish StringDtype(na_value=<NA>) from
    # StringDtype(na_value=nan), which makes merge_asof reject otherwise
    # identical keys read from CSV and Parquet.
    return values.map(lambda value: str(value).casefold() if pd.notna(value) else "").astype(
        object
    )


def _subreddit_key(values: pd.Series) -> pd.Series:
    """Normalize both `foo` and `r/foo` community representations."""
    return values.map(
        lambda value: (
            (
                str(value).strip()[2:]
                if str(value).strip().casefold().startswith("r/")
                else str(value).strip()
            ).casefold()
            if pd.notna(value)
            else ""
        )
    ).astype(object)


def _month_available_at(values: pd.Series) -> pd.Series:
    """Return UTC epoch seconds at which a completed monthly snapshot is usable."""
    parsed = pd.to_datetime(values.astype("string") + "-01", utc=True, errors="coerce")
    available = parsed + pd.offsets.MonthBegin(1)
    # Do not infer the datetime storage unit (ns/us/ms) from astype(int64): it
    # varies across pandas/Arrow versions. Timestamp.timestamp() always returns
    # Unix seconds and therefore matches vote timestamps.
    return available.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)


def _asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: list[str],
    right_time: str,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Perform a backward, strictly-prior as-of join and restore vote row order."""
    feature_columns = list(feature_columns)
    left = left.copy()
    right = right.copy()
    left["timestamp"] = pd.to_numeric(left["timestamp"], errors="coerce").astype("float64")
    right[right_time] = pd.to_numeric(right[right_time], errors="coerce").astype("float64")
    # Enforce the exact same traditional object dtype on both sides. This is
    # deliberately redundant with key creation because Parquet reads can
    # restore pandas' nullable string dtype.
    for column in by:
        left[column] = left[column].astype(object)
        right[column] = right[column].astype(object)
    right_columns = by + [right_time] + feature_columns
    right = right[right_columns].dropna(subset=by + [right_time]).copy()
    right = right.sort_values([right_time] + by).drop_duplicates(by + [right_time], keep="last")
    ordered = left.sort_values(["timestamp"] + by).copy()
    joined = pd.merge_asof(
        ordered,
        right,
        left_on="timestamp",
        right_on=right_time,
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return joined.sort_values("_vote_row").reset_index(drop=True)


def _build_interaction_events(
    reply_graph: pd.DataFrame,
    by_subreddit: bool = False,
) -> pd.DataFrame:
    """Create cumulative undirected global or user-subreddit interaction states."""
    edges = reply_graph.dropna(subset=["from_user", "to_user", "created_utc"]).copy()
    edges["created_utc"] = pd.to_numeric(edges["created_utc"], errors="coerce")
    edges = edges.dropna(subset=["created_utc"])

    base_columns = ["from_user", "to_user", "created_utc"]
    if by_subreddit:
        base_columns.append("subreddit")
    outgoing = edges[base_columns].rename(columns={"from_user": "_user_key", "to_user": "partner"})
    incoming_columns = ["to_user", "from_user", "created_utc"]
    if by_subreddit:
        incoming_columns.append("subreddit")
    incoming = edges[incoming_columns].rename(
        columns={"to_user": "_user_key", "from_user": "partner"}
    )
    events = pd.concat([outgoing, incoming], ignore_index=True)
    events["_user_key"] = _user_key(events["_user_key"])
    events["partner"] = _user_key(events["partner"])
    if by_subreddit:
        events["_subreddit_key"] = _subreddit_key(events["subreddit"])
    events = events[events["_user_key"] != events["partner"]]
    group_columns = ["_user_key", "_subreddit_key"] if by_subreddit else ["_user_key"]
    events = events.sort_values(group_columns + ["created_utc"])

    rows: list[dict] = []
    grouped = events.groupby(
        group_columns if by_subreddit else "_user_key",
        sort=False,
    )
    for group_key, group in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Building local interaction states"
        if by_subreddit
        else "Building interaction states",
        unit="group" if by_subreddit else "user",
    ):
        if by_subreddit:
            username, subreddit = group_key
        else:
            username, subreddit = group_key, None
        partners: set[str] = set()
        total = 0
        # Collapse ties so all events at one instant form one causal state.
        for timestamp, at_time in group.groupby("created_utc", sort=True):
            total += len(at_time)
            partners.update(at_time["partner"].dropna().tolist())
            row = {
                "_user_key": username,
                "interaction_state_at": float(timestamp),
                "prior_n_interaction_partners": len(partners),
                "prior_total_interactions": total,
            }
            if by_subreddit:
                row["_subreddit_key"] = subreddit
                row["local_interaction_state_at"] = row.pop("interaction_state_at")
                row["prior_n_interaction_partners_in_sub"] = row.pop(
                    "prior_n_interaction_partners"
                )
                row["prior_total_interactions_in_sub"] = row.pop("prior_total_interactions")
            rows.append(row)
    return pd.DataFrame(rows)


def build_features(
    votes_path: Path,
    feature_root: Path,
    metadata_path: Path,
    include_collection_score: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build causal per-vote features and return them with a QA report."""
    progress = tqdm(total=7, desc="Building causal features", unit="step")
    try:
        progress.set_postfix_str("loading votes")
        votes = pd.read_csv(votes_path, low_memory=False)
        required = {"username", "item_id", "timestamp", "community"}
        missing = required - set(votes.columns)
        if missing:
            raise ValueError(f"Votes are missing required columns: {sorted(missing)}")

        work = votes[["username", "item_id", "timestamp", "community"]].copy()
        work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
        if work["timestamp"].isna().any():
            raise ValueError(
                f"Votes contain {work['timestamp'].isna().sum():,} invalid timestamps"
            )
        work["_vote_row"] = np.arange(len(work), dtype=np.int64)
        work["_user_key"] = _user_key(work["username"])
        work["_subreddit_key"] = _subreddit_key(work["community"])
        progress.update()

        progress.set_postfix_str("global snapshots")
        global_snapshots = _read_parquet_parts(feature_root / "user_global_snapshots")
        global_snapshots["_user_key"] = _user_key(global_snapshots["username"])
        global_snapshots["global_available_at"] = _month_available_at(
            global_snapshots["year_month"]
        )
        global_features = [
            "cum_n_posts",
            "cum_n_comments",
            "cum_n_contributions",
            "cum_n_distinct_subreddits",
            "cum_n_replies_made",
        ]
        if include_collection_score:
            global_features.append("cum_karma_sum")
        global_snapshots = global_snapshots.rename(
            columns={column: f"prior_{column.removeprefix('cum_')}" for column in global_features}
        )
        renamed_global_features = [
            f"prior_{column.removeprefix('cum_')}" for column in global_features
        ]
        work = _asof_join(
            work,
            global_snapshots,
            ["_user_key"],
            "global_available_at",
            renamed_global_features,
        )
        progress.update()

        progress.set_postfix_str("subreddit snapshots")
        sub_snapshots = _read_parquet_parts(feature_root / "user_subreddit_snapshots")
        sub_snapshots["_user_key"] = _user_key(sub_snapshots["username"])
        sub_snapshots["_subreddit_key"] = _subreddit_key(sub_snapshots["subreddit"])
        sub_snapshots["subreddit_available_at"] = _month_available_at(sub_snapshots["year_month"])
        sub_features = [
            "cum_n_posts_in_sub",
            "cum_n_comments_in_sub",
            "cum_n_months_active_in_sub",
        ]
        if include_collection_score:
            sub_features.append("cum_karma_sum_in_sub")
        sub_snapshots = sub_snapshots.rename(
            columns={column: f"prior_{column.removeprefix('cum_')}" for column in sub_features}
        )
        renamed_sub_features = [f"prior_{column.removeprefix('cum_')}" for column in sub_features]
        work = _asof_join(
            work,
            sub_snapshots,
            ["_user_key", "_subreddit_key"],
            "subreddit_available_at",
            renamed_sub_features,
        )
        progress.update()

        progress.set_postfix_str("reply graph")
        reply_graph = _read_parquet_parts(
            feature_root / "reply_graph",
            columns=["from_user", "to_user", "subreddit", "created_utc"],
        )
        interaction_events = _build_interaction_events(reply_graph)
        if interaction_events.empty:
            work["interaction_state_at"] = np.nan
            work["prior_n_interaction_partners"] = np.nan
            work["prior_total_interactions"] = np.nan
        else:
            work = _asof_join(
                work,
                interaction_events,
                ["_user_key"],
                "interaction_state_at",
                ["prior_n_interaction_partners", "prior_total_interactions"],
            )
        local_interaction_events = _build_interaction_events(reply_graph, by_subreddit=True)
        if local_interaction_events.empty:
            work["local_interaction_state_at"] = np.nan
            work["prior_n_interaction_partners_in_sub"] = np.nan
            work["prior_total_interactions_in_sub"] = np.nan
        else:
            work = _asof_join(
                work,
                local_interaction_events,
                ["_user_key", "_subreddit_key"],
                "local_interaction_state_at",
                ["prior_n_interaction_partners_in_sub", "prior_total_interactions_in_sub"],
            )
        progress.update()

        progress.set_postfix_str("account tenure")
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path, low_memory=False)
            metadata["_user_key"] = _user_key(metadata["username"])
            metadata["account_created_utc"] = pd.to_numeric(
                metadata["account_created_utc"], errors="coerce"
            )
            metadata = metadata.drop_duplicates("_user_key", keep="last")
            work = work.merge(
                metadata[["_user_key", "account_created_utc"]],
                on="_user_key",
                how="left",
            )
            work["tenure_days_at_vote"] = (
                work["timestamp"] - work["account_created_utc"]
            ) / 86400.0
            work.loc[work["tenure_days_at_vote"] < 0, "tenure_days_at_vote"] = np.nan
        else:
            work["account_created_utc"] = np.nan
            work["tenure_days_at_vote"] = np.nan
        progress.update()

        # Missing prior history means a genuine zero for count features, while
        # availability columns retain the distinction for coverage analysis.
        causal_feature_columns = (
            renamed_global_features
            + renamed_sub_features
            + [
                "prior_n_interaction_partners",
                "prior_total_interactions",
                "prior_n_interaction_partners_in_sub",
                "prior_total_interactions_in_sub",
            ]
        )
        work[causal_feature_columns] = work[causal_feature_columns].fillna(0)
        progress.update()

        violations = {
            "global_snapshot_not_prior": int(
                (
                    work["global_available_at"].notna()
                    & (work["global_available_at"] >= work["timestamp"])
                ).sum()
            ),
            "subreddit_snapshot_not_prior": int(
                (
                    work["subreddit_available_at"].notna()
                    & (work["subreddit_available_at"] >= work["timestamp"])
                ).sum()
            ),
            "interaction_not_prior": int(
                (
                    work["interaction_state_at"].notna()
                    & (work["interaction_state_at"] >= work["timestamp"])
                ).sum()
            ),
            "local_interaction_not_prior": int(
                (
                    work["local_interaction_state_at"].notna()
                    & (work["local_interaction_state_at"] >= work["timestamp"])
                ).sum()
            ),
        }
        report = {
            "n_votes": len(work),
            "n_users": int(work["_user_key"].nunique()),
            "coverage": {
                "global_snapshot": float(work["global_available_at"].notna().mean()),
                "subreddit_snapshot": float(work["subreddit_available_at"].notna().mean()),
                "interaction_state": float(work["interaction_state_at"].notna().mean()),
                "local_interaction_state": float(
                    work["local_interaction_state_at"].notna().mean()
                ),
                "account_created": float(work["account_created_utc"].notna().mean()),
            },
            "anti_leakage_violations": violations,
            "include_collection_score": include_collection_score,
        }
        if any(violations.values()):
            raise AssertionError(f"Anti-leakage checks failed: {violations}")
        progress.update()

        output_columns = (
            KEY_COLUMNS
            + ["community"]
            + causal_feature_columns
            + [
                "tenure_days_at_vote",
                "global_available_at",
                "subreddit_available_at",
                "interaction_state_at",
                "local_interaction_state_at",
            ]
        )
        return work[output_columns], report
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes", type=Path, default=DEFAULT_VOTES)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-collection-score",
        action="store_true",
        help="Include non-historical contribution score sums for an explicit ablation.",
    )
    parser.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()

    features, report = build_features(
        args.votes, args.feature_root, args.metadata, args.include_collection_score
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    report_path = args.output.with_suffix(".qa.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sample_path = args.output.with_suffix(".qa_sample.csv")
    features.sample(min(args.sample_size, len(features)), random_state=10).to_csv(
        sample_path, index=False
    )

    print(json.dumps(report, indent=2))
    print(f"Features: {args.output}")
    print(f"QA report: {report_path}")
    print(f"Manual-check sample: {sample_path}")


if __name__ == "__main__":
    main()
