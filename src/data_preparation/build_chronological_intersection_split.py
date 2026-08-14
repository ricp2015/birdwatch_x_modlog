"""Build the common chronological benchmark from the filtered intersection.

The input is the union of the existing random intersection partitions. Those
partitions already contain only items and users that survived the cascade of
all method-specific filters. Items, rather than individual votes, are ordered
by their completion time (last observed vote) so every vote for one moderation
case stays in the same partition and no completed training case ends after a
validation/test case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/splits/reddit/intersection")
DEFAULT_OUTPUT = Path("data/splits/reddit/intersection_chronological")


def load_intersection(split_dir: Path) -> pd.DataFrame:
    """Reconstruct the intersection population from its random partitions."""
    frames = []
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}_votes.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    votes = pd.concat(frames, ignore_index=True)
    required = {"username", "community", "item_id", "timestamp", "vote", "label"}
    missing = required - set(votes.columns)
    if missing:
        raise ValueError(f"Intersection split is missing columns: {sorted(missing)}")
    if votes.duplicated(["username", "item_id"]).any():
        count = int(votes.duplicated(["username", "item_id"]).sum())
        raise ValueError(f"Input contains {count:,} duplicate (username, item_id) votes")
    return votes


def _boundary_after_ties(item_times: pd.DataFrame, target: int, lower: int = 0) -> int:
    """Place a boundary after a timestamp tie, near the requested item count."""
    n_items = len(item_times)
    target = min(max(target, lower + 1), n_items - 1)
    timestamp = item_times.iloc[target - 1]["item_time"]
    boundary = int(item_times["item_time"].searchsorted(timestamp, side="right"))
    if boundary >= n_items:
        # The final tie is too large; place the boundary before it instead.
        boundary = int(item_times["item_time"].searchsorted(timestamp, side="left"))
    if boundary <= lower or boundary >= n_items:
        raise ValueError("Cannot create non-empty chronological partitions without splitting ties")
    return boundary


def build_chronological_splits(
    votes: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return item-level chronological partitions and their validation report."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/validation/test ratios must sum to 1")

    votes = votes.copy()
    votes["timestamp"] = pd.to_datetime(votes["timestamp"], utc=True, errors="coerce")
    if votes["timestamp"].isna().any():
        raise ValueError(f"Input contains {votes['timestamp'].isna().sum():,} invalid timestamps")

    item_times = (
        votes.groupby("item_id", as_index=False)
        .agg(item_first_time=("timestamp", "min"), item_time=("timestamp", "max"))
        .sort_values(["item_time", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    n_items = len(item_times)
    if n_items < 3:
        raise ValueError("At least three items are required")

    train_end = _boundary_after_ties(item_times, int(n_items * train_ratio))
    val_end = _boundary_after_ties(
        item_times, int(n_items * (train_ratio + val_ratio)), lower=train_end
    )
    assignments = {
        "train": set(item_times.iloc[:train_end]["item_id"]),
        "val": set(item_times.iloc[train_end:val_end]["item_id"]),
        "test": set(item_times.iloc[val_end:]["item_id"]),
    }
    splits = {
        name: votes[votes["item_id"].isin(item_ids)]
        .sort_values(["timestamp", "item_id", "username"], kind="stable")
        .reset_index(drop=True)
        for name, item_ids in assignments.items()
    }

    # Dataset integrity and temporal-order checks.
    assert not (assignments["train"] & assignments["val"])
    assert not (assignments["train"] & assignments["test"])
    assert not (assignments["val"] & assignments["test"])
    assert set().union(*assignments.values()) == set(votes["item_id"].unique())
    split_item_times = {
        name: item_times[item_times["item_id"].isin(item_ids)]["item_time"]
        for name, item_ids in assignments.items()
    }
    assert split_item_times["train"].max() < split_item_times["val"].min()
    assert split_item_times["val"].max() < split_item_times["test"].min()
    assert sum(len(frame) for frame in splits.values()) == len(votes)

    all_users = set(votes["username"])
    train_users = set(splits["train"]["username"])
    report: dict = {
        "source": str(DEFAULT_INPUT),
        "split_unit": "item_id",
        "item_time_definition": "maximum vote timestamp per item (case completion)",
        "tie_policy": "equal boundary timestamps stay in the earlier partition",
        "n_source_votes": len(votes),
        "n_source_items": n_items,
        "n_source_users": len(all_users),
        "partitions": {},
        "checks": {
            "item_overlap": 0,
            "all_source_items_preserved": True,
            "all_source_votes_preserved": True,
            "strict_temporal_order": True,
        },
    }
    for name, frame in splits.items():
        labels = frame.drop_duplicates("item_id")["label"].value_counts().to_dict()
        users = set(frame["username"])
        assigned_item_times = item_times[item_times["item_id"].isin(assignments[name])]
        report["partitions"][name] = {
            "n_votes": len(frame),
            "n_items": int(frame["item_id"].nunique()),
            "n_users": int(frame["username"].nunique()),
            "item_fraction": float(frame["item_id"].nunique() / n_items),
            "case_time_start": assigned_item_times["item_time"].min().isoformat(),
            "case_time_end": assigned_item_times["item_time"].max().isoformat(),
            "vote_time_start": frame["timestamp"].min().isoformat(),
            "vote_time_end": frame["timestamp"].max().isoformat(),
            "item_label_counts": {str(key): int(value) for key, value in labels.items()},
            "users_seen_in_train_fraction": (
                1.0 if name == "train" else float(len(users & train_users) / max(len(users), 1))
            ),
        }
    return splits, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading filtered intersection from {args.input_dir} ...")
    votes = load_intersection(args.input_dir)
    splits, report = build_chronological_splits(
        votes, args.train_ratio, args.val_ratio, args.test_ratio
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        path = args.output_dir / f"{name}_votes.parquet"
        frame.to_parquet(path, index=False)
        stats = report["partitions"][name]
        print(
            f"{name:5s}: {stats['n_items']:,} items | {stats['n_votes']:,} votes | "
            f"case completion {stats['case_time_start']} -> {stats['case_time_end']}"
        )

    report["source"] = str(args.input_dir)
    report_path = args.output_dir / "split_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Manifest: {report_path}")


if __name__ == "__main__":
    main()
