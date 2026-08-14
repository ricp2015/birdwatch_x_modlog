"""Discovery and loading helpers for prepared vote splits."""

from pathlib import Path

import pandas as pd


def discover_splits(votes_dir: Path) -> dict[str, Path]:
    """Find fixed and windowed splits under a dataset directory."""
    # Allow every method to receive one exact common benchmark directory.
    if all((votes_dir / f"{name}_votes.parquet").exists() for name in ("train", "val", "test")):
        return {votes_dir.name: votes_dir}

    found: dict[str, Path] = {}
    for name in ("random", "full", "intersection", "intersection_chronological"):
        split_dir = votes_dir / name
        if (split_dir / "train_votes.parquet").exists():
            found[name] = split_dir

    for population in ("full", "intersection"):
        window_root = votes_dir / "windows" / population
        if not window_root.exists():
            continue
        for window_dir in sorted(window_root.glob("w*")):
            if (window_dir / "train_votes.parquet").exists():
                found[f"windows/{population}/{window_dir.name}"] = window_dir
    return found


def load_split_data(
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test votes plus their concatenation."""
    train = pd.read_parquet(split_dir / "train_votes.parquet")
    validation = pd.read_parquet(split_dir / "val_votes.parquet")
    test = pd.read_parquet(split_dir / "test_votes.parquet")
    all_votes = pd.concat([train, validation, test], ignore_index=True)
    return all_votes, train, validation, test


def load_vote_partitions(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, validation, and test vote partitions."""
    _, train, validation, test = load_split_data(split_dir)
    return train, validation, test
