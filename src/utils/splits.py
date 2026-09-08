"""Discovery and loading helpers for prepared vote splits."""

import json
from pathlib import Path

import pandas as pd


def _is_complete_split(path: Path) -> bool:
    return all((path / f"{name}_votes.parquet").exists() for name in ("train", "val", "test"))


def load_split_manifest(split_dir: Path) -> dict:
    """Load the manifest stored beside one concrete TRAIN/VAL/TEST split."""
    path = split_dir / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No split manifest found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def split_dataset(split_dir: Path, split_name: str | None = None) -> str | None:
    """Return the population-facing dataset name without relying on path substrings."""
    try:
        manifest = load_split_manifest(split_dir)
    except FileNotFoundError:
        manifest = {}
    dataset = manifest.get("dataset") or manifest.get("source_split")
    if isinstance(dataset, str):
        return dataset
    population = manifest.get("population")
    if population == "cleaned_pre_density":
        return "full"
    if population == "all_method_intersection":
        return (
            "intersection_chronological"
            if "chronological" in (split_name or "")
            else "intersection"
        )
    if split_name:
        return split_name.split("/", 1)[0]
    return None


def _discover_fold_collection(root: Path, prefix: str = "") -> dict[str, Path]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_type") != "kfold_collection":
        return {}
    found: dict[str, Path] = {}
    for entry in manifest.get("folds", []):
        folder = str(entry.get("folder", ""))
        split_dir = root / folder
        if folder and _is_complete_split(split_dir):
            found[f"{prefix}/{folder}" if prefix else folder] = split_dir
    return found


def discover_splits(votes_dir: Path) -> dict[str, Path]:
    """Find fixed and windowed splits under a dataset directory."""
    # Allow every method to receive one exact common benchmark directory.
    if _is_complete_split(votes_dir):
        return {votes_dir.name: votes_dir}

    root_manifest_path = votes_dir / "manifest.json"
    if root_manifest_path.exists():
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        if root_manifest.get("split_type") == "kfold_collection":
            return _discover_fold_collection(votes_dir)
        if root_manifest.get("split_type") == "kfold_root":
            found: dict[str, Path] = {}
            for entry in root_manifest.get("datasets", []):
                dataset = str(entry.get("dataset", ""))
                folder = str(entry.get("folder", ""))
                if dataset and folder:
                    found.update(_discover_fold_collection(votes_dir / folder, dataset))
            return found

    found: dict[str, Path] = {}
    for name in ("full", "intersection", "intersection_chronological"):
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
