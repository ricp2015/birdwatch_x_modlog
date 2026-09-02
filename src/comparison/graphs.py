"""Generate thesis tables, figures, bootstrap results, and E1 summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.utils.splits import discover_splits, load_vote_partitions  # noqa: E402

log = logging.getLogger(__name__)

FIXED_SPLITS = (
    "full",
    "intersection",
    "intersection_chronological",
)
WINDOW_POPULATIONS = ("full", "intersection")
METRICS = ("macro_f1", "f1_pos", "f1_neg", "roc_auc")
METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "f1_pos": "F1 approve",
    "f1_neg": "F1 remove",
    "roc_auc": "ROC AUC",
}
METADATA_MODES = ("none", "global", "subreddit", "full")
E1_VARIANTS = {
    "cn_inductive": "CN inductive",
    "qsmf": "QSMF",
    "ma_mf": "MA-MF",
    "ma_qsmf": "MA-QSMF",
}


@dataclass(frozen=True)
class MethodSpec:
    """Canonical description of one benchmark method."""

    label: str
    relative_dir: str
    prediction_files: tuple[str, ...] = ("test_predictions.parquet", "item_scores.parquet")
    prediction_columns: tuple[str, ...] = ("y_hat", "prediction", "predicted")
    score_columns: tuple[str, ...] = ("score",)
    main_report: bool = True
    supported_splits: tuple[str, ...] | None = None


METHOD_SPECS: dict[str, MethodSpec] = {
    "Net": MethodSpec("Net", "baselines/BL1_net_vote", prediction_files=("item_scores.parquet",)),
    "WNet": MethodSpec(
        "WNet", "baselines/BL2_net_vote_alpha", prediction_files=("item_scores.parquet",)
    ),
    "SubAdj": MethodSpec(
        "SubAdj",
        "baselines/BL3_net_vote_alpha_subreddit",
        prediction_files=("item_scores.parquet",),
    ),
    "RedditScore": MethodSpec(
        "RedditScore", "baselines/BL4_reddit_score", prediction_files=("item_scores.parquet",)
    ),
    "RedditScore+Sub": MethodSpec(
        "RedditScore+Sub",
        "baselines/BL5_subreddit",
        prediction_files=("item_scores.parquet",),
    ),
    "CN": MethodSpec(
        "CN",
        "cn",
        prediction_files=("item_scores.parquet",),
        score_columns=("i_n", "score"),
    ),
    "CN inductive": MethodSpec("CN inductive", "ma_qsmf/cn_inductive"),
    "QSMF": MethodSpec("QSMF", "ma_qsmf/qsmf"),
    "MA-MF": MethodSpec("MA-MF", "ma_qsmf/ma_mf"),
    "MA-QSMF": MethodSpec("MA-QSMF", "ma_qsmf/ma_qsmf"),
    "NVSE": MethodSpec(
        "NVSE",
        "normvio-skill-extraction",
        prediction_files=("item_scores_test.parquet", "item_scores.parquet"),
        score_columns=("nvse_score", "score"),
    ),
    "VAR": MethodSpec("VAR", "var", prediction_files=("item_scores.parquet",)),
    "Bandit": MethodSpec(
        "Bandit", "bandit", supported_splits=("intersection_chronological",)
    ),
    "GraphPropagation": MethodSpec(
        "GraphPropagation",
        "graph_propagation",
        supported_splits=("intersection_chronological",),
    ),
    "RoleNSGA2": MethodSpec(
        "Role-NSGA2", "role_nsga2", supported_splits=("intersection_chronological",)
    ),
    "VirtualEnsembles": MethodSpec(
        "VirtualEnsembles",
        "virtual_ensembles",
        supported_splits=("intersection_chronological",),
    ),
    "BoC": MethodSpec(
        "BoC", "boc_stacking", supported_splits=("intersection_chronological",)
    ),
}

BOOTSTRAP_TOP_METHODS = ("SEF", "BoC", "VirtualEnsembles", "SubAdj")
BOOTSTRAP_METRICS = ("macro_f1", "f1_neg")

METHOD_COLORS = {
    "CN": "#a61e2a",
    "CN inductive": "#d1495b",
    "QSMF": "#edae49",
    "MA-MF": "#00798c",
    "MA-QSMF": "#30638e",
    "SEF": "#e63946",
    "BoC": "#264653",
    "VirtualEnsembles": "#457b9d",
    "SubAdj": "#2a9d8f",
    "NVSE": "#f4a261",
    "VAR": "#e76f51",
    "Net": "#8d99ae",
    "WNet": "#6a994e",
    "RedditScore": "#7b2cbf",
    "RedditScore+Sub": "#5a189a",
    "Bandit": "#00a6a6",
    "GraphPropagation": "#3a86ff",
    "RoleNSGA2": "#ff6b6b",
}

WINDOW_LABELS = {
    20: "20% training",
    40: "40% training",
    60: "60% training",
    80: "80% training",
    100: "100% training",
}


def parse_result_path(metrics_path: Path, root: Path) -> tuple[str, str] | None:
    """Return the split and raw method key encoded by a result path."""
    try:
        parts = metrics_path.parent.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    if "baselines" in parts:
        marker = parts.index("baselines")
        return "/".join(parts[:marker]), parts[-1]
    return "/".join(parts[:-1]), parts[-1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _test_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    test = metrics.get("test")
    return dict(test) if isinstance(test, Mapping) else dict(metrics)


def _validation_macro_f1(metrics: Mapping[str, Any]) -> float | None:
    validation = metrics.get("val")
    if isinstance(validation, Mapping):
        value = validation.get("macro_f1")
        return float(value) if isinstance(value, (int, float)) else None
    value = metrics.get("val_macro_f1")
    return float(value) if isinstance(value, (int, float)) else None


def _split_result_dir(results_root: Path, split_name: str) -> Path:
    return results_root.joinpath(*split_name.split("/"))


def _fixed_method_dir(results_root: Path, split_name: str, method: str) -> Path | None:
    spec = METHOD_SPECS.get(method)
    if spec is None:
        return None
    return _split_result_dir(results_root, split_name) / spec.relative_dir


def _select_mode(
    candidates: Iterable[tuple[str, Path]],
) -> tuple[str, Path, dict[str, Any]] | None:
    """Select one configuration strictly by validation Macro-F1."""
    eligible: list[tuple[float, int, str, Path, dict[str, Any]]] = []
    for mode, metrics_path in candidates:
        if not metrics_path.exists():
            continue
        metrics = _read_json(metrics_path)
        validation_score = _validation_macro_f1(metrics)
        if validation_score is None or not math.isfinite(validation_score):
            continue
        eligible.append(
            (validation_score, -METADATA_MODES.index(mode), mode, metrics_path.parent, metrics)
        )
    if not eligible:
        return None
    _, _, mode, directory, metrics = max(eligible)
    return mode, directory, metrics


def resolve_method(
    results_root: Path, split_name: str, method: str
) -> tuple[Path, dict[str, Any], str | None] | None:
    """Resolve one canonical method, including validation-only mode selection."""
    if not method_supports_split(method, split_name):
        return None
    if method == "SEF":
        if split_name == "intersection_chronological":
            return None
        selected = _select_mode(
            (
                mode,
                results_root.joinpath(mode, *split_name.split("/"), "expertise", "metrics.json"),
            )
            for mode in METADATA_MODES
        )
        if selected is None:
            return None
        mode, directory, metrics = selected
        return directory, metrics, mode

    directory = _fixed_method_dir(results_root, split_name, method)
    if directory is None:
        return None
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None
    return directory, _read_json(metrics_path), None


def canonical_methods(include_appendix: bool = True) -> list[str]:
    """Return benchmark methods in stable report order."""
    methods = list(METHOD_SPECS)
    if not include_appendix:
        methods = [method for method in methods if METHOD_SPECS[method].main_report]
    methods.append("SEF")
    return methods


def method_supports_split(method: str, split_name: str) -> bool:
    """Return whether a method is defined for the requested evaluation split."""
    if method == "SEF":
        return split_name != "intersection_chronological"
    spec = METHOD_SPECS.get(method)
    return spec is None or spec.supported_splits is None or split_name in spec.supported_splits


def method_label(method: str) -> str:
    """Return the report-facing label for a canonical method key."""
    return METHOD_SPECS.get(method, MethodSpec(method, "")).label


def methods_for_split(split_name: str) -> list[str]:
    """Return methods eligible for paired inference on a split."""
    methods = list(BOOTSTRAP_TOP_METHODS) + ["CN"]
    return [method for method in methods if method_supports_split(method, split_name)]


def _metric_row(
    split_name: str,
    method: str,
    metrics: Mapping[str, Any],
    selected_mode: str | None,
) -> dict[str, Any]:
    payload = _test_payload(metrics)
    n_items = payload.get("n_items", payload.get("n_test", payload.get("n_items_test")))
    n_test_items = payload.get("n_test_items", metrics.get("n_test_items", n_items))
    coverage = payload.get("score_coverage")
    if coverage is None:
        coverage = payload.get("coverage_test", payload.get("coverage"))
    root_coverage = metrics.get("coverage")
    if coverage is None and isinstance(root_coverage, Mapping):
        test_coverage = root_coverage.get("test")
        if isinstance(test_coverage, Mapping):
            eligible = test_coverage.get("eligible_items")
            total = test_coverage.get("total_items")
        else:
            eligible = root_coverage.get("test_eligible_items")
            total = root_coverage.get("test_total_items")
        if (
            isinstance(eligible, (int, float))
            and isinstance(total, (int, float))
            and total
        ):
            n_test_items = int(total)
            coverage = float(eligible) / float(total)
    if coverage is None and n_items is not None and n_test_items:
        coverage = float(n_items) / float(n_test_items)
    if coverage is None:
        coverage = 1.0
    fallback = payload.get("n_fallback_items", metrics.get("n_fallback_items", 0))
    auc_population = payload.get("auc_population")
    if auc_population is None:
        if payload.get("auc_observed_only"):
            auc_population = "observed_external_score_items"
        elif float(coverage) < 0.999999:
            auc_population = "prefix_eligible_test_items"
        else:
            auc_population = "all_scored_test_items"
    row: dict[str, Any] = {
        "split": split_name,
        "method": method,
        "selected_mode": selected_mode,
        "n_items": n_items,
        "n_test_items": n_test_items,
        "score_coverage": float(coverage),
        "n_fallback_items": int(fallback or 0),
        "auc_population": auc_population,
    }
    for metric in METRICS:
        value = payload.get(metric)
        row[metric] = float(value) if isinstance(value, (int, float)) else np.nan
    return row


def collect_benchmark(
    results_root: Path,
    split_names: Sequence[str],
    include_appendix: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect normalized benchmark rows and non-fatal audit findings."""
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for split_name in split_names:
        for method in canonical_methods(include_appendix=include_appendix):
            if not method_supports_split(method, split_name):
                if method == "SEF" and split_name == "intersection_chronological":
                    findings.append(
                        {
                            "severity": "INFO",
                            "code": "unsupported_method_split",
                            "split": split_name,
                            "method": method,
                            "message": (
                                "SEF is intentionally excluded from chronological "
                                "evaluation."
                            ),
                        }
                    )
                continue
            resolved = resolve_method(results_root, split_name, method)
            if resolved is None:
                findings.append(
                    {
                        "severity": "WARNING",
                        "code": "missing_metrics",
                        "split": split_name,
                        "method": method,
                        "message": "No eligible metrics.json was found.",
                    }
                )
                continue
            _, metrics, selected_mode = resolved
            row = _metric_row(split_name, method, metrics, selected_mode)
            rows.append(row)
            if row["score_coverage"] < 0.999999:
                findings.append(
                    {
                        "severity": "WARNING",
                        "code": "partial_score_coverage",
                        "split": split_name,
                        "method": method,
                        "message": f"Score coverage is {row['score_coverage']:.2%}.",
                    }
                )
            if row["auc_population"] != "all_scored_test_items":
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "restricted_auc_population",
                        "split": split_name,
                        "method": method,
                        "message": f"ROC-AUC population: {row['auc_population']}.",
                    }
                )
    return pd.DataFrame(rows), findings


def subset_stats(
    frame: pd.DataFrame, min_vote_thresholds: Sequence[int] = (5, 10, 20)
) -> dict[str, Any]:
    """Calculate coverage, density, and class-balance statistics."""
    if frame.empty:
        base: dict[str, Any] = {
            "n_votes": 0,
            "n_posts": 0,
            "n_users": 0,
            "n_communities": 0,
            "avg_votes_per_user": 0.0,
            "median_votes_per_user": 0.0,
            "avg_votes_per_post": 0.0,
        }
        for threshold in min_vote_thresholds:
            base[f"pct_users_ge_{threshold}_votes"] = 0.0
        return base

    votes_per_user = frame.groupby("username").size()
    votes_per_post = frame.groupby("item_id").size()
    stats: dict[str, Any] = {
        "n_votes": int(len(frame)),
        "n_posts": int(frame["item_id"].nunique()),
        "n_users": int(frame["username"].nunique()),
        "n_communities": int(frame["community"].nunique()) if "community" in frame else None,
        "avg_votes_per_user": float(votes_per_user.mean()),
        "median_votes_per_user": float(votes_per_user.median()),
        "avg_votes_per_post": float(votes_per_post.mean()),
    }
    for threshold in min_vote_thresholds:
        stats[f"pct_users_ge_{threshold}_votes"] = float((votes_per_user >= threshold).mean())
    if "community" in frame:
        user_community_counts = frame.groupby(["username", "community"]).size()
        for threshold in min_vote_thresholds:
            stats[f"pct_user_community_pairs_ge_{threshold}_votes"] = float(
                (user_community_counts >= threshold).mean()
            )
    if "label" in frame:
        labels = frame.drop_duplicates("item_id")["label"]
        stats["pct_label_approve"] = float((labels == 1).mean())
        stats["pct_label_remove"] = float((labels == -1).mean())
    return stats


def collect_split_statistics(splits_root: Path) -> pd.DataFrame:
    """Return one tidy table for every prepared split partition."""
    rows: list[dict[str, Any]] = []
    for split_name, split_dir in discover_splits(splits_root).items():
        train, validation, test = load_vote_partitions(split_dir)
        partitions = {
            "train": train,
            "val": validation,
            "test": test,
            "total": pd.concat([train, validation, test], ignore_index=True),
        }
        for partition, frame in partitions.items():
            rows.append({"split": split_name, "partition": partition, **subset_stats(frame)})
    return pd.DataFrame(rows)


def _prediction_spec(method: str) -> MethodSpec:
    if method == "SEF":
        return MethodSpec(
            "SEF",
            "expertise",
            prediction_files=("item_scores.parquet",),
            prediction_columns=("predicted",),
            score_columns=("meta_proba", "weighted_vote"),
        )
    return METHOD_SPECS[method]


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def load_predictions(results_root: Path, split_name: str, method: str) -> pd.DataFrame:
    """Load and normalize item-level labels, decisions, and optional scores."""
    resolved = resolve_method(results_root, split_name, method)
    if resolved is None:
        raise FileNotFoundError(f"No eligible {method} result for {split_name}")
    directory, metrics, _ = resolved
    spec = _prediction_spec(method)
    prediction_path = _first_existing(directory, spec.prediction_files)
    if prediction_path is None:
        raise FileNotFoundError(f"No prediction file for {method} in {directory}")
    frame = pd.read_parquet(prediction_path)
    required = {"item_id", "label"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"{prediction_path} is missing columns: {missing}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"Duplicate item_id values in {prediction_path}")

    prediction_column = next(
        (column for column in spec.prediction_columns if column in frame.columns), None
    )
    if prediction_column is None and method == "NVSE":
        score_column = next(
            (column for column in spec.score_columns if column in frame.columns), None
        )
        threshold = _test_payload(metrics).get("threshold", metrics.get("threshold"))
        if score_column is not None and isinstance(threshold, (int, float)):
            frame = frame.copy()
            frame["_prediction"] = np.where(frame[score_column] >= threshold, 1, -1)
            prediction_column = "_prediction"
    if prediction_column is None:
        raise ValueError(f"No prediction column recognized in {prediction_path}")

    score_column = next((column for column in spec.score_columns if column in frame.columns), None)
    columns = ["item_id", "label", prediction_column]
    if score_column is not None:
        columns.append(score_column)
    fallback_column = next(
        (column for column in ("is_fallback", "used_fallback") if column in frame.columns),
        None,
    )
    if fallback_column is not None:
        columns.append(fallback_column)
    source_column = next(
        (column for column in ("decision_source", "score_source") if column in frame.columns),
        None,
    )
    if source_column is not None:
        columns.append(source_column)
    normalized = frame[columns].rename(columns={prediction_column: "prediction"}).copy()
    if score_column is not None:
        normalized = normalized.rename(columns={score_column: "score"})
    normalized["item_id"] = normalized["item_id"].astype(str)
    normalized["label"] = normalized["label"].astype(int)
    normalized["prediction"] = normalized["prediction"].astype(int)
    if fallback_column is not None:
        normalized["is_fallback"] = normalized[fallback_column].fillna(False).astype(bool)
        if fallback_column != "is_fallback":
            normalized = normalized.drop(columns=fallback_column)
    elif source_column is not None:
        normalized["is_fallback"] = (
            normalized[source_column]
            .astype("string")
            .str.contains("fallback", case=False, na=False)
        )
    else:
        normalized["is_fallback"] = False
    if source_column is not None:
        normalized = normalized.drop(columns=source_column)
    invalid_labels = set(normalized["label"].unique()) - {-1, 1}
    invalid_predictions = set(normalized["prediction"].unique()) - {-1, 1}
    if invalid_labels or invalid_predictions:
        raise ValueError(
            f"Invalid labels/predictions in {prediction_path}: "
            f"labels={invalid_labels}, predictions={invalid_predictions}"
        )
    return normalized.sort_values("item_id").reset_index(drop=True)


def collect_native_fallback_performance(
    results_root: Path,
    split_names: Sequence[str] = FIXED_SPLITS,
) -> pd.DataFrame:
    """Summarize native, fallback-only, and final hybrid hard-decision performance."""
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        for method in canonical_methods(include_appendix=True):
            if not method_supports_split(method, split_name):
                continue
            try:
                predictions = load_predictions(results_root, split_name, method)
            except FileNotFoundError:
                continue
            fallback = predictions["is_fallback"].astype(bool)
            if not fallback.any():
                continue
            native_frame = predictions.loc[~fallback]
            fallback_frame = predictions.loc[fallback]
            hybrid_metrics = _point_metrics(
                predictions["label"].to_numpy(dtype=int),
                predictions["prediction"].to_numpy(dtype=int),
            )

            def subset_metrics(frame: pd.DataFrame) -> dict[str, float]:
                if frame.empty:
                    return {"macro_f1": np.nan, "f1_neg": np.nan}
                return _point_metrics(
                    frame["label"].to_numpy(dtype=int),
                    frame["prediction"].to_numpy(dtype=int),
                )

            native_metrics = subset_metrics(native_frame)
            fallback_metrics = subset_metrics(fallback_frame)
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "n_total": len(predictions),
                    "n_native": len(native_frame),
                    "native_share": len(native_frame) / len(predictions),
                    "native_macro_f1": native_metrics["macro_f1"],
                    "native_f1_remove": native_metrics["f1_neg"],
                    "n_fallback": len(fallback_frame),
                    "fallback_macro_f1": fallback_metrics["macro_f1"],
                    "fallback_f1_remove": fallback_metrics["f1_neg"],
                    "hybrid_macro_f1": hybrid_metrics["macro_f1"],
                    "hybrid_f1_remove": hybrid_metrics["f1_neg"],
                }
            )
    return pd.DataFrame(rows)


def audit_predictions(
    results_root: Path,
    splits_root: Path,
    split_names: Sequence[str],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    """Audit predictions against TEST membership, labels, and reported metrics."""
    findings: list[dict[str, Any]] = []
    split_directories = discover_splits(splits_root)
    for split_name in split_names:
        split_dir = split_directories.get(split_name)
        if split_dir is None:
            findings.append(
                {
                    "severity": "ERROR",
                    "code": "missing_test_split",
                    "split": split_name,
                    "method": None,
                    "message": "The prepared TEST partition is missing.",
                }
            )
            continue
        test_votes = pd.read_parquet(split_dir / "test_votes.parquet")
        expected = (
            test_votes[["item_id", "label"]]
            .assign(item_id=lambda frame: frame["item_id"].astype(str))
            .drop_duplicates("item_id")
            .set_index("item_id")["label"]
            .astype(int)
        )
        for method in methods:
            if not method_supports_split(method, split_name):
                continue
            try:
                predictions = load_predictions(results_root, split_name, method)
            except FileNotFoundError as exc:
                findings.append(
                    {
                        "severity": "WARNING",
                        "code": "missing_predictions",
                        "split": split_name,
                        "method": method,
                        "message": str(exc),
                    }
                )
                continue
            except ValueError as exc:
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "invalid_predictions",
                        "split": split_name,
                        "method": method,
                        "message": str(exc),
                    }
                )
                continue
            prediction_ids = set(predictions["item_id"])
            expected_ids = set(expected.index)
            unknown_ids = prediction_ids - expected_ids
            missing_ids = expected_ids - prediction_ids
            if unknown_ids:
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "prediction_items_outside_test",
                        "split": split_name,
                        "method": method,
                        "message": f"Predictions contain {len(unknown_ids)} item(s) outside TEST.",
                    }
                )
            if missing_ids:
                findings.append(
                    {
                        "severity": "WARNING",
                        "code": "incomplete_prediction_population",
                        "split": split_name,
                        "method": method,
                        "message": f"Predictions omit {len(missing_ids)} of {len(expected_ids)} TEST items.",
                    }
                )
            shared = predictions[predictions["item_id"].isin(expected_ids)].set_index("item_id")
            expected_labels = expected.loc[shared.index].to_numpy(dtype=int)
            if not np.array_equal(expected_labels, shared["label"].to_numpy(dtype=int)):
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "test_label_mismatch",
                        "split": split_name,
                        "method": method,
                        "message": "Prediction labels disagree with the prepared TEST partition.",
                    }
                )

            resolved = resolve_method(results_root, split_name, method)
            if resolved is not None and len(shared):
                reported = _test_payload(resolved[1])
                recomputed = _point_metrics(
                    shared["label"].to_numpy(dtype=int),
                    shared["prediction"].to_numpy(dtype=int),
                )
                for metric in BOOTSTRAP_METRICS:
                    reported_value = reported.get(metric)
                    if isinstance(reported_value, (int, float)) and not np.isclose(
                        recomputed[metric], float(reported_value), atol=1e-9, rtol=1e-7
                    ):
                        findings.append(
                            {
                                "severity": "ERROR",
                                "code": "metric_prediction_mismatch",
                                "split": split_name,
                                "method": method,
                                "message": (
                                    f"Reported {metric}={float(reported_value):.6f}, but "
                                    f"item predictions yield {recomputed[metric]:.6f}."
                                ),
                            }
                        )
            findings.append(
                {
                    "severity": "INFO",
                    "code": "prediction_population",
                    "split": split_name,
                    "method": method,
                    "message": f"Loaded {len(predictions):,} unique item predictions.",
                }
            )
    return findings


def _safe_f1(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    tn = float(np.sum((labels == -1) & (predictions == -1)))
    fp = float(np.sum((labels == -1) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == -1)))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"macro_f1": (f1_pos + f1_neg) / 2, "f1_neg": f1_neg}


def _bootstrap_metric_samples(
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 250,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute stratified item-bootstrap metric samples using shared draws."""
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == -1)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("Paired bootstrap requires both outcome classes.")
    samples = {
        method: {metric: np.empty(iterations, dtype=float) for metric in BOOTSTRAP_METRICS}
        for method in predictions
    }
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        size = stop - start
        positive_draws = rng.choice(
            positive_indices, size=(size, len(positive_indices)), replace=True
        )
        negative_draws = rng.choice(
            negative_indices, size=(size, len(negative_indices)), replace=True
        )
        for method, method_predictions in predictions.items():
            tp = np.sum(method_predictions[positive_draws] == 1, axis=1).astype(float)
            fn = len(positive_indices) - tp
            tn = np.sum(method_predictions[negative_draws] == -1, axis=1).astype(float)
            fp = len(negative_indices) - tn
            f1_pos = _safe_f1(2 * tp, 2 * tp + fp + fn)
            f1_neg = _safe_f1(2 * tn, 2 * tn + fp + fn)
            samples[method]["macro_f1"][start:stop] = (f1_pos + f1_neg) / 2
            samples[method]["f1_neg"][start:stop] = f1_neg
    return samples


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_bootstrap(
    results_root: Path,
    split_names: Sequence[str] = FIXED_SPLITS,
    iterations: int = 10_000,
    seed: int = 10,
) -> pd.DataFrame:
    """Run the predeclared paired item-level inference families."""
    rows: list[dict[str, Any]] = []
    seed_sequence = np.random.SeedSequence(seed)
    split_seeds = seed_sequence.spawn(len(split_names))
    for split_name, split_seed in zip(split_names, split_seeds):
        eligible = methods_for_split(split_name)
        loaded: dict[str, pd.DataFrame] = {}
        for method in eligible:
            try:
                loaded[method] = load_predictions(results_root, split_name, method)
            except FileNotFoundError:
                log.warning("Skipping missing bootstrap method %s on %s", method, split_name)
        if "CN" not in loaded:
            log.warning("Skipping paired bootstrap for %s because CN is missing", split_name)
            continue

        reference_method = next(iter(loaded))
        reference = loaded[reference_method]
        reference_ids = reference["item_id"].tolist()
        reference_id_set = set(reference_ids)
        labels = reference["label"].to_numpy(dtype=int)
        aligned_predictions: dict[str, np.ndarray] = {}
        for method, frame in loaded.items():
            if set(frame["item_id"]) != reference_id_set:
                raise ValueError(
                    f"Paired bootstrap population mismatch on {split_name}: "
                    f"{reference_method} has {len(reference_ids)} items and {method} has "
                    f"{len(frame)}. No intersection fallback is allowed."
                )
            aligned = frame.set_index("item_id").loc[reference_ids]
            if not np.array_equal(labels, aligned["label"].to_numpy(dtype=int)):
                raise ValueError(
                    f"Ground-truth labels disagree between {reference_method} and {method} "
                    f"on {split_name}."
                )
            aligned_predictions[method] = aligned["prediction"].to_numpy(dtype=int)

        samples = _bootstrap_metric_samples(
            labels,
            aligned_predictions,
            iterations=iterations,
            rng=np.random.default_rng(split_seed),
        )
        point = {
            method: _point_metrics(labels, predictions)
            for method, predictions in aligned_predictions.items()
        }
        top_available = [method for method in BOOTSTRAP_TOP_METHODS if method in loaded]
        families = {
            "top_methods": list(itertools.combinations(top_available, 2)),
            "against_cn": [(method, "CN") for method in top_available],
        }
        for family, pairs in families.items():
            for method_a, method_b in pairs:
                for metric in BOOTSTRAP_METRICS:
                    differences = samples[method_a][metric] - samples[method_b][metric]
                    lower, upper = np.quantile(differences, [0.025, 0.975])
                    left = (np.count_nonzero(differences <= 0) + 1) / (iterations + 1)
                    right = (np.count_nonzero(differences >= 0) + 1) / (iterations + 1)
                    rows.append(
                        {
                            "family": family,
                            "split": split_name,
                            "metric": metric,
                            "method_a": method_a,
                            "method_b": method_b,
                            "estimate_a": point[method_a][metric],
                            "estimate_b": point[method_b][metric],
                            "delta": point[method_a][metric] - point[method_b][metric],
                            "ci_lower": float(lower),
                            "ci_upper": float(upper),
                            "p_value": min(1.0, 2 * min(left, right)),
                            "n_items": len(labels),
                            "iterations": iterations,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm"] = np.nan
    for indices in result.groupby(["family", "metric"]).groups.values():
        result.loc[indices, "p_holm"] = _holm_adjust(result.loc[indices, "p_value"])
    result["significant_holm_0_05"] = result["p_holm"] < 0.05
    return result


def collect_e1(results_root: Path, split_names: Sequence[str]) -> pd.DataFrame:
    """Collect the E1 CN-extension ablation separately from T1."""
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        cn = resolve_method(results_root, split_name, "CN")
        if cn is not None:
            rows.append(_metric_row(split_name, "CN", cn[1], None))
        split_dir = _split_result_dir(results_root, split_name)
        for variant, label in E1_VARIANTS.items():
            metrics_path = split_dir / "ma_qsmf" / variant / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            row = _metric_row(split_name, label, metrics, None)
            row["selected_lambda_y"] = metrics.get("selected_lambda_y")
            row["moderator_aligned"] = metrics.get("moderator_aligned")
            row["learn_rho"] = metrics.get("learn_rho")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    cn_scores = frame.loc[frame["method"] == "CN", ["split", "macro_f1", "f1_neg"]].rename(
        columns={"macro_f1": "cn_macro_f1", "f1_neg": "cn_f1_neg"}
    )
    frame = frame.merge(cn_scores, on="split", how="left")
    frame["delta_macro_f1_vs_cn"] = frame["macro_f1"] - frame["cn_macro_f1"]
    frame["delta_f1_neg_vs_cn"] = frame["f1_neg"] - frame["cn_f1_neg"]
    return frame


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        markdown = frame.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        markdown = "```text\n" + frame.to_string(index=False) + "\n```"
    stem.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Save the report-facing raster and vector versions of one figure."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _method_bar_plot(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    output_stem: Path,
) -> None:
    """Draw the classic coloured method bar chart used by the earlier report."""
    subset = frame.dropna(subset=[metric]).sort_values(metric, ascending=False).copy()
    if subset.empty:
        return
    labels = [method_label(method) for method in subset["method"]]
    values = subset[metric].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(method, "#8c8c8c") for method in subset["method"]]
    width = max(11.0, 0.82 * len(subset))
    fig, axis = plt.subplots(figsize=(width, 6.2))
    bars = axis.bar(labels, values, color=colors, alpha=0.92, width=0.8)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    axis.set_title(title, fontsize=15, pad=14)
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=28, labelsize=9)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _plot_benchmark(frame: pd.DataFrame, output_dir: Path) -> None:
    """Create one coloured bar chart per fixed split and metric."""
    fixed = frame[frame["split"].isin(FIXED_SPLITS)].copy()
    if fixed.empty:
        return
    split_labels = {
        "full": "Full",
        "intersection": "Intersection",
        "intersection_chronological": "Intersection chronological",
    }
    for split_name in FIXED_SPLITS:
        subset = fixed[fixed["split"] == split_name]
        for metric in METRICS:
            _method_bar_plot(
                subset,
                metric,
                f"{METRIC_LABELS[metric]} - {split_labels[split_name]}",
                output_dir / f"{metric}__split_{split_name}",
            )


def _plot_scaling(frame: pd.DataFrame, population: str, output_stem: Path) -> None:
    """Aggregate all training windows of one population into bar-chart figures."""
    prefix = f"windows/{population}/"
    subset = frame[frame["split"].str.startswith(prefix, na=False)].copy()
    if subset.empty:
        return
    subset["window"] = subset["split"].str.rsplit("/", n=1).str[-1].str[1:].astype(int)
    windows = sorted(subset["window"].unique())
    for metric in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharey=True)
        for axis, window in zip(axes.flat, windows):
            window_rows = (
                subset[subset["window"] == window]
                .dropna(subset=[metric])
                .sort_values(metric, ascending=False)
            )
            labels = [method_label(method) for method in window_rows["method"]]
            values = window_rows[metric].to_numpy(dtype=float)
            colors = [
                METHOD_COLORS.get(method, "#8c8c8c") for method in window_rows["method"]
            ]
            bars = axis.bar(labels, values, color=colors, alpha=0.92)
            axis.bar_label(
                bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7
            )
            axis.set_title(WINDOW_LABELS.get(int(window), f"{window}% training"))
            axis.set_ylim(0, 1)
            axis.grid(axis="y", linestyle="--", alpha=0.25)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", rotation=38, labelsize=7)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        for axis in axes.flat[len(windows) :]:
            axis.remove()
        axes[0, 0].set_ylabel(METRIC_LABELS[metric])
        axes[1, 0].set_ylabel(METRIC_LABELS[metric])
        fig.suptitle(
            f"{METRIC_LABELS[metric]} - windowed {population}", fontsize=16, y=1.01
        )
        fig.tight_layout()
        _save_figure(fig, output_stem.parent / f"{metric}__windows_{population}")


def _plot_e1(frame: pd.DataFrame, output_stem: Path) -> None:
    if frame.empty:
        return
    methods = ["CN", *E1_VARIANTS.values()]
    available = [method for method in methods if method in set(frame["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(FIXED_SPLITS))
    for axis, metric in zip(axes, ("macro_f1", "f1_neg")):
        for method in available:
            indexed = frame[frame["method"] == method].set_index("split")
            values = [
                indexed.at[split, metric] if split in indexed.index else np.nan
                for split in FIXED_SPLITS
            ]
            axis.plot(x, values, marker="o", label=method)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(("Full", "Intersection", "Chronological"), rotation=20)
        axis.set_ylim(0, 1)
        axis.grid(linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Test performance")
    axes[0].legend(fontsize=8)
    fig.suptitle("E1: Community Notes extensions", fontsize=14)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_report(
    benchmark: pd.DataFrame,
    split_stats: pd.DataFrame,
    bootstrap: pd.DataFrame,
    e1: pd.DataFrame,
    findings: Sequence[Mapping[str, Any]],
) -> str:
    """Render a compact index of generated, narrative-ready evidence."""
    counts = pd.Series([finding["severity"] for finding in findings]).value_counts()
    lines = [
        "# Benchmark report bundle",
        "",
        "All method variants shown below are selected without using TEST outcomes.",
        "SEF is intentionally excluded from chronological evaluation because its user profiles are static/transductive.",
        "Windowed splits are scaling trajectories sharing validation and test populations; they are not independent replications.",
        "Team Formation v2 denotes the active counterfactual methods evaluated over prefix-eligible community pools; pre-clarification implementations are archived and excluded.",
        "",
        "## Audit summary",
        "",
        f"- Errors: {int(counts.get('ERROR', 0))}",
        f"- Warnings: {int(counts.get('WARNING', 0))}",
        f"- Informational findings: {int(counts.get('INFO', 0))}",
        "",
        "## Generated evidence",
        "",
        f"- Benchmark rows: {len(benchmark)}",
        f"- Split-statistic rows: {len(split_stats)}",
        f"- Paired-bootstrap contrasts: {len(bootstrap)}",
        f"- E1 rows: {len(e1)}",
        "",
        "See `tables/` for machine-readable and Markdown tables and `figures/` for PNG/PDF figures.",
    ]
    return "\n".join(lines) + "\n"


def run_report(
    results_root: Path,
    splits_root: Path,
    output_dir: Path,
    sections: set[str],
    bootstrap_iterations: int,
    seed: int,
    strict_audit: bool = False,
) -> None:
    """Run selected report sections and write a coherent artifact bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    discovered = discover_splits(splits_root)
    split_names = list(discovered)
    if not split_names:
        raise FileNotFoundError(f"No prepared splits found under {splits_root}")

    benchmark = pd.DataFrame()
    split_statistics = pd.DataFrame()
    bootstrap_results = pd.DataFrame()
    e1 = pd.DataFrame()
    findings: list[dict[str, Any]] = []

    needs_benchmark = bool(sections & {"audit", "benchmark", "scaling", "all"})
    if needs_benchmark:
        benchmark, benchmark_findings = collect_benchmark(results_root, split_names)
        findings.extend(benchmark_findings)
        _write_table(benchmark, tables_dir / "benchmark_all_methods")
        main_methods = set(canonical_methods(False))
        main = benchmark[benchmark["method"].isin(main_methods)].copy()
        _write_table(main, tables_dir / "benchmark_main")
        coverage_columns = [
            "split",
            "method",
            "n_items",
            "n_test_items",
            "score_coverage",
            "n_fallback_items",
            "auc_population",
        ]
        _write_table(benchmark[coverage_columns], tables_dir / "coverage_and_fallback")
        native_fallback = collect_native_fallback_performance(results_root)
        _write_table(native_fallback, tables_dir / "native_fallback_performance")
        if sections & {"benchmark", "all"}:
            _plot_benchmark(main, figures_dir)
        if sections & {"scaling", "all"}:
            for population in WINDOW_POPULATIONS:
                _plot_scaling(main, population, figures_dir / f"scaling_{population}")

    if sections & {"splits", "all"}:
        split_statistics = collect_split_statistics(splits_root)
        _write_table(split_statistics, tables_dir / "split_statistics")

    if sections & {"audit", "all"}:
        findings.extend(
            audit_predictions(
                results_root,
                splits_root,
                FIXED_SPLITS,
                canonical_methods(include_appendix=True),
            )
        )

    if sections & {"bootstrap", "all"}:
        bootstrap_results = paired_bootstrap(
            results_root,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        _write_table(bootstrap_results, tables_dir / "paired_bootstrap")

    if sections & {"e1", "all"}:
        e1 = collect_e1(results_root, FIXED_SPLITS)
        _write_table(e1, tables_dir / "e1_ablation")
        _plot_e1(e1, figures_dir / "e1_ablation")

    (output_dir / "audit.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = _render_report(
        benchmark,
        split_statistics,
        bootstrap_results,
        e1,
        findings,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    errors = sum(finding["severity"] == "ERROR" for finding in findings)
    warnings = sum(finding["severity"] == "WARNING" for finding in findings)
    log.info("Report bundle saved to %s", output_dir)
    log.info("Audit completed with %d error(s) and %d warning(s)", errors, warnings)
    if errors and strict_audit:
        raise RuntimeError(f"Report audit found {errors} invalid prediction artifact(s).")
    if errors:
        log.warning(
            "Report generation completed despite audit errors. "
            "See %s; pass --strict-audit to fail on them.",
            output_dir / "audit.json",
        )


def _parse_sections(values: Sequence[str]) -> set[str]:
    valid = {"all", "audit", "splits", "benchmark", "scaling", "bootstrap", "e1"}
    sections = set(values)
    unknown = sections - valid
    if unknown:
        raise ValueError(f"Unknown report section(s): {sorted(unknown)}")
    return {"all"} if "all" in sections else sections


def main() -> None:
    """Run the unified comparison workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        action="append",
        default=None,
        help="Section to run: all, audit, splits, benchmark, scaling, bootstrap, or e1.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/reddit"))
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits/reddit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="Exit with an error when prediction artifacts fail the audit.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(message)s")
    sections = _parse_sections(args.section or ["all"])
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    run_report(
        results_root=args.results_root,
        splits_root=args.splits_root,
        output_dir=args.output_dir,
        sections=sections,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        strict_audit=args.strict_audit,
    )


if __name__ == "__main__":
    main()
