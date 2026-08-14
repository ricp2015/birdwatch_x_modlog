"""Dynamic top-K VAR aggregation with leakage-safe causal user metadata.

This is a separate method by design: ``k_experts_method.py`` remains the
unchanged VAR-only reference implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.tfr_new.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    attach_causal_features,
    load_causal_features,
)
from src.methods.var_core import calculate_var_scores, evaluate_fold  # noqa: E402
from src.utils.splits import discover_splits, load_vote_partitions  # noqa: E402

DEFAULT_VOTES_DIR = Path("data/splits/reddit")
DEFAULT_OUTPUT_ROOT = Path("results/reddit")
K_VALUES = (10,)
MIX_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)

FEATURE_GROUPS = {
    "none": [],
    "global": [
        "prior_n_posts",
        "prior_n_comments",
        "prior_n_distinct_subreddits",
        "prior_n_replies_made",
        "tenure_days_at_vote",
    ],
    "subreddit": [
        "prior_n_posts",
        "prior_n_comments",
        "prior_n_distinct_subreddits",
        "prior_n_replies_made",
        "tenure_days_at_vote",
        "prior_n_posts_in_sub",
        "prior_n_comments_in_sub",
        "prior_n_months_active_in_sub",
    ],
    "full": [
        "prior_n_posts",
        "prior_n_comments",
        "prior_n_distinct_subreddits",
        "prior_n_replies_made",
        "tenure_days_at_vote",
        "prior_n_posts_in_sub",
        "prior_n_comments_in_sub",
        "prior_n_months_active_in_sub",
        "prior_n_interaction_partners",
        "prior_total_interactions",
        "prior_n_interaction_partners_in_sub",
        "prior_total_interactions_in_sub",
    ],
}


def _fit_reference(train: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    """Fit empirical feature distributions using training votes only."""
    reference = {}
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce").dropna().clip(lower=0)
        reference[column] = np.sort(np.log1p(values.to_numpy(dtype=float)))
    return reference


def _percentile(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    """Map values to training empirical percentiles; unknown values are neutral."""
    numeric = pd.to_numeric(values, errors="coerce").clip(lower=0)
    result = np.full(len(numeric), 0.5, dtype=float)
    known = numeric.notna().to_numpy()
    if len(reference) and known.any():
        transformed = np.log1p(numeric[known].to_numpy(dtype=float))
        result[known] = np.searchsorted(reference, transformed, side="right") / len(reference)
    return result


def add_dynamic_scores(
    votes: pd.DataFrame,
    var_scores: pd.DataFrame,
    mode: str,
    mix: float,
    reference: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Combine community VAR rank with causal per-vote metadata rank."""
    out = votes.merge(
        var_scores[["community", "username", "VAR_score"]],
        on=["community", "username"],
        how="left",
    )
    out["VAR_score"] = out["VAR_score"].fillna(0.0)
    out["var_rank"] = out.groupby("community")["VAR_score"].rank(pct=True).fillna(0.0)

    columns = FEATURE_GROUPS[mode]
    if columns:
        components = np.column_stack(
            [_percentile(out[column], reference[column]) for column in columns]
        )
        out["metadata_score"] = components.mean(axis=1)
    else:
        out["metadata_score"] = 0.5
    out["expert_score"] = (
        out["VAR_score"]
        if mix == 0.0
        else (1.0 - mix) * out["var_rank"] + mix * out["metadata_score"]
    )
    return out


def fit_community_scores(
    train: pd.DataFrame,
    var_scores: pd.DataFrame,
    mode: str,
    mix: float,
    reference: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Freeze user-community rankings using training snapshots only."""
    scored_train = add_dynamic_scores(train, var_scores, mode, mix, reference)
    return (
        scored_train.groupby(["community", "username"], as_index=False)["expert_score"]
        .mean()
        .rename(columns={"expert_score": "community_expert_score"})
    )


def select_votes(
    scored: pd.DataFrame,
    k: int,
    selection_mode: str,
    community_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the original per-community or per-item top-K selection."""
    if selection_mode == "per_item":
        return (
            scored.sort_values(["item_id", "expert_score"], ascending=[True, False])
            .groupby(["community", "item_id"], sort=False)
            .head(k)
            .copy()
        )
    user_rank = (
        community_scores.sort_values("community_expert_score", ascending=False)
        .groupby("community", sort=False)
        .head(k)
    )
    return scored.merge(
        user_rank[["community", "username"]],
        on=["community", "username"],
        how="inner",
    )


def aggregate_votes(selected: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Apply the original unweighted or reliability-weighted majority."""
    keys = ["community", "item_id"]
    if strategy == "weighted":
        totals = selected.groupby([*keys, "vote"])["expert_score"].sum()
    else:
        totals = selected.groupby([*keys, "vote"]).size()
    wide = totals.astype(float).unstack("vote", fill_value=0.0)
    negative = wide[-1] if -1 in wide else pd.Series(0.0, index=wide.index)
    positive = wide[1] if 1 in wide else pd.Series(0.0, index=wide.index)
    decisions = wide.reset_index()[keys]
    # Match pandas idxmax on sorted vote labels: ties resolve to -1.
    decisions["predicted"] = np.where(positive.to_numpy() > negative.to_numpy(), 1, -1)
    labels = selected.groupby(keys, as_index=False)["label"].first()
    return decisions.merge(labels, on=keys, how="left", validate="one_to_one")


def run_split(
    split_name: str,
    split_path: Path,
    causal: pd.DataFrame,
    output_root: Path,
) -> dict:
    """Select the representation on validation and evaluate one common test split."""
    train, val, test = load_vote_partitions(split_path)
    train = attach_causal_features(train, causal)
    val = attach_causal_features(val, causal)
    test = attach_causal_features(test, causal)
    var_scores = calculate_var_scores(train)
    all_columns = FEATURE_GROUPS["full"]
    reference = _fit_reference(train, all_columns)

    best = None
    validation_candidates = []
    for mode in FEATURE_GROUPS:
        mixes = (0.0,) if mode == "none" else MIX_VALUES
        for mix in mixes:
            community_scores = fit_community_scores(train, var_scores, mode, mix, reference)
            scored_val = add_dynamic_scores(val, var_scores, mode, mix, reference)
            for k in K_VALUES:
                for selection_mode in ("per_community", "per_item"):
                    selected_val = select_votes(scored_val, k, selection_mode, community_scores)
                    for strategy in ("majority", "weighted"):
                        decisions = aggregate_votes(selected_val, strategy)
                        if decisions.empty or decisions["label"].nunique() < 2:
                            continue
                        macro_f1 = evaluate_fold(decisions)["macro_f1"]
                        candidate = {
                            "mode": mode,
                            "metadata_mix": mix,
                            "k": k,
                            "selection_mode": selection_mode,
                            "strategy": strategy,
                            "validation_macro_f1": macro_f1,
                        }
                        validation_candidates.append(candidate)
                        if best is None or macro_f1 > best["validation_macro_f1"]:
                            best = candidate

    scored_test = add_dynamic_scores(
        test, var_scores, best["mode"], best["metadata_mix"], reference
    )
    community_scores = fit_community_scores(
        train, var_scores, best["mode"], best["metadata_mix"], reference
    )
    selected_voters = select_votes(
        scored_test, best["k"], best["selection_mode"], community_scores
    )
    test_items = aggregate_votes(selected_voters, best["strategy"])
    metrics = evaluate_fold(test_items)
    result = {
        "split": split_name,
        "selected": best,
        "test": metrics,
        "validation_candidates": validation_candidates,
    }

    out_dir = output_root / split_name / "k_experts_causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_items.to_parquet(out_dir / "test_predictions.parquet", index=False)
    selected_voters[
        [
            "item_id",
            "username",
            "community",
            "vote",
            "VAR_score",
            "var_rank",
            "metadata_score",
            "expert_score",
        ]
    ].to_parquet(out_dir / "test_selected_voters.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{split_name}] {best['mode']} mix={best['metadata_mix']:.2f} "
        f"{best['strategy']}/{best['selection_mode']} k={best['k']} "
        f"-> test macro-F1={metrics['macro_f1']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", type=Path, default=DEFAULT_VOTES_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    args = parser.parse_args()

    splits = discover_splits(args.votes_dir)
    if not splits:
        raise FileNotFoundError(f"No prepared splits found under {args.votes_dir}")
    causal = load_causal_features(args.causal_features)
    summary = {
        name: run_split(name, path, causal, args.output_root) for name, path in splits.items()
    }
    summary_path = args.output_root / "k_experts_causal_all_splits.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
