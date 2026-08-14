"""Dynamic skill-and-affinity graph propagation for vote aggregation.

The historical user-case bipartite graph is projected into a user affinity
graph from past co-votes. Personalized PageRank supplies structural proximity;
case-specific attention then combines it with leakage-safe moderator alignment
and causal community/social features before softmax-normalizing the voters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shared_features import (
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    classification_metrics,
    compute_ppr_scores,
    load_and_enrich_splits,
    prepare_voter_representation,
)

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/graph-propagation"


def _percentile_map(values: dict[str, float]) -> dict[str, float]:
    """Convert graph scores to stable percentile ranks."""
    if not values:
        return {}
    series = pd.Series(values, dtype=float)
    return series.rank(pct=True).to_dict()


def add_attention_components(votes: pd.DataFrame, ppr_scores: dict) -> pd.DataFrame:
    """Attach graph, skill, domain-affinity, tenure, and social components."""
    out = votes.copy()
    ppr_rank = _percentile_map(ppr_scores)
    out["attention_graph"] = out["username"].map(ppr_rank).fillna(0.5)
    out["attention_alignment"] = out["moderator_alignment_local"].fillna(0.5)

    local_activity = out["prior_n_posts_in_sub"] + out["prior_n_comments_in_sub"]
    global_activity = out["prior_n_posts"] + out["prior_n_comments"]
    # Community share is the available domain-affinity proxy until textual
    # user-case affinity is materialized for every split.
    out["attention_domain_affinity"] = np.divide(
        local_activity,
        np.maximum(global_activity, 1.0),
    ).clip(0.0, 1.0)
    out["attention_local_experience"] = 1.0 - np.exp(-np.log1p(local_activity) / 4.0)
    tenure_signal = 1.0 - np.exp(
        -np.log1p(out["tenure_days_at_vote"].clip(lower=0.0)) / 6.0
    )
    out["attention_tenure"] = tenure_signal.fillna(0.5)
    social = (
        out["prior_n_interaction_partners_in_sub"]
        + out["prior_total_interactions_in_sub"]
        + out["prior_n_replies_made"]
    )
    out["attention_social_core"] = 1.0 - np.exp(-np.log1p(social) / 4.0)
    return out


def aggregate_with_attention(
    votes: pd.DataFrame,
    ppr_scores: dict,
    component_weights: dict[str, float],
    return_voter_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Compute dynamic voter attention and aggregate each moderation case."""
    annotated = add_attention_components(votes, ppr_scores)
    annotated["attention_logit"] = 0.0
    for component, weight in component_weights.items():
        annotated["attention_logit"] += weight * annotated[component]

    rows = []
    voter_rows = []
    for item_id, group in annotated.groupby("item_id", sort=False):
        logits = group["attention_logit"].to_numpy(dtype=float)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        if return_voter_weights:
            voter_frame = group[
                [
                    "item_id",
                    "username",
                    "community",
                    "label",
                    "vote",
                    "attention_logit",
                    "attention_graph",
                    "attention_alignment",
                    "attention_domain_affinity",
                    "attention_local_experience",
                    "attention_tenure",
                    "attention_social_core",
                ]
            ].copy()
            voter_frame["attention_weight"] = weights
            voter_rows.append(voter_frame)
        rows.append(
            {
                "item_id": item_id,
                "label": group["label"].iloc[0],
                "community": group["community"].iloc[0],
                "score": float(weights @ group["vote"].to_numpy(dtype=float)),
                "n_voters": len(group),
                "effective_team_size": float(1.0 / np.square(weights).sum()),
                "mean_graph_attention": float(group["attention_graph"].mean()),
                "mean_local_alignment": float(group["attention_alignment"].mean()),
            }
        )
    scores = pd.DataFrame(rows)
    if return_voter_weights:
        voter_weights = pd.concat(voter_rows, ignore_index=True) if voter_rows else pd.DataFrame()
        return scores, voter_weights
    return scores


def select_attention_weights(val: pd.DataFrame, ppr_scores: dict) -> tuple[dict, pd.DataFrame]:
    """Select a small interpretable attention family on validation AUC."""
    candidates = {
        "graph_only": {"attention_graph": 1.0},
        "skill_affinity": {
            "attention_graph": 1.0,
            "attention_alignment": 1.0,
            "attention_domain_affinity": 0.75,
            "attention_local_experience": 0.50,
            "attention_tenure": 0.25,
        },
        "full_social": {
            "attention_graph": 1.0,
            "attention_alignment": 1.0,
            "attention_domain_affinity": 0.75,
            "attention_local_experience": 0.50,
            "attention_tenure": 0.25,
            "attention_social_core": 0.50,
        },
    }
    best_name, best_weights, best_scores, best_auc = None, None, None, -np.inf
    from sklearn.metrics import roc_auc_score

    for name, weights in candidates.items():
        scores = aggregate_with_attention(val, ppr_scores, weights)
        try:
            auc = float(roc_auc_score(scores["label"], scores["score"]))
        except ValueError:
            auc = -np.inf
        if auc > best_auc:
            best_name, best_weights, best_scores, best_auc = name, weights, scores, auc
    print(f"Selected attention={best_name} (validation AUC={best_auc:.4f})")
    return {"name": best_name, "weights": best_weights, "validation_auc": best_auc}, best_scores


def main():
    """Build the affinity graph, calibrate on validation, and evaluate test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    args = parser.parse_args()

    votes_dir, out_dir = Path(args.votes_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    train, val, test, _ = prepare_voter_representation(train, val, test)

    print("Building historical skill-and-affinity graph...")
    ppr_scores = compute_ppr_scores(train)
    selected, val_scores = select_attention_weights(val, ppr_scores)
    train_scores = aggregate_with_attention(train, ppr_scores, selected["weights"])
    test_scores, test_voter_weights = aggregate_with_attention(
        test, ppr_scores, selected["weights"], return_voter_weights=True
    )

    thresholds, global_threshold = calibrate_thresholds_per_community(val_scores)
    predictions = {
        "train": apply_thresholds(train_scores, thresholds, global_threshold),
        "val": apply_thresholds(val_scores, thresholds, global_threshold),
        "test": apply_thresholds(test_scores, thresholds, global_threshold),
    }
    results = {name: classification_metrics(frame) for name, frame in predictions.items()}
    results["selected_attention"] = selected
    results["n_graph_users"] = len(ppr_scores)
    with open(out_dir / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    test_voter_weights.to_parquet(out_dir / "test_voter_weights.parquet", index=False)
    pd.DataFrame({"username": ppr_scores.keys(), "ppr": ppr_scores.values()}).to_parquet(
        out_dir / "user_graph_scores.parquet", index=False
    )
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
