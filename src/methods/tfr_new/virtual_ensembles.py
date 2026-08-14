"""Virtual sub-team ensemble built from complementary causal voter profiles."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/virtual-ensembles"
TEAMS = [
    "moderator_aligned",
    "local_experts",
    "social_core",
    "fresh_eyes",
    "experienced_generalists",
]


def add_profile_signals(votes: pd.DataFrame, ppr_scores: dict) -> pd.DataFrame:
    """Construct continuous community-specific role signals."""
    out = votes.copy()
    local = out["prior_n_posts_in_sub"] + out["prior_n_comments_in_sub"]
    global_activity = out["prior_n_posts"] + out["prior_n_comments"]
    out["_local_activity"] = np.log1p(local)
    out["_global_activity"] = np.log1p(global_activity)
    out["_domain_share"] = np.divide(local, np.maximum(global_activity, 1.0)).clip(0, 1)
    out["_social_core"] = np.log1p(
        out["prior_n_interaction_partners_in_sub"]
        + out["prior_total_interactions_in_sub"]
        + out["prior_n_replies_made"]
    )
    ppr = pd.Series(ppr_scores, dtype=float)
    ppr_rank = ppr.rank(pct=True).to_dict() if len(ppr) else {}
    out["_ppr_rank"] = out["username"].map(ppr_rank).fillna(0.5)
    out["_social_role"] = 0.5 * out["_social_core"] + 0.5 * out["_ppr_rank"]
    return out


def fit_team_thresholds(train: pd.DataFrame) -> dict[str, float]:
    """Fit role boundaries once on training profiles."""
    known_tenure = train["tenure_days_at_vote"].dropna()
    return {
        "alignment_hi": float(train["moderator_alignment_local"].quantile(0.70)),
        "alignment_mid": float(train["moderator_alignment_local"].median()),
        "local_hi": float(train["_local_activity"].quantile(0.70)),
        "local_mid": float(train["_local_activity"].median()),
        "global_hi": float(train["_global_activity"].quantile(0.70)),
        "domain_hi": float(train["_domain_share"].quantile(0.70)),
        "domain_lo": float(train["_domain_share"].quantile(0.30)),
        "social_hi": float(train["_social_role"].quantile(0.70)),
        "tenure_lo": float(known_tenure.quantile(0.30)) if len(known_tenure) else 0.0,
    }


def annotate_teams(votes: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Assign overlapping semantic roles; one voter may serve several teams."""
    out = votes.copy()
    out["team_moderator_aligned"] = (
        out["moderator_alignment_local"] >= thresholds["alignment_hi"]
    )
    out["team_local_experts"] = (
        (out["_local_activity"] >= thresholds["local_hi"])
        & (out["_domain_share"] >= thresholds["domain_hi"])
    )
    out["team_social_core"] = out["_social_role"] >= thresholds["social_hi"]
    out["team_fresh_eyes"] = (
        out["tenure_days_at_vote"].notna()
        & (out["tenure_days_at_vote"] <= thresholds["tenure_lo"])
        & (out["_local_activity"] <= thresholds["local_mid"])
        & (out["moderator_alignment_local"] >= thresholds["alignment_mid"])
    )
    out["team_experienced_generalists"] = (
        (out["_global_activity"] >= thresholds["global_hi"])
        & (out["_domain_share"] <= thresholds["domain_lo"])
    )
    return out


def team_verdict(group: pd.DataFrame) -> float:
    """Return a smoothed moderator-alignment weighted team vote."""
    if group.empty:
        return 0.0
    weights = (0.25 + group["moderator_alignment_local"].clip(0, 1)).to_numpy()
    return float(weights @ group["vote"].to_numpy() / weights.sum())


def build_item_features(votes: pd.DataFrame) -> pd.DataFrame:
    """Summarize every overlapping virtual team for each moderation case."""
    rows = []
    for item_id, group in votes.groupby("item_id", sort=False):
        feats = {
            "item_id": item_id,
            "label": group["label"].iloc[0],
            "community": group["community"].iloc[0],
            "n_voters": len(group),
            "raw_vote": float(group["vote"].mean()),
        }
        for team in TEAMS:
            members = group[group[f"team_{team}"]]
            feats[f"verdict_{team}"] = team_verdict(members)
            feats[f"n_{team}"] = len(members)
            feats[f"coverage_{team}"] = float(len(members) / len(group))
            feats[f"alignment_{team}"] = (
                float(members["moderator_alignment_local"].mean()) if len(members) else 0.0
            )
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    """Fit semantic virtual teams and a validation-calibrated meta-learner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    args = parser.parse_args()

    votes_dir, out_dir = Path(args.votes_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    train, val, test, _ = prepare_voter_representation(train, val, test)

    print("Building causal community roles...")
    ppr_scores = compute_ppr_scores(train)
    train = add_profile_signals(train, ppr_scores)
    val = add_profile_signals(val, ppr_scores)
    test = add_profile_signals(test, ppr_scores)
    thresholds = fit_team_thresholds(train)
    annotated = {
        "train": annotate_teams(train, thresholds),
        "val": annotate_teams(val, thresholds),
        "test": annotate_teams(test, thresholds),
    }
    frames = {name: build_item_features(frame) for name, frame in annotated.items()}
    feature_cols = [
        column for column in frames["train"].columns
        if column not in {"item_id", "label", "community"}
    ]
    meta_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2_000, class_weight="balanced")
    )
    meta_model.fit(frames["train"][feature_cols], frames["train"]["label"])

    score_frames = {}
    for name, frame in frames.items():
        positive_idx = int(np.where(meta_model.classes_ == 1)[0][0])
        scored = frame.copy()
        scored["score"] = meta_model.predict_proba(frame[feature_cols])[:, positive_idx]
        score_frames[name] = scored
    community_thresholds, global_threshold = calibrate_thresholds_per_community(
        score_frames["val"]
    )
    predictions = {
        name: apply_thresholds(frame, community_thresholds, global_threshold)
        for name, frame in score_frames.items()
    }
    results = {name: classification_metrics(frame) for name, frame in predictions.items()}
    results["team_thresholds"] = thresholds
    results["n_model_features"] = len(feature_cols)
    results["teams_are_overlapping"] = True
    with open(out_dir / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    membership_columns = [
        "item_id", "username", "community", "label", "vote", *[f"team_{team}" for team in TEAMS]
    ]
    annotated["test"][membership_columns].to_parquet(
        out_dir / "test_team_membership.parquet", index=False
    )
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
