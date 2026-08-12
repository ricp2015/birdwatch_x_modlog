"""Combine votes from reliability- and graph-based voter groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from shared_features import compute_ppr_scores

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/virtual-ensembles"
RELIABILITY_SEED_THR = 0.6


def compute_train_reliability(train: pd.DataFrame) -> dict:
    """Compute training-only user reliability."""
    agree = train["vote"] == train["label"]
    rel = train.assign(agree=agree).groupby("username")["agree"].mean()
    return rel.to_dict()


def assign_team(reliability: float, ppr: float, rel_median: float, ppr_median: float) -> str:
    """Assign a voter to one of four reliability and PageRank groups."""
    if reliability >= rel_median and ppr >= ppr_median:
        return "core"
    if reliability < rel_median and ppr >= ppr_median:
        return "fresh"
    if reliability >= rel_median and ppr < ppr_median:
        return "niche"
    return "other"


def team_verdict(group: pd.DataFrame) -> float:
    """Return the reliability-weighted vote for one group."""
    if len(group) == 0:
        return 0.0
    w = group["_reliability"].clip(lower=0.01)
    return float((w * group["vote"]).sum() / w.sum())


def build_item_features(votes: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per item."""
    rows = []
    for item_id, g in votes.groupby("item_id"):
        feats = {"item_id": item_id, "label": g["label"].iloc[0], "community": g["community"].iloc[0]}
        for team_name in ["core", "fresh", "niche", "other"]:
            team_g = g[g["_team"] == team_name]
            feats[f"verdict_{team_name}"] = team_verdict(team_g)
            feats[f"n_{team_name}"] = len(team_g)
        feats["n_voters"] = len(g)
        rows.append(feats)
    return pd.DataFrame(rows)


def annotate_teams(votes: pd.DataFrame, reliability: dict, ppr_scores: dict) -> pd.DataFrame:
    """Attach reliability, PageRank, and group labels to each vote."""
    votes = votes.copy()
    rel_default = np.median(list(reliability.values())) if reliability else 0.5
    ppr_default = np.median(list(ppr_scores.values())) if ppr_scores else 0.0

    votes["_reliability"] = votes["username"].map(reliability).fillna(rel_default)
    votes["_ppr"] = votes["username"].map(ppr_scores).fillna(ppr_default)

    rel_median = votes["_reliability"].median()
    ppr_median = votes["_ppr"].median()

    votes["_team"] = votes.apply(
        lambda r: assign_team(r["_reliability"], r["_ppr"], rel_median, ppr_median), axis=1
    )
    return votes


def compute_metrics(y_true, y_pred) -> dict:
    """Calculate classification metrics for one split."""
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1),
        "n_items": len(y_true),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def main():
    """Train the group-level ensemble and save its test predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(votes_dir / "train_votes.parquet").sort_values("timestamp")
    val = pd.read_parquet(votes_dir / "val_votes.parquet").sort_values("timestamp")
    test = pd.read_parquet(votes_dir / "test_votes.parquet").sort_values("timestamp")

    print("Computing reliability and PageRank scores...")
    reliability = compute_train_reliability(train)
    ppr_scores = compute_ppr_scores(train, reliability_thr=RELIABILITY_SEED_THR)

    print("Building group features...")
    train_annot = annotate_teams(train, reliability, ppr_scores)
    val_annot = annotate_teams(val, reliability, ppr_scores)
    test_annot = annotate_teams(test, reliability, ppr_scores)

    train_feats = build_item_features(train_annot)
    val_feats = build_item_features(val_annot)
    test_feats = build_item_features(test_annot)

    feature_cols = ["verdict_core", "verdict_fresh", "verdict_niche", "verdict_other",
                     "n_core", "n_fresh", "n_niche", "n_other", "n_voters"]

    print("Training the meta-model...")
    meta_model = LogisticRegression(max_iter=500, class_weight="balanced")
    meta_model.fit(train_feats[feature_cols], train_feats["label"])

    results = {}
    predictions = {}
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        y_pred = meta_model.predict(feats[feature_cols])
        feats = feats.copy()
        feats["y_hat"] = y_pred
        predictions[name] = feats
        results[name] = compute_metrics(feats["label"], y_pred)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
