"""Stack vote aggregates and the most reliable voter features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/boc-stacking"
TOP_K = 5
MIN_VOTES_FOR_RELIABILITY = 3


def compute_train_user_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Compute training-only user reliability."""
    agree = train["vote"] == train["label"]
    stats = train.assign(agree=agree).groupby("username").agg(
        reliability=("agree", "mean"), n_votes=("agree", "size")
    )
    return stats


def user_reliability_lookup(username: str, user_stats: pd.DataFrame, global_default: float) -> tuple:
    """Return a user's reliability and vote count, with a global fallback."""
    if username not in user_stats.index:
        return global_default, 0
    row = user_stats.loc[username]
    if row["n_votes"] < MIN_VOTES_FOR_RELIABILITY:
        return global_default, int(row["n_votes"])
    return float(row["reliability"]), int(row["n_votes"])


def build_item_features(votes: pd.DataFrame, user_stats: pd.DataFrame, global_default: float) -> pd.DataFrame:
    """Summarize each item with aggregate votes and top-voter features."""
    rows = []
    for item_id, g in votes.groupby("item_id"):
        rel_n = [user_reliability_lookup(u, user_stats, global_default) for u in g["username"]]
        rel = np.array([r for r, _ in rel_n])
        n_hist = np.array([n for _, n in rel_n])
        vote = g["vote"].to_numpy()

        feats = {
            "item_id": item_id,
            "label": g["label"].iloc[0],
            "community": g["community"].iloc[0],
            "n_voters": len(g),
            "n_pos": int((vote == 1).sum()),
            "n_neg": int((vote == -1).sum()),
            "net_vote": int(vote.sum()),
            "reliability_weighted_vote": float((rel * vote).sum() / max(rel.sum(), 1e-6)),
        }

        # Keep a fixed number of voter slots per item.
        order = np.argsort(-rel)[:TOP_K]
        for rank in range(TOP_K):
            if rank < len(order):
                i = order[rank]
                feats[f"rank{rank+1}_reliability"] = rel[i]
                feats[f"rank{rank+1}_vote"] = vote[i]
                feats[f"rank{rank+1}_n_hist"] = n_hist[i]
            else:
                feats[f"rank{rank+1}_reliability"] = 0.0
                feats[f"rank{rank+1}_vote"] = 0
                feats[f"rank{rank+1}_n_hist"] = 0

        rows.append(feats)
    return pd.DataFrame(rows)


def select_model(X_train, y_train):
    """Select the final classifier by cross-validated macro F1."""
    candidates = {
        "ridge": LogisticRegression(max_iter=500, class_weight="balanced"),
        "gbc": GradientBoostingClassifier(),
    }
    best_name, best_model, best_score = None, None, -np.inf
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_name, best_model, best_score = name, model, mean_score
    best_model.fit(X_train, y_train)
    print(f"Selected {best_name} (CV macro F1 = {best_score:.4f})")
    return best_model


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
    """Train the stacker and save test predictions and split metrics."""
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

    print("Computing user statistics...")
    user_stats = compute_train_user_stats(train)
    global_default = float((train["vote"] == train["label"]).mean())

    print("Building item features...")
    train_feats = build_item_features(train, user_stats, global_default)
    val_feats = build_item_features(val, user_stats, global_default)
    test_feats = build_item_features(test, user_stats, global_default)

    feature_cols = [c for c in train_feats.columns if c not in ("item_id", "label", "community")]

    print("Selecting the model...")
    model = select_model(train_feats[feature_cols], train_feats["label"])

    results = {}
    predictions = {}
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        y_pred = model.predict(feats[feature_cols])
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
