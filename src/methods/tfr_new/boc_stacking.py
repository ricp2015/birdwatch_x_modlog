"""Stack vote aggregates and the most reliable voter features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shared_features import (
    DEFAULT_CAUSAL_FEATURES,
    load_and_enrich_splits,
    prepare_voter_representation,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/boc-stacking"
TOP_K = 5
MIN_VOTES_FOR_RELIABILITY = 3

METADATA_FEATURES = {
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


def build_item_features(
    votes: pd.DataFrame,
    user_stats: pd.DataFrame,
    global_default: float,
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize each item with aggregate votes and top-voter features."""
    metadata_columns = metadata_columns or []
    rows = []
    grouped = votes.groupby("item_id", sort=False)
    for item_id, g in tqdm(grouped, total=votes["item_id"].nunique(), desc="Item features"):
        if "moderator_alignment_global" in g:
            rel = g["moderator_alignment_global"].fillna(global_default).to_numpy(dtype=float)
            n_hist = g["moderator_alignment_global_n"].fillna(0).to_numpy(dtype=float)
        else:
            rel_n = [
                user_reliability_lookup(user, user_stats, global_default)
                for user in g["username"]
            ]
            rel = np.array([value for value, _ in rel_n])
            n_hist = np.array([count for _, count in rel_n])
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

        # Counts are heavy-tailed, so aggregate log1p-transformed values. Keep
        # only four summaries per signal to limit overfitting on the relatively
        # small number of training items.
        for column in metadata_columns:
            values = pd.to_numeric(g[column], errors="coerce").clip(lower=0)
            transformed = np.log1p(values)
            valid = transformed.notna()
            pos = transformed[(vote == 1) & valid.to_numpy()]
            neg = transformed[(vote == -1) & valid.to_numpy()]
            all_mean = float(transformed[valid].mean()) if valid.any() else 0.0
            pos_mean = float(pos.mean()) if len(pos) else 0.0
            neg_mean = float(neg.mean()) if len(neg) else 0.0
            prefix = f"meta_{column.removeprefix('prior_')}"
            feats[f"{prefix}_mean"] = all_mean
            feats[f"{prefix}_pos_mean"] = pos_mean
            feats[f"{prefix}_neg_mean"] = neg_mean
            feats[f"{prefix}_neg_minus_pos"] = neg_mean - pos_mean

        rows.append(feats)
    return pd.DataFrame(rows)


def select_model(X_train, y_train):
    """Select the final classifier by cross-validated macro F1."""
    candidates = {
        "ridge": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, class_weight="balanced"),
        ),
        "gbc": GradientBoostingClassifier(random_state=10),
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


def positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Return P(label=+1) from a fitted sklearn classifier or pipeline."""
    positive_idx = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(features)[:, positive_idx]


def calibrate_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    """Choose a probability threshold on validation macro F1 only."""
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), y_score]))
    best_threshold = 0.5
    best_f1 = -np.inf
    for threshold in candidates:
        prediction = np.where(y_score >= threshold, 1, -1)
        score = f1_score(y_true, prediction, average="macro")
        if score > best_f1 or (
            np.isclose(score, best_f1) and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Calculate classification metrics for one split."""
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1),
        "n_items": len(y_true),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score if y_score is not None else y_pred)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def run_ablation(
    mode: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Train and evaluate one metadata ablation mode."""
    print(f"\n=== Metadata mode: {mode} ===")
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    print("Computing user statistics...")
    user_stats = compute_train_user_stats(train)
    global_default = float((train["vote"] == train["label"]).mean())

    print("Building item features...")
    metadata_columns = METADATA_FEATURES[mode]
    train_feats = build_item_features(train, user_stats, global_default, metadata_columns)
    val_feats = build_item_features(val, user_stats, global_default, metadata_columns)
    test_feats = build_item_features(test, user_stats, global_default, metadata_columns)

    feature_cols = [c for c in train_feats.columns if c not in ("item_id", "label", "community")]

    print("Selecting the model...")
    model = select_model(train_feats[feature_cols], train_feats["label"])

    val_scores = positive_scores(model, val_feats[feature_cols])
    threshold, calibration_f1 = calibrate_threshold(val_feats["label"], val_scores)
    print(
        f"Validation threshold: {threshold:.4f} "
        f"(validation macro F1 = {calibration_f1:.4f})"
    )

    results = {}
    predictions = {}
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        y_score = positive_scores(model, feats[feature_cols])
        y_pred = np.where(y_score >= threshold, 1, -1)
        feats = feats.copy()
        feats["y_hat"] = y_pred
        feats["score"] = y_score
        predictions[name] = feats
        results[name] = compute_metrics(feats["label"], y_pred, y_score)

    results["metadata_mode"] = mode
    results["n_model_features"] = len(feature_cols)
    results["validation_threshold"] = threshold
    results["validation_threshold_macro_f1"] = calibration_f1
    with open(mode_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    predictions["test"].to_parquet(mode_dir / "test_predictions.parquet", index=False)

    print(json.dumps(results, indent=2))
    return results


def main():
    """Run one or all causal-metadata ablations and save comparable metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument(
        "--metadata-mode",
        choices=["all", *METADATA_FEATURES],
        default="all",
        help="Ablation to run; 'all' runs none/global/subreddit/full.",
    )
    args = parser.parse_args()

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_and_enrich_splits(votes_dir, Path(args.causal_features))
    train, val, test, _ = prepare_voter_representation(train, val, test)
    modes = list(METADATA_FEATURES) if args.metadata_mode == "all" else [args.metadata_mode]

    all_results = {}
    for mode in modes:
        all_results[mode] = run_ablation(mode, train, val, test, out_dir)

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    summary_rows = []
    for mode, result in all_results.items():
        summary_rows.append({"metadata_mode": mode, **result["test"]})
    pd.DataFrame(summary_rows).to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"\nAblation summary saved to {out_dir}")


if __name__ == "__main__":
    main()
