"""Hybrid LinUCB with a doubly robust evaluation estimate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shared_features import calibrate_thresholds_per_community, apply_thresholds

RIDGE_LAMBDA = 1.0
UCB_ALPHA = 0.6
MIN_SUB_SAMPLES = 30


def build_context_features(posts: pd.DataFrame, running_stats: dict) -> pd.DataFrame:
    """Return item features available at the current replay point."""
    return posts


def user_feature_vector(user_state: dict, community: str, base_dim: int) -> np.ndarray:
    """Build the current user-community state vector."""
    key = (user_state.get("username"), community)
    st = user_state["by_user_sub"].get(key)
    if st is None:
        return np.zeros(base_dim)
    n = max(st["n"], 1)
    reliability = st["agree"] / n
    return np.array([reliability, min(np.log1p(n), 5.0), st["last_vote"]])


class HybridLinUCB:
    """Maintain shared and per-user parameters for Hybrid LinUCB."""

    def __init__(self, d_shared: int, d_user: int, alpha: float = UCB_ALPHA,
                 lam: float = RIDGE_LAMBDA):
        """Initialize regularized shared and per-user state."""
        self.alpha = alpha
        self.d_shared = d_shared
        self.d_user = d_user
        self.A0 = lam * np.eye(d_shared)
        self.b0 = np.zeros(d_shared)
        self.per_user = {}  # username -> dict(A, B, Ainv_cached lazily, b)

    def _user_state(self, username: str):
        """Create or return the parameter state for one user."""
        if username not in self.per_user:
            self.per_user[username] = {
                "A": RIDGE_LAMBDA * np.eye(self.d_user),
                "B": np.zeros((self.d_user, self.d_shared)),
                "b": np.zeros(self.d_user),
            }
        return self.per_user[username]

    def score(self, username: str, z_shared: np.ndarray, x_user: np.ndarray) -> float:
        """Score a user from shared and user-specific context."""
        st = self._user_state(username)
        A0_inv = np.linalg.inv(self.A0)
        A_inv = np.linalg.inv(st["A"])
        beta = A0_inv @ self.b0
        theta = A_inv @ (st["b"] - st["B"] @ beta)

        s = float(z_shared @ beta + x_user @ theta)
        # Standard Hybrid LinUCB confidence term.
        var = (
            x_user @ A_inv @ x_user
            + z_shared @ A0_inv @ z_shared
            - 2 * z_shared @ (A0_inv @ st["B"].T @ A_inv @ x_user)
            + x_user @ (A_inv @ st["B"] @ A0_inv @ st["B"].T @ A_inv) @ x_user
        )
        var = max(var, 0.0)
        return s + self.alpha * np.sqrt(var)

    def update(self, username: str, z_shared: np.ndarray, x_user: np.ndarray, reward: float):
        """Update shared and user parameters from one observed reward."""
        st = self._user_state(username)
        A0_inv = np.linalg.inv(self.A0)

        self.A0 += st["B"].T @ np.linalg.inv(st["A"]) @ st["B"]
        self.b0 += st["B"].T @ np.linalg.inv(st["A"]) @ st["b"]

        st["A"] += np.outer(x_user, x_user)
        st["B"] += np.outer(x_user, z_shared)
        st["b"] += reward * x_user

        A0_inv_new = np.linalg.inv(self.A0)
        self.A0 = self.A0 + np.outer(z_shared, z_shared) \
            - st["B"].T @ np.linalg.inv(st["A"]) @ st["B"]
        self.b0 = self.b0 + reward * z_shared \
            - st["B"].T @ np.linalg.inv(st["A"]) @ st["b"]


def replay_train(train_votes: pd.DataFrame, d_shared: int, d_user: int) -> HybridLinUCB:
    """Fit the bandit by replaying training votes in timestamp order."""
    train_votes = train_votes.sort_values("timestamp").reset_index(drop=True)
    bandit = HybridLinUCB(d_shared, d_user)

    # Update state after each observation to avoid temporal leakage.
    user_state = {"by_user_sub": {}}
    sub_running = {}  # community -> {n_posts, n_approve}

    for row in train_votes.itertuples(index=False):
        community = row.community
        stats = sub_running.setdefault(community, {"n": 0, "approve": 0})
        n_seen = max(stats["n"], 1)
        z_shared = np.array([
            stats["approve"] / n_seen,
            min(np.log1p(stats["n"]), 5.0),
            1.0,
        ])
        x_user = user_feature_vector(
            {"username": row.username, "by_user_sub": user_state["by_user_sub"]},
            community, d_user,
        )

        score = bandit.score(row.username, z_shared, x_user)  # noqa: F841
        reward = 1.0 if row.vote == row.label else -1.0
        bandit.update(row.username, z_shared, x_user, reward)

        key = (row.username, community)
        st = user_state["by_user_sub"].setdefault(key, {"n": 0, "agree": 0, "last_vote": 0})
        st["n"] += 1
        st["agree"] += 1 if row.vote == row.label else 0
        st["last_vote"] = row.vote

        # Community statistics are updated last.
        stats["n"] += 1
        stats["approve"] += 1 if row.label == 1 else 0

    bandit._user_state_snapshot = user_state["by_user_sub"]
    bandit._sub_running_snapshot = sub_running
    return bandit


def predict_cases(bandit: HybridLinUCB, votes: pd.DataFrame, d_shared: int, d_user: int) -> pd.DataFrame:
    """Predict with frozen state and softmax-weighted votes."""
    user_state = bandit._user_state_snapshot
    sub_running = bandit._sub_running_snapshot

    rows = []
    for row in votes.itertuples(index=False):
        community = row.community
        stats = sub_running.get(community, {"n": 0, "approve": 0})
        n_seen = max(stats["n"], 1)
        z_shared = np.array([stats["approve"] / n_seen, min(np.log1p(stats["n"]), 5.0), 1.0])
        x_user = user_feature_vector({"username": row.username, "by_user_sub": user_state}, community, d_user)
        trust = bandit.score(row.username, z_shared, x_user)
        rows.append((row.item_id, row.username, row.vote, row.label, community, trust))

    df = pd.DataFrame(rows, columns=["item_id", "username", "vote", "label", "community", "trust"])

    def agg(g):
        """Reduce one item's votes to a trust-weighted score."""
        w = np.exp(g["trust"] - g["trust"].max())
        w = w / w.sum()
        score = float((w * g["vote"]).sum())
        return pd.Series({"label": g["label"].iloc[0], "community": g["community"].iloc[0],
                           "score": score, "n_voters": len(g)})

    return df.groupby("item_id").apply(agg).reset_index()


def estimate_propensity(train_preds: pd.DataFrame) -> LogisticRegression:
    """Estimate approval propensity from training data."""
    X = pd.get_dummies(train_preds[["community"]], columns=["community"])
    X["n_voters"] = train_preds["n_voters"]
    y = (train_preds["label"] == 1).astype(int)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X, y)
    model._columns = X.columns
    return model


def doubly_robust_value(preds: pd.DataFrame, propensity_model: LogisticRegression) -> float:
    """Compute a self-normalized doubly robust value estimate."""
    X = pd.get_dummies(preds[["community"]], columns=["community"])
    for col in propensity_model._columns:
        if col not in X.columns:
            X[col] = 0
    X = X[propensity_model._columns.drop("n_voters") if "n_voters" in propensity_model._columns else propensity_model._columns]
    X["n_voters"] = preds["n_voters"]
    X = X[propensity_model._columns]

    p_c = propensity_model.predict_proba(X)[:, 1]
    p_c = np.clip(p_c, 0.05, 0.95)

    r_c = np.where(preds["y_hat"] == preds["label"], 1.0, -1.0)
    r_hat = 2 * p_c - 1
    indicator = (preds["y_hat"] == preds["label"]).astype(float)

    w = indicator / p_c
    w_sum = w.sum()
    correction = float((w * (r_c - r_hat)).sum() / w_sum) if w_sum > 0 else 0.0
    return float(np.mean(r_hat) + correction)


def compute_metrics(preds: pd.DataFrame) -> dict:
    """Calculate classification metrics from thresholded item scores."""
    from sklearn.metrics import f1_score, roc_auc_score

    y_true = preds["label"]
    y_pred = preds["y_hat"]
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1),
        "n_items": len(preds),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/bandit-dr"


def main():
    """Train the bandit, calibrate scores, and save test predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR,
                         help="Prepared split directory (default: %(default)s)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help="default: %(default)s")
    args = parser.parse_args()

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(votes_dir / "train_votes.parquet").sort_values("timestamp")
    val = pd.read_parquet(votes_dir / "val_votes.parquet").sort_values("timestamp")
    test = pd.read_parquet(votes_dir / "test_votes.parquet").sort_values("timestamp")

    d_shared, d_user = 3, 3

    print("Replaying training votes...")
    bandit = replay_train(train, d_shared, d_user)

    print("Scoring train, validation, and test sets...")
    train_scores = predict_cases(bandit, train, d_shared, d_user)
    val_scores = predict_cases(bandit, val, d_shared, d_user)
    test_scores = predict_cases(bandit, test, d_shared, d_user)

    print("Calibrating community thresholds...")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_scores)
    train_preds = apply_thresholds(train_scores, thresholds, global_threshold)
    val_preds = apply_thresholds(val_scores, thresholds, global_threshold)
    test_preds = apply_thresholds(test_scores, thresholds, global_threshold)

    propensity_model = estimate_propensity(train_preds)

    results = {
        "train": {**compute_metrics(train_preds),
                  "dr_value": doubly_robust_value(train_preds, propensity_model)},
        "val": {**compute_metrics(val_preds),
                "dr_value": doubly_robust_value(val_preds, propensity_model)},
        "test": {**compute_metrics(test_preds),
                 "dr_value": doubly_robust_value(test_preds, propensity_model)},
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    test_preds.to_parquet(out_dir / "test_predictions.parquet", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
