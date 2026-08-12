"""Shared graph scoring and threshold calibration utilities."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score


def compute_ppr_scores(
    train_votes: pd.DataFrame,
    reliability_thr: float = 0.6,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Rank users on a co-voting graph seeded by reliable training voters."""
    users = sorted(train_votes["username"].dropna().unique())
    if not users:
        return {}

    user_index = {user: idx for idx, user in enumerate(users)}
    edge_weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for _, item_votes in train_votes.groupby("item_id"):
        rows = item_votes[["username", "vote"]].drop_duplicates("username").itertuples(index=False)
        for left, right in combinations(rows, 2):
            i, j = user_index[left.username], user_index[right.username]
            weight = 2.0 if left.vote == right.vote else 1.0
            edge_weights[i, j] += weight
            edge_weights[j, i] += weight

    if edge_weights:
        row, col, values = zip(*((i, j, value) for (i, j), value in edge_weights.items()))
        adjacency = sparse.csr_matrix((values, (row, col)), shape=(len(users), len(users)))
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
        transition = sparse.diags(inv_degree) @ adjacency
    else:
        transition = sparse.csr_matrix((len(users), len(users)))

    agreement = (
        train_votes.assign(agree=train_votes["vote"] == train_votes["label"])
        .groupby("username")["agree"]
        .mean()
    )
    seeds = np.array([agreement.get(user, 0.0) >= reliability_thr for user in users], dtype=float)
    personalization = seeds / seeds.sum() if seeds.sum() else np.full(len(users), 1.0 / len(users))

    scores = personalization.copy()
    dangling = np.asarray(transition.sum(axis=1)).ravel() == 0
    for _ in range(max_iter):
        updated = damping * (transition.T @ scores)
        updated += damping * scores[dangling].sum() * personalization
        updated += (1.0 - damping) * personalization
        if np.abs(updated - scores).sum() < tol:
            scores = updated
            break
        scores = updated
    return dict(zip(users, scores.astype(float)))


def calibrate_thresholds_per_community(
    validation_scores: pd.DataFrame,
    min_items: int = 20,
    grid_size: int = 201,
) -> tuple[dict[str, float], float]:
    """Select global and community thresholds by validation macro F1."""
    if validation_scores.empty:
        return {}, 0.0

    def best_threshold(frame: pd.DataFrame) -> float:
        values = frame["score"].to_numpy(dtype=float)
        labels = frame["label"].to_numpy()
        grid = np.linspace(values.min(), values.max(), grid_size) if len(values) else np.array([0.0])
        return float(max(grid, key=lambda threshold: f1_score(
            labels, np.where(values >= threshold, 1, -1), average="macro"
        )))

    global_threshold = best_threshold(validation_scores)
    thresholds = {
        community: best_threshold(group)
        for community, group in validation_scores.groupby("community")
        if len(group) >= min_items and group["label"].nunique() > 1
    }
    return thresholds, global_threshold


def apply_thresholds(
    scores: pd.DataFrame,
    community_thresholds: dict[str, float],
    global_threshold: float,
) -> pd.DataFrame:
    """Apply community thresholds with a global fallback."""
    predictions = scores.copy()
    thresholds = predictions["community"].map(community_thresholds).fillna(global_threshold)
    predictions["y_hat"] = np.where(predictions["score"] >= thresholds, 1, -1)
    return predictions
