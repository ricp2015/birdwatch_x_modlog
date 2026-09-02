"""Leakage-safe content/subreddit vote simulation with per-user heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score


class PostContextEncoder:
    """Encode ``<post content | subreddit>`` from frozen post embeddings.

    PCA and the subreddit vocabulary are fitted on the simulator-training
    prefix only. Unknown communities receive an all-zero community block.
    """

    def __init__(self, embedding_dir: Path, dimensions: int = 16):
        self.embedding_dir = Path(embedding_dir)
        self.requested_dimensions = int(dimensions)
        self.post_ids: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.reducer: PCA | None = None
        self.scale: np.ndarray | None = None
        self.communities: list[str] = []
        self.community_index: dict[str, int] = {}

    def fit(self, votes: pd.DataFrame) -> "PostContextEncoder":
        ids_path = self.embedding_dir / "post_ids.json"
        embeddings_path = self.embedding_dir / "post_embeddings.npy"
        if not ids_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                "Post embeddings are required for counterfactual vote simulation. "
                f"Expected {ids_path} and {embeddings_path}."
            )
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self.post_ids = {str(item_id): index for index, item_id in enumerate(ids)}
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if len(ids) != len(self.embeddings):
            raise ValueError("post_ids.json and post_embeddings.npy have different lengths")

        train_ids = [
            str(item_id)
            for item_id in votes["item_id"].drop_duplicates()
            if str(item_id) in self.post_ids
        ]
        if len(train_ids) < 2:
            raise ValueError("At least two embedded simulator-training posts are required")
        matrix = np.asarray(
            self.embeddings[[self.post_ids[item_id] for item_id in train_ids]],
            dtype=np.float32,
        )
        n_components = min(
            self.requested_dimensions,
            matrix.shape[0] - 1,
            matrix.shape[1],
        )
        self.reducer = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=10,
        ).fit(matrix)
        reduced = self.reducer.transform(matrix)
        self.scale = np.maximum(np.std(reduced, axis=0), 1e-6)
        self.communities = sorted(votes["community"].dropna().astype(str).unique())
        self.community_index = {
            community: index for index, community in enumerate(self.communities)
        }
        return self

    @property
    def content_dimensions(self) -> int:
        return 0 if self.reducer is None else int(self.reducer.n_components_)

    @property
    def output_dimensions(self) -> int:
        return 1 + self.content_dimensions + len(self.communities)

    def content_vector(self, item_id: object) -> tuple[np.ndarray, bool]:
        if self.reducer is None or self.embeddings is None or self.scale is None:
            raise RuntimeError("PostContextEncoder.fit must be called before transform")
        index = self.post_ids.get(str(item_id))
        if index is None:
            return np.zeros(self.content_dimensions, dtype=float), False
        raw = np.asarray(self.embeddings[index], dtype=np.float32).reshape(1, -1)
        reduced = self.reducer.transform(raw)[0]
        return np.tanh(reduced / (3.0 * self.scale)).astype(float), True

    def transform(self, item_ids: pd.Series, communities: pd.Series) -> np.ndarray:
        rows = []
        for item_id, community in zip(item_ids, communities):
            content, _ = self.content_vector(item_id)
            one_hot = np.zeros(len(self.communities), dtype=float)
            community_index = self.community_index.get(str(community))
            if community_index is not None:
                one_hot[community_index] = 1.0
            rows.append(np.concatenate(([1.0], content, one_hot)))
        return np.asarray(rows, dtype=float)

    def one(self, item_id: object, community: object) -> np.ndarray:
        return self.transform(pd.Series([item_id]), pd.Series([community]))[0]

    def coverage(self, votes: pd.DataFrame) -> float:
        item_ids = votes["item_id"].astype(str).drop_duplicates()
        return float(item_ids.isin(self.post_ids).mean()) if len(item_ids) else 0.0


class HierarchicalVoteSimulator:
    """Global ridge vote model with regularized per-user residual heads."""

    def __init__(
        self,
        encoder: PostContextEncoder,
        global_ridge: float = 5.0,
        user_ridge: float = 20.0,
        min_user_votes: int = 2,
    ):
        self.encoder = encoder
        self.global_ridge = float(global_ridge)
        self.user_ridge = float(user_ridge)
        self.min_user_votes = int(min_user_votes)
        self.global_coef: np.ndarray | None = None
        self.user_offsets: dict[str, np.ndarray] = {}
        self.user_counts: dict[str, int] = {}

    @staticmethod
    def _solve(matrix: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
        penalty = ridge * np.eye(matrix.shape[1], dtype=float)
        # Do not penalize the intercept as strongly as the other coefficients.
        penalty[0, 0] = max(ridge * 0.1, 1e-6)
        return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)

    def fit(self, votes: pd.DataFrame) -> "HierarchicalVoteSimulator":
        design = self.encoder.transform(votes["item_id"], votes["community"])
        target = votes["vote"].to_numpy(dtype=float)
        self.global_coef = self._solve(design, target, self.global_ridge)
        global_prediction = design @ self.global_coef
        for username, indices in votes.groupby("username", sort=False).indices.items():
            idx = np.asarray(indices, dtype=int)
            self.user_counts[str(username)] = len(idx)
            if len(idx) < self.min_user_votes:
                continue
            residual = target[idx] - global_prediction[idx]
            # Stronger shrinkage for users with little personal history.
            ridge = self.user_ridge * max(self.min_user_votes / len(idx), 0.25)
            self.user_offsets[str(username)] = self._solve(design[idx], residual, ridge)
        return self

    def predict_scores(
        self,
        item_id: object,
        community: object,
        usernames: list[str],
    ) -> np.ndarray:
        if self.global_coef is None:
            raise RuntimeError("HierarchicalVoteSimulator.fit must be called first")
        context = self.encoder.one(item_id, community)
        scores = np.empty(len(usernames), dtype=float)
        for index, username in enumerate(usernames):
            offset = self.user_offsets.get(str(username))
            coefficient = self.global_coef if offset is None else self.global_coef + offset
            scores[index] = float(context @ coefficient)
        return np.clip(scores, -1.0, 1.0)

    def evaluate_observed(self, votes: pd.DataFrame) -> dict:
        """Measure simulator fidelity only where a historical vote is observed."""
        scores = np.empty(len(votes), dtype=float)
        for (_, group), indices in zip(
            votes.groupby(["item_id", "community"], sort=False),
            votes.groupby(["item_id", "community"], sort=False).indices.values(),
        ):
            idx = np.asarray(indices, dtype=int)
            first = votes.iloc[idx[0]]
            scores[idx] = self.predict_scores(
                first["item_id"], first["community"], votes.iloc[idx]["username"].tolist()
            )
        target = votes["vote"].to_numpy(dtype=int)
        prediction = np.where(scores >= 0, 1, -1)
        result = {
            "macro_f1": float(f1_score(target, prediction, average="macro")),
            "accuracy": float(np.mean(target == prediction)),
            "n_votes": int(len(votes)),
            "post_embedding_coverage": self.encoder.coverage(votes),
        }
        try:
            result["roc_auc"] = float(roc_auc_score((target == 1).astype(int), scores))
        except ValueError:
            result["roc_auc"] = None
        return result
