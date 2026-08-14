"""Causal Hybrid LinUCB voter-trust policy with action-level DR diagnostics.

The policy follows the requested phi(i,c) = [x_c || u_i,s] representation and
softmax vote aggregation. Offline DR is reported only for the observed majority
action; voter-set propensities are not identifiable from this dataset.
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
    load_and_enrich_splits,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

RIDGE_LAMBDA = 1.0
UCB_ALPHA = 0.35
DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_DIR = "results/reddit/intersection/bandit-dr"
DEFAULT_EMBEDDING_DIR = Path("cache/embeddings")
CONTENT_CONTEXT_DIM = 8


def load_case_context_embeddings(
    embedding_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load post embeddings and fit their reduction using training cases only."""
    embeddings_path = embedding_dir / "post_embeddings.npy"
    ids_path = embedding_dir / "post_ids.json"
    if not embeddings_path.exists() or not ids_path.exists():
        print("Post embeddings unavailable: bandit content context disabled.")
        return {}, {"enabled": False, "dimension": 0, "coverage": {}}

    post_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    id_to_index = {str(item_id): index for index, item_id in enumerate(post_ids)}
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if len(post_ids) != len(embeddings):
        raise ValueError(
            f"Post embedding/id count mismatch: {len(embeddings):,} vs {len(post_ids):,}"
        )

    split_ids = {
        name: frame["item_id"].astype(str).drop_duplicates().tolist()
        for name, frame in (("train", train), ("val", val), ("test", test))
    }
    known_train = [item_id for item_id in split_ids["train"] if item_id in id_to_index]
    if len(known_train) < 2:
        print("Too few training post embeddings: bandit content context disabled.")
        return {}, {"enabled": False, "dimension": 0, "coverage": {}}

    train_matrix = np.asarray(
        embeddings[[id_to_index[item_id] for item_id in known_train]], dtype=np.float32
    )
    n_components = min(CONTENT_CONTEXT_DIM, train_matrix.shape[0] - 1, train_matrix.shape[1])
    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=10)
    reducer.fit(train_matrix)

    needed = list(dict.fromkeys(sum(split_ids.values(), [])))
    known = [item_id for item_id in needed if item_id in id_to_index]
    reduced = reducer.transform(
        np.asarray(embeddings[[id_to_index[item_id] for item_id in known]], dtype=np.float32)
    )
    # Bound extreme PCA values without fitting on validation/test data.
    train_scale = np.maximum(np.std(reducer.transform(train_matrix), axis=0), 1e-6)
    context = {
        item_id: np.tanh(vector / (3.0 * train_scale)).astype(float)
        for item_id, vector in zip(known, reduced)
    }
    coverage = {
        name: float(sum(item_id in context for item_id in ids) / max(len(ids), 1))
        for name, ids in split_ids.items()
    }
    print(
        "Post embedding coverage: "
        + ", ".join(f"{name}={value:.2%}" for name, value in coverage.items())
    )
    return context, {
        "enabled": True,
        "dimension": n_components,
        "coverage": coverage,
        "reducer_fit": "training items only",
    }


def causal_user_vector(row, state: dict) -> np.ndarray:
    """Construct u_i,s from prior online alignment and causal profile data."""
    key = (row.username, row.community)
    historical = state.get(key, {"n": 0, "agree": 0, "last_vote": 0})
    n = historical["n"]
    alignment = (historical["agree"] + 2.5) / (n + 5.0)
    tenure_known = bool(row.tenure_available)
    tenure_signal = (
        min(np.log1p(max(row.tenure_days_at_vote, 0.0)) / 9.0, 1.0)
        if tenure_known
        else 0.5
    )
    values = [
        alignment,
        min(np.log1p(n) / 5.0, 1.0),
        float(historical["last_vote"]),
        min(np.log1p(max(row.prior_n_posts + row.prior_n_comments, 0.0)) / 10.0, 1.0),
        min(
            np.log1p(max(row.prior_n_posts_in_sub + row.prior_n_comments_in_sub, 0.0)) / 8.0,
            1.0,
        ),
        min(np.log1p(max(row.prior_n_months_active_in_sub, 0.0)) / 5.0, 1.0),
        tenure_signal,
        float(tenure_known),
        min(np.log1p(max(row.prior_n_interaction_partners, 0.0)) / 7.0, 1.0),
        min(np.log1p(max(row.prior_total_interactions, 0.0)) / 9.0, 1.0),
        min(np.log1p(max(row.prior_total_interactions_in_sub, 0.0)) / 7.0, 1.0),
        1.0,
    ]
    return np.asarray(values, dtype=float)


def case_context(
    community_state: dict,
    community: str,
    n_voters: int,
    content: np.ndarray,
) -> np.ndarray:
    """Construct x_c from causal community state and observable panel size."""
    state = community_state.get(community, {"n": 0, "approve": 0})
    prior_rate = (state["approve"] + 1.0) / (state["n"] + 2.0)
    structural = np.asarray(
        [prior_rate, min(np.log1p(state["n"]) / 8.0, 1.0), min(np.log1p(n_voters) / 4.0, 1.0), 1.0],
        dtype=float,
    )
    return np.concatenate([structural, content])


class HybridLinUCB:
    """Shared case context plus per-user contextual parameters."""

    def __init__(self, d_shared: int, d_user: int, alpha: float = UCB_ALPHA, lam: float = RIDGE_LAMBDA):
        self.alpha, self.d_shared, self.d_user = alpha, d_shared, d_user
        self.A0 = lam * np.eye(d_shared)
        self.b0 = np.zeros(d_shared)
        self.per_user = {}

    def _user_state(self, username: str):
        if username not in self.per_user:
            self.per_user[username] = {
                "A": RIDGE_LAMBDA * np.eye(self.d_user),
                "B": np.zeros((self.d_user, self.d_shared)),
                "b": np.zeros(self.d_user),
            }
        return self.per_user[username]

    def score(self, username: str, z_shared: np.ndarray, x_user: np.ndarray) -> float:
        state = self._user_state(username)
        A0_inv = np.linalg.inv(self.A0)
        A_inv = np.linalg.inv(state["A"])
        beta = A0_inv @ self.b0
        theta = A_inv @ (state["b"] - state["B"] @ beta)
        mean = float(z_shared @ beta + x_user @ theta)
        variance = (
            x_user @ A_inv @ x_user
            + z_shared @ A0_inv @ z_shared
            - 2 * z_shared @ (A0_inv @ state["B"].T @ A_inv @ x_user)
            + x_user @ (A_inv @ state["B"] @ A0_inv @ state["B"].T @ A_inv) @ x_user
        )
        return mean + self.alpha * np.sqrt(max(float(variance), 0.0))

    def update(self, username: str, z_shared: np.ndarray, x_user: np.ndarray, reward: float):
        state = self._user_state(username)
        old_inv = np.linalg.inv(state["A"])
        self.A0 += state["B"].T @ old_inv @ state["B"]
        self.b0 += state["B"].T @ old_inv @ state["b"]
        state["A"] += np.outer(x_user, x_user)
        state["B"] += np.outer(x_user, z_shared)
        state["b"] += reward * x_user
        new_inv = np.linalg.inv(state["A"])
        self.A0 += np.outer(z_shared, z_shared) - state["B"].T @ new_inv @ state["B"]
        self.b0 += reward * z_shared - state["B"].T @ new_inv @ state["b"]


def aggregate_scored_votes(records: list[dict]) -> dict:
    """Softmax-normalize trust and return one community decision score."""
    trust = np.asarray([record["trust"] for record in records], dtype=float)
    weights = np.exp(trust - trust.max())
    weights /= weights.sum()
    for record, weight in zip(records, weights):
        record["trust_weight"] = float(weight)
    votes = np.asarray([record["vote"] for record in records], dtype=float)
    return {
        "item_id": records[0]["item_id"],
        "label": records[0]["label"],
        "community": records[0]["community"],
        "score": float(weights @ votes),
        "raw_score": float(votes.mean()),
        "n_voters": len(records),
        "effective_team_size": float(1.0 / np.square(weights).sum()),
    }


def replay_train(
    train_votes: pd.DataFrame,
    content_context: dict[str, np.ndarray],
    content_dim: int,
) -> tuple[HybridLinUCB, pd.DataFrame]:
    """Replay complete cases chronologically, updating only after each verdict."""
    item_order = (
        train_votes.groupby("item_id")["timestamp"].max().sort_values().index.tolist()
    )
    bandit = HybridLinUCB(d_shared=4 + content_dim, d_user=12)
    user_state: dict = {}
    community_state: dict = {}
    case_rows = []
    indexed = {item_id: group for item_id, group in train_votes.groupby("item_id", sort=False)}

    for item_id in item_order:
        group = indexed[item_id]
        community = group["community"].iloc[0]
        content = content_context.get(str(item_id), np.zeros(content_dim))
        z_shared = case_context(community_state, community, len(group), content)
        scored = []
        pending = []
        for row in group.itertuples(index=False):
            x_user = causal_user_vector(row, user_state)
            trust = bandit.score(row.username, z_shared, x_user)
            scored.append(
                dict(item_id=item_id, username=row.username, vote=row.vote, label=row.label,
                     community=community, trust=trust)
            )
            pending.append((row, x_user))
        case_rows.append(aggregate_scored_votes(scored))

        # The moderator outcome becomes available only after all case voters
        # have been scored, preventing within-case target leakage.
        for row, x_user in pending:
            reward = 1.0 if row.vote == row.label else -1.0
            bandit.update(row.username, z_shared, x_user, reward)
            key = (row.username, community)
            state = user_state.setdefault(key, {"n": 0, "agree": 0, "last_vote": 0})
            state["n"] += 1
            state["agree"] += int(row.vote == row.label)
            state["last_vote"] = row.vote
        state = community_state.setdefault(community, {"n": 0, "approve": 0})
        state["n"] += 1
        state["approve"] += int(group["label"].iloc[0] == 1)

    bandit.user_history = user_state
    bandit.community_history = community_state
    return bandit, pd.DataFrame(case_rows)


def predict_cases(
    bandit: HybridLinUCB,
    votes: pd.DataFrame,
    content_context: dict[str, np.ndarray],
    content_dim: int,
    return_voter_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Score cases with the frozen training policy and causal per-vote profiles."""
    rows = []
    voter_rows = []
    for item_id, group in votes.groupby("item_id", sort=False):
        community = group["community"].iloc[0]
        content = content_context.get(str(item_id), np.zeros(content_dim))
        z_shared = case_context(bandit.community_history, community, len(group), content)
        records = []
        for row in group.itertuples(index=False):
            x_user = causal_user_vector(row, bandit.user_history)
            records.append(
                dict(item_id=item_id, username=row.username, vote=row.vote, label=row.label,
                     community=community, trust=bandit.score(row.username, z_shared, x_user))
            )
        rows.append(aggregate_scored_votes(records))
        if return_voter_weights:
            voter_rows.extend(records)
    scores = pd.DataFrame(rows)
    if return_voter_weights:
        return scores, pd.DataFrame(voter_rows)
    return scores


def _design_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Build an action-evaluation context from subreddit and observable panel size."""
    design = pd.get_dummies(frame[["community"]], columns=["community"], dtype=float)
    design["log_n_voters"] = np.log1p(frame["n_voters"].to_numpy())
    if columns is not None:
        design = design.reindex(columns=columns, fill_value=0.0)
    return design


def fit_action_dr_models(train: pd.DataFrame) -> dict | None:
    """Fit logging-action propensity and outcome models for an action-level DR proxy."""
    observed = np.where(train["raw_score"] >= 0, 1, -1)
    reward = (observed == train["label"].to_numpy()).astype(int)
    if len(np.unique(observed)) < 2 or len(np.unique(reward)) < 2:
        return None
    context = _design_matrix(train)
    propensity = LogisticRegression(max_iter=1_000, class_weight="balanced").fit(
        context, (observed == 1).astype(int)
    )
    outcome_x = context.copy()
    outcome_x["action"] = observed
    outcome = LogisticRegression(max_iter=1_000, class_weight="balanced").fit(outcome_x, reward)
    return {"propensity": propensity, "outcome": outcome, "columns": list(context.columns)}


def action_level_dr_diagnostics(preds: pd.DataFrame, models: dict | None) -> dict | None:
    """Return an internal, explicitly non-reportable action-level DR diagnostic.

    The majority action observed in these data was not generated by a known
    randomized logging policy. Its propensity is estimated observationally,
    and the correction also ignores voter-set selection. Moreover, ordinary
    finite-sample DR terms are not bounded by the reward support. Consequently
    this estimate is useful for implementation diagnostics only, not as a
    policy-value result for the thesis.
    """
    if models is None:
        return None
    context = _design_matrix(preds, models["columns"])
    observed = np.where(preds["raw_score"] >= 0, 1, -1)
    target = preds["y_hat"].to_numpy()
    observed_reward = np.where(observed == preds["label"].to_numpy(), 1.0, -1.0)
    p_plus = models["propensity"].predict_proba(context)[:, 1]
    p_observed = np.where(observed == 1, p_plus, 1.0 - p_plus).clip(0.05, 0.95)

    def predicted_reward(actions: np.ndarray) -> np.ndarray:
        design = context.copy()
        design["action"] = actions
        return 2.0 * models["outcome"].predict_proba(design)[:, 1] - 1.0

    r_target = predicted_reward(target)
    r_observed = predicted_reward(observed)
    correction = (target == observed) / p_observed * (observed_reward - r_observed)
    terms = r_target + correction
    importance = (target == observed) / p_observed
    weight_square_sum = float(np.square(importance).sum())
    effective_sample_size = (
        float(importance.sum() ** 2 / weight_square_sum) if weight_square_sum else 0.0
    )
    estimate = float(np.mean(terms))
    return {
        "estimate_unbounded": estimate,
        "standard_error": float(np.std(terms, ddof=1) / np.sqrt(len(terms)))
        if len(terms) > 1
        else None,
        "effective_sample_size": effective_sample_size,
        "within_reward_bounds": bool(-1.0 <= estimate <= 1.0),
        "report_as_policy_value": False,
    }


def main():
    """Train the causal voter-trust policy and evaluate each fixed split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    args = parser.parse_args()
    votes_dir, out_dir = Path(args.votes_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    content_context, content_report = load_case_context_embeddings(
        args.embedding_dir, train, val, test
    )
    content_dim = int(content_report["dimension"])

    print("Replaying complete training cases...")
    bandit, train_scores = replay_train(train, content_context, content_dim)
    val_scores = predict_cases(bandit, val, content_context, content_dim)
    test_scores, test_voter_weights = predict_cases(
        bandit, test, content_context, content_dim, return_voter_weights=True
    )
    thresholds, global_threshold = calibrate_thresholds_per_community(val_scores)
    predictions = {
        "train": apply_thresholds(train_scores, thresholds, global_threshold),
        "val": apply_thresholds(val_scores, thresholds, global_threshold),
        "test": apply_thresholds(test_scores, thresholds, global_threshold),
    }
    dr_models = fit_action_dr_models(train_scores)
    results = {}
    for name, frame in predictions.items():
        results[name] = classification_metrics(frame)
        results[name]["dr_action_diagnostics"] = action_level_dr_diagnostics(frame, dr_models)
    results["dr_reporting"] = {
        "status": "excluded_from_thesis_metrics",
        "reason": (
            "internal action-level diagnostic based on observationally estimated "
            "majority-action propensities; it does not identify an off-policy value, "
            "does not correct voter-set selection, and finite-sample DR terms are unbounded"
        ),
    }
    results["policy_context"] = {
        "shared_features": 4 + content_dim,
        "user_features": 12,
        "causal_metadata": True,
        "within_case_updates": False,
        "content_embeddings": content_report,
    }
    with open(out_dir / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    test_voter_weights.to_parquet(out_dir / "test_voter_weights.parquet", index=False)
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
