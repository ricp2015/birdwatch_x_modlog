"""Shared VAR scoring and evaluation primitives."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def iterative_cleanup(df, min_user_votes=5, min_sub_users=40):
    """Repeatedly remove sparse users and communities."""
    initial_rows = len(df)
    current = df.copy()
    iteration = 0

    while True:
        iteration += 1
        start_len = len(current)
        user_counts = current["username"].value_counts()
        valid_users = user_counts[user_counts >= min_user_votes].index
        current = current[current["username"].isin(valid_users)]

        community_counts = current.groupby("community")["username"].nunique()
        valid_communities = community_counts[
            community_counts >= min_sub_users
        ].index
        current = current[current["community"].isin(valid_communities)]

        dropped = start_len - len(current)
        if dropped == 0:
            break

    print(
        f"Cleanup: {initial_rows:,} to {len(current):,} rows | "
        f"{current['username'].nunique()} users | {current['community'].nunique()} communities"
    )
    return current


def calculate_var_scores(df, smoothing=10):
    """Calculate smoothed, label-weighted user accuracy by community."""
    clean = df.drop(columns="weight", errors="ignore").copy()
    label_counts = (
        clean.groupby(["community", "label"]).size().reset_index(name="count")
    )
    community_totals = clean.groupby("community").size().reset_index(name="total")
    weights = label_counts.merge(community_totals, on="community")
    weights["weight"] = np.log(weights["total"] / weights["count"])

    weighted = clean.merge(
        weights[["community", "label", "weight"]],
        on=["community", "label"],
        how="left",
    )
    weighted["is_correct"] = (weighted["vote"] == weighted["label"]).astype(int)
    weighted["score_contribution"] = weighted["is_correct"] * weighted["weight"]
    scores = (
        weighted.groupby(["community", "username"])
        .agg(
            total_votes=("item_id", "count"),
            weighted_score_sum=("score_contribution", "sum"),
            raw_correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    scores["VAR_score"] = scores["weighted_score_sum"] / (
        scores["total_votes"] + smoothing
    )
    scores["accuracy"] = scores["raw_correct"] / scores["total_votes"]
    return scores


def evaluate_fold(decisions):
    """Calculate metrics, using a continuous decision score for ROC-AUC."""
    y_true = decisions["label"].values
    y_pred = decisions["predicted"].values
    score_column = next(
        (column for column in ("score", "decision_score") if column in decisions),
        None,
    )
    if score_column is None:
        auc = float("nan")
        auc_score_type = "unavailable_hard_predictions_only"
    else:
        y_score = pd.to_numeric(decisions[score_column], errors="coerce").to_numpy()
        valid = np.isfinite(y_score)
        try:
            auc = float(roc_auc_score((y_true[valid] == 1).astype(int), y_score[valid]))
        except ValueError:
            auc = float("nan")
        auc_score_type = score_column
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,
        "roc_auc_score_type": auc_score_type,
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1, average="binary", zero_division=0),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_pos": precision_score(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        ),
        "recall_pos": recall_score(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        ),
        "precision_neg": precision_score(
            y_true, y_pred, pos_label=-1, average="binary", zero_division=0
        ),
        "recall_neg": recall_score(
            y_true, y_pred, pos_label=-1, average="binary", zero_division=0
        ),
    }

