"""Shared Net Vote fallback for benchmark methods with incomplete native coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

NET_VOTE_THRESHOLD_GRID = np.linspace(-200.0, 200.0, 800)


def net_vote_item_scores(votes: pd.DataFrame) -> pd.DataFrame:
    """Return one Net Vote score and label row per item."""
    required = {"item_id", "community", "label", "vote"}
    missing = required.difference(votes.columns)
    if missing:
        raise ValueError(f"Net Vote fallback input is missing columns: {sorted(missing)}")
    return (
        votes.groupby("item_id", as_index=False)
        .agg(
            community=("community", "first"),
            label=("label", "first"),
            net_vote_score=("vote", "sum"),
        )
    )


def calibrate_net_vote_threshold(
    validation_votes: pd.DataFrame,
    grid: np.ndarray = NET_VOTE_THRESHOLD_GRID,
) -> float:
    """Fit the same global validation threshold used by the Net Vote baseline."""
    scores = net_vote_item_scores(validation_votes)
    if scores.empty:
        raise ValueError("Cannot calibrate the Net Vote fallback on an empty validation set")
    labels = scores["label"].to_numpy(dtype=int)
    values = scores["net_vote_score"].to_numpy(dtype=float)
    macro_f1 = np.array(
        [
            f1_score(
                labels,
                np.where(values >= threshold, 1, -1),
                average="macro",
                zero_division=0,
            )
            for threshold in grid
        ]
    )
    return float(grid[int(np.argmax(macro_f1))])


def add_net_vote_fallback(
    native_predictions: pd.DataFrame,
    evaluation_votes: pd.DataFrame,
    threshold: float,
    *,
    prediction_column: str,
    native_score_column: str,
    native_source: str = "native",
) -> pd.DataFrame:
    """Complete missing item predictions with validation-calibrated Net Vote."""
    required = {"item_id", "label", prediction_column}
    missing = required.difference(native_predictions.columns)
    if missing:
        raise ValueError(f"Native prediction table is missing columns: {sorted(missing)}")
    if native_predictions["item_id"].duplicated().any():
        raise ValueError("Native prediction table contains duplicate item_id values")

    all_items = net_vote_item_scores(evaluation_votes)
    native = native_predictions.copy()
    native["is_fallback"] = False
    native["decision_source"] = native_source
    native["net_vote_score"] = np.nan
    native["net_vote_threshold"] = float(threshold)

    missing_items = all_items.loc[
        ~all_items["item_id"].isin(native["item_id"])
    ].copy()
    if missing_items.empty:
        return native.sort_values("item_id", kind="stable").reset_index(drop=True)

    fallback = missing_items[["item_id", "community", "label", "net_vote_score"]].copy()
    fallback[prediction_column] = np.where(
        fallback["net_vote_score"].to_numpy(dtype=float) >= float(threshold),
        1,
        -1,
    )
    fallback[native_score_column] = np.nan
    fallback["is_fallback"] = True
    fallback["decision_source"] = "net_vote_fallback"
    fallback["net_vote_threshold"] = float(threshold)

    completed = pd.concat([native, fallback], ignore_index=True, sort=False)
    expected = int(evaluation_votes["item_id"].nunique())
    if len(completed) != expected or completed["item_id"].nunique() != expected:
        raise AssertionError("Net Vote fallback did not produce one decision per test item")
    return completed.sort_values("item_id", kind="stable").reset_index(drop=True)
