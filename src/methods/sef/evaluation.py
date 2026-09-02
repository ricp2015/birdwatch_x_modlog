"""Hyperparameter search, prediction, and metrics for SEF."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .runtime import (
    ALPHA,
    BETA,
    GRID_ALPHA,
    GRID_BETA,
    GRID_K,
    K_NEIGHBORS,
    TOP_T,
    log,
)
from .features import _compute_weighted_votes, compute_expert_weights
from .modeling import (
    _items_with_real_expert_signal,
    _net_vote_predictions,
    apply_neg_predictor,
    predict_meta_model,
)

# 9. GRID SEARCH (K/alpha/beta; multi-scale T is learned by the meta-model)


def grid_search_hyperparams(
    val_ids: List[str],
    train_ids: set,
    vote_df: pd.DataFrame,
    prec_df: pd.DataFrame,
    post_emb: np.ndarray,
    post_id2idx: Dict[str, int],
    user_emb: np.ndarray,
    user_id2idx: Dict[str, int],
    faiss_index: faiss.Index,
    lambda_smooth: float,
    use_net_vote_fallback: bool = False,
) -> Tuple[int, int, float, float]:
    """Select expert-weighting parameters on validation data."""
    labels = (
        vote_df[vote_df["item_id"].isin(val_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )
    val_ids = [iid for iid in val_ids if iid in labels]
    if not val_ids:
        return K_NEIGHBORS, TOP_T, ALPHA, BETA

    valid_ab = [(a, b) for a in GRID_ALPHA for b in GRID_BETA if a + b <= 1.0]
    n_combos = len(GRID_K) * len(valid_ab)
    log.info(
        "Hyperparameter search: %d combinations on %d validation posts",
        n_combos,
        len(val_ids),
    )

    best = {"f1": -1.0, "k": K_NEIGHBORS, "t": TOP_T, "alpha": ALPHA, "beta": BETA}

    for k in GRID_K:
        for alpha, beta in valid_ab:
            weights_df = compute_expert_weights(
                val_ids,
                train_ids,
                vote_df,
                prec_df,
                post_emb,
                post_id2idx,
                user_emb,
                user_id2idx,
                faiss_index,
                k,
                alpha,
                beta,
                lambda_smooth,
            )
            # The grid uses the fixed fallback scale only as a cheap proxy for
            # K/alpha/beta.  All TOP_T_LIST scales enter the final meta-model.
            wv = _compute_weighted_votes(val_ids, vote_df, weights_df, TOP_T)
            if use_net_vote_fallback:
                expert_ids = _items_with_real_expert_signal(weights_df)
                raw_mean, _ = _net_vote_predictions(val_ids, vote_df)
                for item_id in val_ids:
                    if item_id not in expert_ids:
                        wv[item_id] = float(raw_mean.get(item_id, 0.0))
            common = [iid for iid in wv if iid in labels]
            if not common:
                continue
            y_true = np.array([labels[iid] for iid in common])
            y_score = np.array([wv[iid] for iid in common])
            best_f1 = -1.0
            for thr in np.linspace(y_score.min(), y_score.max(), 50):
                y_pred = np.where(y_score >= thr, 1, -1)
                f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                best_f1 = max(best_f1, f1)

            if best_f1 > best["f1"]:
                best = {
                    "f1": best_f1,
                    "k": k,
                    "t": TOP_T,
                    "alpha": alpha,
                    "beta": beta,
                }

    log.info(
        "Selected parameters: k=%d, fallback_t=%d, alpha=%.2f, beta=%.2f, macro-F1=%.4f",
        best["k"],
        best["t"],
        best["alpha"],
        best["beta"],
        best["f1"],
    )
    return best["k"], best["t"], best["alpha"], best["beta"]


# 10. PREDICTION


def predict_fold(
    test_ids: List[str],
    vote_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    feat_cols: List[str],
    clf: Any,
    scaler: StandardScaler,
    thresholds: Dict[Optional[str], float],
    top_t: int,
    neg_clf: Optional[Any] = None,
    neg_scaler: Optional[StandardScaler] = None,
    neg_feat_cols: Optional[List[str]] = None,
    use_net_vote_fallback: bool = False,
) -> pd.DataFrame:
    """Generate predictions for one evaluation fold."""
    labels = vote_df[vote_df["item_id"].isin(test_ids)].drop_duplicates("item_id")[
        ["item_id", "label"]
    ]
    post_sub = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )

    # Compute negative-predictor probabilities and add them to the features.
    neg_probas = apply_neg_predictor(feat_df, test_ids, neg_clf, neg_scaler, neg_feat_cols)
    feat_df = feat_df.copy()
    feat_df["neg_expert_proba"] = feat_df["item_id"].map(neg_probas).fillna(0.5)

    meta_preds = predict_meta_model(feat_df, test_ids, clf, scaler, feat_cols)
    wv = _compute_weighted_votes(test_ids, vote_df, weights_df, top_t)
    expert_ids = _items_with_real_expert_signal(weights_df)
    raw_mean, _ = _net_vote_predictions(test_ids, vote_df)

    rows = []
    for item_id in test_ids:
        sub = post_sub.get(item_id)
        thr = thresholds.get(sub, thresholds.get(None, 0.5))
        score = wv.get(item_id, 0.0)
        fallback = item_id not in expert_ids
        meta_pred, meta_proba = meta_preds.get(item_id, (None, None))
        if use_net_vote_fallback and fallback:
            score = float(raw_mean.get(item_id, 0.0))
            meta_proba = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
            predicted = 1 if score >= 0.0 else -1
            score_source = "net_vote_fallback"
        elif meta_proba is not None:
            predicted = 1 if meta_proba >= thr else -1
            score_source = "meta_model"
        else:
            predicted = 1 if score >= 0.0 else -1
            score_source = "weighted_vote_fallback"
        rows.append(
            {
                "item_id": item_id,
                "weighted_vote": score,
                "meta_proba": meta_proba if meta_proba is not None else 0.5,
                "neg_expert_proba": neg_probas.get(item_id, 0.5),
                "predicted": predicted,
                "used_fallback": fallback,
                "score_source": score_source,
                "subreddit": sub,
            }
        )

    pred_df = pd.DataFrame(rows)
    result = pd.merge(labels, pred_df, on="item_id", how="left")
    result["predicted"] = result["predicted"].fillna(1).astype(int)
    return result


# 11. METRICS


def evaluate_fold(decisions: pd.DataFrame) -> Dict:
    """Evaluate fold and return its metrics."""
    y_true = decisions["label"].values
    y_pred = decisions["predicted"].values
    try:
        system_auc = float(
            roc_auc_score(
                (y_true == 1).astype(int),
                decisions["meta_proba"].values
                if "meta_proba" in decisions.columns
                else (y_pred == 1).astype(int),
            )
        )
    except ValueError:
        system_auc = float("nan")

    covered = ~decisions.get("used_fallback", pd.Series(False, index=decisions.index)).astype(bool)
    covered_decisions = decisions.loc[covered]
    if len(covered_decisions) and covered_decisions["label"].nunique() >= 2:
        expert_covered_auc = float(
            roc_auc_score(
                (covered_decisions["label"].to_numpy() == 1).astype(int),
                covered_decisions["meta_proba"].to_numpy(),
            )
        )
    else:
        expert_covered_auc = float("nan")
    is_hybrid = bool(
        "score_source" in decisions and (decisions["score_source"] == "net_vote_fallback").any()
    )
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # Backwards-compatible key: this is the final system score, which is
        # hybrid whenever full-split net-vote fallback is active.
        "roc_auc": system_auc,
        "roc_auc_system": system_auc,
        "roc_auc_expert_covered": expert_covered_auc,
        "roc_auc_system_is_hybrid": is_hybrid,
        "n_auc_expert_covered": int(covered.sum()),
        "n_auc_system": int(len(decisions)),
        "auc_population": (
            "hybrid_meta_plus_net_vote_all_test_items"
            if is_hybrid
            else "meta_model_all_test_items"
        ),
        "auc_expert_covered_population": "items_with_real_expert_signal",
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1, average="binary", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_pos": precision_score(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        ),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        "precision_neg": precision_score(
            y_true, y_pred, pos_label=-1, average="binary", zero_division=0
        ),
        "recall_neg": recall_score(
            y_true, y_pred, pos_label=-1, average="binary", zero_division=0
        ),
    }
