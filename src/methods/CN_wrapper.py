from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.splits import discover_splits, load_split_data
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold
import sys
from pathlib import Path
_SCORING_ROOT = Path(__file__).parent.parent.parent / "external/community-notes/scoring/src"
if not _SCORING_ROOT.exists():
    raise ImportError(
        f"Cannot find the 'scoring' package at {_SCORING_ROOT}. "
        "Expected the Community Notes dependency under external/community-notes/."
    )
if str(_SCORING_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCORING_ROOT))
from scoring import constants as c
from scoring.matrix_factorization.matrix_factorization import MatrixFactorization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# hyper-parameters
THRESHOLD_GRID          = np.linspace(-2.0, 2.0, 400)
BRIDGING_ABS_THRESHOLD  = 0.5
N_FOLDS                 = 5

# CN MatrixFactorization defaults (from mf_base_scorer.py)
MF_INIT_LR              = 0.2
MF_NO_INIT_LR           = 1.0
MF_CONVERGENCE          = 1e-7
MF_USER_FACTOR_LAMBDA   = 0.03
MF_NOTE_FACTOR_LAMBDA   = 0.03
MF_USER_INTERCEPT_LAMBDA= 0.03 * 5
MF_NOTE_INTERCEPT_LAMBDA= 0.03 * 5
MF_GLOBAL_INTERCEPT_LAMBDA = 0.03 * 5

# minimum calibration sample size when falling back from an empty val split
MIN_CAL_FALLBACK_ITEMS = 50


# 1. load data
def load_all_data(dataset_dir: Path) -> pd.DataFrame:
    """Load all data from its configured source."""
    splits_dir = dataset_dir / "random"
    train = pd.read_parquet(splits_dir / "train_votes.parquet")
    val   = pd.read_parquet(splits_dir / "val_votes.parquet")
    test  = pd.read_parquet(splits_dir / "test_votes.parquet")
    all_data = pd.concat([train, val, test], ignore_index=True)
    print(f"Loaded all: {len(all_data)} votes | {all_data['item_id'].nunique()} items")
    return all_data


# 2. mapping between string item_ids (e.g. 't3_dxlv9b') and contiguous int64 indices required by CN's noteIdKey
def build_item_id_map(
    *dfs: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build item id map from the supplied data."""
    all_ids = pd.concat([df["item_id"] for df in dfs]).unique()
    str2int = {sid: i for i, sid in enumerate(all_ids)}
    int2str = {i: sid for sid, i in str2int.items()}
    return str2int, int2str

def to_cn_ratings(
    df: pd.DataFrame,
    str2int: Dict[str, int],
    vote_sign: int = 1,
) -> pd.DataFrame:
    """Convert cn ratings to the required schema."""
    note_ids = df["item_id"].map(str2int)
    n_unknown = note_ids.isna().sum()
    if n_unknown:
        log.warning("to_cn_ratings: dropped %d rows with unknown item_id", n_unknown)
    ratings = pd.DataFrame(
        {
            c.noteIdKey:             note_ids.astype("Int64"),
            c.raterParticipantIdKey: df["username"].astype(str),
            c.helpfulNumKey:         (df["vote"].astype(float) * vote_sign).astype(np.float32),
        }
    )
    ratings = ratings.dropna(subset=[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey])
    ratings[c.noteIdKey] = ratings[c.noteIdKey].astype(np.int64)
    return ratings


# 3. item vote features
def compute_item_vote_features(
    all_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute item vote features from the supplied data."""
    agg = all_df.groupby("item_id").agg(
        n_voters    =("username", "nunique"),
        vote_mean   =("vote",     "mean"),
        vote_std    =("vote",     "std"),
        pct_positive=("vote",    lambda x: (x == 1).mean()),
    ).reset_index()
    p = agg["pct_positive"].clip(1e-9, 1 - 1e-9)
    agg["vote_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    agg["vote_std"]     = agg["vote_std"].fillna(0.0)
    return agg

# 4. run CN algo
def run_cn_mf(
    all_df:    pd.DataFrame,
    vote_sign: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the cn mf workflow."""
    str2int, int2str = build_item_id_map(all_df)
    ratings = to_cn_ratings(all_df, str2int, vote_sign)
    print(f"CN MF | {len(ratings)} ratings | {ratings[c.noteIdKey].nunique()} unique notes | {ratings[c.raterParticipantIdKey].nunique()} unique raters")
    mf = MatrixFactorization(
        initLearningRate         = MF_INIT_LR,
        noInitLearningRate       = MF_NO_INIT_LR,
        convergence              = MF_CONVERGENCE,
        numFactors               = 1,
        useGlobalIntercept       = True,
        log                      = True,
        userFactorLambda         = MF_USER_FACTOR_LAMBDA,
        noteFactorLambda         = MF_NOTE_FACTOR_LAMBDA,
        userInterceptLambda      = MF_USER_INTERCEPT_LAMBDA,
        noteInterceptLambda      = MF_NOTE_INTERCEPT_LAMBDA,
        globalInterceptLambda    = MF_GLOBAL_INTERCEPT_LAMBDA,
        featureCols              = [c.noteIdKey, c.raterParticipantIdKey],
        labelCol                 = c.helpfulNumKey,
    )

    note_params, rater_params, global_intercept = mf.run_mf(ratings)
    note_params["item_id"] = note_params[c.noteIdKey].map(int2str)
    note_params = note_params.drop(columns=[c.noteIdKey])
    return note_params, rater_params, global_intercept


# 5. threshold calibration (grid search) for binary prediction on modlog dataset
def calibrate_threshold(
    note_params: pd.DataFrame,
    val_df:      pd.DataFrame,
    grid:        np.ndarray = THRESHOLD_GRID,
) -> Tuple[float, pd.DataFrame]:
    """Calibrate threshold on validation data."""
    items_val = val_df.drop_duplicates("item_id")[["item_id", "label"]].copy()
    lookup = note_params.set_index("item_id")
    items_val["i_n"] = items_val["item_id"].map(lookup[c.internalNoteInterceptKey]).fillna(0.0)

    y_true = items_val["label"].values
    scores = items_val["i_n"].values

    records = []
    for thr in grid:
        y_pred = np.where(scores >= thr, 1, -1)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
        )
        records.append({"threshold": thr, "precision": p, "recall": r, "f1": f})

    cal_df   = pd.DataFrame(records)
    best     = cal_df.loc[cal_df["f1"].idxmax()]
    best_thr = float(best["threshold"])
    print(f"Best threshold {best_thr:.4f} | val F1 {best['f1']:.4f} (P {best['precision']:.4f}, R {best['recall']:.4f})")
    return best_thr, cal_df


# 6. evaluation
def evaluate(
    note_params:      pd.DataFrame,
    test_df:          pd.DataFrame,
    threshold:        float,
    item_vote_feats:  Optional[pd.DataFrame] = None,
    fallback_scores:  Optional[Dict[str, float]] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Calculate predictions and metrics for one evaluation set."""
    items_test = test_df.drop_duplicates("item_id")[["item_id", "label"]].copy()
    lookup = note_params.set_index("item_id")
    raw_i_n = items_test["item_id"].map(lookup[c.internalNoteInterceptKey])
    n_missing_i_n = int(raw_i_n.isna().sum())
    if fallback_scores:
        fallback_series = items_test["item_id"].map(fallback_scores)
        items_test["i_n"] = raw_i_n.fillna(fallback_series).fillna(0.0)
    else:
        items_test["i_n"] = raw_i_n.fillna(0.0)
    items_test["f_n"] = items_test["item_id"].map(lookup[c.internalNoteFactor1Key]).fillna(0.0)
    items_test["abs_f_n"]        = np.abs(items_test["f_n"])
    items_test["bridging_score"] = -items_test["abs_f_n"]
    items_test["is_bridging"]    = items_test["abs_f_n"] < BRIDGING_ABS_THRESHOLD
    items_test["prediction"]     = np.where(items_test["i_n"] >= threshold, 1, -1)

    if item_vote_feats is not None:
        items_test = items_test.merge(item_vote_feats, on="item_id", how="left")

    y_true = items_test["label"].values
    y_pred = items_test["prediction"].values
    scores = items_test["i_n"].values
    mean_in_pos = float(items_test.loc[items_test["label"] ==  1, "i_n"].mean())
    mean_in_neg = float(items_test.loc[items_test["label"] == -1, "i_n"].mean())
    polarity_correct = mean_in_pos > mean_in_neg
    if polarity_correct:
        print(f"mean i_n: label=+1 {mean_in_pos:.4f} > label=-1 {mean_in_neg:.4f}")
    else:
        log.warning(
            "Polarity Flipped | mean i_n: label=+1 %.4f < label=-1 %.4f "
            "Re-run with vote_sign=-1 to test polarity hypothesis.",
            mean_in_pos, mean_in_neg,
        )
    p,   r,   f,   _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average="macro",  zero_division=0)
    p1,  r1,  f1,  _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1],    average="binary",  zero_division=0)
    p_1, r_1, f_1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1],   average="binary",  pos_label=-1, zero_division=0)
    try:
        auc = float(roc_auc_score((y_true == 1).astype(int), scores))
    except ValueError:
        auc = float("nan")

    prec_curve, rec_curve, thr_curve = precision_recall_curve(
        (y_true == 1).astype(int), scores
    )
    bridging_mask     = items_test["is_bridging"]
    non_bridging_mask = ~bridging_mask
    bridging_correct = float(
        (items_test.loc[bridging_mask, "prediction"] ==
         items_test.loc[bridging_mask, "label"]).mean()
    ) if bridging_mask.any() else float("nan")
    non_bridging_correct = float(
        (items_test.loc[non_bridging_mask, "prediction"] ==
         items_test.loc[non_bridging_mask, "label"]).mean()
    ) if non_bridging_mask.any() else float("nan")
    bridging_label_dist = (
        items_test.loc[bridging_mask, "label"].value_counts().to_dict()
        if bridging_mask.any() else {}
    )
    corr_fn_entropy = float("nan")
    corr_fn_voters  = float("nan")
    if item_vote_feats is not None and "vote_entropy" in items_test.columns:
        valid_rows = items_test[["abs_f_n", "vote_entropy", "n_voters"]].dropna()
        if len(valid_rows) > 10:
            corr_fn_entropy = float(
                np.corrcoef(valid_rows["abs_f_n"], valid_rows["vote_entropy"])[0, 1]
            )
            corr_fn_voters = float(
                np.corrcoef(valid_rows["abs_f_n"], valid_rows["n_voters"])[0, 1]
            )
            print(f"Bridging correlation | |f_n| vs vote_entropy: {corr_fn_entropy:.4f} | |f_n| vs n_voters: {corr_fn_voters:.4f}")
    bridging_entropy_mean     = float("nan")
    non_bridging_entropy_mean = float("nan")
    if "vote_entropy" in items_test.columns:
        bridging_entropy_mean = float(
            items_test.loc[bridging_mask, "vote_entropy"].mean()
        ) if bridging_mask.any() else float("nan")
        non_bridging_entropy_mean = float(
            items_test.loc[non_bridging_mask, "vote_entropy"].mean()
        ) if non_bridging_mask.any() else float("nan")
        print(f"Vote entropy | bridging: {bridging_entropy_mean:.4f} | non-bridging: {non_bridging_entropy_mean:.4f}")
    print(f"Test | macro F1 {f:.4f} | AUC {auc:.4f} | bridging {100 * items_test['is_bridging'].mean():.1f}%")
    metrics = {
        "threshold":                  threshold,
        "macro_precision":            float(p),
        "macro_recall":               float(r),
        "macro_f1":                   float(f),
        "precision_pos":              float(p1),
        "recall_pos":                 float(r1),
        "f1_pos":                     float(f1),
        "precision_neg":              float(p_1),
        "recall_neg":                 float(r_1),
        "f1_neg":                     float(f_1),
        "roc_auc":                    auc,
        "polarity_correct":           polarity_correct,
        "n_items_test":               int(len(items_test)),
        "n_items_missing_i_n":        n_missing_i_n,
        "n_bridging_items":           int(bridging_mask.sum()),
        "bridging_pct":               float(bridging_mask.mean()),
        "bridging_accuracy":          bridging_correct,
        "non_bridging_accuracy":      non_bridging_correct,
        "bridging_label_dist":        {str(k): int(v) for k, v in bridging_label_dist.items()},
        "bridging_vote_entropy_mean": bridging_entropy_mean,
        "non_bridging_vote_entropy":  non_bridging_entropy_mean,
        "corr_abs_fn_vote_entropy":   corr_fn_entropy,
        "corr_abs_fn_n_voters":       corr_fn_voters,
        "mean_i_n_approve":           mean_in_pos,
        "mean_i_n_remove":            mean_in_neg,
        "mean_abs_f_n":               float(items_test["abs_f_n"].mean()),
        "pr_curve_precision":         prec_curve.tolist(),
        "pr_curve_recall":            rec_curve.tolist(),
        "pr_curve_thresholds":        thr_curve.tolist(),
    }
    return metrics, items_test

# 7. user polarisation analysis
def analyze_user_polarization(
    rater_params: pd.DataFrame,
    step1_dir:    Path,
) -> pd.DataFrame:
    """Analyze user polarization and return summary statistics."""
    users_path = step1_dir / "users.parquet"
    if not users_path.exists():
        log.warning("users.parquet not found - skipping polarisation analysis")
        return pd.DataFrame()
    users  = pd.read_parquet(users_path)
    merged = rater_params.rename(
        columns={
            c.raterParticipantIdKey:  "username",
            c.internalRaterInterceptKey: "i_u",
            c.internalRaterFactor1Key:   "f_u",
        }
    ).merge(users, on="username", how="left")

    if "delete_rate" in merged.columns:
        valid = merged[["f_u", "delete_rate"]].dropna()
        if len(valid) > 10:
            corr = float(np.corrcoef(valid["f_u"], valid["delete_rate"])[0, 1])
            print(f"User polarisation | corr(f_u, delete_rate): {corr:.4f}")
    return merged

# 8. build user_params table
def build_user_params(
    rater_params: pd.DataFrame,
    votes_df:     Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build user params from the supplied data."""
    out = rater_params.rename(
        columns={
            c.raterParticipantIdKey:     "username",
            c.internalRaterInterceptKey: "i_u",
            c.internalRaterFactor1Key:   "f_u",
        }
    )[["username", "i_u", "f_u"]]

    if votes_df is not None:
        n_votes = votes_df.groupby("username").size().rename("n_votes")
        out = out.merge(n_votes, on="username", how="left")
    else:
        out["n_votes"] = np.nan

    return out

# 9. save model to .npz
def save_model_npz(
    note_params:      pd.DataFrame,
    rater_params:     pd.DataFrame,
    global_intercept: Optional[float],
    path:             Path,
) -> None:
    """Write model npz to disk."""
    np.savez(
        path,
        mu  = np.array([global_intercept or 0.0]),
        i_u = rater_params[c.internalRaterInterceptKey].values,
        f_u = rater_params[c.internalRaterFactor1Key].values,
        i_n = note_params[c.internalNoteInterceptKey].values,
        f_n = note_params[c.internalNoteFactor1Key].values,
    )
    print(f"Model params saved to {path}")


# 10. save all outputs
def save_outputs(
    output_dir:       Path,
    note_params:      pd.DataFrame,
    rater_params:     pd.DataFrame,
    global_intercept: Optional[float],
    fold_item_scores: List[pd.DataFrame],
    user_params:      pd.DataFrame,
    fold_metrics:     List[Dict],
    fold_cal_dfs:     List[pd.DataFrame],
    user_analysis:    Optional[pd.DataFrame] = None,
) -> None:
    """Write outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_item_scores = pd.concat(
        [df.assign(fold=i) for i, df in enumerate(fold_item_scores)], ignore_index=True
    )
    all_cal_dfs = pd.concat(
        [df.assign(fold=i) for i, df in enumerate(fold_cal_dfs)], ignore_index=True
    )
    all_item_scores.to_parquet(output_dir / "item_scores.parquet",     index=False)
    user_params.to_parquet(    output_dir / "user_params.parquet",     index=False)
    all_cal_dfs.to_parquet(    output_dir / "val_calibration.parquet", index=False)
    save_model_npz(note_params, rater_params, global_intercept, output_dir / "model_params.npz")

    # scalar keys: everything that is int/float/bool (excludes pr_curve lists)
    scalar_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float, bool))]
    agg_metrics: Dict = {"n_folds": N_FOLDS}
    for k in scalar_keys:
        vals = [m[k] for m in fold_metrics if isinstance(m[k], (int, float))]
        agg_metrics[k]           = float(np.mean(vals))
        agg_metrics[f"{k}_std"]  = float(np.std(vals))
        agg_metrics[f"{k}_per_fold"] = vals

    # pr curves per fold stored as list-of-dicts for CI plotting
    agg_metrics["pr_curves_per_fold"] = [
        {
            "precision":  m["pr_curve_precision"],
            "recall":     m["pr_curve_recall"],
            "thresholds": m["pr_curve_thresholds"],
        }
        for m in fold_metrics
    ]

    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(agg_metrics, fh, indent=2)

    # fold_details.parquet: one row per fold, scalar metrics only (no lists)
    fold_rows = [
        {k: v for k, v in m.items() if isinstance(v, (int, float, bool))}
        | {"fold": i}
        for i, m in enumerate(fold_metrics)
    ]
    pd.DataFrame(fold_rows).to_parquet(output_dir / "fold_details.parquet", index=False)

    if user_analysis is not None and not user_analysis.empty:
        user_analysis.to_parquet(output_dir / "user_analysis.parquet", index=False)
        print(f"User analysis saved to {output_dir / 'user_analysis.parquet'}")
    print(f"Step 2 (CN evaluation) outputs saved to {output_dir}")


# main
def evaluate_cn(
    dataset_dir: Path,
    output_dir: Path,
    vote_sign:  int = 1,
) -> Dict:
    """Evaluate cn and return its metrics."""
    log.info(
        "STEP 2: Community Notes(%s) vote_sign=%+d",
        dataset_dir.name, vote_sign,
    )
    all_df = load_all_data(dataset_dir)
    item_vote_feats = compute_item_vote_features(all_df)

    note_params, rater_params, global_intercept = run_cn_mf(all_df, vote_sign)

    print(f"Converged | mean|f_n| {note_params[c.internalNoteFactor1Key].abs().mean():.4f} | mean|f_u| {rater_params[c.internalRaterFactor1Key].abs().mean():.4f}")

    labeled_items = (
        all_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"])
        .reset_index(drop=True)
    )
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=10)

    fold_metrics:     List[Dict]         = []
    fold_item_scores: List[pd.DataFrame] = []
    fold_cal_dfs:     List[pd.DataFrame] = []

    for fold_idx, (val_idx, test_idx) in enumerate(kf.split(labeled_items)):
        log.info("Fold %d/%d", fold_idx + 1, N_FOLDS)
        val_ids  = labeled_items.iloc[val_idx]["item_id"].values
        test_ids = labeled_items.iloc[test_idx]["item_id"].values
        val_df_fold  = all_df[all_df["item_id"].isin(val_ids)]
        test_df_fold = all_df[all_df["item_id"].isin(test_ids)]
        threshold, cal_df    = calibrate_threshold(note_params, val_df_fold)
        metrics, item_scores = evaluate(note_params, test_df_fold, threshold, item_vote_feats)
        fold_metrics.append(metrics)
        fold_item_scores.append(item_scores)
        fold_cal_dfs.append(cal_df)

    user_params   = build_user_params(rater_params, all_df)
    user_analysis = analyze_user_polarization(rater_params, dataset_dir)

    save_outputs(
        output_dir,
        note_params, rater_params, global_intercept,
        fold_item_scores, user_params, fold_metrics, fold_cal_dfs, user_analysis,
    )

    log.info("CN evaluation complete: %s", dataset_dir.name)
    scalar_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float, bool))]
    agg_metrics = {k: float(np.mean([m[k] for m in fold_metrics if isinstance(m[k], (int, float))])) for k in scalar_keys}
    return {
        "note_params":  note_params,
        "rater_params": rater_params,
        "metrics":      agg_metrics,
        "item_scores":  pd.concat(fold_item_scores, ignore_index=True),
    }


# MULTI-SPLIT BENCHMARK  (uses splits produced by prepare_data_step1)
#
# CN's matrix factorization is transductive: it factorizes the (item, user,
# vote) matrix and never sees ground-truth labels during fitting. So, unlike
# TFR/SEF where "train only" matters to avoid leaking labels into a
# supervised model, here the only thing that must come from the right split
# is which votes feed the MF (kept to this split's own item population, so
# no vote from another split/window leaks in) and which labels are used for
# threshold calibration (VAL) vs evaluation (TEST).

def run_single_split_cn(
    split_label: str,
    all_df:      pd.DataFrame,
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    vote_sign:   int = 1,
) -> Tuple[Optional[Dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float]]:
    """Run the single split cn workflow."""
    if test_df.empty or test_df["item_id"].nunique() == 0:
        log.warning("  [%s] empty test set - skipped.", split_label)
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    item_vote_feats = compute_item_vote_features(all_df)
    note_params, rater_params, global_intercept = run_cn_mf(all_df, vote_sign)
    log.info(
        "  [%s] Converged | mean|f_n|=%.4f | mean|f_u|=%.4f",
        split_label,
        note_params[c.internalNoteFactor1Key].abs().mean(),
        rater_params[c.internalRaterFactor1Key].abs().mean(),
    )

    if val_df.empty or val_df["item_id"].nunique() == 0:
        log.warning(
            "  [%s] empty val split - falling back to a calibration sample "
            "carved from TRAIN labels (MF itself never used those labels).",
            split_label,
        )
        train_items = train_df.drop_duplicates("item_id")["item_id"].values
        if len(train_items) == 0:
            log.warning("  [%s] no train items either - skipped.", split_label)
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
        rng = np.random.RandomState(10)
        sample_size = min(len(train_items), max(MIN_CAL_FALLBACK_ITEMS, int(0.2 * len(train_items))))
        sample_ids  = rng.choice(train_items, size=sample_size, replace=False)
        cal_source_df = train_df[train_df["item_id"].isin(sample_ids)]
    else:
        cal_source_df = val_df

    threshold, cal_df    = calibrate_threshold(note_params, cal_source_df)

    # "splits_full" only: force full test coverage using this item's net
    # vote (mean of +1/-1 votes in this split's test set) for any item the
    # MF produced no i_n for, instead of leaving it at a neutral 0.0.
    fallback_scores = None
    if split_label == "splits_full":
        fallback_scores = test_df.groupby("item_id")["vote"].mean().to_dict()

    metrics, item_scores = evaluate(note_params, test_df, threshold, item_vote_feats, fallback_scores)
    if split_label == "splits_full" and metrics.get("n_items_missing_i_n", 0) > 0:
        metrics["used_net_vote_fallback"] = True
        metrics["n_fallback_items"]       = metrics["n_items_missing_i_n"]

    metrics["split"]         = split_label
    metrics["n_train_items"] = int(train_df["item_id"].nunique())
    metrics["n_val_items"]   = int(val_df["item_id"].nunique())

    log.info(
        "  [%s] macro_f1=%.4f  roc_auc=%.4f  f1_pos=%.4f  f1_neg=%.4f  bridging=%.1f%%",
        split_label, metrics["macro_f1"], metrics["roc_auc"],
        metrics["f1_pos"], metrics["f1_neg"], 100 * metrics["bridging_pct"],
    )

    return metrics, item_scores, cal_df, note_params, rater_params, global_intercept


def evaluate_splits(
    votes_dir:  Path,
    output_dir: Path,
    vote_sign:  int = 1,
    method_name: str = "cn",
) -> Dict[str, Dict]:
    """Evaluate splits and return its metrics."""
    splits = discover_splits(votes_dir)
    if not splits:
        log.warning(
            "No split directories found under %s (expected splits/, "
            "splits_full/, splits_intersection/, windowed_folds_full/w*, "
            "windowed_folds_intersection/w*)", votes_dir,
        )
        return {}

    log.info("Found %d split(s): %s", len(splits), list(splits.keys()))
    summary: Dict[str, Dict] = {}

    for split_name, split_path in splits.items():
        log.info("--- split: %s ---", split_name)

        all_df, train_df, val_df, test_df = load_split_data(split_path)

        metrics, item_scores, cal_df, note_params, rater_params, global_intercept = \
            run_single_split_cn(split_name, all_df, train_df, val_df, test_df, vote_sign)

        if metrics is None:
            continue

        user_params   = build_user_params(rater_params, all_df)
        user_analysis = analyze_user_polarization(rater_params, votes_dir)

        split_out_dir = output_dir / split_name / method_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        item_scores.to_parquet(split_out_dir / "item_scores.parquet", index=False)
        user_params.to_parquet(split_out_dir / "user_params.parquet", index=False)
        cal_df.to_parquet(split_out_dir / "val_calibration.parquet", index=False)
        save_model_npz(note_params, rater_params, global_intercept, split_out_dir / "model_params.npz")
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        if user_analysis is not None and not user_analysis.empty:
            user_analysis.to_parquet(split_out_dir / "user_analysis.parquet", index=False)
        log.info("  [%s] Saved -> %s", split_name, split_out_dir)

        summary[split_name] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)

    # strip the large PR-curve arrays for the cross-split summary file
    summary_slim = {
        name: {k: v for k, v in m.items()
               if k not in ("pr_curve_precision", "pr_curve_recall", "pr_curve_thresholds")}
        for name, m in summary.items()
    }
    summary_path = output_dir / "all_splits_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary_slim, fh, indent=2)
    log.info("Summary of all splits saved -> %s", summary_path)

    if summary_slim:
        log.info("=== Cross-split comparison ===")
        for name, m in summary_slim.items():
            log.info("  %-35s macro_F1=%.4f  AUC=%.4f  n_test=%d  bridging=%.1f%%",
                     name, m["macro_f1"], m["roc_auc"], m["n_items_test"],
                     100 * m["bridging_pct"])

    return summary


if __name__ == "__main__":
    evaluate_cn(
        dataset_dir = Path("data/splits/reddit"),
        output_dir  = Path("results/reddit/random/cn"),
    )
    evaluate_splits(
        votes_dir  = Path("data/splits/reddit"),
        output_dir = Path("results/reddit"),
    )
    evaluate_cn(
        dataset_dir = Path("data/splits/reddit"),
        output_dir  = Path("results/reddit/random/cn-inverted"),
        vote_sign  = -1,
    )
    evaluate_splits(
        votes_dir  = Path("data/splits/reddit"),
        output_dir = Path("results/reddit"),
        vote_sign  = -1,
        method_name = "cn-inverted",
    )
    evaluate_cn(
        dataset_dir=Path("data/splits/wikipedia/standard"),
        output_dir=Path("results/wikipedia/standard/random/cn"),
    )
    evaluate_splits(
        votes_dir  = Path("data/splits/wikipedia/standard"),
        output_dir = Path("results/wikipedia/standard"),
    )
