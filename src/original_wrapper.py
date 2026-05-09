from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold
import sys
from pathlib import Path
_SCORING_ROOT = Path(__file__).parent.parent / "communitynotes_main/scoring/src"
if not _SCORING_ROOT.exists():
    raise ImportError(
        f"Cannot find the 'scoring' package at {_SCORING_ROOT}. "
        "Place the unzipped twitter/communitynotes scoring/ folder next to this file."
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


# 1. load data
def load_all_data(step1_dir: Path) -> pd.DataFrame:
    splits_dir = step1_dir / "splits"
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
    all_ids = pd.concat([df["item_id"] for df in dfs]).unique()
    str2int = {sid: i for i, sid in enumerate(all_ids)}
    int2str = {i: sid for sid, i in str2int.items()}
    return str2int, int2str

def to_cn_ratings(
    df: pd.DataFrame,
    str2int: Dict[str, int],
    vote_sign: int = 1,
) -> pd.DataFrame:
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
) -> Tuple[Dict, pd.DataFrame]:
    items_test = test_df.drop_duplicates("item_id")[["item_id", "label"]].copy()
    lookup = note_params.set_index("item_id")
    items_test["i_n"] = items_test["item_id"].map(lookup[c.internalNoteInterceptKey]).fillna(0.0)
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
    users_path = step1_dir / "users.parquet"
    if not users_path.exists():
        log.warning("users.parquet not found — skipping polarisation analysis")
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
def build_user_params(rater_params: pd.DataFrame) -> pd.DataFrame:
    return rater_params.rename(
        columns={
            c.raterParticipantIdKey:     "username",
            c.internalRaterInterceptKey: "i_u",
            c.internalRaterFactor1Key:   "f_u",
        }
    )[["username", "i_u", "f_u"]]

# 9. save model to .npz
def save_model_npz(
    note_params:      pd.DataFrame,
    rater_params:     pd.DataFrame,
    global_intercept: Optional[float],
    path:             Path,
) -> None:
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
def run_step2(
    step1_dir:  Path,
    output_dir: Path,
    vote_sign:  int = 1,
) -> Dict:
    log.info(
        "STEP 2: Community Notes(%s) vote_sign=%+d",
        step1_dir.name, vote_sign,
    )
    all_df = load_all_data(step1_dir)
    item_vote_feats = compute_item_vote_features(all_df)

    note_params, rater_params, global_intercept = run_cn_mf(all_df, vote_sign)

    print(f"Converged | mean|f_n| {note_params[c.internalNoteFactor1Key].abs().mean():.4f} | mean|f_u| {rater_params[c.internalRaterFactor1Key].abs().mean():.4f}")

    labeled_items = (
        all_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"])
        .reset_index(drop=True)
    )
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    user_params   = build_user_params(rater_params)
    user_analysis = analyze_user_polarization(rater_params, step1_dir)

    save_outputs(
        output_dir,
        note_params, rater_params, global_intercept,
        fold_item_scores, user_params, fold_metrics, fold_cal_dfs, user_analysis,
    )

    log.info("Step 2 complete: %s", step1_dir.name)
    scalar_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float, bool))]
    agg_metrics = {k: float(np.mean([m[k] for m in fold_metrics if isinstance(m[k], (int, float))])) for k in scalar_keys}
    return {
        "note_params":  note_params,
        "rater_params": rater_params,
        "metrics":      agg_metrics,
        "item_scores":  pd.concat(fold_item_scores, ignore_index=True),
    }

if __name__ == "__main__":
    run_step2(
        step1_dir  = Path("results/step1/reddit"),
        output_dir = Path("results/step2/reddit_cn"),
    )
    run_step2(
        step1_dir  = Path("results/step1/reddit"),
        output_dir = Path("results/step2/reddit_cn_inverted"),
        vote_sign  = -1,
    )
    run_step2(
        step1_dir=Path("results/step1/wikipedia"),
        output_dir=Path("results/step2/wikipedia"),
    )