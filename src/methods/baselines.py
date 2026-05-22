from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

THRESHOLD_GRID    = np.linspace(-200, 200, 800)
ALPHA_GRID        = np.linspace(1.0, 5.0, 20)
MIN_SUB_VAL_ITEMS = 10
N_FOLDS           = 5
REDDIT_SCORES_PATH = Path("results/step1/reddit/moderated_posts_scores.parquet")


# 1. load full vote dataset (all splits merged)
def load_all_votes(step1_dir: Path) -> pd.DataFrame:
    splits_dir = step1_dir / "splits"
    dfs = []
    for name in ("train_votes", "val_votes", "test_votes"):
        df = pd.read_parquet(splits_dir / f"{name}.parquet")
        dfs.append(df)
    all_votes = pd.concat(dfs, ignore_index=True)
    log.info(
        "Loaded all votes: %d votes | %d items | %d users",
        len(all_votes), all_votes["item_id"].nunique(), all_votes["username"].nunique(),
    )
    return all_votes


# 2. compute score(n) =  n_up - α * n_down. With alpha=1 this is the plain net score (upvotes - downvotes)
def compute_net_scores(
    votes_df: pd.DataFrame,
    alpha:    float = 1.0,
) -> pd.DataFrame:
    agg = votes_df.groupby("item_id").agg(
        community = ("community", "first"),
        n_up      = ("vote", lambda x: (x == 1).sum()),
        n_down    = ("vote", lambda x: (x == -1).sum()),
    ).reset_index()
    agg["n_votes"] = agg["n_up"] + agg["n_down"]
    agg["score"]   = agg["n_up"] - alpha * agg["n_down"]
    return agg


# 3. label lookup
def get_item_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("item_id")[["item_id", "label"]].copy()


# 4. chronological k-fold split at item level
def make_chrono_folds(all_votes: pd.DataFrame, n_folds: int = N_FOLDS) -> List[pd.DataFrame]:
    """
    Sort items by their earliest vote timestamp, divide into n_folds equal
    buckets, return a list of n_folds DataFrames each containing the votes
    for the items in that bucket.
    Chronological order is preserved so earlier items are always in earlier folds.
    """
    item_times = (
        all_votes.groupby("item_id")["timestamp"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"timestamp": "first_vote"})
    )
    n_items = len(item_times)
    fold_size = n_items // n_folds
    folds: List[pd.DataFrame] = []
    for k in range(n_folds):
        start = k * fold_size
        end   = (k + 1) * fold_size if k < n_folds - 1 else n_items
        fold_items = item_times.iloc[start:end]["item_id"].values
        fold_votes = all_votes[all_votes["item_id"].isin(fold_items)].copy()
        folds.append(fold_votes)
        log.info(
            "Fold %d: %d items | %d votes | %s → %s",
            k, len(fold_items), len(fold_votes),
            fold_votes["timestamp"].min().date() if "timestamp" in fold_votes.columns else "?",
            fold_votes["timestamp"].max().date() if "timestamp" in fold_votes.columns else "?",
        )
    return folds


# 5. grid-search over thresholds; maximise macro F1
def calibrate_global_threshold(
    scores_df:  pd.DataFrame,
    val_labels: pd.DataFrame,
    grid:       np.ndarray = THRESHOLD_GRID,
) -> Tuple[float, pd.DataFrame]:
    merged = val_labels.merge(scores_df[["item_id", "score"]], on="item_id", how="left")
    merged["score"] = merged["score"].fillna(0.0)
    y_true = merged["label"].values
    scores = merged["score"].values
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
    return best_thr, cal_df


# 6. alpha selection using inner train→val hold-out
def select_alpha(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    alpha_grid: np.ndarray = ALPHA_GRID,
    thr_grid:   np.ndarray = THRESHOLD_GRID,
) -> float:
    """
    For each alpha, calibrate threshold on train labels, evaluate on val.
    Returns the alpha that maximises val macro F1.
    """
    train_labels = get_item_labels(train_df)
    val_labels   = get_item_labels(val_df)
    best_alpha_f1, best_alpha = -1.0, 1.0

    for alpha in alpha_grid:
        train_scores = compute_net_scores(train_df, alpha=alpha)
        merged_tr = train_labels.merge(train_scores[["item_id", "score"]], on="item_id", how="left")
        merged_tr["score"] = merged_tr["score"].fillna(0.0)
        y_tr, s_tr = merged_tr["label"].values, merged_tr["score"].values

        best_thr_tr, best_f1_tr = 0.0, -1.0
        for thr in thr_grid:
            y_pred = np.where(s_tr >= thr, 1, -1)
            _, _, f, _ = precision_recall_fscore_support(
                y_tr, y_pred, labels=[1, -1], average="macro", zero_division=0
            )
            if f > best_f1_tr:
                best_f1_tr, best_thr_tr = f, thr

        val_scores = compute_net_scores(pd.concat([train_df, val_df], ignore_index=True), alpha=alpha)
        merged_val = val_labels.merge(val_scores[["item_id", "score"]], on="item_id", how="left")
        merged_val["score"] = merged_val["score"].fillna(0.0)
        y_pred_val = np.where(merged_val["score"].values >= best_thr_tr, 1, -1)
        _, _, f_val, _ = precision_recall_fscore_support(
            merged_val["label"].values, y_pred_val, labels=[1, -1], average="macro", zero_division=0
        )
        if f_val > best_alpha_f1:
            best_alpha_f1, best_alpha = f_val, alpha

    return best_alpha


# 7. per-subreddit threshold calibration
def calibrate_per_subreddit(
    scores_df:        pd.DataFrame,
    val_labels_full:  pd.DataFrame,
    global_threshold: float,
    grid:             np.ndarray = THRESHOLD_GRID,
    min_items:        int        = MIN_SUB_VAL_ITEMS,
) -> Dict[str, float]:
    merged = val_labels_full.merge(
        scores_df[["item_id", "score"]], on="item_id", how="left"
    )
    merged["score"] = merged["score"].fillna(0.0)

    sub_thresholds: Dict[str, float] = {}
    for sub, group in merged.groupby("community"):
        n_classes = group["label"].nunique()
        class_counts = group["label"].value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
        if len(group) < min_items or n_classes < 2 or min_class_count < max(3, min_items // 3):
            sub_thresholds[sub] = global_threshold
            continue
        y_true = group["label"].values
        scores = group["score"].values
        best_f1, best_thr = -1.0, global_threshold
        for thr in grid:
            y_pred = np.where(scores >= thr, 1, -1)
            _, _, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
            )
            if f > best_f1:
                best_f1, best_thr = f, thr
        sub_thresholds[sub] = best_thr
    return sub_thresholds


# 8. evaluation on one fold
def evaluate_fold(
    scores_df:    pd.DataFrame,
    test_df:      pd.DataFrame,
    predictions:  np.ndarray,
) -> Dict:
    test_labels = get_item_labels(test_df)
    items_test  = test_labels.merge(scores_df, on="item_id", how="left")
    items_test["score"]      = items_test["score"].fillna(0.0)
    items_test["prediction"] = predictions
    y_true = items_test["label"].values
    y_pred = items_test["prediction"].values
    scores = items_test["score"].values

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

    mean_s_pos = float(items_test.loc[items_test["label"] ==  1, "score"].mean())
    mean_s_neg = float(items_test.loc[items_test["label"] == -1, "score"].mean())

    try:
        prec_curve, rec_curve, thr_curve = precision_recall_curve(
            (y_true == 1).astype(int), scores
        )
        pr_precision = prec_curve.tolist()
        pr_recall    = rec_curve.tolist()
        pr_thresholds = thr_curve.tolist()
    except Exception:
        pr_precision = pr_recall = pr_thresholds = []

    return {
        "macro_precision": float(p),  "macro_recall": float(r),   "macro_f1": float(f),
        "precision_pos":   float(p1), "recall_pos":   float(r1),  "f1_pos":   float(f1),
        "precision_neg":   float(p_1),"recall_neg":   float(r_1), "f1_neg":   float(f_1),
        "roc_auc":         auc,
        "polarity_correct": mean_s_pos > mean_s_neg,
        "n_items":         int(len(items_test)),
        "mean_score_approve": mean_s_pos,
        "mean_score_remove":  mean_s_neg,
        "pr_curve_precision":  pr_precision,
        "pr_curve_recall":     pr_recall,
        "pr_curve_thresholds": pr_thresholds,
    }


# 9. aggregate fold metrics into mean ± std
def aggregate_fold_metrics(fold_results: List[Dict], baseline_name: str, alpha: float) -> Dict:
    scalar_keys = [
        "macro_precision", "macro_recall", "macro_f1",
        "precision_pos", "recall_pos", "f1_pos",
        "precision_neg", "recall_neg", "f1_neg",
        "roc_auc",
    ]
    agg = {"baseline": baseline_name, "alpha": alpha, "n_folds": len(fold_results)}
    for key in scalar_keys:
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        agg[key]              = float(np.mean(vals))
        agg[f"{key}_std"]     = float(np.std(vals))
        agg[f"{key}_per_fold"] = vals
    agg["polarity_correct_per_fold"] = [r["polarity_correct"] for r in fold_results]
    agg["pr_curves_per_fold"] = [
        {
            "precision":  r.get("pr_curve_precision",  []),
            "recall":     r.get("pr_curve_recall",     []),
            "thresholds": r.get("pr_curve_thresholds", []),
        }
        for r in fold_results
    ]
    log.info(
        "%s | macro F1 %.4f ± %.4f | AUC %.4f ± %.4f",
        baseline_name,
        agg["macro_f1"], agg["macro_f1_std"],
        agg["roc_auc"],  agg["roc_auc_std"],
    )
    return agg


# 10. save outputs
def save_outputs(
    output_dir:    Path,
    metrics:       Dict,
    fold_details:  List[Dict],
    baseline_name: str,
) -> None:
    d = output_dir / baseline_name
    d.mkdir(parents=True, exist_ok=True)
    # strip pr curves from fold_details before saving parquet (too large for columnar storage)
    fold_rows = [
        {k: v for k, v in row.items() if not k.startswith("pr_curve")}
        for row in fold_details
    ]
    with open(d / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame(fold_rows).to_parquet(d / "fold_details.parquet", index=False)
    log.info("Baseline '%s' outputs → %s", baseline_name, d)


# 11. run one baseline across all folds
def run_baseline_kfold(
    folds:         List[pd.DataFrame],
    all_votes:     pd.DataFrame,
    baseline_name: str,
    use_alpha:     bool = False,
    use_subreddit: bool = False,
) -> Tuple[Dict, List[Dict]]:
    fold_results  = []
    fold_details  = []
    chosen_alphas = []

    for k, val_fold in enumerate(folds):
        # train = all folds except the current one
        train_fold = pd.concat(
            [folds[i] for i in range(len(folds)) if i != k], ignore_index=True
        )
        train_labels      = get_item_labels(train_fold)
        train_labels_full = train_fold.drop_duplicates("item_id")[["item_id", "label", "community"]].copy()
        val_labels_full   = val_fold.drop_duplicates("item_id")[["item_id", "label", "community"]].copy()

        # scores computed on all_votes for consistency (transductive)
        if use_alpha:
            # use inner train/val split within the train fold for alpha selection
            # split train fold itself 80/20 chronologically for inner cv
            item_times_inner = (
                train_fold.groupby("item_id")["timestamp"].min()
                .sort_values().reset_index()
            )
            n_inner   = len(item_times_inner)
            cut       = int(n_inner * 0.8)
            inner_train_items = item_times_inner.iloc[:cut]["item_id"].values
            inner_val_items   = item_times_inner.iloc[cut:]["item_id"].values
            inner_train = train_fold[train_fold["item_id"].isin(inner_train_items)]
            inner_val   = train_fold[train_fold["item_id"].isin(inner_val_items)]
            alpha = select_alpha(inner_train, inner_val)
            log.info("  Fold %d | alpha=%.2f", k, alpha)
        else:
            alpha = 1.0

        chosen_alphas.append(alpha)
        scores_all = compute_net_scores(all_votes, alpha=alpha)
        threshold, _ = calibrate_global_threshold(scores_all, train_labels)

        if use_subreddit:
            sub_thr = calibrate_per_subreddit(scores_all, train_labels_full, threshold)
            test_items = get_item_labels(val_fold).merge(
                val_fold.drop_duplicates("item_id")[["item_id", "community"]], on="item_id"
            ).merge(scores_all[["item_id", "score"]], on="item_id", how="left")
            test_items["score"] = test_items["score"].fillna(0.0)
            thr_per_item = test_items["community"].map(sub_thr).fillna(threshold)
            preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)
        else:
            test_items = get_item_labels(val_fold).merge(
                scores_all[["item_id", "score"]], on="item_id", how="left"
            )
            test_items["score"] = test_items["score"].fillna(0.0)
            preds = np.where(test_items["score"].values >= threshold, 1, -1)

        fold_metric = evaluate_fold(scores_all, val_fold, preds)
        fold_metric["fold"] = k
        fold_metric["alpha"] = alpha
        fold_metric["threshold"] = threshold
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  Fold %d | macro F1 %.4f | AUC %.4f | alpha=%.2f | thr=%.4f",
            k, fold_metric["macro_f1"], fold_metric["roc_auc"], alpha, threshold,
        )

    mean_alpha = float(np.mean(chosen_alphas))
    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=mean_alpha)
    metrics["alpha_per_fold"] = chosen_alphas
    return metrics, fold_details


# 12. load pre-fetched Reddit scores from Arctic Shift (item_id + score)
def load_external_scores(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["item_id", "score"]].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    log.info(
        "External scores loaded: %d items | %d with valid score",
        len(df), df["score"].notna().sum(),
    )
    return df

def run_bl4_kfold(
    folds:          List[pd.DataFrame],
    external_scores: pd.DataFrame,
    baseline_name:  str,
) -> Tuple[Dict, List[Dict]]:
    """
    BL4 – Score esterno Arctic Shift, una sola soglia globale.
    Preprocessing: compressione log simmetrica (fissa, non fold-specifica) per
    ridurre l'influenza dei post virali con score molto elevato. La griglia di
    threshold è la stessa usata dalle altre baseline per garantire confrontabilità.
    Non si inverte la polarità né si usano soglie multiple.
    """
    fold_results = []
    fold_details  = []

    # compressione log applicata una volta sola a tutti gli item:
    # sign(x) * log1p(|x|) preserva il segno e comprime i valori estremi
    scores_df = external_scores[["item_id"]].copy()
    raw = pd.to_numeric(external_scores["score"], errors="coerce").fillna(0.0)
    scores_df["score"] = np.sign(raw) * np.log1p(np.abs(raw))

    for k, val_fold in enumerate(folds):
        train_fold = pd.concat(
            [folds[i] for i in range(len(folds)) if i != k], ignore_index=True
        )
        train_labels = get_item_labels(train_fold)

        # threshold calibrata sul training fold con griglia standard
        threshold, _ = calibrate_global_threshold(scores_df, train_labels)

        test_items = get_item_labels(val_fold).merge(
            scores_df[["item_id", "score"]], on="item_id", how="left"
        )
        test_items["score"] = test_items["score"].fillna(0.0)
        preds = np.where(test_items["score"].values >= threshold, 1, -1)

        fold_metric = evaluate_fold(scores_df, val_fold, preds)
        fold_metric["fold"]      = k
        fold_metric["alpha"]     = float("nan")
        fold_metric["threshold"] = threshold
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  BL4 Fold %d | macro F1 %.4f | AUC %.4f | thr=%.4f | f1_neg=%.4f",
            k, fold_metric["macro_f1"], fold_metric["roc_auc"], threshold, fold_metric["f1_neg"],
        )

    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=float("nan"))
    return metrics, fold_details

def run_bl5_kfold(
    folds:          List[pd.DataFrame],
    external_scores: pd.DataFrame,
    baseline_name:  str,
    min_sub_items:  int = MIN_SUB_VAL_ITEMS,
) -> Tuple[Dict, List[Dict]]:
    """
    BL5 – Score esterno Arctic Shift con soglie per subreddit.
    Preprocessing identico a BL4, ma in più calibra soglie dedicate per comunità.
    """
    fold_results = []
    fold_details  = []

    # stessa compressione log fissa di BL4
    scores_df = external_scores[["item_id"]].copy()
    raw = pd.to_numeric(external_scores["score"], errors="coerce").fillna(0.0)
    scores_df["score"] = np.sign(raw) * np.log1p(np.abs(raw))

    for k, val_fold in enumerate(folds):
        train_fold = pd.concat(
            [folds[i] for i in range(len(folds)) if i != k], ignore_index=True
        )
        train_labels_full = train_fold.drop_duplicates("item_id")[
            ["item_id", "label", "community"]
        ].copy()
        train_labels = train_labels_full[["item_id", "label"]].copy()

        # Soglia globale
        threshold_global, _ = calibrate_global_threshold(scores_df, train_labels, grid=THRESHOLD_GRID)

        # Soglie per subreddit
        sub_thr = calibrate_per_subreddit(
            scores_df,
            train_labels_full,
            threshold_global,
            grid=THRESHOLD_GRID,
            min_items=min_sub_items,
        )

        # Predizione con soglia per comunità
        test_items = get_item_labels(val_fold).merge(
            val_fold.drop_duplicates("item_id")[["item_id", "community"]],
            on="item_id",
            how="left"
        ).merge(scores_df[["item_id", "score"]], on="item_id", how="left")
        test_items["score"] = test_items["score"].fillna(0.0)

        thr_per_item = test_items["community"].map(sub_thr).fillna(threshold_global)
        preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)

        fold_metric = evaluate_fold(scores_df, val_fold, preds)
        fold_metric["fold"]      = k
        fold_metric["alpha"]     = float("nan")
        fold_metric["threshold"] = threshold_global
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  BL5 Fold %d | macro F1 %.4f | AUC %.4f | thr_glob=%.4f | subreddit soglie attive=%d",
            k,
            fold_metric["macro_f1"],
            fold_metric["roc_auc"],
            threshold_global,
            len([v for v in sub_thr.values() if v != threshold_global]),
        )

    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=float("nan"))
    return metrics, fold_details


# main
def run_baselines(
    step1_dir:  Path,
    output_dir: Path,
    n_folds:    int = N_FOLDS,
) -> Dict[str, Dict]:
    log.info("=" * 60)
    log.info("BASELINES (k-fold, k=%d) — %s", n_folds, step1_dir.name)
    log.info("=" * 60)

    all_votes = load_all_votes(step1_dir)
    folds     = make_chrono_folds(all_votes, n_folds)
    results   = {}

    

    log.info("--- BL4: external Reddit score (Arctic Shift) ---")
    external_scores          = load_external_scores(REDDIT_SCORES_PATH)
    metrics_bl4, details_bl4 = run_bl4_kfold(folds, external_scores, "BL4_reddit_score")
    save_outputs(output_dir, metrics_bl4, details_bl4, "BL4_reddit_score")
    results["BL4_reddit_score"] = metrics_bl4

    log.info("--- BL5: external score (Arctic Shift) – soglie per subreddit ---")
    metrics_bl5, details_bl5 = run_bl5_kfold(folds, external_scores, "BL5_subreddit")
    save_outputs(output_dir, metrics_bl5, details_bl5, "BL5_subreddit")
    results["BL5_subreddit"] = metrics_bl5

    log.info("All baselines complete.")
    return results


if __name__ == "__main__":
    run_baselines(
        step1_dir  = Path("results/step1/reddit"),
        output_dir = Path("results/step2/baselines/reddit"),
    )