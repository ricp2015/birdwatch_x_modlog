"""
Grid search over metadata-based vote filtering thresholds.

For each combination of (min_tenure_days, min_karma) we:
  1. Filter out votes from users below the threshold
  2. Rerun step C (skill extraction) and step D (ranking + eval)
  3. Record macro_F1, AUC, f1_neg, coverage

"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

# Grid to search
MIN_TENURE_GRID = [0, 30, 90, 180, 365]       # days since account creation
MIN_KARMA_GRID  = [0, 10, 100, 500, 1000]     # total karma

THRESHOLD_GRID  = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5
N_FOLDS         = 5
REF_TS          = 1685000000   # ~May 2023, dataset end

CATEGORIES = [
    "spam", "meta-rules", "content", "doxxing",
    "harassment", "hatespeech", "format",
    "off-topic", "trolling", "incivility",
]

ORIGINAL_CSV   = Path("data/processed/final_intersection_dataset.csv")
METADATA_PATH  = Path("data/processed/user_metadata.csv")
SCORES_PATH_DEFAULT = Path("results/reddit/random/team-formation/post_violation_scores.parquet")


# Data loading

def load_all_votes() -> pd.DataFrame:
    """Load all votes from its configured source."""
    votes = pd.read_csv(ORIGINAL_CSV)
    votes["vote"] = votes["vote"].astype(float)
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    return votes[["username", "item_id", "vote", "community", "label"]]


def load_metadata() -> pd.DataFrame:
    """Load metadata from its configured source."""
    meta = pd.read_csv(METADATA_PATH)
    meta = meta[meta["is_suspended"] != True].copy()
    meta["account_created_utc"] = pd.to_numeric(
        meta["account_created_utc"], errors="coerce")
    meta["tenure_days"] = (REF_TS - meta["account_created_utc"]) / 86400
    meta["tenure_days"] = meta["tenure_days"].clip(lower=0)
    meta["total_karma"] = pd.to_numeric(meta["total_karma"], errors="coerce").fillna(0)
    return meta[["username", "tenure_days", "total_karma"]].dropna(subset=["username"])


def filter_votes(votes: pd.DataFrame, meta: pd.DataFrame,
                 min_tenure: int, min_karma: int) -> pd.DataFrame:
    """Filter votes using the selected metadata thresholds."""
    if min_tenure == 0 and min_karma == 0:
        return votes
    eligible = meta[
        (meta["tenure_days"] >= min_tenure) &
        (meta["total_karma"] >= min_karma)
    ]["username"]
    # always keep users not in metadata (unknown -> don't filter them out)
    unknown = set(votes["username"].unique()) - set(meta["username"].unique())
    keep    = set(eligible) | unknown
    return votes[votes["username"].isin(keep)].copy()


# Skill extraction (simplified, mod-agreement only for speed)

def compute_skill(votes: pd.DataFrame, scores: pd.DataFrame,
                  min_votes: int, alpha: float,
                  meta: pd.DataFrame) -> pd.DataFrame:
    """Compute skill from the supplied data."""
    merged = votes.merge(
        scores[["item_id", "top_violation_category"]], on="item_id", how="inner"
    ).dropna(subset=["label", "top_violation_category"])
    merged["agrees"] = (merged["vote"] == merged["label"]).astype(float)

    # feature skill (user-level)
    if alpha < 1.0 and len(meta) > 0:
        meta_f = meta.copy()
        for col in ["total_karma", "tenure_days"]:
            meta_f[f"{col}_rank"] = meta_f[col].rank(pct=True, na_option="bottom")
        meta_f["skill_feat"] = (
            meta_f["total_karma_rank"] * 0.5 +
            meta_f["tenure_days_rank"] * 0.5
        ) - 0.5
        feat_lookup = meta_f.set_index("username")["skill_feat"]
    else:
        feat_lookup = pd.Series(dtype=float)

    records = []
    for (username, community), grp in merged.groupby(["username", "community"]):
        row = {"username": username, "community": community}
        feat = float(feat_lookup.get(username, np.nan)) if len(feat_lookup) else np.nan
        skill_vals = []
        for cat in CATEGORIES:
            cat_grp = grp[grp["top_violation_category"] == cat]
            s_mod = float(cat_grp["agrees"].mean()) - 0.5 \
                    if len(cat_grp) >= min_votes else np.nan
            if not np.isnan(s_mod) and not np.isnan(feat):
                s = alpha * s_mod + (1 - alpha) * feat
            elif not np.isnan(s_mod):
                s = s_mod
            elif not np.isnan(feat) and alpha < 1.0:
                s = feat
            else:
                s = np.nan
            row[f"skill_{cat}"] = s
            if not np.isnan(s):
                skill_vals.append(s)
        row["skill_overall"] = float(np.nanmean(skill_vals)) if skill_vals else np.nan
        records.append(row)

    return pd.DataFrame(records)


# Scoring + evaluation

def compute_item_scores(votes: pd.DataFrame, scores: pd.DataFrame,
                        skills: pd.DataFrame) -> pd.DataFrame:
    """Compute item scores from the supplied data."""
    votes_scores = votes.merge(
        scores[["item_id", "top_violation_category"]], on="item_id", how="left"
    )
    records = []
    for cat in CATEGORIES:
        subset = votes_scores[
            votes_scores["top_violation_category"] == cat].copy()
        if subset.empty:
            continue
        skill_col = f"skill_{cat}"
        subset = subset.merge(
            skills[["username", "community", skill_col]],
            on=["username", "community"], how="left"
        )
        records.append(
            subset[["item_id", "vote", skill_col, "community", "label"]]
            .rename(columns={skill_col: "weight"})
        )
    no_cat = votes_scores[votes_scores["top_violation_category"].isna()].copy()
    no_cat["weight"] = np.nan
    records.append(no_cat[["item_id", "vote", "weight", "community", "label"]])
    all_votes = pd.concat(records, ignore_index=True)

    item_scores = []
    for item_id, grp in all_votes.groupby("item_id"):
        valid = grp.dropna(subset=["weight"])
        denom = valid["weight"].abs().sum()
        if len(valid) == 0 or denom == 0:
            continue
        item_scores.append({
            "item_id":   item_id,
            "community": grp["community"].iloc[0],
            "label":     grp["label"].iloc[0],
            "tfr_score": float((valid["weight"] * valid["vote"]).sum() / denom),
        })
    return pd.DataFrame(item_scores)


def evaluate_kfold(item_scores: pd.DataFrame, n_folds: int = N_FOLDS) -> Dict:
    """Evaluate kfold and return its metrics."""
    labeled = item_scores.dropna(subset=["label"]).reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
    fold_metrics = []
    for val_idx, test_idx in kf.split(labeled):
        val_ids  = labeled.iloc[val_idx]["item_id"].values
        test_ids = labeled.iloc[test_idx]["item_id"].values

        # calibrate
        val = item_scores[item_scores["item_id"].isin(val_ids)].dropna(subset=["label"])
        best_f, best_thr = -1.0, 0.0
        for thr in THRESHOLD_GRID:
            y_pred = np.where(val["tfr_score"].values >= thr, 1, -1)
            _, _, f, _ = precision_recall_fscore_support(
                val["label"].values, y_pred,
                labels=[1, -1], average="macro", zero_division=0)
            if f > best_f:
                best_f, best_thr = f, thr

        # evaluate
        test = item_scores[item_scores["item_id"].isin(test_ids)].dropna(subset=["label"])
        y_true = test["label"].values
        y_pred = np.where(test["tfr_score"].values >= best_thr, 1, -1)
        y_sc   = test["tfr_score"].values

        _, _, f,  _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1,-1], average="macro", zero_division=0)
        _, _, f_, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[-1], average="binary", pos_label=-1, zero_division=0)
        try:
            auc = roc_auc_score((y_true==1).astype(int), y_sc)
        except ValueError:
            auc = float("nan")
        fold_metrics.append({"macro_f1": f, "f1_neg": f_, "roc_auc": auc})

    return {
        "macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "f1_neg":   float(np.mean([m["f1_neg"]   for m in fold_metrics])),
        "roc_auc":  float(np.mean([m["roc_auc"]  for m in fold_metrics])),
        "coverage": float(len(item_scores) / item_scores["item_id"].nunique()),
    }


# Main grid search

def run(scores_path: Path, out_dir: Path, alpha: float) -> None:
    """Run the configured analysis workflow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data (alpha={alpha})...")
    all_votes = load_all_votes()
    meta      = load_metadata()
    scores    = pd.read_parquet(scores_path)

    n_total_users = all_votes["username"].nunique()
    n_total_posts = all_votes["item_id"].nunique()
    print(f"Total users: {n_total_users:,} | Total posts: {n_total_posts:,}")
    print(f"Metadata available for: {meta['username'].nunique():,} users")
    print(f"\nGrid: {len(MIN_TENURE_GRID)} tenure x {len(MIN_KARMA_GRID)} karma "
          f"= {len(MIN_TENURE_GRID)*len(MIN_KARMA_GRID)} combinations\n")

    results = []
    grid = list(product(MIN_TENURE_GRID, MIN_KARMA_GRID))

    for min_tenure, min_karma in tqdm(grid, desc="Grid search"):
        filtered = filter_votes(all_votes, meta, min_tenure, min_karma)
        n_users_kept = filtered["username"].nunique()
        n_votes_kept = len(filtered)
        pct_users    = 100 * n_users_kept / n_total_users

        skills      = compute_skill(filtered, scores, MIN_VOTES_SKILL, alpha, meta)
        item_scores = compute_item_scores(filtered, scores, skills)
        coverage    = len(item_scores) / n_total_posts

        if len(item_scores) < 100:
            print(f"  tenure>={min_tenure:4d}d  karma>={min_karma:5d}  "
                  f"-> too few items ({len(item_scores)}), skipping")
            continue

        metrics = evaluate_kfold(item_scores)

        row = {
            "min_tenure_days": min_tenure,
            "min_karma":       min_karma,
            "n_users_kept":    n_users_kept,
            "pct_users_kept":  round(pct_users, 1),
            "n_votes_kept":    n_votes_kept,
            "n_items_scored":  len(item_scores),
            "coverage":        round(coverage, 4),
            "macro_f1":        round(metrics["macro_f1"], 4),
            "f1_neg":          round(metrics["f1_neg"],   4),
            "roc_auc":         round(metrics["roc_auc"],  4),
        }
        results.append(row)
        print(f"  tenure>={min_tenure:4d}d  karma>={min_karma:5d}  "
              f"users={pct_users:5.1f}%  "
              f"F1={metrics['macro_f1']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}  "
              f"f1_neg={metrics['f1_neg']:.4f}  "
              f"cov={coverage:.3f}")

    df = pd.DataFrame(results)
    out_csv = out_dir / f"filter_search_alpha{alpha:.1f}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved -> {out_csv}")

    # summary: best by each metric
    for metric in ("macro_f1", "roc_auc", "f1_neg"):
        best = df.loc[df[metric].idxmax()]
        print(f"\nBest {metric}: {best[metric]:.4f}  "
              f"(tenure>={int(best['min_tenure_days'])}d, "
              f"karma>={int(best['min_karma'])}, "
              f"users={best['pct_users_kept']}%)")

    # save as JSON too for easy loading
    with open(out_dir / f"filter_search_alpha{alpha:.1f}.json", "w") as fh:
        json.dump(df.to_dict(orient="records"), fh, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse args from the supplied input."""
    p = argparse.ArgumentParser()
    p.add_argument("--scores_path", default=str(SCORES_PATH_DEFAULT))
    p.add_argument("--out_dir",     default="results/diagnostics/team-formation-filter-search")
    p.add_argument("--alpha",       type=float, default=1.0,
                   help="Skill combination weight (1.0=pure mod agreement)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.scores_path), Path(args.out_dir), args.alpha)
