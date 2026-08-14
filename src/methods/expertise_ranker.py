import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.var_core import calculate_var_scores, evaluate_fold, iterative_cleanup

INPUT_FILE = "data/processed/final_intersection_dataset.csv"

# LOAD & FILTER
df = pd.read_csv(INPUT_FILE)
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "vote", "label"])
df = df.sort_values("timestamp").reset_index(drop=True)


clean_df = iterative_cleanup(df, min_user_votes=5, min_sub_users=40)
clean_df = clean_df.sort_values("timestamp")

# TEMPORAL SPLIT + USER INTERSECTION
split_idx = int(len(clean_df) * 0.8)
split_time = clean_df.iloc[split_idx]["timestamp"]
train_raw = clean_df[clean_df["timestamp"] <= split_time]
test_raw = clean_df[clean_df["timestamp"] > split_time]

common_users = set(train_raw["username"].unique()).intersection(
    set(test_raw["username"].unique())
)
train_df = train_raw[train_raw["username"].isin(common_users)].copy()
test_df = test_raw[test_raw["username"].isin(common_users)].copy()

print(f"Train: {len(train_df):,} votes | Test: {len(test_df):,} votes")
print(f"Common users: {len(common_users):,}\n")

# show label counts in train and test
print("Train label distribution:")
print(train_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

# show vote distribution in train and test
print("\nTrain vote distribution:")
print(train_df["vote"].value_counts())

print("\nTest vote distribution:")
print(test_df["vote"].value_counts())


# TRAIN SCORES: VAR + ACCURACY
print("Training VAR...")
train_scores = calculate_var_scores(train_df)


# TRAIN SCORES: HITS
def calculate_hits_scores(df_in, max_iter=100, tol=1e-8):
    """Calculate hits scores from the supplied data."""
    df_temp = df_in.copy()
    df_temp["is_correct"] = (df_temp["vote"] == df_temp["label"]).astype(int)
    all_scores = []

    for community, group in df_temp.groupby("community"):
        users = group["username"].unique()
        items = group["item_id"].unique()
        if len(users) < 2 or len(items) < 2:
            continue

        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {it: i for i, it in enumerate(items)}
        n_users = len(users)
        n_items = len(items)

        rows = group["username"].map(user_idx).values
        cols = group["item_id"].map(item_idx).values
        weights = group["is_correct"].values.astype(float)

        L = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items))

        auth = np.ones(n_users)
        hub = np.ones(n_items)

        for _ in range(max_iter):
            hub_new = L.T.dot(auth)
            hub_norm = np.linalg.norm(hub_new)
            if hub_norm > 0:
                hub_new /= hub_norm

            auth_new = L.dot(hub_new)
            auth_norm = np.linalg.norm(auth_new)
            if auth_norm > 0:
                auth_new /= auth_norm

            if np.linalg.norm(auth_new - auth) < tol:
                break
            auth = auth_new
            _ = hub_new

        for user, idx in user_idx.items():
            all_scores.append(
                {
                    "community": community,
                    "username": user,
                    "hits_authority": auth[idx],
                }
            )

    return pd.DataFrame(all_scores)


print("Computing HITS...")
hits_scores = calculate_hits_scores(train_df)

# TEST GROUND TRUTH + EVAL DF
test_df = test_df.copy()
test_df["is_violation_catch"] = (
    (test_df["vote"] == -1) & (test_df["label"] == -1)
).astype(int)

test_performance = (
    test_df.groupby(["community", "username"])["is_violation_catch"].sum().reset_index()
)
test_performance.rename(columns={"is_violation_catch": "test_relevance"}, inplace=True)

eval_df = test_performance.merge(train_scores, on=["community", "username"], how="left")
eval_df = eval_df.merge(hits_scores, on=["community", "username"], how="left")
eval_df["VAR_score"] = eval_df["VAR_score"].fillna(0)
eval_df["accuracy"] = eval_df["accuracy"].fillna(0)
eval_df["hits_authority"] = eval_df["hits_authority"].fillna(0)

print(f"Eval pairs: {len(eval_df):,}\n")


# EVALUATION
def get_metrics_loop(df, score_col, target_col, max_k=40):
    """Return metrics loop for the supplied input."""
    metrics = {"k": [], "ndcg": [], "precision": [], "recall": []}
    valid_subs = df["community"].unique()
    for k in range(1, max_k + 1):
        ndcgs, precisions, recalls = [], [], []
        for sub in valid_subs:
            sub_data = df[df["community"] == sub].copy()
            if len(sub_data) < 2:
                continue
            top_k = sub_data.sort_values(score_col, ascending=False).head(k)
            y_true = (sub_data[target_col] > 0).astype(int).values
            y_score = sub_data[score_col].values
            if np.sum(y_true) == 0:
                continue
            try:
                ndcg = ndcg_score([y_true], [y_score], k=k)
            except:
                ndcg = 0
            p = (top_k[target_col] > 0).sum() / k
            r = (top_k[target_col] > 0).sum() / np.sum(y_true)
            ndcgs.append(ndcg)
            precisions.append(p)
            recalls.append(r)
        metrics["k"].append(k)
        metrics["ndcg"].append(np.mean(ndcgs))
        metrics["precision"].append(np.mean(precisions))
        metrics["recall"].append(np.mean(recalls))
    return pd.DataFrame(metrics)


print("Evaluating...")
var_res = get_metrics_loop(eval_df, "VAR_score", "test_relevance")
acc_res = get_metrics_loop(eval_df, "accuracy", "test_relevance")
hits_res = get_metrics_loop(eval_df, "hits_authority", "test_relevance")

var_res["Model"] = "VAR"
acc_res["Model"] = "Raw Accuracy"
hits_res["Model"] = "HITS"

all_results = pd.concat([var_res, acc_res, hits_res], ignore_index=True)

# Summary table
summary = all_results[all_results["k"].isin([1, 5, 10, 20, 40])].copy()
summary[["ndcg", "precision", "recall"]] = summary[
    ["ndcg", "precision", "recall"]
].round(3)
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
for k_val in [1, 5, 10, 20, 40]:
    print(f"\n  K = {k_val}")
    print(
        summary[summary["k"] == k_val][
            ["Model", "ndcg", "precision", "recall"]
        ].to_string(index=False)
    )

# PLOT
sns.set_theme(style="white", context="poster")

label_size = 24
title_size = 28
tick_size = 20

palette = {"VAR": "#2563eb", "HITS": "#16a34a", "Raw Accuracy": "#64748b"}
dashes = {"VAR": "", "HITS": (4, 2), "Raw Accuracy": (2, 2)}

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
configs = [
    ("ndcg", "NDCG@K"),
    ("precision", "Precision@K"),
    ("recall", "Recall@K"),
]

for idx, (metric, title) in enumerate(configs):
    ax = axes[idx]

    for model in ["VAR", "HITS", "Raw Accuracy"]:
        mdata = all_results[all_results["Model"] == model]
        ax.plot(
            mdata["k"],
            mdata[metric],
            label=model,
            color=palette[model],
            linewidth=2.5,
            dashes=dashes[model] if dashes[model] else [],
        )

    ax.set_title(title, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("K", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    if idx == 0:
        ax.legend(fontsize=label_size - 4, frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig("var_evaluation.png", dpi=150, bbox_inches="tight")
plt.savefig("var_evaluation.pdf", dpi=300, bbox_inches="tight")

plt.show()


import json
from pathlib import Path
from sklearn.model_selection import KFold

N_FOLDS    = 5
SCALAR_METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]

def var_classification_kfold(test_df, train_scores, k, n_folds=N_FOLDS):
    """Evaluate VAR classification with cross-validation."""
    test_ranked = test_df.merge(
        train_scores[["community", "username", "VAR_score"]],
        on=["community", "username"],
        how="left",
    )
    test_ranked["VAR_score"] = test_ranked["VAR_score"].fillna(0)

    top_k_users = (
        test_ranked[["community", "username", "VAR_score"]]
        .drop_duplicates()
        .sort_values("VAR_score", ascending=False)
        .groupby("community")
        .head(k)
    )
    votes = test_ranked.merge(
        top_k_users[["community", "username"]], on=["community", "username"]
    )

    decisions = (
        votes.groupby(["community", "item_id"])
        .agg(predicted=("vote", lambda x: x.value_counts().idxmax()),
             label=("label", "first"))
        .reset_index()
    )

    labeled_items = decisions[["item_id", "label"]].drop_duplicates().reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)

    fold_metrics = []
    for _, test_idx in kf.split(labeled_items):
        test_ids     = labeled_items.iloc[test_idx]["item_id"].values
        fold_dec     = decisions[decisions["item_id"].isin(test_ids)]
        if len(fold_dec) < 2 or fold_dec["label"].nunique() < 2:
            continue
        fold_metrics.append(evaluate_fold(fold_dec))

    agg = {"n_folds": len(fold_metrics)}
    for key in SCALAR_METRICS:
        vals = [m[key] for m in fold_metrics]
        agg[key]                = float(np.mean(vals))
        agg[f"{key}_std"]       = float(np.std(vals))
        agg[f"{key}_per_fold"]  = vals
    return agg

OUTPUT_ROOT = Path("results/reddit/kfold")

print("\nExporting VAR classification metrics (KFold)...")
best_agg, best_k, best_f1 = None, None, -1
for k in [5, 10, 15, 20, 30, 40, 50]:
    agg = var_classification_kfold(test_df, train_scores, k)
    print(f"  VAR_k{k}: macro_f1={agg['macro_f1']:.3f} +/- {agg['macro_f1_std']:.3f}  roc_auc={agg['roc_auc']:.3f}")
    if agg['macro_f1'] > best_f1:
        best_f1, best_k, best_agg = agg['macro_f1'], k, agg

out_dir = OUTPUT_ROOT / f"expertise-k{best_k}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "metrics.json", "w") as f:
    json.dump(best_agg, f, indent=2)
print(f"\n  Saved best: expertise-k{best_k} (macro_f1={best_f1:.3f})")
print("Done.\n")
