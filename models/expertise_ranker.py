import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "data/processed/final_intersection_dataset.csv"

# LOAD & FILTER
df = pd.read_csv(INPUT_FILE)
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "vote", "label"])
df = df.sort_values("timestamp").reset_index(drop=True)


def iterative_cleanup(df, min_user_votes=5, min_sub_users=40):
    print(f"Starting cleanup on {len(df):,} rows...")
    df_curr = df.copy()
    iteration = 0
    while True:
        iteration += 1
        start_len = len(df_curr)
        user_counts = df_curr["username"].value_counts()
        valid_users = user_counts[user_counts >= min_user_votes].index
        df_curr = df_curr[df_curr["username"].isin(valid_users)]
        sub_user_counts = df_curr.groupby("community")["username"].nunique()
        valid_subs = sub_user_counts[sub_user_counts >= min_sub_users].index
        df_curr = df_curr[df_curr["community"].isin(valid_subs)]
        end_len = len(df_curr)
        dropped = start_len - end_len
        print(
            f"  Iter {iteration}: Dropped {dropped} rows. "
            f"Remaining: {end_len:,} votes | "
            f"{df_curr['username'].nunique()} users | "
            f"{df_curr['community'].nunique()} subs"
        )
        if start_len == end_len:
            break
    print("Cleanup complete.\n")
    return df_curr


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
def calculate_var_scores(df_in):
    cols = [c for c in df_in.columns if c != "weight"]
    df_clean = df_in[cols].copy()
    sub_stats = (
        df_clean.groupby(["community", "label"]).size().reset_index(name="count")
    )
    sub_totals = df_clean.groupby("community").size().reset_index(name="total")
    weights_df = sub_stats.merge(sub_totals, on="community")
    weights_df["weight"] = np.log(weights_df["total"] / weights_df["count"])
    df_weighted = df_clean.merge(
        weights_df[["community", "label", "weight"]],
        on=["community", "label"],
        how="left",
    )
    df_weighted["is_correct"] = (df_weighted["vote"] == df_weighted["label"]).astype(
        int
    )
    df_weighted["score_contribution"] = (
        df_weighted["is_correct"] * df_weighted["weight"]
    )
    user_scores = (
        df_weighted.groupby(["community", "username"])
        .agg(
            total_votes=("item_id", "count"),
            weighted_score_sum=("score_contribution", "sum"),
            raw_correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    LAMBDA = 10
    user_scores["VAR_score"] = user_scores["weighted_score_sum"] / (
        user_scores["total_votes"] + LAMBDA
    )
    user_scores["accuracy"] = user_scores["raw_correct"] / user_scores["total_votes"]
    return user_scores


print("Training VAR...")
train_scores = calculate_var_scores(train_df)


# TRAIN SCORES: HITS
def calculate_hits_scores(df_in, max_iter=100, tol=1e-8):
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