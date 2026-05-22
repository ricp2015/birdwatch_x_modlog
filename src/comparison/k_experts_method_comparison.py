import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold
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
test_raw  = clean_df[clean_df["timestamp"] > split_time]

common_users = set(train_raw["username"].unique()).intersection(
    set(test_raw["username"].unique())
)
train_df = train_raw[train_raw["username"].isin(common_users)].copy()
test_df  = test_raw[test_raw["username"].isin(common_users)].copy()

print(f"Train: {len(train_df):,} votes | Test: {len(test_df):,} votes")
print(f"Common users: {len(common_users):,}\n")


# TRAIN SCORES: VAR
def calculate_var_scores(df_in):
    cols = [c for c in df_in.columns if c != "weight"]
    df_clean = df_in[cols].copy()
    sub_stats  = df_clean.groupby(["community", "label"]).size().reset_index(name="count")
    sub_totals = df_clean.groupby("community").size().reset_index(name="total")
    weights_df = sub_stats.merge(sub_totals, on="community")
    weights_df["weight"] = np.log(weights_df["total"] / weights_df["count"])
    df_weighted = df_clean.merge(
        weights_df[["community", "label", "weight"]], on=["community", "label"], how="left"
    )
    df_weighted["is_correct"] = (df_weighted["vote"] == df_weighted["label"]).astype(int)
    df_weighted["score_contribution"] = df_weighted["is_correct"] * df_weighted["weight"]
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
    user_scores["VAR_score"] = user_scores["weighted_score_sum"] / (user_scores["total_votes"] + LAMBDA)
    user_scores["accuracy"]  = user_scores["raw_correct"] / user_scores["total_votes"]
    return user_scores


print("Training VAR...")
train_scores = calculate_var_scores(train_df)


# ── aggregation strategies ───────────────────────────────────────────────────

N_FOLDS    = 5
K_VALUES   = [1, 2, 3, 5, 10]
OUTPUT_ROOT = Path("results/step2")
SCALAR_METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]


def _base_votes(test_df, train_scores, k):
    """Merge VAR scores into test votes and keep only top-K users per community."""
    test_ranked = test_df.merge(
        train_scores[["community", "username", "VAR_score"]],
        on=["community", "username"], how="left",
    )
    test_ranked["VAR_score"] = test_ranked["VAR_score"].fillna(0)
    top_k_users = (
        test_ranked[["community", "username", "VAR_score"]]
        .drop_duplicates()
        .sort_values("VAR_score", ascending=False)
        .groupby("community").head(k)
    )
    return test_ranked.merge(top_k_users[["community", "username"]], on=["community", "username"])


def _majority(votes):
    """Unweighted majority vote."""
    return (
        votes.groupby(["community", "item_id"])
        .agg(predicted=("vote", lambda x: x.value_counts().idxmax()),
             label=("label", "first"))
        .reset_index()
    )


def _weighted_majority(votes):
    """VAR-score weighted majority: sum scores per candidate label, pick highest."""
    def wm(g):
        return g.groupby("vote")["VAR_score"].sum().idxmax()
    return (
        votes.groupby(["community", "item_id"])
        .apply(lambda g: pd.Series({"predicted": wm(g), "label": g["label"].iloc[0]}))
        .reset_index()
    )


def _top1(votes):
    """Single vote from the highest-VAR user who voted on each item."""
    return (
        votes.sort_values("VAR_score", ascending=False)
        .groupby(["community", "item_id"]).first()
        .reset_index()
        [["community", "item_id", "vote", "label"]]
        .rename(columns={"vote": "predicted"})
    )


def _weighted_threshold(votes):
    """VAR-weighted average of numeric votes; sign determines prediction."""
    majority_class = int(votes["vote"].mode().iloc[0])
    def wt(g):
        w = g["VAR_score"].values
        if w.sum() == 0:
            return majority_class
        return int(np.sign(np.average(g["vote"], weights=w) or majority_class))
    return (
        votes.groupby(["community", "item_id"])
        .apply(lambda g: pd.Series({"predicted": wt(g), "label": g["label"].iloc[0]}))
        .reset_index()
    )


def _consensus(votes):
    """Predict only when all top-K voters agree; else fall back to majority class."""
    majority_class = int(votes["vote"].mode().iloc[0])
    def cons(g):
        u = g["vote"].unique()
        return int(u[0]) if len(u) == 1 else majority_class
    return (
        votes.groupby(["community", "item_id"])
        .apply(lambda g: pd.Series({"predicted": cons(g), "label": g["label"].iloc[0]}))
        .reset_index()
    )


AGGREGATIONS = {
    "majority":  _majority,
    "weighted":  _weighted_majority,
}


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_fold(decisions):
    y_true = decisions["label"].values
    y_pred = decisions["predicted"].values
    labels = sorted(np.unique(y_true))
    return {
        "macro_f1":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "roc_auc":         roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr", labels=labels),
        "f1_pos":          f1_score(y_true, y_pred, pos_label=1,        average="binary", zero_division=0),
        "f1_neg":          f1_score(y_true, y_pred, pos_label=-1,       average="binary", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "macro_recall":    recall_score(y_true, y_pred, average="macro",       zero_division=0),
        "precision_pos":   precision_score(y_true, y_pred, pos_label=1,  average="binary", zero_division=0),
        "recall_pos":      recall_score(y_true, y_pred, pos_label=1,     average="binary", zero_division=0),
        "precision_neg":   precision_score(y_true, y_pred, pos_label=-1, average="binary", zero_division=0),
        "recall_neg":      recall_score(y_true, y_pred, pos_label=-1,    average="binary", zero_division=0),
    }


def kfold_eval(decisions, n_folds=N_FOLDS):
    """KFold aggregation over item-level decisions."""
    labeled_items = decisions[["item_id", "label"]].drop_duplicates().reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for _, test_idx in kf.split(labeled_items):
        test_ids = labeled_items.iloc[test_idx]["item_id"].values
        fold_dec = decisions[decisions["item_id"].isin(test_ids)]
        if len(fold_dec) < 2 or fold_dec["label"].nunique() < 2:
            continue
        fold_metrics.append(evaluate_fold(fold_dec))
    agg = {"n_folds": len(fold_metrics)}
    for key in SCALAR_METRICS:
        vals = [m[key] for m in fold_metrics]
        agg[key]               = float(np.mean(vals))
        agg[f"{key}_std"]      = float(np.std(vals))
        agg[f"{key}_per_fold"] = vals
    return agg


def _base_votes_per_item(test_df, train_scores, k):
    """For each item, keep votes from the top-K highest-VAR users who actually voted it."""
    test_ranked = test_df.merge(
        train_scores[["community", "username", "VAR_score"]],
        on=["community", "username"], how="left",
    )
    test_ranked["VAR_score"] = test_ranked["VAR_score"].fillna(0)
    # rank users within each item by VAR score, keep top-K per item
    test_ranked["rank"] = (
        test_ranked.groupby(["community", "item_id"])["VAR_score"]
        .rank(method="first", ascending=False)
    )
    return test_ranked[test_ranked["rank"] <= k].drop(columns="rank")


def run_aggregation_comparison(test_df, train_scores):
    """Evaluate all aggregation strategies across K values and selection modes."""
    rows = []

    selection_modes = {
        "per_community": _base_votes,
        "per_item":      _base_votes_per_item,
    }

    for agg_name, agg_fn in AGGREGATIONS.items():
        for mode_name, base_fn in selection_modes.items():
            label = f"{agg_name}_{mode_name}"
            best_agg, best_k, best_f1 = None, None, -1
            print(f"\n  [{label}]")

            for k in K_VALUES:
                votes     = base_fn(test_df, train_scores, k)
                decisions = agg_fn(votes)
                agg       = kfold_eval(decisions)
                f1, std, auc, f1n = agg["macro_f1"], agg["macro_f1_std"], agg["roc_auc"], agg["f1_neg"]
                print(f"    k={k:>2}: macro_f1={f1:.3f} ±{std:.3f}  roc_auc={auc:.3f}  f1_neg={f1n:.3f}")
                rows.append({"strategy": label, "k": k,
                             "macro_f1": f1, "macro_f1_std": std,
                             "roc_auc": auc, "f1_neg": f1n, "f1_pos": agg["f1_pos"]})
                if f1 > best_f1:
                    best_f1, best_k, best_agg = f1, k, agg

            out_dir = OUTPUT_ROOT / f"VAR_{label}_k{best_k}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(best_agg, f, indent=2)
            print(f"    → saved best: VAR_{label}_k{best_k} (macro_f1={best_f1:.3f})")

    return pd.DataFrame(rows)


# ── summary + plot ────────────────────────────────────────────────────────────

def print_summary(results_df):
    """Best K per strategy, sorted by macro_f1."""
    best = (
        results_df.sort_values("macro_f1", ascending=False)
        .groupby("strategy").first()
        .reset_index()
        .sort_values("macro_f1", ascending=False)
        [["strategy", "k", "macro_f1", "macro_f1_std", "roc_auc", "f1_neg", "f1_pos"]]
        .round(3)
    )
    print("\n" + "=" * 70)
    print("AGGREGATION COMPARISON — best K per strategy")
    print("=" * 70)
    print(best.to_string(index=False))
    print("=" * 70)


def plot_comparison(results_df):
    sns.set_theme(style="white", context="talk")
    palette = {
        "majority_per_community":  "#2563eb",
        "majority_per_item":       "#93c5fd",
        "weighted_per_community":  "#16a34a",
        "weighted_per_item":       "#86efac",
    }
    metrics_to_plot = [
    ("macro_f1", "Macro F1"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy, grp in results_df.groupby("strategy"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["macro_f1"],
                label=strategy, color=palette.get(strategy, "gray"),
                linewidth=2.5, marker="o", markersize=4)
        if "macro_f1_std" in grp.columns:
            ax.fill_between(grp["k"],
                            grp["macro_f1"] - grp["macro_f1_std"],
                            grp["macro_f1"] + grp["macro_f1_std"],
                            alpha=0.12, color=palette.get(strategy, "gray"))

    ymin = results_df["macro_f1"].min()
    ymax = results_df["macro_f1"].max()
    margin = (ymax - ymin) * 0.3
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_title("Macro F1 by aggregation strategy", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Macro F1")
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig("var_aggregation_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig("var_aggregation_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def show_prediction_examples(test_df, train_scores, k=5, n_examples=5):
    """Show item-level prediction examples with voter details, correct and wrong."""
    votes = _base_votes(test_df, train_scores, k)
    decisions = _majority(votes)

    voter_details = (
        votes[["item_id", "username", "vote", "VAR_score"]]
        .sort_values(["item_id", "VAR_score"], ascending=[True, False])
    )

    # ── voter coverage stats ─────────────────────────────────────────────────
    voters_per_item = voter_details.groupby("item_id")["username"].count()
    coverage = voters_per_item.value_counts().sort_index()
    print(f"\n{'='*65}")
    print(f"  VOTER COVERAGE (k={k}) — how many expert voters per item")
    print(f"{'='*65}")
    print(f"  {'n_voters':>8}  {'n_items':>8}  {'pct':>7}")
    print(f"  {'-'*28}")
    total = len(voters_per_item)
    for n_voters, n_items in coverage.items():
        print(f"  {n_voters:>8}  {n_items:>8}  {100*n_items/total:>6.1f}%")
    print(f"  {'-'*28}")
    print(f"  {'total':>8}  {total:>8}")
    print(f"\n  mean voters/item: {voters_per_item.mean():.2f}  "
          f"median: {voters_per_item.median():.0f}  "
          f"max: {voters_per_item.max()}")

    # ── prediction examples ──────────────────────────────────────────────────
    correct = decisions[decisions["predicted"] == decisions["label"]]
    wrong   = decisions[decisions["predicted"] != decisions["label"]]

    def print_examples(subset, title, n):
        print(f"\n{'='*65}")
        print(f"  {title}  (k={k})")
        print(f"{'='*65}")
        for _, row in subset.sample(min(n, len(subset)), random_state=42).iterrows():
            item   = row["item_id"]
            label  = int(row["label"])
            pred   = int(row["predicted"])
            voters = voter_details[voter_details["item_id"] == item]
            print(f"\n  item: {item}")
            print(f"  label={label:+d}  predicted={pred:+d}  {'✓' if pred == label else '✗'}  "
                  f"({len(voters)} voters)")
            print(f"  {'username':<20} {'vote':>5}  {'VAR_score':>10}")
            print(f"  {'-'*38}")
            for _, v in voters.iterrows():
                print(f"  {v['username']:<20} {int(v['vote']):>+5}  {v['VAR_score']:>10.4f}")

    print_examples(correct, "CORRECT PREDICTIONS", n_examples)
    print_examples(wrong,   "WRONG PREDICTIONS",   n_examples)


# ── run ──────────────────────────────────────────────────────────────────────

print("\nRunning aggregation comparison...")
results_df = run_aggregation_comparison(test_df, train_scores)
print_summary(results_df)
plot_comparison(results_df)
show_prediction_examples(test_df, train_scores, k=100, n_examples=5)
print("Total unique items in test:", test_df["item_id"].nunique())

# quanti item ha votato ciascuno dei top-10 utenti
top10 = train_scores.sort_values("VAR_score", ascending=False).head(10)
for _, row in top10.iterrows():
    n = test_df[test_df["username"] == row["username"]]["item_id"].nunique()
    print(f"  {row['username']:<25} VAR={row['VAR_score']:.3f}  items_voted={n}")