import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.splits import discover_splits, load_vote_partitions
from src.methods.var_core import calculate_var_scores, evaluate_fold, iterative_cleanup
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

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
test_raw  = clean_df[clean_df["timestamp"] > split_time]

common_users = set(train_raw["username"].unique()).intersection(
    set(test_raw["username"].unique())
)
train_df = train_raw[train_raw["username"].isin(common_users)].copy()
test_df  = test_raw[test_raw["username"].isin(common_users)].copy()

print(f"Train: {len(train_df):,} votes | Test: {len(test_df):,} votes")
print(f"Common users: {len(common_users):,}\n")


# TRAIN SCORES: VAR
print("Training VAR...")
train_scores = calculate_var_scores(train_df)


# aggregation strategies

N_FOLDS    = 5
K_VALUES   = [10]
OUTPUT_ROOT = Path("results/reddit")
SCALAR_METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]


def _base_votes(test_df, train_scores, k):
    """Count positive and negative votes for each item."""
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
    """Return the majority-vote prediction for each item."""
    return (
        votes.groupby(["community", "item_id"])
        .agg(predicted=("vote", lambda x: x.value_counts().idxmax()),
             label=("label", "first"))
        .reset_index()
    )


def _weighted_majority(votes):
    """Return reliability-weighted majority predictions."""
    def wm(g):
        """Return the weighted-majority prediction for one item."""
        return g.groupby("vote")["VAR_score"].sum().idxmax()
    return (
        votes.groupby(["community", "item_id"])
        .apply(lambda g: pd.Series({"predicted": wm(g), "label": g["label"].iloc[0]}))
        .reset_index()
    )


def _top1(votes):
    """Return the vote from the highest-ranked user."""
    return (
        votes.sort_values("VAR_score", ascending=False)
        .groupby(["community", "item_id"]).first()
        .reset_index()
        [["community", "item_id", "vote", "label"]]
        .rename(columns={"vote": "predicted"})
    )


def _weighted_threshold(votes):
    """Apply a threshold to reliability-weighted votes."""
    majority_class = int(votes["vote"].mode().iloc[0])
    def wt(g):
        """Return the weighted-threshold prediction for one item."""
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
    """Return predictions only where selected voters agree."""
    majority_class = int(votes["vote"].mode().iloc[0])
    def cons(g):
        """Return the consensus prediction for one item."""
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


# evaluation

def kfold_eval(decisions, n_folds=N_FOLDS):
    """Evaluate one aggregation strategy with cross-validation."""
    labeled_items = decisions[["item_id", "label"]].drop_duplicates().reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
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
    """Count available votes for each item."""
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
    """Run the aggregation comparison workflow."""
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
                print(f"    k={k:>2}: macro_f1={f1:.3f} +/-{std:.3f}  roc_auc={auc:.3f}  f1_neg={f1n:.3f}")
                rows.append({"strategy": label, "k": k,
                             "macro_f1": f1, "macro_f1_std": std,
                             "roc_auc": auc, "f1_neg": f1n, "f1_pos": agg["f1_pos"]})
                if f1 > best_f1:
                    best_f1, best_k, best_agg = f1, k, agg

            out_dir = OUTPUT_ROOT / f"VAR_{label}_k{best_k}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(best_agg, f, indent=2)
            print(f"    -> saved best: VAR_{label}_k{best_k} (macro_f1={best_f1:.3f})")

    return pd.DataFrame(rows)


# summary + plot

def print_summary(results_df):
    """Print summary to the console."""
    best = (
        results_df.sort_values("macro_f1", ascending=False)
        .groupby("strategy").first()
        .reset_index()
        .sort_values("macro_f1", ascending=False)
        [["strategy", "k", "macro_f1", "macro_f1_std", "roc_auc", "f1_neg", "f1_pos"]]
        .round(3)
    )
    print("\n" + "=" * 70)
    print("AGGREGATION COMPARISON - best K per strategy")
    print("=" * 70)
    print(best.to_string(index=False))
    print("=" * 70)


def plot_comparison(results_df):
    """Plot comparison and save the figure."""
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
    """Display prediction examples for inspection."""
    votes = _base_votes(test_df, train_scores, k)
    decisions = _majority(votes)

    voter_details = (
        votes[["item_id", "username", "vote", "VAR_score"]]
        .sort_values(["item_id", "VAR_score"], ascending=[True, False])
    )

    # voter coverage stats
    voters_per_item = voter_details.groupby("item_id")["username"].count()
    coverage = voters_per_item.value_counts().sort_index()
    print(f"\n{'='*65}")
    print(f"  VOTER COVERAGE (k={k}) - how many expert voters per item")
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

    # prediction examples
    correct = decisions[decisions["predicted"] == decisions["label"]]
    wrong   = decisions[decisions["predicted"] != decisions["label"]]

    def print_examples(subset, title, n):
        """Print examples to the console."""
        print(f"\n{'='*65}")
        print(f"  {title}  (k={k})")
        print(f"{'='*65}")
        for _, row in subset.sample(min(n, len(subset)), random_state=10).iterrows():
            item   = row["item_id"]
            label  = int(row["label"])
            pred   = int(row["predicted"])
            voters = voter_details[voter_details["item_id"] == item]
            print(f"\n  item: {item}")
            print(f"  label={label:+d}  predicted={pred:+d}  {'OK' if pred == label else 'FAIL'}  "
                  f"({len(voters)} voters)")
            print(f"  {'username':<20} {'vote':>5}  {'VAR_score':>10}")
            print(f"  {'-'*38}")
            for _, v in voters.iterrows():
                print(f"  {v['username']:<20} {int(v['vote']):>+5}  {v['VAR_score']:>10.4f}")

    print_examples(correct, "CORRECT PREDICTIONS", n_examples)
    print_examples(wrong,   "WRONG PREDICTIONS",   n_examples)


# run

print("\nRunning aggregation comparison...")
results_df = run_aggregation_comparison(test_df, train_scores)
print_summary(results_df)
plot_comparison(results_df)
show_prediction_examples(test_df, train_scores, k=100, n_examples=5)
print("Total unique items in test:", test_df["item_id"].nunique())

# Number of items voted on by each top-10 user.
top10 = train_scores.sort_values("VAR_score", ascending=False).head(10)
for _, row in top10.iterrows():
    n = test_df[test_df["username"] == row["username"]]["item_id"].nunique()
    print(f"  {row['username']:<25} VAR={row['VAR_score']:.3f}  items_voted={n}")


# MULTI-SPLIT BENCHMARK  (uses splits produced by prepare_data_step1)
#
# Everything above runs VAR on its own ad-hoc 80/20 temporal split of the raw
# CSV (with its own iterative density cleanup and a train intersection test user
# restriction), and picks the best K per strategy by looking directly at
# macro_f1 on the TEST set itself - a real leakage, since the same data used
# to pick the hyperparameter is also the one reported as the result.
#
# This section instead evaluates VAR on every split already produced by
# prepare_data_step1 (splits/, splits_full/, splits_intersection/,
# windowed_folds_full/*, windowed_folds_intersection/*):
#   - VAR scores are computed on TRAIN only (calculate_var_scores)
#   - the best (strategy, selection-mode, K) combination is chosen by
#     evaluating on VAL
#   - final metrics are reported on TEST, using that fixed combination
#     (never re-selected against TEST)
#
# It does not touch or re-run anything above; it's purely additive.
#
# NOTE: evaluate_splits() also exports per-split detail files,
# not just metrics.json - this was the one method with zero item/user-level
# output, which made it impossible to join VAR results with user metadata
# for the T1 (user-characteristics) disaggregation analysis. The two new
# files, saved alongside metrics.json for every split:
#
#   voter_scores.parquet  - one row per (item_id, username) among the
#                            top-K voters actually used for that item's
#                            decision (using the winning strategy/mode/K
#                            chosen on VAL), with their VAR_score and raw
#                            vote. This is VAR's equivalent of SEF's
#                            weights.parquet: it tells you WHICH users
#                            drove which decisions, so it can be joined
#                            with karma/tenure/whatever user table.
#   item_scores.parquet   - one row per test item_id with the final
#                            predicted label and ground-truth label
#                            (community/item_id/predicted/label).
#
# Nothing about scoring, K-selection, or the VAL/TEST separation logic
# was changed - this only adds a write step after the winning config is
# already known.

VOTES_DIR_DEFAULT      = Path("data/splits/reddit")
VAR_MULTI_SPLIT_OUTPUT = OUTPUT_ROOT
MIN_CAL_FALLBACK_ITEMS = 50


def _fallback_val_from_train(split_label: str, train_df: pd.DataFrame):
    """Create val from train when validation data is unavailable."""
    train_items = train_df["item_id"].unique()
    if len(train_items) == 0:
        return None
    rng = np.random.RandomState(10)
    sample_size = min(len(train_items), max(MIN_CAL_FALLBACK_ITEMS, int(0.2 * len(train_items))))
    sample_ids = rng.choice(train_items, size=sample_size, replace=False)
    return train_df[train_df["item_id"].isin(sample_ids)]


def run_single_split_var(split_label: str, train_df: pd.DataFrame,
                          val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Run the single split var workflow."""
    if test_df.empty or test_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty test set - skipped.")
        return None, None, None

    val_for_cal = val_df
    if val_df.empty or val_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty val split - falling back to a sample carved from TRAIN.")
        val_for_cal = _fallback_val_from_train(split_label, train_df)
        if val_for_cal is None:
            print(f"  [{split_label}] no train items either - skipped.")
            return None, None, None

    print(f"  [{split_label}] Training VAR on {train_df['item_id'].nunique()} train items...")
    scores = calculate_var_scores(train_df)

    selection_modes = {"per_community": _base_votes, "per_item": _base_votes_per_item}

    best_cfg, best_f1_val = None, -1.0
    for agg_name, agg_fn in AGGREGATIONS.items():
        for mode_name, base_fn in selection_modes.items():
            for k in K_VALUES:
                val_votes = base_fn(val_for_cal, scores, k)
                if val_votes.empty:
                    continue
                val_decisions = agg_fn(val_votes)
                if val_decisions["label"].nunique() < 2:
                    continue
                val_metrics = evaluate_fold(val_decisions)
                if val_metrics["macro_f1"] > best_f1_val:
                    best_f1_val = val_metrics["macro_f1"]
                    best_cfg = {"strategy": agg_name, "mode": mode_name, "k": k}

    if best_cfg is None:
        print(f"  [{split_label}] no valid (strategy, K) found on VAL - skipped.")
        return None, None, None

    agg_fn  = AGGREGATIONS[best_cfg["strategy"]]
    base_fn = selection_modes[best_cfg["mode"]]

    test_votes     = base_fn(test_df, scores, best_cfg["k"])
    test_decisions = agg_fn(test_votes)

    # "splits_full" only: force full test coverage. If none of the top-K
    # selected users voted on an item, that item is silently absent from
    # test_decisions (a real coverage gap for VAR). Add it back using the
    # item's net vote (over ALL voters in this split's test set, not just
    # the top-K), instead of dropping it from evaluation.
    n_fallback = 0
    if split_label == "splits_full":
        covered_ids = set(test_decisions["item_id"].unique()) if not test_decisions.empty else set()
        all_ids     = set(test_df["item_id"].unique())
        missing_ids = all_ids - covered_ids
        if missing_ids:
            n_fallback = len(missing_ids)
            missing_votes = test_df[test_df["item_id"].isin(missing_ids)]
            fallback_rows = (
                missing_votes.groupby(["community", "item_id"])
                .agg(predicted=("vote", lambda x: 1 if x.mean() >= 0 else -1),
                     label=("label", "first"))
                .reset_index()
            )
            test_decisions = pd.concat([test_decisions, fallback_rows], ignore_index=True)

    if test_decisions["label"].nunique() < 2:
        print(f"  [{split_label}] single class in test - skipped.")
        return None, None, None
    test_metrics = evaluate_fold(test_decisions)  # single pass over the whole TEST set, no sub-bucketing

    test_metrics.update({
        "split":         split_label,
        "strategy":      best_cfg["strategy"],
        "mode":          best_cfg["mode"],
        "k":             best_cfg["k"],
        "val_macro_f1":  best_f1_val,
        "n_train_items": int(train_df["item_id"].nunique()),
        "n_val_items":   int(val_df["item_id"].nunique()),
        "n_test_items":  int(test_df["item_id"].nunique()),
    })
    if split_label == "splits_full" and n_fallback > 0:
        test_metrics["used_net_vote_fallback"] = True
        test_metrics["n_fallback_items"]       = n_fallback

    print(
        f"  [{split_label}] best on VAL: {best_cfg['strategy']}_{best_cfg['mode']} k={best_cfg['k']} "
        f"(val macro_f1={best_f1_val:.3f}) -> TEST macro_f1={test_metrics['macro_f1']:.3f} "
        f"roc_auc={test_metrics['roc_auc']:.3f}"
        + (f" | fallback_items={n_fallback}" if split_label == "splits_full" else "")
    )

    # stash the pieces needed for the detail export so step_multi_split
    # doesn't have to recompute base_fn/agg_fn a second time
    test_metrics["_detail_test_votes"]     = test_votes
    test_metrics["_detail_test_decisions"] = test_decisions

    return best_cfg, test_metrics, scores


def evaluate_splits(votes_dir: Path = VOTES_DIR_DEFAULT,
                      output_dir: Path = VAR_MULTI_SPLIT_OUTPUT) -> dict:
    """Evaluate splits and return its metrics."""
    splits = discover_splits(votes_dir)
    if not splits:
        print(f"No split directories found under {votes_dir} "
              f"(expected splits/, splits_full/, splits_intersection/, "
              f"windowed_folds_full/w*, windowed_folds_intersection/w*)")
        return {}

    print(f"Found {len(splits)} split(s): {list(splits.keys())}")
    summary = {}

    for split_name, split_path in splits.items():
        print(f"\n--- split: {split_name} ---")
        train_df, val_df, test_df = load_vote_partitions(split_path)
        best_cfg, metrics, scores = run_single_split_var(split_name, train_df, val_df, test_df)
        if metrics is None:
            continue

        # pull out the detail frames stashed by run_single_split_var, and
        # strip them from the dict before it gets json.dump'd
        test_votes     = metrics.pop("_detail_test_votes")
        test_decisions = metrics.pop("_detail_test_decisions")

        split_out_dir = output_dir / split_name / "var"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        # voter_scores.parquet: who voted, with what VAR_score, on which
        # item - the join key for T1 user-characteristics analysis.
        voter_cols = [c for c in ["community", "item_id", "username", "vote", "VAR_score"]
                      if c in test_votes.columns]
        test_votes[voter_cols].to_parquet(split_out_dir / "voter_scores.parquet", index=False)

        # item_scores.parquet: final decision per test item.
        item_cols = [c for c in ["community", "item_id", "predicted", "label"]
                     if c in test_decisions.columns]
        test_decisions[item_cols].to_parquet(split_out_dir / "item_scores.parquet", index=False)

        # all_user_var_scores.parquet: the FULL per-(community, username)
        # VAR_score table, not top-K filtered - see docstring above for why
        # this (and not voter_scores.parquet) is the right file for T1.
        scores.to_parquet(split_out_dir / "all_user_var_scores.parquet", index=False)

        print(f"  Saved -> {split_out_dir} "
              f"(metrics.json, voter_scores.parquet [{len(test_votes)} rows], "
              f"item_scores.parquet [{len(test_decisions)} rows], "
              f"all_user_var_scores.parquet [{len(scores)} rows, unfiltered])")

        summary[split_name] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_splits_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary of all splits saved -> {summary_path}")

    if summary:
        print("\n=== Cross-split comparison ===")
        for name, m in summary.items():
            print(f"  {name:35s} {m['strategy']}_{m['mode']} k={m['k']:<3} "
                  f"macro_F1={m['macro_f1']:.4f}  AUC={m['roc_auc']:.4f}  n_test={m['n_test_items']}")

    return summary


print("\nRunning multi-split VAR benchmark...")
evaluate_splits()
