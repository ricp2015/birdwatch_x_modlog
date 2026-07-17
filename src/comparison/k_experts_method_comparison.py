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


# ============================================================================
# MULTI-SPLIT BENCHMARK  (uses splits produced by prepare_data_step1)
# ============================================================================
#
# Everything above runs VAR on its own ad-hoc 80/20 temporal split of the raw
# CSV (with its own iterative density cleanup and a train∩test user
# restriction), and picks the best K per strategy by looking directly at
# macro_f1 on the TEST set itself — a real leakage, since the same data used
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

VOTES_DIR_DEFAULT      = Path("results/step1/reddit")
VAR_MULTI_SPLIT_OUTPUT = OUTPUT_ROOT / "VAR_by_split"
MIN_CAL_FALLBACK_ITEMS = 50


def discover_splits(votes_dir: Path) -> dict:
    """
    Find every split directory produced by prepare_data_step1 under votes_dir,
    i.e. any folder containing train_votes.parquet / val_votes.parquet /
    test_votes.parquet.

    Returns a dict: split_label -> directory Path, where split_label is
    e.g. "splits", "splits_full", "splits_intersection",
    "windowed_folds_full/w020", "windowed_folds_intersection/w100", ...
    """
    found = {}
    for name in ("splits", "splits_full", "splits_intersection"):
        d = votes_dir / name
        if (d / "train_votes.parquet").exists():
            found[name] = d
    for tag in ("windowed_folds_full", "windowed_folds_intersection"):
        root = votes_dir / tag
        if root.exists():
            for w_dir in sorted(root.glob("w*")):
                if (w_dir / "train_votes.parquet").exists():
                    found[f"{tag}/{w_dir.name}"] = w_dir
    return found


def load_split_data(split_dir: Path):
    """Load one split's train/val/test parquets (no cleanup re-applied — the
    splits from prepare_data_step1 are already density-filtered)."""
    train = pd.read_parquet(split_dir / "train_votes.parquet")
    val   = pd.read_parquet(split_dir / "val_votes.parquet")
    test  = pd.read_parquet(split_dir / "test_votes.parquet")
    return train, val, test


def _fallback_val_from_train(split_label: str, train_df: pd.DataFrame):
    """Carve a calibration sample out of TRAIN when VAL is empty (e.g.
    windowed_folds at w=100%, where the whole pool goes to train)."""
    train_items = train_df["item_id"].unique()
    if len(train_items) == 0:
        return None
    rng = np.random.RandomState(42)
    sample_size = min(len(train_items), max(MIN_CAL_FALLBACK_ITEMS, int(0.2 * len(train_items))))
    sample_ids = rng.choice(train_items, size=sample_size, replace=False)
    return train_df[train_df["item_id"].isin(sample_ids)]


def run_single_split_var(split_label: str, train_df: pd.DataFrame,
                          val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train VAR scores on TRAIN, select the best (strategy, selection-mode, K)
    combination by evaluating on VAL, then report final metrics on TEST with
    that fixed combination.

    Returns (best_cfg: dict, metrics: dict) or (None, None) if the split has
    no usable data.
    """
    if test_df.empty or test_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty test set — skipped.")
        return None, None

    val_for_cal = val_df
    if val_df.empty or val_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty val split — falling back to a sample carved from TRAIN.")
        val_for_cal = _fallback_val_from_train(split_label, train_df)
        if val_for_cal is None:
            print(f"  [{split_label}] no train items either — skipped.")
            return None, None

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
        print(f"  [{split_label}] no valid (strategy, K) found on VAL — skipped.")
        return None, None

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
        print(f"  [{split_label}] single class in test — skipped.")
        return None, None
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
    return best_cfg, test_metrics


def step_multi_split(votes_dir: Path = VOTES_DIR_DEFAULT,
                      output_dir: Path = VAR_MULTI_SPLIT_OUTPUT) -> dict:
    """
    Run VAR on every split directory found under votes_dir (splits/,
    splits_full/, splits_intersection/, and every window under
    windowed_folds_full/ and windowed_folds_intersection/).

    Results are saved under output_dir/<split_label>/metrics.json, mirroring
    the layout used by the other methods' multi-split benchmarks, plus a
    final all_splits_summary.json comparing every split.
    """
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
        train_df, val_df, test_df = load_split_data(split_path)
        best_cfg, metrics = run_single_split_var(split_name, train_df, val_df, test_df)
        if metrics is None:
            continue

        split_out_dir = output_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"  Saved → {split_out_dir}")

        summary[split_name] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_splits_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary of all splits saved → {summary_path}")

    if summary:
        print("\n=== Cross-split comparison ===")
        for name, m in summary.items():
            print(f"  {name:35s} {m['strategy']}_{m['mode']} k={m['k']:<3} "
                  f"macro_F1={m['macro_f1']:.4f}  AUC={m['roc_auc']:.4f}  n_test={m['n_test_items']}")

    return summary


print("\nRunning multi-split VAR benchmark...")
step_multi_split()