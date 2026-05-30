"""
analyze_signals.py
==================
Analyses discriminability of signals A, B, C with respect to moderator agreement.

For each (U, P) pair with computed expert weights, checks:
  1. Correlation between each signal and concordance with moderator
  2. Discriminability: do high-signal users agree more with moderators?
  3. Breakdown by vote direction (+1 / -1)
  4. Interaction: signal x vote_direction vs concordance

Requires item_scores.parquet from step3 run, plus the original votes CSV.

Usage:
    python analyze_signals.py
    python analyze_signals.py --scores-path results/step3_expert/item_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR   = Path("results/step3_expert")
ORIGINAL_CSV = Path("data/processed/final_intersection_dataset.csv")
N_QUANTILES  = 4   # quartiles for discriminability analysis


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================

def load_data(scores_path: Path, votes_path: Path) -> pd.DataFrame:
    """
    Merge expert weights (from item_scores.parquet) with ground truth labels
    and vote direction. Returns one row per (post, user) pair with columns:
      item_id, username, vote, expert_weight, local_prec, global_prec,
      signal_c, label, concordant
    """
    scores = pd.read_parquet(scores_path)
    log.info("item_scores: %d rows", len(scores))

    votes = pd.read_csv(votes_path, low_memory=False)
    votes["vote"]  = pd.to_numeric(votes["vote"],  errors="coerce")
    votes["label"] = pd.to_numeric(votes["label"], errors="coerce")
    votes = votes[votes["vote"].isin([1, -1]) & votes["label"].isin([1, -1])]
    votes = votes.dropna(subset=["item_id", "username", "vote", "label"])
    votes = votes.drop_duplicates(subset=["username", "item_id"])

    # expert weights are in item_scores only if the step3 code saved them per-user
    # if item_scores has no username column, we need to reload from weights directly
    if "username" not in scores.columns:
        raise ValueError(
            "item_scores.parquet does not contain per-user rows. "
            "Make sure compute_expert_weights output is saved. "
            "See note below."
        )

    # merge label and concordance
    label_map = votes.drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
    scores["label"] = scores["item_id"].map(label_map)
    scores = scores.dropna(subset=["label", "local_prec", "global_prec", "signal_c"])
    scores["label"] = scores["label"].astype(int)

    # concordant: did this user's vote match the moderator?
    scores["concordant"] = (scores["vote"] == scores["label"]).astype(int)

    log.info(
        "Pairs for analysis: %d | concordance rate: %.3f",
        len(scores), scores["concordant"].mean()
    )
    return scores


# ===========================================================================
# 2. CORRELATION ANALYSIS
# ===========================================================================

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson and Spearman correlation between each signal and concordance,
    overall and split by vote direction.
    """
    signals = ["local_prec", "global_prec", "signal_c", "expert_weight"]
    rows = []

    for subset_name, subset in [
        ("all",        df),
        ("vote=+1",    df[df["vote"] ==  1]),
        ("vote=-1",    df[df["vote"] == -1]),
    ]:
        for sig in signals:
            x = subset[sig].values
            y = subset["concordant"].values
            if len(x) < 10:
                continue
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            try:
                auc = roc_auc_score(y, x)
            except Exception:
                auc = float("nan")
            rows.append({
                "subset":   subset_name,
                "signal":   sig,
                "pearson_r":  round(r_p, 4),
                "pearson_p":  round(p_p, 4),
                "spearman_r": round(r_s, 4),
                "spearman_p": round(p_s, 4),
                "auc":        round(auc, 4),
                "n":          len(x),
            })

    result = pd.DataFrame(rows)
    log.info("\n=== CORRELATION: signal vs moderator concordance ===\n%s",
             result.to_string(index=False))
    return result


# ===========================================================================
# 3. DISCRIMINABILITY BY QUANTILE
# ===========================================================================

def discriminability_analysis(df: pd.DataFrame, n_quantiles: int = N_QUANTILES) -> pd.DataFrame:
    """
    Split users into quantiles by each signal. For each quantile, compute:
      - mean concordance rate
      - mean concordance for vote=+1 and vote=-1 separately
    Discriminability = concordance rate should increase monotonically with signal.
    """
    signals = ["local_prec", "global_prec", "signal_c", "expert_weight"]
    rows = []

    for sig in signals:
        try:
            df[f"{sig}_q"] = pd.qcut(df[sig], n_quantiles, labels=False, duplicates="drop")
        except Exception:
            continue

        for q in range(n_quantiles):
            sub = df[df[f"{sig}_q"] == q]
            rows.append({
                "signal":       sig,
                "quantile":     q + 1,
                "n":            len(sub),
                "concordance":  round(sub["concordant"].mean(), 4),
                "conc_vote+1":  round(sub[sub["vote"] ==  1]["concordant"].mean(), 4) if (sub["vote"] == 1).any()  else float("nan"),
                "conc_vote-1":  round(sub[sub["vote"] == -1]["concordant"].mean(), 4) if (sub["vote"] == -1).any() else float("nan"),
                "sig_mean":     round(sub[sig].mean(), 4),
            })

    result = pd.DataFrame(rows)
    log.info("\n=== DISCRIMINABILITY: concordance by signal quantile ===\n%s",
             result.to_string(index=False))
    return result


# ===========================================================================
# 4. INTERACTION: signal x vote_direction
# ===========================================================================

def interaction_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each combination of (signal_quantile, vote_direction), compute
    concordance rate. Tests whether high-signal users who vote -1 are more
    informative for the remove class than low-signal users.
    """
    signals = ["local_prec", "global_prec", "signal_c", "expert_weight"]
    rows = []

    for sig in signals:
        if f"{sig}_q" not in df.columns:
            try:
                df[f"{sig}_q"] = pd.qcut(df[sig], N_QUANTILES, labels=False, duplicates="drop")
            except Exception:
                continue

        for vote_dir in [1, -1]:
            sub = df[df["vote"] == vote_dir]
            for q in sorted(sub[f"{sig}_q"].dropna().unique()):
                qsub = sub[sub[f"{sig}_q"] == q]
                rows.append({
                    "signal":      sig,
                    "vote_dir":    vote_dir,
                    "quantile":    int(q) + 1,
                    "n":           len(qsub),
                    "concordance": round(qsub["concordant"].mean(), 4),
                })

    result = pd.DataFrame(rows)
    log.info("\n=== INTERACTION: signal x vote_direction vs concordance ===\n%s",
             result.to_string(index=False))
    return result


# ===========================================================================
# 5. PLOT
# ===========================================================================

def plot_all(
    corr_df:  pd.DataFrame,
    disc_df:  pd.DataFrame,
    inter_df: pd.DataFrame,
    out_path: Path,
) -> None:
    signals  = ["local_prec", "global_prec", "signal_c", "expert_weight"]
    sig_labels = {
        "local_prec":    "Signal A\n(local precision)",
        "global_prec":   "Signal B\n(subreddit precision)",
        "signal_c":      "Signal C\n(topical similarity)",
        "expert_weight": "Expert weight\n(combined)",
    }

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    colors = {1: "#3B8BD4", -1: "#D45B3B", "all": "#555555"}

    # Row 1: concordance by quantile (all votes)
    for i, sig in enumerate(signals):
        ax  = fig.add_subplot(gs[0, i])
        sub = disc_df[disc_df["signal"] == sig].sort_values("quantile")
        ax.bar(sub["quantile"], sub["concordance"], color=colors["all"], alpha=0.8)
        ax.set_title(sig_labels[sig], fontsize=9)
        ax.set_xlabel("Quantile (low→high)", fontsize=8)
        ax.set_ylabel("Concordance rate" if i == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(disc_df["concordance"].mean(), color="red", linestyle="--",
                   linewidth=0.8, label="mean")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    fig.text(0.01, 0.95, "Row 1: concordance by quantile (all votes)", fontsize=9, fontweight="bold")

    # Row 2: concordance by quantile split by vote direction
    for i, sig in enumerate(signals):
        ax   = fig.add_subplot(gs[1, i])
        sub  = disc_df[disc_df["signal"] == sig].sort_values("quantile")
        qs   = sub["quantile"].values
        w    = 0.35
        ax.bar(qs - w/2, sub["conc_vote+1"], width=w, label="+1", color=colors[1],  alpha=0.8)
        ax.bar(qs + w/2, sub["conc_vote-1"], width=w, label="-1", color=colors[-1], alpha=0.8)
        ax.set_title(sig_labels[sig], fontsize=9)
        ax.set_xlabel("Quantile", fontsize=8)
        ax.set_ylabel("Concordance rate" if i == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    fig.text(0.01, 0.65, "Row 2: concordance by quantile, split by vote direction", fontsize=9, fontweight="bold")

    # Row 3: AUC per signal per subset (bar chart)
    ax3 = fig.add_subplot(gs[2, :2])
    subsets   = ["all", "vote=+1", "vote=-1"]
    x         = np.arange(len(signals))
    bar_w     = 0.25
    sub_colors = ["#555555", "#3B8BD4", "#D45B3B"]
    for j, (sub_name, col) in enumerate(zip(subsets, sub_colors)):
        aucs = [
            corr_df[(corr_df["signal"] == sig) & (corr_df["subset"] == sub_name)]["auc"].values
            for sig in signals
        ]
        aucs = [a[0] if len(a) > 0 else float("nan") for a in aucs]
        ax3.bar(x + j * bar_w, aucs, width=bar_w, label=sub_name, color=col, alpha=0.8)
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random")
    ax3.set_xticks(x + bar_w)
    ax3.set_xticklabels([sig_labels[s].replace("\n", " ") for s in signals], fontsize=8)
    ax3.set_ylabel("AUC (signal → concordance)", fontsize=8)
    ax3.set_title("Row 3: AUC by signal and vote direction", fontsize=9, fontweight="bold")
    ax3.set_ylim(0.4, 1.0)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)

    # Row 3 right: Spearman r heatmap-style
    ax4 = fig.add_subplot(gs[2, 2:])
    pivot = corr_df.pivot(index="signal", columns="subset", values="spearman_r")
    im    = ax4.imshow(pivot.values, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_xticklabels(pivot.columns, fontsize=8)
    ax4.set_yticklabels([sig_labels[s].replace("\n", " ") for s in pivot.index], fontsize=8)
    ax4.set_title("Spearman r: signal vs concordance", fontsize=9, fontweight="bold")
    for ii in range(len(pivot.index)):
        for jj in range(len(pivot.columns)):
            val = pivot.values[ii, jj]
            if not np.isnan(val):
                ax4.text(jj, ii, f"{val:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax4, fraction=0.046)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved to %s", out_path)
    plt.close()


# ===========================================================================
# 6. SAVE RESULTS
# ===========================================================================

def save_results(
    corr_df:  pd.DataFrame,
    disc_df:  pd.DataFrame,
    inter_df: pd.DataFrame,
    out_dir:  Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "signal_correlations.csv",     index=False)
    disc_df.to_csv(out_dir / "signal_discriminability.csv", index=False)
    inter_df.to_csv(out_dir / "signal_interactions.csv",    index=False)
    log.info("CSVs saved to %s", out_dir)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal discriminability analysis")
    parser.add_argument("--scores-path", type=Path,
                        default=OUTPUT_DIR / "weights.parquet")
    parser.add_argument("--votes-path",  type=Path, default=ORIGINAL_CSV)
    parser.add_argument("--out-dir",     type=Path, default=OUTPUT_DIR / "signal_analysis")
    args = parser.parse_args()

    if not args.scores_path.exists():
        raise FileNotFoundError(
            f"{args.scores_path} not found.\n"
            "NOTE: this script requires that item_scores.parquet contains "
            "per-user rows with columns: item_id, username, vote, "
            "local_prec, global_prec, signal_c, expert_weight.\n"
            "In the current step3 code, item_scores only has per-post rows. "
            "To enable this analysis, save weights_df inside run_kfold before "
            "aggregation, e.g.:\n"
            "  weights_df['fold'] = fold_idx\n"
            "  all_weights.append(weights_df)\n"
            "and concatenate at the end."
        )

    df = load_data(args.scores_path, args.votes_path)

    corr_df  = correlation_analysis(df)
    disc_df  = discriminability_analysis(df)
    inter_df = interaction_analysis(df)

    save_results(corr_df, disc_df, inter_df, args.out_dir)
    plot_all(corr_df, disc_df, inter_df, args.out_dir / "signal_analysis.png")

    log.info("Done. Check %s/signal_analysis.png", args.out_dir)


if __name__ == "__main__":
    main()