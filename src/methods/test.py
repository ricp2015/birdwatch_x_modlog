"""
normvio_pairplot.py
===================
Disaggregated analysis of NormVio scores vs moderator decisions.
Produces:
  1. Pairplot: score distributions per category, coloured by label
  2. Boxplot: score per category x label side by side
  3. Barplot: mean score difference (removed - approved) per category

Reads:
    results/step2/team_formation/post_violation_scores.parquet
    data/processed/final_intersection_dataset.csv

Output:
    results/step2/team_formation/normvio_pairplot.png
    results/step2/team_formation/normvio_boxplot.png
    results/step2/team_formation/normvio_barplot.png

Usage:
    python normvio_pairplot.py
    pip install matplotlib seaborn  (if missing)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCORES_PATH = Path("results/step2/team_formation/post_violation_scores.parquet")
CSV_PATH    = Path("data/processed/final_intersection_dataset.csv")
OUT_DIR     = Path("results/step2/team_formation")

CATEGORIES = [
    "spam", "meta-rules", "content", "doxxing",
    "harassment", "hatespeech", "format",
    "off-topic", "trolling", "incivility",
]

# subset for pairplot (keeps it readable and fast)
PLOT_CATS = ["spam", "content", "hatespeech", "off-topic", "incivility", "harassment"]


def load_data() -> pd.DataFrame:
    scores = pd.read_parquet(SCORES_PATH)
    labels = (
        pd.read_csv(CSV_PATH)[["item_id", "label"]]
        .dropna(subset=["label"])
        .drop_duplicates("item_id")
    )
    df = scores.merge(labels, on="item_id", how="inner")
    df["label_str"] = df["label"].map({1: "approved", -1: "removed"})
    print(f"Loaded {len(df):,} posts  "
          f"({(df['label']==-1).sum():,} removed, {(df['label']==1).sum():,} approved)")
    return df


def make_pairplot(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter matrix of NormVio scores, coloured by moderation label."""
    rename_map = {f"score_{c}": c for c in PLOT_CATS}
    plot_df = df[["label_str"] + [f"score_{c}" for c in PLOT_CATS]].copy()
    plot_df = plot_df.rename(columns=rename_map)

    # sample down — pairplot on 140k points is very slow
    # sample each group separately then concat to preserve label_str column
    approved = plot_df[plot_df["label_str"] == "approved"].sample(
        min((plot_df["label_str"] == "approved").sum(), 3000), random_state=42)
    removed  = plot_df[plot_df["label_str"] == "removed"].sample(
        min((plot_df["label_str"] == "removed").sum(),  3000), random_state=42)
    sample = pd.concat([approved, removed], ignore_index=True)

    palette = {"approved": "#2196F3", "removed": "#F44336"}
    g = sns.pairplot(
        sample,
        hue="label_str",
        vars=PLOT_CATS,
        plot_kws={"alpha": 0.15, "s": 8, "rasterized": True},
        diag_kind="kde",
        palette=palette,
        corner=True,
    )
    g.figure.suptitle(
        "NormVio score distributions by category\n(blue=approved, red=removed)",
        y=1.01, fontsize=13,
    )
    g.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Pairplot saved -> {out_path}")


def make_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of scores for all 10 categories, split by approved/removed."""
    rows = []
    for cat in CATEGORIES:
        col = f"score_{cat}"
        tmp = df[["label_str", col]].dropna().copy()
        tmp["category"] = cat
        tmp = tmp.rename(columns={col: "score", "label_str": "label"})
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    palette = {"approved": "#2196F3", "removed": "#F44336"}
    sns.boxplot(
        data=long_df,
        x="category", y="score",
        hue="label",
        palette=palette,
        showfliers=False,
        width=0.6,
        ax=ax,
    )
    ax.set_title(
        "NormVio score distribution per category x moderation label\n(outliers hidden)",
        fontsize=12,
    )
    ax.set_xlabel("NormVio category")
    ax.set_ylabel("Violation probability score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Moderator decision")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Boxplot saved -> {out_path}")


def make_barplot(df: pd.DataFrame, out_path: Path) -> None:
    """Mean score per category x label, and their difference."""
    means = []
    for cat in CATEGORIES:
        col = f"score_{cat}"
        m_rem = float(df.loc[df["label"] == -1, col].mean())
        m_app = float(df.loc[df["label"] ==  1, col].mean())
        means.append({
            "category": cat,
            "removed":  m_rem,
            "approved": m_app,
            "diff":     m_rem - m_app,
        })
    mdf = pd.DataFrame(means).set_index("category")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left panel: mean scores side by side
    mdf[["removed", "approved"]].plot(
        kind="bar", ax=axes[0],
        color=["#F44336", "#2196F3"],
        width=0.7, edgecolor="white",
    )
    axes[0].set_title("Mean NormVio score per category x label")
    axes[0].set_ylabel("Mean violation probability")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(title="label")

    # right panel: difference removed - approved
    colors = ["#4CAF50" if v > 0 else "#FF5722" for v in mdf["diff"]]
    mdf["diff"].plot(kind="bar", ax=axes[1], color=colors, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(
        "Score difference: removed - approved\n(green = model aligns with moderators)"
    )
    axes[1].set_ylabel("Delta mean score")
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Barplot saved -> {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    make_pairplot(df, OUT_DIR / "normvio_pairplot.png")
    make_boxplot( df, OUT_DIR / "normvio_boxplot.png")
    make_barplot( df, OUT_DIR / "normvio_barplot.png")
    print("\nAll plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()