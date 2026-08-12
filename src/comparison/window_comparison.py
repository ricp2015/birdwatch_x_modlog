from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.results import parse_result_path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RUN_LABELS = {
    "reddit_cn":                     "CN",
    "BL1_net_score":                 "Net",
    "BL2_weighted_net_score":        "WNet",
    "BL3_subreddit_adjusted":        "SubAdj",
    "BL4_reddit_score":              "RedditScore",
    "BL5_subreddit":                 "RedditScore+Sub",
    "expertise_ranker_k10":          "ExpertTop10",
    "team_formation":                "TeamFormation",
    "embeddings":                    "Embeddings",
    "reddit_cn_inverted":            "CN_inverted",
    "step3_expert":                  "Embeddings",
    "VAR":                           "ExpertTop10",
    "BL1_net_vote":                  "Net",
    "BL2_net_vote_alpha":            "WNet",
    "BL3_net_vote_alpha_subreddit":  "SubAdj",
}

COLORS = {
    "Embeddings":      "#e63946",
    "RedditScore+Sub": "#457b9d",
    "SubAdj":          "#2a9d8f",
    "TeamFormation":   "#f4a261",
    "ExpertTop10":     "#e67e22",
    "Net":             "#9f7abd",
    "WNet":            "#185116",
    "CN":              "#ab2530",
    "CN_inverted":     "#cdd24f",
    "RedditScore":     "#53107a",
}

HIGHLIGHT = {"Embeddings", "RedditScore+Sub", "SubAdj", "TeamFormation", "ExpertTop10"}

ZORDER = {
    "Embeddings":      10,
    "RedditScore+Sub": 5,
    "SubAdj":          5,
    "TeamFormation":   5,
    "ExpertTop10":     4,
}

LINEWIDTH = {"Embeddings": 2.5}

METRICS = ["f1_neg"]
METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "f1_pos":   "F1 (approve)",
    "f1_neg":   "F1 (remove)",
    "roc_auc":  "ROC AUC",
}

BY_SPLIT_SUFFIX      = "_by_split"
TARGET_FAMILY        = "windows/intersection"
TWO_PART_SPLIT_ROOTS = {"windowed_folds_full", "windowed_folds_intersection"}


def collect_windowed(roots: List[Path], metrics: List[str]) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Collect windowed from the available records."""
    data: Dict[str, Dict[str, Dict[int, float]]] = {m: {} for m in metrics}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("metrics.json"):
            parsed = parse_result_path(p, root)
            if parsed is None:
                continue
            split_name, run_key = parsed
            if not split_name.startswith(TARGET_FAMILY + "/"):
                continue
            window_tag = split_name.split("/")[-1]
            if not window_tag.startswith("w"):
                continue
            try:
                window_pct = int(window_tag[1:])
            except ValueError:
                continue
            m_json = json.loads(p.read_text())
            label  = RUN_LABELS.get(run_key, run_key)
            for metric in metrics:
                value = m_json.get(metric)
                if value is None:
                    continue
                data[metric].setdefault(label, {}).setdefault(window_pct, float(value))
    return data


def _plot_one_axis(ax, method_data: Dict[str, Dict[int, float]],
                   title: str, ylabel: str, show_ylabel: bool,
                   show_legend: bool) -> None:
    """Plot one axis and save the figure."""
    all_windows = sorted({w for m in method_data.values() for w in m})

    def sort_key(label):
        """Return the numeric window sort key."""
        vals   = list(method_data[label].values())
        spread = max(vals) - min(vals) if vals else 0
        mean   = np.mean(vals) if vals else 0
        return (-int(label == "Embeddings"), -spread, -mean)

    methods_sorted = sorted(method_data.keys(), key=sort_key)
    all_values: List[float] = []

    for label in methods_sorted:
        window_map = method_data[label]
        xs = sorted(window_map)
        ys = [window_map[x] for x in xs]
        all_values.extend(ys)

        color = COLORS.get(label, "#888888")
        lw    = LINEWIDTH.get(label, 1.4)
        zo    = ZORDER.get(label, 2)
        alpha = 1.0 if label in HIGHLIGHT else 0.45
        ls    = "-" if label in HIGHLIGHT else "--"

        ax.plot(xs, ys, marker="o", markersize=4,
                color=color, linewidth=lw, linestyle=ls,
                zorder=zo, alpha=alpha, label=label)

    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=7.5,
            framealpha=0.85,
            ncol=2,
            borderpad=0.5,
            handlelength=1.5,
        )

    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Training set size (%)", fontsize=9)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9)
    if all_windows:
        ax.set_xticks(all_windows)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
    if all_values:
        lo, hi = min(all_values), max(all_values)
        margin = max((hi - lo) * 0.35, 0.03)
        ax.set_ylim(lo - margin, hi + margin)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


def plot_window_metrics(data: Dict[str, Dict[str, Dict[int, float]]],
                        metrics: List[str], out: Path) -> None:
    """Plot window metrics and save the figure."""
    metrics_present = [m for m in metrics if data.get(m)]
    if not metrics_present:
        print("No windowed intersection data found.")
        return

    n_rows = len(metrics_present)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 4.2 * n_rows), squeeze=False)

    for row, metric in enumerate(metrics_present):
        ax = axes[row][0]
        _plot_one_axis(
            ax, data[metric],
            title="" if row == 0 else "",
            ylabel=METRIC_LABELS.get(metric, metric),
            show_ylabel=True,
            show_legend=(row == 0),   # legend only on first subplot
        )

    fig.suptitle("Performance vs training set size", fontsize=13, y=1.005)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path,
                        default=[Path("results/reddit")])
    parser.add_argument("--metrics", nargs="+", type=str, default=METRICS)
    parser.add_argument("--out", type=Path,
                        default=Path("results/window_metrics_intersection.png"))
    args = parser.parse_args()
    data = collect_windowed(args.roots, metrics=args.metrics)
    plot_window_metrics(data, args.metrics, args.out)


if __name__ == "__main__":
    main()
