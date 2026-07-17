"""
plot_window_f1.py
=================
Line plot of macro F1 vs window size for all methods.
One figure with two subplots: windowed_folds_full and windowed_folds_intersection.

Usage:
    python plot_window_f1.py
    python plot_window_f1.py --roots results/step2 results/step3_expert --out results/window_f1.png
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Same constants as compare scripts
# ---------------------------------------------------------------------------

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
    "step3_expert":                  "SEF",
    "VAR":                           "VAR",
    "BL1_net_vote":                  "Net",
    "BL2_net_vote_alpha":            "WNet",
    "BL3_net_vote_alpha_subreddit":  "SubAdj",
}

COLORS = {
    "SEF":             "#e63946",
    "RedditScore+Sub": "#457b9d",
    "SubAdj":          "#2a9d8f",
    "TeamFormation":   "#f4a261",
    "VAR":             "#8338ec",
    "Net":             "#adb5bd",
    "WNet":            "#ced4da",
    "CN":              "#6c757d",
    "CN_inverted":     "#adb5bd",
    "RedditScore":     "#dee2e6",
}

ZORDER = {
    "SEF": 10,
    "RedditScore+Sub": 5,
    "SubAdj": 5,
    "TeamFormation": 5,
    "VAR": 4,
}

LINEWIDTH = {
    "SEF": 2.5,
}

BY_SPLIT_SUFFIX = "_by_split"
TWO_PART_SPLIT_ROOTS = {"windowed_folds_full", "windowed_folds_intersection"}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _parse_split_and_run(metrics_path: Path, root: Path) -> Optional[Tuple[str, str]]:
    try:
        rel_parts = metrics_path.parent.relative_to(root).parts
    except ValueError:
        return None
    for i, part in enumerate(rel_parts):
        if not part.endswith(BY_SPLIT_SUFFIX):
            continue
        after = rel_parts[i + 1:]
        if not after:
            return None
        if after[0] in TWO_PART_SPLIT_ROOTS:
            if len(after) < 2:
                return None
            split_name = f"{after[0]}/{after[1]}"
            remainder = after[2:]
        else:
            split_name = after[0]
            remainder = after[1:]
        if remainder:
            run_key = remainder[0]
        elif i > 0:
            run_key = rel_parts[i - 1]
        elif part != "benchmark_by_split":
            run_key = part[: -len(BY_SPLIT_SUFFIX)]
        else:
            run_key = root.name
        return split_name, run_key
    return None


def collect_windowed(
    roots: List[Path],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Returns {family: {method: {window_pct: macro_f1}}}
    family ∈ {"windowed_folds_full", "windowed_folds_intersection"}
    window_pct ∈ {20, 40, 60, 80, 100}
    """
    data: Dict[str, Dict[str, Dict[int, float]]] = {
        "windowed_folds_full": {},
        "windowed_folds_intersection": {},
    }

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("metrics.json"):
            parsed = _parse_split_and_run(p, root)
            if parsed is None:
                continue
            split_name, run_key = parsed

            # only windowed splits
            matched_family = None
            for fam in data:
                if split_name.startswith(fam + "/"):
                    matched_family = fam
                    break
            if matched_family is None:
                continue

            window_tag = split_name.split("/")[-1]  # e.g. "w020"
            if not window_tag.startswith("w"):
                continue
            try:
                window_pct = int(window_tag[1:])
            except ValueError:
                continue

            m = json.loads(p.read_text())
            macro_f1 = m.get("macro_f1")
            if macro_f1 is None:
                continue

            label = RUN_LABELS.get(run_key, run_key)
            if label not in data[matched_family]:
                data[matched_family][label] = {}
            # if duplicate, keep the first
            data[matched_family][label].setdefault(window_pct, float(macro_f1))

    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_window_f1(
    data: Dict[str, Dict[str, Dict[int, float]]],
    out:  Path,
) -> None:
    families = [f for f in ["windowed_folds_full", "windowed_folds_intersection"]
                if data.get(f)]

    if not families:
        print("No windowed data found.")
        return

    fig, axes = plt.subplots(
        1, len(families),
        figsize=(6 * len(families), 5),
        sharey=True,
    )
    if len(families) == 1:
        axes = [axes]

    titles = {
        "windowed_folds_full":         "Full dataset",
        "windowed_folds_intersection": "Intersection dataset",
    }

    for ax, family in zip(axes, families):
        method_data = data[family]
        all_windows = sorted({w for m in method_data.values() for w in m})

        # sort methods: SEF first, then by mean f1 desc, flat methods last
        def sort_key(label):
            vals = list(method_data[label].values())
            spread = max(vals) - min(vals) if vals else 0
            mean   = np.mean(vals) if vals else 0
            return (-int(label == "SEF"), -spread, -mean)

        methods_sorted = sorted(method_data.keys(), key=sort_key)

        for label in methods_sorted:
            window_map = method_data[label]
            xs = sorted(window_map)
            ys = [window_map[x] for x in xs]

            color = COLORS.get(label, "#888888")
            lw    = LINEWIDTH.get(label, 1.4)
            zo    = ZORDER.get(label, 2)
            alpha = 1.0 if label in ZORDER else 0.55
            ls    = "-" if label in ZORDER else "--"

            ax.plot(xs, ys, marker="o", markersize=4,
                    color=color, linewidth=lw, linestyle=ls,
                    zorder=zo, alpha=alpha, label=label)

            # label at end of line
            if xs:
                ax.annotate(
                    label,
                    xy=(xs[-1], ys[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    va="center",
                    alpha=alpha,
                )

        ax.set_title(titles.get(family, family), fontsize=11, pad=8)
        ax.set_xlabel("Window size (%)", fontsize=9)
        ax.set_ylabel("Macro F1" if ax == axes[0] else "", fontsize=9)
        ax.set_xticks(all_windows)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.set_ylim(0.40, 0.82)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    fig.suptitle("Macro F1 vs window size", fontsize=12, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path,
                        default=[Path("results/step2"), Path("results/step3_expert")])
    parser.add_argument("--out", type=Path, default=Path("results/window_f1.png"))
    args = parser.parse_args()

    data = collect_windowed(args.roots)
    plot_window_f1(data, args.out)


if __name__ == "__main__":
    main()