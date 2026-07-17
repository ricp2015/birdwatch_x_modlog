from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]

RUN_LABELS = {
    # legacy single whole-dataset runs (kept for reference / other scripts)
    "reddit_cn":                     "CN",
    "BL1_net_score":                 "Net",
    "BL2_weighted_net_score":        "WNet",
    "BL3_subreddit_adjusted":        "SubAdj",
    "BL4_reddit_score":              "RedditScore",
    "BL5_subreddit":                 "RedditScore+Sub",
    "expertise_ranker_k10":          "ExpertTop10",
    "team_formation":                "TeamFormation",
    "embeddings":                     "Embeddings",
    # new per-split benchmark run keys (see team_formation_ranker.py,
    # step3_expert_finder.py, CN_wrapper.py, baselines.py, var_baseline.py)
    "step3_expert":                  "Embeddings",
    "VAR":                           "ExpertTop10",
    "BL1_net_vote":                  "Net",
    "BL2_net_vote_alpha":            "WNet",
    "BL3_net_vote_alpha_subreddit":  "SubAdj",
}

COLORS = {
    "CN":              "#0647b8",
    "Net":              "#0da25a",
    "WNet":             "#d8b226",
    "SubAdj":           "#b541e6",
    "RedditScore":      "#e67e22",
    "RedditScore+Sub":  "#c0392b",
    "ExpertTop10":      "#16a085",
    "TeamFormation":    "#f39c12",
    "Embeddings":        "#9b59b6",
}

# folder-name suffix that marks a "per-split benchmark" root, e.g.
# .../team_formation/benchmark_by_split/..., .../VAR_by_split/...
BY_SPLIT_SUFFIX = "_by_split"

# split-name path segments that need 2 path parts instead of 1
# (windowed_folds_full/w020, windowed_folds_intersection/w100, ...)
TWO_PART_SPLIT_ROOTS = {"windowed_folds_full", "windowed_folds_intersection"}


# load
def load_metrics(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _parse_split_and_run(metrics_path: Path, root: Path) -> Optional[Tuple[str, str]]:
    """
    Given the path to a metrics.json produced by one of the multi-split
    benchmarks (TFR/SEF/CN/baselines/VAR — anything saved under a folder
    whose name ends in "_by_split"), return (split_name, run_key).

    split_name is e.g. "splits", "splits_full", "splits_intersection",
    "windowed_folds_full/w020", ...

    run_key identifies the method/baseline and is looked up in RUN_LABELS
    for a display label. Handles three folder-layout patterns:
      - <method>/benchmark_by_split/<split>/metrics.json
            -> run_key = "<method>"                       (e.g. team_formation)
      - <method>/benchmark_by_split/<split>/<baseline>/metrics.json
            -> run_key = "<baseline>"                      (e.g. BL1_net_vote)
      - <METHOD>_by_split/<split>/metrics.json
            -> run_key = "<METHOD>"                        (e.g. VAR)
      - benchmark_by_split/<split>/metrics.json  (root IS the method's own dir)
            -> run_key = root.name                         (e.g. step3_expert)

    Returns None if this metrics.json doesn't belong to a per-split
    benchmark (e.g. an older single whole-dataset run — those are simply
    skipped by this script).
    """
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


def collect_by_split(results_roots: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Walk every directory in results_roots for metrics.json files that belong
    to a "*_by_split" benchmark tree, and group them into one DataFrame per
    split (e.g. "splits", "splits_full", "splits_intersection",
    "windowed_folds_full/w020", ...).

    Files that are NOT inside a "*_by_split" tree (older single whole-dataset
    runs) are ignored — this script now only compares per-split results.

    Returns {split_name: DataFrame}, each DataFrame with one row per run
    (method/baseline), sorted by macro_f1 descending.
    """
    per_split_rows: Dict[str, List[Dict]] = {}

    for root in results_roots:
        if not root.exists():
            continue
        for p in root.rglob("metrics.json"):
            parsed = _parse_split_and_run(p, root)
            if parsed is None:
                continue
            split_name, run_key = parsed

            # ignore wikipedia runs, same convention as before
            all_parts = p.parent.relative_to(root).parts
            if any("wikipedia" in part for part in all_parts):
                continue

            m = load_metrics(p)
            label = RUN_LABELS.get(run_key, run_key)

            row = {"run": label}
            for k in METRICS:
                row[k]          = m.get(k, None)
                row[k + "_std"] = m.get(k + "_std", None)

            per_split_rows.setdefault(split_name, []).append(row)

    per_split_df: Dict[str, pd.DataFrame] = {}
    for split_name, rows in per_split_rows.items():
        df = pd.DataFrame(rows)
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        per_split_df[split_name] = df
    return per_split_df


# print
def print_summary(df: pd.DataFrame, split_name: str) -> None:
    print(f"\n=== SUMMARY — split: {split_name} ===\n")
    print(df[["run", "macro_f1", "roc_auc", "f1_pos", "f1_neg"]])


# plot with CI
def plot_metric(df: pd.DataFrame, metric: str, split_name: str, out_dir: Path) -> None:
    plt.figure(figsize=(8, 4))

    colors = [COLORS.get(r, "gray") for r in df["run"]]

    y = df[metric]
    yerr = df.get(metric + "_std")

    # fallback se std non presente
    if yerr is None or yerr.isna().all():
        yerr = None

    plt.bar(
        df["run"],
        y,
        color=colors,
        yerr=yerr,
        capsize=5,
        alpha=0.9,
    )

    plt.title(f"{metric}")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)

    offset = 0.03
    for i, v in enumerate(y):
        if pd.notna(v):
            err = 0
            if yerr is not None and not pd.isna(yerr.iloc[i]):
                err = yerr.iloc[i]
            plt.text(i, v + err + offset, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_split = split_name.replace("/", "_")
    plt.savefig(out_dir / f"{metric}__{safe_split}.png", dpi=1000)
    plt.close()


# main
def compare_by_split(
    results_roots: List[Path],
    plots_dir:     Path = Path("results/step2/comparison_plots"),
) -> Dict[str, pd.DataFrame]:
    """
    Produce one summary + one set of plots PER SPLIT, across every method
    that has been benchmarked with the "*_by_split" layout (TFR, SEF, CN,
    baselines, VAR, ...).

    results_roots should list every top-level results directory that may
    contain a "*_by_split" tree — by default that's results/step2 (TFR, CN,
    baselines, VAR all save under here) and results/step3_expert (SEF saves
    directly under its own step folder).
    """
    per_split_df = collect_by_split(results_roots)

    if not per_split_df:
        print(
            "No per-split data found (looked for '*_by_split' folders under: "
            + ", ".join(str(r) for r in results_roots) + ")"
        )
        return {}

    print(f"Found {len(per_split_df)} split(s): {sorted(per_split_df.keys())}")

    for split_name, df in sorted(per_split_df.items()):
        if df.empty:
            continue
        print_summary(df, split_name)
        for metric in ["macro_f1", "roc_auc", "f1_pos", "f1_neg"]:
            plot_metric(df, metric, split_name, plots_dir)

    return per_split_df


if __name__ == "__main__":
    compare_by_split([Path("results/step2"), Path("results/step3_expert")])