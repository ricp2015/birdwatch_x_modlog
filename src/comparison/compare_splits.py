"""
Minimal benchmark comparison. One table per split, all methods x key metrics.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.results import parse_result_path

import numpy as np
import pandas as pd

METRICS = ["macro_f1", "roc_auc", "f1_pos", "f1_neg"]

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

BY_SPLIT_SUFFIX = "_by_split"
TWO_PART_SPLIT_ROOTS = {"windowed_folds_full", "windowed_folds_intersection"}


def load_metrics(path: Path) -> Dict:
    """Load metrics from its configured source."""
    with open(path) as f:
        return json.load(f)


def collect_by_split(results_roots: List[Path]) -> Dict[str, pd.DataFrame]:
    """Collect by split from the available records."""
    per_split_rows: Dict[str, List[Dict]] = {}
    for root in results_roots:
        if not root.exists():
            continue
        for p in root.rglob("metrics.json"):
            parsed = parse_result_path(p, root)
            if parsed is None:
                continue
            split_name, run_key = parsed
            if any("wikipedia" in part for part in p.parent.relative_to(root).parts):
                continue
            m = load_metrics(p)
            label = RUN_LABELS.get(run_key, run_key)
            row = {"run": label}
            for k in METRICS:
                row[k]          = m.get(k, None)
                row[k + "_std"] = m.get(k + "_std", None)
            per_split_rows.setdefault(split_name, []).append(row)

    out = {}
    for split_name, rows in per_split_rows.items():
        df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        out[split_name] = df
    return out


def build_report(per_split_df: Dict[str, pd.DataFrame]) -> str:
    """Build report from the supplied data."""
    lines = []
    for split_name, df in sorted(per_split_df.items()):
        if df.empty:
            continue

        # header
        lines.append(f"\n=== {split_name} ===\n")
        lines.append(f"  {'Method':<18}" + "".join(f"{m:>16}" for m in METRICS))
        lines.append("  " + "-" * (18 + 16 * len(METRICS)))

        # best value per metric for starring
        best = {m: df[m].max() for m in METRICS}

        for _, row in df.iterrows():
            line = f"  {row['run']:<18}"
            for m in METRICS:
                val = row.get(m)
                std = row.get(m + "_std")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    line += f"{'-':>16}"
                else:
                    star = "*" if abs(val - best[m]) < 1e-6 else " "
                    s = f"{val:.3f}"
                    if std and not np.isnan(std):
                        s += f"+/-{std:.3f}"
                    line += f"{star}{s:>15}"
            lines.append(line)

        # gap row: max - min per metric
        lines.append("  " + "-" * (18 + 16 * len(METRICS)))
        gap_line = f"  {'gap (max-min)':<18}"
        for m in METRICS:
            col = df[m].dropna()
            g = col.max() - col.min() if not col.empty else float("nan")
            gap_line += f"{g:>16.3f}" if not np.isnan(g) else f"{'-':>16}"
        lines.append(gap_line)

    return "\n".join(lines)


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path,
                        default=[Path("results/reddit")])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    per_split_df = collect_by_split(args.roots)
    if not per_split_df:
        print("No per-split data found.")
        sys.exit(1)

    report = build_report(per_split_df)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
