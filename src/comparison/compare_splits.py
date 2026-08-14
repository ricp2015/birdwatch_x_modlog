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

CANONICAL_RUNS = [
    ("Net", "baselines/BL1_net_vote/metrics.json"),
    ("WNet", "baselines/BL2_net_vote_alpha/metrics.json"),
    ("SubAdj", "baselines/BL3_net_vote_alpha_subreddit/metrics.json"),
    ("RedditScore", "baselines/BL4_reddit_score/metrics.json"),
    ("RedditScore+Sub", "baselines/BL5_subreddit/metrics.json"),
    ("CN", "cn/metrics.json"),
    ("SEF", "expertise/metrics.json"),
    ("TeamFormation", "team-formation/metrics.json"),
    ("VAR", "var/metrics.json"),
    ("KExpertsCausal", "k_experts_causal/metrics.json"),
    ("BanditDR", "bandit_dr/metrics.json"),
    ("GraphPropagation", "graph_propagation/metrics.json"),
    ("VirtualEnsembles", "virtual_ensembles/metrics.json"),
]
BOC_MODES = ("none", "global", "subreddit", "full")

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


def _canonical_split_dirs(root: Path) -> Dict[str, Path]:
    """Return the fixed and windowed result directories in benchmark order."""
    found: Dict[str, Path] = {}
    for name in ("random", "full", "intersection", "intersection_chronological"):
        path = root / name
        if path.is_dir():
            found[name] = path
    for population in ("full", "intersection"):
        window_root = root / "windows" / population
        if not window_root.is_dir():
            continue
        for path in sorted(window_root.glob("w*")):
            if path.is_dir():
                found[f"windows/{population}/{path.name}"] = path
    return found


def _test_payload(metrics: Dict) -> Dict:
    """Normalize flat and train/val/test metric schemas."""
    test = metrics.get("test")
    return test if isinstance(test, dict) else metrics


def collect_canonical_suite(root: Path) -> Dict[str, pd.DataFrame]:
    """Collect the 14-method suite and select the BoC metadata mode on VAL."""
    output: Dict[str, pd.DataFrame] = {}
    for split_name, split_dir in _canonical_split_dirs(root).items():
        rows: List[Dict] = []
        for label, relative_path in CANONICAL_RUNS:
            path = split_dir / relative_path
            if not path.exists():
                continue
            payload = _test_payload(load_metrics(path))
            rows.append({"run": label, **{metric: payload.get(metric) for metric in METRICS}})

        boc_candidates = []
        for mode in BOC_MODES:
            path = split_dir / "boc_stacking" / mode / "metrics.json"
            if not path.exists():
                continue
            metrics = load_metrics(path)
            validation = metrics.get("val", {})
            test = metrics.get("test", {})
            boc_candidates.append((validation.get("macro_f1", float("-inf")), mode, test))
        if boc_candidates:
            _, selected_mode, payload = max(boc_candidates, key=lambda candidate: candidate[0])
            rows.append({
                "run": f"BoC[val:{selected_mode}]",
                **{metric: payload.get(metric) for metric in METRICS},
            })

        if rows:
            output[split_name] = (
                pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
            )
    return output


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

    if len(args.roots) == 1:
        per_split_df = collect_canonical_suite(args.roots[0])
    else:
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
