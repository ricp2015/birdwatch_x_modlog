from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import pandas as pd


def discover_splits(votes_dir: Path) -> Dict[str, Path]:
    """
    Find every split directory produced by prepare_data_step1 under votes_dir,
    i.e. any folder containing train_votes.parquet / val_votes.parquet /
    test_votes.parquet.
    """
    found: Dict[str, Path] = {}
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
    train = pd.read_parquet(split_dir / "train_votes.parquet")
    val   = pd.read_parquet(split_dir / "val_votes.parquet")
    test  = pd.read_parquet(split_dir / "test_votes.parquet")
    return train, val, test


def subset_stats(df: pd.DataFrame, min_vote_thresholds: List[int] = (5, 10, 20)) -> Dict:
    """
    Core stats for one subset (train/val/test/total): population size plus
    saturation indicators — the fraction of users/user*community pairs that
    already clear common minimum-vote thresholds. This is the key diagnostic
    for "why doesn't performance change much between small and large
    training windows": if most active users already clear the threshold
    even in the smallest window, per-user statistics were already
    well-estimated and a flat learning curve is expected, not a bug.
    """
    if df.empty:
        stats = {
            "n_votes": 0, "n_posts": 0, "n_users": 0, "n_communities": 0,
            "avg_votes_per_user": 0.0, "avg_votes_per_post": 0.0,
            "median_votes_per_user": 0.0,
        }
        for thr in min_vote_thresholds:
            stats[f"pct_users_ge_{thr}_votes"] = 0.0
        return stats

    n_votes  = len(df)
    n_posts  = df["item_id"].nunique()
    n_users  = df["username"].nunique()
    n_comms  = df["community"].nunique() if "community" in df.columns else None

    votes_per_user = df.groupby("username").size()
    votes_per_post = df.groupby("item_id").size()

    stats = {
        "n_votes":               n_votes,
        "n_posts":               n_posts,
        "n_users":               n_users,
        "n_communities":         n_comms,
        "avg_votes_per_user":    float(votes_per_user.mean()),
        "median_votes_per_user": float(votes_per_user.median()),
        "avg_votes_per_post":    float(votes_per_post.mean()),
    }
    for thr in min_vote_thresholds:
        stats[f"pct_users_ge_{thr}_votes"] = float((votes_per_user >= thr).mean())

    # user x community saturation — this is the unit TFR/SEF actually key
    # their per-category / per-subreddit reliability estimates on
    if "community" in df.columns:
        uc_counts = df.groupby(["username", "community"]).size()
        for thr in min_vote_thresholds:
            stats[f"pct_user_community_pairs_ge_{thr}_votes"] = float((uc_counts >= thr).mean())

    # label balance (approve vs remove), item-level
    if "label" in df.columns:
        item_labels = df.drop_duplicates("item_id")["label"]
        stats["pct_label_approve"] = float((item_labels == 1).mean())
        stats["pct_label_remove"]  = float((item_labels == -1).mean())

    return stats


def render_split_report(split_name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
    subsets = {"train": train, "val": val, "test": test,
               "total": pd.concat([train, val, test], ignore_index=True)}

    all_stats = {name: subset_stats(df) for name, df in subsets.items()}
    all_keys = list(all_stats["total"].keys())

    lines = [f"SPLIT: {split_name}", "-" * (len(split_name) + 7)]

    col_widths = [max(34, max(len(k) for k in all_keys) + 2)] + [12] * len(subsets)
    header = ["metric"] + list(subsets.keys())
    lines.append("  ".join(c.ljust(w) for c, w in zip(header, col_widths)))
    lines.append("  ".join("-" * w for w in col_widths))

    for key in all_keys:
        row = [key]
        for name in subsets:
            v = all_stats[name].get(key)
            if v is None:
                cell = "-"
            elif isinstance(v, float):
                cell = f"{v:.3f}" if v < 1000 else f"{v:,.1f}"
            else:
                cell = f"{v:,}"
            row.append(cell)
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, col_widths)))

    return "\n".join(lines)


def render_cross_split_overview(per_split_stats: Dict[str, Dict]) -> str:
    """
    One row per split, TRAIN-subset only, focused on the numbers that most
    directly explain a flat/non-flat learning curve: n_votes, n_users,
    avg_votes_per_user, and the fraction of users already past common
    minimum-vote thresholds.
    """
    lines = ["CROSS-SPLIT OVERVIEW (TRAIN subset only)", "=" * 90]
    cols = ["split", "n_votes", "n_posts", "n_users",
            "avg_votes/user", "pct_users>=5", "pct_users>=10", "pct_users>=20"]
    widths = [34, 10, 10, 10, 15, 13, 13, 13]
    lines.append("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    lines.append("  ".join("-" * w for w in widths))

    for split_name, stats in per_split_stats.items():
        t = stats["train"]
        row = [
            split_name,
            f"{t.get('n_votes', 0):,}",
            f"{t.get('n_posts', 0):,}",
            f"{t.get('n_users', 0):,}",
            f"{t.get('avg_votes_per_user', 0):.2f}",
            f"{t.get('pct_users_ge_5_votes', 0):.1%}",
            f"{t.get('pct_users_ge_10_votes', 0):.1%}",
            f"{t.get('pct_users_ge_20_votes', 0):.1%}",
        ]
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    return "\n".join(lines)


def compare_split_statistics(
    votes_dir:  Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    splits = discover_splits(votes_dir)
    if not splits:
        print(f"No split directories found under {votes_dir}")
        return {}

    print(f"Found {len(splits)} split(s): {sorted(splits.keys())}\n")

    per_split_stats: Dict[str, Dict] = {}
    all_blocks: List[str] = []

    for split_name in sorted(splits.keys()):
        train, val, test = load_split_data(splits[split_name])
        block = render_split_report(split_name, train, val, test)
        print(block)
        print()
        all_blocks.append(block)

        per_split_stats[split_name] = {
            "train": subset_stats(train),
            "val":   subset_stats(val),
            "test":  subset_stats(test),
        }

    overview = render_cross_split_overview(per_split_stats)
    print(overview)
    all_blocks.append(overview)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "split_statistics.txt", "w") as fh:
            fh.write("\n\n".join(all_blocks) + "\n")
        print(f"\nReport saved -> {output_dir / 'split_statistics.txt'}")

    return per_split_stats


# ===========================================================================
# PERFORMANCE TREND ACROSS WINDOWED FOLDS  (macro_f1 / f1_pos / f1_neg / roc_auc)
# ===========================================================================
#
# Same "*_by_split" discovery logic as compare_methods_text.py, but focused
# on ONE question: for the windowed_folds_* splits (where train size grows
# from w020 to w100 while val/test stay fixed), does f1_pos / f1_neg
# actually move with more training data, or is the whole macro_f1 curve
# flat because the majority class (approve) was already learned at w020
# and the minority class (remove) just gets averaged away?

RUN_LABELS = {
    "reddit_cn":                     "CN",
    "reddit_cn_inverted":            "CN_inverted",
    "step3_expert":                  "SEF",
    "team_formation":                "TeamFormation",
    "VAR":                            "VAR",
    "BL1_net_vote":                  "Net",
    "BL1_net_score":                 "Net",
    "BL2_net_vote_alpha":            "WNet",
    "BL2_weighted_net_score":        "WNet",
    "BL3_net_vote_alpha_subreddit":  "SubAdj",
    "BL3_subreddit_adjusted":        "SubAdj",
    "BL4_reddit_score":              "RedditScore",
    "BL5_subreddit":                 "RedditScore+Sub",
}

BY_SPLIT_SUFFIX = "_by_split"
TWO_PART_SPLIT_ROOTS = {"windowed_folds_full", "windowed_folds_intersection"}


def _parse_split_and_run(metrics_path: Path, root: Path) -> Optional[Tuple[str, str]]:
    """Same parsing logic as compare_methods_text.py — see that file for the
    full explanation of the three folder-layout patterns it handles."""
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


def collect_metrics_by_split(results_roots: List[Path]) -> Dict[str, Dict[str, Dict]]:
    """Returns {split_name: {run_label: metrics_dict}}."""
    out: Dict[str, Dict[str, Dict]] = {}
    for root in results_roots:
        if not root.exists():
            continue
        for p in root.rglob("metrics.json"):
            parsed = _parse_split_and_run(p, root)
            if parsed is None:
                continue
            split_name, run_key = parsed
            all_parts = p.parent.relative_to(root).parts
            if any("wikipedia" in part for part in all_parts):
                continue
            with open(p) as fh:
                m = json.load(fh)
            label = RUN_LABELS.get(run_key, run_key)
            out.setdefault(split_name, {})[label] = m
    return out


def _window_sort_key(split_name: str):
    """Sort 'windowed_folds_full/w020' etc. by tag, then numeric window size."""
    match = re.search(r"/w(\d+)$", split_name)
    w = int(match.group(1)) if match else -1
    tag = split_name.split("/")[0]
    return (tag, w)


def render_windowed_performance_trend(
    per_split_metrics: Dict[str, Dict[str, Dict]],
    runs: Optional[List[str]] = None,
    metrics: List[str] = ("macro_f1", "f1_pos", "f1_neg", "roc_auc"),
) -> str:
    """
    One table per metric: rows = windowed_folds_* splits (sorted w020..w100,
    grouped by full/intersection), columns = runs. Lets you see at a glance
    whether f1_pos/f1_neg actually move with training-set size, even when
    macro_f1 looks flat.
    """
    windowed_splits = sorted(
        (s for s in per_split_metrics if s.startswith("windowed_folds_")),
        key=_window_sort_key,
    )
    if not windowed_splits:
        return "No windowed_folds_* splits found among the collected metrics.json files."

    if runs is None:
        seen = []
        for s in windowed_splits:
            for r in per_split_metrics[s]:
                if r not in seen:
                    seen.append(r)
        runs = seen

    blocks = []
    for metric in metrics:
        lines = [f"WINDOWED PERFORMANCE TREND — {metric}", "=" * 60]
        col_widths = [34] + [14] * len(runs)
        header = ["split"] + runs
        lines.append("  ".join(c.ljust(w) for c, w in zip(header, col_widths)))
        lines.append("  ".join("-" * w for w in col_widths))
        for split_name in windowed_splits:
            row = [split_name]
            for run in runs:
                v = per_split_metrics[split_name].get(run, {}).get(metric)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
            lines.append("  ".join(c.ljust(w) for c, w in zip(row, col_widths)))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def compare_windowed_performance(
    results_roots: List[Path],
    runs:          Optional[List[str]] = None,
    output_dir:    Optional[Path] = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Entry point: collect metrics.json across results_roots and print the
    f1_pos/f1_neg/macro_f1/roc_auc trend across windowed_folds_* splits.
    """
    per_split_metrics = collect_metrics_by_split(results_roots)
    report = render_windowed_performance_trend(per_split_metrics, runs=runs)
    print(report)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "windowed_performance_trend.txt", "w") as fh:
            fh.write(report + "\n")
        print(f"\nReport saved -> {output_dir / 'windowed_performance_trend.txt'}")

    return per_split_metrics


if __name__ == "__main__":
    compare_split_statistics(
        Path("results/step1/reddit"),
        output_dir=Path("results/step1/reddit/split_statistics"),
    )
    compare_windowed_performance(
        [Path("results/step2"), Path("results/step3_expert")],
        runs=["TeamFormation", "Net"],
        output_dir=Path("results/step1/reddit/split_statistics"),
    )