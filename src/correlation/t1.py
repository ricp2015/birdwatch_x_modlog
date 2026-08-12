"""Analyze how user characteristics relate to method-specific reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats



# 1. USER CHARACTERISTICS  (stable, split-independent)

def load_karma_tenure(user_metadata_path: Path) -> pd.DataFrame:
    """Load karma tenure from its configured source."""
    meta = pd.read_csv(user_metadata_path)
    meta["account_created_utc"] = pd.to_numeric(meta.get("account_created_utc"), errors="coerce")
    REF_TS = 1685000000  # ~May 2023, dataset cutoff — same reference used in TFR's _load_feature_skill
    meta["tenure_days"] = ((REF_TS - meta["account_created_utc"]) / 86400).clip(lower=0)
    meta["total_karma"] = pd.to_numeric(meta.get("total_karma"), errors="coerce")
    meta["log_karma"]   = np.log1p(meta["total_karma"].clip(lower=0))
    if "is_suspended" not in meta.columns:
        meta["is_suspended"] = False
    if "has_verified_email" not in meta.columns:
        meta["has_verified_email"] = np.nan
    cols = ["username", "total_karma", "log_karma", "tenure_days",
            "is_suspended", "has_verified_email"]
    return meta[cols].drop_duplicates(subset="username")


def _shannon_entropy(counts: np.ndarray) -> float:
    """Calculate Shannon entropy for observed category counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_activity_diversity(raw_csv_path: Path) -> pd.DataFrame:
    """Compute activity diversity from the supplied data."""
    df = pd.read_csv(raw_csv_path, usecols=["username", "item_id", "community"], low_memory=False)
    df = df.dropna(subset=["username", "community"])

    activity = df.groupby("username").size().rename("n_votes_total")
    diversity = df.groupby("username")["community"].nunique().rename("n_communities")

    entropy_rows = []
    for uname, grp in df.groupby("username")["community"]:
        counts = grp.value_counts().values.astype(float)
        entropy_rows.append({"username": uname, "subreddit_entropy": _shannon_entropy(counts)})
    entropy_df = pd.DataFrame(entropy_rows).set_index("username")["subreddit_entropy"]

    out = pd.concat([activity, diversity, entropy_df], axis=1).reset_index()
    return out


def build_user_characteristics(
    user_metadata_path: Path,
    raw_csv_path: Path,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build user characteristics from the supplied data."""
    if cache_path is not None and cache_path.exists():
        print(f"  Loading cached user characteristics from {cache_path}")
        return pd.read_parquet(cache_path)

    print("  Loading karma/tenure from user_metadata.csv ...")
    karma_tenure = load_karma_tenure(user_metadata_path)
    print(f"    {len(karma_tenure):,} users")

    print("  Computing activity/diversity from raw CSV (full pass) ...")
    activity_diversity = compute_activity_diversity(raw_csv_path)
    print(f"    {len(activity_diversity):,} users")

    merged = karma_tenure.merge(activity_diversity, on="username", how="outer")
    print(f"  Merged user characteristics: {len(merged):,} users "
          f"({merged['total_karma'].notna().sum():,} with karma, "
          f"{merged['n_votes_total'].notna().sum():,} with activity)")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        print(f"  Cached → {cache_path}")

    return merged


# 2. PER-METHOD, PER-SPLIT USER SCORES  (NOT stable across splits)

def load_cn_user_scores(split_dir: Path) -> Optional[pd.DataFrame]:
    """Load cn user scores from its configured source."""
    path = split_dir / "user_params.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).copy()
    if "n_votes" not in df.columns:
        print("    [CN] user_params.parquet has no n_votes column — re-run the patched "
              "CN_wrapper.py to enable the n_signal sensitivity check for CN. Using NaN for now.")
        df["n_votes"] = np.nan
    df = df[["username", "f_u", "i_u", "n_votes"]].rename(
        columns={"i_u": "score_primary", "f_u": "score_polarization", "n_votes": "n_signal"}
    )
    return df


def load_sef_user_scores(split_dir: Path) -> Optional[pd.DataFrame]:
    """Load sef user scores from its configured source."""
    path = split_dir / "weights.parquet"
    if not path.exists():
        return None
    w = pd.read_parquet(path)
    agg = (
        w.groupby("username")
        .agg(
            score_primary=("reliability", "mean"),
            local_prec_mean=("local_prec", "mean"),
            expert_weight_mean=("expert_weight", "mean"),
            n_signal=("item_id", "count"),
        )
        .reset_index()
    )
    return agg


TFR_CATEGORIES = [
    "spam", "meta-rules", "content", "doxxing",
    "harassment", "hatespeech", "format",
    "off-topic", "trolling", "incivility",
]


def load_tfr_user_scores(split_dir: Path) -> Optional[pd.DataFrame]:
    """Load tfr user scores from its configured source."""
    path = split_dir / "user_skill.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).copy()
    if s.empty:
        return None

    mod_cols = [f"skill_mod_{cat}" for cat in TFR_CATEGORIES if f"skill_mod_{cat}" in s.columns]
    if not mod_cols:
        print("    [TFR] no skill_mod_<category> column found; "
              "impossibile costruire uno score karma/tenure-free. Controlla lo schema di user_skill.parquet.")
        return None

    s["n_categories_with_signal"] = s[mod_cols].notna().sum(axis=1)
    s["score_primary"] = s[mod_cols].mean(axis=1, skipna=True)

    n_before = len(s)
    s = s[s["n_categories_with_signal"] > 0].copy()
    n_dropped = n_before - len(s)
    if n_dropped:
        print(f"    [TFR] scartate {n_dropped} / {n_before} righe (username, community) senza "
              f"no usable skill_mod signal (only skill_feat would remain; "
              f"escluse per evitare la circolarita' con karma/tenure)")

    return s.rename(columns={"n_votes": "n_signal"})[
        ["username", "community", "score_primary", "n_signal", "n_categories_with_signal"]
    ]


def load_var_user_scores(split_dir: Path) -> Optional[pd.DataFrame]:
    """Load var user scores from its configured source."""
    path = split_dir / "all_user_var_scores.parquet"
    if not path.exists():
        print(f"    [VAR] all_user_var_scores.parquet not found at {path} — "
              f"falling back to voter_scores.parquet (WARNING: top-K filtered, "
              f"correlations will be biased). Re-run the patched var_baseline.py "
              f"to produce the unfiltered file.")
        return _load_var_user_scores_topk_fallback(split_dir)

    s = pd.read_parquet(path)
    if s.empty:
        return None

    def _weighted_mean(g):
        """Return a count-weighted mean score."""
        w = g["total_votes"].values.astype(float)
        v = g["VAR_score"].values.astype(float)
        if w.sum() == 0:
            return np.nan
        return float(np.average(v, weights=w))

    agg = (
        s.groupby("username")
        .apply(lambda g: pd.Series({
            "score_primary": _weighted_mean(g),
            "n_signal":      g["total_votes"].sum(),
            "n_communities_in_split": g["community"].nunique(),
        }))
        .reset_index()
    )
    return agg


def _load_var_user_scores_topk_fallback(split_dir: Path) -> Optional[pd.DataFrame]:
    """Load var user scores topk fallback from its configured source."""
    path = split_dir / "voter_scores.parquet"
    if not path.exists():
        return None
    v = pd.read_parquet(path)
    if v.empty:
        return None
    per_user_comm = (
        v.groupby(["username", "community"])
        .agg(VAR_score=("VAR_score", "first"), n_rows=("item_id", "count"))
        .reset_index()
    )

    def _weighted_mean(g):
        """Return a count-weighted mean score."""
        w = g["n_rows"].values.astype(float)
        val = g["VAR_score"].values.astype(float)
        if w.sum() == 0:
            return np.nan
        return float(np.average(val, weights=w))

    agg = (
        per_user_comm.groupby("username")
        .apply(lambda g: pd.Series({
            "score_primary": _weighted_mean(g),
            "n_signal":      g["n_rows"].sum(),
            "n_communities_in_split": g["community"].nunique(),
        }))
        .reset_index()
    )
    agg["_topk_biased"] = True
    return agg


METHOD_LOADERS = {
    # name -> (base_dir, loader_fn, cluster_col)
    # cluster_col=None means "already one row per user" (plain Spearman OK).
    # cluster_col="username" means rows are NOT one-per-user (multiple rows
    # per user, e.g. one per community) -> use the cluster bootstrap.
    "CN":  ("results/reddit", load_cn_user_scores,  None),
    "SEF": ("results/reddit", load_sef_user_scores, None),
    "TFR": ("results/reddit", load_tfr_user_scores, "username"),
    "VAR": ("results/reddit", load_var_user_scores, None),
}


# 3. ANALYSIS: correlation + quartile stratification

CHARACTERISTIC_COLS = ["log_karma", "tenure_days", "n_votes_total", "n_communities", "subreddit_entropy"]

# Sensitivity thresholds for minimum per-user signal (n_signal = number of
# votes/rows backing that user's score_primary in this split). Scores based
# on very few votes are noisy — especially for methods with explicit
# shrinkage toward a prior (SEF's Laplace smoothing, VAR's LAMBDA=10 —
# see run_single_split_var / precompute_vote_precision), where "score more
# extreme with more votes" can be an artifact of the shrinkage formula
# itself rather than a real behavioural pattern. Running the same
# correlation at multiple thresholds shows whether an effect survives once
# low-signal users are excluded, instead of trusting a single cutoff.
N_SIGNAL_THRESHOLDS = [0, 5, 20]


def cluster_bootstrap_spearman(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict:
    """Estimate Spearman uncertainty with a cluster bootstrap."""
    rng = np.random.RandomState(seed)
    grouped = df.groupby(cluster_col).indices  # cluster -> positional row indices
    cluster_keys = np.array(list(grouped.keys()))
    n_clusters = len(cluster_keys)

    obs_rho, _ = stats.spearmanr(df[x_col], df[y_col])

    boot_rhos = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_keys, size=n_clusters, replace=True)
        idx = np.concatenate([grouped[k] for k in sampled])
        sub = df.iloc[idx]
        if sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(sub[x_col], sub[y_col])
        if not np.isnan(rho):
            boot_rhos.append(rho)

    if len(boot_rhos) < 20:
        return {
            "spearman_rho": round(float(obs_rho), 4) if not np.isnan(obs_rho) else None,
            "p_value": None, "n": len(df), "n_clusters": n_clusters,
            "clustered": True, "bootstrap_failed": True,
        }

    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    # two-sided bootstrap p-value: 2x the smaller tail crossing zero
    p_boot = float(min(1.0, 2 * min((boot_rhos >= 0).mean(), (boot_rhos <= 0).mean())))

    return {
        "spearman_rho": round(float(obs_rho), 4),
        "p_value": p_boot,
        "n": len(df),
        "n_clusters": n_clusters,
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "clustered": True,
        "n_bootstrap": len(boot_rhos),
    }


def analyze_method(
    method_name: str,
    user_scores: pd.DataFrame,
    characteristics: pd.DataFrame,
    min_n_signal: int = 0,
    cluster_col: Optional[str] = None,
    exclude_suspended: bool = False,
) -> Dict:
    """Analyze method and return summary statistics."""
    chars = characteristics
    if exclude_suspended and "is_suspended" in chars.columns:
        chars = chars[chars["is_suspended"] != True]  # noqa: E712

    merged_full = user_scores.merge(chars, on="username", how="inner")
    n_scored   = len(user_scores)
    n_matched  = len(merged_full)
    n_unique_users_matched = merged_full["username"].nunique()
    coverage   = n_matched / n_scored if n_scored else float("nan")

    if "n_signal" in merged_full.columns and min_n_signal > 0:
        merged = merged_full[merged_full["n_signal"] >= min_n_signal].copy()
    else:
        merged = merged_full

    result: Dict = {
        "method": method_name,
        "min_n_signal": min_n_signal,
        "clustered": cluster_col is not None,
        "exclude_suspended": exclude_suspended,
        "n_rows_with_score": n_scored,
        "n_rows_matched_to_characteristics": n_matched,
        "n_unique_users_matched": n_unique_users_matched,
        "n_rows_after_signal_filter": len(merged),
        "coverage_pct": round(100 * coverage, 1),
        "correlations": {},
        "quartile_tables": {},
    }

    if merged.empty:
        return result

    for col in CHARACTERISTIC_COLS:
        sub = merged[[cluster_col, "score_primary", col]].dropna() if cluster_col \
            else merged[["score_primary", col]].dropna()
        if len(sub) < 10:
            result["correlations"][col] = None
            continue
        if cluster_col:
            result["correlations"][col] = cluster_bootstrap_spearman(
                sub, cluster_col, "score_primary", col
            )
        else:
            rho, pval = stats.spearmanr(sub["score_primary"], sub[col])
            result["correlations"][col] = {
                "spearman_rho": round(float(rho), 4),
                "p_value":      float(pval),
                "n":            len(sub),
                "clustered":    False,
            }

    for col in CHARACTERISTIC_COLS:
        sub = merged[["score_primary", col]].dropna()
        if len(sub) < 20:
            continue
        try:
            sub = sub.copy()
            sub["quartile"] = pd.qcut(sub[col], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
        except ValueError:
            continue
        n_bins_actual = sub["quartile"].nunique()
        table = (
            sub.groupby("quartile", observed=True)["score_primary"]
            .agg(["mean", "median", "std", "count"])
            .round(4)
        )
        result["quartile_tables"][col] = {
            "n_bins_requested": 4,
            "n_bins_actual": int(n_bins_actual),
            "bins_collapsed": bool(n_bins_actual < 4),
            "table": table.to_dict(orient="index"),
        }

    return result


def run_signal_sensitivity(
    method_name: str,
    user_scores: pd.DataFrame,
    characteristics: pd.DataFrame,
    thresholds: list = N_SIGNAL_THRESHOLDS,
    cluster_col: Optional[str] = None,
    exclude_suspended: bool = False,
) -> Dict:
    """Run the signal sensitivity workflow."""
    if "n_signal" not in user_scores.columns:
        return {}
    per_threshold = {}
    for thr in thresholds:
        res = analyze_method(method_name, user_scores, characteristics, min_n_signal=thr,
                              cluster_col=cluster_col, exclude_suspended=exclude_suspended)
        per_threshold[thr] = {
            col: (c["spearman_rho"] if c else None)
            for col, c in res["correlations"].items()
        }
    return per_threshold


def print_sensitivity_report(method_name: str, sensitivity: Dict) -> None:
    """Print sensitivity report to the console."""
    if not sensitivity:
        return
    print(f"  Sensitivity to min_n_signal (Spearman rho at each threshold):")
    thresholds = sorted(sensitivity.keys())
    for col in CHARACTERISTIC_COLS:
        row = [sensitivity[t].get(col) for t in thresholds]
        row_str = "  ".join(
            f"n>={t}:{v:+.3f}" if v is not None else f"n>={t}:  n/a" for t, v in zip(thresholds, row)
        )
        print(f"    {col:20s}  {row_str}")


# FDR correction (Benjamini-Hochberg), applied ONCE across every p-value
# collected from every method x characteristic combination — not per method.
# With 4 methods x 5 characteristics = 20 tests, an uncorrected alpha=0.05
# would be expected to flag ~1 test as "significant" by chance alone even if
# nothing real is going on; BH keeps the false-discovery rate under control
# across the whole family of tests instead of per-test.

def benjamini_hochberg(pvals: list) -> list:
    """Adjust p-values with the Benjamini-Hochberg procedure."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (running minimum from the largest p-value down)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out.tolist()


def apply_fdr_correction(all_results: Dict) -> None:
    """Apply fdr correction to the supplied data."""
    entries = []  # list of (method, col) references into all_results
    pvals = []
    for method_name, result in all_results.items():
        for col, c in result.get("correlations", {}).items():
            if c is not None and c.get("p_value") is not None:
                entries.append((method_name, col))
                pvals.append(c["p_value"])

    if not pvals:
        return

    qvals = benjamini_hochberg(pvals)
    for (method_name, col), q in zip(entries, qvals):
        c = all_results[method_name]["correlations"][col]
        c["q_value"] = round(float(q), 4)
        c["significant_fdr"] = bool(q < 0.05)


def print_method_report(result: Dict) -> None:
    """Print method report to the console."""
    clustered_tag = " [CLUSTERED by username — bootstrap p-value]" if result.get("clustered") else ""
    susp_tag = " [suspended users excluded]" if result.get("exclude_suspended") else ""
    print(f"\n--- {result['method']} (min_n_signal={result.get('min_n_signal', 0)}){clustered_tag}{susp_tag} ---")
    print(f"  Righe con score: {result['n_rows_with_score']:,}  |  "
          f"righe matchate a caratteristiche: {result['n_rows_matched_to_characteristics']:,} "
          f"({result['coverage_pct']}%)  |  matched users: "
          f"{result.get('n_unique_users_matched', 'n/a')}  |  dopo filtro segnale: "
          f"{result.get('n_rows_after_signal_filter', result['n_rows_matched_to_characteristics']):,}")

    if not result["correlations"]:
        print("  [no matched users — skipping correlation/stratification]")
        return

    print("  Spearman correlation (score_primary vs characteristic):")
    print("    Note: sig(p<.05) is uncorrected and shown only for reference.")
    print("    Usa 'q=' / 'FDR-sig' (aggiunto dopo che tutti i metodi sono girati) per la conclusione family-wise.")
    for col, c in result["correlations"].items():
        if c is None:
            print(f"    {col:20s}  [n<10, skipped]")
        elif c.get("bootstrap_failed"):
            print(f"    {col:20s}  rho={c['spearman_rho']}  [bootstrap failed; fewer than 20 valid samples, n_clusters={c['n_clusters']}]")
        elif c["p_value"] is None:
            print(f"    {col:20s}  rho={c['spearman_rho']:+.4f}  p=n/a  n={c['n']}")
        else:
            flag = "  *" if c["p_value"] < 0.05 else ""
            q_str = f"  q={c['q_value']:.4g}{'  **FDR-sig**' if c.get('significant_fdr') else ''}" \
                    if "q_value" in c else ""
            ci_str = f"  CI95=[{c['ci_low']:+.3f},{c['ci_high']:+.3f}]  n_clusters={c['n_clusters']}" \
                     if c.get("clustered") else ""
            print(f"    {col:20s}  rho={c['spearman_rho']:+.4f}  p={c['p_value']:.4g}  n={c['n']}{flag}{q_str}{ci_str}")

    print("  Quartile stratification (mean score_primary by quartile):")
    for col, qinfo in result["quartile_tables"].items():
        collapse_warn = "  [collapsed bins]" if qinfo["bins_collapsed"] else ""
        print(f"    [{col}]  ({qinfo['n_bins_actual']}/{qinfo['n_bins_requested']} bins){collapse_warn}")
        for q, stats_row in qinfo["table"].items():
            print(f"      {q:10s}  mean={stats_row['mean']:+.4f}  "
                  f"median={stats_row['median']:+.4f}  n={int(stats_row['count'])}")


# MAIN

def main() -> None:
    """Run the command-line workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes-dir", type=str, default="data/splits/reddit",
                     help="Root dir with users.parquet / splits (only used for reference paths)")
    ap.add_argument("--split", type=str, default="splits",
                     help="Which split label to analyze (e.g. splits, splits_full, "
                          "windowed_folds_full/w060). Must match a *_by_split/<split> folder "
                          "for each method.")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv",
                     help="Path to Shayan's user_metadata.csv (default: data/processed/user_metadata.csv, "
                          "same path used by team_formation_ranker.py's METADATA_PATH)")
    ap.add_argument("--raw-csv", type=str, default="data/processed/final_intersection_dataset.csv",
                     help="Path to final_intersection_dataset.csv, full pre-filter dataset "
                          "(default: data/processed/final_intersection_dataset.csv, same path "
                          "used as ORIGINAL_CSV in the other scripts)")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--cache-characteristics", action="store_true",
                     help="Cache the merged user-characteristics table to speed up repeated runs")
    ap.add_argument("--min-n-signal", type=int, default=0,
                     help="Minimum n_signal (votes backing a user's score) required to keep that "
                          "user in the main correlation/quartile analysis. 0 = no filter.")
    ap.add_argument("--skip-sensitivity", action="store_true",
                     help="Skip the min_n_signal sensitivity sweep (faster, less robust)")
    ap.add_argument("--exclude-suspended", action="store_true",
                     help="Exclude suspended accounts from the characteristics table before "
                          "matching, mirroring TFR's _load_feature_skill behaviour. OFF by "
                          "default: for T1 the suspended-user pattern may itself be part of "
                          "what you want to detect, not something to filter out silently.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("T1 — User characteristics vs per-method reliability/skill")
    print(f"Split: {args.split}")
    if args.exclude_suspended:
        print("Suspended users excluded")

    cache_path = output_dir / "user_characteristics_cache.parquet" if args.cache_characteristics else None
    characteristics = build_user_characteristics(
        Path(args.user_metadata), Path(args.raw_csv), cache_path,
    )

    all_results = {}
    for method_name, (base_dir, loader, cluster_col) in METHOD_LOADERS.items():
        split_dir = Path(base_dir) / args.split
        print()
        print(f"METHOD: {method_name}  (looking in {split_dir})"
              + (f"  [cluster_col={cluster_col}]" if cluster_col else ""))

        if not split_dir.exists():
            print(f"  [SKIP] split directory not found: {split_dir}")
            continue

        user_scores = loader(split_dir)
        if user_scores is None or user_scores.empty:
            print(f"  [SKIP] no per-user detail file found/usable for {method_name} at {split_dir}")
            continue

        result = analyze_method(method_name, user_scores, characteristics,
                                 min_n_signal=args.min_n_signal, cluster_col=cluster_col,
                                 exclude_suspended=args.exclude_suspended)
        print_method_report(result)

        if not args.skip_sensitivity:
            sensitivity = run_signal_sensitivity(method_name, user_scores, characteristics,
                                                  cluster_col=cluster_col,
                                                  exclude_suspended=args.exclude_suspended)
            print_sensitivity_report(method_name, sensitivity)
            result["n_signal_sensitivity"] = sensitivity

        all_results[method_name] = result

    # FDR correction applied ONCE, jointly across every method x
    # characteristic combination collected above — see apply_fdr_correction()
    print()
    print("Applying Benjamini-Hochberg FDR correction across all "
          f"{sum(len(r['correlations']) for r in all_results.values())} tests...")
    apply_fdr_correction(all_results)
    print("Done. Re-printing summary with FDR-corrected significance:")
    for method_name, result in all_results.items():
        print_method_report(result)

    summary_path = output_dir / f"t1_summary_{args.split.replace('/', '_')}.json"
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
