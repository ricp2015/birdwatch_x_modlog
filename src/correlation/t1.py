"""Analyze how user characteristics relate to method-specific reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# User characteristics


def load_karma_tenure(user_metadata_path: Path) -> pd.DataFrame:
    """Load karma tenure from its configured source."""
    meta = pd.read_csv(user_metadata_path)
    meta["account_created_utc"] = pd.to_numeric(meta.get("account_created_utc"), errors="coerce")
    REF_TS = 1685000000  # Dataset cutoff used by NVSE.
    meta["tenure_days_collection_cutoff"] = ((REF_TS - meta["account_created_utc"]) / 86400).clip(
        lower=0
    )
    meta["total_karma"] = pd.to_numeric(meta.get("total_karma"), errors="coerce")
    meta["log_karma"] = np.log1p(meta["total_karma"].clip(lower=0))
    if "is_suspended" not in meta.columns:
        meta["is_suspended"] = False
    if "has_verified_email" not in meta.columns:
        meta["has_verified_email"] = np.nan
    cols = [
        "username",
        "total_karma",
        "log_karma",
        "tenure_days_collection_cutoff",
        "is_suspended",
        "has_verified_email",
    ]
    return meta[cols].drop_duplicates(subset="username")


def _shannon_entropy(counts: np.ndarray) -> float:
    """Calculate Shannon entropy for observed category counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_activity_diversity(
    causal_features_path: Path,
    train_votes_path: Path,
) -> pd.DataFrame:
    """Return each user's latest strictly prior TRAIN profile.

    Shannon entropy is unavailable without historical per-subreddit counts.
    """
    feature_columns = [
        "username",
        "item_id",
        "timestamp",
        "prior_n_contributions",
        "prior_n_distinct_subreddits",
        "prior_n_replies_made",
        "prior_n_interaction_partners",
        "tenure_days_at_vote",
    ]
    causal = pd.read_parquet(causal_features_path, columns=feature_columns)
    train = pd.read_parquet(train_votes_path, columns=["username", "item_id", "timestamp"])
    for frame in (causal, train):
        frame["username"] = frame["username"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        if pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            frame["timestamp"] = frame["timestamp"].map(
                lambda value: value.timestamp() if pd.notna(value) else np.nan
            )
        else:
            frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    keys = ["username", "item_id", "timestamp"]
    matched = (
        train[keys]
        .drop_duplicates()
        .merge(causal, on=keys, how="left", validate="one_to_one", indicator=True)
    )
    coverage = float((matched["_merge"] == "both").mean()) if len(matched) else 0.0
    if coverage < 0.99:
        raise ValueError(f"Causal T2 join coverage is unexpectedly low: {coverage:.2%}")
    matched = matched.drop(columns="_merge").sort_values(["username", "timestamp"])
    latest = matched.groupby("username", as_index=False).tail(1).copy()
    count_columns = [
        "prior_n_contributions",
        "prior_n_distinct_subreddits",
        "prior_n_replies_made",
        "prior_n_interaction_partners",
    ]
    latest[count_columns] = latest[count_columns].fillna(0.0)
    return latest[["username", *count_columns, "tenure_days_at_vote"]].rename(
        columns={
            "prior_n_contributions": "n_contributions_prior",
            "prior_n_distinct_subreddits": "n_communities_prior",
            "prior_n_replies_made": "n_replies_prior",
            "prior_n_interaction_partners": "n_interaction_partners_prior",
            "tenure_days_at_vote": "tenure_days",
        }
    )


def build_user_characteristics(
    user_metadata_path: Path,
    causal_features_path: Path,
    train_votes_path: Path,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build user characteristics from the supplied data."""
    if cache_path is not None and cache_path.exists():
        print(f"User characteristics cache: {cache_path}")
        return pd.read_parquet(cache_path)

    karma_tenure = load_karma_tenure(user_metadata_path)
    activity_diversity = compute_activity_diversity(causal_features_path, train_votes_path)

    merged = karma_tenure.merge(activity_diversity, on="username", how="outer")
    print(
        f"User characteristics: {len(merged):,} | "
        f"karma={merged['total_karma'].notna().sum():,} | "
        f"activity={merged['n_contributions_prior'].notna().sum():,}"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        print(f"User characteristics saved: {cache_path}")

    return merged


# Per-method, per-split user scores


def load_cn_user_scores(
    split_dir: Path,
    votes_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load cn user scores from its configured source."""
    path = split_dir / "user_params.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).copy()
    if "n_votes" not in df.columns:
        print("CN sensitivity unavailable: user_params.parquet lacks n_votes")
        df["n_votes"] = np.nan
    df = df[["username", "f_u", "i_u", "n_votes"]].rename(
        columns={"i_u": "score_primary", "f_u": "score_polarization", "n_votes": "n_signal"}
    )
    return df


def load_sef_user_scores(
    split_dir: Path,
    votes_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
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


def load_sef_user_community_scores(
    split_dir: Path,
    votes_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Aggregate SEF reliability by user and community for a robustness check."""
    path = split_dir / "weights.parquet"
    if not path.exists() or votes_path is None or not votes_path.exists():
        return None

    weights = pd.read_parquet(path)
    votes = pd.read_parquet(votes_path, columns=["item_id", "community"])
    item_communities = votes.drop_duplicates("item_id")
    scored = weights.merge(item_communities, on="item_id", how="left", validate="many_to_one")
    scored = scored.dropna(subset=["username", "community", "reliability"])
    if scored.empty:
        return None

    return (
        scored.groupby(["username", "community"])
        .agg(
            score_primary=("reliability", "mean"),
            local_prec_mean=("local_prec", "mean"),
            expert_weight_mean=("expert_weight", "mean"),
            n_signal=("item_id", "count"),
        )
        .reset_index()
    )


NVSE_CATEGORIES = [
    "spam",
    "meta-rules",
    "content",
    "doxxing",
    "harassment",
    "hatespeech",
    "format",
    "off-topic",
    "trolling",
    "incivility",
]


def load_nvse_user_scores(
    split_dir: Path,
    votes_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load NVSE user scores from its configured source."""
    path = split_dir / "user_skill.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).copy()
    if s.empty:
        return None

    mod_cols = [f"skill_mod_{cat}" for cat in NVSE_CATEGORIES if f"skill_mod_{cat}" in s.columns]
    if not mod_cols:
        print("NVSE scores unavailable: user_skill.parquet lacks skill_mod columns")
        return None

    s["n_categories_with_signal"] = s[mod_cols].notna().sum(axis=1)
    s["score_primary"] = s[mod_cols].mean(axis=1, skipna=True)

    n_before = len(s)
    s = s[s["n_categories_with_signal"] > 0].copy()
    n_dropped = n_before - len(s)
    if n_dropped:
        print(f"NVSE rows without moderator signal: {n_dropped:,}/{n_before:,}")

    return s.rename(columns={"n_votes": "n_signal"})[
        ["username", "community", "score_primary", "n_signal", "n_categories_with_signal"]
    ]


def load_var_user_scores(
    split_dir: Path,
    votes_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load var user scores from its configured source."""
    path = split_dir / "all_user_var_scores.parquet"
    if not path.exists():
        print(f"VAR fallback: {path} missing; using top-K voter scores")
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
        .apply(
            lambda g: pd.Series(
                {
                    "score_primary": _weighted_mean(g),
                    "n_signal": g["total_votes"].sum(),
                    "n_communities_in_split": g["community"].nunique(),
                }
            )
        )
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
        .apply(
            lambda g: pd.Series(
                {
                    "score_primary": _weighted_mean(g),
                    "n_signal": g["n_rows"].sum(),
                    "n_communities_in_split": g["community"].nunique(),
                }
            )
        )
        .reset_index()
    )
    agg["_topk_biased"] = True
    return agg


METHOD_LOADERS = {
    # Values contain the result root, loader, and optional bootstrap cluster.
    "CN": ("results/reddit", load_cn_user_scores, None),
    "SEF": ("results/reddit", load_sef_user_scores, None),
    "SEF-community": ("results/reddit", load_sef_user_community_scores, "username"),
    "NVSE": ("results/reddit", load_nvse_user_scores, "username"),
    "VAR": ("results/reddit", load_var_user_scores, None),
}
METHOD_RESULT_DIRS = {
    "CN": "cn",
    "SEF": "expertise",
    "SEF-community": "expertise",
    "NVSE": "normvio-skill-extraction",
    "VAR": "var",
}
SEF_MODES = ("none", "global", "subreddit", "full")


def resolve_method_result_dir(
    method_name: str,
    base_dir: str,
    split: str,
    sef_mode: str = "auto",
) -> tuple[Path, Optional[str]]:
    """Resolve canonical outputs, selecting the SEF mode on validation only."""
    root = Path(base_dir)
    if method_name not in {"SEF", "SEF-community"}:
        return root / split / METHOD_RESULT_DIRS[method_name], None
    if "chronological" in split.lower():
        # SEF profiles use complete document histories and lack as-of semantics.
        return root / "__sef_excluded_static_profile__" / split, None

    modes = SEF_MODES if sef_mode == "auto" else (sef_mode,)
    candidates = []
    for mode in modes:
        result_dir = root / mode / split / "expertise"
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation = metrics.get("val") or {}
        macro_f1 = validation.get("macro_f1")
        if macro_f1 is not None:
            candidates.append((float(macro_f1), mode, result_dir))
    if not candidates:
        requested = f"mode={sef_mode}" if sef_mode != "auto" else "any canonical mode"
        return root / "__missing_sef__" / split / requested, None
    _, selected_mode, selected_dir = max(candidates, key=lambda row: row[0])
    return selected_dir, selected_mode


def read_timestamp_semantics(causal_features_path: Path) -> list[str]:
    """Read temporal provenance from a causal-feature artifact."""
    try:
        values = pd.read_parquet(causal_features_path, columns=["timestamp_semantics"])[
            "timestamp_semantics"
        ]
    except Exception as exc:
        # Legacy Parquet files may not expose this optional provenance column.
        print(f"  Timestamp provenance unavailable ({exc})")
        return ["unknown_legacy_artifact"]
    semantics = sorted(values.dropna().astype(str).unique().tolist())
    return semantics or ["unknown_legacy_artifact"]


# Correlation and quartile analysis

CHARACTERISTIC_COLS = [
    "log_karma",
    "tenure_days",
    "n_contributions_prior",
    "n_communities_prior",
]

# Minimum evidence thresholds for the score sensitivity analysis.
N_SIGNAL_THRESHOLDS = [0, 5, 20]


def cluster_bootstrap_spearman(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    n_boot: int = 500,
    seed: int = 10,
) -> Dict:
    """Estimate Spearman uncertainty with a cluster bootstrap."""
    rng = np.random.RandomState(seed)
    grouped = df.groupby(cluster_col).indices
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
            "p_value": None,
            "n": len(df),
            "n_clusters": n_clusters,
            "clustered": True,
            "bootstrap_failed": True,
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
    n_scored = len(user_scores)
    n_matched = len(merged_full)
    n_unique_users_matched = merged_full["username"].nunique()
    coverage = n_matched / n_scored if n_scored else float("nan")

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
        sub = (
            merged[[cluster_col, "score_primary", col]].dropna()
            if cluster_col
            else merged[["score_primary", col]].dropna()
        )
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
                "p_value": float(pval),
                "n": len(sub),
                "clustered": False,
            }

    for col in CHARACTERISTIC_COLS:
        sub = merged[["score_primary", col]].dropna()
        if len(sub) < 20:
            continue
        try:
            sub = sub.copy()
            sub["quartile"] = pd.qcut(
                sub[col], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop"
            )
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

    result["multivariate"] = run_multivariate_analysis(merged, cluster_col=cluster_col)
    result["multivariate_sensitivity"] = run_multivariate_sensitivity(
        merged_full, cluster_col=cluster_col
    )
    result["missing_karma_bias"] = check_missing_karma_bias(merged_full)
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
        res = analyze_method(
            method_name,
            user_scores,
            characteristics,
            min_n_signal=thr,
            cluster_col=cluster_col,
            exclude_suspended=exclude_suspended,
        )
        per_threshold[thr] = {
            col: (c["spearman_rho"] if c else None) for col, c in res["correlations"].items()
        }
    return per_threshold


def run_multivariate_analysis(
    merged: pd.DataFrame,
    cluster_col: Optional[str] = None,
) -> Dict:
    """Fit standardized OLS with HC3 or user-clustered standard errors."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return {"available": False, "reason": "statsmodels is not installed"}

    needed = ["score_primary", *CHARACTERISTIC_COLS]
    if cluster_col:
        needed.append(cluster_col)
    frame = merged[needed].dropna().copy()
    if len(frame) < max(30, len(CHARACTERISTIC_COLS) * 5):
        return {
            "available": True,
            "fitted": False,
            "n": len(frame),
            "reason": "too few complete cases",
        }

    varying = [column for column in CHARACTERISTIC_COLS if frame[column].nunique() > 1]
    x = frame[varying].astype(float)
    x = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(frame["score_primary"].astype(float), x)
    if cluster_col:
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster_col]})
        covariance = f"clustered_by_{cluster_col}"
    else:
        fitted = model.fit(cov_type="HC3")
        covariance = "HC3"

    confidence = fitted.conf_int()
    coefficients = {
        column: {
            "beta_standardized": float(fitted.params[column]),
            "std_error": float(fitted.bse[column]),
            "p_value": float(fitted.pvalues[column]),
            "ci_low": float(confidence.loc[column, 0]),
            "ci_high": float(confidence.loc[column, 1]),
        }
        for column in varying
    }
    vif = {
        column: float(variance_inflation_factor(x[varying].to_numpy(), index))
        for index, column in enumerate(varying)
    }
    return {
        "available": True,
        "fitted": True,
        "n": int(len(frame)),
        "n_users": int(frame[cluster_col].nunique()) if cluster_col else int(len(frame)),
        "covariance": covariance,
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "coefficients": coefficients,
        "vif": vif,
    }


def run_multivariate_sensitivity(
    merged: pd.DataFrame,
    thresholds: list = N_SIGNAL_THRESHOLDS,
    cluster_col: Optional[str] = None,
) -> Dict:
    """Repeat multivariate analysis after minimum-signal filtering."""
    if "n_signal" not in merged.columns:
        return {}
    return {
        str(threshold): run_multivariate_analysis(
            merged[merged["n_signal"] >= threshold], cluster_col=cluster_col
        )
        for threshold in thresholds
    }


def check_missing_karma_bias(merged: pd.DataFrame) -> Dict:
    """Compare method reliability for users with and without observed karma."""
    known = merged.loc[merged["total_karma"].notna(), "score_primary"].dropna()
    missing = merged.loc[merged["total_karma"].isna(), "score_primary"].dropna()
    total = len(known) + len(missing)
    result = {
        "n_known": int(len(known)),
        "n_missing": int(len(missing)),
        "missing_rate": float(len(missing) / total) if total else None,
    }
    if len(known) < 10 or len(missing) < 10:
        return {**result, "tested": False, "reason": "too few observations in one group"}
    statistic, p_value = stats.mannwhitneyu(known, missing, alternative="two-sided")
    rank_biserial = 2.0 * statistic / (len(known) * len(missing)) - 1.0
    return {
        **result,
        "tested": True,
        "known_median": float(known.median()),
        "missing_median": float(missing.median()),
        "mann_whitney_u": float(statistic),
        "p_value": float(p_value),
        "rank_biserial_known_vs_missing": float(rank_biserial),
    }


def print_sensitivity_report(method_name: str, sensitivity: Dict) -> None:
    """Print sensitivity report to the console."""
    if not sensitivity:
        return
    print("  Sensitivity to min_n_signal (Spearman rho at each threshold):")
    thresholds = sorted(sensitivity.keys())
    for col in CHARACTERISTIC_COLS:
        row = [sensitivity[t].get(col) for t in thresholds]
        row_str = "  ".join(
            f"n>={t}:{v:+.3f}" if v is not None else f"n>={t}:  n/a"
            for t, v in zip(thresholds, row)
        )
        print(f"    {col:20s}  {row_str}")


# Apply Benjamini-Hochberg once across all method-characteristic tests.


def benjamini_hochberg(pvals: list) -> list:
    """Adjust p-values with the Benjamini-Hochberg procedure."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value down.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out.tolist()


def apply_fdr_correction(all_results: Dict) -> None:
    """Apply fdr correction to the supplied data."""
    entries = []
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
    clustered_tag = " | clustered" if result.get("clustered") else ""
    susp_tag = " | suspended excluded" if result.get("exclude_suspended") else ""
    print(
        f"{result['method']}: scored={result['n_rows_with_score']:,} | "
        f"matched={result['n_rows_matched_to_characteristics']:,} "
        f"({result['coverage_pct']}%) | "
        f"filtered={result.get('n_rows_after_signal_filter', result['n_rows_matched_to_characteristics']):,}"
        f"{clustered_tag}{susp_tag}"
    )

    if not result["correlations"]:
        print(f"{result['method']}: no matched users")
        return

    for col, c in result["correlations"].items():
        if c is None:
            print(f"  {col}: skipped (n<10)")
        elif c.get("bootstrap_failed"):
            print(
                f"  {col}: rho={c['spearman_rho']} | bootstrap unavailable | "
                f"clusters={c['n_clusters']}"
            )
        elif c["p_value"] is None:
            print(f"  {col}: rho={c['spearman_rho']:+.4f} | p=n/a | n={c['n']}")
        else:
            q_str = (
                f" | q={c['q_value']:.4g} | FDR-significant={c.get('significant_fdr', False)}"
                if "q_value" in c
                else ""
            )
            ci_str = (
                f" | CI95=[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] | clusters={c['n_clusters']}"
                if c.get("clustered")
                else ""
            )
            print(
                f"  {col}: rho={c['spearman_rho']:+.4f} | p={c['p_value']:.4g} | "
                f"n={c['n']}{q_str}{ci_str}"
            )


# CLI


def main() -> None:
    """Run the command-line workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--votes-dir",
        type=str,
        default="data/splits/reddit",
        help="Root dir with users.parquet / splits (only used for reference paths)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="intersection",
        help="Canonical split label, e.g. full, intersection, "
        "intersection_chronological, or windows/full/w060.",
    )
    ap.add_argument(
        "--user-metadata",
        type=str,
        default="data/processed/user_metadata.csv",
        help="Path to Shayan's user_metadata.csv (default: data/processed/user_metadata.csv, "
        "same path used by normvio_skill_extraction.py)",
    )
    ap.add_argument(
        "--causal-features",
        type=str,
        default="data/interim/reddit/features/causal_user_vote_features.parquet",
    )
    ap.add_argument(
        "--sef-mode",
        choices=["auto", *SEF_MODES],
        default="auto",
        help="SEF metadata mode; auto selects it using validation macro-F1.",
    )
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument(
        "--cache-characteristics",
        action="store_true",
        help="Cache the merged user-characteristics table to speed up repeated runs",
    )
    ap.add_argument(
        "--min-n-signal",
        type=int,
        default=0,
        help="Minimum n_signal (votes backing a user's score) required to keep that "
        "user in the main correlation/quartile analysis. 0 = no filter.",
    )
    ap.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the min_n_signal sensitivity sweep (faster, less robust)",
    )
    ap.add_argument(
        "--exclude-suspended",
        action="store_true",
        help="Exclude suspended accounts before matching.",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"T2: split={args.split} | exclude_suspended={args.exclude_suspended}")

    split_path = Path(args.votes_dir) / args.split
    train_votes_path = split_path / "train_votes.parquet"
    cache_name = f"user_characteristics_{args.split.replace('/', '_')}.parquet"
    cache_path = output_dir / cache_name if args.cache_characteristics else None
    characteristics = build_user_characteristics(
        Path(args.user_metadata),
        Path(args.causal_features),
        train_votes_path,
        cache_path,
    )
    timestamp_semantics = read_timestamp_semantics(Path(args.causal_features))

    all_results = {}
    for method_name, (base_dir, loader, cluster_col) in METHOD_LOADERS.items():
        split_dir, selected_sef_mode = resolve_method_result_dir(
            method_name, base_dir, args.split, args.sef_mode
        )
        if not split_dir.exists():
            print(f"Skipped {method_name}: directory missing at {split_dir}")
            continue

        user_scores = loader(split_dir, split_path / "test_votes.parquet")
        if user_scores is None or user_scores.empty:
            print(f"Skipped {method_name}: no usable per-user scores at {split_dir}")
            continue

        result = analyze_method(
            method_name,
            user_scores,
            characteristics,
            min_n_signal=args.min_n_signal,
            cluster_col=cluster_col,
            exclude_suspended=args.exclude_suspended,
        )
        result["timestamp_semantics"] = timestamp_semantics
        result["temporal_estimand"] = "user state at the dataset reference timestamp"
        if selected_sef_mode is not None:
            result["selected_sef_mode"] = selected_sef_mode
        if not args.skip_sensitivity:
            sensitivity = run_signal_sensitivity(
                method_name,
                user_scores,
                characteristics,
                cluster_col=cluster_col,
                exclude_suspended=args.exclude_suspended,
            )
            result["n_signal_sensitivity"] = sensitivity

        all_results[method_name] = result

    # Correct the complete family of method-characteristic tests.
    apply_fdr_correction(all_results)
    for result in all_results.values():
        print_method_report(result)

    summary_path = output_dir / f"t1_summary_{args.split.replace('/', '_')}.json"
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"T2 results: {summary_path}")


if __name__ == "__main__":
    main()
