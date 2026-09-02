"""Test whether Community Notes reliability varies across communities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the user characteristics built for T2.
from t1 import build_user_characteristics


def load_votes_with_labels(split_dir: Path) -> pd.DataFrame:
    """Load votes with labels from its configured source."""
    dfs = []
    for name in ("train_votes", "val_votes", "test_votes"):
        p = split_dir / f"{name}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(f"No train, validation, or test votes found under {split_dir}")
    votes = pd.concat(dfs, ignore_index=True)
    return votes


def compute_per_user_community_agreement(votes: pd.DataFrame) -> pd.DataFrame:
    """Compute per user community agreement from the supplied data."""
    v = votes.dropna(subset=["label", "vote"]).copy()
    v["agree"] = (v["vote"] == v["label"]).astype(float)
    agg = (
        v.groupby(["username", "community"])
        .agg(agreement_rate=("agree", "mean"), n_votes=("agree", "count"))
        .reset_index()
    )
    return agg


def compute_within_user_dispersion(agg: pd.DataFrame, min_communities: int = 2,
                                    min_votes_per_community: int = 1) -> pd.DataFrame:
    """Compute within user dispersion from the supplied data."""
    a = agg[agg["n_votes"] >= min_votes_per_community] if min_votes_per_community > 1 else agg
    counts = a.groupby("username")["community"].nunique()
    multi_users = counts[counts >= min_communities].index
    sub = a[a["username"].isin(multi_users)]
    out = (
        sub.groupby("username")
        .agg(
            n_communities_in_votes=("community", "nunique"),
            agreement_mean=("agreement_rate", "mean"),
            agreement_std=("agreement_rate", "std"),
            agreement_range=("agreement_rate", lambda x: float(x.max() - x.min())),
            total_votes=("n_votes", "sum"),
        )
        .reset_index()
    )
    return out


def _merge_context(dispersion: pd.DataFrame, user_params: pd.DataFrame,
                    characteristics: pd.DataFrame) -> pd.DataFrame:
    """Merge context into the working data."""
    d = dispersion.merge(user_params, on="username", how="left")
    d = d.merge(characteristics[["username", "n_communities_prior"]],
                on="username", how="left")
    median_div = d["n_communities_prior"].median()
    d["diversity_group"] = np.where(
        d["n_communities_prior"] >= median_div, "high", "low"
    )
    return d


def analyze_dispersion(dispersion: pd.DataFrame, label: str) -> Dict:
    """Analyze dispersion and return summary statistics."""
    high = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_std"].dropna()
    low  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_std"].dropna()
    p_disp = None
    if len(high) > 10 and len(low) > 10:
        u_stat, p_disp = stats.mannwhitneyu(high, low, alternative="two-sided")

    high_m = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_mean"].dropna()
    low_m  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_mean"].dropna()
    p_mean = None
    if len(high_m) > 10 and len(low_m) > 10:
        u_stat_m, p_mean = stats.mannwhitneyu(high_m, low_m, alternative="two-sided")

    valid = dispersion[["agreement_std", "i_u"]].dropna()
    rho, p_corr = None, None
    if len(valid) > 10:
        rho, p_corr = stats.spearmanr(valid["agreement_std"], valid["i_u"])

    print(
        f"CN {label}: users={len(dispersion):,} | p-dispersion={p_disp} | "
        f"p-mean={p_mean} | rho={rho}"
    )

    supports_B = bool(p_disp is not None and p_disp < 0.05 and high.median() > low.median())
    also_A = bool(p_mean is not None and p_mean < 0.05 and high_m.median() < low_m.median())

    return {
        "label": label,
        "n": len(dispersion),
        "p_dispersion": p_disp,
        "p_mean_agreement": p_mean,
        "spearman_std_vs_iu": rho,
        "supports_hypothesis_B": supports_B,
        "also_supports_hypothesis_A": also_A,
    }


def run_diagnostic(
    votes_dir: Path,
    split: str,
    cn_dir: Path,
    user_metadata_path: Path,
    causal_features_path: Path,
    output_dir: Path,
    min_votes_robustness: int = 3,
) -> None:
    """Run the diagnostic workflow."""
    split_path = votes_dir / split
    votes = load_votes_with_labels(split_path)

    agg = compute_per_user_community_agreement(votes)

    user_params_path = cn_dir / split / "cn" / "user_params.parquet"
    if not user_params_path.exists():
        raise FileNotFoundError(f"user_params.parquet not found: {user_params_path}")
    user_params = pd.read_parquet(user_params_path)[["username", "i_u", "f_u"]]

    characteristics = build_user_characteristics(
        user_metadata_path,
        causal_features_path,
        split_path / "train_votes.parquet",
    )

    dispersion = compute_within_user_dispersion(agg, min_votes_per_community=1)
    dispersion = _merge_context(dispersion, user_params, characteristics)

    dispersion_robust = compute_within_user_dispersion(agg, min_votes_per_community=min_votes_robustness)
    dispersion_robust = _merge_context(dispersion_robust, user_params, characteristics)

    analyze_dispersion(dispersion, "unfiltered")
    analyze_dispersion(dispersion_robust, f"min-votes-{min_votes_robustness}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}.parquet"
    dispersion.to_parquet(out_path, index=False)
    out_path_robust = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}_robust.parquet"
    dispersion_robust.to_parquet(out_path_robust, index=False)
    print(f"CN diagnostic: {out_path} | robust={out_path_robust}")


def main() -> None:
    """Run the command-line workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes-dir", type=str, default="data/splits/reddit")
    ap.add_argument("--split", type=str, default="intersection")
    ap.add_argument("--cn-dir", type=str, default="results/reddit")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv")
    ap.add_argument("--causal-features", type=str,
                    default="data/interim/reddit/features/causal_user_vote_features.parquet")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--min-votes-robustness", type=int, default=3,
                     help="Minimum votes per user-community pair in the robust variant")
    args = ap.parse_args()

    run_diagnostic(
        votes_dir=Path(args.votes_dir),
        split=args.split,
        cn_dir=Path(args.cn_dir),
        user_metadata_path=Path(args.user_metadata),
        causal_features_path=Path(args.causal_features),
        output_dir=Path(args.output_dir),
        min_votes_robustness=args.min_votes_robustness,
    )


if __name__ == "__main__":
    main()
