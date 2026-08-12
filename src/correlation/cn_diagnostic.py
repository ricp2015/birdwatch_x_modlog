"""Test whether Community Notes reliability varies across communities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the user characteristics built for T1.
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
    print(f"  Loaded votes: {len(votes):,} | users: {votes['username'].nunique():,} | "
          f"community: {votes['community'].nunique():,}")
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
    d = d.merge(characteristics[["username", "n_communities", "subreddit_entropy"]],
                on="username", how="left")
    median_div = d["n_communities"].median()
    d["diversity_group"] = np.where(d["n_communities"] >= median_div, "high", "low")
    return d


def analyze_dispersion(dispersion: pd.DataFrame, label: str) -> Dict:
    """Analyze dispersion and return summary statistics."""
    print()
    print(f"Comparison [{label}]: within-user dispersion by diversity")
    print(f"  Users: {len(dispersion):,}")
    summary = dispersion.groupby("diversity_group")[
        ["agreement_mean", "agreement_std", "agreement_range", "i_u"]
    ].agg(["mean", "median", "count"])
    print(summary.to_string())

    high = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_std"].dropna()
    low  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_std"].dropna()
    p_disp = None
    if len(high) > 10 and len(low) > 10:
        u_stat, p_disp = stats.mannwhitneyu(high, low, alternative="two-sided")
        print(f"\n  Mann-Whitney U for agreement_std: "
              f"p={p_disp:.4g}  (n_high={len(high)}, n_low={len(low)})")
        print(f"    median agreement_std: high={high.median():.4f}  low={low.median():.4f}")
    else:
        print("\n  [not enough observations for the dispersion test]")

    high_m = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_mean"].dropna()
    low_m  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_mean"].dropna()
    p_mean = None
    if len(high_m) > 10 and len(low_m) > 10:
        u_stat_m, p_mean = stats.mannwhitneyu(high_m, low_m, alternative="two-sided")
        print(f"  Mann-Whitney U for agreement_mean: "
              f"p={p_mean:.4g}  (n_high={len(high_m)}, n_low={len(low_m)})")
        print(f"    median agreement_mean: high={high_m.median():.4f}  low={low_m.median():.4f}")

    valid = dispersion[["agreement_std", "i_u"]].dropna()
    rho, p_corr = None, None
    if len(valid) > 10:
        rho, p_corr = stats.spearmanr(valid["agreement_std"], valid["i_u"])
        print(f"\n  Spearman(agreement_std, i_u): rho={rho:+.4f}  p={p_corr:.4g}  n={len(valid)}")

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
    raw_csv_path: Path,
    output_dir: Path,
    min_votes_robustness: int = 3,
) -> None:
    """Run the diagnostic workflow."""
    print(f"CN diversity diagnostic — split: {split}")

    split_path = votes_dir / split
    votes = load_votes_with_labels(split_path)

    print("\nComputing user-community agreement rates...")
    agg = compute_per_user_community_agreement(votes)

    user_params_path = cn_dir / split / "user_params.parquet"
    if not user_params_path.exists():
        raise FileNotFoundError(f"user_params.parquet not found: {user_params_path}")
    user_params = pd.read_parquet(user_params_path)[["username", "i_u", "f_u"]]

    print("Loading user characteristics...")
    characteristics = build_user_characteristics(user_metadata_path, raw_csv_path)

    print("\nVariant 1/2: no minimum community vote count")
    dispersion = compute_within_user_dispersion(agg, min_votes_per_community=1)
    dispersion = _merge_context(dispersion, user_params, characteristics)
    print(f"  Users with at least two communities: {len(dispersion):,}")

    print(f"\nVariant 2/2: communities with at least {min_votes_robustness} votes")
    dispersion_robust = compute_within_user_dispersion(agg, min_votes_per_community=min_votes_robustness)
    dispersion_robust = _merge_context(dispersion_robust, user_params, characteristics)
    print(f"  Users with at least two retained communities: {len(dispersion_robust):,}")

    summary_unfiltered = analyze_dispersion(dispersion, "unfiltered")
    summary_robust = analyze_dispersion(dispersion_robust, f"at least {min_votes_robustness} votes")

    print()
    print("Overall interpretation")
    if summary_unfiltered["supports_hypothesis_B"] and summary_robust["supports_hypothesis_B"]:
        print("  Dispersion remains after filtering sparse communities, supporting hypothesis B.")
    elif summary_unfiltered["supports_hypothesis_B"] and not summary_robust["supports_hypothesis_B"]:
        print("  Dispersion disappears after filtering sparse communities; hypothesis B is inconclusive.")
    elif not summary_unfiltered["supports_hypothesis_B"]:
        print("  No significant dispersion difference was found; hypothesis B is not supported.")

    if summary_unfiltered["also_supports_hypothesis_A"]:
        print("  Mean agreement is also lower for diverse users, so hypothesis A remains plausible.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}.parquet"
    dispersion.to_parquet(out_path, index=False)
    out_path_robust = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}_robust.parquet"
    dispersion_robust.to_parquet(out_path_robust, index=False)
    print(f"\nUser details saved to {out_path}")
    print(f"Robust variant saved to {out_path_robust}")


def main() -> None:
    """Run the command-line workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes-dir", type=str, default="data/splits/reddit")
    ap.add_argument("--split", type=str, default="splits")
    ap.add_argument("--cn-dir", type=str, default="results/reddit")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv")
    ap.add_argument("--raw-csv", type=str, default="data/processed/final_intersection_dataset.csv")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--min-votes-robustness", type=int, default=3,
                     help="Minimum votes per user-community pair in the robust variant")
    args = ap.parse_args()

    run_diagnostic(
        votes_dir=Path(args.votes_dir),
        split=args.split,
        cn_dir=Path(args.cn_dir),
        user_metadata_path=Path(args.user_metadata),
        raw_csv_path=Path(args.raw_csv),
        output_dir=Path(args.output_dir),
        min_votes_robustness=args.min_votes_robustness,
    )


if __name__ == "__main__":
    main()
