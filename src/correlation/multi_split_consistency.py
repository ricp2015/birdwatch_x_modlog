"""Check whether user-characteristic correlations remain consistent across splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from t1 import (
    METHOD_LOADERS,
    build_user_characteristics,
    analyze_method,
    apply_fdr_correction,
)



def run_one_split(
    split: str,
    characteristics: pd.DataFrame,
    min_n_signal: int = 0,
    exclude_suspended: bool = False,
) -> Dict:
    """Run the one split workflow."""
    results = {}
    for method_name, (base_dir, loader, cluster_col) in METHOD_LOADERS.items():
        split_dir = Path(base_dir) / split
        if not split_dir.exists():
            print(f"  [{split}/{method_name}] split directory not found; skipped")
            continue
        user_scores = loader(split_dir)
        if user_scores is None or user_scores.empty:
            print(f"  [{split}/{method_name}] no usable detail file; skipped")
            continue
        result = analyze_method(
            method_name, user_scores, characteristics,
            min_n_signal=min_n_signal, cluster_col=cluster_col,
            exclude_suspended=exclude_suspended,
        )
        results[method_name] = result
    apply_fdr_correction(results)
    return results


def build_consistency_table(per_split_results: Dict[str, Dict]) -> pd.DataFrame:
    """Build consistency table from the supplied data."""
    rows = []
    for split, method_results in per_split_results.items():
        for method_name, result in method_results.items():
            for col, c in result.get("correlations", {}).items():
                if c is None or c.get("p_value") is None:
                    continue
                rows.append({
                    "split": split,
                    "method": method_name,
                    "characteristic": col,
                    "rho": c["spearman_rho"],
                    "p_value": c.get("p_value"),
                    "q_value": c.get("q_value"),
                    "significant_fdr": bool(c.get("significant_fdr", False)),
                    "n": c.get("n"),
                })
    return pd.DataFrame(rows)


def summarize_consistency(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize consistency across groups."""
    if table.empty:
        return pd.DataFrame()

    summary_rows = []
    for (method, col), g in table.groupby(["method", "characteristic"]):
        sig = g[g["significant_fdr"]]
        if sig.empty:
            direction = "never significant"
            sign_consistent = None
        else:
            signs = set(sig["rho"].apply(lambda x: "+" if x > 0 else "-"))
            sign_consistent = len(signs) == 1
            direction = signs.pop() if sign_consistent else "mixed"
        summary_rows.append({
            "method": method,
            "characteristic": col,
            "n_splits_tested": len(g),
            "n_splits_significant": len(sig),
            "direction": direction,
            "sign_consistent": sign_consistent,
            "rho_min": round(float(g["rho"].min()), 3),
            "rho_max": round(float(g["rho"].max()), 3),
        })
    return pd.DataFrame(summary_rows).sort_values(["method", "characteristic"]).reset_index(drop=True)


def print_summary(summary: pd.DataFrame) -> None:
    """Print summary to the console."""
    if summary.empty:
        print("  [no results to summarize]")
        return
    print(f"\n{'method':6s} {'characteristic':20s} {'tested':>7s} {'signif':>7s}  "
          f"{'direction':17s} {'consistent':>10s}  {'rho range':16s}")
    for _, row in summary.iterrows():
        cons = "n/a" if row["sign_consistent"] is None else ("yes" if row["sign_consistent"] else "no")
        flag = "  [unstable sign]" if row["sign_consistent"] is False else ""
        print(f"{row['method']:6s} {row['characteristic']:20s} {row['n_splits_tested']:>7d} "
              f"{row['n_splits_significant']:>7d}  {row['direction']:17s} {cons:>10s}  "
              f"[{row['rho_min']:+.3f}, {row['rho_max']:+.3f}]{flag}")


def main() -> None:
    """Run the command-line workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True,
                     help="Splits to compare")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv")
    ap.add_argument("--raw-csv", type=str, default="data/processed/final_intersection_dataset.csv")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--min-n-signal", type=int, default=0)
    ap.add_argument("--exclude-suspended", action="store_true")
    ap.add_argument("--cache-characteristics", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"T1 multi-split consistency check: {args.splits}")

    cache_path = output_dir / "user_characteristics_cache.parquet" if args.cache_characteristics else None
    characteristics = build_user_characteristics(Path(args.user_metadata), Path(args.raw_csv), cache_path)

    per_split_results = {}
    for split in args.splits:
        print(f"\nSplit: {split}")
        per_split_results[split] = run_one_split(
            split, characteristics,
            min_n_signal=args.min_n_signal, exclude_suspended=args.exclude_suspended,
        )

    table = build_consistency_table(per_split_results)
    summary = summarize_consistency(table)

    print()
    print("Consistency by method and characteristic")
    print_summary(summary)

    table_path = output_dir / "t1_multi_split_raw.csv"
    summary_path = output_dir / "t1_multi_split_summary.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nRaw details saved to {table_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
