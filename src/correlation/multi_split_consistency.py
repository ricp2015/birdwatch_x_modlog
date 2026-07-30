"""
t1_multi_split_consistency.py
==============================
Esegue l'analisi di correlazione univariata di T1 (analyze_method +
correzione FDR) su PIU' split, poi produce un'unica tabella di consistenza:
per ogni (metodo, caratteristica), segno e significativita' FDR della
correlazione su ciascuno split richiesto, piu' un riepilogo su se il
pattern e' consistente ovunque risulti significativo.

Perche' serve: ogni score_primary in T1 e' ricalcolato per ogni split (vedi
il docstring di t1_user_characteristics_analysis.py) — un pattern trovato
su un solo split ("splits") potrebbe essere un caso di quello split
casuale specifico, non un effetto reale. Prima di scrivere un risultato di
T1 in tesi, dovrebbe reggere questo controllo su almeno un altro split
(es. splits_full, o una finestra windowed_folds_full/w0XX).

NOTA: questo script confronta SOLO le correlazioni univariate (non la
regressione multivariata, non il diagnostico CN specifico) — per tenere il
confronto multi-split abbastanza rapido da poter essere lanciato su piu'
split senza aspettare il bootstrap TFR e la regressione ad ogni giro.

Uso
---
    python t1_multi_split_consistency.py \
        --splits splits splits_full windowed_folds_full/w060 \
        --user-metadata data/processed/user_metadata.csv \
        --raw-csv data/processed/final_intersection_dataset.csv \
        --output-dir results/t1_analysis
"""

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

SEP = "=" * 70


def run_one_split(
    split: str,
    characteristics: pd.DataFrame,
    min_n_signal: int = 0,
    exclude_suspended: bool = False,
) -> Dict:
    """Stessa logica del loop principale in t1_user_characteristics_analysis.py
    main(), ma senza stampare i report dettagliati (solo quello che serve
    per la tabella di consistenza) — la correzione FDR e' applicata PER
    SPLIT, indipendentemente per ciascuno, coerentemente con come gira
    normalmente lo script su un singolo split."""
    results = {}
    for method_name, (base_dir, loader, cluster_col) in METHOD_LOADERS.items():
        split_dir = Path(base_dir) / split
        if not split_dir.exists():
            print(f"  [{split}/{method_name}] split directory non trovata — saltato")
            continue
        user_scores = loader(split_dir)
        if user_scores is None or user_scores.empty:
            print(f"  [{split}/{method_name}] nessun file di dettaglio utilizzabile — saltato")
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
    """
    Per ogni (metodo, caratteristica), attraverso tutti gli split presenti:
      - n_splits_tested: su quanti split e' stato testato
      - n_splits_significant: quanti erano FDR-significativi
      - direction: '+' / '-' / 'MISTO' / 'mai significativo'
      - sign_consistent: True se tutte le correlazioni SIGNIFICATIVE hanno
        lo stesso segno (None se non e' mai risultato significativo)
    """
    if table.empty:
        return pd.DataFrame()

    summary_rows = []
    for (method, col), g in table.groupby(["method", "characteristic"]):
        sig = g[g["significant_fdr"]]
        if sig.empty:
            direction = "mai significativo"
            sign_consistent = None
        else:
            signs = set(sig["rho"].apply(lambda x: "+" if x > 0 else "-"))
            sign_consistent = len(signs) == 1
            direction = signs.pop() if sign_consistent else "MISTO"
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
    if summary.empty:
        print("  [nessun risultato da riassumere]")
        return
    print(f"\n{'method':6s} {'characteristic':20s} {'tested':>7s} {'signif':>7s}  "
          f"{'direction':17s} {'consistent':>10s}  {'rho range':16s}")
    for _, row in summary.iterrows():
        cons = "n/a" if row["sign_consistent"] is None else ("SI" if row["sign_consistent"] else "NO")
        flag = "  <-- ATTENZIONE, segno instabile" if row["sign_consistent"] is False else ""
        print(f"{row['method']:6s} {row['characteristic']:20s} {row['n_splits_tested']:>7d} "
              f"{row['n_splits_significant']:>7d}  {row['direction']:17s} {cons:>10s}  "
              f"[{row['rho_min']:+.3f}, {row['rho_max']:+.3f}]{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True,
                     help="Split da confrontare, es: splits splits_full windowed_folds_full/w060")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv")
    ap.add_argument("--raw-csv", type=str, default="data/processed/final_intersection_dataset.csv")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--min-n-signal", type=int, default=0)
    ap.add_argument("--exclude-suspended", action="store_true")
    ap.add_argument("--cache-characteristics", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"T1 multi-split consistency check — split confrontati: {args.splits}")
    print(SEP)

    cache_path = output_dir / "user_characteristics_cache.parquet" if args.cache_characteristics else None
    characteristics = build_user_characteristics(Path(args.user_metadata), Path(args.raw_csv), cache_path)

    per_split_results = {}
    for split in args.splits:
        print(f"\n{SEP}\nSPLIT: {split}\n{SEP}")
        per_split_results[split] = run_one_split(
            split, characteristics,
            min_n_signal=args.min_n_signal, exclude_suspended=args.exclude_suspended,
        )

    table = build_consistency_table(per_split_results)
    summary = summarize_consistency(table)

    print(f"\n{SEP}")
    print("TABELLA DI CONSISTENZA (per metodo x caratteristica, attraverso gli split)")
    print(SEP)
    print_summary(summary)

    table_path = output_dir / "t1_multi_split_raw.csv"
    summary_path = output_dir / "t1_multi_split_summary.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nDettaglio grezzo salvato -> {table_path}")
    print(f"Riepilogo di consistenza salvato -> {summary_path}")


if __name__ == "__main__":
    main()