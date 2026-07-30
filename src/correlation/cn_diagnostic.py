"""
cn_diversity_diagnostic.py
===========================
Diagnostica mirata sul pattern CN osservato in T1: gli utenti con alta
diversita' cross-subreddit hanno i_u che collassa verso zero (quartile piu'
diversificato: i_u medio ~0.0003, quasi zero in valore assoluto), mentre gli
utenti poco diversificati hanno i_u chiaramente positivo.

Due letture possibili, non distinguibili dal solo i_u:

  (A) "Utente meno affidabile": chi vota su tante community eterogenee e'
      genuinamente meno preciso nel rispecchiare le decisioni dei moderatori.

  (B) "Limite strutturale della MF": CN stima UN SOLO fattore/intercetta per
      utente (numFactors=1, un i_u globale). Se la reliability vera di un
      utente e' alta in una community e bassa in un'altra (specificita' per
      community, non un tratto globale), un singolo i_u non puo' rappresentare
      questa variabilita' — puo' finire vicino a zero anche se l'utente e'
      "bravo" ovunque, semplicemente perche' la MF media/comprime segnali
      contrastanti in un solo numero.

Questo script distingue le due ipotesi calcolando, DIRETTAMENTE dai voti
grezzi (bypassando la MF), il tasso di accordo utente-moderatore per ogni
(utente, community) — poi confronta:

  1. La DISPERSIONE di questo tasso di accordo tra le community di uno
     stesso utente (agreement_std). Se l'ipotesi (B) e' corretta, gli utenti
     diversificati dovrebbero avere agreement_std molto piu' alta (la loro
     reliability "vera" varia parecchio a seconda della community) — non
     necessariamente un agreement_mean piu' basso.
  2. Se agreement_std e' effettivamente piu' alta per gli utenti diversificati
     E correlata con un i_u basso, e' un indizio concreto a favore di (B).
  3. Se invece anche l'agreement_mean (non solo la dispersione) e' piu' basso
     per gli utenti diversificati, l'ipotesi (A) resta plausibile insieme
     alla (B) — le due non si escludono a vicenda.

Uso
---
    python cn_diversity_diagnostic.py \
        --votes-dir results/step1/reddit \
        --split splits \
        --cn-dir results/step2/reddit_cn/benchmark_by_split \
        --user-metadata data/processed/user_metadata.csv \
        --raw-csv data/processed/final_intersection_dataset.csv \
        --output-dir results/t1_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# riusa la costruzione delle caratteristiche utente gia' validata in T1,
# invece di duplicarla — stessa fonte (Shayan + raw CSV), stesso significato
from t1 import build_user_characteristics

SEP = "=" * 70


def load_votes_with_labels(split_dir: Path) -> pd.DataFrame:
    """Concatena train/val/test votes di uno split (serve solo vote+label
    grezzi, non le predizioni di nessun metodo)."""
    dfs = []
    for name in ("train_votes", "val_votes", "test_votes"):
        p = split_dir / f"{name}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(f"Nessun train/val/test_votes.parquet trovato sotto {split_dir}")
    votes = pd.concat(dfs, ignore_index=True)
    print(f"  Voti caricati: {len(votes):,} | utenti: {votes['username'].nunique():,} | "
          f"community: {votes['community'].nunique():,}")
    return votes


def compute_per_user_community_agreement(votes: pd.DataFrame) -> pd.DataFrame:
    """
    Tasso di accordo grezzo (vote == label) per ogni (username, community),
    calcolato DIRETTAMENTE dai voti — nessun passaggio attraverso la MF di
    CN. Questa e' la misura "verita' a terra" con cui confrontare i_u.
    """
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
    """
    Per ogni utente con voti in >= min_communities community distinte in
    questo split, calcola la dispersione (std, range) del tasso di accordo
    tra le sue community — la metrica chiave per l'ipotesi (B).

    min_votes_per_community: se >1, scarta prima le righe (utente,
    community) con meno di questa soglia di voti. Motivo: un tasso di
    accordo calcolato su 1-2 voti in una community e' esso stesso rumoroso
    — un utente potrebbe sembrare "disperso tra community" solo perche' una
    delle sue community ha una stima di accordo instabile per pochi dati,
    non perche' la sua reliability vari davvero. Confrontare i risultati
    con min_votes_per_community=1 (default, nessun filtro) e un valore piu'
    alto (es. 3) e' il modo per distinguere le due cose.

    Utenti con una sola community sono esclusi qui per costruzione (non si
    puo' misurare varianza cross-community su un solo punto), ma restano
    nell'analisi principale di T1.
    """
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
    d = dispersion.merge(user_params, on="username", how="left")
    d = d.merge(characteristics[["username", "n_communities", "subreddit_entropy"]],
                on="username", how="left")
    median_div = d["n_communities"].median()
    d["diversity_group"] = np.where(d["n_communities"] >= median_div, "high", "low")
    return d


def analyze_dispersion(dispersion: pd.DataFrame, label: str) -> Dict:
    """Runs the high-vs-low-diversity comparison on one dispersion table
    (either the unfiltered one or the min-votes-per-community robustness
    variant) and returns a small summary dict for the final side-by-side
    comparison, in addition to printing the detail."""
    print(f"\n{SEP}")
    print(f"CONFRONTO [{label}]: dispersione within-user, alta vs bassa diversita'")
    print(SEP)
    print(f"  n utenti in questa variante: {len(dispersion):,}")
    summary = dispersion.groupby("diversity_group")[
        ["agreement_mean", "agreement_std", "agreement_range", "i_u"]
    ].agg(["mean", "median", "count"])
    print(summary.to_string())

    high = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_std"].dropna()
    low  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_std"].dropna()
    p_disp = None
    if len(high) > 10 and len(low) > 10:
        u_stat, p_disp = stats.mannwhitneyu(high, low, alternative="two-sided")
        print(f"\n  Mann-Whitney U su agreement_std (alta vs bassa diversita'): "
              f"p={p_disp:.4g}  (n_high={len(high)}, n_low={len(low)})")
        print(f"    mediana agreement_std: high={high.median():.4f}  low={low.median():.4f}")
    else:
        print("\n  [n troppo basso per il test Mann-Whitney sulla dispersione]")

    high_m = dispersion.loc[dispersion["diversity_group"] == "high", "agreement_mean"].dropna()
    low_m  = dispersion.loc[dispersion["diversity_group"] == "low",  "agreement_mean"].dropna()
    p_mean = None
    if len(high_m) > 10 and len(low_m) > 10:
        u_stat_m, p_mean = stats.mannwhitneyu(high_m, low_m, alternative="two-sided")
        print(f"  Mann-Whitney U su agreement_mean (alta vs bassa diversita'): "
              f"p={p_mean:.4g}  (n_high={len(high_m)}, n_low={len(low_m)})")
        print(f"    mediana agreement_mean: high={high_m.median():.4f}  low={low_m.median():.4f}")

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
    print(SEP)
    print(f"CN diversity diagnostic — split: {split}")
    print(SEP)

    split_path = votes_dir / split
    votes = load_votes_with_labels(split_path)

    print("\nCalcolo tasso di accordo grezzo per (utente, community) dai voti...")
    agg = compute_per_user_community_agreement(votes)

    user_params_path = cn_dir / split / "user_params.parquet"
    if not user_params_path.exists():
        raise FileNotFoundError(f"user_params.parquet non trovato: {user_params_path}")
    user_params = pd.read_parquet(user_params_path)[["username", "i_u", "f_u"]]

    print("Caricamento caratteristiche utente (stessa fonte usata in T1)...")
    characteristics = build_user_characteristics(user_metadata_path, raw_csv_path)

    print("\nVariante 1/2: nessun filtro sul conteggio voti per community...")
    dispersion = compute_within_user_dispersion(agg, min_votes_per_community=1)
    dispersion = _merge_context(dispersion, user_params, characteristics)
    print(f"  Utenti con >=2 community in questo split: {len(dispersion):,}")

    print(f"\nVariante 2/2: solo community con >={min_votes_robustness} voti "
          f"(controllo robustezza — esclude che la dispersione sia rumore da poche osservazioni)...")
    dispersion_robust = compute_within_user_dispersion(agg, min_votes_per_community=min_votes_robustness)
    dispersion_robust = _merge_context(dispersion_robust, user_params, characteristics)
    print(f"  Utenti con >=2 community 'affidabili': {len(dispersion_robust):,}")

    summary_unfiltered = analyze_dispersion(dispersion, "senza filtro")
    summary_robust = analyze_dispersion(dispersion_robust, f"solo community >={min_votes_robustness} voti")

    print(f"\n{SEP}")
    print("INTERPRETAZIONE COMPLESSIVA (confronto tra le due varianti)")
    print(SEP)
    if summary_unfiltered["supports_hypothesis_B"] and summary_robust["supports_hypothesis_B"]:
        print("  -> Il pattern di dispersione REGGE anche filtrando le community con pochi voti: "
              "non e' spiegabile solo da stime rumorose. Supporto solido all'ipotesi (B) — "
              "un i_u singolo non cattura una reliability che varia davvero per community.")
    elif summary_unfiltered["supports_hypothesis_B"] and not summary_robust["supports_hypothesis_B"]:
        print("  -> Il pattern e' presente SENZA filtro ma SPARISCE filtrando le community con "
              "pochi voti: e' plausibile che la dispersione osservata fosse in parte un artefatto "
              "di stime instabili su community con pochi dati, non reliability realmente variabile. "
              "Ipotesi (B) NON confermata in modo solido.")
    elif not summary_unfiltered["supports_hypothesis_B"]:
        print("  -> Nessuna evidenza di dispersione significativamente piu' alta per gli utenti "
              "diversificati, ne' con ne' senza filtro. Ipotesi (B) non supportata da questo "
              "controllo — il pattern osservato in T1 (i_u vicino a zero per alta diversita') "
              "resta da spiegare con altri meccanismi.")

    if summary_unfiltered["also_supports_hypothesis_A"]:
        print("  -> In entrambe le varianti l'accordo medio (non solo la dispersione) e' piu' basso "
              "per gli utenti diversificati — l'ipotesi (A) resta plausibile IN AGGIUNTA alla (B).")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}.parquet"
    dispersion.to_parquet(out_path, index=False)
    out_path_robust = output_dir / f"cn_diversity_diagnostic_{split.replace('/', '_')}_robust.parquet"
    dispersion_robust.to_parquet(out_path_robust, index=False)
    print(f"\nDettaglio per utente salvato -> {out_path}")
    print(f"Variante robusta salvata -> {out_path_robust}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes-dir", type=str, default="results/step1/reddit")
    ap.add_argument("--split", type=str, default="splits")
    ap.add_argument("--cn-dir", type=str, default="results/step2/reddit_cn/benchmark_by_split")
    ap.add_argument("--user-metadata", type=str, default="data/processed/user_metadata.csv")
    ap.add_argument("--raw-csv", type=str, default="data/processed/final_intersection_dataset.csv")
    ap.add_argument("--output-dir", type=str, default="results/t1_analysis")
    ap.add_argument("--min-votes-robustness", type=int, default=3,
                     help="Soglia minima di voti per (utente, community) nella variante di "
                          "robustezza — esclude che la dispersione sia solo rumore da poche osservazioni")
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