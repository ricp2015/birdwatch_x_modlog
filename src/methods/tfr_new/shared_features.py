"""
graph_propagation.py — Opzione 1 (graph propagation) del team_formation_ideas.md.

Costruisce un grafo di co-voto tra utenti (arco pesato per quante volte
votano sugli stessi item, rinforzato se concordi), propaga 'prossimità al
moderatore' con Personalized PageRank (seed = utenti storicamente affidabili),
poi aggrega i voti per item con pesi softmax sullo score PPR — stessa logica
di aggregazione di bandit_dr.py, per confrontabilità diretta.

Fedele al documento originale: qui NON c'è apprendimento online per-caso
(a differenza del bandit) — il punteggio di fiducia è statico per utente,
calcolato una volta su tutto il train, poi congelato per val/test.

Uso:
    python graph_propagation.py --votes-dir data/splits_intersection --out-dir results/graph_propagation/splits_intersection
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from shared_features import compute_ppr_scores, calibrate_thresholds_per_community, apply_thresholds

DEFAULT_VOTES_DIR = "data/splits_intersection"
DEFAULT_OUT_DIR = "results/graph_propagation/splits_intersection"
RELIABILITY_SEED_THR = 0.6


def aggregate_with_ppr(votes: pd.DataFrame, ppr_scores: dict) -> pd.DataFrame:
    votes = votes.copy()
    global_median = np.median(list(ppr_scores.values())) if ppr_scores else 0.0
    votes["_trust"] = votes["username"].map(ppr_scores).fillna(global_median)

    def agg(g):
        w = np.exp(g["_trust"] - g["_trust"].max())
        w = w / w.sum()
        score = float((w * g["vote"]).sum())  # continuo in [-1, 1], NON ancora sogliato
        return pd.Series({"label": g["label"].iloc[0], "community": g["community"].iloc[0],
                           "score": score, "n_voters": len(g)})

    return votes.groupby("item_id").apply(agg).reset_index()


def compute_metrics(preds: pd.DataFrame) -> dict:
    y_true, y_pred = preds["label"], preds["y_hat"]
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1),
        "n_items": len(preds),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(votes_dir / "train_votes.parquet").sort_values("timestamp")
    val = pd.read_parquet(votes_dir / "val_votes.parquet").sort_values("timestamp")
    test = pd.read_parquet(votes_dir / "test_votes.parquet").sort_values("timestamp")

    print("Costruzione grafo di co-voto + Personalized PageRank su train...")
    ppr_scores = compute_ppr_scores(train, reliability_thr=RELIABILITY_SEED_THR)

    print("Aggregazione train/val/test (score continuo)...")
    train_scores = aggregate_with_ppr(train, ppr_scores)
    val_scores = aggregate_with_ppr(val, ppr_scores)
    test_scores = aggregate_with_ppr(test, ppr_scores)

    print("Calibrazione soglia per-community su VAL...")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_scores)
    train_preds = apply_thresholds(train_scores, thresholds, global_threshold)
    val_preds = apply_thresholds(val_scores, thresholds, global_threshold)
    test_preds = apply_thresholds(test_scores, thresholds, global_threshold)

    results = {
        "train": compute_metrics(train_preds),
        "val": compute_metrics(val_preds),
        "test": compute_metrics(test_preds),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    test_preds.to_parquet(out_dir / "test_predictions.parquet", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nSalvato in {out_dir}")


if __name__ == "__main__":
    main()