"""
bandit_dr.py — Opzione 3 (bandit + DR) del team_formation_ideas.md, prototipo
standalone.

Design scelto (vedi discussione):
- LinUCB IBRIDO (Li et al. 2010): componente condivisa theta_0 (stimata sulle
  feature del caso x_c) + componente per-utente theta_i (si affina con le
  osservazioni di quell'utente). Un utente sparso eredita comunque un punteggio
  sensato dalla parte condivisa.
- "Online" = replay in ordine CRONOLOGICO dei voti di TRAIN. Ogni (utente,
  caso) e' trattato come una pull separata. Reward = il voto di quell'utente
  concorda con la decisione storica del moderatore (non con l'aggregato
  y_hat_c, che dipende da pesi ancora in apprendimento -> eviterebbe
  circolarita').
- Aggregazione finale per caso: softmax dei trust score -> voto pesato -> segno.
- Val/Test: NESSUN ulteriore update online. I parametri vengono congelati alla
  fine del replay di train (stesso protocollo train-only degli altri metodi).
- Split: usa gli split GIA' pronti (mai generare nuovi split). Ordina
  cronologicamente train (e, separatamente, val/test per coerenza) prima di
  qualunque elaborazione.
- Correzione Doubly Robust: SOLO in fase di valutazione (non guida il
  training). p_c (propensity) e' approssimato — la probabilita' storica che
  ESATTAMENTE quell'insieme di votanti V_c si formi e' quasi sempre ~0 (gli
  insiemi di votanti quasi mai si ripetono identici), quindi p_c viene stimato
  con un modello di propensity più grezzo: P(azione maggioritaria osservata |
  community, profilo aggregato di reliability dei votanti). Questo è
  un'approssimazione dichiarata, non l'esatta definizione letterale del
  documento originale — da segnalare in tesi.

Uso:
    python bandit_dr.py --votes-dir data/splits_intersection --out-dir results/bandit_dr/splits_intersection

Assunzioni sullo schema dei file (adatta ai nomi reali se diversi):
    train_votes.parquet / val_votes.parquet / test_votes.parquet:
        username, item_id, vote (+1/-1), timestamp, community
    posts.parquet (o colonna 'label' già dentro i vote file):
        item_id, label (+1 approve / -1 remove), community
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shared_features import calibrate_thresholds_per_community, apply_thresholds

RIDGE_LAMBDA = 1.0        # regolarizzazione A_0, A_i (I * lambda)
UCB_ALPHA = 0.6           # peso del termine di confidenza (esplorazione)
MIN_SUB_SAMPLES = 30      # sotto questa soglia, propensity/model fallback globale


# ---------------------------------------------------------------------------
# Feature per (utente, caso) — phi(i, c) = [x_c || u_{i,s}]
# Volutamente minimale per il primo prototipo: si estende dopo con lo score
# di graph propagation (Opzione 1), una volta pronto.
# ---------------------------------------------------------------------------

def build_context_features(posts: pd.DataFrame, running_stats: dict) -> pd.DataFrame:
    """x_c: feature del caso, causali (solo statistiche note FINO a quel punto
    nel replay). running_stats viene aggiornato incrementalmente dal replay,
    non da tutto il train in blocco, per evitare leakage temporale entro lo
    stesso split di training."""
    # placeholder strutturale: il replay aggiorna running_stats per community
    # (n_posts_visti, approve_rate_visto) e li inietta riga per riga.
    return posts


def user_feature_vector(user_state: dict, community: str, base_dim: int) -> np.ndarray:
    """u_{i,s}: stato per-utente-community (reliability corrente, n_voti
    visti, direzione prevalente). Vettore fisso, zero-padded se nuovo."""
    key = (user_state.get("username"), community)
    st = user_state["by_user_sub"].get(key)
    if st is None:
        return np.zeros(base_dim)
    n = max(st["n"], 1)
    reliability = st["agree"] / n
    return np.array([reliability, min(np.log1p(n), 5.0), st["last_vote"]])


# ---------------------------------------------------------------------------
# LinUCB ibrido — replay cronologico
# ---------------------------------------------------------------------------

class HybridLinUCB:
    def __init__(self, d_shared: int, d_user: int, alpha: float = UCB_ALPHA,
                 lam: float = RIDGE_LAMBDA):
        self.alpha = alpha
        self.d_shared = d_shared
        self.d_user = d_user
        self.A0 = lam * np.eye(d_shared)
        self.b0 = np.zeros(d_shared)
        self.per_user = {}  # username -> dict(A, B, Ainv_cached lazily, b)

    def _user_state(self, username: str):
        if username not in self.per_user:
            self.per_user[username] = {
                "A": RIDGE_LAMBDA * np.eye(self.d_user),
                "B": np.zeros((self.d_user, self.d_shared)),
                "b": np.zeros(self.d_user),
            }
        return self.per_user[username]

    def score(self, username: str, z_shared: np.ndarray, x_user: np.ndarray) -> float:
        """z_shared: feature condivise (stesse per tutti gli utenti sul caso c,
        cioè x_c). x_user: feature specifiche utente (u_{i,s}) usate come
        'contesto' del braccio personale, secondo la formulazione ibrida di
        Li et al. 2010."""
        st = self._user_state(username)
        A0_inv = np.linalg.inv(self.A0)
        A_inv = np.linalg.inv(st["A"])
        beta = A0_inv @ self.b0
        theta = A_inv @ (st["b"] - st["B"] @ beta)

        s = float(z_shared @ beta + x_user @ theta)
        # termine di confidenza (approssimazione standard hybrid-LinUCB)
        var = (
            x_user @ A_inv @ x_user
            + z_shared @ A0_inv @ z_shared
            - 2 * z_shared @ (A0_inv @ st["B"].T @ A_inv @ x_user)
            + x_user @ (A_inv @ st["B"] @ A0_inv @ st["B"].T @ A_inv) @ x_user
        )
        var = max(var, 0.0)
        return s + self.alpha * np.sqrt(var)

    def update(self, username: str, z_shared: np.ndarray, x_user: np.ndarray, reward: float):
        st = self._user_state(username)
        A0_inv = np.linalg.inv(self.A0)

        self.A0 += st["B"].T @ np.linalg.inv(st["A"]) @ st["B"]
        self.b0 += st["B"].T @ np.linalg.inv(st["A"]) @ st["b"]

        st["A"] += np.outer(x_user, x_user)
        st["B"] += np.outer(x_user, z_shared)
        st["b"] += reward * x_user

        A0_inv_new = np.linalg.inv(self.A0)
        self.A0 = self.A0 + np.outer(z_shared, z_shared) \
            - st["B"].T @ np.linalg.inv(st["A"]) @ st["B"]
        self.b0 = self.b0 + reward * z_shared \
            - st["B"].T @ np.linalg.inv(st["A"]) @ st["b"]


# ---------------------------------------------------------------------------
# Replay + aggregazione
# ---------------------------------------------------------------------------

def replay_train(train_votes: pd.DataFrame, d_shared: int, d_user: int) -> HybridLinUCB:
    train_votes = train_votes.sort_values("timestamp").reset_index(drop=True)
    bandit = HybridLinUCB(d_shared, d_user)

    # stato causale per costruire feature senza leakage
    user_state = {"by_user_sub": {}}
    sub_running = {}  # community -> {n_posts, n_approve}

    for row in train_votes.itertuples(index=False):
        community = row.community
        stats = sub_running.setdefault(community, {"n": 0, "approve": 0})
        n_seen = max(stats["n"], 1)
        z_shared = np.array([
            stats["approve"] / n_seen,          # approve_rate visto finora
            min(np.log1p(stats["n"]), 5.0),     # log-volume visto finora
            1.0,                                 # intercetta
        ])
        x_user = user_feature_vector(
            {"username": row.username, "by_user_sub": user_state["by_user_sub"]},
            community, d_user,
        )

        score = bandit.score(row.username, z_shared, x_user)  # noqa: F841 (non usato nel replay train)
        reward = 1.0 if row.vote == row.label else -1.0
        bandit.update(row.username, z_shared, x_user, reward)

        key = (row.username, community)
        st = user_state["by_user_sub"].setdefault(key, {"n": 0, "agree": 0, "last_vote": 0})
        st["n"] += 1
        st["agree"] += 1 if row.vote == row.label else 0
        st["last_vote"] = row.vote

        # aggiorna stato community DOPO aver usato le stats pre-update (causale)
        stats["n"] += 1
        stats["approve"] += 1 if row.label == 1 else 0

    bandit._user_state_snapshot = user_state["by_user_sub"]  # per l'inferenza
    bandit._sub_running_snapshot = sub_running
    return bandit


def predict_cases(bandit: HybridLinUCB, votes: pd.DataFrame, d_shared: int, d_user: int) -> pd.DataFrame:
    """Nessun ulteriore update online qui: parametri congelati. Calcola trust
    score, aggrega per softmax-weighted majority per ogni item_id."""
    user_state = bandit._user_state_snapshot
    sub_running = bandit._sub_running_snapshot

    rows = []
    for row in votes.itertuples(index=False):
        community = row.community
        stats = sub_running.get(community, {"n": 0, "approve": 0})
        n_seen = max(stats["n"], 1)
        z_shared = np.array([stats["approve"] / n_seen, min(np.log1p(stats["n"]), 5.0), 1.0])
        x_user = user_feature_vector({"username": row.username, "by_user_sub": user_state}, community, d_user)
        trust = bandit.score(row.username, z_shared, x_user)
        rows.append((row.item_id, row.username, row.vote, row.label, community, trust))

    df = pd.DataFrame(rows, columns=["item_id", "username", "vote", "label", "community", "trust"])

    def agg(g):
        w = np.exp(g["trust"] - g["trust"].max())
        w = w / w.sum()
        score = float((w * g["vote"]).sum())  # continuo in [-1, 1], NON ancora sogliato
        return pd.Series({"label": g["label"].iloc[0], "community": g["community"].iloc[0],
                           "score": score, "n_voters": len(g)})

    return df.groupby("item_id").apply(agg).reset_index()


# ---------------------------------------------------------------------------
# Correzione Doubly Robust (solo valutazione)
# ---------------------------------------------------------------------------

def estimate_propensity(train_preds: pd.DataFrame) -> LogisticRegression:
    """p_c approssimato: P(l'azione maggioritaria osservata sia 'approve') dato
    community + n_voters, stimato SOLO su train. Approssimazione dichiarata
    (non il match esatto dell'insieme di votanti, vedi nota in testa al file)."""
    X = pd.get_dummies(train_preds[["community"]], columns=["community"])
    X["n_voters"] = train_preds["n_voters"]
    y = (train_preds["label"] == 1).astype(int)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X, y)
    model._columns = X.columns
    return model


def doubly_robust_value(preds: pd.DataFrame, propensity_model: LogisticRegression) -> float:
    """preds deve contenere y_hat (post-calibrazione soglia), label, community,
    n_voters. FIX: self-normalized IPW invece di media semplice — la media
    semplice esplode quando p_c e' vicino al floor di clipping (visto in
    pratica: dr_value ~2.09, fuori dal range teorico [-1,1] del reward).
    Self-normalizzando (dividendo per la somma dei pesi anziche' per N) il
    risultato resta limitato e interpretabile."""
    X = pd.get_dummies(preds[["community"]], columns=["community"])
    for col in propensity_model._columns:
        if col not in X.columns:
            X[col] = 0
    X = X[propensity_model._columns.drop("n_voters") if "n_voters" in propensity_model._columns else propensity_model._columns]
    X["n_voters"] = preds["n_voters"]
    X = X[propensity_model._columns]

    p_c = propensity_model.predict_proba(X)[:, 1]
    p_c = np.clip(p_c, 0.05, 0.95)

    r_c = np.where(preds["y_hat"] == preds["label"], 1.0, -1.0)
    r_hat = 2 * p_c - 1
    indicator = (preds["y_hat"] == preds["label"]).astype(float)

    w = indicator / p_c
    w_sum = w.sum()
    correction = float((w * (r_c - r_hat)).sum() / w_sum) if w_sum > 0 else 0.0
    return float(np.mean(r_hat) + correction)


# ---------------------------------------------------------------------------
# Metriche standard (coerenti col resto della pipeline)
# ---------------------------------------------------------------------------

def compute_metrics(preds: pd.DataFrame) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    y_true = preds["label"]
    y_pred = preds["y_hat"]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_VOTES_DIR = "data/splits_intersection"
DEFAULT_OUT_DIR = "results/bandit_dr/splits_intersection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR,
                         help="cartella dello split gia' pronto (default: %(default)s)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help="default: %(default)s")
    args = parser.parse_args()

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(votes_dir / "train_votes.parquet").sort_values("timestamp")
    val = pd.read_parquet(votes_dir / "val_votes.parquet").sort_values("timestamp")
    test = pd.read_parquet(votes_dir / "test_votes.parquet").sort_values("timestamp")

    d_shared, d_user = 3, 3  # coerente con build_context_features / user_feature_vector sopra

    print("Replay LinUCB su train (ordine cronologico)...")
    bandit = replay_train(train, d_shared, d_user)

    print("Inferenza su train/val/test (score continuo, non ancora sogliato)...")
    train_scores = predict_cases(bandit, train, d_shared, d_user)
    val_scores = predict_cases(bandit, val, d_shared, d_user)
    test_scores = predict_cases(bandit, test, d_shared, d_user)

    print("Calibrazione soglia per-community su VAL...")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_scores)
    train_preds = apply_thresholds(train_scores, thresholds, global_threshold)
    val_preds = apply_thresholds(val_scores, thresholds, global_threshold)
    test_preds = apply_thresholds(test_scores, thresholds, global_threshold)

    propensity_model = estimate_propensity(train_preds)

    results = {
        "train": {**compute_metrics(train_preds),
                  "dr_value": doubly_robust_value(train_preds, propensity_model)},
        "val": {**compute_metrics(val_preds),
                "dr_value": doubly_robust_value(val_preds, propensity_model)},
        "test": {**compute_metrics(test_preds),
                 "dr_value": doubly_robust_value(test_preds, propensity_model)},
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    test_preds.to_parquet(out_dir / "test_predictions.parquet", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nSalvato in {out_dir}")


if __name__ == "__main__":
    main()