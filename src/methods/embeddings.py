"""
step3_expert_finder.py  (v2 — fixed)
=====================================
Semantic Expert Finder (SEF) — v2 corregge i problemi principali di v1:

FIX 1 — Signal B implementato correttamente
  Prima: signal B era "sub_p" (precisione subreddit) — solo comportamentale.
  Ora:   signal B = cosine_sim(user_profile_emb, post_emb) — veramente topico.

FIX 2 — Formula a tre componenti
  Prima:  score = alpha * local_p + (1-alpha) * sub_p
  Ora:    score = alpha * local_p + beta * sub_p + (1-alpha-beta) * signal_b
  Grid search esteso a beta (con constraint alpha+beta <= 1).

FIX 3 — Threshold per-subreddit
  Ogni subreddit ha policy diverse. Un threshold globale non cattura questo.
  Ora viene calibrato un threshold per subreddit (con fallback globale se
  il subreddit ha meno di MIN_SUB_SAMPLES campioni).

FIX 4 — Gestione sbilanciamento classi
  Prima: calibrazione su macro F1 puro → bias verso la classe positiva.
  Ora:   weighted F1 con class_weight durante calibrazione + meta-model
         con class_weight='balanced' che impara la soglia ottimale.

FIX 5 — Meta-model downstream
  Dopo aver calcolato i pesi degli esperti, si estraggono feature per post
  (weighted vote a vari T, statistiche sugli esperti, base rate subreddit)
  e si allena una LogisticRegression. Questo strato impara automaticamente
  i pesi ottimali tra i segnali e compensa lo sbilanciamento.

FIX 6 — Default aggiornati
  MIN_USER_VOTES: 3 → 10  (stime di precisione più affidabili)
  K_NEIGHBORS:    50 → 100 (copertura locale migliore per Signal A)

Scores
------
  A  Behavioral-local    precisione su post semanticamente simili (FAISS)
  B  Behavioral-global   precisione storica per subreddit
  C  Topical             cosine_sim(user_profile_emb, post_emb)

  expert_weight(u, P) = alpha * A(u,P) + beta * B(u,P) + (1-alpha-beta) * C(u,P)

  Predizione finale:
    Stage 1: weighted vote degli esperti → feature vector per post
    Stage 2: LogisticRegression(class_weight='balanced') su feature vector
"""

from __future__ import annotations

import argparse
import json
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import faiss
import numpy as np
import pandas as pd
import subprocess
import sys
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEP1_DIR    = Path("results/step1/reddit")
FETCH_DIR    = Path("results/step3_expert")
OUTPUT_DIR   = Path("results/step3_expert")
ORIGINAL_CSV = Path("data/processed/final_intersection_dataset.csv")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 64

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
K_NEIGHBORS    = 100    # FIX 6: era 50, ora 100 per più copertura locale
MIN_USER_VOTES = 10     # FIX 6: era 3, ora 10 per stime più affidabili
ALPHA          = 0.3    # peso Signal A (local behavioral)
BETA           = 0.3    # peso Signal B (global precision)  — NUOVO
                        # peso Signal C (topical) = 1-alpha-beta = 0.4
LAMBDA_SMOOTH  = 2.0
TOP_T          = 10
CAL_SAMPLE     = 500

MIN_SUB_SAMPLES = 50    # FIX 3: soglia minima per threshold per-subreddit

# ---------------------------------------------------------------------------
# Grid search — mantenuto piccolo per velocità
# ---------------------------------------------------------------------------
# Razionale delle scelte:
#   K:     v1 mostrava best_k ∈ {20,100} → testiamo solo 2 valori
#   T:     v1 mostrava best_t ∈ {5,20}   → testiamo solo 2 valori
#   alpha: v1 mostrava alpha=0 dominante → 0.0 e 0.3 sufficienti
#   beta:  nuovo parametro → 0.0 e 0.3
#   Combinazioni valide (alpha+beta ≤ 1): 4  →  totale 2×2×4 = 16 per fold
#   VAL_GRID_SAMPLE: 500 post (era 2000) → 4× più veloce
#   CAL_SAMPLE: 500 (era 1000)
GRID_K     = [20, 100]
GRID_T     = [5, 20]
GRID_ALPHA = [0.0, 0.3]
GRID_BETA  = [0.0, 0.3]
# combinazioni valide: alpha + beta <= 1.0
VAL_GRID_SAMPLE = 500    # post campionati dal training per il grid search

# ---------------------------------------------------------------------------
# Meta-model feature columns (FIX 5)
# ---------------------------------------------------------------------------
TOP_T_LIST    = [5, 10, 20]   # T diversi per estrarre weighted vote
META_FEAT_COLS: List[str] = []   # popolato dinamicamente

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
N_FOLDS = 5

SCALAR_METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# global usato da compute_expert_weights (invariato rispetto a v1)
post_ids_ordered: List[str] = []


# ===========================================================================
# 1.  DATA LOADING  (invariato da v1)
# ===========================================================================

def load_votes(input_csv: Optional[Path] = None) -> pd.DataFrame:
    for csv_path in [p for p in [input_csv, ORIGINAL_CSV] if p is not None]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            df["vote"]  = pd.to_numeric(df["vote"],  errors="coerce")
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])].copy()
            df["vote"]  = df["vote"].astype(int)
            df["label"] = df["label"].astype(int)
            df = df.dropna(subset=["item_id", "username", "vote", "label"])
            df = df.drop_duplicates(subset=["username", "item_id"])
            log.info(
                "Votes loaded from %s: %d rows | %d posts | %d users | %d communities",
                csv_path, len(df), df["item_id"].nunique(),
                df["username"].nunique(), df["community"].nunique(),
            )
            return df

    log.warning("Original CSV not found — falling back to filtered_votes.parquet.")
    path = STEP1_DIR / "filtered_votes.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        dfs = [
            pd.read_parquet(STEP1_DIR / "splits" / f"{s}_votes.parquet")
            for s in ("train", "val", "test")
            if (STEP1_DIR / "splits" / f"{s}_votes.parquet").exists()
        ]
        df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"]  = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    log.info("Votes loaded: %d rows | %d posts | %d users", len(df),
             df["item_id"].nunique(), df["username"].nunique())
    return df


def load_post_texts() -> pd.DataFrame:
    path = FETCH_DIR / "post_texts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Post texts not found at {path}.")
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["text"].notna()].copy()
    log.info("Post texts: %d with text / %d total", len(df), before)
    return df


def load_user_documents() -> pd.DataFrame:
    path = FETCH_DIR / "user_documents.parquet"
    if not path.exists():
        log.warning("User documents not found — Signal C (topical) sarà 0 per tutti.")
        return pd.DataFrame(columns=["username", "text"])
    df = pd.read_parquet(path)
    df = df[df["text"].notna()].copy()
    log.info("User documents: %d docs | %d users", len(df), df["username"].nunique())
    return df


# ===========================================================================
# 2.  EMBEDDINGS  (invariato da v1 — subprocess per evitare macOS segfault)
# ===========================================================================

def _get_model() -> str:
    return EMBEDDING_MODEL


def _encode_texts(
    texts:      List[str],
    model_name: str,
    normalize:  bool = True,
    batch_size: int  = BATCH_SIZE,
) -> np.ndarray:
    import pickle
    helper = Path(__file__).parent / "embed_helper.py"
    if not helper.exists():
        raise FileNotFoundError(f"embed_helper.py non trovato a {helper}.")

    with tempfile.TemporaryDirectory() as tmp:
        texts_path = os.path.join(tmp, "texts.pkl")
        out_path   = os.path.join(tmp, "embeddings.npy")
        with open(texts_path, "wb") as f:
            pickle.dump(texts, f)
        cmd = [
            sys.executable, str(helper),
            "--texts-path", texts_path,
            "--out-path",   out_path,
            "--model-name", model_name,
            "--batch-size", str(batch_size),
        ]
        if normalize:
            cmd.append("--normalize")
        log.info("Spawning embed_helper per %d testi …", len(texts))
        subprocess.run(cmd, check=True)
        vecs = np.load(out_path).astype(np.float32)
    log.info("  → embeddings shape: %s", vecs.shape)
    return vecs


def build_post_embeddings(
    post_texts: pd.DataFrame,
    embed_dir:  Path,
    model:      Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    emb_path    = embed_dir / "post_embeddings.npy"
    ids_path    = embed_dir / "post_ids.json"
    partial_dir = embed_dir / "post_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached post embeddings …")
        embeddings = np.load(emb_path)
        item_ids   = json.load(open(ids_path))
        return embeddings, item_ids

    model_name = model if model is not None else _get_model()
    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(exist_ok=True)

    df           = post_texts.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    all_texts    = df["text"].tolist()
    all_item_ids = df["item_id"].tolist()
    n_total      = len(all_texts)
    n_chunks     = (n_total + chunk_size - 1) // chunk_size
    done         = len(sorted(partial_dir.glob("chunk_*.npy")))

    for ci in range(done, n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=True)
        np.save(partial_dir / f"chunk_{ci:05d}.npy", vecs)

    chunks     = [np.load(partial_dir / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(emb_path, embeddings)
    json.dump(all_item_ids, open(ids_path, "w"))
    log.info("Post embeddings salvati: shape %s", embeddings.shape)
    return embeddings, all_item_ids


def build_user_embeddings(
    user_docs:  pd.DataFrame,
    embed_dir:  Path,
    model:      Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    emb_path    = embed_dir / "user_embeddings.npy"
    ids_path    = embed_dir / "user_ids.json"
    partial_dir = embed_dir / "user_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached user embeddings …")
        embeddings = np.load(emb_path)
        usernames  = json.load(open(ids_path))
        return embeddings, usernames

    if user_docs.empty:
        log.warning("Nessun documento utente — user embeddings vuoti.")
        embed_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        json.dump([], open(ids_path, "w"))
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    model_name = model if model is not None else _get_model()
    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(exist_ok=True)

    user_docs = user_docs.reset_index(drop=True)
    all_texts = user_docs["text"].tolist()
    n_total   = len(all_texts)
    n_chunks  = (n_total + chunk_size - 1) // chunk_size
    done      = len(sorted(partial_dir.glob("udoc_chunk_*.npy")))

    for ci in range(done, n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=False)
        np.save(partial_dir / f"udoc_chunk_{ci:05d}.npy", vecs)

    chunks         = [np.load(partial_dir / f"udoc_chunk_{i:05d}.npy") for i in range(n_chunks)]
    doc_embeddings = np.concatenate(chunks, axis=0).astype(np.float32)

    user_docs["_emb_idx"] = np.arange(len(user_docs))
    usernames, user_vecs  = [], []
    for uname, group in tqdm(user_docs.groupby("username"), desc="Averaging user profiles"):
        mean_vec = doc_embeddings[group["_emb_idx"].values].mean(axis=0)
        norm     = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        usernames.append(uname)
        user_vecs.append(mean_vec)

    user_matrix = np.stack(user_vecs, axis=0).astype(np.float32)
    np.save(emb_path, user_matrix)
    json.dump(usernames, open(ids_path, "w"))
    log.info("User embeddings salvati: shape %s", user_matrix.shape)
    return user_matrix, usernames


# ===========================================================================
# 3.  FAISS INDEX  (invariato)
# ===========================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    log.info("FAISS index built: %d vectors", index.ntotal)
    return index


# ===========================================================================
# 4.  SIGNAL COMPUTATION  — FIX 1+2: Signal B + formula a 3 componenti
# ===========================================================================

def precompute_vote_precision(
    train_votes:    pd.DataFrame,
    lambda_smooth:  float,
    min_user_votes: int,
) -> pd.DataFrame:
    """
    Precision per utente per direzione (con e senza subreddit).
    Identico a v1 — questa parte era già corretta.
    """
    tv = train_votes.copy()
    tv["correct"] = (tv["vote"] == tv["label"]).astype(int)

    user_totals = tv.groupby("username")["vote"].count()
    valid_users = user_totals[user_totals >= min_user_votes].index
    tv = tv[tv["username"].isin(valid_users)]

    rows = []
    for (uname, direction), grp in tv.groupby(["username", "vote"]):
        n   = len(grp)
        n_c = grp["correct"].sum()
        rows.append({
            "username":    uname,
            "direction":   int(direction),
            "subreddit":   None,
            "prec":        (n_c + lambda_smooth) / (n + 2 * lambda_smooth),
            "n_votes_dir": n,
        })
    for (uname, sub, direction), grp in tv.groupby(["username", "community", "vote"]):
        n   = len(grp)
        n_c = grp["correct"].sum()
        rows.append({
            "username":    uname,
            "direction":   int(direction),
            "subreddit":   sub,
            "prec":        (n_c + lambda_smooth) / (n + 2 * lambda_smooth),
            "n_votes_dir": n,
        })

    return pd.DataFrame(rows)


def compute_subreddit_base_rates(train_votes: pd.DataFrame) -> Dict[str, float]:
    """
    FIX 3 — base rate (frazione approved) per subreddit su dati di training.
    Usata come feature nel meta-model e come fallback per threshold per-subreddit.
    """
    rates = {}
    for sub, grp in train_votes.drop_duplicates("item_id").groupby("community"):
        approved = (grp["label"] == 1).sum()
        rates[sub] = float(approved / max(len(grp), 1))
    return rates


def compute_expert_weights(
    test_item_ids:  List[str],
    train_item_ids: set,
    vote_df:        pd.DataFrame,
    prec_df:        pd.DataFrame,
    post_emb:       np.ndarray,
    post_id2idx:    Dict[str, int],
    user_emb:       np.ndarray,          # FIX 1: ora passato e usato
    user_id2idx:    Dict[str, int],      # FIX 1: ora passato e usato
    faiss_index:    faiss.Index,
    k_neighbors:    int,
    alpha:          float,
    beta:           float,               # FIX 2: nuovo parametro
    lambda_smooth:  float,
) -> pd.DataFrame:
    """
    Calcola il peso esperto per ogni coppia (post_test, votante).

    Tre segnali (FIX 1+2):
      A  local_p   = precisione locale su K vicini FAISS (behavioral, locale)
      B  sub_p     = precisione storica subreddit (behavioral, globale)
      C  signal_b  = cosine_sim(user_profile_emb, post_emb) (topico)

      expert_weight = alpha * A + beta * B + (1-alpha-beta) * C

    In v1 Signal C non era implementato — user_emb veniva ignorato.
    """
    # lookup rapidi per precisioni
    global_prec_map: Dict[tuple, float] = {}
    sub_prec_map:    Dict[tuple, float] = {}
    for _, r in prec_df.iterrows():
        key = (r["username"], r["direction"])
        if r["subreddit"] is None:
            global_prec_map[key] = float(r["prec"])
        else:
            sub_prec_map[(r["username"], r["direction"], r["subreddit"])] = float(r["prec"])

    gamma = 1.0 - alpha - beta   # peso Signal C (topico)

    # voti di training per lookup FAISS locale
    train_votes = vote_df[vote_df["item_id"].isin(train_item_ids)].copy()
    train_votes["correct"] = (train_votes["vote"] == train_votes["label"]).astype(int)
    item_votes: Dict[str, pd.DataFrame] = {
        iid: grp[["username", "vote", "correct"]]
        for iid, grp in train_votes.groupby("item_id")
    }

    post_sub = (
        vote_df[vote_df["item_id"].isin(test_item_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )
    test_votes = (
        vote_df[vote_df["item_id"].isin(test_item_ids)]
        [["item_id", "username", "vote"]]
    )

    rows = []
    for item_id in test_item_ids:
        if item_id not in post_id2idx:
            continue

        subreddit = post_sub.get(item_id)
        p_idx     = post_id2idx[item_id]
        query_vec = post_emb[p_idx : p_idx + 1]   # shape (1, 384)
        p_vec     = post_emb[p_idx]                # shape (384,) per Signal C

        # FAISS: K vicini più simili nel training set
        raw_sims, raw_idxs = faiss_index.search(query_vec, k_neighbors + 1)
        neighbours: List[Tuple[str, float]] = []
        for sim, j in zip(raw_sims[0], raw_idxs[0]):
            if j < 0:
                continue
            nid = post_ids_ordered[j]
            if nid == item_id or nid not in train_item_ids:
                continue
            neighbours.append((nid, max(float(sim), 0.0)))

        mean_sim = np.mean([s for _, s in neighbours]) if neighbours else 1.0

        voters_on_post = test_votes[test_votes["item_id"] == item_id]

        for _, vrow in voters_on_post.iterrows():
            uname     = vrow["username"]
            direction = int(vrow["vote"])

            # ── Signal B: precisione globale/subreddit (behavioral) ─────────
            laplace_default = lambda_smooth / (2 * lambda_smooth)   # 0.5
            global_p = global_prec_map.get((uname, direction), laplace_default)
            sub_p    = sub_prec_map.get((uname, direction, subreddit), global_p)

            # ── Signal A: precisione locale sui K vicini ────────────────────
            loc_correct = 0.0
            loc_weight  = 0.0
            for nid, sim in neighbours:
                if nid not in item_votes:
                    continue
                nv = item_votes[nid]
                user_rows = nv[nv["username"] == uname]
                for _, ur in user_rows.iterrows():
                    if int(ur["vote"]) == direction:
                        loc_correct += sim * float(ur["correct"])
                        loc_weight  += sim

            # Bayesian shrinkage: pull verso sub_p quando copertura locale bassa
            local_p = (loc_correct + lambda_smooth * sub_p * mean_sim) / \
                      (loc_weight  + lambda_smooth * mean_sim)

            # ── Signal C: similarità topica utente–post  (FIX 1) ───────────
            # Cosine sim tra profilo-embedding utente e embedding del post.
            # Entrambi L2-normalizzati → dot product = cosine sim ∈ [-1,1].
            # Clipping a [0, 1]: similarità negativa non ha interpretazione utile.
            if uname in user_id2idx and user_emb.shape[0] > 0:
                u_vec    = user_emb[user_id2idx[uname]]
                raw_sim  = float(np.dot(u_vec, p_vec))
                signal_c = max(0.0, raw_sim)
            else:
                # Fallback: usa sub_p se non abbiamo l'embedding utente
                signal_c = sub_p

            # ── Peso esperto finale (FIX 2) ──────────────────────────────────
            # expert_weight = alpha*A + beta*B + gamma*C  (gamma = 1-alpha-beta)
            expert_w = alpha * local_p + beta * sub_p + gamma * signal_c

            rows.append({
                "item_id":       item_id,
                "username":      uname,
                "vote":          direction,
                "expert_weight": float(expert_w),
                "local_prec":    float(local_p),
                "global_prec":   float(sub_p),
                "signal_c":      float(signal_c),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "item_id","username","vote","expert_weight",
            "local_prec","global_prec","signal_c"
        ])
    return pd.DataFrame(rows)


# ===========================================================================
# 5.  FEATURE EXTRACTION  (FIX 5: meta-model)
# ===========================================================================

def extract_post_features(
    item_ids:      List[str],
    vote_df:       pd.DataFrame,
    weights_df:    pd.DataFrame,
    base_rates:    Dict[str, float],
    top_t_list:    List[int] = TOP_T_LIST,
) -> pd.DataFrame:
    """
    Estrae un feature vector per post da usare nel meta-model.

    Feature:
      - raw vote mean, std, n_voters, pct_positive
      - weighted_vote a vari T
      - statistiche sui pesi degli esperti
      - media signal_c (topico) dei top-5 esperti
      - base_rate approvazione del subreddit
    """
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][
        ["item_id", "username", "vote"]
    ]
    post_sub = (
        vote_df[vote_df["item_id"].isin(item_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )

    rows = []
    for item_id in item_ids:
        rv  = raw_votes[raw_votes["item_id"] == item_id]["vote"].values
        pw  = weights_df[weights_df["item_id"] == item_id] if not weights_df.empty else pd.DataFrame()
        sub = post_sub.get(item_id, "__unknown__")

        feats: Dict[str, float] = {}

        # Raw vote statistics
        feats["raw_vote_mean"] = float(rv.mean())          if len(rv) > 0 else 0.0
        feats["raw_vote_std"]  = float(rv.std())           if len(rv) > 1 else 0.0
        feats["n_voters"]      = float(len(rv))
        feats["pct_positive"]  = float((rv == 1).mean())   if len(rv) > 0 else 0.5

        # Weighted vote a diversi T
        for t in top_t_list:
            wv = _compute_weighted_votes([item_id], vote_df, weights_df, t)
            feats[f"wv_top{t}"] = wv.get(item_id, feats["raw_vote_mean"])

        # Statistiche sugli esperti
        if not pw.empty and len(pw) > 0:
            feats["mean_expert_weight"] = float(pw["expert_weight"].mean())
            feats["std_expert_weight"]  = float(pw["expert_weight"].std()) if len(pw) > 1 else 0.0
            feats["n_experts"]          = float(len(pw))
            top5 = pw.nlargest(5, "expert_weight")
            feats["top5_mean_local"]    = float(top5["local_prec"].mean())
            feats["top5_mean_global"]   = float(top5["global_prec"].mean())
            feats["top5_mean_signal_c"] = float(top5["signal_c"].mean())
            feats["top5_weighted_vote"] = float(
                np.dot(top5["expert_weight"].values, top5["vote"].values.astype(float))
                / top5["expert_weight"].values.sum()
            )
        else:
            feats["mean_expert_weight"] = 0.0
            feats["std_expert_weight"]  = 0.0
            feats["n_experts"]          = 0.0
            feats["top5_mean_local"]    = 0.5
            feats["top5_mean_global"]   = 0.5
            feats["top5_mean_signal_c"] = 0.0
            feats["top5_weighted_vote"] = feats["raw_vote_mean"]

        # Subreddit base rate (tasso storico di approvazione)
        feats["sub_approve_rate"] = base_rates.get(sub, 0.5)

        feats["item_id"] = item_id
        rows.append(feats)

    return pd.DataFrame(rows)


def get_meta_feature_cols(feat_df: pd.DataFrame) -> List[str]:
    return [c for c in feat_df.columns if c != "item_id"]


def train_meta_model(
    feat_df: pd.DataFrame,
    labels:  Dict[str, int],
    feat_cols: List[str],
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    FIX 4+5: LogisticRegression con class_weight='balanced' per gestire
    lo sbilanciamento tra approve e remove.
    """
    items = feat_df[feat_df["item_id"].isin(labels)]["item_id"].tolist()
    X = feat_df.set_index("item_id").loc[items, feat_cols].values.astype(float)
    y = np.array([labels[i] for i in items])

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    clf = LogisticRegression(
        class_weight="balanced",   # FIX 4: compensa lo sbilanciamento
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_s, y)
    log.info(
        "  Meta-model coef sample: %s …",
        dict(zip(feat_cols[:4], clf.coef_[0][:4].round(3)))
    )
    return clf, scaler


def predict_meta_model(
    feat_df:   pd.DataFrame,
    item_ids:  List[str],
    clf:       LogisticRegression,
    scaler:    StandardScaler,
    feat_cols: List[str],
) -> Dict[str, Tuple[int, float]]:
    """
    Ritorna {item_id: (predicted_label, proba_pos)}.
    """
    df = feat_df.set_index("item_id")
    result = {}
    for item_id in item_ids:
        if item_id not in df.index:
            result[item_id] = (1, 0.5)
            continue
        x = df.loc[item_id, feat_cols].values.astype(float).reshape(1, -1)
        x_s   = scaler.transform(x)
        pred  = int(clf.predict(x_s)[0])
        proba = float(clf.predict_proba(x_s)[0, list(clf.classes_).index(1)])
        result[item_id] = (pred, proba)
    return result


# ===========================================================================
# 6.  WEIGHTED VOTE  (invariato nella logica, esteso per feature extraction)
# ===========================================================================

def _compute_weighted_votes(
    item_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    top_t:      int,
) -> Dict[str, float]:
    if weights_df.empty:
        raw = vote_df[vote_df["item_id"].isin(item_ids)][["item_id","vote"]]
        return {
            iid: float(grp["vote"].mean())
            for iid, grp in raw.groupby("item_id")
        }

    w = weights_df[weights_df["item_id"].isin(item_ids)]
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id","username","vote"]]

    result = {}
    for item_id in item_ids:
        pw = w[w["item_id"] == item_id]
        if pw.empty:
            rv = raw_votes[raw_votes["item_id"] == item_id]["vote"].values
            result[item_id] = float(rv.mean()) if len(rv) > 0 else 0.0
            continue
        top     = pw.nlargest(top_t, "expert_weight")
        weights = top["expert_weight"].values.clip(min=1e-9)
        votes   = top["vote"].values.astype(float)
        result[item_id] = float(np.dot(weights, votes) / weights.sum())
    return result


# ===========================================================================
# 7.  THRESHOLD CALIBRATION  (FIX 3+4: per-subreddit + weighted F1)
# ===========================================================================

def calibrate_threshold_weighted(
    sample_ids:  List[str],
    vote_df:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    top_t:       int,
    neg_weight:  float = 2.0,   # FIX 4: peso extra sulla classe negativa
) -> float:
    """
    Cerca il threshold che massimizza F1 pesata, dando più importanza
    alla classe -1 (remove), storicamente sottorappresentata.
    """
    labels = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )
    wv_scores = _compute_weighted_votes(sample_ids, vote_df, weights_df, top_t)
    if not wv_scores:
        return 0.0

    common = [iid for iid in wv_scores if iid in labels]
    if not common:
        return 0.0

    y_true  = np.array([labels[iid] for iid in common])
    y_score = np.array([wv_scores[iid] for iid in common])

    # pesi campione: penalizza la classe + (approve) che è sovrarappresentata
    sample_w = np.where(y_true == -1, neg_weight, 1.0)

    best_thr, best_f1 = 0.0, -1.0
    for thr in np.linspace(float(y_score.min()), float(y_score.max()), 100):
        y_pred = np.where(y_score >= thr, 1, -1)
        f1 = f1_score(y_true, y_pred, average="macro",
                      sample_weight=sample_w, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def calibrate_thresholds_per_subreddit(
    sample_ids:  List[str],
    vote_df:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    top_t:       int,
    min_samples: int = MIN_SUB_SAMPLES,
) -> Dict[Optional[str], float]:
    """
    FIX 3: threshold separato per ogni subreddit con abbastanza campioni.
    Ritorna dict {subreddit: threshold, None: global_threshold}.
    """
    global_thr = calibrate_threshold_weighted(
        sample_ids, vote_df, weights_df, top_t
    )
    thresholds: Dict[Optional[str], float] = {None: global_thr}

    post_sub = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )
    sub_groups: Dict[str, List[str]] = {}
    for iid in sample_ids:
        sub = post_sub.get(iid)
        if sub:
            sub_groups.setdefault(sub, []).append(iid)

    for sub, ids in sub_groups.items():
        if len(ids) < min_samples:
            thresholds[sub] = global_thr   # fallback
            continue
        thr = calibrate_threshold_weighted(ids, vote_df, weights_df, top_t)
        thresholds[sub] = thr
        log.debug("  Sub %s: threshold=%.4f (n=%d)", sub, thr, len(ids))

    log.info(
        "  Thresholds: global=%.4f | %d subreddit-specific",
        global_thr, sum(1 for k in thresholds if k is not None)
    )
    return thresholds


# ===========================================================================
# 8.  GRID SEARCH  (FIX 2: aggiunto beta)
# ===========================================================================

def grid_search_hyperparams(
    val_ids:       List[str],
    train_ids:     set,
    vote_df:       pd.DataFrame,
    prec_df:       pd.DataFrame,
    post_emb:      np.ndarray,
    post_id2idx:   Dict[str, int],
    user_emb:      np.ndarray,
    user_id2idx:   Dict[str, int],
    faiss_index:   faiss.Index,
    lambda_smooth: float,
) -> Tuple[int, int, float, float]:
    """
    Grid search su K × T × alpha × beta sul validation set.
    Ottimizza macro F1.
    Ritorna (best_k, best_t, best_alpha, best_beta).
    """
    labels = (
        vote_df[vote_df["item_id"].isin(val_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )
    val_ids = [iid for iid in val_ids if iid in labels]
    if not val_ids:
        return K_NEIGHBORS, TOP_T, ALPHA, BETA

    best = {"f1": -1.0, "k": K_NEIGHBORS, "t": TOP_T, "alpha": ALPHA, "beta": BETA}

    # Precomputa embeddings FAISS per tutti i K valori possibili
    max_k = max(GRID_K)
    raw_sims_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for item_id in val_ids:
        if item_id not in post_id2idx:
            continue
        qv = post_emb[post_id2idx[item_id] : post_id2idx[item_id] + 1]
        sims, idxs = faiss_index.search(qv, max_k + 1)
        raw_sims_cache[item_id] = (sims[0], idxs[0])

    n_combos = (
        len(GRID_K) * len(GRID_T)
        * sum(1 for a in GRID_ALPHA for b in GRID_BETA if a + b <= 1.0)
    )
    log.info("  Grid search: %d combinazioni su %d val post …", n_combos, len(val_ids))

    for alpha in GRID_ALPHA:
        for beta in [b for b in GRID_BETA if alpha + b <= 1.0]:
            gamma = 1.0 - alpha - beta
            for k in GRID_K:
                # Calcola i pesi con questi iper-parametri
                weights_df = compute_expert_weights(
                    val_ids, train_ids, vote_df, prec_df,
                    post_emb, post_id2idx,
                    user_emb, user_id2idx,
                    faiss_index, k, alpha, beta, lambda_smooth,
                )
                for t in GRID_T:
                    wv = _compute_weighted_votes(val_ids, vote_df, weights_df, t)
                    common = [iid for iid in wv if iid in labels]
                    if not common:
                        continue
                    y_true  = np.array([labels[iid] for iid in common])
                    y_score = np.array([wv[iid] for iid in common])

                    # Threshold ricercato sulla stessa val (leggermente ottimistico
                    # ma consistente tra combinazioni)
                    best_thr, best_f1_t = 0.0, -1.0
                    for thr in np.linspace(y_score.min(), y_score.max(), 50):
                        y_pred = np.where(y_score >= thr, 1, -1)
                        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                        if f1 > best_f1_t:
                            best_f1_t, best_thr = f1, float(thr)

                    if best_f1_t > best["f1"]:
                        best = {"f1": best_f1_t, "k": k, "t": t,
                                "alpha": alpha, "beta": beta}

    log.info(
        "  Best: k=%d  t=%d  alpha=%.2f  beta=%.2f  macro_f1=%.4f",
        best["k"], best["t"], best["alpha"], best["beta"], best["f1"]
    )
    return best["k"], best["t"], best["alpha"], best["beta"]


# ===========================================================================
# 9.  PREDICTION  (aggiornato: usa thresholds per-subreddit + meta-model)
# ===========================================================================

def predict_fold(
    test_ids:    List[str],
    vote_df:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    feat_df:     pd.DataFrame,
    feat_cols:   List[str],
    clf:         LogisticRegression,
    scaler:      StandardScaler,
    thresholds:  Dict[Optional[str], float],
    top_t:       int,
) -> pd.DataFrame:
    """
    Predizione a due stadi:
      Stage 1: weighted vote degli esperti (per feature extraction)
      Stage 2: meta-model LogisticRegression su feature per post

    Il meta-model usa class_weight='balanced' e supera il threshold fisso.
    """
    labels = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id")[["item_id", "label"]]
    )
    post_sub = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )

    # Predizione meta-model
    meta_preds = predict_meta_model(feat_df, test_ids, clf, scaler, feat_cols)

    # Predizione con threshold per-subreddit (Stage 1, come baseline)
    wv = _compute_weighted_votes(test_ids, vote_df, weights_df, top_t)
    w_ids = set(weights_df["item_id"].unique()) if not weights_df.empty else set()

    rows = []
    for item_id in test_ids:
        sub      = post_sub.get(item_id)
        thr      = thresholds.get(sub, thresholds.get(None, 0.0))
        score    = wv.get(item_id, 0.0)
        fallback = item_id not in w_ids

        # Stage 2: usa il meta-model se disponibile
        meta_pred, meta_proba = meta_preds.get(item_id, (None, None))
        predicted = meta_pred if meta_pred is not None else (1 if score >= thr else -1)

        rows.append({
            "item_id":       item_id,
            "weighted_vote": score,
            "meta_proba":    meta_proba if meta_proba is not None else 0.5,
            "predicted":     predicted,
            "used_fallback": fallback,
            "subreddit":     sub,
        })

    pred_df = pd.DataFrame(rows)
    result  = pd.merge(labels, pred_df, on="item_id", how="left")
    result["predicted"] = result["predicted"].fillna(1).astype(int)
    return result


# ===========================================================================
# 10.  METRICS  (invariato)
# ===========================================================================

def evaluate_fold(decisions: pd.DataFrame) -> Dict:
    y_true = decisions["label"].values
    y_pred = decisions["predicted"].values
    try:
        auc = float(roc_auc_score(
            (y_true == 1).astype(int),
            decisions["meta_proba"].values if "meta_proba" in decisions.columns
            else (y_pred == 1).astype(int),
        ))
    except ValueError:
        auc = float("nan")
    return {
        "macro_f1":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "roc_auc":         auc,
        "f1_pos":          f1_score(y_true, y_pred, pos_label=1,  average="binary", zero_division=0),
        "f1_neg":          f1_score(y_true, y_pred, pos_label=-1, average="binary", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro",   zero_division=0),
        "macro_recall":    recall_score(y_true, y_pred, average="macro",      zero_division=0),
        "precision_pos":   precision_score(y_true, y_pred, pos_label=1,  average="binary", zero_division=0),
        "recall_pos":      recall_score(y_true, y_pred, pos_label=1,     average="binary", zero_division=0),
        "precision_neg":   precision_score(y_true, y_pred, pos_label=-1, average="binary", zero_division=0),
        "recall_neg":      recall_score(y_true, y_pred, pos_label=-1,    average="binary", zero_division=0),
    }


# ===========================================================================
# 11.  KFold  (aggiornato: grid search su beta, meta-model, threshold per-sub)
# ===========================================================================

def run_kfold(
    vote_df:       pd.DataFrame,
    post_emb:      np.ndarray,
    post_id_list:  List[str],
    user_emb:      np.ndarray,
    user_id_list:  List[str],
    faiss_index:   faiss.Index,
    default_k:     int,
    default_t:     int,
    default_alpha: float,
    default_beta:  float,
    min_coverage:  int,
    do_grid_search: bool = True,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    global post_ids_ordered
    post_ids_ordered = post_id_list

    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}
    user_id2idx = {uid: i for i, uid in enumerate(user_id_list)}

    labeled_items = (
        vote_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"])
        .reset_index(drop=True)
    )
    log.info("Labeled items per KFold: %d", len(labeled_items))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics:     List[Dict]         = []
    fold_item_scores: List[pd.DataFrame] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(labeled_items)):
        log.info("─── Fold %d / %d ───", fold_idx + 1, N_FOLDS)

        train_ids = set(labeled_items.iloc[train_val_idx]["item_id"].tolist())
        test_ids  = set(labeled_items.iloc[test_idx]["item_id"].tolist())
        log.info("  train=%d  test=%d", len(train_ids), len(test_ids))

        # ── precisioni dal training fold ─────────────────────────────────────
        train_vote_df = vote_df[vote_df["item_id"].isin(train_ids)]
        prec_df = precompute_vote_precision(train_vote_df, LAMBDA_SMOOTH, min_coverage)
        base_rates = compute_subreddit_base_rates(train_vote_df)
        log.info("  reliable users (>=%d votes): %d",
                 min_coverage, prec_df["username"].nunique())

        # ── grid search su una porzione di validation interna ────────────────
        if do_grid_search:
            train_list = list(train_ids)
            np.random.shuffle(train_list)
            val_grid_ids = train_list[:min(VAL_GRID_SAMPLE, len(train_list))]
            train_grid   = set(train_list[min(VAL_GRID_SAMPLE, len(train_list)):])

            k, t, alpha, beta = grid_search_hyperparams(
                val_grid_ids, train_grid, vote_df, prec_df,
                post_emb, post_id2idx,
                user_emb, user_id2idx,
                faiss_index, LAMBDA_SMOOTH,
            )
        else:
            k, t, alpha, beta = default_k, default_t, default_alpha, default_beta

        # ── pesi esperti su post di test ─────────────────────────────────────
        test_id_list = list(test_ids)
        weights_df   = compute_expert_weights(
            test_id_list, train_ids, vote_df, prec_df,
            post_emb, post_id2idx,
            user_emb, user_id2idx,
            faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
        )

        # ── calibrazione threshold per-subreddit su campione training ────────
        train_sample = list(train_ids)
        np.random.shuffle(train_sample)
        train_sample = train_sample[:min(CAL_SAMPLE, len(train_sample))]

        weights_cal = compute_expert_weights(
            train_sample,
            train_ids - set(train_sample),
            vote_df, prec_df,
            post_emb, post_id2idx,
            user_emb, user_id2idx,
            faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
        )
        thresholds = calibrate_thresholds_per_subreddit(
            train_sample, vote_df, weights_cal, t
        )

        # ── meta-model: training su tutto il training fold ───────────────────
        # Feature extraction sui post di training
        log.info("  Extracting training features per meta-model …")
        train_feat_df = extract_post_features(
            train_sample, vote_df, weights_cal, base_rates
        )
        feat_cols = get_meta_feature_cols(train_feat_df)

        train_labels = (
            vote_df[vote_df["item_id"].isin(train_sample)]
            .drop_duplicates("item_id")
            .set_index("item_id")["label"]
            .to_dict()
        )
        clf, scaler = train_meta_model(train_feat_df, train_labels, feat_cols)

        # Feature extraction sui post di test
        test_feat_df = extract_post_features(
            test_id_list, vote_df, weights_df, base_rates
        )

        # ── predizione ───────────────────────────────────────────────────────
        decisions = predict_fold(
            test_id_list, vote_df, weights_df,
            test_feat_df, feat_cols, clf, scaler,
            thresholds, t,
        )
        decisions["fold"]  = fold_idx
        decisions["k"]     = k
        decisions["top_t"] = t
        decisions["alpha"] = alpha
        decisions["beta"]  = beta

        if decisions["label"].nunique() < 2:
            log.warning("Fold %d: una sola classe nel test — skip.", fold_idx + 1)
            continue

        metrics = evaluate_fold(decisions)
        metrics.update({
            "fold":         fold_idx,
            "k":            k,
            "top_t":        t,
            "alpha":        alpha,
            "beta":         beta,
            "gamma":        1.0 - alpha - beta,
            "threshold":    thresholds.get(None, 0.0),
            "n_test":       len(decisions),
            "fallback_pct": float(decisions["used_fallback"].mean()),
        })

        log.info(
            "  macro_f1=%.4f  roc_auc=%.4f  f1_pos=%.4f  f1_neg=%.4f  "
            "fallback=%.1f%%  alpha=%.2f  beta=%.2f  gamma=%.2f",
            metrics["macro_f1"], metrics["roc_auc"],
            metrics["f1_pos"],   metrics["f1_neg"],
            100 * metrics["fallback_pct"],
            alpha, beta, 1.0 - alpha - beta,
        )

        fold_metrics.append(metrics)
        fold_item_scores.append(decisions)

    return fold_metrics, fold_item_scores


# ===========================================================================
# 12.  OUTPUT SAVING  (invariato nella struttura, aggiunto gamma)
# ===========================================================================

def save_outputs(
    fold_metrics:     List[Dict],
    fold_item_scores: List[pd.DataFrame],
    output_dir:       Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = pd.concat(fold_item_scores, ignore_index=True)
    all_scores.to_parquet(output_dir / "item_scores.parquet", index=False)

    fold_rows = [
        {k: v for k, v in m.items() if isinstance(v, (int, float, bool))}
        for m in fold_metrics
    ]
    pd.DataFrame(fold_rows).to_parquet(output_dir / "fold_details.parquet", index=False)

    agg: Dict = {"n_folds": len(fold_metrics)}
    for key in SCALAR_METRICS:
        vals = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
        agg[key]               = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"]      = float(np.std(vals))  if vals else float("nan")
        agg[f"{key}_per_fold"] = vals

    agg["best_k_per_fold"]     = [m.get("k")     for m in fold_metrics]
    agg["best_t_per_fold"]     = [m.get("top_t") for m in fold_metrics]
    agg["best_alpha_per_fold"] = [m.get("alpha") for m in fold_metrics]
    agg["best_beta_per_fold"]  = [m.get("beta")  for m in fold_metrics]
    agg["best_gamma_per_fold"] = [m.get("gamma") for m in fold_metrics]
    agg["fallback_pct"]        = float(np.mean([m.get("fallback_pct", 0) for m in fold_metrics]))

    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)

    log.info(
        "Outputs salvati in %s\n"
        "  macro_f1 = %.4f ± %.4f\n"
        "  roc_auc  = %.4f ± %.4f\n"
        "  f1_pos   = %.4f  f1_neg = %.4f",
        output_dir,
        agg["macro_f1"], agg["macro_f1_std"],
        agg["roc_auc"],  agg["roc_auc_std"],
        agg["f1_pos"],   agg["f1_neg"],
    )


# ===========================================================================
# 13.  OPTIONAL NEO4J  (invariato)
# ===========================================================================

def build_neo4j_graph(
    vote_df:      pd.DataFrame,
    post_texts:   pd.DataFrame,
    post_emb:     np.ndarray,
    post_id_list: List[str],
    faiss_index:  faiss.Index,
    neo4j_uri:    str,
    neo4j_user:   str,
    neo4j_pass:   str,
    k_similar:    int = 50,
) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.error("neo4j non installato. Esegui: pip install neo4j")
        return

    driver     = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}
    text_ids    = set(post_id_list)

    def _chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE")

        post_rows = vote_df.drop_duplicates("item_id")[["item_id", "community"]].to_dict("records")
        for chunk in _chunks(post_rows, 500):
            session.run(
                """
                UNWIND $rows AS r
                MERGE (p:Post {id: r.item_id})
                SET p.subreddit = r.community, p.has_text = r.has_text
                MERGE (c:Community {id: r.community})
                MERGE (p)-[:IN]->(c)
                """,
                rows=[{**r, "has_text": r["item_id"] in text_ids} for r in chunk],
            )

        vote_rows = vote_df[["username", "item_id", "vote", "label"]].to_dict("records")
        for chunk in _chunks(vote_rows, 500):
            session.run(
                """
                UNWIND $rows AS r
                MERGE (u:User {id: r.username})
                MERGE (p:Post {id: r.item_id})
                MERGE (u)-[v:VOTED]->(p)
                SET v.vote = r.vote, v.label = r.label, v.correct = (r.vote = r.label)
                """,
                rows=chunk,
            )

        sim_rows = []
        for i, pid in enumerate(tqdm(post_id_list, desc="FAISS → Neo4j")):
            qv = post_emb[i : i + 1]
            sims, idxs = faiss_index.search(qv, k_similar + 1)
            for rank, (sim, j) in enumerate(zip(sims[0], idxs[0])):
                if j < 0 or post_id_list[j] == pid:
                    continue
                sim_rows.append({"src": pid, "dst": post_id_list[j], "sim": float(sim), "rank": rank})
            if len(sim_rows) >= 2000:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (src:Post {id: r.src}), (dst:Post {id: r.dst})
                    MERGE (src)-[s:SIMILAR_TO]->(dst)
                    SET s.sim = r.sim, s.rank = r.rank
                    """,
                    rows=sim_rows,
                )
                sim_rows = []
        if sim_rows:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (src:Post {id: r.src}), (dst:Post {id: r.dst})
                MERGE (src)-[s:SIMILAR_TO]->(dst)
                SET s.sim = r.sim, s.rank = r.rank
                """,
                rows=sim_rows,
            )

    driver.close()
    log.info("Neo4j build completo.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Expert Finder v2")
    parser.add_argument("--k-neighbors",    type=int,   default=K_NEIGHBORS)
    parser.add_argument("--top-t",          type=int,   default=TOP_T)
    parser.add_argument("--alpha",          type=float, default=ALPHA,
                        help="Peso Signal A (local behavioral)")
    parser.add_argument("--beta",           type=float, default=BETA,
                        help="Peso Signal B (global precision). gamma=1-alpha-beta è Signal C.")
    parser.add_argument("--min-user-votes", type=int,   default=MIN_USER_VOTES)
    parser.add_argument("--output-dir",     type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--no-grid-search", action="store_true",
                        help="Salta il grid search, usa i default")
    parser.add_argument("--input-csv",      type=Path,  default=None)
    parser.add_argument("--build-neo4j",    action="store_true")
    parser.add_argument("--neo4j-uri",      default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user",     default="neo4j")
    parser.add_argument("--neo4j-pass",     default="password")
    args = parser.parse_args()

    assert args.alpha + args.beta <= 1.0, \
        f"alpha ({args.alpha}) + beta ({args.beta}) deve essere <= 1.0"

    output_dir = args.output_dir
    embed_dir  = output_dir / "embeddings"

    log.info("=" * 60)
    log.info("STEP 3 — Semantic Expert Finder (v2)")
    log.info("  alpha=%.2f (local)  beta=%.2f (global)  gamma=%.2f (topical)",
             args.alpha, args.beta, 1.0 - args.alpha - args.beta)
    log.info("=" * 60)

    # ── carica dati ──────────────────────────────────────────────────────────
    vote_df    = load_votes(args.input_csv)
    post_texts = load_post_texts()
    user_docs  = load_user_documents()

    # ── embeddings (con cache) ────────────────────────────────────────────────
    model = _get_model()
    post_emb, post_id_list = build_post_embeddings(post_texts, embed_dir, model)
    user_emb, user_id_list = build_user_embeddings(user_docs,  embed_dir, model)

    log.info(
        "User embeddings: %d utenti con profilo topico su %d totali (%.1f%%)",
        len(user_id_list),
        vote_df["username"].nunique(),
        100 * len(user_id_list) / max(vote_df["username"].nunique(), 1),
    )

    # ── FAISS ────────────────────────────────────────────────────────────────
    faiss_index = build_faiss_index(post_emb)

    # ── KFold ────────────────────────────────────────────────────────────────
    fold_metrics, fold_item_scores = run_kfold(
        vote_df        = vote_df,
        post_emb       = post_emb,
        post_id_list   = post_id_list,
        user_emb       = user_emb,
        user_id_list   = user_id_list,
        faiss_index    = faiss_index,
        default_k      = args.k_neighbors,
        default_t      = args.top_t,
        default_alpha  = args.alpha,
        default_beta   = args.beta,
        min_coverage   = args.min_user_votes,
        do_grid_search = not args.no_grid_search,
    )

    # ── salva ────────────────────────────────────────────────────────────────
    save_outputs(fold_metrics, fold_item_scores, output_dir)

    # ── Neo4j (opzionale) ────────────────────────────────────────────────────
    if args.build_neo4j:
        build_neo4j_graph(
            vote_df      = vote_df,
            post_texts   = post_texts,
            post_emb     = post_emb,
            post_id_list = post_id_list,
            faiss_index  = faiss_index,
            neo4j_uri    = args.neo4j_uri,
            neo4j_user   = args.neo4j_user,
            neo4j_pass   = args.neo4j_pass,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()