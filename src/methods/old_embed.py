"""
step3_expert_finder.py  (v5)
============================
Semantic Expert Finder — changes vs v4:

  V5-1  Systematic disagreement: users who consistently vote AGAINST moderators
        are now treated as informative sources. Reliability = |prec - 0.5|,
        effective_vote = vote * sign(prec - 0.5). Users with prec < 0.5 have
        their vote flipped; their weight is based on how far they are from
        random (0.5), not from 1.0.

  V5-2  Richer subreddit features: sub_approve_rate is joined by
        sub_approve_rate_text (text posts only), sub_approve_std (temporal
        variance across months), and sub_n_posts (size signal).

  V5-3  Disaggregated per-rank expert features: instead of top-N mean,
        the meta-model receives individual weight/local/global/signal_c
        for each of the top RANK_N experts per direction (pos and neg),
        sorted by expert_weight descending. Missing slots are zero-padded.
        Suggested by supervisor: "instead of mean and std, report
        vote1, vote2, ..., vote5".

Signals (unchanged from v4):
  A  local_prec       FAISS-neighbour precision (sim-weighted, shrinkage)
  B  sub_prec         historical subreddit precision
  C  signal_c_directed = cosine_sim if vote=-1, else 1-cosine_sim

  expert_weight = alpha*A + beta*B + (1-alpha-beta)*C_directed
  effective_vote = vote * sign(prec - 0.5)   [V5-1]
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
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
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 64

# ---------------------------------------------------------------------------
# Hyperparameter defaults
# ---------------------------------------------------------------------------
K_NEIGHBORS    = 5
MIN_USER_VOTES = 10
ALPHA          = 0.4
BETA           = 0.4
LAMBDA_SMOOTH  = 2.0
TOP_T          = 1
CAL_SAMPLE     = 500
MIN_SUB_SAMPLES = 50

# Number of per-rank expert slots in feature vector (V5-3)
RANK_N = 5

# ---------------------------------------------------------------------------
# Grid search — fixed params, skip with --no-grid-search
# ---------------------------------------------------------------------------
GRID_K     = [5, 10]
GRID_T     = [1, 3, 5, 10]
GRID_ALPHA = [0.2, 0.3, 0.4, 0.5, 0.6]
GRID_BETA  = [0.2, 0.3, 0.4, 0.5, 0.6]
VAL_GRID_SAMPLE = 1000

# ---------------------------------------------------------------------------
# Meta-model feature T values
# ---------------------------------------------------------------------------
TOP_T_LIST = [1, 3, 5, 20]          # mirrors grid T values

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

post_ids_ordered: List[str] = []


# ===========================================================================
# 1. DATA LOADING
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
                "Votes: %d rows | %d posts | %d users | %d communities",
                len(df), df["item_id"].nunique(),
                df["username"].nunique(), df["community"].nunique(),
            )
            return df

    log.warning("CSV not found — falling back to parquet.")
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
    log.info("Votes: %d rows | %d posts | %d users", len(df),
             df["item_id"].nunique(), df["username"].nunique())
    return df


def load_post_texts() -> pd.DataFrame:
    path = FETCH_DIR / "post_texts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Post texts not found at {path}.")
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["text"].notna()].copy()
    log.info("Post texts: %d / %d with text", len(df), before)
    return df


def load_user_documents() -> pd.DataFrame:
    path = FETCH_DIR / "user_documents.parquet"
    if not path.exists():
        log.warning("User documents not found — Signal C will be 0.")
        return pd.DataFrame(columns=["username", "text"])
    df = pd.read_parquet(path)
    df = df[df["text"].notna()].copy()
    log.info("User docs: %d docs | %d users", len(df), df["username"].nunique())
    return df


# ===========================================================================
# 2. EMBEDDINGS
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
        raise FileNotFoundError(f"embed_helper.py not found at {helper}.")
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
        log.info("Encoding %d texts …", len(texts))
        subprocess.run(cmd, check=True)
        vecs = np.load(out_path).astype(np.float32)
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
        return np.load(emb_path), json.load(open(ids_path))

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
    log.info("Post embeddings: %s", embeddings.shape)
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
        return np.load(emb_path), json.load(open(ids_path))

    if user_docs.empty:
        log.warning("No user docs — user embeddings empty.")
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
    log.info("User embeddings: %s", user_matrix.shape)
    return user_matrix, usernames


# ===========================================================================
# 3. FAISS INDEX
# ===========================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    log.info("FAISS index: %d vectors", index.ntotal)
    return index


# ===========================================================================
# 4. SIGNAL COMPUTATION
# ===========================================================================

def precompute_vote_precision(
    train_votes:    pd.DataFrame,
    lambda_smooth:  float,
    min_user_votes: int,
) -> pd.DataFrame:
    """
    Per-user, per-direction precision with Laplace smoothing.

    V5-1: adds two columns:
      reliability  = |prec - 0.5|   (how far from random, regardless of direction)
      bias_sign    = sign(prec - 0.5)  (+1 if user tends to agree with mods,
                                        -1 if user tends to systematically disagree)
    Users with bias_sign=-1 are contrarian — their votes will be flipped in
    compute_expert_weights so that systematic disagreement becomes useful signal.
    """
    tv = train_votes.copy()
    tv["correct"] = (tv["vote"] == tv["label"]).astype(int)

    valid_users = tv.groupby("username")["vote"].count()
    valid_users = valid_users[valid_users >= min_user_votes].index
    tv = tv[tv["username"].isin(valid_users)]

    rows = []
    for (uname, direction), grp in tv.groupby(["username", "vote"]):
        n, nc = len(grp), grp["correct"].sum()
        prec  = (nc + lambda_smooth) / (n + 2 * lambda_smooth)
        rows.append({
            "username": uname, "direction": int(direction), "subreddit": None,
            "prec": prec, "n_votes_dir": n,
            "reliability": abs(prec - 0.5),
            "bias_sign":   1 if prec >= 0.5 else -1,
        })
    for (uname, sub, direction), grp in tv.groupby(["username", "community", "vote"]):
        n, nc = len(grp), grp["correct"].sum()
        prec  = (nc + lambda_smooth) / (n + 2 * lambda_smooth)
        rows.append({
            "username": uname, "direction": int(direction), "subreddit": sub,
            "prec": prec, "n_votes_dir": n,
            "reliability": abs(prec - 0.5),
            "bias_sign":   1 if prec >= 0.5 else -1,
        })
    return pd.DataFrame(rows)


def compute_subreddit_base_rates(train_votes: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    V5-2: richer subreddit context features.
    Returns dict {subreddit: {approve_rate, approve_rate_text, approve_std, n_posts}}.
      approve_rate      — overall historical approval rate
      approve_rate_text — approval rate for text-only posts (selftext not null/empty)
      approve_std       — std of monthly approval rates (temporal variance)
      n_posts           — log(1 + n_posts) as subreddit size signal
    """
    # per-post label (one row per item)
    posts = train_votes.drop_duplicates("item_id").copy()

    # detect text posts: if dataset has a 'selftext' column use it, else fallback
    has_selftext = "selftext" in posts.columns

    # parse date if available for temporal variance
    has_date = "created_utc" in posts.columns or "date" in posts.columns
    date_col  = "created_utc" if "created_utc" in posts.columns else ("date" if "date" in posts.columns else None)

    result = {}
    for sub, grp in posts.groupby("community"):
        approved = (grp["label"] == 1)
        rate     = float(approved.mean())
        n        = len(grp)

        # text-post approval rate
        if has_selftext:
            text_mask = grp["selftext"].notna() & (grp["selftext"] != "")
            rate_text = float(approved[text_mask].mean()) if text_mask.any() else rate
        else:
            rate_text = rate

        # temporal variance: std of monthly approval rates
        if date_col is not None:
            try:
                grp2 = grp.copy()
                grp2["_month"] = pd.to_datetime(grp2[date_col], unit="s", errors="coerce").dt.to_period("M")
                monthly = grp2.groupby("_month")["label"].apply(lambda x: (x == 1).mean())
                approve_std = float(monthly.std()) if len(monthly) > 1 else 0.0
            except Exception:
                approve_std = 0.0
        else:
            approve_std = 0.0

        result[sub] = {
            "approve_rate":      rate,
            "approve_rate_text": rate_text,
            "approve_std":       approve_std,
            "n_posts":           float(np.log1p(n)),
        }
    return result


def compute_expert_weights(
    test_item_ids:  List[str],
    train_item_ids: set,
    vote_df:        pd.DataFrame,
    prec_df:        pd.DataFrame,
    post_emb:       np.ndarray,
    post_id2idx:    Dict[str, int],
    user_emb:       np.ndarray,
    user_id2idx:    Dict[str, int],
    faiss_index:    faiss.Index,
    k_neighbors:    int,
    alpha:          float,
    beta:           float,
    lambda_smooth:  float,
) -> pd.DataFrame:
    """
    expert_weight(u, P) = alpha*A + beta*B + gamma*C_directed
      A = local precision on FAISS neighbours (sim-weighted, Bayesian shrinkage)
      B = subreddit-level historical precision
      C = signal_c_directed (direction-aware topical similarity)

    V5-1: systematic disagreement.
      reliability(u) = |prec - 0.5|  — how informative the user is regardless of direction
      effective_vote = vote * bias_sign  — flipped for contrarian users
      expert_weight is based on reliability, not raw precision, so contrarians
      with prec=0.1 get the same weight as concordants with prec=0.9.
    """
    # build lookup maps: (username, direction) -> (prec, reliability, bias_sign)
    global_map: Dict[tuple, tuple] = {}   # -> (prec, reliability, bias_sign)
    sub_map:    Dict[tuple, tuple] = {}
    for _, r in prec_df.iterrows():
        val = (float(r["prec"]), float(r["reliability"]), int(r["bias_sign"]))
        if r["subreddit"] is None:
            global_map[(r["username"], r["direction"])] = val
        else:
            sub_map[(r["username"], r["direction"], r["subreddit"])] = val

    laplace_default = lambda_smooth / (2 * lambda_smooth)  # 0.5 -> reliability=0, bias_sign=1

    gamma = 1.0 - alpha - beta

    train_votes = vote_df[vote_df["item_id"].isin(train_item_ids)].copy()
    train_votes["correct"] = (train_votes["vote"] == train_votes["label"]).astype(int)
    item_votes: Dict[str, pd.DataFrame] = {
        iid: grp[["username", "vote", "correct"]]
        for iid, grp in train_votes.groupby("item_id")
    }

    post_sub = (
        vote_df[vote_df["item_id"].isin(test_item_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )
    test_votes = vote_df[vote_df["item_id"].isin(test_item_ids)][["item_id", "username", "vote"]]

    rows = []
    for item_id in test_item_ids:
        if item_id not in post_id2idx:
            continue

        subreddit = post_sub.get(item_id)
        p_idx     = post_id2idx[item_id]
        query_vec = post_emb[p_idx : p_idx + 1]
        p_vec     = post_emb[p_idx]

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

        for _, vrow in test_votes[test_votes["item_id"] == item_id].iterrows():
            uname     = vrow["username"]
            direction = int(vrow["vote"])

            # lookup precision, reliability, bias_sign
            g_val  = global_map.get((uname, direction), (laplace_default, 0.0, 1))
            s_val  = sub_map.get((uname, direction, subreddit), g_val)
            sub_p, reliability, bias_sign = s_val

            # effective_vote: flipped for contrarians (bias_sign=-1)
            effective_vote = direction * bias_sign

            # Signal A: local precision on FAISS neighbours
            # use reliability (|prec-0.5|) as the correctness signal so that
            # contrarians contribute positively when their effective_vote is used
            loc_correct = loc_weight = 0.0
            for nid, sim in neighbours:
                if nid not in item_votes:
                    continue
                for _, ur in item_votes[nid][item_votes[nid]["username"] == uname].iterrows():
                    if int(ur["vote"]) == direction:
                        # correct = 1 if user agreed with mod; for contrarians
                        # "correct" means they disagreed with mod (bias_sign=-1),
                        # so we use bias_sign to align the signal
                        aligned = float(ur["correct"]) if bias_sign == 1 else (1.0 - float(ur["correct"]))
                        loc_correct += sim * aligned
                        loc_weight  += sim

            local_reliability = (loc_correct + lambda_smooth * reliability * mean_sim) / \
                                 (loc_weight  + lambda_smooth * mean_sim)

            # Signal C — direction-aware (V4-1 unchanged)
            if uname in user_id2idx and user_emb.shape[0] > 0:
                raw_sim = max(0.0, float(np.dot(user_emb[user_id2idx[uname]], p_vec)))
            else:
                raw_sim = reliability  # fallback: use reliability as proxy
            signal_c          = raw_sim
            signal_c_directed = raw_sim if direction == -1 else (1.0 - raw_sim)

            # expert_weight based on reliability (V5-1), not raw precision
            expert_w = alpha * local_reliability + beta * reliability + gamma * signal_c_directed

            rows.append({
                "item_id":           item_id,
                "username":          uname,
                "vote":              direction,
                "effective_vote":    effective_vote,
                "expert_weight":     float(expert_w),
                "local_prec":        float(local_reliability),
                "global_prec":       float(sub_p),
                "reliability":       float(reliability),
                "bias_sign":         int(bias_sign),
                "signal_c":          float(signal_c),
                "signal_c_directed": float(signal_c_directed),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "item_id", "username", "vote", "effective_vote", "expert_weight",
            "local_prec", "global_prec", "reliability", "bias_sign",
            "signal_c", "signal_c_directed",
        ])
    return pd.DataFrame(rows)


# ===========================================================================
# 5. FEATURE EXTRACTION  (V4-2 early fusion + V5-2 subreddit + V5-3 per-rank)
# ===========================================================================

def extract_post_features(
    item_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    base_rates: Dict[str, Dict[str, float]],
    top_t_list: List[int] = TOP_T_LIST,
    rank_n:     int = RANK_N,
) -> pd.DataFrame:
    """
    Feature vector per post for the meta-model.

    V5-2: richer subreddit features (approve_rate, approve_rate_text,
          approve_std, n_posts) instead of a single approve_rate scalar.

    V5-3: per-rank disaggregated expert features (supervisor suggestion).
          Instead of top-N mean/std, the meta-model receives individual
          weight, local_prec, global_prec, reliability, signal_c_directed,
          effective_vote for rank-1 through rank-RANK_N experts,
          separately for pos and neg directions. Missing slots are 0-padded.

    V4-2: direction-disaggregated aggregates kept for complementarity.
    """
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "username", "vote"]]
    post_sub  = (
        vote_df[vote_df["item_id"].isin(item_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )
    vote_col = "effective_vote" if (not weights_df.empty and "effective_vote" in weights_df.columns) else "vote"

    # default subreddit stats for unknown subs
    default_sub = {"approve_rate": 0.5, "approve_rate_text": 0.5, "approve_std": 0.0, "n_posts": 0.0}

    rows = []
    for item_id in item_ids:
        rv  = raw_votes[raw_votes["item_id"] == item_id]["vote"].values
        pw  = weights_df[weights_df["item_id"] == item_id] if not weights_df.empty else pd.DataFrame()
        sub = post_sub.get(item_id, "__unknown__")

        feats: Dict[str, float] = {}

        # --- raw vote stats ---
        feats["raw_vote_mean"] = float(rv.mean()) if len(rv) > 0 else 0.0
        feats["raw_vote_std"]  = float(rv.std())  if len(rv) > 1 else 0.0
        feats["n_voters"]      = float(len(rv))
        feats["n_upvoters"]    = float((rv ==  1).sum())
        feats["n_downvoters"]  = float((rv == -1).sum())

        # --- weighted vote at multiple T ---
        for t in top_t_list:
            wv = _compute_weighted_votes([item_id], vote_df, weights_df, t)
            feats[f"wv_top{t}"] = wv.get(item_id, feats["raw_vote_mean"])

        min_t = min(top_t_list)
        feats["expert_crowd_divergence"] = feats[f"wv_top{min_t}"] - feats["raw_vote_mean"]

        # --- V5-2: subreddit context features ---
        sr = base_rates.get(sub, default_sub)
        feats["sub_approve_rate"]      = sr["approve_rate"]
        feats["sub_approve_rate_text"] = sr["approve_rate_text"]
        feats["sub_approve_std"]       = sr["approve_std"]
        feats["sub_n_posts"]           = sr["n_posts"]

        if not pw.empty:
            feats["mean_expert_weight"] = float(pw["expert_weight"].mean())
            feats["std_expert_weight"]  = float(pw["expert_weight"].std()) if len(pw) > 1 else 0.0
            feats["n_experts"]          = float(len(pw))

            pw_pos = pw[pw["vote"] ==  1].nlargest(rank_n, "expert_weight").reset_index(drop=True)
            pw_neg = pw[pw["vote"] == -1].nlargest(rank_n, "expert_weight").reset_index(drop=True)

            # V4-2 aggregates (kept)
            for tag, top in [("pos", pw_pos), ("neg", pw_neg)]:
                dv = 1.0 if tag == "pos" else -1.0
                if not top.empty:
                    w = top["expert_weight"].values.clip(min=1e-9)
                    feats[f"n_{tag}_experts"]              = float(len(pw[pw["vote"] == (1 if tag=="pos" else -1)]))
                    feats[f"sum_weight_{tag}"]             = float(w.sum())
                    feats[f"top{rank_n}_{tag}_mean_local"] = float(top["local_prec"].mean())
                    feats[f"top{rank_n}_{tag}_mean_global"]= float(top["global_prec"].mean())
                    feats[f"top{rank_n}_{tag}_mean_signal_c"] = float(top["signal_c_directed"].mean())
                    feats[f"top{rank_n}_{tag}_weighted_vote"] = float(
                        np.dot(w, top[vote_col].values.astype(float)) / w.sum()
                    )
                else:
                    feats[f"n_{tag}_experts"]              = 0.0
                    feats[f"sum_weight_{tag}"]             = 0.0
                    feats[f"top{rank_n}_{tag}_mean_local"] = 0.5
                    feats[f"top{rank_n}_{tag}_mean_global"]= 0.5
                    feats[f"top{rank_n}_{tag}_mean_signal_c"] = 0.5
                    feats[f"top{rank_n}_{tag}_weighted_vote"] = dv

            # V5-3: per-rank individual features
            for tag, top in [("pos", pw_pos), ("neg", pw_neg)]:
                dv = 1.0 if tag == "pos" else -1.0
                for r in range(rank_n):
                    if r < len(top):
                        row = top.iloc[r]
                        feats[f"r{r+1}_{tag}_weight"]      = float(row["expert_weight"])
                        feats[f"r{r+1}_{tag}_local"]       = float(row["local_prec"])
                        feats[f"r{r+1}_{tag}_global"]      = float(row["global_prec"])
                        feats[f"r{r+1}_{tag}_reliability"] = float(row.get("reliability", 0.0))
                        feats[f"r{r+1}_{tag}_signal_c"]    = float(row["signal_c_directed"])
                        feats[f"r{r+1}_{tag}_evote"]       = float(row[vote_col])
                    else:
                        # zero-pad missing slots
                        feats[f"r{r+1}_{tag}_weight"]      = 0.0
                        feats[f"r{r+1}_{tag}_local"]       = 0.0
                        feats[f"r{r+1}_{tag}_global"]      = 0.0
                        feats[f"r{r+1}_{tag}_reliability"] = 0.0
                        feats[f"r{r+1}_{tag}_signal_c"]    = 0.0
                        feats[f"r{r+1}_{tag}_evote"]       = dv
        else:
            feats["mean_expert_weight"] = 0.0
            feats["std_expert_weight"]  = 0.0
            feats["n_experts"]          = 0.0
            for tag, dv in [("pos", 1.0), ("neg", -1.0)]:
                feats[f"n_{tag}_experts"]              = 0.0
                feats[f"sum_weight_{tag}"]             = 0.0
                feats[f"top{rank_n}_{tag}_mean_local"] = 0.5
                feats[f"top{rank_n}_{tag}_mean_global"]= 0.5
                feats[f"top{rank_n}_{tag}_mean_signal_c"] = 0.5
                feats[f"top{rank_n}_{tag}_weighted_vote"] = dv
                for r in range(rank_n):
                    feats[f"r{r+1}_{tag}_weight"]      = 0.0
                    feats[f"r{r+1}_{tag}_local"]       = 0.0
                    feats[f"r{r+1}_{tag}_global"]      = 0.0
                    feats[f"r{r+1}_{tag}_reliability"] = 0.0
                    feats[f"r{r+1}_{tag}_signal_c"]    = 0.0
                    feats[f"r{r+1}_{tag}_evote"]       = dv

        feats["item_id"] = item_id
        rows.append(feats)

    return pd.DataFrame(rows)


def get_meta_feature_cols(feat_df: pd.DataFrame) -> List[str]:
    return [c for c in feat_df.columns if c != "item_id"]


# ===========================================================================
# 6. META-MODEL  (I5: GradientBoostingClassifier replaces LogisticRegression)
# ===========================================================================

def train_meta_model(
    feat_df:   pd.DataFrame,
    labels:    Dict[str, int],
    feat_cols: List[str],
) -> Tuple[GradientBoostingClassifier, StandardScaler]:
    """
    I5: GradientBoostingClassifier.
    Handles correlated features and non-linear interactions natively.
    subsample=0.8 provides implicit regularisation against overfitting.
    Class imbalance handled via sample_weight (balanced reweighting).
    """
    items = feat_df[feat_df["item_id"].isin(labels)]["item_id"].tolist()
    X = feat_df.set_index("item_id").loc[items, feat_cols].values.astype(float)
    y = np.array([labels[i] for i in items])

    # balanced sample weights
    counts   = {c: (y == c).sum() for c in np.unique(y)}
    max_cnt  = max(counts.values())
    sw       = np.array([max_cnt / counts[yi] for yi in y])

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators      = 200,
        learning_rate     = 0.05,
        max_depth         = 4,
        subsample         = 0.8,
        min_samples_leaf  = 20,
        random_state      = 42,
    )
    clf.fit(X_s, y, sample_weight=sw)

    # log top-3 feature importances
    top3 = sorted(zip(feat_cols, clf.feature_importances_),
                  key=lambda x: x[1], reverse=True)[:3]
    log.info("  Meta-model top features: %s",
             {k: round(v, 3) for k, v in top3})
    return clf, scaler


def predict_meta_model(
    feat_df:   pd.DataFrame,
    item_ids:  List[str],
    clf:       GradientBoostingClassifier,
    scaler:    StandardScaler,
    feat_cols: List[str],
) -> Dict[str, Tuple[int, float]]:
    """Returns {item_id: (predicted_label, proba_positive)}."""
    df = feat_df.set_index("item_id")
    result = {}
    for item_id in item_ids:
        if item_id not in df.index:
            result[item_id] = (1, 0.5)
            continue
        x     = df.loc[item_id, feat_cols].values.astype(float).reshape(1, -1)
        x_s   = scaler.transform(x)
        pred  = int(clf.predict(x_s)[0])
        proba = float(clf.predict_proba(x_s)[0, list(clf.classes_).index(1)])
        result[item_id] = (pred, proba)
    return result


# ===========================================================================
# 7. WEIGHTED VOTE
# ===========================================================================

def _compute_weighted_votes(
    item_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    top_t:      int,
) -> Dict[str, float]:
    """Weighted vote using effective_vote (flipped for contrarians) if available."""
    if weights_df.empty:
        raw = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "vote"]]
        return {iid: float(grp["vote"].mean()) for iid, grp in raw.groupby("item_id")}

    w         = weights_df[weights_df["item_id"].isin(item_ids)]
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "username", "vote"]]
    vote_col  = "effective_vote" if "effective_vote" in w.columns else "vote"

    result = {}
    for item_id in item_ids:
        pw = w[w["item_id"] == item_id]
        if pw.empty:
            rv = raw_votes[raw_votes["item_id"] == item_id]["vote"].values
            result[item_id] = float(rv.mean()) if len(rv) > 0 else 0.0
            continue
        top     = pw.nlargest(top_t, "expert_weight")
        weights = top["expert_weight"].values.clip(min=1e-9)
        votes   = top[vote_col].values.astype(float)
        result[item_id] = float(np.dot(weights, votes) / weights.sum())
    return result


# ===========================================================================
# 8. THRESHOLD CALIBRATION
# ===========================================================================

def calibrate_threshold_weighted(
    sample_ids:  List[str],
    vote_df:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    top_t:       int,
    neg_weight:  float = 2.0,
) -> float:
    """Find threshold maximising weighted macro-F1 (neg class upweighted 2x)."""
    labels = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
    )
    wv_scores = _compute_weighted_votes(sample_ids, vote_df, weights_df, top_t)
    common    = [iid for iid in wv_scores if iid in labels]
    if not common:
        return 0.0

    y_true   = np.array([labels[iid] for iid in common])
    y_score  = np.array([wv_scores[iid] for iid in common])
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
    """Per-subreddit threshold; falls back to global for small subreddits."""
    global_thr = calibrate_threshold_weighted(sample_ids, vote_df, weights_df, top_t)
    thresholds: Dict[Optional[str], float] = {None: global_thr}

    post_sub = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )
    sub_groups: Dict[str, List[str]] = {}
    for iid in sample_ids:
        sub = post_sub.get(iid)
        if sub:
            sub_groups.setdefault(sub, []).append(iid)

    n_specific = 0
    for sub, ids in sub_groups.items():
        if len(ids) < min_samples:
            thresholds[sub] = global_thr
        else:
            thresholds[sub] = calibrate_threshold_weighted(ids, vote_df, weights_df, top_t)
            n_specific += 1

    log.info("  Thresholds: global=%.4f | %d subreddit-specific", global_thr, n_specific)
    return thresholds


# ===========================================================================
# 9. GRID SEARCH  (I3+I4: extended T grid, pure-C combo covered by (0,0))
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
    """Grid over K x T x alpha x beta; optimises macro-F1 on val sample."""
    labels  = (
        vote_df[vote_df["item_id"].isin(val_ids)]
        .drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
    )
    val_ids = [iid for iid in val_ids if iid in labels]
    if not val_ids:
        return K_NEIGHBORS, TOP_T, ALPHA, BETA

    best = {"f1": -1.0, "k": K_NEIGHBORS, "t": TOP_T, "alpha": ALPHA, "beta": BETA}

    n_combos = (
        len(GRID_K) * len(GRID_T)
        * sum(1 for a in GRID_ALPHA for b in GRID_BETA if a + b <= 1.0)
    )
    log.info("  Grid search: %d combos on %d val posts …", n_combos, len(val_ids))

    for alpha in GRID_ALPHA:
        for beta in [b for b in GRID_BETA if alpha + b <= 1.0]:
            for k in GRID_K:
                weights_df = compute_expert_weights(
                    val_ids, train_ids, vote_df, prec_df,
                    post_emb, post_id2idx, user_emb, user_id2idx,
                    faiss_index, k, alpha, beta, lambda_smooth,
                )
                for t in GRID_T:
                    wv     = _compute_weighted_votes(val_ids, vote_df, weights_df, t)
                    common = [iid for iid in wv if iid in labels]
                    if not common:
                        continue
                    y_true  = np.array([labels[iid] for iid in common])
                    y_score = np.array([wv[iid] for iid in common])

                    best_thr, best_f1_t = 0.0, -1.0
                    for thr in np.linspace(y_score.min(), y_score.max(), 50):
                        y_pred = np.where(y_score >= thr, 1, -1)
                        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                        if f1 > best_f1_t:
                            best_f1_t, best_thr = f1, float(thr)

                    if best_f1_t > best["f1"]:
                        best = {"f1": best_f1_t, "k": k, "t": t, "alpha": alpha, "beta": beta}

    log.info("  Best: k=%d t=%d alpha=%.2f beta=%.2f macro_f1=%.4f",
             best["k"], best["t"], best["alpha"], best["beta"], best["f1"])
    return best["k"], best["t"], best["alpha"], best["beta"]


# ===========================================================================
# 10. PREDICTION
# ===========================================================================

def predict_fold(
    test_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    feat_df:    pd.DataFrame,
    feat_cols:  List[str],
    clf:        GradientBoostingClassifier,
    scaler:     StandardScaler,
    thresholds: Dict[Optional[str], float],
    top_t:      int,
) -> pd.DataFrame:
    labels = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id")[["item_id", "label"]]
    )
    post_sub = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )

    meta_preds = predict_meta_model(feat_df, test_ids, clf, scaler, feat_cols)
    wv         = _compute_weighted_votes(test_ids, vote_df, weights_df, top_t)
    w_ids      = set(weights_df["item_id"].unique()) if not weights_df.empty else set()

    rows = []
    for item_id in test_ids:
        sub      = post_sub.get(item_id)
        thr      = thresholds.get(sub, thresholds.get(None, 0.0))
        score    = wv.get(item_id, 0.0)
        fallback = item_id not in w_ids
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
# 11. METRICS
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
# 12. KFOLD
# ===========================================================================

def run_kfold(
    vote_df:        pd.DataFrame,
    post_emb:       np.ndarray,
    post_id_list:   List[str],
    user_emb:       np.ndarray,
    user_id_list:   List[str],
    faiss_index:    faiss.Index,
    default_k:      int,
    default_t:      int,
    default_alpha:  float,
    default_beta:   float,
    min_coverage:   int,
    do_grid_search: bool = True,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    global post_ids_ordered
    post_ids_ordered = post_id_list

    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}
    user_id2idx = {uid: i for i, uid in enumerate(user_id_list)}

    labeled_items = (
        vote_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"]).reset_index(drop=True)
    )
    log.info("Labeled items: %d", len(labeled_items))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics:     List[Dict]         = []
    fold_item_scores: List[pd.DataFrame] = []
    fold_weights:     List[pd.DataFrame] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(labeled_items)):
        log.info("--- Fold %d / %d ---", fold_idx + 1, N_FOLDS)

        train_ids    = set(labeled_items.iloc[train_val_idx]["item_id"].tolist())
        test_ids     = set(labeled_items.iloc[test_idx]["item_id"].tolist())
        log.info("  train=%d  test=%d", len(train_ids), len(test_ids))

        train_vote_df = vote_df[vote_df["item_id"].isin(train_ids)]
        prec_df       = precompute_vote_precision(train_vote_df, LAMBDA_SMOOTH, min_coverage)
        base_rates    = compute_subreddit_base_rates(train_vote_df)
        log.info("  reliable users (>=%d votes): %d", min_coverage, prec_df["username"].nunique())

        if do_grid_search:
            train_list   = list(train_ids)
            np.random.shuffle(train_list)
            val_grid_ids = train_list[:min(VAL_GRID_SAMPLE, len(train_list))]
            train_grid   = set(train_list[min(VAL_GRID_SAMPLE, len(train_list)):])
            k, t, alpha, beta = grid_search_hyperparams(
                val_grid_ids, train_grid, vote_df, prec_df,
                post_emb, post_id2idx, user_emb, user_id2idx,
                faiss_index, LAMBDA_SMOOTH,
            )
        else:
            k, t, alpha, beta = default_k, default_t, default_alpha, default_beta

        test_id_list = list(test_ids)
        weights_df   = compute_expert_weights(
            test_id_list, train_ids, vote_df, prec_df,
            post_emb, post_id2idx, user_emb, user_id2idx,
            faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
        )

        train_sample = list(train_ids)
        np.random.shuffle(train_sample)
        train_sample = train_sample[:min(CAL_SAMPLE, len(train_sample))]

        weights_cal = compute_expert_weights(
            train_sample, train_ids - set(train_sample), vote_df, prec_df,
            post_emb, post_id2idx, user_emb, user_id2idx,
            faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
        )
        thresholds = calibrate_thresholds_per_subreddit(train_sample, vote_df, weights_cal, t)

        log.info("  Extracting features for meta-model …")
        train_feat_df = extract_post_features(train_sample, vote_df, weights_cal, base_rates)
        feat_cols     = get_meta_feature_cols(train_feat_df)
        train_labels  = (
            vote_df[vote_df["item_id"].isin(train_sample)]
            .drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
        )
        clf, scaler = train_meta_model(train_feat_df, train_labels, feat_cols)

        test_feat_df = extract_post_features(test_id_list, vote_df, weights_df, base_rates)

        decisions = predict_fold(
            test_id_list, vote_df, weights_df,
            test_feat_df, feat_cols, clf, scaler, thresholds, t,
        )
        decisions["fold"]  = fold_idx
        decisions["k"]     = k
        decisions["top_t"] = t
        decisions["alpha"] = alpha
        decisions["beta"]  = beta

        if decisions["label"].nunique() < 2:
            log.warning("Fold %d: single class in test — skipped.", fold_idx + 1)
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
            100 * metrics["fallback_pct"], alpha, beta, 1.0 - alpha - beta,
        )

        fold_metrics.append(metrics)
        fold_item_scores.append(decisions)
        weights_df["fold"] = fold_idx
        fold_weights.append(weights_df)

    return fold_metrics, fold_item_scores, fold_weights


# ===========================================================================
# 13. OUTPUT
# ===========================================================================

def save_outputs(
    fold_metrics:     List[Dict],
    fold_item_scores: List[pd.DataFrame],
    fold_weights:     List[pd.DataFrame],
    output_dir:       Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(fold_item_scores, ignore_index=True).to_parquet(
        output_dir / "item_scores.parquet", index=False
    )
    pd.concat(fold_weights, ignore_index=True).to_parquet(
        output_dir / "weights.parquet", index=False
    )
    pd.DataFrame([
        {k: v for k, v in m.items() if isinstance(v, (int, float, bool))}
        for m in fold_metrics
    ]).to_parquet(output_dir / "fold_details.parquet", index=False)

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
        "Saved to %s | macro_f1=%.4f±%.4f  roc_auc=%.4f±%.4f  f1_pos=%.4f  f1_neg=%.4f",
        output_dir,
        agg["macro_f1"], agg["macro_f1_std"],
        agg["roc_auc"],  agg["roc_auc_std"],
        agg["f1_pos"],   agg["f1_neg"],
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Expert Finder v4")
    parser.add_argument("--k-neighbors",    type=int,   default=K_NEIGHBORS)
    parser.add_argument("--top-t",          type=int,   default=TOP_T)
    parser.add_argument("--alpha",          type=float, default=ALPHA)
    parser.add_argument("--beta",           type=float, default=BETA)
    parser.add_argument("--min-user-votes", type=int,   default=MIN_USER_VOTES)
    parser.add_argument("--output-dir",     type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--no-grid-search", action="store_true")
    parser.add_argument("--input-csv",      type=Path,  default=None)
    args = parser.parse_args()

    assert args.alpha + args.beta <= 1.0, \
        f"alpha + beta must be <= 1.0 (got {args.alpha + args.beta:.2f})"

    embed_dir = args.output_dir / "embeddings"

    log.info("=== SEF v4 | alpha=%.2f beta=%.2f gamma=%.2f ===",
             args.alpha, args.beta, 1.0 - args.alpha - args.beta)

    vote_df    = load_votes(args.input_csv)
    post_texts = load_post_texts()
    user_docs  = load_user_documents()

    model = _get_model()
    post_emb, post_id_list = build_post_embeddings(post_texts, embed_dir, model)
    user_emb, user_id_list = build_user_embeddings(user_docs,  embed_dir, model)
    log.info("User profiles: %d / %d (%.1f%%)",
             len(user_id_list), vote_df["username"].nunique(),
             100 * len(user_id_list) / max(vote_df["username"].nunique(), 1))

    faiss_index = build_faiss_index(post_emb)

    fold_metrics, fold_item_scores, fold_weights = run_kfold(
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

    save_outputs(fold_metrics, fold_item_scores, fold_weights, args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()