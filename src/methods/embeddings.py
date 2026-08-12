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
from src.utils.splits import discover_splits
import subprocess
import sys
import tempfile
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log_msg = "xgboost not installed - skipping XGB. Run: pip install xgboost"

# Paths
STEP1_DIR    = Path("data/interim/reddit")
SPLITS_DIR   = Path("data/splits/reddit")
FETCH_DIR    = Path("data/interim/reddit")
OUTPUT_DIR   = Path("results/reddit")
CACHE_DIR    = Path("cache/embeddings")
ORIGINAL_CSV = Path("data/processed/final_intersection_dataset.csv")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 64

# Hyperparameter defaults
K_NEIGHBORS    = 5
MIN_USER_VOTES = 10
ALPHA          = 0.5
BETA           = 0.3
LAMBDA_SMOOTH  = 2.0
TOP_T          = 3
CAL_SAMPLE     = 4000
MIN_SUB_SAMPLES = 50

# Number of per-rank expert slots in feature vector
RANK_N = 5

# Minimum reliable downvoter experts required to activate the neg predictor
MIN_NEG_EXPERTS = 1
NEG_RELIABILITY_THR = 0.1   # minimum reliability to count as "reliable"

# Grid search
GRID_K     = [5, 10]
GRID_T     = [1, 3, 5]
GRID_ALPHA = [0.2, 0.3, 0.4, 0.5, 0.6]
GRID_BETA  = [0.2, 0.3, 0.4, 0.5, 0.6]
# valid combos: alpha+beta <= 1.0
VAL_GRID_SAMPLE = 3000

# Meta-model feature T values
TOP_T_LIST = [1, 3, 5, 20]          # mirrors grid T values

# Evaluation
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


# 1. DATA LOADING

def load_votes(input_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load votes from its configured source."""
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

    log.warning("CSV not found - falling back to parquet.")
    path = STEP1_DIR / "filtered_votes.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        dfs = [
            pd.read_parquet(SPLITS_DIR / "random" / f"{s}_votes.parquet")
            for s in ("train", "val", "test")
            if (SPLITS_DIR / "random" / f"{s}_votes.parquet").exists()
        ]
        df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"]  = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    log.info("Votes: %d rows | %d posts | %d users", len(df),
             df["item_id"].nunique(), df["username"].nunique())
    return df


def _load_split_votes(path: Path) -> pd.DataFrame:
    """Load split votes from its configured source."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"]  = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def load_post_texts() -> pd.DataFrame:
    """Load post texts from its configured source."""
    path = FETCH_DIR / "post_texts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Post texts not found at {path}.")
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["text"].notna()].copy()
    log.info("Post texts: %d / %d with text", len(df), before)
    return df


def load_user_documents() -> pd.DataFrame:
    """Load user documents from its configured source."""
    path = FETCH_DIR / "user_documents.parquet"
    if not path.exists():
        log.warning("User documents not found - Signal C will be 0.")
        return pd.DataFrame(columns=["username", "text"])
    df = pd.read_parquet(path)
    df = df[df["text"].notna()].copy()
    log.info("User docs: %d docs | %d users", len(df), df["username"].nunique())
    return df


# 2. EMBEDDINGS

def _get_model() -> str:
    """Return model for the supplied input."""
    return EMBEDDING_MODEL


def _encode_texts(
    texts:      List[str],
    model_name: str,
    normalize:  bool = True,
    batch_size: int  = BATCH_SIZE,
) -> np.ndarray:
    """Encode text batches with the configured embedding model."""
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
        log.info("Encoding %d texts ...", len(texts))
        subprocess.run(cmd, check=True)
        vecs = np.load(out_path).astype(np.float32)
    return vecs


def build_post_embeddings(
    post_texts: pd.DataFrame,
    embed_dir:  Path,
    model:      Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Build post embeddings from the supplied data."""
    emb_path    = embed_dir / "post_embeddings.npy"
    ids_path    = embed_dir / "post_ids.json"
    partial_dir = embed_dir / "post_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached post embeddings ...")
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
    """Build user embeddings from the supplied data."""
    emb_path    = embed_dir / "user_embeddings.npy"
    ids_path    = embed_dir / "user_ids.json"
    partial_dir = embed_dir / "user_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached user embeddings ...")
        return np.load(emb_path), json.load(open(ids_path))

    if user_docs.empty:
        log.warning("No user docs - user embeddings empty.")
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


# 3. FAISS INDEX

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build faiss index from the supplied data."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    log.info("FAISS index: %d vectors", index.ntotal)
    return index


# 4. SIGNAL COMPUTATION

def precompute_vote_precision(
    train_votes:    pd.DataFrame,
    lambda_smooth:  float,
    min_user_votes: int,
) -> pd.DataFrame:
    """Precompute vote precision for reuse."""
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
    """Compute subreddit base rates from the supplied data."""
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
    """Compute expert weights from the supplied data."""
    # lookup tables
    global_map: Dict[tuple, tuple] = {}
    sub_map:    Dict[tuple, tuple] = {}
    for row in prec_df.itertuples(index=False):
        val = (float(row.prec), float(row.reliability), int(row.bias_sign))
        if row.subreddit is None:
            global_map[(row.username, row.direction)] = val
        else:
            sub_map[(row.username, row.direction, row.subreddit)] = val

    laplace_default = lambda_smooth / (2 * lambda_smooth)
    gamma = 1.0 - alpha - beta

    # training vote lookup: {item_id: {username: [(vote, correct)]}}
    train_votes = vote_df[vote_df["item_id"].isin(train_item_ids)].copy()
    train_votes["correct"] = (train_votes["vote"] == train_votes["label"]).astype(np.int8)
    train_lookup: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for row in train_votes.itertuples(index=False):
        d = train_lookup.setdefault(row.item_id, {})
        d.setdefault(row.username, []).append((int(row.vote), float(row.correct)))

    valid_ids = [iid for iid in test_item_ids if iid in post_id2idx]
    if not valid_ids:
        return pd.DataFrame(columns=[
            "item_id", "username", "vote", "effective_vote", "expert_weight",
            "local_prec", "global_prec", "reliability", "bias_sign",
            "signal_c", "signal_c_directed",
        ])

    post_sub = (
        vote_df[vote_df["item_id"].isin(valid_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )
    test_votes_by_post: Dict[str, List[Tuple[str, int]]] = {}
    for row in vote_df[vote_df["item_id"].isin(valid_ids)][["item_id","username","vote"]].itertuples(index=False):
        test_votes_by_post.setdefault(row.item_id, []).append((row.username, int(row.vote)))

    # V6-2a: batch FAISS
    post_indices = [post_id2idx[iid] for iid in valid_ids]
    batch_sims, batch_idxs = faiss_index.search(post_emb[post_indices], k_neighbors + 1)

    rows = []
    train_ids_set = train_item_ids

    for idx, item_id in enumerate(valid_ids):
        subreddit = post_sub.get(item_id)
        p_vec     = post_emb[post_indices[idx]]

        # neighbours
        neighbours: List[Tuple[str, float]] = []
        for sim, j in zip(batch_sims[idx], batch_idxs[idx]):
            if j < 0:
                continue
            nid = post_ids_ordered[j]
            if nid != item_id and nid in train_ids_set:
                neighbours.append((nid, max(float(sim), 0.0)))
        mean_sim = float(np.mean([s for _, s in neighbours])) if neighbours else 1.0

        voters = test_votes_by_post.get(item_id, [])
        if not voters:
            continue

        # V6-2b: Signal C for all voters at once
        known = [(i, u) for i, (u, _) in enumerate(voters) if u in user_id2idx]
        sig_c_map: Dict[str, float] = {}
        if known and user_emb.shape[0] > 0:
            uidxs   = [user_id2idx[u] for _, u in known]
            raw_c   = np.clip(user_emb[uidxs] @ p_vec, 0.0, 1.0)
            for (_, u), c in zip(known, raw_c):
                sig_c_map[u] = float(c)

        for uname, direction in voters:
            g_val = global_map.get((uname, direction), (laplace_default, 0.0, 1))
            s_val = sub_map.get((uname, direction, subreddit), g_val)
            sub_p, reliability, bias_sign = s_val
            effective_vote = direction * bias_sign

            # Signal A via dict lookup
            loc_correct = loc_weight = 0.0
            for nid, sim in neighbours:
                ndata = train_lookup.get(nid, {}).get(uname)
                if ndata is None:
                    continue
                for v, correct in ndata:
                    if v == direction:
                        aligned = correct if bias_sign == 1 else (1.0 - correct)
                        loc_correct += sim * aligned
                        loc_weight  += sim

            local_rel = (loc_correct + lambda_smooth * reliability * mean_sim) /                         (loc_weight  + lambda_smooth * mean_sim)

            raw_c      = sig_c_map.get(uname, reliability)
            signal_c_d = raw_c if direction == -1 else (1.0 - raw_c)
            expert_w   = alpha * local_rel + beta * reliability + gamma * signal_c_d

            rows.append({
                "item_id":           item_id,
                "username":          uname,
                "vote":              direction,
                "effective_vote":    effective_vote,
                "expert_weight":     float(expert_w),
                "local_prec":        float(local_rel),
                "global_prec":       float(sub_p),
                "reliability":       float(reliability),
                "bias_sign":         int(bias_sign),
                "signal_c":          float(raw_c),
                "signal_c_directed": float(signal_c_d),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "item_id", "username", "vote", "effective_vote", "expert_weight",
            "local_prec", "global_prec", "reliability", "bias_sign",
            "signal_c", "signal_c_directed",
        ])
    return pd.DataFrame(rows)

# 5. FEATURE EXTRACTION  (V4-2 early fusion + V5-2 subreddit + V5-3 per-rank)

def extract_post_features(
    item_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    base_rates: Dict[str, Dict[str, float]],
    top_t_list: List[int] = TOP_T_LIST,
    rank_n:     int = RANK_N,
) -> pd.DataFrame:
    """Extract post features from the supplied records."""
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

        # raw vote stats
        feats["raw_vote_mean"] = float(rv.mean()) if len(rv) > 0 else 0.0
        feats["raw_vote_std"]  = float(rv.std())  if len(rv) > 1 else 0.0
        feats["n_voters"]      = float(len(rv))
        feats["n_upvoters"]    = float((rv ==  1).sum())
        feats["n_downvoters"]  = float((rv == -1).sum())

        # weighted vote at multiple T
        for t in top_t_list:
            wv = _compute_weighted_votes([item_id], vote_df, weights_df, t)
            feats[f"wv_top{t}"] = wv.get(item_id, feats["raw_vote_mean"])

        min_t = min(top_t_list)
        feats["expert_crowd_divergence"] = feats[f"wv_top{min_t}"] - feats["raw_vote_mean"]

        # V5-2: subreddit context features
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

        # V7-3: downvoter-specific features for the negative predictor
        if not pw.empty:
            pw_neg_all = pw[pw["vote"] == -1]
            reliable_neg = pw_neg_all[
                pw_neg_all.get("reliability", pd.Series(dtype=float)).reindex(pw_neg_all.index, fill_value=0.0) > NEG_RELIABILITY_THR
            ] if "reliability" in pw_neg_all.columns else pw_neg_all
            feats["n_neg_reliable"] = float(len(reliable_neg))
            feats["neg_expert_coverage"] = float(len(reliable_neg) / max(len(pw_neg_all), 1))
            if not reliable_neg.empty:
                top_neg_r = reliable_neg.nlargest(5, "expert_weight")
                w_neg_r   = top_neg_r["expert_weight"].values.clip(min=1e-9)
                feats["wv_neg_only"] = float(
                    np.dot(w_neg_r, top_neg_r[vote_col].values.astype(float)) / w_neg_r.sum()
                )
            else:
                feats["wv_neg_only"] = 0.0
        else:
            feats["n_neg_reliable"]      = 0.0
            feats["neg_expert_coverage"] = 0.0
            feats["wv_neg_only"]         = 0.0

        # placeholder for neg predictor proba - filled later in run_kfold / run_single_split
        feats["neg_expert_proba"] = 0.5

        rows.append(feats)

    return pd.DataFrame(rows)


def get_meta_feature_cols(feat_df: pd.DataFrame) -> List[str]:
    """Return meta feature cols for the supplied input."""
    return [c for c in feat_df.columns if c != "item_id"]


# 6. META-MODEL  (V7-2: model selection - Ridge / XGB / GBT)

class _XGBWrapper:
    """Implement xgbwrapper."""
    def __init__(self, clf):
        """Initialize the instance."""
        self._clf    = clf
        self.classes_ = np.array([-1, 1])

    def predict(self, X):
        """Predict labels for the supplied features."""
        raw = self._clf.predict(X)
        return np.where(raw == 1, 1, -1)

    def predict_proba(self, X):
        """Predict class probabilities for the supplied features."""
        # columns: [proba_class0(=-1), proba_class1(=+1)]
        return self._clf.predict_proba(X)   # already [p0, p1]

    def get_params(self, deep=True):
        """Return estimator parameters for scikit-learn compatibility."""
        return self._clf.get_params(deep=deep)

    @property
    def feature_importances_(self):
        """Return feature weights in estimator order."""
        return self._clf.feature_importances_


def _build_candidates(y: np.ndarray) -> List[Tuple[str, Any]]:
    """Build candidates from the supplied data."""
    candidates = [
        ("Ridge", LogisticRegression(
            penalty="l2", C=0.1, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
        )),
        ("GBT", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=20, random_state=10,
        )),
    ]
    if HAS_XGBOOST:
        pos_count  = int((y == 1).sum())
        neg_count  = int((y == -1).sum())
        scale_pos  = neg_count / max(pos_count, 1)
        candidates.append(("XGB", XGBClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            verbosity=0, random_state=10,
        )))
    return candidates


def train_meta_model(
    feat_df:   pd.DataFrame,
    labels:    Dict[str, int],
    feat_cols: List[str],
) -> Tuple[Any, StandardScaler, str]:
    """Train meta model on the supplied features."""
    items = feat_df[feat_df["item_id"].isin(labels)]["item_id"].tolist()
    X = feat_df.set_index("item_id").loc[items, feat_cols].values.astype(float)
    y = np.array([labels[i] for i in items])

    counts  = {c: (y == c).sum() for c in np.unique(y)}
    max_cnt = max(counts.values())
    sw      = np.array([max_cnt / counts[yi] for yi in y])

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    candidates = _build_candidates(y)
    best_name, best_clf, best_score = None, None, -1.0

    for name, clf in candidates:
        try:
            if name == "Ridge":
                cv_scores = cross_val_score(
                    clf, X_s, y, cv=3, scoring="f1_macro", n_jobs=-1
                )
            else:
                from sklearn.model_selection import StratifiedKFold
                skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
                scores = []
                # XGB needs 0/1 labels
                y_fit = (y == 1).astype(int) if name == "XGB" else y
                for tr, va in skf.split(X_s, y):
                    clf_cv = type(clf)(**clf.get_params())
                    clf_cv.fit(X_s[tr], y_fit[tr], sample_weight=sw[tr])
                    raw_pred = clf_cv.predict(X_s[va])
                    # remap XGB predictions back to -1/+1
                    pred_va = np.where(raw_pred == 1, 1, -1) if name == "XGB" else raw_pred
                    scores.append(f1_score(y[va], pred_va, average="macro", zero_division=0))
                cv_scores = np.array(scores)

            mean_f1 = float(cv_scores.mean())
            log.info("    %s CV macro_f1=%.4f +/- %.4f", name, mean_f1, cv_scores.std())

            if mean_f1 > best_score:
                best_score, best_name, best_clf = mean_f1, name, clf
        except Exception as e:
            log.warning("    %s failed: %s", name, e)

    # fit best on full training data
    if best_name == "Ridge":
        best_clf.fit(X_s, y)
    elif best_name == "XGB":
        y_fit = (y == 1).astype(int)
        best_clf.fit(X_s, y_fit, sample_weight=sw)
        # wrap clf so predict/predict_proba return -1/+1
        best_clf = _XGBWrapper(best_clf)
    else:
        best_clf.fit(X_s, y, sample_weight=sw)

    # feature importances
    actual_clf = best_clf._clf if isinstance(best_clf, _XGBWrapper) else best_clf
    if hasattr(actual_clf, "feature_importances_"):
        top3 = sorted(zip(feat_cols, actual_clf.feature_importances_),
                      key=lambda x: x[1], reverse=True)[:3]
    elif hasattr(actual_clf, "coef_"):
        top3 = sorted(zip(feat_cols, np.abs(actual_clf.coef_[0])),
                      key=lambda x: x[1], reverse=True)[:3]
    else:
        top3 = []

    log.info("  Meta-model winner: %s (CV f1=%.4f) | top: %s",
             best_name, best_score,
             {k: round(v, 3) for k, v in top3})
    return best_clf, scaler, best_name


# 6b. NEGATIVE PREDICTOR  (V7-1)

def train_neg_predictor(
    feat_df:   pd.DataFrame,
    labels:    Dict[str, int],
    feat_cols: List[str],
) -> Tuple[Optional[Any], Optional[StandardScaler]]:
    """Train neg predictor on the supplied features."""
    neg_feat_cols = [c for c in feat_cols if
                     any(tag in c for tag in ["neg", "wv_neg", "n_neg", "neg_expert"])]
    if not neg_feat_cols:
        return None, None

    df = feat_df[feat_df["item_id"].isin(labels)].copy()
    df["_label"] = df["item_id"].map(labels)

    # keep only posts with at least one reliable downvoter expert
    mask = df["n_neg_reliable"] > 0 if "n_neg_reliable" in df.columns else pd.Series(True, index=df.index)
    df_neg = df[mask]

    if len(df_neg) < 50 or df_neg["_label"].nunique() < 2:
        log.info("  Neg predictor: insufficient data (%d posts) - skipped.", len(df_neg))
        return None, None

    X = df_neg[neg_feat_cols].values.astype(float)
    y = df_neg["_label"].values.astype(int)

    counts  = {c: (y == c).sum() for c in np.unique(y)}
    max_cnt = max(counts.values())
    sw      = np.array([max_cnt / counts[yi] for yi in y])

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced",
        max_iter=1000, solver="lbfgs",
    )
    clf.fit(X_s, y)
    log.info("  Neg predictor: trained on %d posts (%d remove, %d approve)",
             len(y), (y==-1).sum(), (y==1).sum())
    return clf, scaler, neg_feat_cols


def apply_neg_predictor(
    feat_df:        pd.DataFrame,
    item_ids:       List[str],
    neg_clf:        Optional[Any],
    neg_scaler:     Optional[StandardScaler],
    neg_feat_cols:  Optional[List[str]],
) -> Dict[str, float]:
    """Apply neg predictor to the supplied data."""
    result = {iid: 0.5 for iid in item_ids}
    if neg_clf is None or neg_feat_cols is None:
        return result

    df = feat_df.set_index("item_id")
    # only predict for posts with at least one reliable neg expert
    eligible = [iid for iid in item_ids
                if iid in df.index and df.loc[iid, "n_neg_reliable"] > 0]
    if not eligible:
        return result

    X   = df.loc[eligible, neg_feat_cols].values.astype(float)
    X_s = neg_scaler.transform(X)
    # proba of class -1 (remove)
    neg_class_idx = list(neg_clf.classes_).index(-1)
    probas        = neg_clf.predict_proba(X_s)[:, neg_class_idx]
    for iid, p in zip(eligible, probas):
        result[iid] = float(p)
    return result


def predict_meta_model(
    feat_df:   pd.DataFrame,
    item_ids:  List[str],
    clf:       Any,
    scaler:    StandardScaler,
    feat_cols: List[str],
) -> Dict[str, Tuple[int, float]]:
    """Predict item labels with the fitted meta-model."""
    df = feat_df.set_index("item_id")
    pos_class_idx = list(clf.classes_).index(1)

    known  = [iid for iid in item_ids if iid in df.index]
    result: Dict[str, Tuple[int, float]] = {iid: (1, 0.5) for iid in item_ids}
    if not known:
        return result

    X      = df.loc[known, feat_cols].values.astype(float)
    X_s    = scaler.transform(X)
    preds  = clf.predict(X_s)
    probas = clf.predict_proba(X_s)[:, pos_class_idx]
    for iid, pred, proba in zip(known, preds, probas):
        result[iid] = (int(pred), float(proba))
    return result


# 7. WEIGHTED VOTE

def _compute_weighted_votes(
    item_ids:   List[str],
    vote_df:    pd.DataFrame,
    weights_df: pd.DataFrame,
    top_t:      int,
) -> Dict[str, float]:
    """Compute weighted votes from the supplied data."""
    if weights_df.empty:
        raw = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "vote"]]
        return {iid: float(g["vote"].mean()) for iid, g in raw.groupby("item_id")}

    vote_col = "effective_vote" if "effective_vote" in weights_df.columns else "vote"
    w = weights_df[weights_df["item_id"].isin(item_ids)][
        ["item_id", "expert_weight", vote_col]
    ].copy()

    # raw vote fallback for posts not in weights_df
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "vote"]]
    raw_mean  = raw_votes.groupby("item_id")["vote"].mean().to_dict()

    if w.empty:
        return {iid: raw_mean.get(iid, 0.0) for iid in item_ids}

    # keep top-T per post by expert_weight, then compute weighted mean
    w_sorted  = w.sort_values("expert_weight", ascending=False)
    w_top     = w_sorted.groupby("item_id", sort=False).head(top_t)
    w_top     = w_top.copy()
    w_top["expert_weight"] = w_top["expert_weight"].clip(lower=1e-9)
    w_top["wv_num"] = w_top["expert_weight"] * w_top[vote_col]

    agg = w_top.groupby("item_id").agg(
        num=("wv_num", "sum"),
        den=("expert_weight", "sum"),
    )
    agg["wv"] = agg["num"] / agg["den"]
    result = agg["wv"].to_dict()

    # fill posts missing from weights_df
    for iid in item_ids:
        if iid not in result:
            result[iid] = raw_mean.get(iid, 0.0)
    return result


# 8. THRESHOLD CALIBRATION

def calibrate_threshold_weighted(
    sample_ids:  List[str],
    vote_df:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    top_t:       int,
    neg_weight:  float = 2.0,
) -> float:
    """Calibrate threshold weighted on validation data."""
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
    """Calibrate thresholds per subreddit on validation data."""
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


# 9. GRID SEARCH  (I3+I4: extended T grid, pure-C combo covered by (0,0))

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
    """Select expert-weighting parameters on validation data."""
    labels  = (
        vote_df[vote_df["item_id"].isin(val_ids)]
        .drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
    )
    val_ids = [iid for iid in val_ids if iid in labels]
    if not val_ids:
        return K_NEIGHBORS, TOP_T, ALPHA, BETA

    valid_ab = [(a, b) for a in GRID_ALPHA for b in GRID_BETA if a + b <= 1.0]
    n_combos = len(GRID_K) * len(valid_ab) * len(GRID_T)
    log.info("  Grid search: %d combos on %d val posts ...", n_combos, len(val_ids))

    best = {"f1": -1.0, "k": K_NEIGHBORS, "t": TOP_T, "alpha": ALPHA, "beta": BETA}

    for k in GRID_K:
        # precompute weights once per (k, alpha, beta) - T is free to vary after
        for alpha, beta in valid_ab:
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
                y_score = np.array([wv[iid]     for iid in common])

                best_f1_t = -1.0
                for thr in np.linspace(y_score.min(), y_score.max(), 50):
                    y_pred = np.where(y_score >= thr, 1, -1)
                    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                    if f1 > best_f1_t:
                        best_f1_t = f1

                if best_f1_t > best["f1"]:
                    best = {"f1": best_f1_t, "k": k, "t": t, "alpha": alpha, "beta": beta}

    log.info("  Best: k=%d t=%d alpha=%.2f beta=%.2f macro_f1=%.4f",
             best["k"], best["t"], best["alpha"], best["beta"], best["f1"])
    return best["k"], best["t"], best["alpha"], best["beta"]


# 10. PREDICTION

def predict_fold(
    test_ids:       List[str],
    vote_df:        pd.DataFrame,
    weights_df:     pd.DataFrame,
    feat_df:        pd.DataFrame,
    feat_cols:      List[str],
    clf:            Any,
    scaler:         StandardScaler,
    thresholds:     Dict[Optional[str], float],
    top_t:          int,
    neg_clf:        Optional[Any]           = None,
    neg_scaler:     Optional[StandardScaler] = None,
    neg_feat_cols:  Optional[List[str]]     = None,
) -> pd.DataFrame:
    """Generate predictions for one evaluation fold."""
    labels = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id")[["item_id", "label"]]
    )
    post_sub = (
        vote_df[vote_df["item_id"].isin(test_ids)]
        .drop_duplicates("item_id").set_index("item_id")["community"].to_dict()
    )

    # V7-1: compute neg predictor probabilities and inject into feat_df
    neg_probas = apply_neg_predictor(feat_df, test_ids, neg_clf, neg_scaler, neg_feat_cols)
    feat_df = feat_df.copy()
    feat_df["neg_expert_proba"] = feat_df["item_id"].map(neg_probas).fillna(0.5)

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
            "item_id":        item_id,
            "weighted_vote":  score,
            "meta_proba":     meta_proba if meta_proba is not None else 0.5,
            "neg_expert_proba": neg_probas.get(item_id, 0.5),
            "predicted":      predicted,
            "used_fallback":  fallback,
            "subreddit":      sub,
        })

    pred_df = pd.DataFrame(rows)
    result  = pd.merge(labels, pred_df, on="item_id", how="left")
    result["predicted"] = result["predicted"].fillna(1).astype(int)
    return result


# 11. METRICS

def evaluate_fold(decisions: pd.DataFrame) -> Dict:
    """Evaluate fold and return its metrics."""
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


# 12. SINGLE-SPLIT EVALUATION  (core logic, reused by K-fold and multi-split)

def run_single_split(
    train_ids:      set,
    val_ids:        Optional[set],
    test_ids:       set,
    vote_df:        pd.DataFrame,
    post_emb:       np.ndarray,
    post_id2idx:    Dict[str, int],
    user_emb:       np.ndarray,
    user_id2idx:    Dict[str, int],
    faiss_index:    faiss.Index,
    default_k:      int,
    default_t:      int,
    default_alpha:  float,
    default_beta:   float,
    min_coverage:   int,
    do_grid_search: bool = True,
    split_label:    str = "",
) -> Tuple[Optional[Dict], pd.DataFrame, pd.DataFrame]:
    """Run the single split workflow."""
    train_vote_df = vote_df[vote_df["item_id"].isin(train_ids)]
    prec_df       = precompute_vote_precision(train_vote_df, LAMBDA_SMOOTH, min_coverage)
    base_rates    = compute_subreddit_base_rates(train_vote_df)
    log.info("  [%s] train=%d val=%d test=%d | reliable users (>=%d votes): %d",
             split_label, len(train_ids), len(val_ids or []), len(test_ids),
             min_coverage, prec_df["username"].nunique())

    # hyperparameter selection: use real val if we have one, else carve from train
    if val_ids:
        val_grid_ids = list(val_ids)
        train_grid   = train_ids
    else:
        train_list   = list(train_ids)
        np.random.shuffle(train_list)
        val_grid_ids = train_list[:min(VAL_GRID_SAMPLE, len(train_list))]
        train_grid   = set(train_list[min(VAL_GRID_SAMPLE, len(train_list)):])

    if do_grid_search and val_grid_ids:
        k, t, alpha, beta = grid_search_hyperparams(
            val_grid_ids, train_grid, vote_df, prec_df,
            post_emb, post_id2idx, user_emb, user_id2idx,
            faiss_index, LAMBDA_SMOOTH,
        )
    else:
        k, t, alpha, beta = default_k, default_t, default_alpha, default_beta

    # expert weights for the TEST set (weighted using TRAIN-only precision)
    test_id_list = list(test_ids)
    weights_df = compute_expert_weights(
        test_id_list, train_ids, vote_df, prec_df,
        post_emb, post_id2idx, user_emb, user_id2idx,
        faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
    )

    # training sample for the meta-model (subsampled from TRAIN only)
    train_sample = list(train_ids)
    np.random.shuffle(train_sample)
    train_sample = train_sample[:min(CAL_SAMPLE, len(train_sample))]

    weights_train_sample = compute_expert_weights(
        train_sample, train_ids - set(train_sample), vote_df, prec_df,
        post_emb, post_id2idx, user_emb, user_id2idx,
        faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
    )

    # threshold calibration: on real VAL if available, else on train_sample
    if val_ids:
        cal_ids     = list(val_ids)
        weights_cal = compute_expert_weights(
            cal_ids, train_ids, vote_df, prec_df,
            post_emb, post_id2idx, user_emb, user_id2idx,
            faiss_index, k, alpha, beta, LAMBDA_SMOOTH,
        )
    else:
        cal_ids     = train_sample
        weights_cal = weights_train_sample
    thresholds = calibrate_thresholds_per_subreddit(cal_ids, vote_df, weights_cal, t)

    log.info("  [%s] Extracting features for meta-model ...", split_label)
    train_feat_df = extract_post_features(train_sample, vote_df, weights_train_sample, base_rates)
    feat_cols     = get_meta_feature_cols(train_feat_df)
    train_labels  = (
        vote_df[vote_df["item_id"].isin(train_sample)]
        .drop_duplicates("item_id").set_index("item_id")["label"].to_dict()
    )

    # V7-1: train negative predictor on training sample only
    neg_result = train_neg_predictor(train_feat_df, train_labels, feat_cols)
    if len(neg_result) == 3:
        neg_clf, neg_scaler, neg_feat_cols = neg_result
    else:
        neg_clf, neg_scaler, neg_feat_cols = None, None, None

    if neg_clf is not None:
        neg_probas_train = apply_neg_predictor(
            train_feat_df, train_sample, neg_clf, neg_scaler, neg_feat_cols
        )
        train_feat_df = train_feat_df.copy()
        train_feat_df["neg_expert_proba"] = \
            train_feat_df["item_id"].map(neg_probas_train).fillna(0.5)
        feat_cols = get_meta_feature_cols(train_feat_df)

    # V7-2: model selection, trained only on TRAIN
    clf, scaler, model_name = train_meta_model(train_feat_df, train_labels, feat_cols)

    test_feat_df = extract_post_features(test_id_list, vote_df, weights_df, base_rates)

    decisions = predict_fold(
        test_id_list, vote_df, weights_df,
        test_feat_df, feat_cols, clf, scaler, thresholds, t,
        neg_clf=neg_clf, neg_scaler=neg_scaler, neg_feat_cols=neg_feat_cols,
    )
    decisions["split"] = split_label
    decisions["k"]     = k
    decisions["top_t"] = t
    decisions["alpha"] = alpha
    decisions["beta"]  = beta

    if decisions["label"].nunique() < 2:
        log.warning("  [%s] single class in test - skipped.", split_label)
        return None, decisions, weights_df

    metrics = evaluate_fold(decisions)
    metrics.update({
        "split":             split_label,
        "k":                 k,
        "top_t":             t,
        "alpha":             alpha,
        "beta":              beta,
        "gamma":             1.0 - alpha - beta,
        "threshold":         thresholds.get(None, 0.0),
        "n_train_items":     len(train_ids),
        "n_val_items":       len(val_ids) if val_ids else 0,
        "n_test":            len(decisions),
        "fallback_pct":      float(decisions["used_fallback"].mean()),
        "model_name":        model_name,
        "has_neg_predictor": int(neg_clf is not None),
    })

    log.info(
        "  [%s] macro_f1=%.4f  roc_auc=%.4f  f1_pos=%.4f  f1_neg=%.4f  "
        "model=%s  neg_pred=%s  alpha=%.2f  beta=%.2f  gamma=%.2f",
        split_label, metrics["macro_f1"], metrics["roc_auc"],
        metrics["f1_pos"], metrics["f1_neg"],
        model_name, "yes" if neg_clf is not None else "no",
        alpha, beta, 1.0 - alpha - beta,
    )

    weights_df = weights_df.copy()
    weights_df["split"] = split_label
    return metrics, decisions, weights_df


# 12b. KFOLD  (legacy: random split, val carved out of train per fold)

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
) -> Tuple[List[Dict], List[pd.DataFrame], List[pd.DataFrame]]:
    """Run the kfold workflow."""
    global post_ids_ordered
    post_ids_ordered = post_id_list

    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}
    user_id2idx = {uid: i for i, uid in enumerate(user_id_list)}

    labeled_items = (
        vote_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"]).reset_index(drop=True)
    )
    log.info("Labeled items: %d", len(labeled_items))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=10)
    fold_metrics:     List[Dict]         = []
    fold_item_scores: List[pd.DataFrame] = []
    fold_weights:     List[pd.DataFrame] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(labeled_items)):
        log.info("--- Fold %d / %d ---", fold_idx + 1, N_FOLDS)

        train_ids = set(labeled_items.iloc[train_val_idx]["item_id"].tolist())
        test_ids  = set(labeled_items.iloc[test_idx]["item_id"].tolist())

        metrics, decisions, weights_df = run_single_split(
            train_ids, None, test_ids,
            vote_df, post_emb, post_id2idx, user_emb, user_id2idx, faiss_index,
            default_k, default_t, default_alpha, default_beta,
            min_coverage, do_grid_search, split_label=f"fold{fold_idx}",
        )
        if metrics is None:
            continue

        metrics["fold"]    = fold_idx
        decisions["fold"]  = fold_idx
        weights_df["fold"] = fold_idx

        fold_metrics.append(metrics)
        fold_item_scores.append(decisions)
        fold_weights.append(weights_df)

    return fold_metrics, fold_item_scores, fold_weights


# 12c. MULTI-SPLIT BENCHMARK  (uses splits produced by prepare_data_step1)

def evaluate_splits(
    votes_dir:      Path,
    output_dir:     Path,
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
) -> Dict[str, Dict]:
    """Evaluate splits and return its metrics."""
    global post_ids_ordered
    post_ids_ordered = post_id_list
    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}
    user_id2idx = {uid: i for i, uid in enumerate(user_id_list)}

    splits = discover_splits(votes_dir)
    if not splits:
        log.warning("No split directories found under %s "
                    "(expected splits/, splits_full/, splits_intersection/, "
                    "windowed_folds_full/w*, windowed_folds_intersection/w*)", votes_dir)
        return {}

    log.info("Found %d split(s): %s", len(splits), list(splits.keys()))
    summary: Dict[str, Dict] = {}

    for split_name, split_path in splits.items():
        log.info("--- split: %s ---", split_name)

        train_votes = _load_split_votes(split_path / "train_votes.parquet")
        val_votes   = _load_split_votes(split_path / "val_votes.parquet")
        test_votes  = _load_split_votes(split_path / "test_votes.parquet")

        # this split's own votes only - never mixes in votes from outside it
        split_vote_df = pd.concat([train_votes, val_votes, test_votes], ignore_index=True)

        train_ids = set(train_votes["item_id"].unique())
        val_ids   = set(val_votes["item_id"].unique())
        test_ids  = set(test_votes["item_id"].unique())

        if not test_ids:
            log.warning("  [%s] empty test set - skipped.", split_name)
            continue

        metrics, decisions, weights_df = run_single_split(
            train_ids, val_ids, test_ids,
            split_vote_df, post_emb, post_id2idx, user_emb, user_id2idx, faiss_index,
            default_k, default_t, default_alpha, default_beta,
            min_coverage, do_grid_search, split_label=split_name,
        )
        if metrics is None:
            continue

        # "splits_full" only: for items where compute_expert_weights found no
        # expert (used_fallback=True), weighted_vote already equals the raw
        # net vote (see _compute_weighted_votes' fallback branch). Force the
        # final decision for those items to follow that net vote directly,
        # instead of the meta-model's prediction on a degenerate/zero-padded
        # feature vector - this measures how much performance degrades when
        # every item must get a decision, using the simplest possible signal
        # for items with no real expert coverage.
        if split_name == "splits_full" and decisions["used_fallback"].any():
            fb = decisions["used_fallback"]
            n_fallback = int(fb.sum())
            decisions = decisions.copy()
            decisions.loc[fb, "predicted"]   = np.where(decisions.loc[fb, "weighted_vote"] >= 0, 1, -1)
            decisions.loc[fb, "meta_proba"]  = np.clip((decisions.loc[fb, "weighted_vote"] + 1) / 2, 0.0, 1.0)
            metrics.update(evaluate_fold(decisions))  # overwrite macro_f1/roc_auc/etc, keep k/alpha/model_name/...
            metrics.update({
                "split": split_name, "n_test": len(decisions),
                "fallback_pct": float(decisions["used_fallback"].mean()),
                "used_net_vote_fallback": True,
                "n_fallback_items": n_fallback,
            })
            log.info("  [%s] net-vote fallback forced on %d/%d items -> "
                     "macro_f1=%.4f  roc_auc=%.4f",
                     split_name, n_fallback, len(decisions),
                     metrics["macro_f1"], metrics["roc_auc"])

        split_out_dir = output_dir / split_name / "expertise"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        decisions.to_parquet(split_out_dir / "item_scores.parquet", index=False)
        weights_df.to_parquet(split_out_dir / "weights.parquet", index=False)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("  [%s] Saved -> %s", split_name, split_out_dir)

        summary[split_name] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_splits_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary of all splits saved -> %s", summary_path)

    if summary:
        log.info("=== Cross-split comparison ===")
        for name, m in summary.items():
            log.info("  %-35s macro_F1=%.4f  AUC=%.4f  n_test=%d  model=%s",
                     name, m["macro_f1"], m["roc_auc"], m["n_test"], m["model_name"])

    return summary


# 13. OUTPUT

def save_outputs(
    fold_metrics:     List[Dict],
    fold_item_scores: List[pd.DataFrame],
    fold_weights:     List[pd.DataFrame],
    output_dir:       Path,
) -> None:
    """Write outputs to disk."""
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

    agg["best_k_per_fold"]     = [m.get("k")          for m in fold_metrics]
    agg["best_t_per_fold"]     = [m.get("top_t")       for m in fold_metrics]
    agg["best_alpha_per_fold"] = [m.get("alpha")       for m in fold_metrics]
    agg["best_beta_per_fold"]  = [m.get("beta")        for m in fold_metrics]
    agg["best_gamma_per_fold"] = [m.get("gamma")       for m in fold_metrics]
    agg["model_per_fold"]      = [m.get("model_name")  for m in fold_metrics]
    agg["neg_pred_per_fold"]   = [m.get("has_neg_predictor") for m in fold_metrics]
    agg["fallback_pct"]        = float(np.mean([m.get("fallback_pct", 0) for m in fold_metrics]))

    # model selection summary
    from collections import Counter
    model_counts = Counter(agg["model_per_fold"])
    agg["model_wins"] = dict(model_counts)
    log.info("  Model wins: %s", model_counts)

    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)

    log.info(
        "Saved to %s | macro_f1=%.4f+/-%.4f  roc_auc=%.4f+/-%.4f  f1_pos=%.4f  f1_neg=%.4f",
        output_dir,
        agg["macro_f1"], agg["macro_f1_std"],
        agg["roc_auc"],  agg["roc_auc_std"],
        agg["f1_pos"],   agg["f1_neg"],
    )


# ENTRY POINT

def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser(description="Semantic Expert Finder v7")
    parser.add_argument("--k-neighbors",    type=int,   default=K_NEIGHBORS)
    parser.add_argument("--top-t",          type=int,   default=TOP_T)
    parser.add_argument("--alpha",          type=float, default=ALPHA)
    parser.add_argument("--beta",           type=float, default=BETA)
    parser.add_argument("--min-user-votes", type=int,   default=MIN_USER_VOTES)
    parser.add_argument("--output-dir",     type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--no-grid-search", action="store_true")
    parser.add_argument("--input-csv",      type=Path,  default=None)
    parser.add_argument("--votes-dir",      type=Path,  default=SPLITS_DIR,
                         help="Directory with the splits produced by prepare_data_step1 "
                              "(splits/, splits_full/, splits_intersection/, windowed_folds_*)")
    parser.add_argument("--tasks", nargs="+", default=["kfold", "splits"],
                         choices=["kfold", "splits"],
                         help="kfold = legacy random K-fold benchmark; "
                              "splits = benchmark on every prepared split")
    parser.add_argument("--skip-tasks", nargs="+", default=[],
                         choices=["kfold", "splits"])
    args = parser.parse_args()

    assert args.alpha + args.beta <= 1.0, \
        f"alpha + beta must be <= 1.0 (got {args.alpha + args.beta:.2f})"

    tasks_to_run = [task for task in args.tasks if task not in args.skip_tasks]

    embed_dir = CACHE_DIR

    log.info("=== SEF v7 | alpha=%.2f beta=%.2f gamma=%.2f | steps=%s ===",
             args.alpha, args.beta, 1.0 - args.alpha - args.beta, tasks_to_run)

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

    if "kfold" in tasks_to_run:
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
        save_outputs(fold_metrics, fold_item_scores, fold_weights,
                     args.output_dir / "kfold" / "expertise")

    if "splits" in tasks_to_run:
        evaluate_splits(
            votes_dir      = args.votes_dir,
            output_dir     = args.output_dir,
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

    log.info("Done.")


if __name__ == "__main__":
    main()
