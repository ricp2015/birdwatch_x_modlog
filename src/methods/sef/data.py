"""Vote, text, embedding-cache, and FAISS data operations for SEF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from .runtime import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FETCH_DIR,
    ORIGINAL_CSV,
    SPLITS_DIR,
    STEP1_DIR,
    log,
)

# 1. DATA LOADING


def load_votes(input_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load votes from its configured source."""
    for csv_path in [p for p in [input_csv, ORIGINAL_CSV] if p is not None]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            df["vote"] = pd.to_numeric(df["vote"], errors="coerce")
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])].copy()
            df["vote"] = df["vote"].astype(int)
            df["label"] = df["label"].astype(int)
            df = df.dropna(subset=["item_id", "username", "vote", "label"])
            df = df.drop_duplicates(subset=["username", "item_id"])
            log.info(
                "Votes: %d rows, %d posts, %d users, %d communities",
                len(df),
                df["item_id"].nunique(),
                df["username"].nunique(),
                df["community"].nunique(),
            )
            return df

    log.warning("CSV not found; falling back to Parquet files.")
    path = STEP1_DIR / "filtered_votes.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        dfs = [
            pd.read_parquet(SPLITS_DIR / "intersection" / f"{s}_votes.parquet")
            for s in ("train", "val", "test")
            if (SPLITS_DIR / "intersection" / f"{s}_votes.parquet").exists()
        ]
        df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"] = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    log.info(
        "Votes: %d rows, %d posts, %d users",
        len(df),
        df["item_id"].nunique(),
        df["username"].nunique(),
    )
    return df


def _load_split_votes(path: Path) -> pd.DataFrame:
    """Load split votes from its configured source."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"] = df["vote"].astype(int)
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
    log.info("Post texts available: %d of %d", len(df), before)
    return df


def load_user_documents() -> pd.DataFrame:
    """Load user documents from its configured source."""
    path = FETCH_DIR / "user_documents.parquet"
    if not path.exists():
        log.warning("User documents not found; Signal C will use the neutral value 0.5.")
        return pd.DataFrame(columns=["username", "text"])
    df = pd.read_parquet(path)
    df = df[df["text"].notna()].copy()
    log.info("User documents: %d documents, %d users", len(df), df["username"].nunique())
    return df


# 2. EMBEDDINGS


def _get_model() -> str:
    """Return model for the supplied input."""
    return EMBEDDING_MODEL


def _encode_texts(
    texts: List[str],
    model_name: str,
    normalize: bool = True,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode text batches without relying on an external utility script."""
    from sentence_transformers import SentenceTransformer

    log.info("Encoding %d texts", len(texts))
    encoder = SentenceTransformer(model_name)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)


def build_post_embeddings(
    post_texts: pd.DataFrame,
    embed_dir: Path,
    model: Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Build post embeddings from the supplied data."""
    emb_path = embed_dir / "post_embeddings.npy"
    ids_path = embed_dir / "post_ids.json"
    partial_dir = embed_dir / "post_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached post embeddings")
        return np.load(emb_path), json.load(open(ids_path))

    model_name = model if model is not None else _get_model()
    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(exist_ok=True)

    df = post_texts.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    all_texts = df["text"].tolist()
    all_item_ids = df["item_id"].tolist()
    n_total = len(all_texts)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    done = len(sorted(partial_dir.glob("chunk_*.npy")))

    for ci in range(done, n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=True)
        np.save(partial_dir / f"chunk_{ci:05d}.npy", vecs)

    chunks = [np.load(partial_dir / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(emb_path, embeddings)
    json.dump(all_item_ids, open(ids_path, "w"))
    log.info("Post embeddings: %s", embeddings.shape)
    return embeddings, all_item_ids


def build_user_embeddings(
    user_docs: pd.DataFrame,
    embed_dir: Path,
    model: Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Build user embeddings from the supplied data."""
    emb_path = embed_dir / "user_embeddings.npy"
    ids_path = embed_dir / "user_ids.json"
    partial_dir = embed_dir / "user_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached user embeddings")
        return np.load(emb_path), json.load(open(ids_path))

    if user_docs.empty:
        log.warning("No user documents found; user embeddings will be empty.")
        embed_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        json.dump([], open(ids_path, "w"))
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    model_name = model if model is not None else _get_model()
    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(exist_ok=True)

    user_docs = user_docs.reset_index(drop=True)
    all_texts = user_docs["text"].tolist()
    n_total = len(all_texts)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    done = len(sorted(partial_dir.glob("udoc_chunk_*.npy")))

    for ci in range(done, n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=False)
        np.save(partial_dir / f"udoc_chunk_{ci:05d}.npy", vecs)

    chunks = [np.load(partial_dir / f"udoc_chunk_{i:05d}.npy") for i in range(n_chunks)]
    doc_embeddings = np.concatenate(chunks, axis=0).astype(np.float32)

    user_docs["_emb_idx"] = np.arange(len(user_docs))
    usernames, user_vecs = [], []
    for uname, group in tqdm(user_docs.groupby("username"), desc="Averaging user profiles"):
        mean_vec = doc_embeddings[group["_emb_idx"].values].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
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


def build_faiss_index(embeddings: np.ndarray, *, announce: bool = False) -> faiss.Index:
    """Build faiss index from the supplied data."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if announce:
        log.info("FAISS index ready: %d vectors", index.ntotal)
    return index
