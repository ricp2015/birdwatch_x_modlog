"""NormVio model loading and post-category scoring for NVSE."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Dict, List
import unicodedata

from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.splits import discover_splits  # noqa: E402
from src.utils.tabular import read_table  # noqa: E402

# Constants
BERT_TYPE = "DeepPavlov/bert-base-cased-conversational"
HIDDEN_SIZE = 768
MAX_LENGTH = 120
DROPOUT = 0.1
NUM_CLASSES = 2

CATEGORIES = [
    "spam",
    "meta-rules",
    "content",
    "harassment",
    "hatespeech",
    "format",
    "off-topic",
    "trolling",
    "incivility",
]

THRESHOLD_GRID = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5  # min votes per userxcommunityxcategory to compute skill


# Model definitions from NormVio


class EncoderBERT(nn.Module):
    """Implement encoder bert."""

    def __init__(self, device: torch.device, model_source: str | Path | None = None):
        """Initialize the instance."""
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained(str(model_source or resolve_bert_source())).to(
            device
        )

    def forward(self, input_seq, input_lengths):
        """Run a forward pass through the model."""
        input_seq = input_seq.T  # Convert sequence-first to batch-first.
        mask_ids = (input_seq != 0).long()
        token_ids = torch.ones_like(input_seq)
        return self.model(input_ids=input_seq, attention_mask=mask_ids, token_type_ids=token_ids)[
            1
        ]  # pooled CLS


class SingleTargetClf(nn.Module):
    """Implement single target clf."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float, device: torch.device):
        """Initialize the instance."""
        super().__init__()
        self.device = device
        self.layer1 = nn.Linear(hidden_size, hidden_size).to(device)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, num_classes).to(device)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, ctx_out, dialog_lengths):
        """Run a forward pass through the model."""
        # ctx_out: [1, batch, hidden]
        lengths = (
            dialog_lengths.unsqueeze(0).unsqueeze(2).expand(1, -1, ctx_out.size(2)).to(self.device)
        )
        last = torch.gather(ctx_out, 0, lengths - 1).squeeze(0)  # [batch, hidden]
        if last.dim() == 1:
            last = last.unsqueeze(0)
        h = self.layer2_act(self.layer2(self.drop(self.layer1_act(self.layer1(self.drop(last))))))
        return self.clf(self.drop(h))


class Predictor(nn.Module):
    """Implement predictor."""

    def __init__(self, encoder: EncoderBERT, clf: SingleTargetClf):
        """Initialize the instance."""
        super().__init__()
        self.encoder = encoder
        self.clf = clf

    def forward(self, input_batch, dialog_lengths, utt_lengths):
        """Run a forward pass through the model."""
        utt_hidden = self.encoder(input_batch, utt_lengths)  # [batch, hidden]
        ctx_out = utt_hidden.unsqueeze(0)  # [1, batch, hidden]
        return self.clf(ctx_out, dialog_lengths)


# Utilities


def get_device(device_str: str) -> torch.device:
    """Return device for the supplied input."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unicode_to_ascii(s: str) -> str:
    """Normalize text to ASCII characters."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def resolve_bert_source(model_id: str = BERT_TYPE) -> Path | str:
    """Prefer the exact locally cached Hugging Face snapshot when available."""
    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except FileNotFoundError:
        return model_id


def load_model(
    model_dir: Path,
    device: torch.device,
    model_source: str | Path | None = None,
) -> Predictor:
    """Load model from its configured source."""
    pt_path = model_dir / "finetuned_model.pt"
    # Legacy checkpoints use pickle serialization.
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    enc = EncoderBERT(device, model_source=model_source)
    enc.load_state_dict(ckpt["en"])
    enc.eval()
    clf = SingleTargetClf(HIDDEN_SIZE, NUM_CLASSES, DROPOUT, device)
    clf.load_state_dict(ckpt["atk_clf"])
    clf.eval()
    return Predictor(enc, clf).to(device)


def _clean_text_chunk(chunk: List[str]) -> List[str]:
    """Strip and NFD-normalize one chunk of texts (top-level for multiprocessing)."""
    return [unicode_to_ascii(str(t).strip()) if t else "" for t in chunk]


def _clean_texts(texts: List[str]) -> List[str]:
    """NFD-normalize every text, spreading the work across CPU cores at scale."""
    n = len(texts)
    if n < 50_000:
        return _clean_text_chunk(texts)
    workers = min(32, os.cpu_count() or 1)
    chunk_size = max(1, math.ceil(n / (workers * 4)))
    chunks = [texts[i : i + chunk_size] for i in range(0, n, chunk_size)]
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            cleaned_chunks = list(pool.map(_clean_text_chunk, chunks))
        return [text for chunk in cleaned_chunks for text in chunk]
    except Exception as exc:
        print(f"Parallel text cleaning failed ({exc}); falling back to sequential", flush=True)
        return _clean_text_chunk(texts)


def _tokenize_all(
    cleaned_texts: List[str],
    tokenizer: BertTokenizer,
    max_length: int,
    tokenize_chunk: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize every text exactly once, shared across every category's model."""
    n = len(cleaned_texts)
    valid_mask = np.fromiter((bool(t) for t in cleaned_texts), dtype=bool, count=n)
    lengths = np.zeros(n, dtype=np.int32)
    encoded: List[List[int] | None] = [None] * n
    valid_positions = np.nonzero(valid_mask)[0]
    for start in range(0, len(valid_positions), tokenize_chunk):
        pos_chunk = valid_positions[start : start + tokenize_chunk]
        text_chunk = [cleaned_texts[i] for i in pos_chunk]
        batch = tokenizer(
            text_chunk, add_special_tokens=True, truncation=True, max_length=max_length
        )
        for pos, ids in zip(pos_chunk, batch["input_ids"]):
            encoded[pos] = ids
            lengths[pos] = len(ids)
    max_len = int(lengths.max()) if len(valid_positions) else 1
    token_ids = np.zeros((n, max(max_len, 1)), dtype=np.int32)
    for pos in valid_positions:
        ids = encoded[pos]
        token_ids[pos, : len(ids)] = ids
    return token_ids, lengths, valid_mask


def _visible_devices(device: torch.device) -> List[torch.device]:
    """Fan a bare 'cuda' device out across every visible GPU; respect an explicit index."""
    if device.type != "cuda" or device.index is not None:
        return [device]
    if not torch.cuda.is_available():
        return [device]
    count = torch.cuda.device_count()
    if count <= 1:
        return [device]
    return [torch.device(f"cuda:{i}") for i in range(count)]


def _shard_categories(
    categories: List[str], devices: List[torch.device]
) -> Dict[torch.device, List[str]]:
    """Round-robin distribute independent category models across devices."""
    shards: Dict[torch.device, List[str]] = {d: [] for d in devices}
    for i, cat in enumerate(categories):
        shards[devices[i % len(devices)]].append(cat)
    return shards


_OOM_ERRORS: tuple[type[Exception], ...] = (
    (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)
)


@torch.no_grad()
def _forward_probs_with_oom_backoff(
    predictor: Predictor,
    inp: torch.Tensor,
    d_lens_full: torch.Tensor,
    lengths_full: torch.Tensor,
    idx: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Score one batch slice, halving it and retrying on a CUDA out-of-memory error."""
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=inp.device)
    try:
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = predictor(inp[:, idx_t], d_lens_full[idx_t], lengths_full[idx_t])
        else:
            logits = predictor(inp[:, idx_t], d_lens_full[idx_t], lengths_full[idx_t])
        return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    except _OOM_ERRORS as exc:
        if _OOM_ERRORS == (RuntimeError,) and "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(idx) == 1:
            raise
        mid = len(idx) // 2
        left = _forward_probs_with_oom_backoff(
            predictor, inp, d_lens_full, lengths_full, idx[:mid], device, use_amp
        )
        right = _forward_probs_with_oom_backoff(
            predictor, inp, d_lens_full, lengths_full, idx[mid:], device, use_amp
        )
        return np.concatenate([left, right])


def _length_sorted_batches(
    lengths: np.ndarray, valid_mask: np.ndarray, batch_size: int
) -> List[np.ndarray]:
    """Group valid texts into batches sorted by token length to minimize padding waste."""
    valid_idx = np.nonzero(valid_mask)[0]
    if len(valid_idx) == 0:
        return []
    order = valid_idx[np.argsort(lengths[valid_idx], kind="stable")]
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def _score_categories_worker(
    device_str: str,
    categories: List[str],
    models_dir: Path,
    model_source: str | Path | None,
    token_ids: np.ndarray,
    lengths: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Score every text against a shard of category models resident on one device."""
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    n = len(lengths)
    scores: Dict[str, np.ndarray] = {cat: np.full(n, np.nan) for cat in categories}
    predictors: Dict[str, Predictor] = {}
    for cat in categories:
        try:
            predictors[cat] = load_model(models_dir / cat, device, model_source=model_source)
        except Exception as exc:
            import traceback

            print(f"Failed to load {cat} on {device}: {exc}")
            traceback.print_exc()
    if not predictors:
        return scores

    batches = _length_sorted_batches(lengths, valid_mask, batch_size)
    use_amp = device.type == "cuda"
    total = len(batches)
    for batch_num, idx in enumerate(batches):
        if batch_num == 0 or (batch_num + 1) % 200 == 0 or batch_num + 1 == total:
            print(
                f"[{device}] {','.join(predictors)}: batch {batch_num + 1}/{total} "
                f"(size={len(idx)})",
                flush=True,
            )
        local_max_len = int(lengths[idx].max())
        sub = token_ids[idx][:, :local_max_len]
        inp = torch.from_numpy(np.ascontiguousarray(sub)).long().to(device).T
        lengths_t = torch.from_numpy(np.ascontiguousarray(lengths[idx])).long().to(device)
        d_lens = torch.ones(len(idx), dtype=torch.long, device=device)
        local_positions = np.arange(len(idx))
        for cat, predictor in predictors.items():
            try:
                probs = _forward_probs_with_oom_backoff(
                    predictor, inp, d_lens, lengths_t, local_positions, device, use_amp
                )
                scores[cat][idx] = probs
            except Exception as exc:
                print(f"Batch scoring failed for {cat} on {device}: {exc}")

    del predictors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def run_inference_all_categories(
    texts: List[str],
    models_dir: Path,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int,
    model_source: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """Score every text against every category, tokenizing once and fanning across GPUs."""
    model_source = model_source or resolve_bert_source()
    print(f"BERT source: {model_source}")

    available = [cat for cat in CATEGORIES if (models_dir / cat).exists()]
    missing = [cat for cat in CATEGORIES if cat not in available]
    for cat in missing:
        print(f"Skipped {cat}: model missing at {models_dir / cat}")
    if not available:
        return {cat: np.full(len(texts), np.nan) for cat in CATEGORIES}

    print(f"Tokenizing {len(texts):,} texts once for {len(available)} categories", flush=True)
    cleaned = _clean_texts(texts)
    token_ids, lengths, valid_mask = _tokenize_all(cleaned, tokenizer, MAX_LENGTH)
    print(f"Tokenized: {int(valid_mask.sum()):,}/{len(texts):,} non-empty texts", flush=True)

    devices = _visible_devices(device)
    shards = {
        dev: cats for dev, cats in _shard_categories(available, devices).items() if cats
    }
    print(
        "Scoring devices: "
        + ", ".join(f"{dev}={len(cats)} categories" for dev, cats in shards.items()),
        flush=True,
    )

    results: Dict[str, np.ndarray] = {}
    if len(shards) > 1:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(shards), mp_context=ctx) as pool:
                futures = [
                    pool.submit(
                        _score_categories_worker,
                        str(dev),
                        cats,
                        models_dir,
                        model_source,
                        token_ids,
                        lengths,
                        valid_mask,
                        batch_size,
                    )
                    for dev, cats in shards.items()
                ]
                for future in futures:
                    results.update(future.result())
        except Exception as exc:
            print(f"Multi-device scoring failed ({exc}); retrying sequentially", flush=True)
            results = {}

    if not results:
        for dev, cats in shards.items():
            results.update(
                _score_categories_worker(
                    str(dev), cats, models_dir, model_source, token_ids, lengths, valid_mask,
                    batch_size,
                )
            )

    for cat in missing:
        results[cat] = np.full(len(texts), np.nan)
    return {cat: results[cat] for cat in CATEGORIES}


def load_votes(votes_dir: Path) -> pd.DataFrame:
    """Load the union of the explicitly configured prepared splits."""
    splits = discover_splits(votes_dir)
    if not splits:
        raise FileNotFoundError(f"No complete prepared split found under {votes_dir}")
    frames = []
    for split_path in splits.values():
        for partition in ("train", "val", "test"):
            frames.append(pd.read_parquet(split_path / f"{partition}_votes.parquet"))
    votes = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["username", "item_id"], keep="first"
    )
    print(
        f"Votes: {len(votes):,} rows | {votes['item_id'].nunique():,} posts | source={votes_dir}"
    )
    votes["vote"] = votes["vote"].astype(float)
    # Labels are +1 for approve and -1 for remove.
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    required = ["username", "item_id", "timestamp", "vote", "community", "label"]
    missing = set(required) - set(votes.columns)
    if missing:
        raise ValueError(
            f"Votes are missing columns required for causal metadata: {sorted(missing)}"
        )
    return votes[required]


def _load_split_votes(path: Path) -> pd.DataFrame:
    """Load split votes from its configured source."""
    votes = pd.read_parquet(path)
    votes["vote"] = votes["vote"].astype(float)
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    else:
        votes["label"] = votes["label"].astype(float)
    required = ["username", "item_id", "timestamp", "vote", "community", "label"]
    missing = set(required) - set(votes.columns)
    if missing:
        raise ValueError(f"Split votes are missing required columns: {sorted(missing)}")
    return votes[required]


# Post violation scoring


def _clean_text_column(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index)
    return series.astype("string").fillna("").str.strip()


def _build_post_texts(posts: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of building one ``r/<sub> <title> <body>`` string per row."""
    community = _clean_text_column(
        posts["community"] if "community" in posts.columns else posts.get("subreddit"),
        posts.index,
    )
    is_r_prefixed = community.str.slice(0, 2).str.casefold() == "r/"
    community = community.mask(is_r_prefixed, community.str.slice(2))

    title = _clean_text_column(posts.get("title"), posts.index)
    # Canonical rows use raw ``selftext``; legacy rows use combined ``text``.
    body_field = "selftext" if "selftext" in posts.columns else "text"
    body = _clean_text_column(posts.get(body_field), posts.index)
    body = body.mask(body.isin(["[removed]", "[deleted]"]), "")

    content = (title + " " + body).str.strip()
    text = "r/" + community + " " + content
    return text.mask(content == "", "").astype(object)


POST_TEXTS_PATH = Path("data/interim/reddit/auxiliary/post_texts.parquet")
DEFAULT_VIOLATION_SCORES = Path("cache/nvse/post_violation_scores.parquet")


def score_violations(
    votes_dir: Path,
    docs_path: Path,
    models_dir: Path,
    out_path: Path,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int,
    max_posts: int = None,
    post_texts_path: Path = POST_TEXTS_PATH,
    model_source: str | Path | None = None,
) -> pd.DataFrame:
    """Score every post with the available violation models."""
    votes = load_votes(votes_dir)
    items = votes.drop_duplicates("item_id")[["item_id", "community"]].copy()

    # Use canonical post texts, then the low-coverage legacy document fallback.
    if post_texts_path.exists():
        post_texts = read_table(post_texts_path)
        required = {"item_id", "title", "selftext"}
        missing = required - set(post_texts.columns)
        if missing:
            raise ValueError(
                f"{post_texts_path} is missing canonical post-text columns: {sorted(missing)}"
            )
        post_texts = post_texts[["item_id", "title", "selftext"]]
        posts = items.merge(post_texts, on="item_id", how="left")
        n_missing = posts["title"].isna().sum()
        source = post_texts_path
    else:
        print(
            f"Post text fallback: {post_texts_path} missing; "
            "run `python -m src.data_preparation.fetch_reddit_auxiliary_data "
            "--section post-texts`"
        )
        docs = pd.read_parquet(docs_path)
        if "item_id" not in docs and "thing_id" in docs:
            docs = docs.rename(columns={"thing_id": "item_id"})
        required = {"source", "item_id", "text"}
        missing = required - set(docs.columns)
        if missing:
            raise ValueError(f"{docs_path} is missing fallback columns: {sorted(missing)}")
        docs = docs[docs["source"] == "post"].copy()
        bare = ~docs["item_id"].astype(str).str.startswith("t3_")
        docs.loc[bare, "item_id"] = "t3_" + docs.loc[bare, "item_id"].astype(str)
        docs = docs.drop_duplicates("item_id")
        posts = items.merge(docs[["item_id", "text"]], on="item_id", how="left")
        n_missing = posts["text"].isna().sum()
        source = docs_path

    print(f"Post texts: {len(posts):,} | missing={n_missing:,} | source={source}")

    if max_posts is not None:
        posts = posts.head(max_posts)
        print(f"Post limit: {max_posts:,}")
    posts["input_text"] = _build_post_texts(posts)
    texts = posts["input_text"].tolist()

    scores = run_inference_all_categories(
        texts,
        models_dir,
        tokenizer,
        device,
        batch_size,
        model_source=model_source,
    )

    for cat, arr in scores.items():
        posts[f"score_{cat}"] = arr

    score_mat = np.stack([scores[c] for c in CATEGORIES], axis=1)
    # All-NaN rows indicate that no category model loaded.
    all_nan_rows = np.all(np.isnan(score_mat), axis=1)
    top_cats, top_scores = [], []
    for i, row in enumerate(score_mat):
        if all_nan_rows[i]:
            top_cats.append(None)
            top_scores.append(np.nan)
        else:
            top_cats.append(CATEGORIES[int(np.nanargmax(row))])
            top_scores.append(float(np.nanmax(row)))
    posts["top_violation_category"] = top_cats
    posts["top_violation_score"] = top_scores
    if all_nan_rows.all():
        print("No violation model produced scores; check PyTorch and model files")

    keep = ["item_id", "community", "top_violation_category", "top_violation_score"] + [
        f"score_{c}" for c in CATEGORIES
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    posts[keep].to_parquet(out_path, index=False)
    n_scored = posts["top_violation_category"].notna().sum()
    print(f"Violation scores: {out_path} | scored={n_scored:,}/{len(posts):,}")
    return posts[keep]
