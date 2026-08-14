"""
Steps:
Post violation scorer     - runs NormVio violation detectors on every post
User skill extractor      - correlates user votes with violation scores (full dataset)
Ranker + benchmark        - weighted vote by skill, random K-fold evaluation (legacy)
Multi-split benchmark     - same ranker, evaluated on every split produced by prepare_data_step1
(splits/, splits_full/, splits_intersection/, windowed_folds_full/*, windowed_folds_intersection/*).
Skill is computed on each split's TRAIN votes only, threshold is calibrated on VAL, 
metrics are reported on TEST.

Architecture:
NormVio (CRAFT): EncoderBERT (DeepPavlov/bert-base-cased-conversational)
-> context encoder disabled (use_context=False, posts have no prior conversation)
-> SingleTargetClf -> binary output (violation / clean)

Models: external/normvio/normvio_redditmodels/<category>/finetuned_model.pt
One checkpoint per norm category; loaded one at a time to save memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.tfr_new.shared_features import (
    DEFAULT_CAUSAL_FEATURES,
    attach_causal_features,
    load_causal_features,
)
from src.utils.splits import discover_splits
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# Constants
BERT_TYPE   = "DeepPavlov/bert-base-cased-conversational"
HIDDEN_SIZE = 768
MAX_LENGTH  = 120
DROPOUT     = 0.1
NUM_CLASSES = 2

CATEGORIES = [
    "spam", "meta-rules", "content",
    "harassment", "hatespeech", "format",
    "off-topic", "trolling", "incivility",
]

THRESHOLD_GRID  = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5   # min votes per userxcommunityxcategory to compute skill


# MODEL DEFINITIONS  (mirrors NormVio/src/models.py)

class EncoderBERT(nn.Module):
    """Implement encoder bert."""
    def __init__(self, device: torch.device):
        """Initialize the instance."""
        super().__init__()
        self.device = device
        self.model  = BertModel.from_pretrained(BERT_TYPE).to(device)

    def forward(self, input_seq, input_lengths):
        """Run a forward pass through the model."""
        input_seq = input_seq.T                          # [seq, batch] -> [batch, seq]
        mask_ids  = (input_seq != 0).long()
        token_ids = torch.ones_like(input_seq)
        return self.model(input_ids=input_seq,
                          attention_mask=mask_ids,
                          token_type_ids=token_ids)[1]  # pooled CLS


class SingleTargetClf(nn.Module):
    """Implement single target clf."""
    def __init__(self, hidden_size: int, num_classes: int,
                 dropout: float, device: torch.device):
        """Initialize the instance."""
        super().__init__()
        self.device     = device
        self.layer1     = nn.Linear(hidden_size, hidden_size).to(device)
        self.layer1_act = nn.LeakyReLU()
        self.layer2     = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.layer2_act = nn.LeakyReLU()
        self.clf        = nn.Linear(hidden_size // 2, num_classes).to(device)
        self.drop       = nn.Dropout(p=dropout)

    def forward(self, ctx_out, dialog_lengths):
        """Run a forward pass through the model."""
        # ctx_out: [1, batch, hidden]
        lengths = (dialog_lengths.unsqueeze(0).unsqueeze(2)
                                 .expand(1, -1, ctx_out.size(2)).to(self.device))
        last = torch.gather(ctx_out, 0, lengths - 1).squeeze(0)  # [batch, hidden]
        if last.dim() == 1:
            last = last.unsqueeze(0)
        h = self.layer2_act(self.layer2(self.drop(
            self.layer1_act(self.layer1(self.drop(last))))))
        return self.clf(self.drop(h))


class Predictor(nn.Module):
    """Implement predictor."""
    def __init__(self, encoder: EncoderBERT, clf: SingleTargetClf):
        """Initialize the instance."""
        super().__init__()
        self.encoder = encoder
        self.clf     = clf

    def forward(self, input_batch, dialog_lengths, utt_lengths):
        """Run a forward pass through the model."""
        utt_hidden = self.encoder(input_batch, utt_lengths)  # [batch, hidden]
        ctx_out    = utt_hidden.unsqueeze(0)                 # [1, batch, hidden]
        return self.clf(ctx_out, dialog_lengths)


# SHARED UTILITIES

def get_device(device_str: str) -> torch.device:
    """Return device for the supplied input."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unicode_to_ascii(s: str) -> str:
    """Normalize text to ASCII characters."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def encode_text(text: str, tokenizer: BertTokenizer) -> List[int]:
    """Tokenize and encode a batch of texts."""
    cleaned = unicode_to_ascii(text.strip())
    if not cleaned:
        return []
    return tokenizer.encode(cleaned, add_special_tokens=True,
                            truncation=True, max_length=MAX_LENGTH)


def load_model(model_dir: Path, device: torch.device) -> Predictor:
    """Load model from its configured source."""
    pt_path = model_dir / "finetuned_model.pt"
    # weights_only=False required: .pt files are pickle-based (legacy format).
    # torch 2.2 supports this flag directly.
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    enc  = EncoderBERT(device)
    enc.load_state_dict(ckpt["en"])
    enc.eval()
    clf = SingleTargetClf(HIDDEN_SIZE, NUM_CLASSES, DROPOUT, device)
    clf.load_state_dict(ckpt["atk_clf"])
    clf.eval()
    return Predictor(enc, clf).to(device)


@torch.no_grad()
def score_texts(
    texts:      List[str],
    predictor:  Predictor,
    tokenizer:  BertTokenizer,
    device:     torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict violation probabilities for a batch of texts."""
    probs     = np.full(len(texts), np.nan)
    valid_idx = [i for i, t in enumerate(texts) if t.strip()]

    n_batches = (len(valid_idx) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(valid_idx), batch_size)):
        if batch_num == 0:
            print(f"      inference: {len(valid_idx)} texts, {n_batches} batches ...", flush=True)
        chunk      = valid_idx[start: start + batch_size]
        batch_texts = [texts[i] for i in chunk]
        encoded    = [encode_text(t, tokenizer) for t in batch_texts]

        nonempty = [(li, e) for li, e in enumerate(encoded) if e]
        if not nonempty:
            continue
        local_idx, enc_clean = zip(*nonempty)

        max_len = max(len(e) for e in enc_clean)
        padded  = [list(e) + [0] * (max_len - len(e)) for e in enc_clean]

        inp     = torch.LongTensor(padded).T.to(device)
        lengths = torch.tensor([len(e) for e in enc_clean], dtype=torch.long).to(device)
        d_lens  = torch.ones(len(enc_clean), dtype=torch.long).to(device)

        logits  = predictor(inp, d_lens, lengths)
        p       = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        for li, pi in zip(local_idx, p):
            probs[chunk[li]] = float(pi)

    return probs


def run_inference_all_categories(
    texts:      List[str],
    models_dir: Path,
    tokenizer:  BertTokenizer,
    device:     torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Run the inference all categories workflow."""
    scores: Dict[str, np.ndarray] = {}
    for cat in tqdm(CATEGORIES, desc="  NormVio categories"):
        model_dir = models_dir / cat
        if not model_dir.exists():
            print(f"    [SKIP] no model for '{cat}' at {model_dir}")
            scores[cat] = np.full(len(texts), np.nan)
            continue
        try:
            predictor = load_model(model_dir, device)
            probs     = score_texts(texts, predictor, tokenizer, device, batch_size)
            scores[cat] = probs
            print(f"    {cat:15s}  mean={np.nanmean(probs):.3f}  "
                  f"frac>0.5={np.nanmean(probs > 0.5):.3f}")
        except Exception as exc:
            import traceback
            print(f"    [ERROR] {cat}: {exc}")
            traceback.print_exc()
            scores[cat] = np.full(len(texts), np.nan)
        finally:
            try:
                del predictor
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return scores


ORIGINAL_CSV = Path("data/processed/final_intersection_dataset.csv")

def load_votes(votes_dir: Path) -> pd.DataFrame:
    """Load votes from its configured source."""
    if ORIGINAL_CSV.exists():
        votes = pd.read_csv(ORIGINAL_CSV)
        print(f"  Loaded votes from {ORIGINAL_CSV}: {len(votes):,} rows, "
              f"{votes['item_id'].nunique():,} unique posts")
    else:
        print(f"  [WARNING] {ORIGINAL_CSV} not found, falling back to splits/")
        splits_dir = votes_dir / "splits"
        dfs = []
        for split in ("train_votes", "val_votes", "test_votes"):
            p = splits_dir / f"{split}.parquet"
            if p.exists():
                dfs.append(pd.read_parquet(p))
        votes = pd.concat(dfs, ignore_index=True)
    votes["vote"] = votes["vote"].astype(float)
    # label: 1=approve, -1=remove (may be missing for some rows)
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    required = ["username", "item_id", "timestamp", "vote", "community", "label"]
    missing = set(required) - set(votes.columns)
    if missing:
        raise ValueError(f"Votes are missing columns required for causal metadata: {sorted(missing)}")
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


# STEP B - Post violation scorer

def _build_post_text(row: pd.Series) -> str:
    """Build post text from the supplied data."""
    subreddit = str(row.get("community", row.get("subreddit", ""))).lstrip("r/").strip()
    title     = str(row.get("title", "") or "").strip()
    body      = str(row.get("text",  "") or "").strip()
    body      = "" if body in ("[removed]", "[deleted]") else body
    content   = (title + " " + body).strip()
    return f"r/{subreddit} {content}" if content else ""


POST_TEXTS_PATH = Path("data/interim/reddit/post_texts.parquet")
DEFAULT_VIOLATION_SCORES = Path(
    "results/reddit/kfold/team-formation/post_violation_scores.parquet"
)

def score_violations(votes_dir: Path, docs_path: Path, models_dir: Path,
           out_path: Path, tokenizer: BertTokenizer,
           device: torch.device, batch_size: int,
           max_posts: int = None) -> pd.DataFrame:
    """Score every post with the available violation models."""
    print("\n=== STEP B: Post violation scorer ===")

    votes = load_votes(votes_dir)
    items = (votes.drop_duplicates("item_id")[["item_id", "community"]]
                  .copy())

    # Source 1: post_texts.parquet from fetch_expert_data.py (preferred, ~73k posts)
    # Source 2: user_documents.parquet (low overlap ~433/73k)
    if POST_TEXTS_PATH.exists():
        post_texts = pd.read_parquet(POST_TEXTS_PATH)[["item_id", "title", "text"]]
        posts = items.merge(post_texts, on="item_id", how="left")
        n_missing = posts["title"].isna().sum()
        print(f"  Source: {POST_TEXTS_PATH}")
        print(f"  Posts: {len(posts):,} unique | missing text: {n_missing:,}")
    else:
        print(f"  [WARNING] {POST_TEXTS_PATH} not found.")
        print(f"  Run: python src/data_preparation/fetch_expert_data.py --posts-only")
        print(f"  Falling back to user_documents.parquet (low overlap ~433/73k)")
        docs = pd.read_parquet(docs_path)
        docs = docs[docs["source"] == "post"].drop_duplicates("thing_id")
        items["thing_id"] = items["item_id"].str.replace("t3_", "", regex=False)
        posts = items.merge(docs[["thing_id", "title", "text"]],
                            on="thing_id", how="left")
        n_missing = posts["title"].isna().sum()
        print(f"  Posts: {len(posts):,} unique | missing text: {n_missing:,}")

    if max_posts is not None:
        posts = posts.head(max_posts)
        print(f"  [TEST MODE] limited to {max_posts} posts")
    posts["input_text"] = posts.apply(_build_post_text, axis=1)
    texts = posts["input_text"].tolist()

    scores = run_inference_all_categories(texts, models_dir,
                                          tokenizer, device, batch_size)

    for cat, arr in scores.items():
        posts[f"score_{cat}"] = arr

    score_mat = np.stack([scores[c] for c in CATEGORIES], axis=1)
    # guard: rows where ALL categories are NaN -> no model loaded successfully
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
    posts["top_violation_score"]    = top_scores
    if all_nan_rows.all():
        print("  [WARNING] ALL models failed - check torch version and .pt files")

    keep = (["item_id", "community", "top_violation_category", "top_violation_score"]
            + [f"score_{c}" for c in CATEGORIES])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    posts[keep].to_parquet(out_path, index=False)
    print(f"  Saved -> {out_path}")
    print(posts["top_violation_category"].value_counts().to_string())
    return posts[keep]


# STEP C - User skill extractor
# Skill definition (v2 - moderator agreement per NormVio category):
#
#   NormVio is used as a TOPIC MODEL only - it assigns each post to a
#   norm category (spam, incivility, ...) regardless of whether the post
#   was actually removed. The skill of user u in category c is defined as:
#
#       skill(u, c) = fraction of votes on category-c posts where
#                     vote(u, i) agrees with label(i)
#                   = P(vote == label | category == c)
#
#   where agreement means: upvote (+1) on approved post (+1),
#   or downvote (-1) on removed post (-1).
#
#   This avoids using NormVio scores as violation detectors (which fail
#   on Reddit posts) and avoids data leakage because:
#     - categories come from NormVio (not ground truth)
#     - skill is computed on temporally prior posts (chronological split)
#
# Agreement is computed on the TRAIN split only; skill is then applied
# to score posts in VAL/TEST.
#
#   - estimate_user_skill() computes skill on the whole dataset (legacy
#     behaviour, kept for backwards compatibility / the CLI "C" step).
#   - compute_user_skill() is the same logic factored out so step_E can
#     call it once per split, passing only that split's TRAIN votes.

CAUSAL_FEATURES_PATH = DEFAULT_CAUSAL_FEATURES
CAUSAL_PROFILE_COLUMNS = [
    "prior_n_posts",
    "prior_n_comments",
    "prior_n_distinct_subreddits",
    "prior_n_replies_made",
    "tenure_days_at_vote",
    "prior_n_posts_in_sub",
    "prior_n_comments_in_sub",
    "prior_n_months_active_in_sub",
    "prior_n_interaction_partners",
    "prior_total_interactions",
    "prior_n_interaction_partners_in_sub",
    "prior_total_interactions_in_sub",
]


def _load_causal_feature_skill(
    votes: pd.DataFrame,
    causal_features: pd.DataFrame | Path,
) -> pd.Series:
    """Build a train-only user-community prior from causal per-vote snapshots."""
    causal = (
        load_causal_features(causal_features)
        if isinstance(causal_features, Path)
        else causal_features
    )
    enriched = attach_causal_features(votes, causal)
    components = []
    for column in CAUSAL_PROFILE_COLUMNS:
        values = pd.to_numeric(enriched[column], errors="coerce")
        if column == "tenure_days_at_vote":
            signal = 1.0 - np.exp(-np.log1p(values.clip(lower=0)) / 7.0)
            signal = signal.fillna(0.5)
        else:
            signal = 1.0 - np.exp(-np.log1p(values.fillna(0).clip(lower=0)) / 5.0)
        components.append(signal.to_numpy(dtype=float))
    enriched["_causal_profile"] = np.column_stack(components).mean(axis=1) - 0.5
    result = enriched.groupby(["username", "community"])["_causal_profile"].mean()
    print(
        f"  Causal feature skill computed for {len(result):,} user-community profiles "
        f"(mean={result.mean():.4f}, std={result.std():.4f})"
    )
    return result


def compute_user_skill(
    votes:         pd.DataFrame,
    scores:        pd.DataFrame,
    min_votes:     int,
    causal_features: pd.DataFrame | Path = CAUSAL_FEATURES_PATH,
    alpha:         float = 0.5,
    verbose:       bool = True,
) -> pd.DataFrame:
    """Compute user skill from the supplied data."""
    scores_cat = scores[["item_id", "top_violation_category"]]
    merged = votes.merge(scores_cat, on="item_id", how="inner")
    merged = merged.dropna(subset=["label", "top_violation_category"])
    if verbose:
        print(f"  Votes with category + label: {len(merged):,}")

    merged["agrees"] = (merged["vote"] == merged["label"]).astype(float)

    skill_feat = (
        _load_causal_feature_skill(votes, causal_features)
        if alpha < 1.0
        else pd.Series(dtype=float)
    )
    use_meta   = len(skill_feat) > 0 and alpha < 1.0

    records = []
    iterator = merged.groupby(["username", "community"])
    if verbose:
        iterator = tqdm(iterator, desc="  userxcommunity")
    for (username, community), grp in iterator:
        row = {"username": username, "community": community,
               "n_votes": len(grp)}

        feat = float(skill_feat.get((username, community), np.nan)) if use_meta else np.nan

        skill_vals = []
        for cat in CATEGORIES:
            cat_grp = grp[grp["top_violation_category"] == cat]

            if len(cat_grp) >= min_votes:
                s_mod = float(cat_grp["agrees"].mean()) - 0.5
            else:
                s_mod = np.nan

            if not np.isnan(s_mod) and not np.isnan(feat):
                s = alpha * s_mod + (1 - alpha) * feat
            elif not np.isnan(s_mod):
                s = s_mod          # no metadata -> fall back to mod agreement
            elif not np.isnan(feat) and alpha < 1.0:
                s = feat           # no mod agreement -> fall back to metadata
            else:
                s = np.nan

            row[f"skill_{cat}"]     = s
            row[f"skill_mod_{cat}"] = s_mod   # keep raw components for analysis
            if not np.isnan(s):
                skill_vals.append(s)

        row["skill_feat"]    = feat
        row["skill_overall"] = float(np.nanmean(skill_vals)) if skill_vals else np.nan
        records.append(row)

    return pd.DataFrame(records)


def estimate_user_skill(votes_dir: Path, scores_path: Path,
           out_path: Path, min_votes: int,
           causal_features: pd.DataFrame | Path = CAUSAL_FEATURES_PATH,
           alpha: float = 0.5) -> pd.DataFrame:
    """Estimate user-community skill from training votes."""
    print(f"\n=== STEP C: User skill extractor (alpha={alpha:.2f}) ===")

    votes  = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)

    out_df = compute_user_skill(votes, scores, min_votes, causal_features, alpha, verbose=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_valid = out_df["skill_overall"].notna().sum()
    print(f"  Usersxcommunities: {len(out_df):,}")
    print(f"  With valid overall skill: {n_valid:,}")
    if n_valid > 0:
        print(f"  Mean skill_overall: {out_df['skill_overall'].mean():.4f}")
        print(f"  Skill > 0 (better than chance): "
              f"{(out_df['skill_overall'] > 0).sum():,} / {n_valid:,}")
    print(f"  Saved -> {out_path}")
    return out_df


# STEP D - Team Formation Ranker + benchmark (legacy: random K-fold)

def _compute_tfr_scores(
    votes:    pd.DataFrame,
    scores:   pd.DataFrame,
    skills:   pd.DataFrame,
) -> pd.DataFrame:
    """Compute tfr scores from the supplied data."""
    items = (votes.drop_duplicates("item_id")[["item_id", "community", "label"]]
                  .merge(scores[["item_id", "top_violation_category"]],
                         on="item_id", how="left"))

    # vectorised approach: merge votes with skill per item's top category
    # We do one pass per category to avoid O(N^2) iterrows
    votes_scores = votes.merge(
        scores[["item_id", "top_violation_category"]],
        on="item_id", how="left"
    )

    records = []
    if len(skills) > 0:
        for cat in CATEGORIES:
            subset = votes_scores[votes_scores["top_violation_category"] == cat].copy()
            if subset.empty:
                continue
            # attach skill for this category
            skill_col = f"skill_{cat}"
            subset = subset.merge(
                skills[["username", "community", skill_col]],
                on=["username", "community"], how="left"
            )
            records.append(subset[["item_id", "vote", skill_col, "community", "label"]]
                           .rename(columns={skill_col: "weight"}))

    # items with no category (NaN top_violation_category), or no skills table at all
    no_cat = votes_scores[votes_scores["top_violation_category"].isna()].copy()
    no_cat["weight"] = np.nan
    records.append(no_cat[["item_id", "vote", "weight", "community", "label"]])

    if len(skills) == 0:
        # nothing to weight by - every vote is unweighted
        rest = votes_scores[votes_scores["top_violation_category"].notna()].copy()
        rest["weight"] = np.nan
        records.append(rest[["item_id", "vote", "weight", "community", "label"]])

    all_votes = pd.concat(records, ignore_index=True)

    # aggregate per item
    item_scores = []
    for item_id, grp in all_votes.groupby("item_id"):
        community = grp["community"].iloc[0]
        label     = grp["label"].iloc[0]
        top_cat   = (votes_scores.loc[votes_scores["item_id"] == item_id,
                                      "top_violation_category"]
                                  .iloc[0] if len(votes_scores[votes_scores["item_id"] == item_id]) else None)

        valid = grp.dropna(subset=["weight"])
        denom = valid["weight"].abs().sum()

        if len(valid) == 0 or denom == 0:
            # no skilled voter available - skip this item entirely
            continue
        tfr_score = float((valid["weight"] * valid["vote"]).sum() / denom)
        item_scores.append({
            "item_id":      item_id,
            "community":    community,
            "label":        label,
            "tfr_score":    tfr_score,
            "top_category": top_cat,
        })

    return pd.DataFrame(item_scores)


def _calibrate(item_scores: pd.DataFrame, val_ids: np.ndarray) -> float:
    """Select a decision threshold on validation scores."""
    val    = item_scores[item_scores["item_id"].isin(val_ids)].dropna(subset=["label"])
    best_f, best_thr = -1.0, 0.0
    for thr in THRESHOLD_GRID:
        y_pred = np.where(val["tfr_score"].values >= thr, 1, -1)
        _, _, f, _ = precision_recall_fscore_support(
            val["label"].values, y_pred,
            labels=[1, -1], average="macro", zero_division=0)
        if f > best_f:
            best_f, best_thr = f, thr
    return best_thr


def _evaluate(item_scores: pd.DataFrame,
              test_ids: np.ndarray, threshold: float) -> Dict:
    """Calculate metrics for scored items."""
    test   = item_scores[item_scores["item_id"].isin(test_ids)].dropna(subset=["label"])
    y_true = test["label"].values
    y_pred = np.where(test["tfr_score"].values >= threshold, 1, -1)
    y_sc   = test["tfr_score"].values

    p,  r,  f,  _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average="macro",  zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1],    average="binary",  zero_division=0)
    p_, r_, f_, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1],   average="binary",  pos_label=-1, zero_division=0)
    try:
        auc = float(roc_auc_score((y_true == 1).astype(int), y_sc))
    except ValueError:
        auc = float("nan")

    return {
        "threshold":       threshold,
        "macro_f1":        float(f),
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "f1_pos":          float(f1),   # approve class (label=+1)
        "f1_neg":          float(f_),   # remove  class (label=-1)
        "f1_approve":      float(f1),   # alias
        "f1_remove":       float(f_),   # alias
        "precision_pos":   float(p1),
        "recall_pos":      float(r1),
        "precision_neg":   float(p_),
        "recall_neg":      float(r_),
        "roc_auc":         auc,
        "n_items":         len(test),
    }


def evaluate_kfold(votes_dir: Path, scores_path: Path, skills_path: Path,
           out_dir: Path, n_folds: int) -> Dict:
    """Evaluate kfold and return its metrics."""
    print("\n=== STEP D: Team Formation Ranker + benchmark (random K-fold, legacy) ===")

    votes  = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)
    skills = pd.read_parquet(skills_path)

    print("  Computing TFR scores ...")
    item_scores = _compute_tfr_scores(votes, scores, skills)
    n_total = votes["item_id"].nunique()
    n_scored = len(item_scores)
    print(f"  Items scored by TFR: {n_scored:,} / {n_total:,} "
          f"({100*n_scored/n_total:.1f}% coverage - rest have no skilled voter)")

    labeled = item_scores.dropna(subset=["label"]).reset_index(drop=True)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=10)

    fold_metrics: List[Dict] = []
    fold_frames:  List[pd.DataFrame] = []

    for fold_idx, (val_idx, test_idx) in enumerate(kf.split(labeled)):
        val_ids  = labeled.iloc[val_idx]["item_id"].values
        test_ids = labeled.iloc[test_idx]["item_id"].values

        threshold = _calibrate(item_scores, val_ids)
        metrics   = _evaluate(item_scores, test_ids, threshold)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        frame = item_scores[item_scores["item_id"].isin(test_ids)].copy()
        frame["prediction"] = np.where(frame["tfr_score"] >= threshold, 1, -1)
        frame["fold"]       = fold_idx
        fold_frames.append(frame)

        print(f"  Fold {fold_idx+1}/{n_folds} | thr={threshold:.3f} | "
              f"macro_F1={metrics['macro_f1']:.4f} | AUC={metrics['roc_auc']:.4f} | "
)

    scalar_keys = [k for k, v in fold_metrics[0].items()
                   if isinstance(v, (int, float)) and k != "fold"]
    agg: Dict = {"n_folds": n_folds}
    for k in scalar_keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(float(m[k]))]
        agg[k]               = float(np.mean(vals))
        agg[f"{k}_std"]      = float(np.std(vals))
        agg[f"{k}_per_fold"] = [m[k] for m in fold_metrics]

    print(f"\n  === TFR Summary ({n_folds}-fold) ===")
    for k in ("macro_f1", "roc_auc", "f1_pos", "f1_neg", "macro_precision", "macro_recall"):
        print(f"    {k:20s}  {agg[k]:.4f} +/- {agg[f'{k}_std']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(fold_frames, ignore_index=True).to_parquet(
        out_dir / "item_scores.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    pd.DataFrame(fold_metrics).to_parquet(out_dir / "fold_details.parquet", index=False)
    print(f"  Saved -> {out_dir}/")
    return agg


# STEP E - Multi-split benchmark (uses splits produced by prepare_data_step1)

def _add_net_vote_fallback(
    item_scores: pd.DataFrame,
    votes:       pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing item scores with unweighted net votes."""
    covered_ids = set(item_scores["item_id"].unique()) if not item_scores.empty else set()
    all_ids     = set(votes["item_id"].unique())
    missing_ids = all_ids - covered_ids
    if not missing_ids:
        return item_scores

    missing_votes = votes[votes["item_id"].isin(missing_ids)]
    net_vote = (
        missing_votes.groupby("item_id")
        .agg(community=("community", "first"),
             label=("label", "first"),
             tfr_score=("vote", "mean"))
        .reset_index()
    )
    net_vote["top_category"] = None
    net_vote["is_fallback"]   = True

    item_scores = item_scores.copy()
    if "is_fallback" not in item_scores.columns:
        item_scores["is_fallback"] = False

    return pd.concat([item_scores, net_vote], ignore_index=True)


def evaluate_splits(
    votes_dir:     Path,
    scores_path:   Path,
    out_dir:       Path,
    min_votes:     int   = MIN_VOTES_SKILL,
    causal_features_path: Path = CAUSAL_FEATURES_PATH,
    alpha:         float = 0.5,
) -> Dict[str, Dict]:
    """Evaluate splits and return its metrics."""
    print("\n=== STEP E: TFR benchmark on every prepared split ===")

    scores = pd.read_parquet(scores_path)
    causal = load_causal_features(causal_features_path)
    splits = discover_splits(votes_dir)

    if not splits:
        print(f"  [WARNING] no split directories found under {votes_dir} "
              f"(expected splits/, splits_full/, splits_intersection/, "
              f"windowed_folds_full/w*, windowed_folds_intersection/w*)")
        return {}

    print(f"  Found {len(splits)} split(s): {list(splits.keys())}")
    summary: Dict[str, Dict] = {}

    for split_name, split_path in splits.items():
        print(f"\n  --- split: {split_name} ---")

        train_votes = _load_split_votes(split_path / "train_votes.parquet")
        val_votes   = _load_split_votes(split_path / "val_votes.parquet")
        test_votes  = _load_split_votes(split_path / "test_votes.parquet")

        skills = compute_user_skill(
            train_votes, scores, min_votes, causal, alpha, verbose=False
        )
        n_valid_skill = int(skills["skill_overall"].notna().sum()) if len(skills) else 0
        print(f"    Skill computed on TRAIN only: {len(skills):,} userxcommunity rows "
              f"({n_valid_skill:,} with a valid overall skill)")

        item_scores_val  = _compute_tfr_scores(val_votes,  scores, skills)
        item_scores_test = _compute_tfr_scores(test_votes, scores, skills)

        if item_scores_val.empty or item_scores_test.empty:
            print(f"    [SKIP] no scoreable items in val or test for split '{split_name}' "
                  f"(val={len(item_scores_val)}, test={len(item_scores_test)})")
            continue

        threshold = _calibrate(item_scores_val, item_scores_val["item_id"].values)

        # "splits_full" only: force full test coverage using net-vote as a
        # fallback score for items TFR couldn't score (no skilled voter),
        # instead of silently dropping them. This measures how much
        # performance degrades when every item must get a decision.
        if split_name == "splits_full":
            n_before = len(item_scores_test)
            item_scores_test = _add_net_vote_fallback(item_scores_test, test_votes)
            n_fallback = len(item_scores_test) - n_before
        else:
            n_fallback = 0

        metrics = _evaluate(item_scores_test, item_scores_test["item_id"].values, threshold)

        n_test_items = test_votes["item_id"].nunique()
        metrics["split"]         = split_name
        metrics["n_train_items"] = train_votes["item_id"].nunique()
        metrics["n_val_items"]   = val_votes["item_id"].nunique()
        metrics["n_test_items"]  = n_test_items
        metrics["coverage_test"] = len(item_scores_test) / max(n_test_items, 1)
        metrics["user_profile"] = "causal train-only activity/tenure/social metadata"
        metrics["collection_time_karma_used"] = False
        if split_name == "splits_full":
            metrics["used_net_vote_fallback"] = True
            metrics["n_fallback_items"]        = n_fallback

        print(f"    thr={threshold:.3f} | macro_F1={metrics['macro_f1']:.4f} | "
              f"AUC={metrics['roc_auc']:.4f} | coverage={metrics['coverage_test']:.1%}"
              + (f" | fallback_items={n_fallback}" if split_name == "splits_full" else ""))

        split_out_dir = out_dir / split_name / "team-formation"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        skills.to_parquet(split_out_dir / "user_skill.parquet", index=False)
        item_scores_val.to_parquet(split_out_dir / "item_scores_val.parquet", index=False)
        item_scores_test.to_parquet(split_out_dir / "item_scores_test.parquet", index=False)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"    Saved -> {split_out_dir}/")

        summary[split_name] = metrics

    summary_path = out_dir / "all_splits_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary of all splits saved -> {summary_path}")

    if summary:
        print("\n  === Cross-split comparison ===")
        print(f"    {'split':35s}  {'macro_F1':>9s}  {'AUC':>7s}  {'n_test':>7s}  {'coverage':>9s}")
        for name, m in summary.items():
            print(f"    {name:35s}  {m['macro_f1']:9.4f}  {m['roc_auc']:7.4f}  "
                  f"{m['n_test_items']:7d}  {m['coverage_test']:9.1%}")

    return summary


# CLI

def parse_args() -> argparse.Namespace:
    """Parse args from the supplied input."""
    p = argparse.ArgumentParser(
        description="Team Formation Ranker - full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--votes_dir",  default="data/splits/reddit")
    p.add_argument("--docs",       default="data/interim/reddit/history/user_documents.parquet")
    p.add_argument("--models",     default="external/normvio/normvio_redditmodels",
                   help="Dir with one sub-folder per category (finetuned_model.pt)")
    p.add_argument("--out_dir",    default="results/reddit/random/team-formation")
    p.add_argument(
        "--violation_scores",
        default=str(DEFAULT_VIOLATION_SCORES),
        help=(
            "Shared NormVio score cache keyed by item_id. It is reused by every "
            "split and is independent of out_dir."
        ),
    )
    p.add_argument(
        "--force-rescore",
        action="store_true",
        help="Recompute and overwrite the shared NormVio score cache.",
    )
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default="",
                   help="cuda | cpu (auto-detected if empty)")
    p.add_argument("--n_folds",     type=int, default=5,
                   help="Number of folds for the legacy random K-fold benchmark (step D)")
    p.add_argument("--min_votes",   type=int, default=MIN_VOTES_SKILL,
                   help="Min votes per userxcommunityxcategory to compute skill")
    p.add_argument(
        "--causal_features",
        default=str(CAUSAL_FEATURES_PATH),
        help="Per-vote causal user feature parquet",
    )
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="Weight for moderator-agreement skill vs metadata skill (1.0=pure mod, 0.0=pure meta)")

    tasks = ["score", "skill", "kfold", "splits"]
    p.add_argument("--tasks", nargs="+", default=tasks, choices=tasks,
                   help="Tasks to run (default: all)")
    p.add_argument("--max_posts",   type=int, default=None,
                   help="Limit number of posts scored in step B (for testing)")
    p.add_argument("--skip-tasks", nargs="+", default=[], choices=tasks,
                   help="Tasks to skip when existing output files can be reused")
    return p.parse_args()


def main():
    """Run the command-line workflow."""
    args    = parse_args()
    device  = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | torch {torch.__version__} | file: {__file__}")
    print(f"Tasks to run: {args.tasks}")
    if args.skip_tasks:
        print(f"Tasks skipped: {args.skip_tasks}")

    tasks_to_run = [task for task in args.tasks if task not in args.skip_tasks]

    # intermediate file paths
    viol_scores_path = Path(args.violation_scores)
    user_skill_path  = out_dir / "user_skill.parquet"
    per_split_dir    = Path("results/reddit")

    if "score" in tasks_to_run:
        if viol_scores_path.exists() and not args.force_rescore:
            print(f"\nReusing shared violation scores: {viol_scores_path}")
        else:
            print("\nLoading BERT tokenizer ...")
            tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)
            score_violations(Path(args.votes_dir), Path(args.docs),
                   Path(args.models), viol_scores_path,
                   tokenizer, device, args.batch_size,
                   max_posts=args.max_posts)

    downstream_tasks = {"skill", "kfold", "splits"}.intersection(tasks_to_run)
    if downstream_tasks and not viol_scores_path.exists():
        raise FileNotFoundError(
            f"Shared violation scores not found at {viol_scores_path}. "
            "Run the 'score' task once or pass --violation_scores to an existing cache."
        )

    if "skill" in tasks_to_run:
        estimate_user_skill(Path(args.votes_dir), viol_scores_path,
               user_skill_path, args.min_votes,
               causal_features=Path(args.causal_features),
               alpha=args.alpha)

    if "kfold" in tasks_to_run:
        evaluate_kfold(Path(args.votes_dir), viol_scores_path,
               user_skill_path, out_dir, args.n_folds)

    if "splits" in tasks_to_run:
        evaluate_splits(Path(args.votes_dir), viol_scores_path,
                            per_split_dir, args.min_votes,
                            Path(args.causal_features), args.alpha)

    print("\nDone.")


if __name__ == "__main__":
    main()
