"""
team_formation_ranker.py
========================
Full pipeline for the Team Formation Ranker (TFR) method.

Steps
-----
  B  Post violation scorer     — runs NormVio violation detectors on every post
  C  User skill extractor      — correlates user votes with violation scores (full dataset)
  D  Ranker + benchmark        — weighted vote by skill, random K-fold evaluation (legacy)
  E  Multi-split benchmark     — same ranker, evaluated on every split produced by
                                  prepare_data_step1 (splits/, splits_full/,
                                  splits_intersection/, windowed_folds_full/*,
                                  windowed_folds_intersection/*). Skill is computed
                                  on each split's TRAIN votes only, threshold is
                                  calibrated on VAL, metrics are reported on TEST.

Run all steps (default)
-----------------------
    python team_formation_ranker.py

    # or explicitly:
    python team_formation_ranker.py \
        --votes_dir results/step1/reddit \
        --docs      results/step1/reddit/history/user_documents.parquet \
        --models    normvio/normvio_redditmodels \
        --out_dir   results/step2/team_formation

Run individual steps
--------------------
    python team_formation_ranker.py --steps C D   # assumes B output already exists
    python team_formation_ranker.py --steps E      # only the per-split benchmark (needs B output)

Architecture
------------
NormVio (CRAFT): EncoderBERT (DeepPavlov/bert-base-cased-conversational)
→ context encoder disabled (use_context=False, posts have no prior conversation)
→ SingleTargetClf → binary output (violation / clean)

Models: normvio/normvio_redditmodels/<category>/finetuned_model.pt
One checkpoint per norm category; loaded one at a time to save memory.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BERT_TYPE   = "DeepPavlov/bert-base-cased-conversational"
HIDDEN_SIZE = 768
MAX_LENGTH  = 120
DROPOUT     = 0.1
NUM_CLASSES = 2

CATEGORIES = [
    "spam", "meta-rules", "content", "doxxing",
    "harassment", "hatespeech", "format",
    "off-topic", "trolling", "incivility",
]

THRESHOLD_GRID  = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5   # min votes per user×community×category to compute skill


# ============================================================================
# MODEL DEFINITIONS  (mirrors NormVio/src/models.py)
# ============================================================================

class EncoderBERT(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model  = BertModel.from_pretrained(BERT_TYPE).to(device)

    def forward(self, input_seq, input_lengths):
        input_seq = input_seq.T                          # [seq, batch] → [batch, seq]
        mask_ids  = (input_seq != 0).long()
        token_ids = torch.ones_like(input_seq)
        return self.model(input_ids=input_seq,
                          attention_mask=mask_ids,
                          token_type_ids=token_ids)[1]  # pooled CLS


class SingleTargetClf(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int,
                 dropout: float, device: torch.device):
        super().__init__()
        self.device     = device
        self.layer1     = nn.Linear(hidden_size, hidden_size).to(device)
        self.layer1_act = nn.LeakyReLU()
        self.layer2     = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.layer2_act = nn.LeakyReLU()
        self.clf        = nn.Linear(hidden_size // 2, num_classes).to(device)
        self.drop       = nn.Dropout(p=dropout)

    def forward(self, ctx_out, dialog_lengths):
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
    def __init__(self, encoder: EncoderBERT, clf: SingleTargetClf):
        super().__init__()
        self.encoder = encoder
        self.clf     = clf

    def forward(self, input_batch, dialog_lengths, utt_lengths):
        utt_hidden = self.encoder(input_batch, utt_lengths)  # [batch, hidden]
        ctx_out    = utt_hidden.unsqueeze(0)                 # [1, batch, hidden]
        return self.clf(ctx_out, dialog_lengths)


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def get_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unicode_to_ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def encode_text(text: str, tokenizer: BertTokenizer) -> List[int]:
    cleaned = unicode_to_ascii(text.strip())
    if not cleaned:
        return []
    return tokenizer.encode(cleaned, add_special_tokens=True,
                            truncation=True, max_length=MAX_LENGTH)


def load_model(model_dir: Path, device: torch.device) -> Predictor:
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
    """Return P(violation) for each text. Empty/unrecoverable texts → NaN."""
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
    """Run all category models; return dict cat → P(violation) array."""
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
    """
    Load all votes from the original intersection dataset (pre-density filter),
    which contains all 73k+ vote rows across all posts.
    Falls back to the split parquets if the CSV is not found.
    """
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
    return votes[["username", "item_id", "vote", "community", "label"]]


def _load_split_votes(path: Path) -> pd.DataFrame:
    """
    Load one train/val/test parquet produced by prepare_data_step1 and
    reduce it to the columns the TFR pipeline needs, with the same dtype
    handling as load_votes().
    """
    votes = pd.read_parquet(path)
    votes["vote"] = votes["vote"].astype(float)
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    else:
        votes["label"] = votes["label"].astype(float)
    return votes[["username", "item_id", "vote", "community", "label"]]


def discover_splits(votes_dir: Path) -> Dict[str, Path]:
    """
    Find every split directory produced by prepare_data_step1 under votes_dir,
    i.e. any folder containing train_votes.parquet / val_votes.parquet /
    test_votes.parquet.

    Returns a dict: split_label -> directory Path, where split_label is
    e.g. "splits", "splits_full", "splits_intersection",
    "windowed_folds_full/w020", "windowed_folds_intersection/w100", ...
    """
    found: Dict[str, Path] = {}

    # simple three-way splits
    for name in ("splits", "splits_full", "splits_intersection"):
        d = votes_dir / name
        if (d / "train_votes.parquet").exists():
            found[name] = d

    # windowed folds (one train/val/fixed-test per window size)
    for tag in ("windowed_folds_full", "windowed_folds_intersection"):
        root = votes_dir / tag
        if root.exists():
            for w_dir in sorted(root.glob("w*")):
                if (w_dir / "train_votes.parquet").exists():
                    found[f"{tag}/{w_dir.name}"] = w_dir

    return found


# ============================================================================
# STEP B — Post violation scorer
# ============================================================================

def _build_post_text(row: pd.Series) -> str:
    subreddit = str(row.get("community", row.get("subreddit", ""))).lstrip("r/").strip()
    title     = str(row.get("title", "") or "").strip()
    body      = str(row.get("text",  "") or "").strip()
    body      = "" if body in ("[removed]", "[deleted]") else body
    content   = (title + " " + body).strip()
    return f"r/{subreddit} {content}" if content else ""


POST_TEXTS_PATH = Path("results/step3_expert/post_texts.parquet")

def step_B(votes_dir: Path, docs_path: Path, models_dir: Path,
           out_path: Path, tokenizer: BertTokenizer,
           device: torch.device, batch_size: int,
           max_posts: int = None) -> pd.DataFrame:
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
    # guard: rows where ALL categories are NaN → no model loaded successfully
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
        print("  [WARNING] ALL models failed — check torch version and .pt files")

    keep = (["item_id", "community", "top_violation_category", "top_violation_score"]
            + [f"score_{c}" for c in CATEGORIES])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    posts[keep].to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(posts["top_violation_category"].value_counts().to_string())
    return posts[keep]


# ============================================================================
# STEP C — User skill extractor
# ============================================================================
# Skill definition (v2 — moderator agreement per NormVio category):
#
#   NormVio is used as a TOPIC MODEL only — it assigns each post to a
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
#   - step_C() (below) computes skill on the WHOLE dataset (legacy
#     behaviour, kept for backwards compatibility / the CLI "C" step).
#   - compute_user_skill() is the same logic factored out so step_E can
#     call it once per split, passing only that split's TRAIN votes.

METADATA_PATH = Path("data/processed/user_metadata.csv")

def _load_feature_skill(metadata_path: Path) -> pd.Series:
    """
    Compute a metadata-based skill score for each user, normalised to [-0.5, +0.5].

    Features used:
      - log(total_karma + 1)   — proxy for community standing
      - tenure_days            — days since account creation at dataset cut-off
      - is_suspended           — suspended users get NaN (excluded)
      - has_verified_email     — small credibility signal

    Each feature is percentile-ranked across all users, averaged, then
    shifted to [-0.5, +0.5] so that the median user has skill_feat = 0.
    Returns a Series indexed by username.
    """
    if not metadata_path.exists():
        print(f"  [WARNING] metadata not found at {metadata_path} — skipping feature skill")
        return pd.Series(dtype=float)

    meta = pd.read_csv(metadata_path)

    # exclude suspended accounts entirely
    meta = meta[meta["is_suspended"] != True].copy()

    # tenure in days from account creation to a fixed reference point
    meta["account_created_utc"] = pd.to_numeric(meta["account_created_utc"], errors="coerce")
    REF_TS = 1685000000  # ~May 2023, end of dataset window
    meta["tenure_days"] = (REF_TS - meta["account_created_utc"]) / 86400
    meta["tenure_days"] = meta["tenure_days"].clip(lower=0)

    meta["log_karma"]  = np.log1p(meta["total_karma"].fillna(0).clip(lower=0))
    meta["email_bonus"] = meta["has_verified_email"].fillna(False).astype(float)

    # percentile rank each feature → [0, 1]
    for col in ["log_karma", "tenure_days"]:
        meta[f"{col}_rank"] = meta[col].rank(pct=True, na_option="bottom")

    # composite: karma and tenure equally weighted, small email bonus
    meta["feat_score"] = (
        meta["log_karma_rank"] * 0.45 +
        meta["tenure_days_rank"] * 0.45 +
        meta["email_bonus"] * 0.10
    )

    # shift to [-0.5, +0.5]
    meta["skill_feat"] = meta["feat_score"] - 0.5

    result = meta.set_index("username")["skill_feat"]
    print(f"  Feature skill computed for {len(result):,} users "
          f"(mean={result.mean():.4f}, std={result.std():.4f})")
    return result


def compute_user_skill(
    votes:         pd.DataFrame,
    scores:        pd.DataFrame,
    min_votes:     int,
    metadata_path: Path = METADATA_PATH,
    alpha:         float = 0.5,
    verbose:       bool = True,
) -> pd.DataFrame:
    """
    Core skill computation, factored out of step_C so it can be run on any
    votes subset (e.g. a single split's TRAIN votes in step_E), not just the
    full dataset.

    Composite skill = alpha * skill_mod + (1-alpha) * skill_feat

    skill_mod:  moderator-agreement per NormVio category [-0.5, +0.5],
                computed only from the `votes` passed in (i.e. TRAIN votes
                when called per-split).
    skill_feat: metadata-based (karma + tenure + email)  [-0.5, +0.5]

    alpha=1.0 → pure moderator agreement
    alpha=0.0 → pure metadata
    alpha=0.5 → equal combination (default)
    """
    scores_cat = scores[["item_id", "top_violation_category"]]
    merged = votes.merge(scores_cat, on="item_id", how="inner")
    merged = merged.dropna(subset=["label", "top_violation_category"])
    if verbose:
        print(f"  Votes with category + label: {len(merged):,}")

    merged["agrees"] = (merged["vote"] == merged["label"]).astype(float)

    skill_feat = _load_feature_skill(metadata_path) if alpha < 1.0 else pd.Series(dtype=float)
    use_meta   = len(skill_feat) > 0 and alpha < 1.0

    records = []
    iterator = merged.groupby(["username", "community"])
    if verbose:
        iterator = tqdm(iterator, desc="  user×community")
    for (username, community), grp in iterator:
        row = {"username": username, "community": community,
               "n_votes": len(grp)}

        feat = float(skill_feat.get(username, np.nan)) if use_meta else np.nan

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
                s = s_mod          # no metadata → fall back to mod agreement
            elif not np.isnan(feat) and alpha < 1.0:
                s = feat           # no mod agreement → fall back to metadata
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


def step_C(votes_dir: Path, scores_path: Path,
           out_path: Path, min_votes: int,
           metadata_path: Path = METADATA_PATH,
           alpha: float = 0.5) -> pd.DataFrame:
    """
    Legacy CLI step: computes skill on the WHOLE dataset (load_votes(votes_dir),
    i.e. the original 73k-vote CSV), unrestricted by any split. Kept for
    backwards compatibility with the "C" step and with step_D's random KFold.

    If skill_mod is NaN (not enough votes in category) but skill_feat
    exists, skill_feat is used as fallback — increasing coverage.
    """
    print(f"\n=== STEP C: User skill extractor (alpha={alpha:.2f}) ===")

    votes  = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)

    out_df = compute_user_skill(votes, scores, min_votes, metadata_path, alpha, verbose=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_valid = out_df["skill_overall"].notna().sum()
    print(f"  Users×communities: {len(out_df):,}")
    print(f"  With valid overall skill: {n_valid:,}")
    if n_valid > 0:
        print(f"  Mean skill_overall: {out_df['skill_overall'].mean():.4f}")
        print(f"  Skill > 0 (better than chance): "
              f"{(out_df['skill_overall'] > 0).sum():,} / {n_valid:,}")
    print(f"  Saved → {out_path}")
    return out_df


# ============================================================================
# STEP D — Team Formation Ranker + benchmark (legacy: random K-fold)
# ============================================================================

def _compute_tfr_scores(
    votes:    pd.DataFrame,
    scores:   pd.DataFrame,
    skills:   pd.DataFrame,
) -> pd.DataFrame:
    items = (votes.drop_duplicates("item_id")[["item_id", "community", "label"]]
                  .merge(scores[["item_id", "top_violation_category"]],
                         on="item_id", how="left"))

    # vectorised approach: merge votes with skill per item's top category
    # We do one pass per category to avoid O(N²) iterrows
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
        # nothing to weight by — every vote is unweighted
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
            # no skilled voter available — skip this item entirely
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


def step_D(votes_dir: Path, scores_path: Path, skills_path: Path,
           out_dir: Path, n_folds: int) -> Dict:
    """
    Legacy benchmark: skill computed once on the whole dataset (step_C),
    then a random sklearn KFold cross-validation is used purely to get
    stable performance estimates. Does NOT use the splits produced by
    prepare_data_step1 — see step_E for that.
    """
    print("\n=== STEP D: Team Formation Ranker + benchmark (random K-fold, legacy) ===")

    votes  = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)
    skills = pd.read_parquet(skills_path)

    print("  Computing TFR scores …")
    item_scores = _compute_tfr_scores(votes, scores, skills)
    n_total = votes["item_id"].nunique()
    n_scored = len(item_scores)
    print(f"  Items scored by TFR: {n_scored:,} / {n_total:,} "
          f"({100*n_scored/n_total:.1f}% coverage — rest have no skilled voter)")

    labeled = item_scores.dropna(subset=["label"]).reset_index(drop=True)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=42)

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
        print(f"    {k:20s}  {agg[k]:.4f} ± {agg[f'{k}_std']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(fold_frames, ignore_index=True).to_parquet(
        out_dir / "item_scores.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    pd.DataFrame(fold_metrics).to_parquet(out_dir / "fold_details.parquet", index=False)
    print(f"  Saved → {out_dir}/")
    return agg


# ============================================================================
# STEP E — Multi-split benchmark (uses splits produced by prepare_data_step1)
# ============================================================================

def _add_net_vote_fallback(
    item_scores: pd.DataFrame,
    votes:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Force full coverage: for every item_id present in `votes` but missing
    from `item_scores` (i.e. TFR found no skilled voter for it), add a row
    using the item's net vote (mean of +1/-1 votes) as a stand-in score.

    Net vote lies in [-1, 1], the same range as tfr_score (a weighted
    average of skill*vote), so the SAME calibrated threshold can be applied
    to both real and fallback rows without rescaling.

    Only used for the "splits_full" benchmark, where every test item must
    receive a prediction instead of being silently dropped for lack of
    coverage.
    """
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


def step_E_multi_split(
    votes_dir:     Path,
    scores_path:   Path,
    out_dir:       Path,
    min_votes:     int   = MIN_VOTES_SKILL,
    metadata_path: Path  = METADATA_PATH,
    alpha:         float = 0.5,
) -> Dict[str, Dict]:
    """
    Run the TFR ranker on every split directory found under votes_dir
    (splits/, splits_full/, splits_intersection/, and every window under
    windowed_folds_full/ and windowed_folds_intersection/).

    For each split:
      - skill is computed with compute_user_skill() using ONLY that split's
        train_votes.parquet (no leakage from val/test)
      - item scores are computed for val_votes.parquet and test_votes.parquet
        using that split's skill table
      - the decision threshold is calibrated on val, then applied to test
      - results are saved under out_dir/<split_label>/, mirroring the
        directory layout produced by prepare_data_step1:

          out_dir/
            splits/{user_skill,item_scores_val,item_scores_test}.parquet + metrics.json
            splits_full/...
            splits_intersection/...
            windowed_folds_full/w020/...
            windowed_folds_full/w040/...
            ...
            windowed_folds_intersection/w020/...
            ...
            all_splits_summary.json   <- metrics for every split, for easy comparison
    """
    print("\n=== STEP E: TFR benchmark on every prepared split ===")

    scores = pd.read_parquet(scores_path)
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
            train_votes, scores, min_votes, metadata_path, alpha, verbose=False
        )
        n_valid_skill = int(skills["skill_overall"].notna().sum()) if len(skills) else 0
        print(f"    Skill computed on TRAIN only: {len(skills):,} user×community rows "
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
        if split_name == "splits_full":
            metrics["used_net_vote_fallback"] = True
            metrics["n_fallback_items"]        = n_fallback

        print(f"    thr={threshold:.3f} | macro_F1={metrics['macro_f1']:.4f} | "
              f"AUC={metrics['roc_auc']:.4f} | coverage={metrics['coverage_test']:.1%}"
              + (f" | fallback_items={n_fallback}" if split_name == "splits_full" else ""))

        split_out_dir = out_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        skills.to_parquet(split_out_dir / "user_skill.parquet", index=False)
        item_scores_val.to_parquet(split_out_dir / "item_scores_val.parquet", index=False)
        item_scores_test.to_parquet(split_out_dir / "item_scores_test.parquet", index=False)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"    Saved → {split_out_dir}/")

        summary[split_name] = metrics

    summary_path = out_dir / "all_splits_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary of all splits saved → {summary_path}")

    if summary:
        print("\n  === Cross-split comparison ===")
        print(f"    {'split':35s}  {'macro_F1':>9s}  {'AUC':>7s}  {'n_test':>7s}  {'coverage':>9s}")
        for name, m in summary.items():
            print(f"    {name:35s}  {m['macro_f1']:9.4f}  {m['roc_auc']:7.4f}  "
                  f"{m['n_test_items']:7d}  {m['coverage_test']:9.1%}")

    return summary


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Team Formation Ranker — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--votes_dir",  default="results/step1/reddit")
    p.add_argument("--docs",       default="results/step1/reddit/history/user_documents.parquet")
    p.add_argument("--models",     default="normvio/normvio_redditmodels",
                   help="Dir with one sub-folder per category (finetuned_model.pt)")
    p.add_argument("--out_dir",    default="results/step2/team_formation")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default="",
                   help="cuda | cpu (auto-detected if empty)")
    p.add_argument("--n_folds",     type=int, default=5,
                   help="Number of folds for the legacy random K-fold benchmark (step D)")
    p.add_argument("--min_votes",   type=int, default=MIN_VOTES_SKILL,
                   help="Min votes per user×community×category to compute skill")
    p.add_argument("--metadata_path", default="data/processed/user_metadata.csv",
                   help="Path to user metadata CSV from Shayan")
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="Weight for moderator-agreement skill vs metadata skill (1.0=pure mod, 0.0=pure meta)")

    p.add_argument("--steps",       nargs="+", default=["B", "C", "D", "E"],
                   choices=["B", "C", "D", "E"],
                   help="Steps to run (default: all, including E = per-split benchmark)")
    p.add_argument("--max_posts",   type=int, default=None,
                   help="Limit number of posts scored in step B (for testing)")
    p.add_argument("--skip_steps",  nargs="+", default=[],
                   choices=["B", "C", "D", "E"],
                   help="Steps to skip (use existing output files)")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | torch {torch.__version__} | file: {__file__}")
    print(f"Steps to run: {args.steps}")
    if args.skip_steps:
        print(f"Steps skipped: {args.skip_steps}")

    steps_to_run = [s for s in args.steps if s not in args.skip_steps]

    # intermediate file paths
    viol_scores_path = out_dir / "post_violation_scores.parquet"
    user_skill_path  = out_dir / "user_skill.parquet"
    per_split_dir    = out_dir / "benchmark_by_split"

    if "B" in steps_to_run:
        print("\nLoading BERT tokenizer ...")
        tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)
        step_B(Path(args.votes_dir), Path(args.docs),
               Path(args.models), viol_scores_path,
               tokenizer, device, args.batch_size,
               max_posts=args.max_posts)

    if "C" in steps_to_run:
        step_C(Path(args.votes_dir), viol_scores_path,
               user_skill_path, args.min_votes,
               metadata_path=Path(args.metadata_path),
               alpha=args.alpha)

    if "D" in steps_to_run:
        step_D(Path(args.votes_dir), viol_scores_path,
               user_skill_path, out_dir, args.n_folds)

    if "E" in steps_to_run:
        step_E_multi_split(Path(args.votes_dir), viol_scores_path,
                            per_split_dir, args.min_votes,
                            Path(args.metadata_path), args.alpha)

    print("\nDone.")


if __name__ == "__main__":
    main()