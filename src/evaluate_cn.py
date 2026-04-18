from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
)
from tqdm import trange

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE           = 0.01   # applied after per-parameter count normalisation
L2_LAMBDA               = 0.005  # L2 regularisation weight — validate against paper
N_EPOCHS                = 1000   # increased: Reddit had not converged at 500
RANDOM_SEED             = 42
INIT_STD                = 0.1    # factor init σ; smaller → stable without clipping
#   With INIT_STD=0.1: |f_u·f_n| ≈ 0.01 early on → gradients well-behaved
#   With old INIT_STD=0.5: |f_u·f_n| ≈ 0.25 → clipping was needed for stability
PATIENCE                = 50
MIN_DELTA               = 1e-5   # minimum improvement to reset patience counter
LOG_EVERY               = 20
THRESHOLD_GRID          = np.linspace(-2.0, 2.0, 400)
BRIDGING_ABS_THRESHOLD  = 0.5

# BATCH_SIZE — controls the optimisation scheme:
#   None  → full-batch GD (one gradient step per epoch; fast; current behaviour)
#   int   → mini-batch SGD (multiple steps per epoch; closer to Birdwatch paper)
#           Recommended: 512 for Reddit (~115 steps/epoch),
#                        4096 for Wikipedia (~143 steps/epoch)
BATCH_SIZE: Optional[int] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_splits(step1_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = step1_dir / "splits"
    train = pd.read_parquet(splits_dir / "train_votes.parquet")
    val   = pd.read_parquet(splits_dir / "val_votes.parquet")
    test  = pd.read_parquet(splits_dir / "test_votes.parquet")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        log.info("Loaded %s: %d votes | %d items", name, len(df), df["item_id"].nunique())
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Index mapping
# ---------------------------------------------------------------------------
def build_index_maps(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    all_users = pd.concat([train["username"], val["username"], test["username"]]).unique()
    all_items = pd.concat([train["item_id"],  val["item_id"],  test["item_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {n: i for i, n in enumerate(all_items)}
    log.info("Index maps: %d users | %d items", len(user2idx), len(item2idx))
    return user2idx, item2idx


def encode(
    df:        pd.DataFrame,
    user2idx:  Dict,
    item2idx:  Dict,
    vote_sign: int = 1,
) -> np.ndarray:
    """
    Convert a vote dataframe to a (N, 3) array [user_idx, item_idx, vote].

    vote_sign: +1 (default) leaves votes unchanged.
               -1 inverts all votes — use to test the Reddit polarity hypothesis
               (upvotes anti-correlated with mod approval).
    """
    u_idx = df["username"].map(user2idx)
    n_idx = df["item_id"].map(item2idx)
    r_val = df["vote"].astype(float) * vote_sign
    mask  = u_idx.notna() & n_idx.notna()
    n_unk = (~mask).sum()
    if n_unk:
        log.warning("Dropped %d rows with unknown user/item during encode", n_unk)
    return np.column_stack([
        u_idx[mask].values.astype(np.int32),
        n_idx[mask].values.astype(np.int32),
        r_val[mask].values,
    ])


# ---------------------------------------------------------------------------
# 3. Per-item vote features (computed from the full vote matrix)
# ---------------------------------------------------------------------------
def compute_item_vote_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-item features derived from votes only (no labels used).

    Columns
    -------
    item_id         : debate / post identifier
    n_voters        : total number of distinct voters
    vote_mean       : mean vote (+1/-1); positive = majority keep/approve
    vote_std        : std of votes; higher = more disagreement
    vote_entropy    : binary entropy of vote distribution; max at 50/50 split
    pct_positive    : fraction of +1 votes
    """
    all_votes = pd.concat([train_df, val_df, test_df], ignore_index=True)
    agg = all_votes.groupby("item_id").agg(
        n_voters   =("username", "nunique"),
        vote_mean  =("vote",     "mean"),
        vote_std   =("vote",     "std"),
        pct_positive=("vote",    lambda x: (x == 1).mean()),
    ).reset_index()

    # binary entropy: H = -p*log2(p) - (1-p)*log2(1-p), clipped to avoid log(0)
    p = agg["pct_positive"].clip(1e-9, 1 - 1e-9)
    agg["vote_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    agg["vote_std"]     = agg["vote_std"].fillna(0.0)

    return agg


# ---------------------------------------------------------------------------
# 4. Birdwatch MF model
# ---------------------------------------------------------------------------
class BirdwatchMF:
    """
    Birdwatch matrix factorisation (Wojcik et al. 2022 / arxiv 2210.15723).

    Model:   r̂_{u,n} = μ + i_u + i_n + f_u · f_n

    Alignment with the paper
    -------------------------
    • Loss:         sum_{(u,n)} (r_{u,n} - r̂_{u,n})² + λ·‖params‖²  (eq. 1)
    • Optimiser:    SGD (paper); approximated here with full-batch / mini-batch GD.
    • Count norm:   each parameter's gradient is the MEAN error over its
                    observations — the batch-GD analogue of SGD where every
                    sample contributes equally to each parameter per epoch
                    regardless of popularity.
    • Simultaneous factor update: f_u and f_n are updated using their PRE-step
                    values (both snapshots taken before either parameter moves),
                    which matches the intended gradient semantics.
    • No clipping:  stability is preserved by the smaller factor initialisation
                    (INIT_STD = 0.1), which keeps f_u·f_n ≈ 0.01 early on.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        lr:   float = LEARNING_RATE,
        lam:  float = L2_LAMBDA,
        seed: int   = RANDOM_SEED,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.mu  = 0.0
        self.i_u = np.zeros(n_users)
        self.i_n = np.zeros(n_items)
        self.f_u = rng.normal(0, INIT_STD, n_users)
        self.f_n = rng.normal(0, INIT_STD, n_items)
        self.lr  = lr
        self.lam = lam

    # ------------------------------------------------------------------
    def predict(self, u_idx: np.ndarray, n_idx: np.ndarray) -> np.ndarray:
        return (
            self.mu
            + self.i_u[u_idx]
            + self.i_n[n_idx]
            + self.f_u[u_idx] * self.f_n[n_idx]
        )

    # ------------------------------------------------------------------
    def _step(self, u: np.ndarray, n: np.ndarray, r: np.ndarray) -> float:
        r_hat = self.predict(u, n)
        err   = r - r_hat
        mse   = float(np.mean(err ** 2))

        counts_u = np.bincount(u, minlength=len(self.i_u)).astype(float)
        counts_n = np.bincount(n, minlength=len(self.i_n)).astype(float)
        safe_u   = np.maximum(counts_u, 1)
        safe_n   = np.maximum(counts_n, 1)

        # Global intercept
        self.mu -= self.lr * (-2.0 * err.mean() + 2.0 * self.lam * self.mu)

        # User bias
        d_iu = np.zeros_like(self.i_u)
        np.add.at(d_iu, u, -2.0 * err)
        self.i_u -= self.lr * (d_iu / safe_u + 2.0 * self.lam * self.i_u)

        # Item bias (primary ranking score used for classification)
        d_in = np.zeros_like(self.i_n)
        np.add.at(d_in, n, -2.0 * err)
        self.i_n -= self.lr * (d_in / safe_n + 2.0 * self.lam * self.i_n)

        # Simultaneous factor update — snapshot both before modifying either
        f_u_prev = self.f_u.copy()
        f_n_prev = self.f_n.copy()

        d_fu = np.zeros_like(self.f_u)
        np.add.at(d_fu, u, -2.0 * err * f_n_prev[n])
        self.f_u -= self.lr * (d_fu / safe_u + 2.0 * self.lam * self.f_u)

        d_fn = np.zeros_like(self.f_n)
        np.add.at(d_fn, n, -2.0 * err * f_u_prev[u])
        self.f_n -= self.lr * (d_fn / safe_n + 2.0 * self.lam * self.f_n)

        return mse

    # ------------------------------------------------------------------
    def fit(
        self,
        all_data:   np.ndarray,
        val_data:   np.ndarray,
        n_epochs:   int           = N_EPOCHS,
        batch_size: Optional[int] = BATCH_SIZE,
    ) -> Dict:
        """
        Train on all_data; monitor reconstruction MSE on val_data for early stopping.
        Labels are never seen during training.
        """
        history: Dict = {"train_mse": [], "val_mse": [], "mean_abs_fn": []}
        best_val, patience_counter = float("inf"), 0

        for epoch in trange(n_epochs, desc="Training"):
            idx       = np.random.permutation(len(all_data))
            shuf_data = all_data[idx]

            if batch_size is None:
                train_mse = self._step(
                    shuf_data[:, 0].astype(np.int32),
                    shuf_data[:, 1].astype(np.int32),
                    shuf_data[:, 2],
                )
            else:
                losses = []
                for start in range(0, len(shuf_data), batch_size):
                    b = shuf_data[start : start + batch_size]
                    losses.append(self._step(
                        b[:, 0].astype(np.int32),
                        b[:, 1].astype(np.int32),
                        b[:, 2],
                    ))
                train_mse = float(np.mean(losses))

            u_v = val_data[:, 0].astype(np.int32)
            n_v = val_data[:, 1].astype(np.int32)
            r_v = val_data[:, 2]
            val_mse = float(np.mean((r_v - self.predict(u_v, n_v)) ** 2))

            if epoch % LOG_EVERY == 0:
                mean_fn = float(np.mean(np.abs(self.f_n)))
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                history["mean_abs_fn"].append(mean_fn)
                log.info(
                    "Epoch %4d | train MSE %.4f | val MSE %.4f | mean|f_n| %.4f",
                    epoch, train_mse, val_mse, mean_fn,
                )

            # early stopping with minimum improvement threshold
            if best_val - val_mse > MIN_DELTA:
                best_val, patience_counter = val_mse, 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d (best val MSE %.4f)", epoch, best_val)
                break

        return history

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        np.savez(
            path,
            mu=np.array([self.mu]),
            i_u=self.i_u, f_u=self.f_u,
            i_n=self.i_n, f_n=self.f_n,
        )
        log.info("Model saved → %s", path)


# ---------------------------------------------------------------------------
# 5. Threshold calibration on validation set
# ---------------------------------------------------------------------------
def calibrate_threshold(
    model:    BirdwatchMF,
    val_df:   pd.DataFrame,
    item2idx: Dict[str, int],
    grid:     np.ndarray = THRESHOLD_GRID,
) -> Tuple[float, pd.DataFrame]:
    """
    Grid search over i_n thresholds; maximise macro F1 on val labels.
    """
    items_val = val_df.drop_duplicates("item_id")[["item_id", "label"]].copy()
    idxs  = items_val["item_id"].map(item2idx)
    valid = idxs.notna()
    scores = np.zeros(len(items_val))
    scores[valid.values] = model.i_n[idxs[valid].astype(int).values]
    items_val["i_n"] = scores
    y_true = items_val["label"].values

    records = []
    for thr in grid:
        y_pred = np.where(scores >= thr, 1, -1)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
        )
        records.append({"threshold": thr, "precision": p, "recall": r, "f1": f})

    cal_df   = pd.DataFrame(records)
    best     = cal_df.loc[cal_df["f1"].idxmax()]
    best_thr = float(best["threshold"])
    log.info(
        "Best threshold %.4f → val F1 %.4f (P %.4f, R %.4f)",
        best_thr, best["f1"], best["precision"], best["recall"],
    )
    return best_thr, cal_df


# ---------------------------------------------------------------------------
# 6. Evaluation on test set
# ---------------------------------------------------------------------------
def evaluate(
    model:            BirdwatchMF,
    test_df:          pd.DataFrame,
    item2idx:         Dict[str, int],
    threshold:        float,
    item_vote_feats:  Optional[pd.DataFrame] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Evaluate model on test items and compute bridging analysis.

    item_vote_feats : output of compute_item_vote_features(); if provided,
                      bridging metrics are correlated with vote diversity.

    Polarity diagnostic
    -------------------
    AUC < 0.5 means i_n is anti-correlated with the ground-truth label.
    On Reddit this is expected: upvotes are anti-correlated with mod approval.
    Re-run with vote_sign=-1 to test the polarity hypothesis.
    """
    items_test = test_df.drop_duplicates("item_id")[["item_id", "label"]].copy()

    idxs  = items_test["item_id"].map(item2idx)
    valid = idxs.notna()
    i_n_vals = np.zeros(len(items_test))
    f_n_vals = np.zeros(len(items_test))
    i_n_vals[valid.values] = model.i_n[idxs[valid].astype(int).values]
    f_n_vals[valid.values] = model.f_n[idxs[valid].astype(int).values]

    items_test["i_n"]            = i_n_vals
    items_test["f_n"]            = f_n_vals
    items_test["abs_f_n"]        = np.abs(f_n_vals)
    items_test["bridging_score"] = -np.abs(f_n_vals)   # higher = more bridging
    items_test["is_bridging"]    = np.abs(f_n_vals) < BRIDGING_ABS_THRESHOLD
    items_test["prediction"]     = np.where(i_n_vals >= threshold, 1, -1)

    # join vote-diversity features if available
    if item_vote_feats is not None:
        items_test = items_test.merge(item_vote_feats, on="item_id", how="left")

    y_true = items_test["label"].values
    y_pred = items_test["prediction"].values
    scores = items_test["i_n"].values

    # ---- Polarity diagnostic ----
    mean_in_pos = float(items_test.loc[items_test["label"] ==  1, "i_n"].mean())
    mean_in_neg = float(items_test.loc[items_test["label"] == -1, "i_n"].mean())
    polarity_correct = mean_in_pos > mean_in_neg
    if polarity_correct:
        log.info("Polarity ✓ | mean i_n: label=+1 %.4f > label=-1 %.4f", mean_in_pos, mean_in_neg)
    else:
        log.warning(
            "Polarity INVERTED | mean i_n: label=+1 %.4f < label=-1 %.4f "
            "→ i_n is anti-correlated with ground truth. "
            "Re-run with vote_sign=-1 to test polarity hypothesis.",
            mean_in_pos, mean_in_neg,
        )

    # ---- Standard metrics ----
    p,   r,   f,   _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average="macro",  zero_division=0)
    p1,  r1,  f1,  _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1],    average="binary",  zero_division=0)
    p_1, r_1, f_1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1],   average="binary",  pos_label=-1, zero_division=0)
    try:
        auc = float(roc_auc_score((y_true == 1).astype(int), scores))
    except ValueError:
        auc = float("nan")

    prec_curve, rec_curve, thr_curve = precision_recall_curve(
        (y_true == 1).astype(int), scores
    )

    # ---- Bridging analysis ----
    bridging_mask     = items_test["is_bridging"]
    non_bridging_mask = ~bridging_mask

    # accuracy of bridging vs non-bridging items
    bridging_correct     = float((items_test.loc[bridging_mask,     "prediction"] ==
                                  items_test.loc[bridging_mask,     "label"]).mean()) \
                           if bridging_mask.any() else float("nan")
    non_bridging_correct = float((items_test.loc[non_bridging_mask, "prediction"] ==
                                  items_test.loc[non_bridging_mask, "label"]).mean()) \
                           if non_bridging_mask.any() else float("nan")

    # label distribution within bridging items
    bridging_label_dist = items_test.loc[bridging_mask, "label"].value_counts().to_dict() \
                          if bridging_mask.any() else {}

    # correlation of |f_n| with vote entropy (requires item_vote_feats)
    corr_fn_entropy = float("nan")
    corr_fn_voters  = float("nan")
    if item_vote_feats is not None and "vote_entropy" in items_test.columns:
        valid_rows = items_test[["abs_f_n", "vote_entropy", "n_voters"]].dropna()
        if len(valid_rows) > 10:
            corr_fn_entropy = float(np.corrcoef(valid_rows["abs_f_n"], valid_rows["vote_entropy"])[0, 1])
            corr_fn_voters  = float(np.corrcoef(valid_rows["abs_f_n"], valid_rows["n_voters"])[0, 1])
            log.info(
                "Bridging correlation | |f_n| vs vote_entropy: %.4f | |f_n| vs n_voters: %.4f",
                corr_fn_entropy, corr_fn_voters,
            )

    # mean vote entropy for bridging vs non-bridging (if available)
    bridging_entropy_mean     = float("nan")
    non_bridging_entropy_mean = float("nan")
    if "vote_entropy" in items_test.columns:
        bridging_entropy_mean     = float(items_test.loc[bridging_mask,     "vote_entropy"].mean()) \
                                    if bridging_mask.any() else float("nan")
        non_bridging_entropy_mean = float(items_test.loc[non_bridging_mask, "vote_entropy"].mean()) \
                                    if non_bridging_mask.any() else float("nan")
        log.info(
            "Vote entropy | bridging items: %.4f | non-bridging items: %.4f",
            bridging_entropy_mean, non_bridging_entropy_mean,
        )

    log.info(
        "Bridging accuracy | bridging: %.4f | non-bridging: %.4f",
        bridging_correct, non_bridging_correct,
    )
    log.info(
        "Test | macro F1 %.4f | AUC %.4f | bridging %.1f%%",
        f, auc, 100 * items_test["is_bridging"].mean(),
    )

    metrics = {
        "threshold":                  threshold,
        "macro_precision":            float(p),
        "macro_recall":               float(r),
        "macro_f1":                   float(f),
        "precision_pos":              float(p1),
        "recall_pos":                 float(r1),
        "f1_pos":                     float(f1),
        "precision_neg":              float(p_1),
        "recall_neg":                 float(r_1),
        "f1_neg":                     float(f_1),
        "roc_auc":                    auc,
        "polarity_correct":           polarity_correct,
        "n_items_test":               int(len(items_test)),
        "n_bridging_items":           int(bridging_mask.sum()),
        "bridging_pct":               float(bridging_mask.mean()),
        "bridging_accuracy":          bridging_correct,
        "non_bridging_accuracy":      non_bridging_correct,
        "bridging_label_dist":        {str(k): int(v) for k, v in bridging_label_dist.items()},
        "bridging_vote_entropy_mean": bridging_entropy_mean,
        "non_bridging_vote_entropy":  non_bridging_entropy_mean,
        "corr_abs_fn_vote_entropy":   corr_fn_entropy,
        "corr_abs_fn_n_voters":       corr_fn_voters,
        "mean_i_n_approve":           mean_in_pos,
        "mean_i_n_remove":            mean_in_neg,
        "mean_abs_f_n":               float(np.abs(items_test["f_n"]).mean()),
        "pr_curve_precision":         prec_curve.tolist(),
        "pr_curve_recall":            rec_curve.tolist(),
        "pr_curve_thresholds":        thr_curve.tolist(),
    }
    return metrics, items_test


# ---------------------------------------------------------------------------
# 7. User polarisation analysis
# ---------------------------------------------------------------------------
def analyze_user_polarization(
    model:    BirdwatchMF,
    user2idx: Dict[str, int],
    step1_dir: Path,
) -> pd.DataFrame:
    """
    Build a per-user table combining model factors with metadata from users.parquet.

    The f_u dimension captures ideological positioning: on Wikipedia AfD,
    users with high f_u are expected to be inclusionists (low delete_rate)
    and users with low f_u deletionists (high delete_rate).  On Reddit the
    same dimension should correlate with the propensity to upvote posts that
    moderators later approve.

    Returns a DataFrame saved to step1_dir/../step2_<platform>/user_analysis.parquet.
    """
    users_path = step1_dir / "users.parquet"
    if not users_path.exists():
        log.warning("users.parquet not found at %s — skipping polarisation analysis", users_path)
        return pd.DataFrame()

    users = pd.read_parquet(users_path)
    user_list = list(user2idx.keys())
    idxs      = [user2idx[u] for u in user_list]

    params = pd.DataFrame({
        "username": user_list,
        "i_u":      model.i_u[idxs],
        "f_u":      model.f_u[idxs],
    })
    merged = params.merge(users, on="username", how="left")

    # correlation between f_u and delete_rate (available on both platforms)
    if "delete_rate" in merged.columns:
        valid = merged[["f_u", "delete_rate"]].dropna()
        if len(valid) > 10:
            corr = float(np.corrcoef(valid["f_u"], valid["delete_rate"])[0, 1])
            log.info(
                "User polarisation | corr(f_u, delete_rate): %.4f  "
                "(negative = high f_u ↔ keep/approve tendency)",
                corr,
            )

    return merged


# ---------------------------------------------------------------------------
# 8. User parameters table (minimal, for backwards compatibility)
# ---------------------------------------------------------------------------
def build_user_params(
    model:    BirdwatchMF,
    user2idx: Dict[str, int],
) -> pd.DataFrame:
    users = list(user2idx.keys())
    idxs  = [user2idx[u] for u in users]
    return pd.DataFrame({
        "username": users,
        "i_u":      model.i_u[idxs],
        "f_u":      model.f_u[idxs],
    })


# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
def save_outputs(
    output_dir:   Path,
    model:        BirdwatchMF,
    item_scores:  pd.DataFrame,
    user_params:  pd.DataFrame,
    metrics:      Dict,
    cal_df:       pd.DataFrame,
    user_analysis: Optional[pd.DataFrame] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_scores.to_parquet(output_dir / "item_scores.parquet",     index=False)
    user_params.to_parquet(output_dir / "user_params.parquet",     index=False)
    cal_df.to_parquet(     output_dir / "val_calibration.parquet", index=False)
    model.save(            output_dir / "model_params.npz")
    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    if user_analysis is not None and not user_analysis.empty:
        user_analysis.to_parquet(output_dir / "user_analysis.parquet", index=False)
        log.info("User analysis saved → %s", output_dir / "user_analysis.parquet")
    log.info("Step 2 outputs → %s", output_dir)


# ---------------------------------------------------------------------------
# 10. Main entry point
# ---------------------------------------------------------------------------
def run_step2(
    step1_dir:  Path,
    output_dir: Path,
    n_epochs:   int           = N_EPOCHS,
    lr:         float         = LEARNING_RATE,
    lam:        float         = L2_LAMBDA,
    batch_size: Optional[int] = BATCH_SIZE,
    vote_sign:  int           = 1,
) -> Dict:
    """
    Parameters
    ----------
    step1_dir   : directory produced by step1 (contains splits/ sub-folder).
    output_dir  : where to write model, scores, and metrics.
    vote_sign   : +1 (default) or -1 (invert all votes).
                  Use -1 on Reddit to test whether upvotes being anti-correlated
                  with mod approval explains the inverted AUC.
    batch_size  : None = full-batch GD; int = mini-batch SGD.
    """
    log.info("=" * 55)
    log.info(
        "STEP 2 — Birdwatch MF  (%s)  vote_sign=%+d  batch=%s",
        step1_dir.name, vote_sign,
        "full" if batch_size is None else str(batch_size),
    )
    log.info("=" * 55)

    train_df, val_df, test_df = load_splits(step1_dir)
    user2idx, item2idx = build_index_maps(train_df, val_df, test_df)

    # per-item vote features (computed before any label is used)
    item_vote_feats = compute_item_vote_features(train_df, val_df, test_df)

    train_data = encode(train_df, user2idx, item2idx, vote_sign)
    val_data   = encode(val_df,   user2idx, item2idx, vote_sign)
    test_data  = encode(test_df,  user2idx, item2idx, vote_sign)

    # transductive: train on the full vote matrix so every item gets a
    # reliably estimated i_n; labels are never passed to the model
    all_data = np.vstack([train_data, val_data, test_data])

    model = BirdwatchMF(
        n_users=len(user2idx),
        n_items=len(item2idx),
        lr=lr, lam=lam,
    )
    model.fit(all_data, val_data, n_epochs=n_epochs, batch_size=batch_size)

    log.info(
        "Converged | mean|f_n| %.4f | mean|f_u| %.4f",
        np.mean(np.abs(model.f_n)), np.mean(np.abs(model.f_u)),
    )

    threshold, cal_df    = calibrate_threshold(model, val_df, item2idx)
    metrics, item_scores = evaluate(model, test_df, item2idx, threshold, item_vote_feats)
    user_params          = build_user_params(model, user2idx)
    user_analysis        = analyze_user_polarization(model, user2idx, step1_dir)
    save_outputs(output_dir, model, item_scores, user_params, metrics, cal_df, user_analysis)

    log.info("Step 2 complete — %s", step1_dir.name)
    return {"model": model, "metrics": metrics, "item_scores": item_scores}


if __name__ == "__main__":
    run_step2(
        step1_dir=Path("results/step1/reddit"),
        output_dir=Path("results/step2/reddit"),
    )

