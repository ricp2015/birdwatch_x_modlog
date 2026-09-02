"""Expert-signal computation and post-level feature extraction for SEF."""

from __future__ import annotations

from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from . import runtime
from .runtime import (
    LAMBDA_SMOOTH,
    NEG_RELIABILITY_THR,
    RANK_N,
    TOP_T_LIST,
    _causal_metadata_scores,
    log,
)
from .data import build_faiss_index

# 4. SIGNAL COMPUTATION


def _compute_weighted_votes(
    item_ids: List[str],
    vote_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    top_t: int,
) -> Dict[str, float]:
    """Compute weighted votes, falling back to raw vote means when needed."""
    if weights_df.empty:
        raw = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "vote"]]
        return {iid: float(group["vote"].mean()) for iid, group in raw.groupby("item_id")}

    vote_col = "effective_vote" if "effective_vote" in weights_df.columns else "vote"
    weights = weights_df[weights_df["item_id"].isin(item_ids)][
        ["item_id", "expert_weight", vote_col]
    ].copy()
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "vote"]]
    raw_mean = raw_votes.groupby("item_id")["vote"].mean().to_dict()

    if weights.empty:
        return {item_id: raw_mean.get(item_id, 0.0) for item_id in item_ids}

    top_weights = (
        weights.sort_values("expert_weight", ascending=False)
        .groupby("item_id", sort=False)
        .head(top_t)
        .copy()
    )
    top_weights["expert_weight"] = top_weights["expert_weight"].clip(lower=1e-9)
    top_weights["wv_num"] = top_weights["expert_weight"] * top_weights[vote_col]
    aggregate = top_weights.groupby("item_id").agg(
        num=("wv_num", "sum"),
        den=("expert_weight", "sum"),
    )
    result = (aggregate["num"] / aggregate["den"]).to_dict()
    for item_id in item_ids:
        if item_id not in result:
            result[item_id] = raw_mean.get(item_id, 0.0)
    return result


def precompute_vote_precision(
    train_votes: pd.DataFrame,
    lambda_smooth: float,
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
        prec = (nc + lambda_smooth) / (n + 2 * lambda_smooth)
        rows.append(
            {
                "username": uname,
                "direction": int(direction),
                "subreddit": None,
                "prec": prec,
                "n_votes_dir": n,
                "reliability": abs(prec - 0.5),
                "bias_sign": 1 if prec >= 0.5 else -1,
            }
        )
    for (uname, sub, direction), grp in tv.groupby(["username", "community", "vote"]):
        n, nc = len(grp), grp["correct"].sum()
        prec = (nc + lambda_smooth) / (n + 2 * lambda_smooth)
        rows.append(
            {
                "username": uname,
                "direction": int(direction),
                "subreddit": sub,
                "prec": prec,
                "n_votes_dir": n,
                "reliability": abs(prec - 0.5),
                "bias_sign": 1 if prec >= 0.5 else -1,
            }
        )
    return pd.DataFrame(rows)


def compute_subreddit_base_rates(train_votes: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute subreddit base rates from the supplied data."""
    # per-post label (one row per item)
    posts = train_votes.drop_duplicates("item_id").copy()

    # detect text posts: if dataset has a 'selftext' column use it, else fallback
    has_selftext = "selftext" in posts.columns

    # parse date if available for temporal variance
    date_col = (
        "created_utc"
        if "created_utc" in posts.columns
        else ("date" if "date" in posts.columns else None)
    )

    result = {}
    for sub, grp in posts.groupby("community"):
        approved = grp["label"] == 1
        rate = float(approved.mean())
        n = len(grp)

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
                grp2["_month"] = pd.to_datetime(
                    grp2[date_col], unit="s", errors="coerce"
                ).dt.to_period("M")
                monthly = grp2.groupby("_month")["label"].apply(lambda x: (x == 1).mean())
                approve_std = float(monthly.std()) if len(monthly) > 1 else 0.0
            except Exception:
                approve_std = 0.0
        else:
            approve_std = 0.0

        result[sub] = {
            "approve_rate": rate,
            "approve_rate_text": rate_text,
            "approve_std": approve_std,
            "n_posts": float(np.log1p(n)),
        }
    return result


def compute_expert_weights(
    test_item_ids: List[str],
    train_item_ids: set,
    vote_df: pd.DataFrame,
    prec_df: pd.DataFrame,
    post_emb: np.ndarray,
    post_id2idx: Dict[str, int],
    user_emb: np.ndarray,
    user_id2idx: Dict[str, int],
    faiss_index: faiss.Index,
    k_neighbors: int,
    alpha: float,
    beta: float,
    lambda_smooth: float,
) -> pd.DataFrame:
    """Compute weights from exactly K nearest training-reference items.

    ``faiss_index`` remains in the public signature for compatibility.
    """
    metadata_map = _causal_metadata_scores(vote_df, test_item_ids)
    # lookup tables
    global_map: Dict[tuple, tuple] = {}
    sub_map: Dict[tuple, tuple] = {}
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
        return pd.DataFrame(
            columns=[
                "item_id",
                "username",
                "vote",
                "effective_vote",
                "expert_weight",
                "semantic_expert_weight",
                "causal_metadata_score",
                "local_prec",
                "contextual_prec",
                "reliability",
                "bias_sign",
                "signal_c",
                "signal_c_directed",
                "has_history",
                "has_local_evidence",
                "has_user_embedding",
                "has_real_expert_signal",
            ]
        )

    post_sub = (
        vote_df[vote_df["item_id"].isin(valid_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )
    test_votes_by_post: Dict[str, List[Tuple[str, int]]] = {}
    for row in vote_df[vote_df["item_id"].isin(valid_ids)][
        ["item_id", "username", "vote"]
    ].itertuples(index=False):
        test_votes_by_post.setdefault(row.item_id, []).append((row.username, int(row.vote)))

    # Build the exact reference-only index.  IndexFlatIP is exact and cheap to
    # rebuild for the item sets used by each split/fold.
    post_indices = [post_id2idx[iid] for iid in valid_ids]
    reference_ids = sorted(
        (iid for iid in train_item_ids if iid in post_id2idx),
        key=post_id2idx.__getitem__,
    )
    if reference_ids and k_neighbors > 0:
        reference_matrix = post_emb[[post_id2idx[iid] for iid in reference_ids]]
        reference_index = build_faiss_index(reference_matrix)
        search_k = min(k_neighbors, len(reference_ids))
        batch_sims, batch_idxs = reference_index.search(post_emb[post_indices], search_k)
    else:
        batch_sims = np.empty((len(post_indices), 0), dtype=np.float32)
        batch_idxs = np.empty((len(post_indices), 0), dtype=np.int64)

    rows = []
    for idx, item_id in enumerate(valid_ids):
        subreddit = post_sub.get(item_id)
        p_vec = post_emb[post_indices[idx]]

        # neighbours
        neighbours: List[Tuple[str, float]] = []
        for sim, j in zip(batch_sims[idx], batch_idxs[idx]):
            if j < 0:
                continue
            nid = reference_ids[j]
            if nid != item_id:
                neighbours.append((nid, max(float(sim), 0.0)))
        mean_sim = float(np.mean([s for _, s in neighbours])) if neighbours else 1.0

        voters = test_votes_by_post.get(item_id, [])
        if not voters:
            continue

        # Segnale C per tutti i votanti in un'unica operazione.
        known = [(i, u) for i, (u, _) in enumerate(voters) if u in user_id2idx]
        sig_c_map: Dict[str, float] = {}
        if known and user_emb.shape[0] > 0:
            uidxs = [user_id2idx[u] for _, u in known]
            raw_c = np.clip(user_emb[uidxs] @ p_vec, 0.0, 1.0)
            for (_, u), c in zip(known, raw_c):
                sig_c_map[u] = float(c)

        for uname, direction in voters:
            has_history = (uname, direction, subreddit) in sub_map or (
                uname,
                direction,
            ) in global_map
            g_val = global_map.get((uname, direction), (laplace_default, 0.0, 1))
            s_val = sub_map.get((uname, direction, subreddit), g_val)
            contextual_p, reliability, bias_sign = s_val
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
                        loc_weight += sim

            local_denominator = loc_weight + lambda_smooth * mean_sim
            if local_denominator <= 1e-12:
                local_rel = reliability
            else:
                local_rel = (
                    loc_correct + lambda_smooth * reliability * mean_sim
                ) / local_denominator

            has_local_evidence = loc_weight > 0.0
            has_user_embedding = uname in sig_c_map
            # Unknown profiles are neutral.  Falling back to reliability made
            # unseen upvoters systematically more influential than downvoters.
            raw_c = sig_c_map.get(uname, 0.5)
            signal_c_d = raw_c if direction == -1 else (1.0 - raw_c)
            semantic_w = alpha * local_rel + beta * reliability + gamma * signal_c_d
            metadata_score = metadata_map.get((item_id, uname), 0.5)
            expert_w = (
                (1.0 - runtime.causal_metadata_weight) * semantic_w
                + runtime.causal_metadata_weight * metadata_score
                if runtime.causal_metadata_mode != "none"
                else semantic_w
            )

            rows.append(
                {
                    "item_id": item_id,
                    "username": uname,
                    "vote": direction,
                    "effective_vote": effective_vote,
                    "expert_weight": float(expert_w),
                    "semantic_expert_weight": float(semantic_w),
                    "causal_metadata_score": float(metadata_score),
                    "local_prec": float(local_rel),
                    "contextual_prec": float(contextual_p),
                    "reliability": float(reliability),
                    "bias_sign": int(bias_sign),
                    "signal_c": float(raw_c),
                    "signal_c_directed": float(signal_c_d),
                    "has_history": bool(has_history),
                    "has_local_evidence": bool(has_local_evidence),
                    "has_user_embedding": bool(has_user_embedding),
                    "has_real_expert_signal": bool(
                        has_history or has_local_evidence or has_user_embedding
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "item_id",
                "username",
                "vote",
                "effective_vote",
                "expert_weight",
                "semantic_expert_weight",
                "causal_metadata_score",
                "local_prec",
                "contextual_prec",
                "reliability",
                "bias_sign",
                "signal_c",
                "signal_c_directed",
                "has_history",
                "has_local_evidence",
                "has_user_embedding",
                "has_real_expert_signal",
            ]
        )
    return pd.DataFrame(rows)


# 5. FEATURE EXTRACTION


def extract_post_features(
    item_ids: List[str],
    vote_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    base_rates: Dict[str, Dict[str, float]],
    top_t_list: List[int] = TOP_T_LIST,
    rank_n: int = RANK_N,
) -> pd.DataFrame:
    """Extract post features from the supplied records."""
    raw_votes = vote_df[vote_df["item_id"].isin(item_ids)][["item_id", "username", "vote"]]
    post_sub = (
        vote_df[vote_df["item_id"].isin(item_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )
    vote_col = (
        "effective_vote"
        if (not weights_df.empty and "effective_vote" in weights_df.columns)
        else "vote"
    )

    # default subreddit stats for unknown subs
    default_sub = {
        "approve_rate": 0.5,
        "approve_rate_text": 0.5,
        "approve_std": 0.0,
        "n_posts": 0.0,
    }

    rows = []
    for item_id in item_ids:
        rv = raw_votes[raw_votes["item_id"] == item_id]["vote"].values
        pw = (
            weights_df[weights_df["item_id"] == item_id]
            if not weights_df.empty
            else pd.DataFrame()
        )
        sub = post_sub.get(item_id, "__unknown__")

        feats: Dict[str, float] = {}

        # raw vote stats
        feats["raw_vote_mean"] = float(rv.mean()) if len(rv) > 0 else 0.0
        feats["raw_vote_std"] = float(rv.std()) if len(rv) > 1 else 0.0
        feats["n_voters"] = float(len(rv))
        feats["n_upvoters"] = float((rv == 1).sum())
        feats["n_downvoters"] = float((rv == -1).sum())

        # weighted vote at multiple T
        for t in top_t_list:
            wv = _compute_weighted_votes([item_id], vote_df, weights_df, t)
            feats[f"wv_top{t}"] = wv.get(item_id, feats["raw_vote_mean"])

        min_t = min(top_t_list)
        feats["expert_crowd_divergence"] = feats[f"wv_top{min_t}"] - feats["raw_vote_mean"]

        # Feature di contesto del subreddit.
        sr = base_rates.get(sub, default_sub)
        feats["sub_approve_rate"] = sr["approve_rate"]
        feats["sub_approve_rate_text"] = sr["approve_rate_text"]
        feats["sub_approve_std"] = sr["approve_std"]
        feats["sub_n_posts"] = sr["n_posts"]

        if not pw.empty:
            feats["mean_expert_weight"] = float(pw["expert_weight"].mean())
            feats["std_expert_weight"] = float(pw["expert_weight"].std()) if len(pw) > 1 else 0.0
            feats["n_experts"] = float(len(pw))

            pw_pos = pw[pw["vote"] == 1].nlargest(rank_n, "expert_weight").reset_index(drop=True)
            pw_neg = pw[pw["vote"] == -1].nlargest(rank_n, "expert_weight").reset_index(drop=True)

            # Aggregati sulle competenze.
            for tag, top in [("pos", pw_pos), ("neg", pw_neg)]:
                dv = 1.0 if tag == "pos" else -1.0
                if not top.empty:
                    w = top["expert_weight"].values.clip(min=1e-9)
                    feats[f"n_{tag}_experts"] = float(
                        len(pw[pw["vote"] == (1 if tag == "pos" else -1)])
                    )
                    feats[f"sum_weight_{tag}"] = float(w.sum())
                    feats[f"top{rank_n}_{tag}_mean_local"] = float(top["local_prec"].mean())
                    feats[f"top{rank_n}_{tag}_mean_contextual"] = float(
                        top["contextual_prec"].mean()
                    )
                    feats[f"top{rank_n}_{tag}_mean_signal_c"] = float(
                        top["signal_c_directed"].mean()
                    )
                    feats[f"top{rank_n}_{tag}_weighted_vote"] = float(
                        np.dot(w, top[vote_col].values.astype(float)) / w.sum()
                    )
                else:
                    feats[f"n_{tag}_experts"] = 0.0
                    feats[f"sum_weight_{tag}"] = 0.0
                    feats[f"top{rank_n}_{tag}_mean_local"] = 0.5
                    feats[f"top{rank_n}_{tag}_mean_contextual"] = 0.5
                    feats[f"top{rank_n}_{tag}_mean_signal_c"] = 0.5
                    feats[f"top{rank_n}_{tag}_weighted_vote"] = dv

            # Feature individuali per posizione.
            for tag, top in [("pos", pw_pos), ("neg", pw_neg)]:
                dv = 1.0 if tag == "pos" else -1.0
                for r in range(rank_n):
                    if r < len(top):
                        row = top.iloc[r]
                        feats[f"r{r + 1}_{tag}_weight"] = float(row["expert_weight"])
                        feats[f"r{r + 1}_{tag}_local"] = float(row["local_prec"])
                        feats[f"r{r + 1}_{tag}_contextual"] = float(row["contextual_prec"])
                        feats[f"r{r + 1}_{tag}_reliability"] = float(row.get("reliability", 0.0))
                        feats[f"r{r + 1}_{tag}_signal_c"] = float(row["signal_c_directed"])
                        feats[f"r{r + 1}_{tag}_evote"] = float(row[vote_col])
                    else:
                        # zero-pad missing slots
                        feats[f"r{r + 1}_{tag}_weight"] = 0.0
                        feats[f"r{r + 1}_{tag}_local"] = 0.0
                        feats[f"r{r + 1}_{tag}_contextual"] = 0.0
                        feats[f"r{r + 1}_{tag}_reliability"] = 0.0
                        feats[f"r{r + 1}_{tag}_signal_c"] = 0.0
                        feats[f"r{r + 1}_{tag}_evote"] = dv
        else:
            feats["mean_expert_weight"] = 0.0
            feats["std_expert_weight"] = 0.0
            feats["n_experts"] = 0.0
            for tag, dv in [("pos", 1.0), ("neg", -1.0)]:
                feats[f"n_{tag}_experts"] = 0.0
                feats[f"sum_weight_{tag}"] = 0.0
                feats[f"top{rank_n}_{tag}_mean_local"] = 0.5
                feats[f"top{rank_n}_{tag}_mean_contextual"] = 0.5
                feats[f"top{rank_n}_{tag}_mean_signal_c"] = 0.5
                feats[f"top{rank_n}_{tag}_weighted_vote"] = dv
                for r in range(rank_n):
                    feats[f"r{r + 1}_{tag}_weight"] = 0.0
                    feats[f"r{r + 1}_{tag}_local"] = 0.0
                    feats[f"r{r + 1}_{tag}_contextual"] = 0.0
                    feats[f"r{r + 1}_{tag}_reliability"] = 0.0
                    feats[f"r{r + 1}_{tag}_signal_c"] = 0.0
                    feats[f"r{r + 1}_{tag}_evote"] = dv

        feats["item_id"] = item_id

        # Negative-vote features for the dedicated predictor.
        if not pw.empty:
            pw_neg_all = pw[pw["vote"] == -1]
            reliable_neg = (
                pw_neg_all[
                    pw_neg_all.get("reliability", pd.Series(dtype=float)).reindex(
                        pw_neg_all.index, fill_value=0.0
                    )
                    > NEG_RELIABILITY_THR
                ]
                if "reliability" in pw_neg_all.columns
                else pw_neg_all
            )
            feats["n_neg_reliable"] = float(len(reliable_neg))
            feats["neg_expert_coverage"] = float(len(reliable_neg) / max(len(pw_neg_all), 1))
            if not reliable_neg.empty:
                top_neg_r = reliable_neg.nlargest(5, "expert_weight")
                w_neg_r = top_neg_r["expert_weight"].values.clip(min=1e-9)
                feats["wv_neg_only"] = float(
                    np.dot(w_neg_r, top_neg_r[vote_col].values.astype(float)) / w_neg_r.sum()
                )
            else:
                feats["wv_neg_only"] = 0.0
        else:
            feats["n_neg_reliable"] = 0.0
            feats["neg_expert_coverage"] = 0.0
            feats["wv_neg_only"] = 0.0

        # placeholder for neg predictor proba - filled later in run_kfold / run_single_split
        feats["neg_expert_proba"] = 0.5

        rows.append(feats)

    return pd.DataFrame(rows)


def get_meta_feature_cols(feat_df: pd.DataFrame) -> List[str]:
    """Return meta feature cols for the supplied input."""
    return [c for c in feat_df.columns if c != "item_id"]


def build_reference_heldout_train_features(
    target_ids: List[str],
    reference_ids: set,
    vote_df: pd.DataFrame,
    post_emb: np.ndarray,
    post_id2idx: Dict[str, int],
    user_emb: np.ndarray,
    user_id2idx: Dict[str, int],
    faiss_index: faiss.Index,
    k_neighbors: int,
    alpha: float,
    beta: float,
    min_coverage: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build TRAIN features from a label-disjoint reference partition."""
    overlap = set(target_ids) & set(reference_ids)
    if overlap:
        raise ValueError("SEF meta targets and feature-reference items must be disjoint")
    if not target_ids or not reference_ids:
        raise ValueError("SEF reference-heldout encoding requires two non-empty partitions")

    reference_votes = vote_df[vote_df["item_id"].isin(reference_ids)]
    precision = precompute_vote_precision(reference_votes, LAMBDA_SMOOTH, min_coverage)
    base_rates = compute_subreddit_base_rates(reference_votes)
    weights = compute_expert_weights(
        target_ids,
        reference_ids,
        vote_df,
        precision,
        post_emb,
        post_id2idx,
        user_emb,
        user_id2idx,
        faiss_index,
        k_neighbors,
        alpha,
        beta,
        LAMBDA_SMOOTH,
    )
    features = extract_post_features(target_ids, vote_df, weights, base_rates)
    log.info(
        "Meta-model features: %d disjoint reference items, %d target items",
        len(reference_ids),
        len(target_ids),
    )
    return features, weights
