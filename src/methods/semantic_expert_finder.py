"""Command-line entry point for the Semantic Expert Finder method."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.sef import runtime  # noqa: E402
from src.methods.sef.data import (  # noqa: E402
    _get_model,
    build_faiss_index,
    build_post_embeddings,
    build_temporal_user_profiles,
    collect_temporal_profile_requests,
    load_post_texts,
    load_user_documents,
    load_votes,
)
from src.methods.sef.experiments import (  # noqa: E402
    evaluate_splits,
    run_kfold,
    save_outputs,
)
from src.methods.sef.runtime import (  # noqa: E402
    ALPHA,
    BETA,
    CACHE_DIR,
    K_NEIGHBORS,
    MIN_USER_VOTES,
    OUTPUT_DIR,
    SPLITS_DIR,
    TOP_T,
    log,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    attach_causal_features,
    load_causal_features,
)

# ENTRY POINT


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser(description="Semantic Expert Finder")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument(
        "--top-t",
        type=int,
        default=TOP_T,
        help="Weighted-vote scale used only for diagnostics/fallback; not grid-searched.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--min-user-votes", type=int, default=MIN_USER_VOTES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    hyperparam_group = parser.add_mutually_exclusive_group()
    hyperparam_group.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip grid search and use the CLI K/alpha/beta defaults for every split.",
    )
    hyperparam_group.add_argument(
        "--reuse-hyperparams",
        action="store_true",
        help="Skip grid search and reuse each split's existing VAL-selected "
        "K/alpha/beta from its metrics.json; --top-t remains diagnostic/fallback-only.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=10,
        help="Seed for deterministic TRAIN sampling.",
    )
    parser.add_argument("--causal-metadata-weight", type=float, default=0.20)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument(
        "--input-csv",
        "--input",
        dest="input_csv",
        type=Path,
        default=None,
        help="Base vote table (CSV, Parquet, JSON, or JSONL).",
    )
    parser.add_argument(
        "--post-texts",
        type=Path,
        default=runtime.FETCH_DIR / "post_texts.parquet",
    )
    parser.add_argument(
        "--user-documents",
        type=Path,
        default=runtime.FETCH_DIR / "user_documents.parquet",
    )
    parser.add_argument("--embedding-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--embedding-model",
        default=runtime.EMBEDDING_MODEL,
        help="SentenceTransformers model id or local model directory.",
    )
    parser.add_argument(
        "--votes-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Root containing the 13 canonical prepared splits, or one direct split.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["splits"],
        choices=["kfold", "splits"],
        help="splits = the 13 canonical benchmark splits (default); "
        "kfold = optional legacy shuffled K-fold benchmark",
    )
    parser.add_argument("--skip-tasks", nargs="+", default=[], choices=["kfold", "splits"])
    args = parser.parse_args()

    if not 0.0 <= args.causal_metadata_weight <= 1.0:
        parser.error("--causal-metadata-weight must be between 0 and 1")

    assert args.alpha + args.beta <= 1.0, (
        f"alpha + beta must be <= 1.0 (got {args.alpha + args.beta:.2f})"
    )

    tasks_to_run = [task for task in args.tasks if task not in args.skip_tasks]
    if args.reuse_hyperparams and "kfold" in tasks_to_run:
        parser.error("--reuse-hyperparams is supported for canonical splits, not legacy kfold")

    embed_dir = args.embedding_dir

    log.info(
        "Starting SEF: alpha=%.2f, beta=%.2f, gamma=%.2f, tasks=%s",
        args.alpha,
        args.beta,
        1.0 - args.alpha - args.beta,
        tasks_to_run,
    )

    runtime.causal_metadata_weight = args.causal_metadata_weight
    runtime.random_seed = args.random_seed
    runtime.causal_feature_table = load_causal_features(args.causal_features)

    base_vote_df = load_votes(args.input_csv)
    enriched_vote_df = attach_causal_features(base_vote_df, runtime.causal_feature_table)
    post_texts = load_post_texts(args.post_texts)
    user_docs = load_user_documents(args.user_documents)
    profile_requests = collect_temporal_profile_requests(base_vote_df, args.votes_dir)

    model = args.embedding_model or _get_model()
    post_emb, post_id_list = build_post_embeddings(post_texts, embed_dir, model)
    user_profiles = build_temporal_user_profiles(user_docs, profile_requests, embed_dir, model)
    log.info(
        "User profiles: %d / %d (%.1f%%)",
        user_profiles.n_users_with_history,
        base_vote_df["username"].nunique(),
        100 * user_profiles.n_users_with_history / max(base_vote_df["username"].nunique(), 1),
    )

    faiss_index = build_faiss_index(post_emb, announce=True)

    log.info("Using the full causal metadata profile; output: %s", args.output_dir)

    if "kfold" in tasks_to_run:
        fold_metrics, fold_item_scores, fold_weights = run_kfold(
            vote_df=enriched_vote_df,
            post_emb=post_emb,
            post_id_list=post_id_list,
            user_profiles=user_profiles,
            faiss_index=faiss_index,
            default_k=args.k_neighbors,
            default_t=args.top_t,
            default_alpha=args.alpha,
            default_beta=args.beta,
            min_coverage=args.min_user_votes,
            do_grid_search=not args.no_grid_search,
        )
        save_outputs(
            fold_metrics,
            fold_item_scores,
            fold_weights,
            args.output_dir / "kfold" / "expertise",
        )

    if "splits" in tasks_to_run:
        evaluate_splits(
            votes_dir=args.votes_dir,
            output_dir=args.output_dir,
            post_emb=post_emb,
            post_id_list=post_id_list,
            user_profiles=user_profiles,
            faiss_index=faiss_index,
            default_k=args.k_neighbors,
            default_t=args.top_t,
            default_alpha=args.alpha,
            default_beta=args.beta,
            min_coverage=args.min_user_votes,
            do_grid_search=not args.no_grid_search,
            reuse_hyperparams=args.reuse_hyperparams,
        )

    log.info("Processing complete.")


if __name__ == "__main__":
    main()
