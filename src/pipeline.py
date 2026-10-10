"""Reproducible Typer interface for the complete Reddit analysis pipeline."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Annotated, Mapping, Sequence

import typer

from src.reproducibility import (
    ReproductionConfig,
    load_reproduction_config,
    nested_exclusions,
    resolve_method_selection,
    validate_reproduction_config,
)
from src.utils.parallel import CommandJob, resolve_worker_count, run_command_jobs
from src.utils.splits import discover_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/processed/final_intersection_dataset.csv")
DEFAULT_SPLITS = Path("data/splits/reddit")
DEFAULT_RESULTS = Path("results/reddit")
DEFAULT_REPORT = Path("results/report")
KFOLD_DIRNAME = "kfold"
DEFAULT_CAUSAL_FEATURES = Path("data/interim/reddit/features/causal_user_vote_features.parquet")
DEFAULT_NVSE_SCORES = Path("cache/nvse/post_violation_scores.parquet")
DEFAULT_NVSE_WORK = Path("cache/nvse/work")
DEFAULT_POST_TEXTS = Path("data/interim/reddit/auxiliary/post_texts.parquet")
DEFAULT_USER_DOCUMENTS = Path("data/interim/reddit/auxiliary/user_documents.parquet")
DEFAULT_EXTERNAL_SCORES = Path("data/interim/reddit/auxiliary/moderated_posts_scores.parquet")
DEFAULT_NORMVIO_MODELS = Path("external/normvio/normvio_redditmodels")
DEFAULT_NORMVIO_BERT = "DeepPavlov/bert-base-cased-conversational"
DEFAULT_EMBEDDING_CACHE = Path("cache/embeddings")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

METHOD_MODULES = {
    "baselines": "src.methods.baselines",
    "community-notes": "src.methods.community_notes",
    "var": "src.methods.expert_selection.var",
    "nvse": "src.methods.normvio_skill_extraction",
    "sef": "src.methods.semantic_expert_finder",
    "ma-qsmf": "src.methods.ma_qsmf",
    "team-formation": "src.methods.team_formation_v2.run_all",
}
TEAM_FORMATION_METHODS = (
    "bandit",
    "graph_propagation",
    "virtual_ensembles",
    "boc_stacking",
    "role_nsga2",
)
BASELINE_METHODS = (
    "BL1_net_vote",
    "BL2_net_vote_alpha",
    "BL3_net_vote_alpha_subreddit",
    "BL4_reddit_score",
    "BL5_subreddit",
)
MA_QSMF_BRANCHES = (
    ("cn_inductive", "ma_mf"),
    ("qsmf", "ma_qsmf"),
)
TEAM_FORMATION_MODULES = {
    "bandit": "src.methods.team_formation_v2.bandit.pipeline",
    "graph_propagation": "src.methods.team_formation_v2.graph_propagation.pipeline",
    "virtual_ensembles": "src.methods.team_formation_v2.virtual_ensembles.pipeline",
    "boc_stacking": "src.methods.team_formation_v2.boc_stacking.pipeline",
    "role_nsga2": "src.methods.team_formation_v2.role_nsga2.pipeline",
}
NESTED_METHODS = {"team-formation": TEAM_FORMATION_METHODS}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Reproduce data preparation, model evaluation, and thesis figures.",
)
prepare_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="One-time, versioned data and cache preparation.",
)
run_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Repeatable model evaluation and reporting.",
)
app.add_typer(prepare_app, name="prepare")
app.add_typer(run_app, name="run")


def _python_module(module: str, *arguments: object) -> list[str]:
    serialized = [
        value.as_posix() if isinstance(value, Path) else str(value) for value in arguments
    ]
    return [Path(sys.executable).as_posix(), "-B", "-m", module, *serialized]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git_state() -> dict[str, object]:
    base = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    commit = subprocess.run(
        [*base, "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [*base, "status", "--short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _input_fingerprints(splits_root: Path, extra_inputs: Sequence[Path] = ()) -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
        *extra_inputs,
    ]
    root = splits_root if splits_root.is_absolute() else PROJECT_ROOT / splits_root
    if root.exists():
        paths.extend(sorted(root.rglob("*manifest.json")))
        paths.extend(sorted(root.rglob("*_votes.parquet")))
    return {_relative(path): digest for path in paths if (digest := _sha256(path)) is not None}


def _source_tree_sha256() -> str:
    """Fingerprint source contents too, including an uncommitted working tree."""
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        digest.update(_relative(path).encode("utf-8"))
        file_digest = _sha256(path)
        if file_digest is not None:
            digest.update(file_digest.encode("ascii"))
    return digest.hexdigest()


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)


def _write_run_manifest(
    manifest_path: Path,
    category: str,
    steps: list[dict[str, object]],
    splits_root: Path,
    extra_inputs: Sequence[Path] = (),
) -> None:
    destination = manifest_path if manifest_path.is_absolute() else PROJECT_ROOT / manifest_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "category": category,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": PROJECT_ROOT.as_posix(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_state(),
        "source_tree_sha256": _source_tree_sha256(),
        "input_fingerprints": _input_fingerprints(splits_root, extra_inputs),
        "steps": steps,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Manifest: {_relative(destination)}")


def _execute(
    commands: Sequence[tuple[str, list[str]] | CommandJob],
    *,
    category: str,
    manifest_path: Path,
    splits_root: Path,
    dry_run: bool,
    continue_on_error: bool = False,
    extra_inputs: Sequence[Path] = (),
    n_workers: int = 1,
    threads_per_worker: int = 1,
    resource_limits: Mapping[str, int] | None = None,
) -> None:
    jobs = [
        command if isinstance(command, CommandJob) else CommandJob(command[0], command[1])
        for command in commands
    ]
    for job in jobs:
        typer.echo(f"\n[{job.name}] {_display_command(job.command)}")
    results = run_command_jobs(
        jobs,
        cwd=PROJECT_ROOT,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
        announce=lambda name: typer.echo(f"[{name}] started", err=True),
        resource_limits=resource_limits,
    )
    records: list[dict[str, object]] = [
        {
            "name": result.name,
            "command": result.command,
            "dry_run": dry_run,
            "started_at_utc": result.started_at_utc,
            "ended_at_utc": result.ended_at_utc,
            "duration_seconds": result.duration_seconds,
            "return_code": result.return_code,
            "status": result.status,
            "resource": result.resource,
            "environment_overrides": dict(result.environment or {}),
        }
        for result in results
    ]
    failure = any(result.return_code not in (None, 0) for result in results)
    _write_run_manifest(
        manifest_path,
        category,
        records,
        splits_root,
        extra_inputs=extra_inputs,
    )
    if failure:
        raise typer.Exit(code=1)


def _prepared(path: Path, force: bool) -> bool:
    target = path if path.is_absolute() else PROJECT_ROOT / path
    if target.exists() and not force:
        typer.echo(f"Already prepared: {_relative(target)} (use --force to rebuild)")
        return True
    return False


def _scoped_root(root: Path, scope: str) -> Path:
    """Append one evaluation scope exactly once."""
    return root if root.name == scope else root / scope


def _load_config(path: Path) -> ReproductionConfig:
    try:
        return load_reproduction_config(path, PROJECT_ROOT, METHOD_MODULES, NESTED_METHODS)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


def _direct_config(
    input_path: Path,
    dataset: str,
    methods: Sequence[str],
    *,
    post_texts: Path | None,
    user_documents: Path | None,
    user_contributions: Path | None,
    user_metadata: Path | None,
    external_scores: Path | None,
    download_user_documents: bool,
    n_user_documents: int,
    kfold: bool,
    force: bool,
    graphs: bool,
    device: str,
    excluded_methods: Sequence[str] = (),
) -> ReproductionConfig:
    """Build the conventional layout used by the simple ``--input`` interface."""
    if not dataset or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in dataset
    ):
        raise typer.BadParameter(
            "dataset must contain only letters, digits, '.', '_' or '-'",
            param_hint="--dataset",
        )
    try:
        selected, normalized_exclusions = resolve_method_selection(
            methods,
            excluded_methods,
            valid_methods=tuple(METHOD_MODULES),
            nested_methods=NESTED_METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
    interim = PROJECT_ROOT / "data" / "interim" / dataset
    processed = PROJECT_ROOT / "data" / "processed" / dataset
    cache = PROJECT_ROOT / "cache" / dataset
    if dataset == "reddit":
        processed = PROJECT_ROOT / "data" / "processed"
        cache = PROJECT_ROOT / "cache"
    default_user_contributions = PROJECT_ROOT / "data" / "raw" / dataset / "user_contributions"
    report = PROJECT_ROOT / "results" / "report" / dataset
    if dataset == "reddit":
        default_user_contributions = (
            PROJECT_ROOT / "data" / "raw" / "user_contributions" / "user_contributions"
        )
        report = PROJECT_ROOT / "results" / "report"

    def supplied(path: Path | None, default: Path) -> Path:
        if path is None:
            return default
        return path if path.is_absolute() else PROJECT_ROOT / path

    auxiliary = interim / "auxiliary"
    return ReproductionConfig(
        source=None,
        dataset=dataset,
        votes=input_path if input_path.is_absolute() else PROJECT_ROOT / input_path,
        post_texts=supplied(post_texts, auxiliary / "post_texts.parquet"),
        user_documents=supplied(user_documents, auxiliary / "user_documents.parquet"),
        user_contributions=supplied(user_contributions, default_user_contributions),
        user_metadata=supplied(user_metadata, processed / "user_metadata.csv"),
        external_scores=supplied(external_scores, auxiliary / "moderated_posts_scores.parquet"),
        normvio_models=PROJECT_ROOT / "external" / "normvio" / "normvio_redditmodels",
        normvio_bert=DEFAULT_NORMVIO_BERT,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        interim=interim,
        splits=PROJECT_ROOT / "data" / "splits" / dataset,
        results=PROJECT_ROOT / "results" / dataset,
        report=report,
        cache=cache,
        causal_features=interim / "features" / "causal_user_vote_features.parquet",
        nvse_scores=cache / "nvse" / "post_violation_scores.parquet",
        embedding_cache=cache / "embeddings",
        causal_features_are_precomputed=False,
        nvse_scores_are_precomputed=False,
        methods=selected,
        excluded_methods=normalized_exclusions,
        prepare_dataset=True,
        prepare_user_features=True,
        prepare_nvse_scores=True,
        download_user_documents=download_user_documents,
        n_user_documents=n_user_documents,
        graphs=graphs,
        kfold=kfold,
        force=force,
        continue_on_error=False,
        device=device,
        seed=10,
        reuse_sef_hyperparameters=False,
        skip_existing_team_methods=False,
        graph_sections=("all",),
    )


def _preflight(config: ReproductionConfig) -> None:
    errors, warnings = validate_reproduction_config(config, PROJECT_ROOT)
    for message in warnings:
        typer.echo(f"WARNING: {message}", err=True)
    if errors:
        for message in errors:
            typer.echo(f"ERROR: {message}", err=True)
        raise typer.Exit(code=2)
    typer.echo(
        f"Preflight OK: dataset={config.dataset} | methods={','.join(config.methods)} | "
        f"splits={_relative(config.splits)}"
        + (f" | excluded={','.join(config.excluded_methods)}" if config.excluded_methods else "")
    )


def _optional_path(path: Path | None, fallback: Path) -> Path:
    return path if path is not None else fallback


def _local_model_files(model: str) -> list[Path]:
    """Resolve a local path/cached Hugging Face snapshot for provenance hashing."""
    candidate = Path(model)
    candidate = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not candidate.is_dir():
        try:
            from huggingface_hub import snapshot_download

            candidate = Path(snapshot_download(model, local_files_only=True))
        except Exception:
            return []
    return sorted(path for path in candidate.rglob("*") if path.is_file())


def _parallel_methods_command(
    config: ReproductionConfig,
    *,
    splits_root: Path,
    results_root: Path,
    manifest: Path,
    n_workers: int,
    threads_per_worker: int,
    memory_workers: int,
    gpu_devices: Sequence[str],
) -> list[str]:
    """Build a nested method-stage command while preserving stage dependencies."""

    post_texts = _optional_path(config.post_texts, config.interim / "missing_post_texts")
    user_documents = _optional_path(
        config.user_documents, config.interim / "auxiliary" / "user_documents.parquet"
    )
    external_scores = _optional_path(
        config.external_scores, config.interim / "missing_external_scores"
    )
    args: list[object] = [
        "run",
        "methods",
        "--splits-root",
        splits_root,
        "--results-root",
        results_root,
        "--input",
        config.votes,
        "--causal-features",
        config.causal_features,
        "--nvse-scores",
        config.nvse_scores,
        "--nvse-work",
        config.cache / "nvse" / "work",
        "--post-texts",
        post_texts,
        "--user-documents",
        user_documents,
        "--external-scores",
        external_scores,
        "--embedding-cache",
        config.embedding_cache,
        "--embedding-model",
        config.embedding_model,
        "--device",
        config.device,
        "--n-workers",
        n_workers,
        "--threads-per-worker",
        threads_per_worker,
        "--memory-workers",
        memory_workers,
        "--manifest",
        manifest,
    ]
    for method in config.methods:
        args.extend(("--method", method))
    for excluded in config.excluded_methods:
        args.extend(("--exclude-method", excluded))
    for gpu in gpu_devices:
        args.extend(("--gpu-device", gpu))
    if config.reuse_sef_hyperparameters:
        args.append("--reuse-sef-hyperparameters")
    if config.skip_existing_team_methods:
        args.append("--skip-existing-team-methods")
    if config.continue_on_error:
        args.append("--continue-on-error")
    return _python_module("src.pipeline", *args)


def _reproduction_commands(
    config: ReproductionConfig,
    *,
    n_workers: int = 1,
    threads_per_worker: int = 1,
    memory_workers: int = 1,
    gpu_devices: Sequence[str] = (),
) -> list[tuple[str, list[str]]]:
    """Build the complete preparation/evaluation/report plan for one dataset."""
    commands: list[tuple[str, list[str]]] = []
    datasets_dir = config.interim / "datasets"
    dataset_marker = datasets_dir / "prepare_reddit_data.manifest.json"
    dataset_scheduled = config.prepare_dataset and (
        config.force or not dataset_marker.exists() or not bool(discover_splits(config.splits))
    )
    if dataset_scheduled:
        command = _python_module(
            "src.data_preparation.prepare_reddit_data",
            "--input",
            config.votes,
            "--output-dir",
            datasets_dir,
            "--splits-dir",
            config.splits,
            "--seed",
            config.seed,
        )
        if config.external_scores is not None:
            command.extend(("--external-scores", config.external_scores.as_posix()))
        if config.post_texts is not None:
            command.extend(("--post-texts", config.post_texts.as_posix()))
        commands.append(("prepare-dataset", command))

    needs_user_documents = "sef" in config.methods
    user_documents = _optional_path(
        config.user_documents, config.interim / "auxiliary" / "user_documents.parquet"
    )
    user_documents_scheduled = needs_user_documents and not user_documents.is_file()
    if (
        user_documents_scheduled
        and config.user_contributions is not None
        and config.user_contributions.is_dir()
    ):
        commands.append(
            (
                "prepare-user-documents-local",
                _python_module(
                    "src.data_preparation.build_user_documents",
                    "--input-dir",
                    config.user_contributions,
                    "--users-from",
                    config.votes,
                    "--output",
                    user_documents,
                    "--n-user-docs",
                    config.n_user_documents,
                ),
            )
        )

    if needs_user_documents and config.download_user_documents:
        commands.append(
            (
                "download-missing-user-documents",
                _python_module(
                    "src.data_preparation.fetch_reddit_auxiliary_data",
                    "--section",
                    "user-documents",
                    "--input-csv",
                    config.votes,
                    "--output-dir",
                    user_documents.parent,
                    "--user-documents-output",
                    user_documents,
                    "--n-user-docs",
                    config.n_user_documents,
                    "--missing-user-documents-only",
                ),
            )
        )

    causal_needed = bool({"nvse", "sef", "team-formation"} & set(config.methods))
    if (
        not config.causal_features_are_precomputed
        and config.prepare_user_features
        and causal_needed
        and (config.force or not config.causal_features.exists())
    ):
        build_state = config.interim / "build_state"
        command = _python_module(
            "src.data_preparation.build_user_history_features",
            "--stage",
            "all",
            "--input-dir",
            config.user_contributions,
            "--author-index",
            build_state / "author_index.sqlite",
            "--feature-root",
            build_state / "user_temporal_features",
            "--votes",
            config.votes,
            "--metadata",
            _optional_path(config.user_metadata, config.interim / "missing_user_metadata"),
            "--post-texts",
            _optional_path(config.post_texts, config.interim / "missing_post_texts"),
            "--output",
            config.causal_features,
        )
        if config.force:
            command.extend(("--rebuild-index", "--rebuild-snapshots"))
        commands.append(("prepare-causal-user-features", command))

    if (
        "nvse" in config.methods
        and not config.nvse_scores_are_precomputed
        and config.prepare_nvse_scores
        and (config.force or not config.nvse_scores.exists())
    ):
        command = _python_module(
            "src.methods.normvio_skill_extraction",
            "--votes_dir",
            config.splits,
            "--post_texts",
            config.post_texts,
            "--docs",
            _optional_path(config.user_documents, config.interim / "missing_user_documents"),
            "--models",
            config.normvio_models,
            "--bert_model",
            config.normvio_bert,
            "--violation_scores",
            config.nvse_scores,
            "--out_dir",
            config.cache / "nvse" / "work",
            "--tasks",
            "score",
            "--device",
            config.device,
        )
        if config.force:
            command.append("--force-rescore")
        commands.append(("prepare-nvse-scores", command))

    kfold_root = _scoped_root(config.splits, KFOLD_DIRNAME)
    if (
        config.kfold
        and not dataset_scheduled
        and (config.force or not (kfold_root / "manifest.json").exists())
    ):
        commands.append(
            (
                "prepare-kfold",
                _python_module(
                    "src.data_preparation.prepare_kfold_splits",
                    "--splits-root",
                    config.splits,
                    "--output-root",
                    kfold_root,
                    "--seed",
                    config.seed,
                ),
            )
        )

    commands.append(
        (
            "methods",
            _parallel_methods_command(
                config,
                splits_root=config.splits,
                results_root=config.results,
                manifest=config.results / "reproducibility" / "latest_methods_run.json",
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_workers=memory_workers,
                gpu_devices=gpu_devices,
            ),
        )
    )

    if config.kfold:
        kfold_results = _scoped_root(config.results, KFOLD_DIRNAME)
        commands.append(
            (
                "kfold-methods",
                _parallel_methods_command(
                    config,
                    splits_root=kfold_root,
                    results_root=kfold_results,
                    manifest=(
                        config.results
                        / "reproducibility"
                        / "latest_kfold_methods_run.json"
                    ),
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    memory_workers=memory_workers,
                    gpu_devices=gpu_devices,
                ),
            )
        )
        commands.append(
            (
                "kfold-graphs",
                _python_module(
                    "src.comparison.graphs",
                    "--k-fold-only",
                    "--splits-root",
                    kfold_root,
                    "--results-root",
                    kfold_results,
                    "--output-dir",
                    config.report / "kfold",
                ),
            )
        )

    if config.graphs:
        graph_args: list[object] = [
            "--results-root",
            config.results,
            "--splits-root",
            config.splits,
            "--output-dir",
            config.report,
        ]
        for section in config.graph_sections:
            graph_args.extend(("--section", section))
        commands.append(("graphs", _python_module("src.comparison.graphs", *graph_args)))
    return commands


@app.command("validate")
def validate_config(
    config_path: Annotated[Path, typer.Option("--config", "-c")],
) -> None:
    """Validate one dataset config, every schema, and external dependency."""
    _preflight(_load_config(config_path))


@app.command("reproduce")
def reproduce(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Votes table (CSV, Parquet, JSON, or JSONL). Required without --config.",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Name used to isolate prepared data and outputs."),
    ] = "reddit",
    method: Annotated[
        list[str] | None,
        typer.Option(
            "--method",
            "-m",
            help="Method family to run. Repeat for multiple methods; omitted means all.",
        ),
    ] = None,
    exclude_method: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-method",
            "-x",
            help="Method name to omit, such as bandit or sef; repeat for several.",
        ),
    ] = None,
    post_texts: Annotated[
        Path | None,
        typer.Option(help="Local post-text table; otherwise use the conventional path."),
    ] = None,
    user_documents: Annotated[
        Path | None,
        typer.Option(help="Local historical user-document table for SEF."),
    ] = None,
    user_contributions: Annotated[
        Path | None,
        typer.Option(help="Directory containing one contribution JSONL per user."),
    ] = None,
    user_metadata: Annotated[
        Path | None,
        typer.Option(help="Local account metadata table used for tenure features."),
    ] = None,
    external_scores: Annotated[
        Path | None,
        typer.Option(help="Local per-item Reddit/Arctic Shift score table."),
    ] = None,
    download_user_documents: Annotated[
        bool,
        typer.Option(
            "--download-user-documents",
            help="Top up missing local user documents from Arctic Shift.",
        ),
    ] = False,
    n_user_documents: Annotated[
        int,
        typer.Option(
            "--n-user-documents",
            min=2,
            help="Maximum local documents per user for semantic embeddings.",
        ),
    ] = 150,
    kfold: Annotated[
        bool,
        typer.Option("--k-fold", help="Also run the prepared five-fold evaluation."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild prepared splits and caches."),
    ] = False,
    graphs: Annotated[
        bool,
        typer.Option("--graphs/--no-graphs", help="Enable or skip final figures."),
    ] = True,
    device: Annotated[
        str,
        typer.Option("--device", help="Execution device: cpu, cuda, or cuda:N."),
    ] = "cpu",
    gpu_device: Annotated[
        list[str] | None,
        typer.Option(
            "--gpu-device",
            help="GPU scheduling lane for methods; repeat for multiple GPUs.",
        ),
    ] = None,
    n_workers: Annotated[
        int,
        typer.Option(
            min=0,
            help="Concurrent method/split processes; 0 selects a conservative value.",
        ),
    ] = 1,
    threads_per_worker: Annotated[
        int,
        typer.Option(min=1, help="BLAS/OpenMP threads available to each method process."),
    ] = 1,
    memory_workers: Annotated[
        int,
        typer.Option(min=1, help="Maximum concurrent memory-heavy Team Formation jobs."),
    ] = 1,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional advanced config for non-standard paths."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print the plan without executing it."),
    ] = False,
) -> None:
    """Prepare, evaluate, and render one dataset in a single launch."""
    if config_path is not None and input_path is not None:
        raise typer.BadParameter("use either --input or --config, not both")
    if config_path is not None:
        config = _load_config(config_path)
        if exclude_method:
            try:
                selected, normalized_exclusions = resolve_method_selection(
                    config.methods,
                    [*config.excluded_methods, *exclude_method],
                    valid_methods=tuple(METHOD_MODULES),
                    nested_methods=NESTED_METHODS,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
            config = replace(
                config,
                methods=selected,
                excluded_methods=normalized_exclusions,
            )
    elif input_path is not None:
        config = _direct_config(
            input_path,
            dataset,
            method or ["all"],
            excluded_methods=exclude_method or [],
            post_texts=post_texts,
            user_documents=user_documents,
            user_contributions=user_contributions,
            user_metadata=user_metadata,
            external_scores=external_scores,
            download_user_documents=download_user_documents,
            n_user_documents=n_user_documents,
            kfold=kfold,
            force=force,
            graphs=graphs,
            device=device,
        )
    else:
        raise typer.BadParameter("provide the votes table with --input", param_hint="--input")
    _preflight(config)
    commands = _reproduction_commands(
        config,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_workers=memory_workers,
        gpu_devices=gpu_device or (),
    )
    if not commands:
        typer.echo("Nothing to run: every configured stage is disabled or already prepared.")
        return
    extra_inputs = [
        path
        for path in (
            config.source,
            config.votes,
            config.post_texts,
            config.user_documents,
            config.user_metadata,
            config.external_scores,
            config.causal_features,
            config.nvse_scores,
        )
        if path is not None and path.is_file()
    ]
    command_names = {name for name, _ in commands}
    if "prepare-nvse-scores" in command_names and config.normvio_models is not None:
        extra_inputs.extend(
            config.normvio_models / category / "finetuned_model.pt"
            for category in (
                "spam",
                "meta-rules",
                "content",
                "harassment",
                "hatespeech",
                "format",
                "off-topic",
                "trolling",
                "incivility",
            )
        )
        extra_inputs.extend(_local_model_files(config.normvio_bert))
    if "sef" in config.methods:
        extra_inputs.extend(_local_model_files(config.embedding_model))
    _execute(
        commands,
        category="reproduce",
        manifest_path=config.results / "reproducibility" / "latest_run.json",
        splits_root=config.splits,
        dry_run=dry_run,
        continue_on_error=config.continue_on_error,
        extra_inputs=extra_inputs,
    )


@prepare_app.command("auxiliary")
def prepare_auxiliary(
    section: Annotated[
        str,
        typer.Option(
            help="post-texts, user-documents, reddit-scores, moderated-items, or all."
        ),
    ] = "all",
    input_csv: Annotated[
        Path,
        typer.Option(help="Table supplying item_id, username, and timestamp."),
    ] = DEFAULT_INPUT,
    votes: Annotated[
        Path,
        typer.Option(help="Vote table used for Reddit score acquisition."),
    ] = DEFAULT_INPUT,
    output_dir: Annotated[
        Path,
        typer.Option(help="Destination for the fetched Parquet tables and manifest."),
    ] = Path("data/interim/reddit/auxiliary"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the acquisition command without running it."),
    ] = False,
) -> None:
    """Acquire optional Arctic Shift inputs once (network access required)."""
    allowed = {"post-texts", "user-documents", "reddit-scores", "moderated-items", "all"}
    if section not in allowed:
        raise typer.BadParameter(f"section must be one of {sorted(allowed)}")
    command = _python_module(
        "src.data_preparation.fetch_reddit_auxiliary_data",
        "--section",
        section,
        "--input-csv",
        input_csv,
        "--votes",
        votes,
        "--output-dir",
        output_dir,
    )
    _execute(
        [("auxiliary", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_auxiliary_run.json"),
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("user-documents")
def prepare_user_documents(
    input_dir: Annotated[
        Path,
        typer.Option(help="Directory containing one contribution JSONL per user."),
    ] = Path("data/raw/user_contributions/user_contributions"),
    users_from: Annotated[
        Path,
        typer.Option(help="Votes table selecting users and the relevant time window."),
    ] = DEFAULT_INPUT,
    output: Annotated[
        Path,
        typer.Option(help="Destination user_documents.parquet."),
    ] = DEFAULT_USER_DOCUMENTS,
    n_user_docs: Annotated[
        int,
        typer.Option(min=2, help="Maximum documents per user, split between posts/comments."),
    ] = 150,
    download: Annotated[
        bool,
        typer.Option(
            "--download",
            help="Top up users below the local document limit with Arctic Shift.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build user documents locally; optionally top up gaps from Arctic Shift."""
    commands: list[tuple[str, list[str]]] = []
    target = output if output.is_absolute() else PROJECT_ROOT / output
    if force or not target.is_file():
        command = _python_module(
            "src.data_preparation.build_user_documents",
            "--input-dir",
            input_dir,
            "--users-from",
            users_from,
            "--output",
            output,
            "--n-user-docs",
            n_user_docs,
        )
        if force:
            command.append("--force")
        commands.append(("user-documents-local", command))
    else:
        typer.echo(f"Already prepared: {_relative(target)}")
    if download:
        commands.append(
            (
                "user-documents-arctic-shift-top-up",
                _python_module(
                    "src.data_preparation.fetch_reddit_auxiliary_data",
                    "--section",
                    "user-documents",
                    "--input-csv",
                    users_from,
                    "--output-dir",
                    output.parent,
                    "--user-documents-output",
                    output,
                    "--n-user-docs",
                    n_user_docs,
                    "--missing-user-documents-only",
                ),
            )
        )
    if not commands:
        return
    _execute(
        commands,
        category="prepare",
        manifest_path=output.parent / "prepare_user_documents_run.json",
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("dataset")
def prepare_dataset(
    input_path: Annotated[Path, typer.Option("--input")] = DEFAULT_INPUT,
    output_dir: Annotated[Path, typer.Option()] = Path("data/interim/reddit/datasets"),
    splits_dir: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    external_scores: Annotated[Path | None, typer.Option()] = None,
    post_texts: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 10,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Create the cleaned dataset and all canonical splits once."""
    marker = output_dir / "prepare_reddit_data.manifest.json"
    if _prepared(marker, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.prepare_reddit_data",
        "--input",
        input_path,
        "--output-dir",
        output_dir,
        "--splits-dir",
        splits_dir,
        "--seed",
        seed,
    )
    if external_scores is not None:
        command.extend(("--external-scores", external_scores.as_posix()))
    if post_texts is not None:
        command.extend(("--post-texts", post_texts.as_posix()))
    _execute(
        [("dataset-and-splits", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_dataset_run.json"),
        splits_root=splits_dir,
        dry_run=dry_run,
    )


@prepare_app.command("kfold")
def prepare_kfold(
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Defaults to <splits-root>/kfold."),
    ] = None,
    folds: Annotated[int, typer.Option(min=3)] = 5,
    seed: Annotated[int, typer.Option()] = 10,
    chronological_initial_train_fraction: Annotated[
        float,
        typer.Option(min=0.05, max=0.90),
    ] = 0.40,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build random and walk-forward folds from the three canonical splits."""
    destination = output_root or _scoped_root(splits_root, KFOLD_DIRNAME)
    marker = destination / "manifest.json"
    if _prepared(marker, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.prepare_kfold_splits",
        "--splits-root",
        splits_root,
        "--output-root",
        destination,
        "--n-folds",
        folds,
        "--seed",
        seed,
        "--chronological-initial-train-fraction",
        chronological_initial_train_fraction,
    )
    _execute(
        [("kfold-splits", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_kfold_run.json"),
        splits_root=destination,
        dry_run=dry_run,
    )


@prepare_app.command("user-features")
def prepare_user_features(
    input_dir: Annotated[Path, typer.Option()] = Path(
        "data/raw/user_contributions/user_contributions"
    ),
    votes: Annotated[Path, typer.Option()] = DEFAULT_INPUT,
    metadata: Annotated[Path, typer.Option()] = Path("data/processed/user_metadata.csv"),
    post_texts: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/auxiliary/post_texts.parquet"
    ),
    output: Annotated[Path, typer.Option()] = DEFAULT_CAUSAL_FEATURES,
    author_index: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/build_state/author_index.sqlite"
    ),
    feature_root: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/build_state/user_temporal_features"
    ),
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build causal, per-vote user-history features once."""
    if _prepared(output, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.build_user_history_features",
        "--stage",
        "all",
        "--input-dir",
        input_dir,
        "--votes",
        votes,
        "--metadata",
        metadata,
        "--post-texts",
        post_texts,
        "--author-index",
        author_index,
        "--feature-root",
        feature_root,
        "--output",
        output,
    )
    if force:
        command.extend(("--rebuild-index", "--rebuild-snapshots"))
    _execute(
        [("causal-user-features", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_user_features_run.json"),
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("nvse-cache")
def prepare_nvse_cache(
    votes_dir: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output: Annotated[Path, typer.Option()] = DEFAULT_NVSE_SCORES,
    post_texts: Annotated[Path, typer.Option()] = DEFAULT_POST_TEXTS,
    user_documents: Annotated[Path, typer.Option()] = DEFAULT_USER_DOCUMENTS,
    models: Annotated[Path, typer.Option()] = DEFAULT_NORMVIO_MODELS,
    bert_model: Annotated[str, typer.Option()] = DEFAULT_NORMVIO_BERT,
    work_dir: Annotated[Path, typer.Option()] = DEFAULT_NVSE_WORK,
    device: Annotated[str, typer.Option(help="cpu, cuda, or empty for auto-detect.")] = "",
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Compute the split-independent NormVio post-score cache once."""
    if _prepared(output, force) and not dry_run:
        return
    command = _python_module(
        "src.methods.normvio_skill_extraction",
        "--votes_dir",
        votes_dir,
        "--violation_scores",
        output,
        "--post_texts",
        post_texts,
        "--docs",
        user_documents,
        "--models",
        models,
        "--bert_model",
        bert_model,
        "--out_dir",
        work_dir,
        "--tasks",
        "score",
        "--device",
        device,
    )
    if force:
        command.append("--force-rescore")
    _execute(
        [("nvse-post-scores", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_nvse_cache_run.json"),
        splits_root=votes_dir,
        dry_run=dry_run,
    )


def _split_output_parent(results_root: Path, split_name: str) -> Path:
    """Return the root that makes a direct split retain its canonical result path."""

    return results_root.joinpath(*split_name.split("/")).parent


def _gpu_assignments(device: str, gpu_devices: Sequence[str]) -> list[str]:
    configured = [value.removeprefix("cuda:") for value in gpu_devices if value != ""]
    if configured:
        return configured
    if device.startswith("cuda:"):
        return [device.split(":", 1)[1]]
    if device == "cuda":
        return ["0"]
    return []


def _granular_method_jobs(
    selected: Sequence[str],
    splits_root: Path,
    results_root: Path,
    input_path: Path,
    causal_features: Path,
    nvse_scores: Path,
    nvse_work: Path,
    post_texts: Path,
    user_documents: Path,
    external_scores: Path,
    embedding_cache: Path,
    embedding_model: str,
    reuse_sef_hyperparameters: bool,
    excluded_methods: Sequence[str],
    device: str,
    gpu_devices: Sequence[str],
    skip_existing_team_methods: bool,
    n_workers: int,
) -> tuple[list[CommandJob], dict[str, int]]:
    """Build independent jobs covering every one of the 18 benchmark methods."""

    splits = discover_splits(splits_root)
    if not splits:
        raise typer.BadParameter(f"No prepared splits found below {splits_root}")
    jobs: list[CommandJob] = []
    gpus = _gpu_assignments(device, gpu_devices)
    gpu_cursor = 0

    def gpu_job(name: str, command: list[str]) -> CommandJob:
        nonlocal gpu_cursor
        if not gpus:
            return CommandJob(name, command, resource="cpu")
        gpu = gpus[gpu_cursor % len(gpus)]
        gpu_cursor += 1
        return CommandJob(
            name,
            command,
            environment={"CUDA_VISIBLE_DEVICES": gpu},
            resource=f"gpu:{gpu}",
        )

    for split_name, split_path in splits.items():
        output_parent = _split_output_parent(results_root, split_name)
        split_result = results_root.joinpath(*split_name.split("/"))

        if "baselines" in selected:
            for baseline in BASELINE_METHODS:
                jobs.append(
                    CommandJob(
                        f"{split_name}/baseline/{baseline}",
                        _python_module(
                            "src.methods.baselines",
                            "--votes-dir",
                            split_path,
                            "--output-dir",
                            output_parent,
                            "--external-scores",
                            external_scores,
                            "--baseline",
                            baseline,
                        ),
                    )
                )
        if "community-notes" in selected:
            jobs.append(
                CommandJob(
                    f"{split_name}/cn",
                    _python_module(
                        "src.methods.community_notes",
                        "--votes-dir",
                        split_path,
                        "--output-dir",
                        output_parent,
                    ),
                )
            )
        if "var" in selected:
            jobs.append(
                CommandJob(
                    f"{split_name}/var",
                    _python_module(
                        "src.methods.expert_selection.var",
                        "--votes-dir",
                        split_path,
                        "--output-dir",
                        output_parent,
                    ),
                )
            )
        if "nvse" in selected:
            jobs.append(
                CommandJob(
                    f"{split_name}/nvse",
                    _python_module(
                        "src.methods.normvio_skill_extraction",
                        "--votes_dir",
                        split_path,
                        "--violation_scores",
                        nvse_scores,
                        "--causal_features",
                        causal_features,
                        "--results_dir",
                        output_parent,
                        "--out_dir",
                        nvse_work,
                        "--tasks",
                        "splits",
                    ),
                )
            )
        if "ma-qsmf" in selected:
            for variants in MA_QSMF_BRANCHES:
                command = _python_module(
                    "src.methods.ma_qsmf",
                    "--votes-dir",
                    split_path,
                    "--output-dir",
                    output_parent,
                    "--variants",
                    *variants,
                    "--device",
                    "cuda" if gpus else device,
                )
                jobs.append(gpu_job(f"{split_name}/{'-'.join(variants)}", command))
        if "team-formation" in selected:
            excluded = set(
                nested_exclusions("team-formation", excluded_methods, NESTED_METHODS)
            )
            for method in TEAM_FORMATION_METHODS:
                if method in excluded:
                    continue
                if skip_existing_team_methods:
                    from src.methods.team_formation_v2.run_all import has_active_metrics

                    if has_active_metrics(split_result / method / "metrics.json", method):
                        continue
                command = _python_module(
                    TEAM_FORMATION_MODULES[method],
                    "--votes-dir",
                    split_path,
                    "--out-dir",
                    split_result / method,
                    "--causal-features",
                    causal_features,
                    "--embedding-dir",
                    embedding_cache,
                )
                if method != "boc_stacking":
                    command.extend(("--team-size", "3"))
                jobs.append(CommandJob(f"{split_name}/{method}", command, resource="memory"))

    # SEF deliberately remains one shared-cache producer/consumer process.  Its
    # expensive hyperparameter work is optimized internally; launching one copy
    # per split would duplicate the base vote table and temporal profile cache.
    if "sef" in selected:
        args: list[object] = [
            "--votes-dir",
            splits_root,
            "--output-dir",
            results_root,
            "--causal-features",
            causal_features,
            "--input-csv",
            input_path,
            "--post-texts",
            post_texts,
            "--user-documents",
            user_documents,
            "--embedding-dir",
            embedding_cache,
            "--embedding-model",
            embedding_model,
            "--tasks",
            "splits",
            "--n-workers",
            resolve_worker_count(n_workers),
        ]
        if reuse_sef_hyperparameters:
            args.append("--reuse-hyperparams")
        command = _python_module("src.methods.semantic_expert_finder", *args)
        jobs.append(gpu_job("sef/all-splits", command))

    resource_limits = {"memory": 1}
    resource_limits.update({f"gpu:{gpu}": 1 for gpu in gpus})
    return jobs, resource_limits


@run_app.command("methods")
def run_methods(
    method: Annotated[
        list[str] | None,
        typer.Option(
            "--method",
            "-m",
            help="Repeat for a subset; omit to run every canonical method family.",
        ),
    ] = None,
    exclude_method: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-method",
            "-x",
            help="Method name to omit, such as bandit or sef; repeat for several.",
        ),
    ] = None,
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    input_path: Annotated[Path, typer.Option("--input")] = DEFAULT_INPUT,
    causal_features: Annotated[Path, typer.Option()] = DEFAULT_CAUSAL_FEATURES,
    nvse_scores: Annotated[Path, typer.Option()] = DEFAULT_NVSE_SCORES,
    nvse_work: Annotated[Path, typer.Option()] = DEFAULT_NVSE_WORK,
    post_texts: Annotated[Path, typer.Option()] = DEFAULT_POST_TEXTS,
    user_documents: Annotated[Path, typer.Option()] = DEFAULT_USER_DOCUMENTS,
    external_scores: Annotated[Path, typer.Option()] = DEFAULT_EXTERNAL_SCORES,
    embedding_cache: Annotated[Path, typer.Option()] = DEFAULT_EMBEDDING_CACHE,
    embedding_model: Annotated[str, typer.Option()] = DEFAULT_EMBEDDING_MODEL,
    device: Annotated[
        str,
        typer.Option(help="Device for MA-QSMF: cpu, cuda, or cuda:N."),
    ] = "cpu",
    gpu_device: Annotated[
        list[str] | None,
        typer.Option(
            "--gpu-device",
            help="GPU index for resource-aware scheduling; repeat for multiple GPUs.",
        ),
    ] = None,
    k_fold: Annotated[
        bool,
        typer.Option(
            "--k-fold",
            help="Run only the prepared folds under <splits-root>/kfold and write under "
            "<results-root>/kfold.",
        ),
    ] = False,
    reuse_sef_hyperparameters: Annotated[
        bool,
        typer.Option(help="Reuse each split's previously VAL-selected SEF configuration."),
    ] = False,
    skip_existing_team_methods: Annotated[
        bool,
        typer.Option(help="Reuse valid Team Formation v2 outputs."),
    ] = False,
    n_workers: Annotated[
        int,
        typer.Option(
            min=0,
            help="Concurrent method/split processes; 0 chooses a conservative automatic value.",
        ),
    ] = 1,
    threads_per_worker: Annotated[
        int,
        typer.Option(min=1, help="Native BLAS/OpenMP threads available to each process."),
    ] = 1,
    memory_workers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum concurrent memory-heavy Team Formation jobs.",
        ),
    ] = 1,
    continue_on_error: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Fit/evaluate selected methods on every discovered canonical split."""
    requested = method or ["all"]
    try:
        selected, normalized_exclusions = resolve_method_selection(
            requested,
            exclude_method or [],
            valid_methods=tuple(METHOD_MODULES),
            nested_methods=NESTED_METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
    evaluation_splits_root = _scoped_root(splits_root, KFOLD_DIRNAME) if k_fold else splits_root
    evaluation_results_root = _scoped_root(results_root, KFOLD_DIRNAME) if k_fold else results_root
    if k_fold and not (evaluation_splits_root / "manifest.json").exists() and not dry_run:
        raise typer.BadParameter(
            f"K-fold splits are not prepared at {evaluation_splits_root}; run "
            "`python -m src.pipeline prepare kfold` first.",
            param_hint="--k-fold",
        )
    commands, resource_limits = _granular_method_jobs(
        selected,
        evaluation_splits_root,
        evaluation_results_root,
        input_path,
        causal_features,
        nvse_scores,
        nvse_work,
        post_texts,
        user_documents,
        external_scores,
        embedding_cache,
        embedding_model,
        reuse_sef_hyperparameters,
        normalized_exclusions,
        device,
        gpu_device or [],
        skip_existing_team_methods,
        n_workers,
    )
    resource_limits["memory"] = memory_workers
    manifest_path = manifest or Path(
        "results/reproducibility/"
        + ("latest_kfold_methods_run.json" if k_fold else "latest_methods_run.json")
    )
    _execute(
        commands,
        category="methods",
        manifest_path=manifest_path,
        splits_root=evaluation_splits_root,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        resource_limits=resource_limits,
    )
    if k_fold and not dry_run:
        _execute(
            [
                (
                    "kfold-graphs",
                    _python_module(
                        "src.comparison.graphs",
                        "--k-fold-only",
                        "--splits-root",
                        evaluation_splits_root,
                        "--results-root",
                        evaluation_results_root,
                        "--output-dir",
                        DEFAULT_REPORT / "kfold",
                    ),
                )
            ],
            category="kfold-graphs",
            manifest_path=Path("results/reproducibility/latest_kfold_graphs_run.json"),
            splits_root=evaluation_splits_root,
            dry_run=False,
        )


@run_app.command("kfold-graphs")
def run_kfold_graphs(
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path, typer.Option()] = Path(
        "results/reproducibility/latest_kfold_graphs_run.json"
    ),
) -> None:
    """Generate method-comparison graphs with mean and SD across K-fold runs."""
    evaluation_splits_root = _scoped_root(splits_root, KFOLD_DIRNAME)
    evaluation_results_root = _scoped_root(results_root, KFOLD_DIRNAME)
    destination = output_dir or DEFAULT_REPORT / "kfold"
    command = _python_module(
        "src.comparison.graphs",
        "--k-fold-only",
        "--splits-root",
        evaluation_splits_root,
        "--results-root",
        evaluation_results_root,
        "--output-dir",
        destination,
    )
    _execute(
        [("kfold-graphs", command)],
        category="report",
        manifest_path=manifest,
        splits_root=evaluation_splits_root,
        dry_run=dry_run,
    )


@run_app.command("graphs")
def run_graphs(
    section: Annotated[
        list[str] | None,
        typer.Option(
            "--section",
            help="Repeat to select sections; omit for the complete report.",
        ),
    ] = None,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_REPORT,
    strict_audit: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path, typer.Option()] = Path(
        "results/reproducibility/latest_graphs_run.json"
    ),
) -> None:
    """Regenerate graphs/tables from all discovered split and method outputs."""
    arguments: list[object] = [
        "--results-root",
        results_root,
        "--splits-root",
        splits_root,
        "--output-dir",
        output_dir,
    ]
    for value in section or ["all"]:
        arguments.extend(("--section", value))
    if strict_audit:
        arguments.append("--strict-audit")
    _execute(
        [("graphs", _python_module("src.comparison.graphs", *arguments))],
        category="graphs",
        manifest_path=manifest,
        splits_root=splits_root,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
