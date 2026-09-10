"""Configuration and fail-fast checks for one-shot reproducible runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.splits import discover_splits
from src.utils.tabular import read_table, table_columns

VOTE_COLUMNS = {"username", "community", "item_id", "timestamp", "vote", "label"}
CAUSAL_COLUMNS = {
    "username",
    "item_id",
    "timestamp",
    "community",
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
}
NORMVIO_CATEGORIES = (
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
NVSE_COLUMNS = {
    "item_id",
    "community",
    "top_violation_category",
    "top_violation_score",
    *(f"score_{category}" for category in NORMVIO_CATEGORIES),
}


@dataclass(frozen=True)
class ReproductionConfig:
    """Normalized paths and switches read from one JSON configuration."""

    source: Path | None
    dataset: str
    votes: Path
    post_texts: Path | None
    user_documents: Path | None
    user_contributions: Path | None
    user_metadata: Path | None
    external_scores: Path | None
    normvio_models: Path | None
    normvio_bert: str
    embedding_model: str
    interim: Path
    splits: Path
    results: Path
    report: Path
    cache: Path
    causal_features: Path
    nvse_scores: Path
    embedding_cache: Path
    causal_features_are_precomputed: bool
    nvse_scores_are_precomputed: bool
    methods: tuple[str, ...]
    prepare_dataset: bool
    prepare_user_features: bool
    prepare_nvse_scores: bool
    download_user_documents: bool
    n_user_documents: int
    graphs: bool
    kfold: bool
    force: bool
    continue_on_error: bool
    device: str
    seed: int
    reuse_sef_hyperparameters: bool
    skip_existing_team_methods: bool
    graph_sections: tuple[str, ...]


def _check_keys(payload: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown {section} key(s): {sorted(unknown)}")


def _path(project_root: Path, value: Any, *, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string or null")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _required_path(project_root: Path, value: Any, *, field: str) -> Path:
    path = _path(project_root, value, field=field)
    if path is None:
        raise ValueError(f"{field} is required")
    return path


def _boolean(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"run.{key} must be true or false")
    return value


def _integer(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"run.{key} must be an integer")
    return value


def load_reproduction_config(
    config_path: Path,
    project_root: Path,
    valid_methods: Iterable[str],
) -> ReproductionConfig:
    """Load a strict versioned JSON configuration and resolve its paths."""
    source = config_path if config_path.is_absolute() else project_root / config_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The reproduction config must be one JSON object")
    _check_keys(payload, {"schema_version", "dataset", "inputs", "outputs", "run"}, "top-level")
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", dataset):
        raise ValueError("dataset must contain only letters, digits, '.', '_' or '-'")

    inputs = payload.get("inputs", {})
    outputs = payload.get("outputs", {})
    run = payload.get("run", {})
    if not all(isinstance(section, dict) for section in (inputs, outputs, run)):
        raise ValueError("inputs, outputs, and run must be JSON objects")
    _check_keys(
        inputs,
        {
            "votes",
            "post_texts",
            "user_documents",
            "user_contributions",
            "user_metadata",
            "external_scores",
            "normvio_models",
            "normvio_bert",
            "embedding_model",
            "causal_features",
            "nvse_scores",
            "embedding_cache",
        },
        "inputs",
    )
    _check_keys(outputs, {"interim", "splits", "results", "report", "cache"}, "outputs")
    _check_keys(
        run,
        {
            "methods",
            "prepare_dataset",
            "prepare_user_features",
            "prepare_nvse_scores",
            "download_user_documents",
            "n_user_documents",
            "graphs",
            "kfold",
            "force",
            "continue_on_error",
            "device",
            "seed",
            "reuse_sef_hyperparameters",
            "skip_existing_team_methods",
            "graph_sections",
        },
        "run",
    )

    interim = _required_path(
        project_root,
        outputs.get("interim", f"data/interim/{dataset}"),
        field="outputs.interim",
    )
    cache = _required_path(
        project_root, outputs.get("cache", f"cache/{dataset}"), field="outputs.cache"
    )
    raw_methods = run.get("methods", ["all"])
    if (
        not isinstance(raw_methods, list)
        or not raw_methods
        or not all(isinstance(value, str) for value in raw_methods)
    ):
        raise ValueError("run.methods must be a non-empty JSON array of strings")
    valid = set(valid_methods)
    if raw_methods == ["all"]:
        methods = tuple(valid_methods)
    else:
        unknown = set(raw_methods) - valid
        if unknown:
            raise ValueError(f"Unknown run.methods value(s): {sorted(unknown)}")
        if "all" in raw_methods:
            raise ValueError("'all' cannot be combined with individual methods")
        requested = set(raw_methods)
        methods = tuple(method for method in valid_methods if method in requested)

    causal = _path(project_root, inputs.get("causal_features"), field="inputs.causal_features")
    nvse = _path(project_root, inputs.get("nvse_scores"), field="inputs.nvse_scores")
    embedding = _path(project_root, inputs.get("embedding_cache"), field="inputs.embedding_cache")
    sections = run.get("graph_sections", ["all"])
    if (
        not isinstance(sections, list)
        or not sections
        or not all(isinstance(value, str) for value in sections)
    ):
        raise ValueError("run.graph_sections must be a non-empty JSON array of strings")
    normvio_bert = inputs.get("normvio_bert", "DeepPavlov/bert-base-cased-conversational")
    embedding_model = inputs.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if not isinstance(normvio_bert, str) or not normvio_bert.strip():
        raise ValueError("inputs.normvio_bert must be a non-empty model id/path string")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("inputs.embedding_model must be a non-empty model id/path string")
    device = run.get("device", "")
    if not isinstance(device, str):
        raise ValueError("run.device must be a string")
    n_user_documents = _integer(run, "n_user_documents", 150)
    if n_user_documents < 2:
        raise ValueError("run.n_user_documents must be at least 2")

    return ReproductionConfig(
        source=source,
        dataset=dataset,
        votes=_required_path(project_root, inputs.get("votes"), field="inputs.votes"),
        post_texts=_path(project_root, inputs.get("post_texts"), field="inputs.post_texts"),
        user_documents=_path(
            project_root, inputs.get("user_documents"), field="inputs.user_documents"
        ),
        user_contributions=_path(
            project_root, inputs.get("user_contributions"), field="inputs.user_contributions"
        ),
        user_metadata=_path(
            project_root, inputs.get("user_metadata"), field="inputs.user_metadata"
        ),
        external_scores=_path(
            project_root, inputs.get("external_scores"), field="inputs.external_scores"
        ),
        normvio_models=_path(
            project_root, inputs.get("normvio_models"), field="inputs.normvio_models"
        ),
        normvio_bert=normvio_bert,
        embedding_model=embedding_model,
        interim=interim,
        splits=_required_path(
            project_root, outputs.get("splits", f"data/splits/{dataset}"), field="outputs.splits"
        ),
        results=_required_path(
            project_root, outputs.get("results", f"results/{dataset}"), field="outputs.results"
        ),
        report=_required_path(
            project_root,
            outputs.get("report", f"results/report/{dataset}"),
            field="outputs.report",
        ),
        cache=cache,
        causal_features=causal or interim / "features" / "causal_user_vote_features.parquet",
        nvse_scores=nvse or cache / "nvse" / "post_violation_scores.parquet",
        embedding_cache=embedding or cache / "embeddings",
        causal_features_are_precomputed=causal is not None,
        nvse_scores_are_precomputed=nvse is not None,
        methods=methods,
        prepare_dataset=_boolean(run, "prepare_dataset", True),
        prepare_user_features=_boolean(run, "prepare_user_features", True),
        prepare_nvse_scores=_boolean(run, "prepare_nvse_scores", True),
        download_user_documents=_boolean(run, "download_user_documents", False),
        n_user_documents=n_user_documents,
        graphs=_boolean(run, "graphs", True),
        kfold=_boolean(run, "kfold", False),
        force=_boolean(run, "force", False),
        continue_on_error=_boolean(run, "continue_on_error", False),
        device=device,
        seed=_integer(run, "seed", 10),
        reuse_sef_hyperparameters=_boolean(run, "reuse_sef_hyperparameters", False),
        skip_existing_team_methods=_boolean(run, "skip_existing_team_methods", False),
        graph_sections=tuple(sections),
    )


def _validate_table(
    path: Path | None,
    required: set[str],
    label: str,
    errors: list[str],
) -> None:
    if path is None or not path.is_file():
        errors.append(f"{label}: missing file ({path})")
        return
    try:
        columns = set(table_columns(path))
    except Exception as exc:
        errors.append(f"{label}: cannot read {path}: {exc}")
        return
    missing = required - columns
    if missing:
        errors.append(f"{label}: {path} is missing {sorted(missing)}")


def _validate_vote_sample(path: Path, errors: list[str]) -> None:
    """Check value-level invariants on a bounded source-table sample."""
    try:
        sample = read_table(path, columns=VOTE_COLUMNS, nrows=2_000)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"inputs.votes: cannot sample {path}: {exc}")
        return
    if sample.empty:
        errors.append(f"inputs.votes is empty: {path}")
        return
    if sample[list(VOTE_COLUMNS)].isna().any(axis=None):
        errors.append("inputs.votes sample contains null core values")
    for column in ("vote", "label"):
        values = np.asarray(sample[column])
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        if np.isnan(numeric).any() or not set(np.unique(numeric)).issubset({-1.0, 1.0}):
            errors.append(f"inputs.votes.{column} must contain only -1 or +1")
    timestamps = pd.to_numeric(sample["timestamp"], errors="coerce")
    if timestamps.isna().any():
        errors.append("inputs.votes.timestamp must be numeric Unix seconds")


def _model_is_local_or_cached(model: str, project_root: Path) -> bool:
    candidate = Path(model)
    candidate = candidate if candidate.is_absolute() else project_root / candidate
    if candidate.is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False


def _model_is_explicit_local_path(model: str, project_root: Path) -> bool:
    candidate = Path(model)
    candidate = candidate if candidate.is_absolute() else project_root / candidate
    return candidate.is_dir()


def _validate_contribution_jsonl(input_dir: Path, errors: list[str]) -> None:
    """Validate a bounded sample of the per-user JSONL contract."""
    sampled = 0
    for path in sorted(input_dir.glob("*.jsonl"))[:3]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Invalid JSON in {path}: {exc}")
                    break
                if not isinstance(record, dict):
                    errors.append(f"Expected JSON objects in {path}")
                    break
                missing = {"subreddit", "created_utc"} - set(record)
                if "id" not in record and "name" not in record:
                    missing.add("id or name")
                is_comment = "body" in record and "parent_id" in record
                is_post = "title" in record
                if not is_comment and not is_post:
                    missing.add("comment(body,parent_id) or post(title)")
                if missing:
                    errors.append(f"Contribution sample {path} is missing {sorted(missing)}")
                sampled += 1
                break
    if not sampled:
        errors.append(f"No non-empty contribution JSONL record found in {input_dir}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reproduction_config(
    config: ReproductionConfig,
    project_root: Path,
) -> tuple[list[str], list[str]]:
    """Return all fail-fast errors and non-fatal reproducibility warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    packages = {"numpy", "pandas", "pyarrow", "sklearn", "scipy", "statsmodels"}
    if "nvse" in config.methods:
        packages.update({"torch", "transformers", "huggingface_hub"})
    if "sef" in config.methods:
        packages.update({"faiss", "sentence_transformers", "xgboost"})
    if "team-formation" in config.methods:
        packages.add("xgboost")
    if "community-notes" in config.methods:
        packages.add("wandb")
    if config.graphs:
        packages.update({"matplotlib", "seaborn"})
    missing_packages = sorted(name for name in packages if importlib.util.find_spec(name) is None)
    if missing_packages:
        errors.append(
            "Missing Python package(s): "
            + ", ".join(missing_packages)
            + "; install requirements.txt in the interpreter launching Typer"
        )
    if config.device and not re.fullmatch(r"cpu|cuda(?::\d+)?", config.device):
        errors.append("run.device must be empty, 'cpu', 'cuda', or 'cuda:<index>'")
    if config.device.startswith("cuda") and importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            errors.append(f"run.device={config.device!r}, but PyTorch cannot access CUDA")

    _validate_table(config.votes, VOTE_COLUMNS, "inputs.votes", errors)
    if config.votes.is_file():
        _validate_vote_sample(config.votes, errors)

    dataset_marker = config.interim / "datasets" / "prepare_reddit_data.manifest.json"
    if config.prepare_dataset and dataset_marker.is_file() and not config.force:
        try:
            dataset_manifest = json.loads(dataset_marker.read_text(encoding="utf-8"))
            recorded = dataset_manifest.get("input")
            if recorded:
                recorded_path = Path(recorded)
                recorded_path = (
                    recorded_path if recorded_path.is_absolute() else project_root / recorded_path
                )
                if recorded_path.resolve() != config.votes.resolve():
                    errors.append(
                        f"Prepared dataset at {config.interim} belongs to {recorded_path}, "
                        f"not {config.votes}; choose dataset-specific outputs or set "
                        "run.force=true"
                    )
            recorded_hash = dataset_manifest.get("input_sha256")
            if recorded_hash:
                if _sha256_file(config.votes) != recorded_hash:
                    errors.append(
                        f"Configured votes changed after splits were prepared: {config.votes}; "
                        "set run.force=true to rebuild"
                    )
            else:
                warnings.append(
                    f"Legacy dataset manifest has no source hash: {dataset_marker}; "
                    "the existing split/source identity cannot be proven until rebuilt"
                )
            optional_manifest = dataset_manifest.get("optional_inputs", {})
            for key, current_path in (
                ("moderated_posts_scores", config.external_scores),
                ("post_texts", config.post_texts),
            ):
                recorded_optional = optional_manifest.get(key, {})
                recorded_optional_hash = recorded_optional.get("sha256")
                if current_path is not None and current_path.is_file() and recorded_optional_hash:
                    if _sha256_file(current_path) != recorded_optional_hash:
                        errors.append(
                            f"{key} changed after the intersection split was prepared: "
                            f"{current_path}; set run.force=true to rebuild"
                        )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"Cannot validate existing dataset manifest {dataset_marker}: {exc}")

    splits_ready = bool(discover_splits(config.splits))
    if not config.prepare_dataset and not splits_ready:
        errors.append(f"Prepared splits are missing under {config.splits}")

    needs_causal = bool({"nvse", "sef", "team-formation"} & set(config.methods))
    will_build_causal = (
        needs_causal
        and not config.causal_features_are_precomputed
        and config.prepare_user_features
        and (config.force or not config.causal_features.is_file())
    )
    if config.causal_features_are_precomputed and not config.causal_features.is_file():
        errors.append(
            f"Configured precomputed causal features are missing: {config.causal_features}"
        )
    if needs_causal and not config.causal_features.is_file() and not will_build_causal:
        errors.append(f"Causal feature table is missing: {config.causal_features}")
    if will_build_causal:
        if config.user_contributions is None or not config.user_contributions.is_dir():
            errors.append(
                f"inputs.user_contributions must be a directory of per-user JSONL files "
                f"to build causal features ({config.user_contributions})"
            )
        elif not next(config.user_contributions.glob("*.jsonl"), None):
            errors.append(f"No *.jsonl files found in {config.user_contributions}")
        else:
            _validate_contribution_jsonl(config.user_contributions, errors)
        if config.user_metadata is None or not config.user_metadata.is_file():
            warnings.append("User metadata is absent; account-tenure features will be missing")
        else:
            _validate_table(
                config.user_metadata,
                {"username", "account_created_utc"},
                "inputs.user_metadata",
                errors,
            )

    needs_post_texts = "sef" in config.methods or (
        "nvse" in config.methods
        and not config.nvse_scores_are_precomputed
        and config.prepare_nvse_scores
        and (config.force or not config.nvse_scores.is_file())
    )
    if needs_post_texts:
        required = {"item_id", "text"} if "sef" in config.methods else {"item_id"}
        if (
            "nvse" in config.methods
            and not config.nvse_scores_are_precomputed
            and (config.force or not config.nvse_scores.is_file())
        ):
            required |= {"title", "selftext"}
        _validate_table(config.post_texts, required, "inputs.post_texts", errors)

    if "sef" in config.methods:
        user_documents_path = (
            config.user_documents
            if config.user_documents is not None
            else config.interim / "auxiliary" / "user_documents.parquet"
        )
        user_documents_ready = user_documents_path.is_file()
        can_build_user_documents = (
            config.user_contributions is not None
            and config.user_contributions.is_dir()
            and next(config.user_contributions.glob("*.jsonl"), None) is not None
        )
        if user_documents_ready:
            _validate_table(
                user_documents_path,
                {"username", "created_utc", "text"},
                "inputs.user_documents",
                errors,
            )
        elif can_build_user_documents:
            _validate_contribution_jsonl(config.user_contributions, errors)
        elif not can_build_user_documents and not config.download_user_documents:
            errors.append(
                "inputs.user_documents is missing and cannot be built: provide "
                "inputs.user_contributions or enable run.download_user_documents"
            )
        if not _model_is_local_or_cached(config.embedding_model, project_root):
            warnings.append(
                f"Embedding model will require network download: {config.embedding_model}"
            )
        if not _model_is_explicit_local_path(config.embedding_model, project_root):
            warnings.append(
                "For archival reproducibility, replace inputs.embedding_model with an exact "
                "local Hugging Face snapshot directory"
            )

    needs_nvse_score_build = (
        "nvse" in config.methods
        and not config.nvse_scores_are_precomputed
        and config.prepare_nvse_scores
        and (config.force or not config.nvse_scores.is_file())
    )
    if config.nvse_scores_are_precomputed and not config.nvse_scores.is_file():
        errors.append(f"Configured precomputed NVSE scores are missing: {config.nvse_scores}")
    if (
        "nvse" in config.methods
        and not config.nvse_scores.is_file()
        and not needs_nvse_score_build
    ):
        errors.append(f"NVSE score cache is missing: {config.nvse_scores}")
    if needs_nvse_score_build:
        if config.normvio_models is None:
            errors.append("inputs.normvio_models is required to build NVSE scores")
        else:
            for category in NORMVIO_CATEGORIES:
                checkpoint = config.normvio_models / category / "finetuned_model.pt"
                if not checkpoint.is_file():
                    errors.append(f"NormVio checkpoint missing: {checkpoint}")
        if not _model_is_local_or_cached(config.normvio_bert, project_root):
            warnings.append(f"NormVio BERT will require network download: {config.normvio_bert}")
        if not _model_is_explicit_local_path(config.normvio_bert, project_root):
            warnings.append(
                "For archival reproducibility, replace inputs.normvio_bert with an exact "
                "local Hugging Face snapshot directory"
            )

    if config.causal_features.is_file():
        _validate_table(config.causal_features, CAUSAL_COLUMNS, "causal feature cache", errors)
    if config.nvse_scores.is_file():
        _validate_table(config.nvse_scores, NVSE_COLUMNS, "NVSE score cache", errors)

    if "team-formation" in config.methods and "sef" not in config.methods:
        ids_path = config.embedding_cache / "post_ids.json"
        embeddings_path = config.embedding_cache / "post_embeddings.npy"
        if not ids_path.is_file() or not embeddings_path.is_file():
            errors.append(
                "Team Formation without SEF requires precomputed post_ids.json and "
                f"post_embeddings.npy under {config.embedding_cache}"
            )
        else:
            try:
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                matrix = np.load(embeddings_path, mmap_mode="r")
                if matrix.ndim != 2 or len(ids) != len(matrix):
                    errors.append(
                        f"Invalid/misaligned embedding cache under {config.embedding_cache}"
                    )
            except Exception as exc:
                errors.append(f"Cannot validate embedding cache {config.embedding_cache}: {exc}")

    if "baselines" in config.methods:
        if config.external_scores is None or not config.external_scores.is_file():
            errors.append(
                "inputs.external_scores is required when the complete baselines family is selected"
            )
        else:
            _validate_table(
                config.external_scores,
                {"item_id", "score"},
                "inputs.external_scores",
                errors,
            )

    if "community-notes" in config.methods:
        scoring_root = project_root / "external/community-notes/scoring/src"
        if not scoring_root.is_dir():
            errors.append(f"Community Notes source tree is missing: {scoring_root}")

    if config.seed != 10:
        errors.append(
            "run.seed must be 10: several scientific runners intentionally fix their "
            "internal random_state to 10"
        )
    if config.kfold and config.reuse_sef_hyperparameters and "sef" in config.methods:
        errors.append("SEF hyperparameter reuse is not supported for K-fold evaluation")
    return errors, warnings
