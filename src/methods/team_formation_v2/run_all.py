"""Run every active counterfactual team-formation method on prepared splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.team_formation_v2.shared_features import load_split_manifest  # noqa: E402
from src.utils.splits import discover_splits  # noqa: E402

CHRONOLOGICAL_SPLIT = "intersection_chronological"
DEFAULT_VOTES_DIR = "data/splits/reddit"
DEFAULT_OUT_ROOT = "results/reddit"
DEFAULT_CAUSAL_FEATURES = "data/interim/reddit/features/causal_user_vote_features.parquet"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"
DEFAULT_TEAM_SIZE = 3

METHODS = {
    "bandit": "src.methods.team_formation_v2.bandit.pipeline",
    "graph_propagation": "src.methods.team_formation_v2.graph_propagation.pipeline",
    "virtual_ensembles": "src.methods.team_formation_v2.virtual_ensembles.pipeline",
    "boc_stacking": "src.methods.team_formation_v2.boc_stacking.pipeline",
    "role_nsga2": "src.methods.team_formation_v2.role_nsga2.pipeline",
}
PANEL_METHODS = frozenset(METHODS) - {"boc_stacking"}


def write_summary(path: Path, summary: dict[str, str]) -> None:
    """Persist progress so a long multi-split run can be inspected or resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _configured_team_size(payload: dict[str, object]) -> int | None:
    for section in ("protocol", "ensemble", "composition"):
        values = payload.get(section)
        if isinstance(values, dict) and isinstance(values.get("team_size"), int):
            return int(values["team_size"])
    return None


def has_active_metrics(path: Path, method: str) -> bool:
    """Recognize only outputs produced under the simulated-team protocol."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    if not isinstance(protocol, dict) or protocol.get("simulated_estimand") is not True:
        return False
    return method not in PANEL_METHODS or _configured_team_size(payload) == DEFAULT_TEAM_SIZE


def resolve_splits(votes_path: Path) -> dict[str, Path]:
    """Resolve one direct prepared split or every canonical split below a root."""
    splits = discover_splits(votes_path)
    direct = all(
        (votes_path / f"{part}_votes.parquet").exists() for part in ("train", "val", "test")
    )
    if (
        direct
        and votes_path.parent.name in {"full", "intersection"}
        and votes_path.parent.parent.name == "windows"
    ):
        splits = {f"windows/{votes_path.parent.name}/{votes_path.name}": votes_path}
    if not splits:
        raise FileNotFoundError(f"No complete prepared split found under {votes_path}")
    incomplete = {
        name: [
            part
            for part in ("train", "val", "test")
            if not (path / f"{part}_votes.parquet").exists()
        ]
        for name, path in splits.items()
    }
    incomplete = {name: parts for name, parts in incomplete.items() if parts}
    if incomplete:
        raise FileNotFoundError(f"Incomplete prepared splits: {incomplete}")
    return splits


def resolve_chronological_split(votes_path: Path) -> Path:
    """Backward-compatible resolver used by the chronological K-sensitivity runner."""
    splits = resolve_splits(votes_path)
    if CHRONOLOGICAL_SPLIT in splits:
        return splits[CHRONOLOGICAL_SPLIT]
    if len(splits) == 1:
        name, path = next(iter(splits.items()))
        if name == CHRONOLOGICAL_SPLIT:
            return path
    raise FileNotFoundError(f"{CHRONOLOGICAL_SPLIT} was not found under {votes_path}")


def annotate_outer_protocol(metrics_path: Path, split_name: str, split_path: Path) -> None:
    """Record whether an output has a prospective or seeded-item outer protocol."""
    if not metrics_path.exists():
        return
    manifest = load_split_manifest(split_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    split_type = manifest.get("split_type", "unknown")
    cv_protocol = manifest.get("cv_protocol")
    checks = manifest.get("checks", {})
    chronological = (
        split_type == "chronological" or cv_protocol == "expanding_window_chronological"
    )
    prospective = chronological and checks.get("strict_temporal_order") is True
    payload["outer_split_protocol"] = {
        "split": split_name,
        "split_type": split_type,
        "cv_protocol": cv_protocol,
        "strict_temporal_order": bool(prospective),
        "prospective_interpretation": bool(prospective),
        "note": (
            "Outer TRAIN strictly precedes VAL/TEST."
            if prospective
            else "Comparative seeded-item/window benchmark; the internal TRAIN prefix is "
            "chronological, but outer TRAIN need not precede VAL/TEST."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_command(
    method: str,
    split_path: Path,
    out_dir: Path,
    causal_features: str,
    embedding_dir: str,
) -> list[str]:
    """Build one module command with the shared counterfactual protocol inputs."""
    command = [
        sys.executable,
        "-B",
        "-m",
        METHODS[method],
        "--votes-dir",
        str(split_path),
        "--out-dir",
        str(out_dir),
        "--causal-features",
        causal_features,
        "--embedding-dir",
        embedding_dir,
    ]
    if method in PANEL_METHODS:
        command.extend(("--team-size", str(DEFAULT_TEAM_SIZE)))
    return command


def main() -> None:
    """Run every active method on all discovered prepared splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--causal-features", default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
        help="Active methods to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split-by-method execution matrix without starting subprocesses.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a method when its active metrics.json already exists for that split.",
    )
    args = parser.parse_args()

    splits = resolve_splits(Path(args.votes_dir))
    out_root = Path(args.out_root)
    print(f"Prepared splits: {len(splits)}")

    summary_filename = "team_formation_plan.json" if args.dry_run else "team_formation.json"
    summary_path = out_root / "summaries" / summary_filename
    summary: dict[str, str] = {}
    for split_name, split_path in splits.items():
        for method in args.methods:
            out_dir = out_root.joinpath(*split_name.split("/"), method)
            run_name = f"{split_name}/{method}"
            command = build_command(
                method,
                split_path,
                out_dir,
                args.causal_features,
                args.embedding_dir,
            )
            print(f"Running {run_name}")
            if args.skip_existing and has_active_metrics(out_dir / "metrics.json", method):
                summary[run_name] = "SKIPPED (active metrics.json exists)"
                print(summary[run_name])
                write_summary(summary_path, summary)
                continue
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                summary[run_name] = "DRY RUN"
                write_summary(summary_path, summary)
                continue
            result = subprocess.run(command, cwd=_PROJECT_ROOT, check=False)
            summary[run_name] = (
                "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            )
            if result.returncode == 0:
                annotate_outer_protocol(out_dir / "metrics.json", split_name, split_path)
            write_summary(summary_path, summary)

    print("Summary")
    for run_name, status in summary.items():
        print(f"{run_name}: {status}")

    write_summary(summary_path, summary)
    failures = {name: status for name, status in summary.items() if status.startswith("FAILED")}
    if failures:
        print(f"Failed runs: {len(failures)} | details={summary_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
