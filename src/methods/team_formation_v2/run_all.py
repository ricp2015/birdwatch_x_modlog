"""Run every active counterfactual team-formation method chronologically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_preparation.interim_paths import CAUSAL_USER_VOTE_FEATURES  # noqa: E402

CHRONOLOGICAL_SPLIT = "intersection_chronological"
DEFAULT_VOTES_DIR = f"data/splits/reddit/{CHRONOLOGICAL_SPLIT}"
DEFAULT_OUT_ROOT = "results/reddit"
DEFAULT_CAUSAL_FEATURES = str(CAUSAL_USER_VOTE_FEATURES)
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
    """Persist progress so a long chronological run can be inspected or resumed."""
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


def resolve_chronological_split(votes_path: Path) -> Path:
    """Resolve the sole valid pre-case split accepted by this runner."""
    direct = all(
        (votes_path / f"{partition}_votes.parquet").exists()
        for partition in ("train", "val", "test")
    )
    split_path = votes_path if direct else votes_path / CHRONOLOGICAL_SPLIT
    if split_path.name != CHRONOLOGICAL_SPLIT or not all(
        (split_path / f"{partition}_votes.parquet").exists()
        for partition in ("train", "val", "test")
    ):
        raise FileNotFoundError(
            "Team-formation-v2 requires the complete "
            f"{CHRONOLOGICAL_SPLIT} split; received {votes_path}"
        )
    return split_path


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
    """Run every active method on the strictly ordered pre-case split."""
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
        help="Print the chronological execution matrix without starting subprocesses.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a method when its active chronological metrics.json already exists.",
    )
    args = parser.parse_args()

    split_path = resolve_chronological_split(Path(args.votes_dir))
    out_root = Path(args.out_root)
    print(f"Chronological split: {split_path}")

    summary_filename = "tfr_v2_run_plan.json" if args.dry_run else "tfr_v2_run_summary.json"
    summary_path = out_root / summary_filename
    summary: dict[str, str] = {}
    for method in args.methods:
        out_dir = out_root / CHRONOLOGICAL_SPLIT / method
        run_name = f"{CHRONOLOGICAL_SPLIT}/{method}"
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
        write_summary(summary_path, summary)

    print("Summary")
    for run_name, status in summary.items():
        print(f"{run_name}: {status}")

    write_summary(summary_path, summary)
    failures = {
        name: status
        for name, status in summary.items()
        if status.startswith("FAILED")
    }
    if failures:
        print(f"Failed runs: {len(failures)} | details={summary_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
