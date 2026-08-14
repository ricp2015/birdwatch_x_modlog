"""Run every team-formation method on every discovered prepared split."""

import argparse
from pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.splits import discover_splits  # noqa: E402

DEFAULT_VOTES_DIR = "data/splits/reddit"
DEFAULT_OUT_ROOT = "results/reddit"
DEFAULT_CAUSAL_FEATURES = "data/interim/reddit/causal_user_vote_features.parquet"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"

METHODS = ["bandit_dr", "graph_propagation", "virtual_ensembles", "boc_stacking"]


def main():
    """Run every new method on every prepared split and report exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--causal-features", default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    args = parser.parse_args()

    votes_root = Path(args.votes_dir)
    out_root = Path(args.out_root)
    splits = discover_splits(votes_root)
    if not splits:
        raise FileNotFoundError(
            f"No prepared splits found under {votes_root}. Expected fixed or windowed splits."
        )

    print(f"Discovered {len(splits)} split(s):")
    for split_name, split_path in splits.items():
        print(f"  {split_name}: {split_path}")

    script_dir = Path(__file__).resolve().parent
    summary = {}

    for split_name, split_path in splits.items():
        for method in METHODS:
            out_dir = out_root / split_name / method
            run_name = f"{split_name}/{method}"
            print(f"\nRunning {run_name}")
            cmd = [
                sys.executable,
                str(script_dir / f"{method}.py"),
                "--votes-dir",
                str(split_path),
                "--out-dir",
                str(out_dir),
                "--causal-features",
                args.causal_features,
            ]
            if method == "bandit_dr":
                cmd.extend(["--embedding-dir", args.embedding_dir])
            result = subprocess.run(cmd)
            summary[run_name] = (
                "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            )

    print("\nSummary")
    for method, status in summary.items():
        print(f"{method}: {status}")


if __name__ == "__main__":
    main()
