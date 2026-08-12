"""Run every team-formation method on one prepared split."""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection"
DEFAULT_OUT_ROOT = "results/reddit/intersection"

METHODS = ["bandit_dr", "graph_propagation", "virtual_ensembles", "boc_stacking"]


def main():
    """Run each method as a subprocess and report its exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary = {}

    for method in METHODS:
        out_dir = str(Path(args.out_root) / method)
        print(f"\nRunning {method}")
        cmd = [
            sys.executable, str(script_dir / f"{method}.py"),
            "--votes-dir", args.votes_dir,
            "--out-dir", out_dir,
        ]
        result = subprocess.run(cmd)
        summary[method] = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"

    print("\nSummary")
    for method, status in summary.items():
        print(f"{method}: {status}")


if __name__ == "__main__":
    main()
