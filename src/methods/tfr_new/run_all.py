"""
run_all_methods.py — lancia bandit_dr, graph_propagation, virtual_ensembles e
boc_stacking direttamente (ognuno chiamato con --votes-dir/--out-dir), senza
passare da run_multi_split.py — adatto perché qui c'è un solo split fisso
(splits_intersection), non serve la discovery multi-split.

Uso:
    python run_all_methods.py
    python run_all_methods.py --votes-dir results/step1/reddit/splits_intersection --out-root results/step4
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_VOTES_DIR = "results/step1/reddit/splits_intersection"
DEFAULT_OUT_ROOT = "results/step4"

METHODS = ["bandit_dr", "graph_propagation", "virtual_ensembles", "boc_stacking"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary = {}

    for method in METHODS:
        out_dir = str(Path(args.out_root) / method)
        print(f"\n{'=' * 60}\n{method}\n{'=' * 60}")
        cmd = [
            sys.executable, str(script_dir / f"{method}.py"),
            "--votes-dir", args.votes_dir,
            "--out-dir", out_dir,
        ]
        result = subprocess.run(cmd)
        summary[method] = "OK" if result.returncode == 0 else f"FALLITO (exit {result.returncode})"

    print(f"\n{'=' * 60}\nRiepilogo finale\n{'=' * 60}")
    for method, status in summary.items():
        print(f"{method}: {status}")


if __name__ == "__main__":
    main()