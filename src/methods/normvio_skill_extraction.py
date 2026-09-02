"""Command-line entry point for NormVio + Skill Extraction."""

from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.nvse.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
