"""Helpers for reading the semantic result directory layout."""

from pathlib import Path


def parse_result_path(metrics_path: Path, root: Path) -> tuple[str, str] | None:
    """Return the evaluation and method encoded by a metrics path."""
    try:
        parts = metrics_path.parent.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    if "baselines" in parts:
        marker = parts.index("baselines")
        return "/".join(parts[:marker]), parts[-1]
    return "/".join(parts[:-1]), parts[-1]
