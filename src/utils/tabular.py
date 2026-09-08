"""Small, shared readers for user-supplied tabular inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson"}


def read_table(
    path: Path,
    *,
    columns: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a flat CSV, Parquet, JSON array, or JSON-lines table.

    ``columns`` is enforced for JSON inputs after reading because pandas does not
    expose a uniform projection argument for both JSON encodings.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    requested = list(columns) if columns is not None else None
    if suffix == ".csv":
        frame = pd.read_csv(path, usecols=requested, nrows=nrows, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        if nrows is None:
            frame = pd.read_parquet(path, columns=requested)
        else:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            frame = parquet.read_row_group(0, columns=requested).slice(0, nrows).to_pandas()
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True, nrows=nrows)
    elif suffix == ".json":
        frame = pd.read_json(path)
        if nrows is not None:
            frame = frame.head(nrows)
    else:
        raise ValueError(
            f"Unsupported table format {suffix!r} for {path}; "
            f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )

    if requested is not None:
        missing = set(requested) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[requested]
    return frame


def table_columns(path: Path) -> list[str]:
    """Return columns while reading at most one logical record where possible."""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"Expected one JSON object per line in {path}")
                    return list(record)
        return []
    return list(read_table(path, nrows=1).columns)
