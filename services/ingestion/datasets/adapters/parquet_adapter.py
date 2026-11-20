"""Optional Parquet adapter.

Only used when ``pyarrow`` is importable (``PARQUET_ENABLED`` defaults to its
availability).  Otherwise the caller records ``adapter_unavailable`` rather than
failing the dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..config import Config
from . import AdapterError, Record, StreamSummary


def iter_records(
    path: Path, *, config: Config, summary: StreamSummary
) -> Iterator[Record]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AdapterError("adapter_unavailable", "pyarrow is not installed") from exc

    try:
        source = parquet.ParquetFile(path)
        fields = tuple(str(name) for name in source.schema_arrow.names)
        if len(fields) > config.max_fields:
            raise AdapterError(
                "limits_exceeded", f"parquet has {len(fields)} columns (cap {config.max_fields})"
            )
        summary.fields = fields
        for batch in source.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if summary.count >= config.max_records_per_file:
                    summary.truncated = True
                    return
                for value in row.values():
                    if isinstance(value, str) and len(value) > config.max_cell_chars:
                        raise AdapterError(
                            "limits_exceeded",
                            f"cell exceeds {config.max_cell_chars} characters",
                        )
                yield Record(summary.count, "row", row)
                summary.count += 1
    except AdapterError:
        raise
    except Exception as exc:  # pyarrow raises its own exception hierarchy
        raise AdapterError("parse_error", f"parquet could not be read: {type(exc).__name__}") from exc
    if summary.count == 0:
        raise AdapterError("empty_content", "parquet file contains no rows")
