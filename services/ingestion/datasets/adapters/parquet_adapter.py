"""Optional Parquet adapter.

Only used when ``pyarrow`` is importable (``PARQUET_ENABLED`` defaults to its
availability).  Otherwise the caller records ``adapter_unavailable`` rather than
failing the dataset.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from . import (
    AdapterError,
    ParseResult,
    Record,
    _Collector,
    _file_size,
    _summary,
)


def parse(path: Path, *, config: Config) -> ParseResult:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AdapterError("adapter_unavailable", "pyarrow is not installed") from exc

    collector = _Collector(max_sample=config.adapter_max_sample_records)
    try:
        source = parquet.ParquetFile(path)
        fields = tuple(str(name) for name in source.schema_arrow.names)
        if len(fields) > config.max_fields:
            raise AdapterError(
                "limits_exceeded", f"parquet has {len(fields)} columns (cap {config.max_fields})"
            )
        for batch in source.iter_batches(batch_size=1024):
            rows = batch.to_pylist()
            for row in rows:
                if collector.count >= config.max_records_per_file:
                    collector.truncated = True
                    break
                for value in row.values():
                    if isinstance(value, str) and len(value) > config.max_cell_chars:
                        raise AdapterError(
                            "limits_exceeded",
                            f"cell exceeds {config.max_cell_chars} characters",
                        )
                collector.add(Record(collector.count, "row", row))
            if collector.truncated:
                break
    except AdapterError:
        raise
    except Exception as exc:  # pyarrow raises its own exception hierarchy
        raise AdapterError("parse_error", f"parquet could not be read: {type(exc).__name__}") from exc
    if collector.count == 0:
        return ParseResult(
            adapter="parquet",
            status="skipped",
            reason_code="empty_content",
            detail="parquet file contains no rows",
            fields=fields,
            bytes_read=_file_size(path),
            media_kind="parquet",
        )
    return _summary(
        "parquet",
        collector,
        fields=fields,
        warnings=(),
        encoding=None,
        bytes_read=_file_size(path),
        media_kind="parquet",
    )
