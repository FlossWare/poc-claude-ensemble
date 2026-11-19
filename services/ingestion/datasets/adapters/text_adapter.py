"""Plain-text and Markdown adapter.

Each non-empty line becomes one record.  Markdown front matter is carried as a
record of its own (as inert text) so a downstream consumer can decide what to do
with it; nothing is interpreted or executed.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from . import (
    AdapterError,
    ParseResult,
    Record,
    _Collector,
    _decode_probe,
    _file_size,
    _summary,
)


def parse(path: Path, *, config: Config, kind: str) -> ParseResult:
    encoding, warnings = _decode_probe(path, config)
    collector = _Collector(max_sample=config.adapter_max_sample_records)
    truncated = False
    try:
        with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
            for line_number, line in enumerate(handle):
                if collector.count >= config.max_records_per_file:
                    truncated = True
                    break
                if len(line) > config.max_line_chars:
                    raise AdapterError(
                        "limits_exceeded",
                        f"line {line_number + 1} exceeds {config.max_line_chars} characters",
                    )
                text = line.rstrip("\r\n")
                if not text.strip():
                    continue
                collector.add(
                    Record(collector.count, "text", {"line": line_number + 1, "text": text})
                )
    except UnicodeDecodeError as exc:
        raise AdapterError("encoding_error", "text is not valid in its encoding") from exc
    if collector.count == 0:
        return ParseResult(
            adapter=kind,
            status="skipped",
            reason_code="empty_content",
            detail="file contains no non-empty lines",
            warnings=warnings,
            encoding=encoding,
            bytes_read=_file_size(path),
            media_kind=kind,
        )
    collector.truncated = truncated
    return _summary(
        kind,
        collector,
        fields=("line", "text"),
        warnings=warnings,
        encoding=encoding,
        bytes_read=_file_size(path),
        media_kind=kind,
    )
