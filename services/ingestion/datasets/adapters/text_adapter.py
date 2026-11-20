"""Plain-text and Markdown adapter.

Each non-empty line becomes one record.  Markdown front matter is carried as a
record of its own (as inert text) so a downstream consumer can decide what to do
with it; nothing is interpreted or executed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..config import Config
from . import AdapterError, Record, StreamSummary, _decode_probe


def iter_records(
    path: Path, *, config: Config, kind: str, summary: StreamSummary
) -> Iterator[Record]:
    encoding, warnings = _decode_probe(path, config)
    summary.encoding = encoding
    summary.fields = ("line", "text")
    summary.warn(*warnings)
    try:
        with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
            for line_number, line in enumerate(handle):
                if summary.count >= config.max_records_per_file:
                    summary.truncated = True
                    return
                if len(line) > config.max_line_chars:
                    raise AdapterError(
                        "limits_exceeded",
                        f"line {line_number + 1} exceeds {config.max_line_chars} characters",
                    )
                text = line.rstrip("\r\n")
                if not text.strip():
                    continue
                yield Record(
                    summary.count,
                    "text",
                    {"line": line_number + 1, "text": text},
                )
                summary.count += 1
    except UnicodeDecodeError as exc:
        raise AdapterError("encoding_error", "text is not valid in its encoding") from exc
    if summary.count == 0:
        raise AdapterError("empty_content", "file contains no non-empty lines")
