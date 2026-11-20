"""JSON and JSONL/NDJSON adapter.

``json.loads`` is used strictly for data: no object hooks, no ``eval``.  Depth
and record counts are bounded before any structure is walked, so a deeply
nested document cannot exhaust the stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..config import Config
from . import AdapterError, Record, StreamSummary, _decode_probe, _file_size


def _check_depth(value: object, max_depth: int) -> None:
    """Iteratively verify the nesting depth of already-parsed JSON."""
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise AdapterError(
                "limits_exceeded", f"JSON nesting exceeds {max_depth} levels"
            )
        if isinstance(current, dict):
            for item in current.values():
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            for item in current:
                stack.append((item, depth + 1))


def _field_names(value: object, max_fields: int) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    if len(value) > max_fields:
        raise AdapterError(
            "limits_exceeded", f"object has {len(value)} fields (cap {max_fields})"
        )
    return tuple(str(key) for key in value.keys())


def _loads(text: str, max_depth: int, max_fields: int) -> object:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise AdapterError("parse_error", f"invalid JSON: {exc}") from exc
    _check_depth(value, max_depth)
    _field_names(value, max_fields)
    return value


def iter_records(
    path: Path, *, config: Config, kind: str, summary: StreamSummary
) -> Iterator[Record]:
    if kind == "jsonl":
        yield from _iter_jsonl(path, config=config, summary=summary)
    else:
        yield from _iter_json(path, config=config, summary=summary)


def _iter_json(path: Path, *, config: Config, summary: StreamSummary) -> Iterator[Record]:
    size = _file_size(path)
    if size > config.max_file_bytes:
        raise AdapterError(
            "limits_exceeded", f"JSON document exceeds {config.max_file_bytes} bytes"
        )
    encoding, warnings = _decode_probe(path, config)
    summary.encoding = encoding
    summary.warn(*warnings)
    try:
        text = path.read_text(encoding=encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise AdapterError("encoding_error", "JSON is not valid in its encoding") from exc
    if len(text) > config.max_line_chars and len(text.splitlines()) <= 1:
        # A single line past the cap is still valid JSON; the cap protects
        # line-oriented consumers, so only warn here.
        summary.warn(f"single_line_json:{len(text)}")
    value = _loads(text, config.max_json_depth, config.max_fields)
    if isinstance(value, list):
        for index, item in enumerate(value):
            if index >= config.max_records_per_file:
                summary.truncated = True
                return
            fields = _field_names(item, config.max_fields)
            if not summary.fields and fields:
                summary.fields = fields
            summary.count += 1
            yield Record(index, "value", item)
    elif isinstance(value, dict):
        summary.fields = _field_names(value, config.max_fields)
        summary.count = 1
        yield Record(0, "object", value)
    else:
        summary.count = 1
        yield Record(0, "value", value)
    if summary.count == 0:
        raise AdapterError("empty_content", "JSON document contains no records")


def _iter_jsonl(path: Path, *, config: Config, summary: StreamSummary) -> Iterator[Record]:
    encoding, warnings = _decode_probe(path, config)
    summary.encoding = encoding
    summary.warn(*warnings)
    malformed = 0
    try:
        with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
            for line_number, line in enumerate(handle):
                if summary.count >= config.max_records_per_file:
                    summary.truncated = True
                    break
                if len(line) > config.max_line_chars:
                    raise AdapterError(
                        "limits_exceeded",
                        f"line {line_number + 1} exceeds {config.max_line_chars} characters",
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except ValueError:
                    malformed += 1
                    # A few junk lines (trailing comma, truncated tail) must
                    # not discard an otherwise usable file; too many must.
                    if malformed > 10:
                        raise AdapterError(
                            "parse_error",
                            f"{malformed} malformed JSONL lines (first at line {line_number + 1})",
                        )
                    continue
                _check_depth(value, config.max_json_depth)
                field_names = _field_names(value, config.max_fields)
                if not summary.fields:
                    summary.fields = field_names
                kind = "object" if isinstance(value, dict) else "value"
                yield Record(summary.count, kind, value)
                summary.count += 1
    except UnicodeDecodeError as exc:
        raise AdapterError("encoding_error", "JSONL is not valid in its encoding") from exc
    if malformed:
        summary.warn(f"malformed_lines:{malformed}")
    if summary.count == 0:
        raise AdapterError("empty_content", "JSONL file contains no records")
