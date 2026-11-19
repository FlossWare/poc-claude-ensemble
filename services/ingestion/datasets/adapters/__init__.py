"""PR 3: format adapters that turn acquired bytes into normalized records.

Each adapter is a generator of :class:`Record` objects plus a bounded summary.
Parsing never interprets content: CSV cells stay inert text, JSON is never
evaluated, XML is rejected outright if it carries a DOCTYPE or ENTITY, and every
loop is capped by configuration (records, fields, cell size, line length, JSON
depth, XML depth).

PR 4 owns the text projection, sharding and indexing.  This module stops at
"normalized records", which is why :func:`parse_file` returns counts plus a
small bounded sample rather than materialising a 100,000-row table in memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..model import sanitize_nested

ADAPTER_VERSION = 2

#: Extensions that map onto an adapter.
EXTENSION_KINDS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".json": "json",
    ".geojson": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".parquet": "parquet",
    ".pdf": "pdf",
}

MEDIA_TYPE_KINDS = {
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/csv": "csv",
    "application/json": "json",
    "application/geo+json": "json",
    "application/x-ndjson": "jsonl",
    "application/ndjson": "jsonl",
    "application/jsonl": "jsonl",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/plain": "text",
    "text/markdown": "markdown",
    "application/parquet": "parquet",
    "application/vnd.apache.parquet": "parquet",
    "application/pdf": "pdf",
}

TEXT_KINDS = frozenset({"csv", "tsv", "json", "jsonl", "xml", "text", "markdown"})

class AdapterError(Exception):
    """A file could not be parsed into records."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


@dataclass(frozen=True)
class Record:
    """One normalized record.

    ``kind`` is ``object`` (a JSON object), ``row`` (a tabular row),
    ``value`` (a scalar or array element) or ``text`` (a line of text).
    ``data`` is already JSON-safe.
    """

    index: int
    kind: str
    data: object

    def to_artifact(self) -> dict[str, object]:
        return {"index": self.index, "kind": self.kind, "data": self.data}


@dataclass(frozen=True)
class ParseResult:
    adapter: str | None
    status: str
    reason_code: str | None = None
    detail: str | None = None
    record_count: int = 0
    truncated: bool = False
    fields: tuple[str, ...] = ()
    sample: tuple[Record, ...] = ()
    warnings: tuple[str, ...] = ()
    encoding: str | None = None
    bytes_read: int = 0
    media_kind: str | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "adapter_version": ADAPTER_VERSION,
            "adapter": self.adapter,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:500] if self.detail else None,
            "record_count": self.record_count,
            "truncated": self.truncated,
            "fields": list(self.fields),
            "sample": [record.to_artifact() for record in self.sample],
            "warnings": list(self.warnings),
            "encoding": self.encoding,
            "bytes_read": self.bytes_read,
            "media_kind": self.media_kind,
        }


def normalize_kind(
    name: str | None, media_type: str | None, prefix: bytes | None
) -> str | None:
    """Decide which adapter handles a file.

    The extension wins (a ``.jsonl`` URL served as ``text/plain`` is still
    JSONL), then the media type, then a magic-byte sniff.
    """
    if name:
        lowered = name.lower().rsplit("/", 1)[-1]
        for suffix, kind in sorted(
            EXTENSION_KINDS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if lowered.endswith(suffix):
                return kind
    if media_type:
        base = media_type.split(";", 1)[0].strip().lower()
        if base in MEDIA_TYPE_KINDS:
            return MEDIA_TYPE_KINDS[base]
    return sniff_kind(prefix)


def sniff_kind(prefix: bytes | None) -> str | None:
    if not prefix:
        return None
    stripped = prefix.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json"
    if stripped.startswith(b"<?xml"):
        return "xml"
    return None


def _decode_probe(path: Path, config: Config) -> tuple[str, tuple[str, ...]]:
    """Pick an encoding from the first bytes; ``utf-8-sig`` then ``latin-1``."""
    try:
        with path.open("rb") as handle:
            head = handle.read(64 * 1024)
    except OSError as exc:
        raise AdapterError("parse_error", f"file could not be opened: {type(exc).__name__}") from exc
    for encoding in ("utf-8-sig",):
        try:
            head.decode(encoding)
            return encoding, ()
        except UnicodeDecodeError:
            continue
    if config.adapter_encoding_fallback:
        return "latin-1", ("encoding_fallback:latin-1",)
    raise AdapterError("encoding_error", "file is not valid UTF-8")


def _bounded_lines(handle, max_line_chars: int):
    """Yield decoded lines, refusing any line past the configured cap."""
    for line in handle:
        if len(line) > max_line_chars:
            raise AdapterError(
                "limits_exceeded", f"line exceeds {max_line_chars} characters"
            )
        yield line


@dataclass
class _Collector:
    """Accumulates counts, observed fields and a bounded sample."""

    max_sample: int
    sample: list[Record] = field(default_factory=list)
    count: int = 0
    truncated: bool = False

    def add(self, record: Record) -> None:
        self.count += 1
        if len(self.sample) < self.max_sample:
            self.sample.append(record)


def _summary(
    adapter: str,
    collector: _Collector,
    *,
    fields: tuple[str, ...],
    warnings: tuple[str, ...],
    encoding: str | None,
    bytes_read: int,
    media_kind: str,
) -> ParseResult:
    return ParseResult(
        adapter=adapter,
        status="parsed",
        record_count=collector.count,
        truncated=collector.truncated,
        fields=fields,
        sample=tuple(
            Record(record.index, record.kind, sanitize_nested(record.data))
            for record in collector.sample
        ),
        warnings=warnings,
        encoding=encoding,
        bytes_read=bytes_read,
        media_kind=media_kind,
    )


def parse_file(
    path: Path,
    *,
    name: str | None,
    media_type: str | None,
    config: Config,
    kind: str | None = None,
) -> ParseResult:
    """Parse one file into a bounded summary of normalized records."""
    path = Path(path)
    try:
        prefix = _read_prefix(path, config.acquire_sniff_prefix_bytes)
    except OSError as exc:
        return _failed(None, "parse_error", f"file could not be opened: {type(exc).__name__}")
    resolved = kind or normalize_kind(name, media_type, prefix)
    if resolved is None or resolved in {"archive"}:
        return ParseResult(
            adapter=None,
            status="skipped",
            reason_code="unsupported_format",
            detail=f"no adapter for {name or media_type or 'unknown content'}",
            media_kind=resolved,
        )
    if resolved == "pdf":
        # PDF extraction is deliberately disabled for datasets (design §3).
        return ParseResult(
            adapter="pdf",
            status="skipped",
            reason_code="adapter_unavailable",
            detail="PDF extraction is disabled for datasets",
            media_kind="pdf",
        )
    if resolved == "parquet" and not config.parquet_enabled:
        return ParseResult(
            adapter=None,
            status="skipped",
            reason_code="adapter_unavailable",
            detail="pyarrow is not installed",
            media_kind=resolved,
        )
    try:
        # Imported lazily so an unrelated adapter failure cannot cascade.
        from . import csv_adapter, json_adapter, text_adapter, xml_adapter

        if resolved in {"csv", "tsv"}:
            return csv_adapter.parse(
                path,
                name=name,
                media_type=media_type,
                config=config,
                kind=resolved,
            )
        if resolved in {"json", "jsonl"}:
            return json_adapter.parse(
                path,
                name=name,
                media_type=media_type,
                config=config,
                kind=resolved,
            )
        if resolved == "xml":
            return xml_adapter.parse(path, config=config)
        if resolved in {"text", "markdown"}:
            return text_adapter.parse(path, config=config, kind=resolved)
        if resolved == "parquet":
            from . import parquet_adapter

            return parquet_adapter.parse(path, config=config)
    except AdapterError as exc:
        return _failed(resolved, exc.code, exc.detail)
    except (ValueError, UnicodeError, OSError, EOFError) as exc:
        return _failed(resolved, "parse_error", type(exc).__name__)
    return ParseResult(
        adapter=None,
        status="skipped",
        reason_code="unsupported_format",
        detail=f"no adapter for {resolved}",
        media_kind=resolved,
    )


def _failed(adapter: str | None, code: str, detail: str) -> ParseResult:
    return ParseResult(
        adapter=adapter,
        status="failed",
        reason_code=code,
        detail=detail,
        media_kind=adapter,
    )


def _read_prefix(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(max(1, limit))


def _file_size(path: Path) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


__all__ = [
    "ADAPTER_VERSION",
    "AdapterError",
    "ParseResult",
    "Record",
    "normalize_kind",
    "parse_file",
    "sniff_kind",
]
