#!/usr/bin/env python3
"""Read-only, bounded export over the Knowledge SQLite store.

The gateway exposes ingested corpus records to external applications without
giving them a SQLite handle or a filesystem path. This module owns that read
path:

* The database is opened per request through a ``mode=ro`` URI. Export traffic
  never creates the schema, never writes, and never shares a connection with
  the Knowledge service's ingestion writes.
* Records are ordered by ``source_id`` and then ``revision_sequence`` and
  paged with an opaque keyset cursor, so appends and updates between page
  requests cannot duplicate or skip the rows a client is already walking.
* Every response is bounded twice: per-record content is truncated to a
  configured number of characters with an explicit ``content_truncated`` flag,
  and a page stops early when its accumulated content budget is reached. The
  returned ``next_cursor`` always resumes at the first omitted record.

There are no mutation endpoints here, by design.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = os.path.expanduser(
    "~/.FlossWare/poc-claude-ensemble/data/sqlite/knowledge.sqlite3"
)
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
MAX_ID_CHARS = 512
DEFAULT_MAX_CONTENT_CHARS = 65_536
DEFAULT_MAX_PAGE_CONTENT_CHARS = 4 * 1_048_576
CURSOR_VERSION = 1
FILTER_NAMES = ("origin", "source_id")
REQUIRED_COLUMNS = {
    "source_id",
    "revision_sequence",
    "title",
    "origin",
    "locator",
    "content_sha256",
    "content",
    "tombstoned",
}


class CorpusError(ValueError):
    """A client-visible export error carrying the shared envelope code/status."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def encode_cursor(source_id: str, revision_sequence: int) -> str:
    """Encode the keyset position as URL-safe base64 without padding.

    ``source_id`` is arbitrary text, so it cannot be embedded in a plain
    ``id:sequence`` string without an escaping scheme. JSON plus base64 is
    unambiguous and keeps the cursor opaque to clients.
    """
    payload = json.dumps(
        {
            "v": CURSOR_VERSION,
            "source_id": source_id,
            "revision_sequence": revision_sequence,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> tuple[str, int]:
    """Return ``(source_id, revision_sequence)`` or raise a 400 CorpusError."""
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise CorpusError("invalid_request", "malformed cursor")
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError):
        raise CorpusError("invalid_request", "malformed cursor") from None
    if not isinstance(payload, dict) or payload.get("v") != CURSOR_VERSION:
        raise CorpusError("invalid_request", "malformed cursor")
    source_id = payload.get("source_id")
    sequence = payload.get("revision_sequence")
    if not isinstance(source_id, str) or not source_id:
        raise CorpusError("invalid_request", "malformed cursor")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise CorpusError("invalid_request", "malformed cursor")
    return source_id, sequence


class CorpusExportStore:
    """A read-only view of Knowledge records for the gateway."""

    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.expanduser(
            db_path
            or os.environ.get("POC_CLAUDE_ENSEMBLE_CORPUS_DB")
            or os.environ.get("POC_CLAUDE_ENSEMBLE_KNOWLEDGE_DB")
            or DEFAULT_DB_PATH
        )
        self.max_content_chars = _env_positive_int(
            "POC_CLAUDE_ENSEMBLE_CORPUS_MAX_CONTENT_CHARS", DEFAULT_MAX_CONTENT_CHARS
        )
        page_budget = _env_positive_int(
            "POC_CLAUDE_ENSEMBLE_CORPUS_MAX_PAGE_CONTENT_CHARS",
            DEFAULT_MAX_PAGE_CONTENT_CHARS,
        )
        # A record must always fit inside a page, otherwise the first record of
        # every page could exceed the budget and pagination could stall.
        self.max_page_content_chars = max(page_budget, self.max_content_chars)

    @contextmanager
    def read_only_connection(self) -> Iterator[sqlite3.Connection]:
        """Open the store with ``mode=ro`` and close it at the end of the request.

        A short busy timeout keeps an export request from waiting behind a
        writer for long. The connection itself can never take a write lock.
        """
        path = os.path.abspath(self.db_path)
        if not os.path.isfile(path):
            raise CorpusError("corpus_unavailable", "corpus store is not available", status=503)
        uri = Path(path).as_uri() + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        except sqlite3.Error as exc:
            raise CorpusError("corpus_unavailable", "corpus store cannot be opened read-only", status=503) from exc
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _columns(conn: sqlite3.Connection) -> set[str]:
        return {row["name"] for row in conn.execute("PRAGMA table_info(records)")}

    @staticmethod
    def _require_schema(columns: set[str]) -> None:
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise CorpusError(
                "corpus_unavailable",
                "corpus store schema is not compatible: missing " + ", ".join(missing),
                status=503,
            )

    @staticmethod
    def _limit(value: str) -> int:
        if not isinstance(value, str) or not value.isascii() or not value.isdigit():
            raise CorpusError("invalid_request", "limit must be an integer")
        limit = int(value)
        if limit < 1 or limit > MAX_LIMIT:
            raise CorpusError("invalid_request", f"limit must be between 1 and {MAX_LIMIT}")
        return limit

    @staticmethod
    def _filters(parameters: dict[str, list[str]]) -> dict[str, str]:
        filters: dict[str, str] = {}
        for name in FILTER_NAMES:
            if name not in parameters:
                continue
            value = parameters[name][0]
            if not value or len(value) > MAX_ID_CHARS:
                raise CorpusError(
                    "invalid_request",
                    f"{name} must be a non-empty string of at most {MAX_ID_CHARS} characters",
                )
            filters[name] = value
        return filters

    def records(self, parameters: dict[str, list[str]] | None = None) -> dict[str, Any]:
        """Return one bounded, filtered page of records."""
        parameters = parameters or {}
        supported = set(FILTER_NAMES) | {"limit", "cursor"}
        unknown = sorted(set(parameters) - supported)
        if unknown:
            raise CorpusError("invalid_request", f"unsupported query parameter: {unknown[0]}")
        if any(len(values) != 1 for values in parameters.values()):
            raise CorpusError("invalid_request", "each query parameter may be supplied only once")
        limit = self._limit(parameters.get("limit", [str(DEFAULT_LIMIT)])[0])
        filters = self._filters(parameters)
        cursor = decode_cursor(parameters["cursor"][0]) if "cursor" in parameters else None

        where = ["tombstoned = 0"]
        where_args: list[Any] = []
        for name in FILTER_NAMES:
            if name in filters:
                where.append(f"{name} = ?")
                where_args.append(filters[name])
        page_where = list(where)
        page_args = list(where_args)
        if cursor is not None:
            page_where.append("(source_id > ? OR (source_id = ? AND revision_sequence > ?))")
            page_args.extend((cursor[0], cursor[0], cursor[1]))

        try:
            with self.read_only_connection() as conn:
                columns = self._columns(conn)
                self._require_schema(columns)
                retrieved_column = next(
                    (name for name in ("retrieved", "retrieved_at") if name in columns), None
                )
                select = (
                    "SELECT source_id, revision_sequence, title, origin, locator, content_sha256, "
                    "substr(content, 1, ?) AS content_excerpt, length(content) AS content_length"
                )
                if retrieved_column:
                    select += f", {retrieved_column}"
                # The excerpt cap is the first bind parameter because it appears in the
                # SELECT list; filters and the cursor follow in SQL order.
                rows = conn.execute(
                    select
                    + " FROM records WHERE "
                    + " AND ".join(page_where)
                    + " ORDER BY source_id ASC, revision_sequence ASC LIMIT ?",
                    [self.max_content_chars, *page_args, limit + 1],
                ).fetchall()
                total = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM records WHERE " + " AND ".join(where), where_args
                    ).fetchone()[0]
                )
        except sqlite3.Error as exc:
            raise CorpusError("corpus_unavailable", "corpus store read failed", status=503) from exc

        selected: list[sqlite3.Row] = []
        used_chars = 0
        budget_exhausted = False
        for row in rows:
            if len(selected) >= limit:
                break
            excerpt_chars = len(row["content_excerpt"])
            if selected and used_chars + excerpt_chars > self.max_page_content_chars:
                budget_exhausted = True
                break
            selected.append(row)
            used_chars += excerpt_chars
        has_more = len(selected) < len(rows) or budget_exhausted

        records = []
        for row in selected:
            record: dict[str, Any] = {
                "source_id": row["source_id"],
                "revision_sequence": int(row["revision_sequence"]),
                "title": row["title"],
                "origin": row["origin"],
                "locator": row["locator"],
                "content_sha256": row["content_sha256"],
                "content": row["content_excerpt"],
                "content_length": int(row["content_length"]),
                "content_truncated": int(row["content_length"]) > len(row["content_excerpt"]),
            }
            if retrieved_column:
                record["retrieved"] = row[retrieved_column]
            records.append(record)

        next_cursor = None
        if has_more and selected:
            last = selected[-1]
            next_cursor = encode_cursor(last["source_id"], int(last["revision_sequence"]))
        return {
            "ok": True,
            "records": records,
            "next_cursor": next_cursor,
            "total_hint": total,
            "limit": limit,
            "returned": len(records),
            "content_budget_exhausted": budget_exhausted,
        }

    def origins(self) -> dict[str, Any]:
        """Return each live origin with its record count, ordered by origin."""
        try:
            with self.read_only_connection() as conn:
                self._require_schema(self._columns(conn))
                rows = conn.execute(
                    "SELECT origin, COUNT(*) AS count FROM records WHERE tombstoned = 0 "
                    "GROUP BY origin ORDER BY origin ASC"
                ).fetchall()
        except sqlite3.Error as exc:
            raise CorpusError("corpus_unavailable", "corpus store read failed", status=503) from exc
        return {
            "ok": True,
            "origins": [{"origin": row["origin"], "count": int(row["count"])} for row in rows],
        }
