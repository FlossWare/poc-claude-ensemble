#!/usr/bin/env python3
"""Loopback REST index service: the source manifest that owns revision order, tombstones, and readiness.

Index does not hold canonical content. It records, per source, the highest revision sequence it has
accepted, the profile versions that revision was built with, and any tombstone. Writers consult it before
publishing, so an older revision can never overwrite a newer one, and a deleted source stays deleted.

State is a SQLite database, written in one transaction per request. Re-sending the same revision is
idempotent; a different revision at the same sequence is a conflict; a lower sequence is superseded.

Endpoints
  GET  /health
  GET  /ready
  POST /revisions     {"source_id", "revision_sequence", "revision_id", "profiles": {...}} -> decision
  POST /tombstones    {"source_id", "revision_sequence"} -> decision
  GET  /sources/{source_id}   current state of one source
  GET  /manifest              bounded listing of sources (limit query parameter, max 500)
"""

import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DB_PATH = os.environ.get("ENSEMBLE_INDEX_DB", os.path.expanduser(
    "~/.FlossWare/poc-claude-ensemble/data/sqlite/manifest.sqlite3"))
MAX_BODY_BYTES = 1_048_576
MAX_ID_CHARS = 512
MAX_MANIFEST = 500
_LOCK = threading.Lock()


class IndexError_(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        high_sequence INTEGER NOT NULL,
        revision_id TEXT NOT NULL,
        profiles TEXT NOT NULL,
        tombstoned INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
    return conn


def _validate_id(value, field):
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS:
        raise IndexError_("invalid_request", f"{field} must be a non-empty string of at most {MAX_ID_CHARS} characters")
    return value


def _validate_sequence(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IndexError_("invalid_request", "revision_sequence must be a positive integer")
    return value


def _row(row):
    return {"source_id": row["source_id"], "high_sequence": row["high_sequence"],
            "revision_id": row["revision_id"], "profiles": json.loads(row["profiles"]),
            "tombstoned": bool(row["tombstoned"]), "updated_at": row["updated_at"]}


def record_revision(source_id, sequence, revision_id, profiles):
    """Decide whether a revision may publish. Returns the decision and the current state."""
    source_id = _validate_id(source_id, "source_id")
    sequence = _validate_sequence(sequence)
    revision_id = _validate_id(revision_id, "revision_id")
    if not isinstance(profiles, dict) or not profiles:
        raise IndexError_("invalid_request", "profiles must be a non-empty object naming the profile versions used")
    profile_json = json.dumps(profiles, sort_keys=True)
    with _LOCK:
        conn = _connect()
        try:
            with conn:
                row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO sources (source_id, high_sequence, revision_id, profiles) VALUES (?,?,?,?)",
                                 (source_id, sequence, revision_id, profile_json))
                    decision = "accepted"
                elif row["tombstoned"]:
                    decision = "tombstoned"
                elif sequence > row["high_sequence"]:
                    conn.execute("UPDATE sources SET high_sequence=?, revision_id=?, profiles=?, "
                                 "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE source_id=?",
                                 (sequence, revision_id, profile_json, source_id))
                    decision = "accepted"
                elif sequence == row["high_sequence"]:
                    if row["revision_id"] == revision_id and row["profiles"] == profile_json:
                        decision = "duplicate"
                    else:
                        decision = "conflict"
                else:
                    decision = "superseded"
                current = _row(conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone())
        finally:
            conn.close()
    return {"decision": decision, "source": current}


def record_tombstone(source_id, sequence):
    source_id = _validate_id(source_id, "source_id")
    sequence = _validate_sequence(sequence)
    with _LOCK:
        conn = _connect()
        try:
            with conn:
                row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO sources (source_id, high_sequence, revision_id, profiles, tombstoned) "
                                 "VALUES (?,?,?,?,1)", (source_id, sequence, "tombstone", "{}"))
                    decision = "tombstoned"
                elif sequence >= row["high_sequence"]:
                    conn.execute("UPDATE sources SET high_sequence=?, tombstoned=1, "
                                 "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE source_id=?",
                                 (max(sequence, row["high_sequence"]), source_id))
                    decision = "tombstoned"
                else:
                    decision = "superseded"
                current = _row(conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone())
        finally:
            conn.close()
    return {"decision": decision, "source": current}


def get_source(source_id):
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    finally:
        conn.close()
    return _row(row) if row else None


def list_sources(limit):
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM sources ORDER BY source_id LIMIT ?", (limit,)).fetchall()
    finally:
        conn.close()
    return [_row(r) for r in rows]


class IndexHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleIndex/1"

    def _send(self, status, payload):
        body = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._send(status, {"ok": False, "error": {"code": code, "message": message}})

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/health":
                self._send(200, {"ok": True, "service": "index"})
            elif url.path == "/ready":
                conn = _connect()
                conn.execute("SELECT 1").fetchone()
                conn.close()
                self._send(200, {"ok": True, "ready": True})
            elif url.path.startswith("/sources/"):
                source = get_source(unquote(url.path[len("/sources/"):]))
                if source is None:
                    self._error(404, "not_found", "unknown source")
                else:
                    self._send(200, {"ok": True, "source": source})
            elif url.path == "/manifest":
                try:
                    limit = int(parse_qs(url.query).get("limit", ["100"])[0])
                except ValueError:
                    self._error(400, "invalid_request", "limit must be an integer")
                    return
                if not 1 <= limit <= MAX_MANIFEST:
                    self._error(400, "invalid_request", f"limit must be between 1 and {MAX_MANIFEST}")
                    return
                self._send(200, {"ok": True, "sources": list_sources(limit)})
            else:
                self._error(404, "not_found", "unknown path")
        except Exception as exc:  # reported safely; never a traceback to the caller
            self._error(503, "service_unavailable", type(exc).__name__)

    def do_POST(self):
        if self.path not in {"/revisions", "/tombstones"}:
            self._error(404, "not_found", "unknown path")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(400, "invalid_request", "request body must be between 1 byte and 1 MiB")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid_request", "body must be UTF-8 JSON")
            return
        if not isinstance(body, dict):
            self._error(400, "invalid_request", "body must be a JSON object")
            return
        try:
            if self.path == "/revisions":
                result = record_revision(body.get("source_id"), body.get("revision_sequence"),
                                         body.get("revision_id"), body.get("profiles"))
            else:
                result = record_tombstone(body.get("source_id"), body.get("revision_sequence"))
        except IndexError_ as exc:
            self._error(exc.status, exc.code, str(exc))
            return
        except sqlite3.Error as exc:
            self._error(503, "service_unavailable", type(exc).__name__)
            return
        self._send(200, dict(result, ok=True))

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("index service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), IndexHandler)


def main():
    host = os.environ.get("ENSEMBLE_INDEX_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_INDEX_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
