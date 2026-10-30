#!/usr/bin/env python3
"""Loopback REST knowledge service: sourced records with revisions, provenance, and deletion.

Knowledge owns the content. Before a revision is stored, it asks Index (over REST) whether the revision
may publish, so an older revision can never replace a newer one and a deleted source stays deleted.
Only accepted or duplicate decisions are stored. Every record keeps its origin and locator.

Endpoints
  GET  /health
  GET  /ready
  POST /records                      {"source_id", "revision_sequence", "revision_id", "title", "content",
                                      "provenance": {"origin", "locator"}, "profiles"} -> decision
  GET  /records/{source_id}          current record
  POST /records/{source_id}/tombstone  {"revision_sequence"} -> tombstone decision
"""

import hashlib
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8781
INDEX_URL = os.environ.get("ENSEMBLE_INDEX_URL", "http://127.0.0.1:8777")
DB_PATH = os.environ.get("ENSEMBLE_KNOWLEDGE_DB", os.path.expanduser(
    "~/.FlossWare/data/sqlite/v1/knowledge.sqlite3"))
MAX_BODY_BYTES = 8 * 1_048_576
MAX_CONTENT_CHARS = 4_000_000
MAX_ID_CHARS = 512
INDEX_TIMEOUT = 5
_LOCK = threading.Lock()


class KnowledgeError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS records (
        source_id TEXT PRIMARY KEY, revision_sequence INTEGER NOT NULL, revision_id TEXT NOT NULL,
        title TEXT NOT NULL, content TEXT NOT NULL, content_sha256 TEXT NOT NULL,
        origin TEXT NOT NULL, locator TEXT NOT NULL, profiles TEXT NOT NULL, tombstoned INTEGER NOT NULL DEFAULT 0)""")
    return conn


def _index_call(path, payload):
    """Ask Index for a decision. Index unreachable is an explicit failure, never a silent accept."""
    request = urllib.request.Request(f"{INDEX_URL}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=INDEX_TIMEOUT) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        with exc:
            detail = json.loads(exc.read()).get("error", {}).get("message", str(exc))
        raise KnowledgeError("index_rejected", f"index returned HTTP {exc.code}: {detail}", status=502) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise KnowledgeError("index_unavailable", f"index unreachable: {exc}", status=503) from exc
    if body.get("ok") is not True:
        raise KnowledgeError("index_rejected", "index returned an unexpected response", status=502)
    return body["decision"]


def _id(value, field):
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS:
        raise KnowledgeError("invalid_request", f"{field} must be a non-empty string of at most {MAX_ID_CHARS} characters")
    return value


def store_record(body):
    source_id = _id(body.get("source_id"), "source_id")
    revision_id = _id(body.get("revision_id"), "revision_id")
    sequence = body.get("revision_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise KnowledgeError("invalid_request", "revision_sequence must be a positive integer")
    title, content = body.get("title"), body.get("content")
    if not isinstance(title, str) or not title:
        raise KnowledgeError("invalid_request", "title must be a non-empty string")
    if not isinstance(content, str) or not content:
        raise KnowledgeError("invalid_request", "content must be a non-empty string")
    if len(content) > MAX_CONTENT_CHARS:
        raise KnowledgeError("payload_too_large", f"content is limited to {MAX_CONTENT_CHARS} characters", status=413)
    provenance = body.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("origin"), str) \
            or not isinstance(provenance.get("locator"), str):
        raise KnowledgeError("invalid_request", "provenance must include string origin and locator")
    profiles = body.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise KnowledgeError("invalid_request", "profiles must be a non-empty object")

    decision = _index_call("/revisions", {"source_id": source_id, "revision_sequence": sequence,
                                          "revision_id": revision_id, "profiles": profiles})
    if decision not in ("accepted", "duplicate"):
        return {"decision": decision, "source_id": source_id, "stored": False}
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with _LOCK:
        conn = _connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO records (source_id, revision_sequence, revision_id, title, content,"
                    " content_sha256, origin, locator, profiles, tombstoned) VALUES (?,?,?,?,?,?,?,?,?,0)",
                    (source_id, sequence, revision_id, title, content, digest,
                     provenance["origin"], provenance["locator"], json.dumps(profiles, sort_keys=True)))
        finally:
            conn.close()
    return {"decision": decision, "source_id": source_id, "stored": True, "content_sha256": digest}


def get_record(source_id):
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM records WHERE source_id = ?", (source_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"source_id": row["source_id"], "revision_sequence": row["revision_sequence"],
            "revision_id": row["revision_id"], "title": row["title"], "content": row["content"],
            "content_sha256": row["content_sha256"], "provenance": {"origin": row["origin"], "locator": row["locator"]},
            "profiles": json.loads(row["profiles"]), "tombstoned": bool(row["tombstoned"])}


def tombstone_record(source_id, sequence):
    source_id = _id(source_id, "source_id")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise KnowledgeError("invalid_request", "revision_sequence must be a positive integer")
    decision = _index_call("/tombstones", {"source_id": source_id, "revision_sequence": sequence})
    if decision == "tombstoned":
        with _LOCK:
            conn = _connect()
            try:
                with conn:
                    conn.execute("UPDATE records SET tombstoned = 1 WHERE source_id = ?", (source_id,))
            finally:
                conn.close()
    return {"decision": decision, "source_id": source_id}


class KnowledgeHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleKnowledge/1"

    def _send(self, status, payload):
        body = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._send(status, {"ok": False, "error": {"code": code, "message": message}})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            self._send(200, {"ok": True, "service": "knowledge"})
        elif url.path == "/ready":
            self._send(200, {"ok": True, "ready": True, "index": INDEX_URL})
        elif url.path.startswith("/records/"):
            record = get_record(unquote(url.path[len("/records/"):]))
            if record is None:
                self._error(404, "not_found", "unknown record")
            else:
                self._send(200, {"ok": True, "record": record})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        url = urlparse(self.path)
        parts = url.path.strip("/").split("/")
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(400, "invalid_request", "request body must be between 1 byte and 8 MiB")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise KnowledgeError("invalid_request", "body must be a JSON object")
            if url.path == "/records":
                result = store_record(body)
            elif len(parts) == 3 and parts[0] == "records" and parts[2] == "tombstone":
                result = tombstone_record(unquote(parts[1]), body.get("revision_sequence"))
            else:
                self._error(404, "not_found", "unknown path")
                return
        except KnowledgeError as exc:
            self._error(exc.status, exc.code, str(exc))
            return
        except (ValueError, UnicodeDecodeError) as exc:
            self._error(400, "invalid_request", str(exc))
            return
        self._send(200, dict(result, ok=True))

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("knowledge service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), KnowledgeHandler)


def main():
    host = os.environ.get("ENSEMBLE_KNOWLEDGE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_KNOWLEDGE_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
