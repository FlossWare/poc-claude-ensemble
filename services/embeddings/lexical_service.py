#!/usr/bin/env python3
"""Loopback REST lexical retrieval service backed by the SQLite FTS5 transcript index.

Text is analyzed by the Tokenization service (lexical-v1) on the way in and on the way out,
so an index term and a query term always come from the same analyzer. Each analyzed term
is stored in messages.analyzed and indexed by FTS5 with Porter stemming. Query terms are
OR-combined and ranked by BM25.

The service is the only writer to the transcript database through REST. POST /documents adds
sessions and messages incrementally and never rewrites rows that already exist. A document
whose analysis fails is not written at all. Rows without an analyzed column are invisible
to search until the migration in migrate_lexical_tokenization.py has filled them.

Endpoints
  GET  /health
  POST /search     {"query": "...", "top_k": n} -> ranked candidates with source info
  POST /documents  {"session": {...}, "messages": [...]} -> incremental, idempotent insert

Depends on the Tokenization service (ENSEMBLE_TOKENIZATION_URL). When it is unavailable,
search and document writes return 503 rather than an unanalyzed result.
"""

import hashlib
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8772
DB_PATH = os.environ.get("ENSEMBLE_LEXICAL_DB", os.path.expanduser(
    "~/.FlossWare/poc-claude-ensemble/data/sqlite/transcripts.db"))
MAX_BODY_BYTES = 65_536
MAX_DOCUMENT_BODY_BYTES = 8 * 1_048_576
MAX_QUERY_CHARS = 2_000
MAX_TOP_K = 500
MAX_MESSAGES_PER_DOCUMENT = 500
MAX_MESSAGE_CHARS = 200_000
TOKENIZATION_URL = os.environ.get("ENSEMBLE_TOKENIZATION_URL", "http://127.0.0.1:8776")
TOKENIZE_PROFILE = "lexical-v1"
TOKENIZE_TIMEOUT = 10
# The FTS tokenizer keeps the Tokenization service's compound identifiers (paths, dotted names,
# dashed names, UUIDs) as single tokens. Analyzed text is space-separated, so spaces still split terms.
FTS_TOKENIZE = "porter unicode61 remove_diacritics 2 tokenchars '-._:/'"
_WRITE_LOCK = threading.Lock()


class AnalysisUnavailable(Exception):
    """The Tokenization service could not analyze text. Nothing is written or searched."""


def analyze(text):
    """Return the lexical-v1 terms for text, in order. Whole identifiers and their parts both appear."""
    request = urllib.request.Request(
        f"{TOKENIZATION_URL}/tokenize",
        data=json.dumps({"profile": TOKENIZE_PROFILE, "text": text}).encode("utf-8"),
        method="POST", headers={"content-type": "application/json", "accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TOKENIZE_TIMEOUT) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AnalysisUnavailable(f"tokenization: {exc}") from exc
    if not isinstance(body, dict) or body.get("ok") is not True or not isinstance(body.get("tokens"), list):
        raise AnalysisUnavailable("tokenization: unexpected response")
    return [token["term"] for token in body["tokens"]]


def match_expression(terms):
    """Quote each distinct term so FTS5 treats it as one token, and OR the terms together."""
    seen, quoted = set(), []
    for term in terms:
        if term not in seen:
            seen.add(term)
            quoted.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(quoted)


def _connect_read():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10, check_same_thread=False)


def _connect_write():
    return sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)


def search(query, top_k):
    terms = analyze(query)
    if not terms:
        return []
    match = match_expression(terms)
    conn = _connect_read()
    try:
        rows = conn.execute(
            """SELECT m.text, m.role, m.kind, m.session_id, s.project, s.is_subagent
               FROM messages_fts
               JOIN messages m ON m.id = messages_fts.rowid
               JOIN sessions s ON s.file = m.file
               WHERE messages_fts MATCH ?
               ORDER BY bm25(messages_fts)
               LIMIT ?""",
            (match, top_k * 4),
        ).fetchall()
    finally:
        conn.close()
    seen, results = set(), []
    for text, role, kind, session_id, project, is_subagent in rows:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "text_sha": key, "text": text, "role": role, "kind": kind,
            "session_id": session_id, "project": project, "is_subagent": bool(is_subagent),
        })
        if len(results) == top_k:
            break
    return results


def ingest_document(session, messages):
    """Add one session and its messages. Idempotent: an existing (file, uuid, kind, text) is skipped.

    Analysis happens before any lock is taken or row is written, so a Tokenization failure
    leaves the database untouched and the caller can retry the whole document.
    """
    file_path = session["file"]
    analyzed = [" ".join(analyze(message["text"])) for message in messages]
    inserted = skipped = 0
    with _WRITE_LOCK:
        conn = _connect_write()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, source, project, title, first_ts, last_ts,"
                    " file, is_subagent, bytes) VALUES (?,?,?,?,?,?,?,?,?)",
                    (session["session_id"], session.get("source", "orchestrator"), session["project"],
                     session.get("title"), session.get("first_ts"), session.get("last_ts"), file_path,
                     int(bool(session.get("is_subagent"))), session.get("bytes", 0)),
                )
                for message, terms in zip(messages, analyzed):
                    exists = conn.execute(
                        "SELECT 1 FROM messages WHERE file = ? AND uuid = ? AND kind = ? AND text = ? LIMIT 1",
                        (file_path, message["uuid"], message.get("kind", "text"), message["text"]),
                    ).fetchone()
                    if exists:
                        skipped += 1
                        continue
                    cursor = conn.execute(
                        "INSERT INTO messages (session_id, uuid, role, kind, ts, text, file, analyzed)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (session["session_id"], message["uuid"], message["role"], message.get("kind", "text"),
                         message.get("ts", ""), message["text"], file_path, terms),
                    )
                    conn.execute("INSERT INTO messages_fts (rowid, analyzed) VALUES (?, ?)",
                                 (cursor.lastrowid, terms))
                    inserted += 1
        finally:
            conn.close()
    return {"ok": True, "session_id": session["session_id"], "inserted": inserted, "already_present": skipped}


def _validate_document(body):
    session = body.get("session")
    messages = body.get("messages")
    if not isinstance(session, dict):
        raise ValueError("session must be an object")
    for key in ("session_id", "project", "file"):
        if not isinstance(session.get(key), str) or not session.get(key):
            raise ValueError(f"session.{key} must be a non-empty string")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if len(messages) > MAX_MESSAGES_PER_DOCUMENT:
        raise ValueError(f"at most {MAX_MESSAGES_PER_DOCUMENT} messages per document")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if not isinstance(message.get("uuid"), str) or message.get("role") not in ("user", "assistant"):
            raise ValueError("each message needs a uuid and a role of user or assistant")
        if message.get("kind", "text") not in ("text", "thinking", "tool_use", "tool_result"):
            raise ValueError("kind must be text, thinking, tool_use, or tool_result")
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each message needs non-empty text")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f"each message is limited to {MAX_MESSAGE_CHARS} characters")
    return session, messages


class LexicalHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleLexical/1"

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
        if self.path == "/health":
            try:
                conn = _connect_read()
                conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
                conn.close()
                self._send(200, {"ok": True, "service": "lexical", "backend": "sqlite-fts5"})
            except sqlite3.Error as exc:
                self._error(503, "service_unavailable", str(exc))
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path not in {"/search", "/documents"}:
            self._error(404, "not_found", "unknown path")
            return
        limit = MAX_DOCUMENT_BODY_BYTES if self.path == "/documents" else MAX_BODY_BYTES
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > limit:
            self._error(400, "invalid_request", f"request body must be between 1 byte and {limit} bytes")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid_request", "body must be UTF-8 JSON")
            return
        if not isinstance(body, dict):
            self._error(400, "invalid_request", "body must be a JSON object")
            return
        if self.path == "/documents":
            try:
                session, messages = _validate_document(body)
                self._send(200, ingest_document(session, messages))
            except ValueError as exc:
                self._error(400, "invalid_request", str(exc))
            except (sqlite3.Error, AnalysisUnavailable) as exc:
                self._error(503, "service_unavailable", str(exc))
            return
        query, top_k = body.get("query"), body.get("top_k", 10)
        if not isinstance(query, str) or not query.strip():
            self._error(400, "invalid_request", "query must be a non-empty string")
            return
        if len(query) > MAX_QUERY_CHARS:
            self._error(413, "payload_too_large", f"query is limited to {MAX_QUERY_CHARS} characters")
            return
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            self._error(400, "invalid_request", f"top_k must be an integer from 1 to {MAX_TOP_K}")
            return
        try:
            self._send(200, {"ok": True, "results": search(query, top_k)})
        except (sqlite3.Error, AnalysisUnavailable) as exc:
            self._error(503, "service_unavailable", str(exc))

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("lexical service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), LexicalHandler)


def main():
    host = os.environ.get("ENSEMBLE_LEXICAL_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_LEXICAL_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
