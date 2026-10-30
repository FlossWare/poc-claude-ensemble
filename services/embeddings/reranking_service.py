#!/usr/bin/env python3
"""Loopback REST reranking service backed by a local cross-encoder (ms-marco-MiniLM-L6-v2).

A cross-encoder reads each query and candidate together, which is more precise than
comparing vectors. It only ever scores a small candidate set supplied by the caller.
The model is loaded from disk; nothing is downloaded at runtime.

Endpoints
  GET  /health
  POST /rerank   {"query": "...", "documents": ["...", ...]} -> {"scores": [...]}
                 scores are aligned with the input order and are specific to this model
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8774
MODEL_DIR = os.path.expanduser(os.environ.get("ENSEMBLE_RERANK_MODEL_DIR",
    "~/.FlossWare/models/rerankers/cross-encoder/ms-marco-MiniLM-L6-v2/233902d25c440f23af6f7d6e94d2946bac0bee0a"))
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2 (local)"
MAX_BODY_BYTES = 1_048_576
MAX_DOCUMENTS = 50
MAX_DOCUMENT_CHARS = 4_000
MAX_QUERY_CHARS = 2_000

_model = None
_load_error = None
_MODEL_LOCK = threading.Lock()   # the cross-encoder is called from one thread at a time


def _load():
    global _model, _load_error
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_DIR, device="cpu")
    except Exception as exc:  # reported via /health and 503 responses
        _load_error = str(exc)


class RerankHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleRerank/1"

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
            ready = _model is not None
            self._send(200 if ready else 503, {"ok": ready, "service": "reranking", "model_loaded": ready,
                                              "model": MODEL_ID, "error": None if ready else _load_error})
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/rerank":
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
        query, documents = body.get("query"), body.get("documents")
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            self._error(400, "invalid_request", f"query must be a non-empty string of at most {MAX_QUERY_CHARS} characters")
            return
        if not isinstance(documents, list) or not documents or len(documents) > MAX_DOCUMENTS:
            self._error(400, "invalid_request", f"documents must be a list of 1 to {MAX_DOCUMENTS} strings")
            return
        for document in documents:
            if not isinstance(document, str) or len(document) > MAX_DOCUMENT_CHARS:
                self._error(400, "invalid_request", f"each document must be a string of at most {MAX_DOCUMENT_CHARS} characters")
                return
        if _model is None:
            self._error(503, "service_unavailable", f"model not loaded: {_load_error}")
            return
        with _MODEL_LOCK:
            scores = _model.predict([(query, document) for document in documents])
        self._send(200, {"ok": True, "model": MODEL_ID, "scores": [float(s) for s in scores]})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("reranking service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), RerankHandler)


def main():
    host = os.environ.get("ENSEMBLE_RERANK_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_RERANK_PORT", str(DEFAULT_PORT)))
    _load()
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
