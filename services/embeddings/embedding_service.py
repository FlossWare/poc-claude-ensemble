#!/usr/bin/env python3
"""Loopback REST embedding service for Claude Ensemble.

Endpoints
  GET  /health       liveness; reports whether the model loaded
  GET  /model        model identity, dimensions, limits
  POST /embeddings   {"texts": [...], "mode": "document" | "query"}
                     -> {"ok": true, "model": ..., "dimensions": ..., "embeddings": [[...], ...]}

Only loopback binds are accepted. Model calls are serialized with a lock.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_lib import DIMENSIONS, MODEL_ID, Embedder  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
MAX_BODY_BYTES = 1_048_576
MAX_TEXTS = 64
MAX_TEXT_CHARS = 65_536
MODES = {"document", "query"}

_embedder = None
_load_error = None
_MODEL_LOCK = threading.Lock()   # one inference at a time; model execution is serialized here


def _model_metadata():
    info = getattr(_embedder, "info", {}) if _embedder is not None else {}
    return {
        "backend": info.get("backend"),
        "device": info.get("device"),
        "precision": info.get("precision"),
        "variant": info.get("variant"),
        "profile": info.get("profile"),
        "fallback_reason": info.get("fallback_reason"),
    }


def _load():
    global _embedder, _load_error
    try:
        _embedder = Embedder(threads=int(os.environ.get("ENSEMBLE_EMBED_THREADS", "8")))
    except Exception as exc:  # reported via /health and 503 responses
        _load_error = str(exc)


class EmbeddingHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleEmbedding/1"

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
            ready = _embedder is not None
            self._send(200 if ready else 503, {"ok": ready, "service": "embedding", "model_loaded": ready,
                                              "error": None if ready else _load_error})
            return
        if self.path == "/model":
            self._send(200, {"ok": True, "model": MODEL_ID, "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
                             "dimensions": DIMENSIONS, "max_texts": MAX_TEXTS,
                             "max_text_chars": MAX_TEXT_CHARS, "modes": sorted(MODES),
                             **_model_metadata()})
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/embeddings":
            self._error(404, "not_found", "unknown path")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            self._error(400, "invalid_request", "request body is required")
            return
        if length > MAX_BODY_BYTES:
            self._error(413, "payload_too_large", "request body exceeds 1 MiB")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid_request", "body must be UTF-8 JSON")
            return
        if not isinstance(body, dict):
            self._error(400, "invalid_request", "body must be a JSON object")
            return
        texts, mode = body.get("texts"), body.get("mode", "document")
        if mode not in MODES:
            self._error(400, "invalid_request", "mode must be 'document' or 'query'")
            return
        if not isinstance(texts, list) or not texts:
            self._error(400, "invalid_request", "texts must be a non-empty list")
            return
        if len(texts) > MAX_TEXTS:
            self._error(413, "payload_too_large", f"at most {MAX_TEXTS} texts per request")
            return
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                self._error(400, "invalid_request", "every text must be a non-empty string")
                return
            if len(text) > MAX_TEXT_CHARS:
                self._error(413, "payload_too_large", f"each text is limited to {MAX_TEXT_CHARS} characters")
                return
        if _embedder is None:
            self._error(503, "service_unavailable", f"model not loaded: {_load_error}")
            return
        with _MODEL_LOCK:
            if mode == "query":
                vectors = _embedder.embed_queries(texts)
            else:
                vectors = _embedder.embed_documents(texts)
        self._send(200, {"ok": True, "model": MODEL_ID, "dimensions": DIMENSIONS,
                         "profile": getattr(_embedder, "profile", None), "embeddings": vectors.tolist()})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("embedding service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), EmbeddingHandler)


def main():
    host = os.environ.get("ENSEMBLE_EMBED_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_EMBED_PORT", str(DEFAULT_PORT)))
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
