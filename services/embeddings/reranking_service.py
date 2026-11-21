#!/usr/bin/env python3
"""Loopback REST reranking service backed by a local cross-encoder (ms-marco-MiniLM-L6-v2).

A cross-encoder reads each query and candidate together, which is more precise than
comparing vectors. It only ever scores a small candidate set supplied by the caller.
The model is loaded from disk; nothing is downloaded at runtime.

Backends (mirrors the embedding service pattern in embed_lib.py):

  * ``openvino``  -- the IR shipped in the model directory, compiled for the
    configured device. ``GPU`` (the Arc iGPU, FP16) is the default when the
    device is available; ``CPU`` is the fallback. Measured on this host:
    ~7500 pairs/s on GPU vs ~80 pairs/s for sentence-transformers CPU, with
    identical rankings (the IR logits reproduce the CrossEncoder scores).
  * ``sentence-transformers`` -- the original backend, kept as an explicit
    override and as the fallback when OpenVINO cannot load the model.

Selection: ``POC_CLAUDE_ENSEMBLE_RERANK_BACKEND`` (``auto`` | ``openvino`` |
``sentence-transformers``, default ``auto``) and
``POC_CLAUDE_ENSEMBLE_RERANK_DEVICE`` (``GPU`` | ``CPU``, default ``GPU``).
``auto`` prefers OpenVINO and falls back to sentence-transformers only if the
IR or the device fails to load; the fallback is reported honestly via
``/health`` and in the ``backend`` field of every response.

Score contract: raw logits, aligned with the input document order -- exactly
what ``CrossEncoder.predict`` returned before, so existing callers see no
change when the backend switches.

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
MODEL_DIR = os.path.expanduser(os.environ.get("POC_CLAUDE_ENSEMBLE_RERANK_MODEL_DIR",
    "~/.FlossWare/models/rerankers/cross-encoder/ms-marco-MiniLM-L6-v2/233902d25c440f23af6f7d6e94d2946bac0bee0a"))
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2 (local)"
IR_PATH = os.path.join(MODEL_DIR, "openvino", "openvino_model.xml")
MAX_BODY_BYTES = 1_048_576
MAX_DOCUMENTS = 50
MAX_DOCUMENT_CHARS = 4_000
MAX_QUERY_CHARS = 2_000
MAX_LENGTH = 512

BACKEND_ENV = "POC_CLAUDE_ENSEMBLE_RERANK_BACKEND"
DEVICE_ENV = "POC_CLAUDE_ENSEMBLE_RERANK_DEVICE"

_model = None
_load_error = None
_MODEL_LOCK = threading.Lock()   # the cross-encoder is called from one thread at a time


class _OpenVinoReranker:
    """Cross-encoder scoring through the local OpenVINO IR.

    ``predict(pairs)`` returns raw logits, matching the sentence-transformers
    contract for this model (its default activation is the identity).
    """

    def __init__(self, device: str):
        import numpy as np
        import openvino as ov
        from tokenizers import Tokenizer

        self._np = np
        self.runtime = f"openvino/{ov.__version__}"
        self.device = device
        self._core = ov.Core()
        available = self._core.available_devices
        # OpenVINO lists multi-instance devices as GPU.0, GPU.1, ... while
        # compile_model accepts the bare "GPU" alias for any of them.
        if not any(d == device or d.startswith(device + ".") for d in available):
            raise RuntimeError(
                f"requested device {device!r} not available (have {available})")
        self._compiled = self._core.compile_model(IR_PATH, device)
        self._tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_LENGTH)
        self._tokenizer.enable_padding(direction="right", pad_id=0, pad_type_id=0)

    def predict(self, pairs):
        np = self._np
        encodings = self._tokenizer.encode_batch(
            [(query, document) for query, document in pairs])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        output = self._compiled({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        logits = np.asarray(output[self._compiled.output(0)], dtype=np.float32).reshape(-1)
        return [float(value) for value in logits]


def _load():
    global _model, _load_error
    backend = os.environ.get(BACKEND_ENV, "auto").strip().lower() or "auto"
    device = os.environ.get(DEVICE_ENV, "GPU").strip().upper() or "GPU"
    _load_error = None
    if backend in ("auto", "openvino"):
        try:
            _model = _OpenVinoReranker(device)
            return
        except Exception as exc:
            if backend == "openvino":
                _load_error = f"openvino backend failed: {exc}"
                return
            _load_error = f"openvino backend failed ({exc}); falling back to sentence-transformers"
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_DIR, device="cpu")
    except Exception as exc:  # reported via /health and 503 responses
        _load_error = str(exc) if _load_error is None else f"{_load_error}; {exc}"


def _backend_name():
    if _model is None:
        return None
    return "openvino" if isinstance(_model, _OpenVinoReranker) else "sentence-transformers"


class RerankHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleRerank/2"

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
                                              "model": MODEL_ID,
                                              "backend": _backend_name(),
                                              "device": getattr(_model, "device", "cpu"),
                                              "error": None if ready else _load_error})
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
        self._send(200, {"ok": True, "model": MODEL_ID, "backend": _backend_name(),
                         "scores": [float(s) for s in scores]})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("reranking service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), RerankHandler)


def main():
    host = os.environ.get("POC_CLAUDE_ENSEMBLE_RERANK_HOST", DEFAULT_HOST)
    port = int(os.environ.get("POC_CLAUDE_ENSEMBLE_RERANK_PORT", str(DEFAULT_PORT)))
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
