#!/usr/bin/env python3
"""Loopback REST vector service backed by ChromaDB.

It is the only writer to the vector store. Queries are exact: all vectors in a
collection are scored with a dot product, since vectors are normalized. The
matrix is cached per collection and rebuilt when the collection's count changes.

Endpoints
  GET  /health
  GET  /collections                          names and counts
  POST /collections/{name}/existing          {"ids": [...]} -> ids already stored
  POST /collections/{name}/upsert            {"ids", "embeddings", "documents", "metadatas"}
  POST /collections/{name}/query             {"embedding": [...], "top_k": n} -> exact nearest
  POST /collections/{name}/delete-superseded {"url", "keep_sha256"} -> removes other revisions of a source
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_lib import DIMENSIONS  # noqa: E402
from vector_store import client, collection  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8771
MAX_BODY_BYTES = 16 * 1_048_576
MAX_UPSERT = 256
MAX_TOP_K = 200

_lock = threading.Lock()
_cache = {}  # name -> {"ids", "documents", "buf" (grows by doubling), "rows", "index"}


def _ensure(name):
    """Load a collection into memory once; later upserts are applied to it incrementally."""
    entry = _cache.get(name)
    if entry is not None:
        return entry
    col = collection(name)
    ids, documents, metadatas, vectors = [], [], [], []
    for offset in range(0, col.count(), 2000):   # paged: one large read exceeds SQLite's variable limit
        page = col.get(include=["embeddings", "documents", "metadatas"], limit=2000, offset=offset)
        ids.extend(page["ids"])
        documents.extend(page["documents"])
        metadatas.extend(dict(m or {}) for m in page["metadatas"])
        vectors.extend(page["embeddings"])
    capacity = max(1024, 2 * len(ids))
    buf = np.zeros((capacity, DIMENSIONS), dtype=np.float32)
    if ids:
        buf[: len(ids)] = np.asarray(vectors, dtype=np.float32)
    entry = {"ids": ids, "documents": documents, "metadatas": metadatas, "buf": buf, "rows": len(ids),
             "index": {record_id: row for row, record_id in enumerate(ids)}}
    _cache[name] = entry
    return entry


def _apply_upsert(name, ids, documents, matrix, metadatas):
    entry = _ensure(name)
    for record_id, document, vector, metadata in zip(ids, documents, matrix, metadatas):
        row = entry["index"].get(record_id)
        if row is None:
            if entry["rows"] == entry["buf"].shape[0]:
                grown = np.zeros((2 * entry["buf"].shape[0], DIMENSIONS), dtype=np.float32)
                grown[: entry["rows"]] = entry["buf"][: entry["rows"]]
                entry["buf"] = grown
            row = entry["rows"]
            entry["rows"] += 1
            entry["index"][record_id] = row
            entry["ids"].append(record_id)
            entry["documents"].append(document)
            entry["metadatas"].append(metadata)
        else:
            entry["documents"][row] = document
            entry["metadatas"][row] = metadata
        entry["buf"][row] = vector


class VectorHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleVector/1"

    def _send(self, status, payload):
        body = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._send(status, {"ok": False, "error": {"code": code, "message": message}})

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body must be between 1 byte and 16 MiB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    def _route(self):
        parts = self.path.strip("/").split("/")
        return parts

    def do_GET(self):
        parts = self._route()
        if parts == ["health"]:
            try:
                client().heartbeat()
                self._send(200, {"ok": True, "service": "vector", "backend": "chromadb"})
            except Exception as exc:
                self._error(503, "service_unavailable", str(exc))
            return
        if parts == ["collections"]:
            cols = [{"name": c.name, "count": c.count()} for c in client().list_collections()]
            self._send(200, {"ok": True, "collections": cols})
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        parts = self._route()
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", str(exc))
            return
        if len(parts) == 3 and parts[0] == "collections":
            name, action = parts[1], parts[2]
            try:
                if action == "existing":
                    ids = body.get("ids", [])
                    if not isinstance(ids, list):
                        raise ValueError("ids must be a list")
                    found = []
                    col = collection(name)
                    for start in range(0, len(ids), 1000):
                        found.extend(col.get(ids=ids[start : start + 1000], include=[])["ids"])
                    self._send(200, {"ok": True, "found": found})
                    return
                if action == "upsert":
                    self._upsert(name, body)
                    return
                if action == "metadata":
                    self._metadata(name, body)
                    return
                if action == "delete-superseded":
                    self._delete_superseded(name, body)
                    return
                if action == "query":
                    self._query(name, body)
                    return
            except ValueError as exc:
                self._error(400, "invalid_request", str(exc))
                return
            except RuntimeError as exc:
                self._error(409, "collection_mismatch", str(exc))
                return
            except Exception as exc:
                self._error(500, "internal_error", str(exc))
                return
        self._error(404, "not_found", "unknown path")

    def _upsert(self, name, body):
        ids, embeddings = body.get("ids"), body.get("embeddings")
        documents, metadatas = body.get("documents"), body.get("metadatas")
        if not (isinstance(ids, list) and isinstance(embeddings, list) and isinstance(documents, list)):
            raise ValueError("ids, embeddings, and documents must be lists")
        if not (len(ids) == len(embeddings) == len(documents)) or not ids:
            raise ValueError("ids, embeddings, and documents must be non-empty and the same length")
        if len(ids) > MAX_UPSERT:
            raise ValueError(f"at most {MAX_UPSERT} records per upsert")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != DIMENSIONS or not np.isfinite(matrix).all():
            raise ValueError(f"embeddings must be finite vectors of {DIMENSIONS} dimensions")
        if not isinstance(metadatas, list) or len(metadatas) != len(ids):
            raise ValueError("metadatas must be a list matching ids in length")
        # ChromaDB rejects empty metadata objects, so every record must carry at least one attribute.
        if not all(isinstance(m, dict) and m for m in metadatas):
            raise ValueError("each record needs a non-empty metadata object")
        with _lock:
            collection(name).upsert(ids=ids, embeddings=matrix.tolist(), documents=documents, metadatas=metadatas)
            _apply_upsert(name, ids, documents, matrix, metadatas)
        self._send(200, {"ok": True, "upserted": len(ids)})

    def _metadata(self, name, body):
        ids, metadatas = body.get("ids"), body.get("metadatas")
        if not (isinstance(ids, list) and isinstance(metadatas, list) and len(ids) == len(metadatas) and ids):
            raise ValueError("ids and metadatas must be non-empty lists of equal length")
        if len(ids) > MAX_UPSERT:
            raise ValueError(f"at most {MAX_UPSERT} records per metadata update")
        with _lock:
            col = collection(name)
            present = set(col.get(ids=ids, include=[])["ids"])
            keep = [(i, m) for i, m in zip(ids, metadatas) if i in present]
            if keep:
                col.update(ids=[i for i, _ in keep], metadatas=[m for _, m in keep])
                entry = _cache.get(name)
                if entry is not None:
                    for record_id, metadata in keep:
                        row = entry["index"].get(record_id)
                        if row is not None:
                            entry["metadatas"][row] = dict(metadata)
        self._send(200, {"ok": True, "updated": len(keep)})

    def _delete_superseded(self, name, body):
        """Remove vectors of other revisions of one source. Used when a document's content changes."""
        url, keep = body.get("url"), body.get("keep_sha256")
        if not isinstance(url, str) or not url or len(url) > 2048:
            raise ValueError("url must be a non-empty string of at most 2048 characters")
        if not isinstance(keep, str) or len(keep) != 64 or any(c not in "0123456789abcdef" for c in keep):
            raise ValueError("keep_sha256 must be a lowercase hex SHA-256")
        with _lock:
            col = collection(name)
            stale = col.get(where={"$and": [{"url": {"$eq": url}}, {"content_sha256": {"$ne": keep}}]},
                            include=[])["ids"]
            if stale:
                col.delete(ids=stale)
                _cache.pop(name, None)   # the in-memory matrix is rebuilt from the store on next use
        self._send(200, {"ok": True, "deleted": len(stale)})

    def _query(self, name, body):
        vector = body.get("embedding")
        top_k = body.get("top_k", 10)
        if not isinstance(vector, list):
            raise ValueError("embedding must be a list")
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (DIMENSIONS,) or not np.isfinite(query).all():
            raise ValueError(f"embedding must be a finite vector of {DIMENSIONS} dimensions")
        with _lock:
            entry = _ensure(name)
            rows = entry["rows"]
            if rows == 0:
                self._send(200, {"ok": True, "results": []})
                return
            scores = entry["buf"][:rows] @ query
            order = np.argsort(-scores)[:top_k]
            results = [{"id": entry["ids"][i], "document": entry["documents"][i], "score": float(scores[i]),
                        "metadata": entry["metadatas"][i]}
                       for i in order]
        self._send(200, {"ok": True, "results": results})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("vector service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), VectorHandler)


def main():
    host = os.environ.get("ENSEMBLE_VECTOR_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_VECTOR_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
