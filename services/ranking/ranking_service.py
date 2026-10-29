#!/usr/bin/env python3
"""Loopback REST ranking: fuses ranked candidate lists and explains every result's position.

Reciprocal rank fusion: each list contributes 1/(k + rank) for every candidate it contains. Each result
returns its fused score and the per-list contribution, so the order can always be explained. Ties break
on the candidate key, so the same input always gives the same order.

Endpoints
  GET  /health
  GET  /ready
  POST /rank   {"lists": {"lexical": [{"key", "rank"}, ...], "vector": [...]}, "top_k": n, "k": 60}
               -> {"results": [{"key", "score", "contributions": {"lexical": 0.016, ...}}]}
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8780
MAX_BODY_BYTES = 4 * 1_048_576
MAX_LISTS = 8
MAX_CANDIDATES_PER_LIST = 2000
MAX_TOP_K = 500
DEFAULT_K = 60


def fuse(lists, top_k, k=DEFAULT_K):
    if not isinstance(lists, dict) or not lists:
        raise ValueError("lists must be a non-empty object of named ranked lists")
    if len(lists) > MAX_LISTS:
        raise ValueError(f"at most {MAX_LISTS} lists")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
    scores, contributions = {}, {}
    for name, items in lists.items():
        if not isinstance(name, str) or not isinstance(items, list):
            raise ValueError("each list must be a named array of {key, rank}")
        if len(items) > MAX_CANDIDATES_PER_LIST:
            raise ValueError(f"at most {MAX_CANDIDATES_PER_LIST} candidates per list")
        seen = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str) or not item["key"]:
                raise ValueError("every candidate needs a non-empty string key")
            rank = item.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("every candidate needs a positive integer rank")
            if item["key"] in seen:
                continue   # a duplicate within one list adds nothing
            seen.add(item["key"])
            share = 1.0 / (k + rank)
            scores[item["key"]] = scores.get(item["key"], 0.0) + share
            contributions.setdefault(item["key"], {})[name] = round(share, 6)
    order = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [{"key": key, "score": round(scores[key], 6), "contributions": contributions[key]} for key in order]


class RankingHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleRanking/1"

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
            self._send(200, {"ok": True, "service": "ranking"})
        elif self.path == "/ready":
            self._send(200, {"ok": True, "ready": True, "method": "reciprocal-rank-fusion"})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/rank":
            self._error(404, "not_found", "unknown path")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(400, "invalid_request", "request body must be between 1 byte and 4 MiB")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            results = fuse(body.get("lists"), body.get("top_k", 10), body.get("k", DEFAULT_K))
        except (ValueError, UnicodeDecodeError) as exc:
            self._error(400, "invalid_request", str(exc))
            return
        self._send(200, {"ok": True, "results": results})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("ranking service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), RankingHandler)


def main():
    host = os.environ.get("ENSEMBLE_RANKING_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_RANKING_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
