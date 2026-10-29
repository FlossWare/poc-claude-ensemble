#!/usr/bin/env python3
"""Loopback REST tokenization service: versioned, deterministic analyzers for index and query text.

Index and query text must be analyzed the same way, or matches silently fail. Analyzers are
versioned profiles, so a change in tokenization is an explicit new version rather than silent drift.

Profiles
  lexical-v1  Unicode case folding; letter and digit runs of 2+ characters; compound identifiers
              (paths, UUIDs, dotted or dashed names) are kept whole AND split into components, so
              "memory-service" matches both "memory-service" and "memory". Offsets are code-point exact.

Endpoints
  GET  /health
  GET  /ready
  GET  /profiles
  POST /tokenize   {"profile": "lexical-v1", "text": "..."} -> {"tokens": [{"term", "start", "end", "kind"}]}
                   kind is "whole" for an identifier kept intact, or "part" for a component of one.
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8776
MAX_BODY_BYTES = 8 * 1_048_576
MAX_TEXT_CHARS = 2_000_000
PROFILES = {"lexical-v1": {"version": "1", "min_length": 2}}

# A compound identifier: letters and digits joined by . _ : - (paths, UUIDs, dotted names).
COMPOUND = re.compile(r"[^\W_]+(?:[._:\-/][^\W_]+)+", re.UNICODE)
SIMPLE = re.compile(r"[^\W_]+", re.UNICODE)


class TokenizationError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def tokenize(profile, text):
    if profile not in PROFILES:
        raise TokenizationError("unsupported_profile", f"unknown tokenization profile: {profile}", status=422)
    if not isinstance(text, str):
        raise TokenizationError("invalid_request", "text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise TokenizationError("payload_too_large", f"text is limited to {MAX_TEXT_CHARS} characters", status=413)
    min_length = PROFILES[profile]["min_length"]
    folded = text.casefold()
    if len(folded) != len(text):
        # casefold can change length (e.g. German sharp s). Offsets must still point into the original
        # text, so fall back to per-character folding that keeps one output character per input character.
        folded = "".join(ch.casefold() if len(ch.casefold()) == 1 else ch.lower() for ch in text)
    tokens, spans = [], []
    for match in COMPOUND.finditer(folded):
        term = match.group(0)
        if len(term) >= min_length:
            tokens.append({"term": term, "start": match.start(), "end": match.end(), "kind": "whole"})
            spans.append((match.start(), match.end()))
    for match in SIMPLE.finditer(folded):
        term = match.group(0)
        if len(term) < min_length:
            continue
        inside_whole = any(s <= match.start() and match.end() <= e for s, e in spans)
        if inside_whole and (match.start(), match.end()) in spans:
            continue
        tokens.append({"term": term, "start": match.start(), "end": match.end(),
                       "kind": "part" if inside_whole else "whole"})
    tokens.sort(key=lambda t: (t["start"], t["end"], t["kind"]))
    return tokens


def describe_profiles():
    return [{"name": name, "version": spec["version"], "parameters": {"min_length": spec["min_length"]}}
            for name, spec in sorted(PROFILES.items())]


class TokenizationHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleTokenization/1"

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
        if self.path == "/health":
            self._send(200, {"ok": True, "service": "tokenization"})
        elif self.path == "/ready":
            self._send(200, {"ok": True, "ready": True, "profiles": sorted(PROFILES)})
        elif self.path == "/profiles":
            self._send(200, {"ok": True, "profiles": describe_profiles()})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/tokenize":
            self._error(404, "not_found", "unknown path")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(400, "invalid_request", "request body must be between 1 byte and 8 MiB")
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
            tokens = tokenize(body.get("profile"), body.get("text"))
        except TokenizationError as exc:
            self._error(exc.status, exc.code, str(exc))
            return
        self._send(200, {"ok": True, "profile": {"name": body["profile"], "version": PROFILES[body["profile"]]["version"]},
                         "tokens": tokens})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("tokenization service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), TokenizationHandler)


def main():
    host = os.environ.get("ENSEMBLE_TOKENIZATION_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_TOKENIZATION_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
