#!/usr/bin/env python3
"""Loopback REST entity and identifier extraction. Preserves exact spelling and reports uncertainty.

Exact identifiers are matched by pattern and reported with exact=true and confidence 1.0: UUIDs, URLs,
file paths, issue keys (CPSEARCH-10177), version strings, and dotted or dashed technical names.
Candidate names that are not certain (capitalised words that may be a product or a person) are reported
with exact=false and a lower confidence, so callers never treat a guess as an identifier.

Endpoints
  GET  /health
  GET  /ready
  POST /extract   {"text": "..."} -> {"entities": [{"type", "text", "start", "end", "exact", "confidence"}]}
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8778
MAX_BODY_BYTES = 4 * 1_048_576
MAX_TEXT_CHARS = 1_000_000

PATTERNS = [
    ("url", re.compile(r"https?://[^\s<>\"')]+"), True),
    ("uuid", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), True),
    ("issue_key", re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,7}\b"), True),
    ("path", re.compile(r"(?<![\w/])(?:~|\.{1,2})?/[\w.\-/]+|(?<![\w/])[\w.\-]+(?:/[\w.\-]+)+"), True),
    ("file", re.compile(r"\b[\w\-]+\.(?:py|js|ts|sh|md|json|yaml|yml|toml|txt|java|go|rs|c|h|sql|service)\b"), True),
    ("version", re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"), True),
    ("technical_name", re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+\b"), True),
    ("capitalised_name", re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"), False),
]
CONFIDENCE_EXACT = 1.0
CONFIDENCE_UNCERTAIN = 0.4


def extract(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise OverflowError(f"text is limited to {MAX_TEXT_CHARS} characters")
    found, taken = [], []
    for kind, pattern, exact in PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(not (end <= s or start >= e) for s, e in taken):
                continue
            if exact:
                taken.append((start, end))
            found.append({"type": kind, "text": match.group(0), "start": start, "end": end,
                          "exact": exact, "confidence": CONFIDENCE_EXACT if exact else CONFIDENCE_UNCERTAIN})
    found.sort(key=lambda e: (e["start"], e["end"]))
    return found


class ExtractionHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleExtraction/1"

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
            self._send(200, {"ok": True, "service": "extraction"})
        elif self.path == "/ready":
            self._send(200, {"ok": True, "ready": True})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/extract":
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
            entities = extract(body.get("text"))
        except OverflowError as exc:
            self._error(413, "payload_too_large", str(exc))
            return
        except (ValueError, UnicodeDecodeError) as exc:
            self._error(400, "invalid_request", str(exc))
            return
        self._send(200, {"ok": True, "entities": entities})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("extraction service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), ExtractionHandler)


def main():
    host = os.environ.get("ENSEMBLE_EXTRACTION_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_EXTRACTION_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
