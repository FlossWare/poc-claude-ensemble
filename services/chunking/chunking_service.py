#!/usr/bin/env python3
"""Loopback REST chunking service: deterministic, versioned chunk boundaries with provenance.

Each profile is a name and a version. Given the same profile, source ID, and text, the service
always returns the same chunks with the same IDs. Chunk IDs include the profile name and version,
so a chunk from one profile can never be mistaken for a chunk from another.

Profiles
  window-v1    fixed character windows with overlap (1800 chars, 200 overlap)
  markdown-v1  splits at Markdown headings; oversized sections fall back to windows (2000 chars)
  message-v1   one chunk per message, clipped at 20000 chars

Endpoints
  GET  /health
  GET  /ready         true when every profile is loaded
  GET  /profiles      name, version, and parameters of each profile
  POST /chunk         {"profile": "...", "source_id": "...", "media_type": "text/plain|text/markdown",
                       "text": "..."} -> {"chunks": [{"chunk_id", "index", "start", "end", "header", "text"}]}
"""

import hashlib
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8775
MAX_BODY_BYTES = 8 * 1_048_576
MAX_TEXT_CHARS = 4_000_000
MAX_SOURCE_ID_CHARS = 512
MEDIA_TYPES = ("text/plain", "text/markdown")

PROFILES = {
    "window-v1": {"version": "1", "strategy": "window", "size": 1800, "overlap": 200},
    "markdown-v1": {"version": "1", "strategy": "markdown", "size": 2000},
    "message-v1": {"version": "1", "strategy": "message", "size": 20000},
}

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class ChunkingError(ValueError):
    """Raised for invalid requests. `code` distinguishes unknown profiles from malformed input."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def _chunk_id(profile, source_id, start, end):
    key = f"{profile}|{PROFILES[profile]['version']}|{source_id}|{start}|{end}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _windows(text, size, overlap):
    """(start, end) character spans. Overlap keeps context across boundaries."""
    if not text:
        return []
    spans, start, step = [], 0, size - overlap
    while True:
        end = min(start + size, len(text))
        spans.append((start, end))
        if end >= len(text):
            return spans
        start += step


def _markdown_sections(text):
    """(start, end, header) spans, one per heading; text before the first heading gets header ''."""
    spans, line_starts, pos = [], [], 0
    for line in text.splitlines(keepends=True):
        line_starts.append(pos)
        pos += len(line)
    headings = []
    for offset, line in zip(line_starts, text.splitlines()):
        match = HEADING.match(line)
        if match:
            headings.append((offset, match.group(2).strip()))
    if not headings or headings[0][0] > 0:
        headings.insert(0, (0, ""))
    for index, (offset, header) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        if end > offset:
            spans.append((offset, end, header))
    return spans


def chunk(profile_name, source_id, text, media_type):
    """Pure function: profile, source, and text in; chunk records out."""
    if profile_name not in PROFILES:
        raise ChunkingError("unsupported_profile", f"unknown chunking profile: {profile_name}", status=422)
    if not isinstance(source_id, str) or not source_id or len(source_id) > MAX_SOURCE_ID_CHARS:
        raise ChunkingError("invalid_request", "source_id must be a non-empty string of at most 512 characters")
    if media_type not in MEDIA_TYPES:
        raise ChunkingError("unsupported_media_type", "media_type must be text/plain or text/markdown", status=415)
    if not isinstance(text, str):
        raise ChunkingError("invalid_request", "text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise ChunkingError("payload_too_large", f"text is limited to {MAX_TEXT_CHARS} characters", status=413)

    spec = PROFILES[profile_name]
    records = []
    if spec["strategy"] == "window":
        spans = [(s, e, "") for s, e in _windows(text, spec["size"], spec["overlap"])]
    elif spec["strategy"] == "message":
        spans = [(0, min(len(text), spec["size"]), "")] if text else []
    else:  # markdown: headings define sections; sections over the size limit are windowed
        spans = []
        for start, end, header in _markdown_sections(text) if media_type == "text/markdown" else [(0, len(text), "")]:
            section_len = end - start
            if section_len <= spec["size"]:
                spans.append((start, end, header))
            else:
                spans.extend((start + s, start + e, header)
                             for s, e in _windows(text[start:end], spec["size"], 0))
    for index, (start, end, header) in enumerate(spans):
        if end <= start:
            continue
        records.append({"chunk_id": _chunk_id(profile_name, source_id, start, end), "index": index,
                        "start": start, "end": end, "header": header, "text": text[start:end]})
    return records


def describe_profiles():
    return [{"name": name, "version": spec["version"],
             "parameters": {k: v for k, v in spec.items() if k != "version"}}
            for name, spec in sorted(PROFILES.items())]


class ChunkingHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleChunking/1"

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
            self._send(200, {"ok": True, "service": "chunking"})
        elif self.path == "/ready":
            self._send(200, {"ok": True, "ready": True, "profiles": sorted(PROFILES)})
        elif self.path == "/profiles":
            self._send(200, {"ok": True, "profiles": describe_profiles()})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/chunk":
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
            records = chunk(body.get("profile"), body.get("source_id"), body.get("text"),
                            body.get("media_type", "text/plain"))
        except ChunkingError as exc:
            self._error(exc.status, exc.code, str(exc))
            return
        self._send(200, {"ok": True, "profile": {"name": body["profile"], "version": PROFILES[body["profile"]]["version"]},
                         "chunks": records})

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("chunking service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), ChunkingHandler)


def main():
    host = os.environ.get("ENSEMBLE_CHUNKING_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_CHUNKING_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
