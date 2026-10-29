#!/usr/bin/env python3
"""Loopback REST intent classification. Returns a label, a confidence, and the rules that fired.

Rules are explicit and inspectable: each label has cue patterns, and every match is returned as evidence.
Confidence is the share of matched cues for the winning label, scaled by how far it leads the runner-up.
When nothing matches, the label is "other" with confidence 0.0, so a caller can tell "no evidence" apart
from a confident answer.

Labels: history (what did we do, last time), debug (why does X fail), howto (how do I, how does),
lookup (what is, where is, which), change (implement, add, fix, refactor), other.

Endpoints
  GET  /health
  GET  /ready
  POST /classify   {"text": "..."} -> {"label", "confidence", "evidence": [{"rule", "text"}], "scores"}
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8779
MAX_BODY_BYTES = 262_144
MAX_TEXT_CHARS = 20_000

RULES = {
    "history": [r"\bwhat did we\b", r"\blast time\b", r"\bearlier\b", r"\bwe (?:decided|discussed|did)\b",
                r"\bremember\b", r"\bpreviously\b", r"\bwhat have (?:we|you)\b"],
    "debug": [r"\bwhy (?:is|does|did|isn'?t|doesn'?t|am i)\b", r"\b(?:error|fails?|failing|broken|crash\w*|exception|timeout)\b",
              r"\bnot working\b", r"\bwhat'?s wrong\b"],
    "howto": [r"\bhow (?:do|does|can|should|to)\b", r"\bsteps? to\b", r"\bexample of\b"],
    "lookup": [r"\bwhat (?:is|are)\b", r"\bwhere (?:is|are|do)\b", r"\bwhich\b", r"\bwho\b", r"\bshow me\b"],
    "change": [r"\b(?:implement|add|build|fix|refactor|rename|remove|delete|update|change)\b",
               r"\bcan you (?:make|write|create)\b"],
}
COMPILED = {label: [re.compile(p, re.IGNORECASE) for p in patterns] for label, patterns in RULES.items()}


def classify(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise OverflowError(f"text is limited to {MAX_TEXT_CHARS} characters")
    evidence, counts = [], {}
    for label, patterns in COMPILED.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                counts[label] = counts.get(label, 0) + 1
                evidence.append({"rule": label, "text": match.group(0)})
    if not counts:
        return {"label": "other", "confidence": 0.0, "evidence": [], "scores": {}}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    label, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    total = sum(counts.values())
    confidence = round((top / total) * (top / (top + runner_up)), 3)
    return {"label": label, "confidence": confidence, "evidence": evidence, "scores": dict(ranked)}


class IntentHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleIntent/1"

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
            self._send(200, {"ok": True, "service": "intent"})
        elif self.path == "/ready":
            self._send(200, {"ok": True, "ready": True, "labels": sorted(RULES)})
        else:
            self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path != "/classify":
            self._error(404, "not_found", "unknown path")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(400, "invalid_request", "request body must be between 1 byte and 256 KiB")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            result = classify(body.get("text"))
        except OverflowError as exc:
            self._error(413, "payload_too_large", str(exc))
            return
        except (ValueError, UnicodeDecodeError) as exc:
            self._error(400, "invalid_request", str(exc))
            return
        self._send(200, dict(result, ok=True))

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("intent service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), IntentHandler)


def main():
    host = os.environ.get("ENSEMBLE_INTENT_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_INTENT_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
