#!/usr/bin/env python3
"""Live acceptance check for POC document and URI ingestion.

Drives the deployed gateway (/api/v1/ingestion/*) with a generated temporary text
fixture, then verifies discovery through Knowledge, Lexical, and Vector and the
session-events ledger. It never fetches external URLs.

Writes happen only when --confirm-ingest is supplied. Without it the run stops after
the preview, which is non-mutating.

Exit codes: 0 passed, 1 failed, 2 preview verified and awaiting confirmation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TIMEOUT_SECONDS = 60
REQUIRED_LEDGER_EVENTS = ("ingestion.started", "ingestion.completed")
FORBIDDEN_LEDGER_KEYS = {"content", "text", "prompt", "payload", "headers", "authorization", "password", "api_key"}
FORBIDDEN_LEDGER_MARKERS = ("bearer ", "authorization:", "api_key=", "password=")


@dataclass
class Config:
    gateway: str = os.environ.get("POC_CLAUDE_ENSEMBLE_GATEWAY_URL", "http://127.0.0.1:8080/api/v1")
    knowledge: str = os.environ.get("POC_CLAUDE_ENSEMBLE_KNOWLEDGE_URL", "http://127.0.0.1:8781")
    lexical: str = os.environ.get("POC_CLAUDE_ENSEMBLE_LEXICAL_URL", "http://127.0.0.1:8772")
    vector: str = os.environ.get("POC_CLAUDE_ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
    embedding: str = os.environ.get("POC_CLAUDE_ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
    confirm: bool = False


class AcceptanceFailure(Exception):
    def __init__(self, step: str, detail: str):
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _call(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, _decode(exc.read())


def _quoted(source_id: str) -> str:
    return urllib.parse.quote(source_id, safe="")


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys |= _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _walk_keys(child)
    return keys


def _cleanup(config: Config, source_id: str) -> dict[str, Any]:
    """Tombstone the Knowledge record. Lexical and Vector rows have no delete API and remain."""
    status, body = _call("POST", f"{config.knowledge}/records/{_quoted(source_id)}/tombstone",
                         {"revision_sequence": time.time_ns()})
    tombstoned = status == 200 and isinstance(body, dict) and body.get("decision") == "tombstoned"
    return {
        "knowledge_tombstoned": tombstoned,
        "http_status": status,
        "residual_indexes": [] if status == 404 else ["lexical", "vector"],
        "note": "Lexical and Vector rows for this generated fixture are not removed by any public API.",
    }


def run(config: Config) -> dict[str, Any]:
    report: dict[str, Any] = {"outcome": "failed", "steps": [], "failed_step": None, "detail": None, "cleanup": None}
    state: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        # Detail explains a failure; passing checks report only their name.
        report["steps"].append({"name": name, "ok": bool(ok), "detail": "" if ok else detail})
        if not ok:
            raise AcceptanceFailure(name, detail or "check failed")

    def passed(name: str, detail: str = "") -> None:
        check(name, True, detail)

    try:
        _run_steps(config, state, check, passed, report)
    except AcceptanceFailure as exc:
        report["failed_step"], report["detail"] = exc.step, exc.detail
    except (urllib.error.URLError, OSError, ValueError) as exc:
        report["failed_step"], report["detail"] = "transport", f"{type(exc).__name__}: {exc}"
    finally:
        if config.confirm and state.get("source_id"):
            report["cleanup"] = _cleanup(config, state["source_id"])
    if report.get("outcome") != "awaiting_confirmation" and report["failed_step"] is None:
        report["outcome"] = "passed"
    return report


def _run_steps(config: Config, state: dict[str, Any], check: Callable[..., None],
               passed: Callable[..., None], report: dict[str, Any]) -> None:
    gateway = config.gateway.rstrip("/")

    status, body = _call("GET", gateway + "/health")
    check("gateway health", status == 200 and isinstance(body, dict) and body.get("ok") is True, f"HTTP {status}")
    status, body = _call("GET", gateway + "/ready")
    check("gateway ready (graph, memory)", status == 200 and isinstance(body, dict) and body.get("ready") is True,
          f"HTTP {status}")
    status, body = _call("GET", gateway + "/ingestion/health")
    check("ingestion health via gateway",
          status == 200 and isinstance(body, dict) and body.get("ok") is True and "http" in body.get("schemes", []),
          f"HTTP {status}")

    token = "ceacceptance" + uuid.uuid4().hex
    text = f"Acceptance fixture {token}. Ingestion must index this sentence for search."
    expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"ce-acceptance-{uuid.uuid4().hex[:12]}.txt"
    with tempfile.TemporaryDirectory(prefix="ce-ingest-acceptance-") as temporary:
        fixture = Path(temporary) / filename
        fixture.write_text(text, encoding="utf-8")
        content = base64.b64encode(fixture.read_bytes()).decode("ascii")
    passed("temporary fixture created", f"{len(text)} characters")

    status, preview = _call("POST", gateway + "/ingestion/preview", {
        "filename": filename, "media_type": "text/plain", "content_base64": content})
    check("preview accepted", status == 200 and isinstance(preview, dict) and preview.get("ok") is True,
          f"HTTP {status}")
    source = preview.get("source", {})
    check("preview is non-mutating", preview.get("persisted") is False, "persisted must be false")
    check("preview content hash matches fixture", source.get("content_sha256") == expected_sha,
          "content_sha256 mismatch")
    check("preview origin and media type", source.get("origin") == "upload" and source.get("media_type") == "text/plain",
          f"origin={source.get('origin')} media_type={source.get('media_type')}")
    check("preview contains fixture text", token in str(preview.get("preview", "")), "token missing from preview")
    state.update(preview_id=preview["preview_id"], source_id=source["source_id"])

    if not config.confirm:
        report["outcome"] = "awaiting_confirmation"
        report["steps"].append({"name": "confirmation gate", "ok": True,
                                "detail": "stopped before ingestion; re-run with --confirm-ingest"})
        return

    status, result = _call("POST", gateway + "/ingestion/ingest", {"preview_id": state["preview_id"]})
    check("ingestion completed", status == 200 and isinstance(result, dict) and result.get("status") == "completed",
          f"HTTP {status} status={result.get('status') if isinstance(result, dict) else None}")
    check("ingestion stored chunks and vectors",
          result.get("chunks", 0) >= 1 and result.get("vectors") == result.get("chunks"),
          f"chunks={result.get('chunks')} vectors={result.get('vectors')}")
    check("ingestion hash matches preview", result.get("content_sha256") == expected_sha, "content_sha256 mismatch")
    ingestion_id = result.get("ingestion_id")
    state["ingestion_id"] = ingestion_id

    status, replay = _call("POST", gateway + "/ingestion/ingest", {"preview_id": state["preview_id"]})
    check("replay of consumed preview refused",
          status == 410 and isinstance(replay, dict) and replay.get("error_code") == "preview_expired",
          f"HTTP {status}")

    source_id = state["source_id"]
    status, record = _call("GET", f"{config.knowledge}/records/{_quoted(source_id)}")
    check("knowledge record discoverable",
          status == 200 and isinstance(record, dict) and record.get("content_sha256") == expected_sha
          and record.get("tombstoned") is False, f"HTTP {status}")

    status, lexical = _call("POST", f"{config.lexical}/knowledge/search", {"query": token, "top_k": 10})
    lexical_hits = lexical.get("results", []) if isinstance(lexical, dict) else []
    check("lexical search finds fixture",
          status == 200 and any(hit.get("url") == source_id for hit in lexical_hits),
          f"HTTP {status}, {len(lexical_hits)} hits")

    status, embedded = _call("POST", f"{config.embedding}/embeddings", {"texts": [text], "mode": "query"})
    vectors = embedded.get("embeddings") if isinstance(embedded, dict) else None
    check("query embedding available", status == 200 and isinstance(vectors, list) and len(vectors) == 1,
          f"HTTP {status}")
    status, vector_hits = _call("POST", f"{config.vector}/collections/knowledge_chunks/query",
                                {"embedding": vectors[0], "top_k": 20})
    matches = vector_hits.get("results", []) if isinstance(vector_hits, dict) else []
    check("vector search finds fixture",
          status == 200 and any((hit.get("metadata") or {}).get("url") == source_id for hit in matches),
          f"HTTP {status}, {len(matches)} hits")

    status, history = _call("GET", gateway + "/ingestion/history?limit=100")
    entries = history.get("entries", []) if isinstance(history, dict) else []
    mine = [entry for entry in entries if entry.get("correlation_id") == ingestion_id]
    names = {entry.get("event_name") for entry in mine}
    check("ledger records lifecycle events", status == 200 and all(name in names for name in REQUIRED_LEDGER_EVENTS),
          f"HTTP {status}, events={sorted(n for n in names if n)}")
    completed = [entry for entry in mine if entry.get("event_name") == "ingestion.completed"]
    check("ledger completion status", bool(completed) and completed[-1].get("status") == "completed", "")

    serialized = json.dumps(mine, sort_keys=True).lower()
    leaked_keys = _walk_keys(mine) & FORBIDDEN_LEDGER_KEYS
    check("ledger has no forbidden keys", not leaked_keys, f"keys={sorted(leaked_keys)}")
    check("ledger has no fixture text or secret markers",
          text.lower() not in serialized and not any(marker in serialized for marker in FORBIDDEN_LEDGER_MARKERS),
          "fixture text or credential marker found in ledger")
    check("ledger source ids carry no query strings", all("?" not in entry.get("source_id", "") for entry in mine),
          "query string found in ledger source_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    parser.add_argument("--gateway", default=defaults.gateway)
    parser.add_argument("--knowledge", default=defaults.knowledge)
    parser.add_argument("--lexical", default=defaults.lexical)
    parser.add_argument("--vector", default=defaults.vector)
    parser.add_argument("--embedding", default=defaults.embedding)
    parser.add_argument("--confirm-ingest", action="store_true",
                        help="permit the write path; without it the run stops after the non-mutating preview")
    args = parser.parse_args(argv)
    config = Config(gateway=args.gateway, knowledge=args.knowledge, lexical=args.lexical,
                    vector=args.vector, embedding=args.embedding, confirm=args.confirm_ingest)
    report = run(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["outcome"] == "passed":
        return 0
    if report["outcome"] == "awaiting_confirmation":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
