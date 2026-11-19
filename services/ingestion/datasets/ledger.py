"""One deterministic summary event in the existing append-only Memory ledger.

The event id is a pure function of the catalog snapshot and run flags, so a
re-run posts the same id and ``append-once`` treats it as a duplicate.  The
payload contains only numeric and enum projections: no URLs, no content and
no query strings.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .catalog import CatalogSnapshot
from .config import Config
from .model import canonical_json, sha256_hex

DEFAULT_MEMORY_URL = "http://127.0.0.1:8767"


def event_input_hash(config: Config, snapshot: CatalogSnapshot) -> str:
    return sha256_hex(
        canonical_json(
            {
                "schema_version": config.schema_version,
                "catalog_repo": snapshot.repo,
                "catalog_commit": snapshot.commit,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "probe_enabled": bool(config.probe_enabled),
                "limit": config.limit,
            }
        )
    )


def summary_event_id(config: Config, snapshot: CatalogSnapshot) -> str:
    return sha256_hex(
        canonical_json(
            {
                "entry_id": "apd:catalog",
                "stage": "inventory",
                "input_hash": event_input_hash(config, snapshot),
                "outcome_code": "inventory_complete",
                "attempt": 1,
            }
        )
    )


def build_summary_entry(
    *,
    event_id: str,
    run_id: str,
    snapshot: CatalogSnapshot,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    manifest_changed: bool,
    outcomes_added: int,
    probe_checked: int,
) -> dict[str, object]:
    counts = report.get("counts") or {}
    assert isinstance(counts, dict)
    return {
        "event_id": event_id,
        "event_type": "dataset_inventory",
        "event_name": "dataset.inventory.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "catalog_commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "discovered": int(counts.get("discovered") or 0),
        "status_counts": _int_map(counts.get("status")),
        "disposition_counts": _int_map(counts.get("disposition")),
        "access_counts": _int_map(counts.get("access")),
        "license_family_counts": _int_map(counts.get("license_family")),
        "format_counts": _int_map(counts.get("format")),
        "probe_enabled": bool(report.get("probe_enabled")),
        "probe_checked": int(probe_checked),
        "limit": report.get("limit"),
        "invariant_ok": bool((report.get("invariant") or {}).get("ok")),
        "exit_code": int(exit_code),
        "duration_ms": int(duration_ms),
        "manifest_changed": bool(manifest_changed),
        "outcomes_added": int(outcomes_added),
    }


def _int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool):
            result[key] = item
    return dict(sorted(result.items()))


def _default_post(url: str, payload: dict[str, object], timeout: float):
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(64 * 1024)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(64 * 1024)


def post_summary(
    config: Config,
    *,
    event_id: str,
    entry: dict[str, object],
    post=None,
) -> dict[str, object]:
    if not config.ledger_enabled:
        return {"enabled": False, "posted": False, "duplicate": False}
    base = os.environ.get("POC_CLAUDE_ENSEMBLE_MEMORY_URL", DEFAULT_MEMORY_URL).rstrip("/")
    url = base + "/memory/append-once"
    payload = {"name": "session_events", "event_id": event_id, "entry": entry}
    try:
        status, body = (post or _default_post)(url, payload, 5)
    except Exception as exc:  # observability is best-effort and never rolls back inventory
        return {
            "enabled": True,
            "posted": False,
            "duplicate": False,
            "error": type(exc).__name__,
        }
    duplicate = False
    if status == 409:
        duplicate = True
    try:
        decoded = json.loads(body or b"{}")
    except ValueError:
        decoded = {}
    if isinstance(decoded, dict) and decoded.get("duplicate") is True:
        duplicate = True
    posted = 200 <= status < 300 or duplicate
    return {
        "enabled": True,
        "posted": posted,
        "duplicate": duplicate,
        "status": status,
    }
