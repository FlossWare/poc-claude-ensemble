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


def resolution_event_id(
    config: Config,
    snapshot: CatalogSnapshot,
    *,
    resolver_version: int,
    selection_hash: str,
    sample: int | None,
    limit: int | None,
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "entry_id": "apd:catalog",
                "stage": "resolve",
                "input_hash": sha256_hex(
                    canonical_json(
                        {
                            "schema_version": config.schema_version,
                            "catalog_repo": snapshot.repo,
                            "catalog_commit": snapshot.commit,
                            "snapshot_sha256": snapshot.snapshot_sha256,
                            "resolver_version": resolver_version,
                            "selection_hash": selection_hash,
                            "sample": sample,
                            "limit": limit,
                        }
                    )
                ),
                "outcome_code": "resolution_complete",
                "attempt": 1,
            }
        )
    )


def build_resolution_entry(
    *,
    event_id: str,
    run_id: str,
    snapshot: CatalogSnapshot,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    manifest_changed: bool,
    outcomes_added: int,
    resolver_version: int,
) -> dict[str, object]:
    """Numeric and enum projections only: no URLs, content or query strings."""
    counts = report.get("counts") or {}
    assert isinstance(counts, dict)
    funnel = report.get("funnel") or {}
    assert isinstance(funnel, dict)
    return {
        "event_id": event_id,
        "event_type": "dataset_resolution",
        "event_name": "dataset.resolve.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "resolver_version": int(resolver_version),
        "catalog_commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "discovered": int(counts.get("discovered") or 0),
        "status_counts": _int_map(counts.get("status")),
        "reason_counts": _int_map(counts.get("reason")),
        "method_counts": _int_map(counts.get("method")),
        "kind_counts": _int_map(counts.get("kind")),
        "disposition_counts": _int_map(counts.get("disposition")),
        "license_policy_counts": _int_map(counts.get("license_policy")),
        "resolved": int(funnel.get("resolved") or 0),
        "unresolved": int(funnel.get("unresolved") or 0),
        "deferred": int(funnel.get("deferred") or 0),
        "failed": int(funnel.get("failed") or 0),
        "sample": report.get("sample"),
        "limit": report.get("limit"),
        "invariant_ok": bool((report.get("invariant") or {}).get("ok")),
        "exit_code": int(exit_code),
        "duration_ms": int(duration_ms),
        "manifest_changed": bool(manifest_changed),
        "outcomes_added": int(outcomes_added),
    }


def acquisition_event_id(
    config: Config,
    snapshot: CatalogSnapshot,
    *,
    acquire_version: int,
    selection_hash: str,
    sample: int | None,
    limit: int | None,
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "entry_id": "apd:catalog",
                "stage": "acquire",
                "input_hash": sha256_hex(
                    canonical_json(
                        {
                            "schema_version": config.schema_version,
                            "catalog_repo": snapshot.repo,
                            "catalog_commit": snapshot.commit,
                            "snapshot_sha256": snapshot.snapshot_sha256,
                            "acquire_version": acquire_version,
                            "selection_hash": selection_hash,
                            "sample": sample,
                            "limit": limit,
                        }
                    )
                ),
                "outcome_code": "acquisition_complete",
                "attempt": 1,
            }
        )
    )


def build_acquisition_entry(
    *,
    event_id: str,
    run_id: str,
    snapshot: CatalogSnapshot,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    manifest_changed: bool,
    outcomes_added: int,
    acquire_version: int,
    adapter_version: int,
) -> dict[str, object]:
    """Numeric and enum projections only: no URLs, content, hashes or query strings."""
    counts = report.get("counts") or {}
    assert isinstance(counts, dict)
    funnel = report.get("funnel") or {}
    assert isinstance(funnel, dict)
    adapter = counts.get("adapter") if isinstance(counts.get("adapter"), dict) else {}
    return {
        "event_id": event_id,
        "event_type": "dataset_acquisition",
        "event_name": "dataset.acquire.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "acquire_version": int(acquire_version),
        "adapter_version": int(adapter_version),
        "catalog_commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "discovered": int(counts.get("discovered") or 0),
        "status_counts": _int_map(counts.get("status")),
        "reason_counts": _int_map(counts.get("reason")),
        "parse_status_counts": _int_map(counts.get("parse_status")),
        "adapter_counts": _int_map(adapter),
        "license_policy_counts": _int_map(counts.get("license_policy")),
        "acquired": int(funnel.get("acquired") or 0),
        "parsed": int(funnel.get("parsed") or 0),
        "skipped": int(funnel.get("skipped") or 0),
        "deferred": int(funnel.get("deferred") or 0),
        "failed": int(funnel.get("failed") or 0),
        "files": int(funnel.get("files") or 0),
        "bytes": int(funnel.get("bytes") or 0),
        "records": int(funnel.get("records") or 0),
        "sample": report.get("sample"),
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


def integration_event_id(
    config: Config,
    snapshot: CatalogSnapshot,
    *,
    integrate_version: int,
    selection_hash: str,
    sample: int | None,
    limit: int | None,
    include_metadata_only: bool,
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "entry_id": "apd:catalog",
                "stage": "integrate",
                "input_hash": sha256_hex(
                    canonical_json(
                        {
                            "schema_version": config.schema_version,
                            "catalog_repo": snapshot.repo,
                            "catalog_commit": snapshot.commit,
                            "snapshot_sha256": snapshot.snapshot_sha256,
                            "integrate_version": integrate_version,
                            "selection_hash": selection_hash,
                            "sample": sample,
                            "limit": limit,
                            "include_metadata_only": bool(include_metadata_only),
                        }
                    )
                ),
                "outcome_code": "integration_complete",
                "attempt": 1,
            }
        )
    )


def build_integration_entry(
    *,
    event_id: str,
    run_id: str,
    snapshot: CatalogSnapshot,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    manifest_changed: bool,
    outcomes_added: int,
    integrate_version: int,
    normalize_version: int,
    indexing_version: int,
    adapter_version: int,
) -> dict[str, object]:
    """Numeric and enum projections only: no URLs, content, hashes or query strings."""
    counts = report.get("counts") or {}
    assert isinstance(counts, dict)
    funnel = report.get("funnel") or {}
    assert isinstance(funnel, dict)
    decisions = counts.get("knowledge_decision") if isinstance(counts.get("knowledge_decision"), dict) else {}
    record_kinds = counts.get("record_kind") if isinstance(counts.get("record_kind"), dict) else {}
    return {
        "event_id": event_id,
        "event_type": "dataset_integration",
        "event_name": "dataset.integrate.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "integrate_version": int(integrate_version),
        "normalize_version": int(normalize_version),
        "indexing_version": int(indexing_version),
        "adapter_version": int(adapter_version),
        "catalog_commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "discovered": int(counts.get("discovered") or 0),
        "status_counts": _int_map(counts.get("status")),
        "reason_counts": _int_map(counts.get("reason")),
        "knowledge_decision_counts": _int_map(decisions),
        "record_kind_counts": _int_map(record_kinds),
        "indexed": int(funnel.get("indexed") or 0),
        "metadata_indexed": int(funnel.get("metadata_indexed") or 0),
        "unchanged": int(funnel.get("unchanged") or 0),
        "skipped": int(funnel.get("skipped") or 0),
        "deferred": int(funnel.get("deferred") or 0),
        "failed": int(funnel.get("failed") or 0),
        "records": int(funnel.get("records") or 0),
        "shards": int(funnel.get("shards") or 0),
        "knowledge_records": int(funnel.get("knowledge_records") or 0),
        "chunks": int(funnel.get("chunks") or 0),
        "vectors": int(funnel.get("vectors") or 0),
        "sample": report.get("sample"),
        "limit": report.get("limit"),
        "invariant_ok": bool((report.get("invariant") or {}).get("ok")),
        "exit_code": int(exit_code),
        "duration_ms": int(duration_ms),
        "manifest_changed": bool(manifest_changed),
        "outcomes_added": int(outcomes_added),
    }


# ---------------------------------------------------------------------------
# PR 5: graph projection and retrieval verification
# ---------------------------------------------------------------------------


def _event_id(stage: str, input_hash: str, outcome_code: str) -> str:
    return sha256_hex(
        canonical_json(
            {
                "entry_id": "apd:catalog",
                "stage": stage,
                "input_hash": input_hash,
                "outcome_code": outcome_code,
                "attempt": 1,
            }
        )
    )


def graph_event_id(
    config: Config,
    snapshot: CatalogSnapshot,
    *,
    graph_version: int,
    selection_hash: str,
    sample: int | None,
    limit: int | None,
    dry_run: bool,
) -> str:
    return _event_id(
        "graph",
        sha256_hex(
            canonical_json(
                {
                    "schema_version": config.schema_version,
                    "catalog_repo": snapshot.repo,
                    "catalog_commit": snapshot.commit,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "graph_version": graph_version,
                    "selection_hash": selection_hash,
                    "sample": sample,
                    "limit": limit,
                    "dry_run": bool(dry_run),
                }
            )
        ),
        "graph_projection_complete",
    )


def build_graph_entry(
    *,
    event_id: str,
    run_id: str,
    snapshot: CatalogSnapshot,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    graph_version: int,
    outcomes_added: int,
    dry_run: bool,
) -> dict[str, object]:
    """Numeric and enum projections only: no node ids, URLs or content."""
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    funnel = report.get("funnel") if isinstance(report.get("funnel"), dict) else {}
    resume_state = report.get("resume") if isinstance(report.get("resume"), dict) else {}
    assert isinstance(counts, dict)
    assert isinstance(funnel, dict)
    assert isinstance(resume_state, dict)
    return {
        "event_id": event_id,
        "event_type": "dataset_graph",
        "event_name": "dataset.graph.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "graph_version": int(graph_version),
        "catalog_commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "dry_run": bool(dry_run),
        "discovered": int(counts.get("discovered") or 0),
        "status_counts": _int_map(counts.get("status")),
        "reason_counts": _int_map(counts.get("reason")),
        "projected": int(funnel.get("projected") or 0),
        "created": int(funnel.get("created") or 0),
        "unchanged": int(funnel.get("unchanged") or 0),
        "skipped": int(funnel.get("skipped") or 0),
        "deferred": int(funnel.get("deferred") or 0),
        "failed": int(funnel.get("failed") or 0),
        "resumed": int(funnel.get("resumed") or 0),
        "nodes_created": int(funnel.get("nodes_created") or 0),
        "edges_created": int(funnel.get("edges_created") or 0),
        "retried": int(resume_state.get("retried") or 0),
        "retry_exhausted": int(resume_state.get("retry_exhausted") or 0),
        "invariant_ok": bool((report.get("invariant") or {}).get("ok")),
        "exit_code": int(exit_code),
        "duration_ms": int(duration_ms),
        "outcomes_added": int(outcomes_added),
    }


def verification_event_id(
    config: Config,
    *,
    catalog_commit: str,
    verify_version: int,
    query_hash: str,
    top_k: int,
) -> str:
    return _event_id(
        "verify",
        sha256_hex(
            canonical_json(
                {
                    "schema_version": config.schema_version,
                    "catalog_commit": catalog_commit,
                    "verify_version": verify_version,
                    "query_hash": query_hash,
                    "top_k": top_k,
                }
            )
        ),
        "retrieval_verification_complete",
    )


def build_verification_entry(
    *,
    event_id: str,
    run_id: str,
    catalog_commit: str,
    report: dict[str, object],
    exit_code: int,
    duration_ms: int,
    verify_version: int,
) -> dict[str, object]:
    """Leg outcomes and counts only: never the queries or the retrieved text."""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    vector_store = report.get("vector_store") if isinstance(report.get("vector_store"), dict) else {}
    assert isinstance(summary, dict)
    assert isinstance(vector_store, dict)
    legs = summary.get("legs") if isinstance(summary.get("legs"), dict) else {}
    assert isinstance(legs, dict)

    def _leg_status(name: str) -> str:
        value = legs.get(name) if isinstance(legs.get(name), dict) else {}
        assert isinstance(value, dict)
        return str(value.get("status") or "unknown")

    return {
        "event_id": event_id,
        "event_type": "dataset_verification",
        "event_name": "dataset.verify.completed",
        "source": "poc-dataset-ingest",
        "producer": "poc-claude-ensemble",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "correlation_id": run_id,
        "execution_id": run_id,
        "schema_version": 1,
        "verify_version": int(verify_version),
        "catalog_commit": catalog_commit,
        "queries": int(report.get("query_count") or 0),
        "top_k": int(report.get("top_k") or 0),
        "lexical_status": _leg_status("lexical"),
        "vector_status": _leg_status("vector"),
        "hybrid_status": _leg_status("hybrid"),
        "lexical_matched": int((legs.get("lexical") or {}).get("matched") or 0)
        if isinstance(legs.get("lexical"), dict)
        else 0,
        "vector_matched": int((legs.get("vector") or {}).get("matched") or 0)
        if isinstance(legs.get("vector"), dict)
        else 0,
        "vector_functional": bool(vector_store.get("functional")),
        "graph_verified": (summary.get("graph") or {}).get("verified")
        if isinstance(summary.get("graph"), dict)
        else None,
        "ok": bool(summary.get("ok")),
        "problem_count": len(summary.get("problems") or []),
        "exit_code": int(exit_code),
        "duration_ms": int(duration_ms),
    }
