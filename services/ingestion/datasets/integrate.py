"""PR 4: per-entry normalization and integration.

This module joins the pure normalizer (:mod:`.normalize`) to the shared
retrieval orchestration (:mod:`.indexing`) and produces the explicit, per-record
outcomes the design requires:

* every record carries ``indexed``, ``unchanged``, ``skipped`` or ``failed``;
* every record that is not ``indexed``/``unchanged`` carries a ``reason_code``;
* every entry carries exactly one terminal status.

It also owns the handoff from PR 3: the acquisition manifest is read as data, so
no download, resolution or parsing policy is re-litigated here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from . import adapters, indexing, manifest, model, normalize
from .config import Config

INTEGRATE_VERSION = 1

#: Reasons that justify another attempt without operator action.
RETRYABLE_REASONS = frozenset(
    {
        "dependency_unavailable",
        "downstream_failure",
        "invalid_downstream_response",
        "index_conflict",
        "index_unavailable",
        "not_selected",
        "timeout",
        "not_acquired",
        "encoding_error",
        "limits_exceeded",
        "adapter_unavailable",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EntryIntegration:
    """The terminal result for one entry, with no record content retained."""

    entry_id: str
    status: str
    reason_code: str | None = None
    detail: str | None = None
    retryable: bool = False
    metadata: dict[str, object] | None = None
    record_outcomes: tuple[dict[str, object], ...] = ()
    files: tuple[dict[str, object], ...] = ()
    records_total: int = 0
    shards: int = 0
    chunks: int = 0
    vectors: int = 0
    truncated: bool = False
    tombstoned: tuple[str, ...] = ()

    def to_artifact(self) -> dict[str, object]:
        content = [item for item in self.record_outcomes if item.get("kind") == "content"]
        return {
            "integrate_version": INTEGRATE_VERSION,
            "normalize_version": normalize.NORMALIZE_VERSION,
            "indexing_version": indexing.INDEXING_VERSION,
            "adapter_version": adapters.ADAPTER_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:500] if self.detail else None,
            "retryable": self.retryable,
            "records_total": self.records_total,
            "shards": self.shards,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "truncated": self.truncated,
            "metadata": self.metadata,
            "content": {
                "shards": len(content),
                "indexed": sum(1 for item in content if item.get("status") == "indexed"),
                "unchanged": sum(1 for item in content if item.get("status") == "unchanged"),
                "skipped": sum(1 for item in content if item.get("status") == "skipped"),
                "failed": sum(1 for item in content if item.get("status") == "failed"),
                "records": list(content),
            },
            "files": list(self.files),
            "tombstoned": list(self.tombstoned),
        }


# ---------------------------------------------------------------------------
# Acquisition handoff
# ---------------------------------------------------------------------------


def load_acquisitions(path: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    """Read the PR 3 acquisition manifest into ``entry_id -> acquisition``."""
    acquisitions: dict[str, dict[str, object]] = {}
    commit: str | None = None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                entry_id = row.get("entry_id")
                acquisition = row.get("acquisition")
                if isinstance(entry_id, str) and isinstance(acquisition, dict):
                    acquisitions[entry_id] = acquisition
                    catalog = row.get("catalog")
                    if commit is None and isinstance(catalog, dict):
                        value = catalog.get("commit")
                        commit = value if isinstance(value, str) else None
    except OSError:
        return {}, None
    return acquisitions, commit


def select_entries(
    rows: Sequence[dict[str, object]],
    acquisitions: dict[str, dict[str, object]],
    *,
    entries: Sequence[str] | None = None,
    sample: int | None = None,
    limit: int | None = None,
    include_metadata_only: bool = False,
) -> tuple[set[str], list[str], bool]:
    """Choose which entries to integrate.

    Default: entries PR 3 acquired (``acquired``/``unchanged``).  With
    ``include_metadata_only`` also entries whose license policy allows metadata
    but not content, which then receive a metadata record only.
    """
    candidates: list[str] = []
    known: set[str] = set()
    for row in rows:
        entry_id = str(row.get("entry_id") or "")
        known.add(entry_id)
        acquisition = acquisitions.get(entry_id)
        if acquisition is not None and acquisition.get("status") in {"acquired", "unchanged"}:
            candidates.append(entry_id)
        elif include_metadata_only and normalize.license_policy(row) == "metadata_only_review":
            candidates.append(entry_id)
    candidates.sort()
    if entries:
        wanted = list(dict.fromkeys(entries))
        unknown = [entry_id for entry_id in wanted if entry_id not in known]
        accepted = {entry_id for entry_id in wanted if entry_id in set(candidates)}
        return accepted, unknown, False
    if sample is not None:
        chosen = candidates[: max(0, sample)]
        return set(chosen), [], len(chosen) < len(candidates)
    if limit is not None:
        chosen = candidates[: max(0, limit)]
        return set(chosen), [], len(chosen) < len(candidates)
    return set(candidates), [], False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def _row_label(row: dict[str, object]) -> str:
    return str(row.get("entry_id") or "")


def _skip(
    entry_id: str,
    status: str,
    reason_code: str,
    *,
    detail: str | None = None,
    retryable: bool | None = None,
) -> EntryIntegration:
    if retryable is None:
        retryable = reason_code in RETRYABLE_REASONS
    return EntryIntegration(
        entry_id=entry_id,
        status=status,
        reason_code=reason_code,
        detail=detail,
        retryable=retryable,
    )


def _entry_status(
    normalization: normalize.EntryNormalization,
    outcomes: Sequence[indexing.RecordOutcome],
) -> tuple[str, str | None, str | None, bool]:
    """Collapse record outcomes into one terminal entry status.

    A content failure dominates: an entry with one unindexed shard is
    ``deferred`` (retryable) or ``failed`` even when its metadata record was
    stored, so the incomplete work is never reported as success.
    """
    metadata = [item for item in outcomes if item.kind == "metadata"]
    content = [item for item in outcomes if item.kind == "content"]
    indexed_content = [item for item in content if item.status == "indexed"]
    indexed_metadata = [item for item in metadata if item.status == "indexed"]
    unchanged = [item for item in outcomes if item.status == "unchanged"]
    failed = [item for item in outcomes if item.status == "failed"]
    skipped = [item for item in outcomes if item.status == "skipped"]

    content_failed = [item for item in content if item.status == "failed"]
    content_skipped = [item for item in content if item.status == "skipped"]
    worst = content_failed[0] if content_failed else (content_skipped[0] if content_skipped else None)
    if worst is not None:
        reason = worst.reason_code or ("index_failed" if content_failed else "skipped")
        retryable = reason in RETRYABLE_REASONS
        return ("deferred" if retryable else "failed"), reason, worst.detail, retryable
    if indexed_content:
        return "indexed", None, None, False
    if indexed_metadata:
        return "metadata_indexed", normalization.content_reason, None, False
    if failed:
        reason = failed[0].reason_code or "index_failed"
        retryable = reason in RETRYABLE_REASONS
        return ("deferred" if retryable else "failed"), reason, failed[0].detail, retryable
    if unchanged:
        return "unchanged", "content_unchanged", None, False
    if skipped:
        reason = skipped[0].reason_code or "skipped"
        return "skipped", reason, skipped[0].detail, reason in RETRYABLE_REASONS
    reason = normalization.content_reason or "empty_content"
    return "metadata_indexed", reason, None, False


def integrate_entry(
    row: dict[str, object],
    acquisition: dict[str, object] | None,
    *,
    commit: str,
    config: Config,
    endpoints: indexing.Endpoints | None = None,
    journal: indexing.CompletionJournal | None = None,
    post=None,
) -> EntryIntegration:
    """Normalize one entry and publish every record, returning explicit outcomes."""
    post = post or indexing.post_json
    entry_id = _row_label(row)
    if acquisition is None:
        acquisition_status = None
    else:
        acquisition_status = str(acquisition.get("status") or "")
    metadata_only = (
        config.integration_metadata_only
        and normalize.license_policy(row) == "metadata_only_review"
    )
    if (
        acquisition is not None
        and acquisition_status not in {"acquired", "unchanged"}
        and not metadata_only
    ):
        reason = str(acquisition.get("reason_code") or "not_acquired")
        status = "deferred" if reason in RETRYABLE_REASONS else "skipped"
        return _skip(
            entry_id,
            status,
            reason,
            detail=acquisition.get("detail")
            if isinstance(acquisition.get("detail"), str)
            else None,
        )

    normalization = normalize.normalize_entry(
        row, acquisition, commit=commit, profiles=indexing.PROFILES, config=config
    )
    endpoints = endpoints or indexing.Endpoints.from_config(config)
    journal = journal or indexing.CompletionJournal(
        Path(config.data_root) / "cache" / "integrations",
        enabled=config.integration_journal_enabled,
    )

    records: list[normalize.NormalizedRecord] = []
    if normalization.metadata is not None:
        records.append(normalization.metadata)
    records.extend(normalization.records)

    outcomes: list[indexing.RecordOutcome] = []
    for record in records:
        outcomes.append(
            indexing.integrate_record(
                record,
                endpoints=endpoints,
                journal=journal,
                profiles=indexing.PROFILES,
                timeout=config.integration_timeout,
                post=post,
            )
        )

    tombstoned: list[str] = []
    newly_indexed = any(item.kind == "content" and item.status == "indexed" for item in outcomes)
    if newly_indexed:
        for file_outcome in normalization.files:
            if file_outcome.status != "parsed":
                continue
            source_url = normalize.canonical_source_url(file_outcome.uri, file_outcome.member)
            tombstoned.extend(
                indexing.tombstone_surplus_shards(
                    entry_id=entry_id,
                    source_url=source_url,
                    shard_count=file_outcome.shards,
                    endpoints=endpoints,
                    knowledge_sequence=1,
                )
            )

    status, reason, detail, retryable = _entry_status(normalization, outcomes)
    metadata_outcome = next((item for item in outcomes if item.kind == "metadata"), None)
    return EntryIntegration(
        entry_id=entry_id,
        status=status,
        reason_code=reason,
        detail=detail,
        retryable=retryable,
        metadata=metadata_outcome.to_artifact() if metadata_outcome is not None else (
            {"kind": "metadata", "status": "skipped", "reason_code": normalization.metadata_reason}
            if normalization.metadata_reason
            else None
        ),
        record_outcomes=tuple(item.to_artifact() for item in outcomes),
        files=tuple(item.to_artifact() for item in normalization.files),
        records_total=normalization.records_total,
        shards=normalization.shards,
        chunks=sum(item.chunks for item in outcomes),
        vectors=sum(item.vectors for item in outcomes),
        truncated=normalization.truncated,
        tombstoned=tuple(tombstoned),
    )


# ---------------------------------------------------------------------------
# Manifest, outcomes and accounting
# ---------------------------------------------------------------------------


def integration_manifest_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    lines = [model.canonical_json(row) for row in sorted(rows, key=lambda item: str(item.get("entry_id")))]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_integration_manifest(
    rows: Sequence[dict[str, object]], path: Path
) -> tuple[bool, Path]:
    data = integration_manifest_bytes(rows)
    try:
        existing = path.read_bytes()
    except OSError:
        existing = None
    if existing == data:
        return False, path
    manifest.atomic_write(path, data)
    return True, path


def integration_outcome_rows(
    rows: Sequence[dict[str, object]],
    integrations: Sequence[EntryIntegration],
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> list[dict[str, object]]:
    """One append-only outcome row per record (plus one for record-less entries)."""
    by_entry = {item.entry_id: item for item in integrations}
    outcomes: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        integration = by_entry.get(entry_id)
        if integration is None:
            continue
        artifact = integration.to_artifact()
        records = list(artifact.get("content", {}).get("records") or [])
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            records.insert(0, metadata)
        if not records:
            records = [
                {
                    "source_id": entry_id,
                    "kind": "entry",
                    "status": integration.status,
                    "reason_code": integration.reason_code,
                    "detail": integration.detail,
                    "content_sha256": None,
                    "revision_id": None,
                    "revision_sequence": None,
                    "knowledge_decision": None,
                    "chunks": 0,
                    "vectors": 0,
                }
            ]
        catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
        assert isinstance(catalog, dict)
        for record in records:
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or integration.status)
            reason_code = record.get("reason_code")
            event_id = model.sha256_hex(
                model.canonical_json(
                    {
                        "entry_id": entry_id,
                        "stage": "integrate",
                        "input_hash": record.get("revision_id") or row.get("metadata_sha256"),
                        "outcome_code": f"{status}:{reason_code or ''}",
                        "attempt": 1,
                    }
                )
            )
            outcomes.append(
                {
                    "schema_version": 1,
                    "event_id": event_id,
                    "run_id": run_id,
                    "entry_id": entry_id,
                    "catalog_commit": catalog_commit,
                    "stage": "integrate",
                    "status": status,
                    "reason_code": reason_code,
                    "detail": record.get("detail"),
                    "source_uri": model.sanitize_url(record.get("locator"))
                    or model.sanitize_url(row.get("homepage"))
                    or catalog.get("meta_url"),
                    "source_sha256": record.get("content_sha256"),
                    "bytes": None,
                    "etag": None,
                    "last_modified": None,
                    "retrieved_at": retrieved_at,
                    "attempts": 1,
                    "retryable": reason_code in RETRYABLE_REASONS,
                    "next_retry_at": None,
                    "knowledge_source_id": record.get("source_id"),
                    "revision_id": record.get("revision_id"),
                    "revision_sequence": record.get("revision_sequence"),
                    "knowledge_decision": record.get("knowledge_decision"),
                    "record_kind": record.get("kind"),
                    "shard_index": record.get("shard_index"),
                    "shard_count": record.get("shard_count"),
                    "chunks": int(record.get("chunks") or 0),
                    "vectors": int(record.get("vectors") or 0),
                }
            )
    return outcomes


@dataclass
class IntegrationAccounting:
    ok: bool
    discovered: int
    status_total: int
    status_counts: dict[str, int]
    missing_reason: list[str]
    problems: list[str]


def check_accounting(
    rows: Sequence[dict[str, object]],
    integrations: Sequence[EntryIntegration],
    discovered: int,
) -> IntegrationAccounting:
    """``discovered == statuses`` and every non-success status has a reason."""
    by_entry = {item.entry_id: item for item in integrations}
    status_counts: dict[str, int] = {}
    missing: list[str] = []
    problems: list[str] = []
    for row in rows:
        entry_id = str(row.get("entry_id") or "")
        integration = by_entry.get(entry_id)
        if integration is None:
            problems.append(f"missing_integration:{entry_id}")
            continue
        status = integration.status
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in manifest.VALID_STATUSES:
            problems.append(f"invalid_status:{status}")
        if status not in manifest.SUCCESS_STATUSES and not integration.reason_code:
            missing.append(entry_id)
    total = sum(status_counts.values())
    if total != discovered:
        problems.append(f"status_total:{total}!=discovered:{discovered}")
    if missing:
        problems.append(f"missing_reason:{len(missing)}")
    return IntegrationAccounting(
        ok=not problems,
        discovered=discovered,
        status_total=total,
        status_counts=dict(sorted(status_counts.items())),
        missing_reason=sorted(missing),
        problems=sorted(set(problems)),
    )


def run_id_for(config: Config, commit: str, snapshot_sha256: str, selection: Iterable[str]) -> str:
    return model.sha256_hex(
        model.canonical_json(
            {
                "schema_version": config.schema_version,
                "catalog_commit": commit,
                "snapshot_sha256": snapshot_sha256,
                "integrate_version": INTEGRATE_VERSION,
                "selection": sorted(selection),
            }
        )
    )
