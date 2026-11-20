"""Manifest construction, accounting invariant and append-only outcomes.

One manifest row is emitted for every discovered catalog file, including
malformed and unsafe ones, so ``discovered == len(rows)`` can be asserted.
Manifest rows contain no per-run timestamps, making a re-run on the same
snapshot byte-identical.
"""
from __future__ import annotations

import json
import os
import stat
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from . import classify, licenses, model
from .catalog import CatalogFile, CatalogSnapshot
from .config import Config
from .model import sanitize_url

SUCCESS_STATUSES = frozenset({"indexed", "metadata_indexed"})
VALID_STATUSES = frozenset(
    {"indexed", "metadata_indexed", "unchanged", "deferred", "skipped", "failed"}
)

RETRYABLE_REASONS = frozenset(
    {
        "catalog_snapshot_failed",
        "landing_page_only",
        "resolver_not_implemented",
        "doi_unresolved",
        "blocked",
        "not_found",
        "server_error",
        "rate_limited",
        "timeout",
        "dns_failure",
        "tls_failure",
        "connection_failed",
        "oversized_declared",
        "oversized_stream",
        "checksum_mismatch",
        "partial_download",
        "disk_guard",
        "run_budget_exhausted",
        "unsupported_format",
        "adapter_unavailable",
        "encoding_error",
        "limits_exceeded",
        "index_conflict",
        "dependency_unavailable",
        "graph_failed",
        "ledger_failed",
    }
)


@dataclass
class BuildResult:
    rows: list[dict[str, object]]
    discovered: int
    unhandled: int
    truncated: bool


@dataclass
class Accounting:
    ok: bool
    discovered: int
    status_total: int
    status_counts: dict[str, int]
    missing_reason: list[str]
    problems: list[str]


def _meta_url(snapshot: CatalogSnapshot, catalog_file: CatalogFile) -> str:
    readme = snapshot.readme_entries.get(catalog_file.relpath)
    if readme and readme.get("url"):
        return readme["url"]
    quoted = urllib.parse.quote(catalog_file.relpath, safe="/")
    return f"https://github.com/{snapshot.repo}/tree/{snapshot.commit}/core/{quoted}"


def _readme_status(snapshot: CatalogSnapshot, catalog_file: CatalogFile) -> str:
    readme = snapshot.readme_entries.get(catalog_file.relpath)
    return readme.get("status", "missing") if readme else "missing"


def _catalog_block(
    snapshot: CatalogSnapshot, catalog_file: CatalogFile, meta_url: str, readme_status: str
) -> dict[str, object]:
    return {
        "repo": snapshot.repo,
        "commit": snapshot.commit,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "mode": snapshot.mode,
        "path": catalog_file.relpath,
        "file_sha256": catalog_file.sha256 or None,
        "bytes": catalog_file.size,
        "meta_url": sanitize_url(meta_url),
        "readme_status": readme_status,
    }


def _license_block(
    snapshot: CatalogSnapshot,
    result: licenses.LicenseResult,
    meta_url: str,
    config: Config,
) -> dict[str, object]:
    policy, reason, review = licenses.policy_for(result, config.license_allow_families)
    evidence: list[dict[str, object]] = [
        {"type": "metadata", "url": sanitize_url(meta_url), "retrieved_at": None}
    ]
    for url in result.evidence_urls:
        evidence.append(
            {"type": "license_file", "url": sanitize_url(url), "retrieved_at": None}
        )
    return {
        "catalog": snapshot.catalog_license,
        "metadata_repo": snapshot.metadata_repo_license,
        "dataset": {
            # Preserve free text verbatim, but never persist query strings in
            # a value that is itself a URL.
            "declared_raw": model.sanitize_nested(result.declared_raw),
            "normalized": result.normalized,
            "family": result.family,
            "ambiguous": result.ambiguous,
            "matched_families": list(result.matched_families),
            "policy": policy,
            "reason_code": reason,
            "review_required": bool(result.review_required or review),
            "evidence": evidence,
        },
    }


def _unknown_license_block(snapshot: CatalogSnapshot, meta_url: str) -> dict[str, object]:
    return {
        "catalog": snapshot.catalog_license,
        "metadata_repo": snapshot.metadata_repo_license,
        "dataset": {
            "declared_raw": None,
            "normalized": None,
            "family": "unknown",
            "ambiguous": False,
            "matched_families": [],
            "policy": "metadata_only_review",
            "reason_code": "license_unknown_metadata_only",
            "review_required": True,
            "evidence": [
                {"type": "metadata", "url": sanitize_url(meta_url), "retrieved_at": None}
            ],
        },
    }


def _unknown_classes() -> dict[str, object]:
    return {
        "access": "unknown",
        "access_reason": None,
        "size": {"class": "unknown", "bytes": None, "source": "metadata"},
        "format": ["unknown"],
        "scheme": [],
        "scheme_unsupported": [],
    }


def _failed_row(
    snapshot: CatalogSnapshot,
    catalog_file: CatalogFile,
    entry_id: str,
    meta_url: str,
    readme_status: str,
    reason_code: str,
    detail: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "catalog": _catalog_block(snapshot, catalog_file, meta_url, readme_status),
        "title": None,
        "category": catalog_file.category,
        "homepage": None,
        "description": None,
        "keywords": [],
        "tags": [],
        "publisher": [],
        "organization": [],
        "sources": [],
        "version": None,
        "issued_time": None,
        "temporal": None,
        "spatial": None,
        "extra": {},
        "metadata_sha256": model.sha256_hex(
            model.canonical_json(
                {
                    "path": catalog_file.relpath,
                    "file_sha256": catalog_file.sha256,
                    "reason_code": reason_code,
                }
            )
        ),
        "license": _unknown_license_block(snapshot, meta_url),
        "classes": _unknown_classes(),
        "probe": None,
        "disposition": "metadata_only_review",
        "status": "failed",
        "reason_code": reason_code,
        "next_action": "fix_metadata",
        "warnings": [],
        "error": detail[:500],
    }


def _budget_row(
    snapshot: CatalogSnapshot,
    catalog_file: CatalogFile,
    entry_id: str,
    meta_url: str,
    readme_status: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "catalog": _catalog_block(snapshot, catalog_file, meta_url, readme_status),
        "title": None,
        "category": catalog_file.category,
        "homepage": None,
        "description": None,
        "keywords": [],
        "tags": [],
        "publisher": [],
        "organization": [],
        "sources": [],
        "version": None,
        "issued_time": None,
        "temporal": None,
        "spatial": None,
        "extra": {},
        "metadata_sha256": model.sha256_hex(
            model.canonical_json(
                {
                    "path": catalog_file.relpath,
                    "file_sha256": catalog_file.sha256,
                }
            )
        ),
        "license": _unknown_license_block(snapshot, meta_url),
        "classes": _unknown_classes(),
        "probe": None,
        "disposition": "metadata_only_review",
        "status": "deferred",
        "reason_code": "run_budget_exhausted",
        "next_action": "resume",
        "warnings": [],
        "error": None,
    }


def _read_snapshot_file(
    core_dir: Path, catalog_file: CatalogFile, config: Config
) -> bytes:
    target = core_dir.joinpath(*catalog_file.relpath.split("/"))
    core_real = os.path.realpath(core_dir)
    target_real = os.path.realpath(target)
    if target_real != core_real and not target_real.startswith(core_real + os.sep):
        raise model.EntryError("catalog_entry_invalid", "entry path escapes the catalog root")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    chunks: list[bytes] = []
    total = 0
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise model.EntryError("catalog_entry_invalid", "entry is not a regular file")
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > config.catalog_max_file_bytes:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    if total > config.catalog_max_file_bytes:
        raise model.EntryError("limits_exceeded", "entry file exceeds the byte limit")
    payload = b"".join(chunks)
    if catalog_file.sha256 and model.sha256_hex(payload) != catalog_file.sha256:
        raise model.EntryError("catalog_entry_invalid", "entry changed during inventory")
    return payload


def _row_from_normalized(
    snapshot: CatalogSnapshot,
    catalog_file: CatalogFile,
    normalized: dict[str, object],
    config: Config,
) -> dict[str, object]:
    raw_fields = normalized["normalized"]
    assert isinstance(raw_fields, dict)
    fields = model.sanitize_nested(raw_fields)
    assert isinstance(fields, dict)
    homepage = sanitize_url(fields.get("homepage"))
    sources: list[dict[str, object]] = []
    for source in fields.get("sources") or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        item["access_url"] = sanitize_url(item.get("access_url"))
        sources.append(item)
    meta_url = str(normalized["meta_url"] or "")
    row: dict[str, object] = {
        "schema_version": 1,
        "entry_id": normalized["entry_id"],
        "catalog": _catalog_block(
            snapshot, catalog_file, meta_url, str(normalized["readme_status"])
        ),
        "title": fields.get("title"),
        "category": fields.get("category"),
        "homepage": homepage,
        "description": fields.get("description"),
        "keywords": fields.get("keywords") or [],
        "tags": fields.get("tags") or [],
        "publisher": fields.get("publisher") or [],
        "organization": fields.get("organization") or [],
        "sources": sources,
        "version": fields.get("version"),
        "issued_time": fields.get("issued_time"),
        "temporal": fields.get("temporal"),
        "spatial": fields.get("spatial"),
        "extra": model.sanitize_nested(normalized.get("extra") or {}),
        "metadata_sha256": normalized["metadata_sha256"],
        "license": _license_block(
            snapshot, normalized["license"], meta_url, config  # type: ignore[arg-type]
        ),
        "classes": _unknown_classes(),
        "probe": None,
        "disposition": "metadata_only_review",
        "status": "deferred",
        "reason_code": "resolver_not_implemented",
        "next_action": "resolve",
        "warnings": normalized.get("warnings") or [],
        "error": None,
    }
    return row


def build_rows(
    snapshot: CatalogSnapshot,
    config: Config,
    *,
    probe_results: dict[str, object] | None = None,
    limit: int | None = None,
) -> BuildResult:
    probe_results = probe_results or {}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    unhandled = 0
    truncated = False
    for index, catalog_file in enumerate(snapshot.files):
        meta_url = _meta_url(snapshot, catalog_file)
        readme_status = _readme_status(snapshot, catalog_file)
        try:
            entry_id = model.entry_id_for(catalog_file.category, catalog_file.filename)
        except model.EntryError:
            entry_id = f"apd:invalid/{model.sha256_hex(catalog_file.relpath)[:16]}"
        if entry_id in seen:
            rows.append(
                _failed_row(
                    snapshot,
                    catalog_file,
                    entry_id,
                    meta_url,
                    readme_status,
                    "catalog_duplicate_id",
                    "duplicate entry identifier",
                )
            )
            continue
        seen.add(entry_id)
        if catalog_file.unsafe_reason:
            code = (
                "limits_exceeded"
                if catalog_file.unsafe_reason == "oversized"
                else "catalog_entry_invalid"
            )
            rows.append(
                _failed_row(
                    snapshot,
                    catalog_file,
                    entry_id,
                    meta_url,
                    readme_status,
                    code,
                    f"unsafe catalog file: {catalog_file.unsafe_reason}",
                )
            )
            continue
        if limit is not None and index >= limit:
            truncated = True
            rows.append(
                _budget_row(snapshot, catalog_file, entry_id, meta_url, readme_status)
            )
            continue
        try:
            payload = _read_snapshot_file(snapshot.core_dir, catalog_file, config)
            document, warnings = model.load_yaml_entry(payload, config)
            normalized = model.normalize_entry(
                document,
                category=catalog_file.category,
                filename=catalog_file.filename,
                relpath=catalog_file.relpath,
                meta_url=meta_url,
                readme_status=readme_status,
                warnings=warnings,
            )
            row = _row_from_normalized(snapshot, catalog_file, normalized, config)
            classify.classify_row(row, config, probe_results.get(entry_id))
            rows.append(row)
        except model.EntryError as exc:
            rows.append(
                _failed_row(
                    snapshot,
                    catalog_file,
                    entry_id,
                    meta_url,
                    readme_status,
                    exc.code,
                    exc.detail,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive per-entry isolation
            unhandled += 1
            rows.append(
                _failed_row(
                    snapshot,
                    catalog_file,
                    entry_id,
                    meta_url,
                    readme_status,
                    "catalog_entry_invalid",
                    type(exc).__name__,
                )
            )
    rows.sort(key=lambda row: str(row["entry_id"]))
    return BuildResult(
        rows=rows,
        discovered=len(snapshot.files),
        unhandled=unhandled,
        truncated=truncated,
    )


def check_accounting(rows: list[dict[str, object]], discovered: int) -> Accounting:
    status_counts: dict[str, int] = {}
    missing_reason: list[str] = []
    problems: list[str] = []
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in VALID_STATUSES:
            problems.append(f"invalid_status:{status}")
        if status not in SUCCESS_STATUSES and not row.get("reason_code"):
            missing_reason.append(str(row.get("entry_id")))
    status_total = sum(status_counts.values())
    ok = True
    if len(rows) != discovered:
        ok = False
        problems.append(f"row_count:{len(rows)}!=discovered:{discovered}")
    if status_total != discovered:
        ok = False
        problems.append(f"status_total:{status_total}!=discovered:{discovered}")
    if missing_reason:
        ok = False
        problems.append(f"missing_reason:{len(missing_reason)}")
    if problems:
        ok = False
    return Accounting(
        ok=ok,
        discovered=discovered,
        status_total=status_total,
        status_counts=dict(sorted(status_counts.items())),
        missing_reason=sorted(missing_reason),
        problems=sorted(set(problems)),
    )


def atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def load_rows(path: Path) -> list[dict[str, object]] | None:
    """Read a persisted inventory manifest as data, or ``None`` if unusable.

    The graph stage prefers this over re-parsing the catalog: it is the same
    PR 1 rows, and reading them means a projection resumes from the manifest
    instead of from scratch.  A malformed or empty file is treated as absent.
    """
    rows: list[dict[str, object]] = []
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except OSError:
        return None
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                return None
            if not isinstance(row, dict) or "entry_id" not in row:
                return None
            rows.append(row)
    return rows or None


def rows_match_snapshot(
    rows: list[dict[str, object]], snapshot: CatalogSnapshot
) -> bool:
    """Whether persisted rows still describe this catalog snapshot."""
    if len(rows) != len(snapshot.files):
        return False
    for row in rows:
        catalog = row.get("catalog")
        if not isinstance(catalog, dict):
            return False
        if catalog.get("commit") != snapshot.commit:
            return False
        if catalog.get("snapshot_sha256") not in (None, snapshot.snapshot_sha256):
            return False
    return True


def manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    ordered = sorted(rows, key=lambda row: str(row.get("entry_id")))
    return "".join(model.canonical_json(row) + "\n" for row in ordered).encode("utf-8")


def write_manifest(rows: list[dict[str, object]], path: Path) -> tuple[bool, Path]:
    data = manifest_bytes(rows)
    existing = None
    try:
        existing = path.read_bytes()
    except OSError:
        existing = None
    changed = existing != data
    if changed:
        atomic_write(path, data)
    return changed, path


def _outcome_row(
    row: dict[str, object], run_id: str, catalog_commit: str, retrieved_at: str
) -> dict[str, object]:
    status = str(row.get("status"))
    reason_code = row.get("reason_code")
    outcome_code = f"{status}:{reason_code or ''}"
    event_id = model.sha256_hex(
        model.canonical_json(
            {
                "entry_id": row.get("entry_id"),
                "stage": "classify",
                "input_hash": row.get("metadata_sha256"),
                "outcome_code": outcome_code,
                "attempt": 1,
            }
        )
    )
    catalog = row.get("catalog") or {}
    classes = row.get("classes") or {}
    dataset_license = (row.get("license") or {}).get("dataset") or {}
    return {
        "schema_version": 1,
        "event_id": event_id,
        "run_id": run_id,
        "entry_id": row.get("entry_id"),
        "catalog_commit": catalog_commit,
        "stage": "classify",
        "status": status,
        "reason_code": reason_code,
        "detail": row.get("error"),
        "source_uri": row.get("homepage"),
        "source_sha256": catalog.get("file_sha256") if isinstance(catalog, dict) else None,
        "bytes": catalog.get("bytes") if isinstance(catalog, dict) else None,
        "etag": None,
        "last_modified": None,
        "retrieved_at": retrieved_at,
        "attempts": 1,
        "retryable": reason_code in RETRYABLE_REASONS,
        "next_retry_at": None,
        "knowledge_source_id": None,
        "revision_id": None,
        "revision_sequence": None,
        "knowledge_decision": None,
        "license_family": dataset_license.get("family")
        if isinstance(dataset_license, dict)
        else None,
        "access": classes.get("access") if isinstance(classes, dict) else None,
        "format": classes.get("format") if isinstance(classes, dict) else None,
    }


def append_outcome_rows(outcomes: list[dict[str, object]], path: Path) -> int:
    """Append outcome rows, deduplicating by deterministic ``event_id``."""
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except ValueError:
                    continue
                event_id = existing.get("event_id") if isinstance(existing, dict) else None
                if isinstance(event_id, str):
                    seen.add(event_id)
    except OSError:
        pass
    new_rows: list[dict[str, object]] = []
    for outcome in outcomes:
        event_id = str(outcome.get("event_id") or "")
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        new_rows.append(outcome)
    if not new_rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            descriptor = -1
            for outcome in new_rows:
                handle.write(model.canonical_json(outcome) + "\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return len(new_rows)


def append_outcomes(
    rows: list[dict[str, object]],
    path: Path,
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> int:
    outcomes = [
        _outcome_row(row, run_id, catalog_commit, retrieved_at)
        for row in sorted(rows, key=lambda item: str(item.get("entry_id")))
    ]
    return append_outcome_rows(outcomes, path)


def run_id_for(config: Config, snapshot: CatalogSnapshot) -> str:
    return model.sha256_hex(
        model.canonical_json(
            {
                "schema_version": config.schema_version,
                "catalog_repo": snapshot.repo,
                "commit": snapshot.commit,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "probe_enabled": bool(config.probe_enabled),
                "limit": config.limit,
            }
        )
    )[:16]


def write_run_report(report: dict[str, object], data_root: Path, run_id: str) -> Path:
    path = data_root / "runs" / f"{run_id}.json"
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    atomic_write(path, data)
    return path


def latest_run(data_root: Path) -> dict[str, object] | None:
    directory = data_root / "runs"
    try:
        candidates = sorted(
            directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
        )
    except OSError:
        return None
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("run_file", str(candidate))
            return payload
    return None
