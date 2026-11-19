"""Inventory report: funnel counts, per-entry rows, JSON and Markdown."""
from __future__ import annotations

import json
from collections import Counter

from .catalog import CatalogSnapshot
from .manifest import Accounting


def _count(rows: list[dict[str, object]], getter) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = getter(row)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                counter[str(item)] += 1
        else:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _classes(row: dict[str, object]) -> dict[str, object]:
    value = row.get("classes")
    return value if isinstance(value, dict) else {}


def _size_class(row: dict[str, object]) -> str | None:
    size = _classes(row).get("size")
    if isinstance(size, dict):
        value = size.get("class")
        return str(value) if value is not None else None
    return None


def _license_family(row: dict[str, object]) -> str | None:
    value = (row.get("license") or {}).get("dataset") or {}
    family = value.get("family") if isinstance(value, dict) else None
    return str(family) if family is not None else None


def _probe(row: dict[str, object]) -> dict[str, object] | None:
    value = row.get("probe")
    return value if isinstance(value, dict) else None


def _compact_entry(row: dict[str, object]) -> dict[str, object]:
    classes = _classes(row)
    license_info = (row.get("license") or {}).get("dataset") or {}
    probe = _probe(row)
    return {
        "entry_id": row.get("entry_id"),
        "title": row.get("title"),
        "category": row.get("category"),
        "status": row.get("status"),
        "reason_code": row.get("reason_code"),
        "disposition": row.get("disposition"),
        "next_action": row.get("next_action"),
        "license_family": license_info.get("family") if isinstance(license_info, dict) else None,
        "license_normalized": license_info.get("normalized")
        if isinstance(license_info, dict)
        else None,
        "access": classes.get("access"),
        "access_reason": classes.get("access_reason"),
        "size_class": _size_class(row),
        "formats": classes.get("format") or [],
        "schemes": classes.get("scheme") or [],
        "scheme_unsupported": classes.get("scheme_unsupported") or [],
        "readme_status": (row.get("catalog") or {}).get("readme_status"),
        "probe_status": probe.get("status") if isinstance(probe, dict) else None,
        "warnings": row.get("warnings") or [],
    }


def build_report(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    snapshot: CatalogSnapshot,
    accounting: Accounting,
    generated_at: str,
    duration_ms: int,
    exit_code: int,
    manifest_path: str,
    manifest_changed: bool,
    outcomes_added: int,
    ledger_result: dict[str, object],
    probe_checked: int,
    probe_enabled: bool,
    limit: int | None,
    unhandled: int,
    truncated: bool,
) -> dict[str, object]:
    status_counts = dict(accounting.status_counts)
    entry_paths = {str(row.get("catalog", {}).get("path")) for row in rows}
    missing_from_readme = sorted(
        relpath for relpath in entry_paths if relpath not in snapshot.readme_entries
    )
    extra_in_readme = sorted(
        relpath for relpath in snapshot.readme_entries if relpath not in entry_paths
    )
    counts = {
        "status": status_counts,
        "disposition": _count(rows, lambda row: row.get("disposition")),
        "access": _count(rows, lambda row: _classes(row).get("access")),
        "size_class": _count(rows, _size_class),
        "license_family": _count(rows, _license_family),
        "format": _count(rows, lambda row: _classes(row).get("format")),
        "scheme": _count(rows, lambda row: _classes(row).get("scheme")),
        "scheme_unsupported": _count(
            rows, lambda row: _classes(row).get("scheme_unsupported")
        ),
        "readme_status": _count(
            rows, lambda row: (row.get("catalog") or {}).get("readme_status")
        ),
        "probe_access": _count(
            rows,
            lambda row: (
                _classes(row).get("access") if _probe(row) is not None else None
            ),
        ),
        "probe_reason": _count(
            rows,
            lambda row: (
                _probe(row).get("reason_code") if _probe(row) is not None else None
            ),
        ),
    }
    funnel = {
        "discovered": accounting.discovered,
        "resolved": 0,
        "acquired": 0,
        "parsed": 0,
        "indexed": 0,
        "unchanged": status_counts.get("unchanged", 0),
        "skipped": status_counts.get("skipped", 0),
        "failed": status_counts.get("failed", 0),
        "deferred": status_counts.get("deferred", 0),
    }
    readme = {
        "available": snapshot.readme_path is not None,
        "path": str(snapshot.readme_path) if snapshot.readme_path is not None else None,
        "meta_links": snapshot.readme_meta_total,
        "entries_with_status": sum(
            1
            for row in rows
            if (row.get("catalog") or {}).get("readme_status") in {"OK", "FIXME"}
        ),
        "entries_missing_status": len(missing_from_readme),
        "missing_from_readme": missing_from_readme[:100],
        "extra_in_readme": extra_in_readme[:100],
        "warnings": snapshot.readme_warnings,
    }
    return {
        "schema_version": 1,
        "report_type": "inventory",
        "run_id": run_id,
        "generated_at": generated_at,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "limit": limit,
        "truncated": truncated,
        "probe_enabled": probe_enabled,
        "probe_checked": probe_checked,
        "catalog": {
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "mode": snapshot.mode,
            "entries": len(rows),
            "discovered_files": len(snapshot.files),
            "catalog_license": snapshot.catalog_license,
            "metadata_repo_license": snapshot.metadata_repo_license,
            "warnings": snapshot.warnings,
        },
        "counts": {"discovered": accounting.discovered, **counts},
        "funnel": funnel,
        "invariant": {
            "ok": accounting.ok,
            "discovered": accounting.discovered,
            "status_total": accounting.status_total,
            "missing_reason": accounting.missing_reason[:100],
            "problems": accounting.problems,
        },
        "readme": readme,
        "manifest": {"path": manifest_path, "changed": manifest_changed},
        "outcomes_added": outcomes_added,
        "unhandled": unhandled,
        "ledger": ledger_result,
        "entries": [_compact_entry(row) for row in rows],
    }


def _md(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _table(title: str, mapping: dict[str, int]) -> list[str]:
    lines = [f"### {title}", "", "| Value | Count |", "|---|---:|"]
    if not mapping:
        lines.append("| _(none)_ | 0 |")
    for key, value in mapping.items():
        lines.append(f"| {_md(key)} | {value} |")
    lines.append("")
    return lines


def _license_policy(row: dict[str, object]) -> str | None:
    license_block = row.get("license")
    dataset = license_block.get("dataset") if isinstance(license_block, dict) else None
    value = dataset.get("policy") if isinstance(dataset, dict) else None
    return str(value) if value is not None else None


def build_resolution_report(
    rows: list[dict[str, object]],
    resolutions: list[object],
    *,
    run_id: str,
    snapshot: CatalogSnapshot,
    accounting: object,
    generated_at: str,
    duration_ms: int,
    exit_code: int,
    manifest_path: str,
    manifest_changed: bool,
    outcomes_added: int,
    ledger_result: dict[str, object],
    resolver_version: int,
    sample: int | None,
    limit: int | None,
    unhandled: int,
    truncated: bool,
    cache_hits: int,
    cache_misses: int,
) -> dict[str, object]:
    by_entry = {resolution.entry_id: resolution for resolution in resolutions}  # type: ignore[attr-defined]
    status_counts = dict(accounting.status_counts)  # type: ignore[attr-defined]
    entries: list[dict[str, object]] = []
    for row in rows:
        entry_id = str(row.get("entry_id") or "")
        resolution = by_entry.get(entry_id)
        if resolution is None:
            continue
        artifact = resolution.to_artifact()  # type: ignore[attr-defined]
        entries.append(
            {
                "entry_id": entry_id,
                "title": row.get("title"),
                "category": row.get("category"),
                "disposition": row.get("disposition"),
                "license_policy": _license_policy(row),
                "status": artifact.get("status"),
                "reason_code": artifact.get("reason_code"),
                "method": artifact.get("method"),
                "kind": artifact.get("kind"),
                "source_count": artifact.get("source_count"),
                "total_sources": artifact.get("total_sources"),
                "version": artifact.get("version"),
                "next_action": artifact.get("next_action"),
            }
        )
    entries.sort(key=lambda entry: str(entry.get("entry_id")))
    counts = {
        "status": status_counts,
        "reason": _count(entries, lambda entry: entry.get("reason_code")),
        "method": _count(entries, lambda entry: entry.get("method")),
        "kind": _count(entries, lambda entry: entry.get("kind")),
        "disposition": _count(entries, lambda entry: entry.get("disposition")),
        "license_policy": _count(entries, lambda entry: entry.get("license_policy")),
        "category": _count(entries, lambda entry: entry.get("category")),
    }
    funnel = {
        "discovered": accounting.discovered,  # type: ignore[attr-defined]
        "resolved": status_counts.get("resolved", 0),
        "unresolved": status_counts.get("unresolved", 0),
        "deferred": status_counts.get("deferred", 0),
        "failed": status_counts.get("failed", 0),
        "acquired": 0,
        "parsed": 0,
        "indexed": 0,
    }
    return {
        "schema_version": 1,
        "report_type": "resolution",
        "run_id": run_id,
        "generated_at": generated_at,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "sample": sample,
        "limit": limit,
        "truncated": truncated,
        "resolver_version": resolver_version,
        "catalog": {
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "mode": snapshot.mode,
            "entries": len(entries),
            "discovered_files": len(snapshot.files),
            "warnings": snapshot.warnings,
        },
        "counts": {"discovered": accounting.discovered, **counts},  # type: ignore[attr-defined]
        "funnel": funnel,
        "invariant": {
            "ok": accounting.ok,  # type: ignore[attr-defined]
            "discovered": accounting.discovered,  # type: ignore[attr-defined]
            "status_total": accounting.status_total,  # type: ignore[attr-defined]
            "missing_reason": accounting.missing_reason[:100],  # type: ignore[attr-defined]
            "problems": accounting.problems,  # type: ignore[attr-defined]
        },
        "manifest": {"path": manifest_path, "changed": manifest_changed},
        "outcomes_added": outcomes_added,
        "unhandled": unhandled,
        "cache": {"hits": cache_hits, "misses": cache_misses},
        "ledger": ledger_result,
        "entries": entries,
    }


def render_resolution_markdown(report: dict[str, object]) -> str:
    catalog = report.get("catalog") or {}
    counts = report.get("counts") or {}
    funnel = report.get("funnel") or {}
    invariant = report.get("invariant") or {}
    cache = report.get("cache") or {}
    assert isinstance(catalog, dict)
    assert isinstance(counts, dict)
    assert isinstance(funnel, dict)
    assert isinstance(invariant, dict)
    assert isinstance(cache, dict)
    lines: list[str] = [
        "# Dataset resolution report",
        "",
        f"- Run: `{report.get('run_id')}`",
        f"- Generated: {report.get('generated_at')}",
        f"- Catalog: `{catalog.get('repo')}` @ `{catalog.get('commit')}`",
        f"- Resolver version: {report.get('resolver_version')}",
        f"- Sample: {report.get('sample')} / Limit: {report.get('limit')}",
        f"- Truncated: {report.get('truncated')}",
        f"- Exit code: {report.get('exit_code')}",
        f"- Cache: {cache.get('hits')} hit(s), {cache.get('misses')} miss(es)",
        "",
        "## Accounting invariant",
        "",
        f"- Holds: **{invariant.get('ok')}**",
        f"- Discovered: {invariant.get('discovered')}",
        f"- Current statuses: {invariant.get('status_total')}",
        f"- Missing reason codes: {len(invariant.get('missing_reason') or [])}",
    ]
    problems = invariant.get("problems") or []
    if problems:
        lines.append(f"- Problems: {', '.join(_md(item) for item in problems)}")
    lines.extend(["", "## Funnel", "", "| Stage | Count |", "|---|---:|"])
    for key, value in funnel.items():
        lines.append(f"| {_md(key)} | {value} |")
    lines.append("")
    for key in (
        "status",
        "reason",
        "method",
        "kind",
        "disposition",
        "license_policy",
        "category",
    ):
        lines.extend(_table(key.replace("_", " ").title(), counts.get(key) or {}))
    ledger = report.get("ledger") or {}
    lines.extend(
        [
            "## Ledger",
            "",
            f"- Enabled: {ledger.get('enabled')}",
            f"- Posted: {ledger.get('posted')}",
            f"- Duplicate (append-once): {ledger.get('duplicate')}",
            f"- Error: {ledger.get('error') or '(none)'}",
            "",
            "## Entries",
            "",
            "| Entry | Status | Reason | Method | Kind | Sources |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for entry in report.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        lines.append(
            "| {entry_id} | {status} | {reason} | {method} | {kind} | {sources} |".format(
                entry_id=_md(entry.get("entry_id")),
                status=_md(entry.get("status")),
                reason=_md(entry.get("reason_code")),
                method=_md(entry.get("method")),
                kind=_md(entry.get("kind")),
                sources=entry.get("source_count"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict[str, object]) -> str:
    if report.get("report_type") == "resolution":
        return render_resolution_markdown(report)
    catalog = report.get("catalog") or {}
    counts = report.get("counts") or {}
    funnel = report.get("funnel") or {}
    invariant = report.get("invariant") or {}
    readme = report.get("readme") or {}
    assert isinstance(catalog, dict)
    assert isinstance(counts, dict)
    assert isinstance(funnel, dict)
    assert isinstance(invariant, dict)
    assert isinstance(readme, dict)
    lines: list[str] = [
        "# Dataset inventory report",
        "",
        f"- Run: `{report.get('run_id')}`",
        f"- Generated: {report.get('generated_at')}",
        f"- Catalog: `{catalog.get('repo')}` @ `{catalog.get('commit')}`",
        f"- Snapshot SHA-256: `{catalog.get('snapshot_sha256')}`",
        f"- Exit code: {report.get('exit_code')}",
        f"- Probe enabled: {report.get('probe_enabled')} ({report.get('probe_checked')} checked)",
        "",
        "## Accounting invariant",
        "",
        f"- Holds: **{invariant.get('ok')}**",
        f"- Discovered: {invariant.get('discovered')}",
        f"- Current statuses: {invariant.get('status_total')}",
        f"- Missing reason codes: {len(invariant.get('missing_reason') or [])}",
    ]
    problems = invariant.get("problems") or []
    if problems:
        lines.append(f"- Problems: {', '.join(_md(item) for item in problems)}")
    lines.extend(["", "## Funnel", "", "| Stage | Count |", "|---|---:|"])
    for key, value in funnel.items():
        lines.append(f"| {_md(key)} | {value} |")
    lines.append("")
    for key in (
        "status",
        "disposition",
        "license_family",
        "access",
        "size_class",
        "format",
        "scheme",
        "scheme_unsupported",
        "readme_status",
    ):
        lines.extend(_table(key.replace("_", " ").title(), counts.get(key) or {}))
    lines.extend(
        [
            "## README reconciliation",
            "",
            f"- Available: {readme.get('available')}",
            f"- Meta links: {readme.get('meta_links')}",
            f"- Entries with OK/FIXME status: {readme.get('entries_with_status')}",
            f"- Entries missing status: {readme.get('entries_missing_status')}",
            f"- Warnings: {', '.join(_md(item) for item in readme.get('warnings') or []) or '(none)'}",
            "",
        ]
    )
    ledger = report.get("ledger") or {}
    lines.extend(
        [
            "## Ledger",
            "",
            f"- Enabled: {ledger.get('enabled')}",
            f"- Posted: {ledger.get('posted')}",
            f"- Duplicate (append-once): {ledger.get('duplicate')}",
            f"- Error: {ledger.get('error') or '(none)'}",
            "",
            "## Entries",
            "",
            "| Entry | Status | Reason | License | Access | Formats |",
            "|---|---|---|---|---|---|",
        ]
    )
    for entry in report.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        lines.append(
            "| {entry_id} | {status} | {reason} | {license} | {access} | {formats} |".format(
                entry_id=_md(entry.get("entry_id")),
                status=_md(entry.get("status")),
                reason=_md(entry.get("reason_code")),
                license=_md(entry.get("license_family")),
                access=_md(entry.get("access")),
                formats=_md(", ".join(str(item) for item in entry.get("formats") or [])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
