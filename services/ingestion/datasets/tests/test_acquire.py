"""Acquisition: bounds, checksums, resume, budgets and per-entry outcomes."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import pytest

from ingestion.datasets import acquire, adapters
from ingestion.datasets.netguard import NetguardError
from ingestion.datasets.resolver import Resolution, ResolvedSource

from ingestion.datasets.tests.helpers import (
    FakeDiskUsage,
    FakeStreamResponse,
    FakeStreamTransport,
    make_config,
)

URL = "https://data.example.org/table.csv"


def _row(
    entry_id: str = "apd:Test/full.yml",
    *,
    policy: str = "index_content",
    reason: str | None = None,
    status: str = "deferred",
    formats: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "metadata_sha256": hashlib.sha256(entry_id.encode()).hexdigest(),
        "catalog": {
            "repo": "awesomedata/apd-core",
            "commit": "a" * 40,
            "path": "Test/full.yml",
            "meta_url": "https://github.com/awesomedata/apd-core",
        },
        "title": "Full",
        "category": "Test",
        "disposition": policy,
        "license": {
            "dataset": {
                "family": "cc-by",
                "policy": policy,
                "reason_code": reason,
                "review_required": policy != "index_content",
            }
        },
        "classes": {"format": formats or ["csv"]},
        "status": status,
        "reason_code": "resolver_not_implemented" if status == "deferred" else None,
        "error": None,
    }


def _source(uri: str = URL, **changes) -> ResolvedSource:
    values = {
        "uri": uri,
        "kind": "direct_file",
        "method": "direct_file",
        "media_type": "text/csv",
        "size_bytes": None,
        "member": uri.rsplit("/", 1)[-1],
    }
    values.update(changes)
    return ResolvedSource(**values)


def _resolution(
    entry_id: str = "apd:Test/full.yml",
    sources: tuple[ResolvedSource, ...] | None = None,
    *,
    status: str = "resolved",
    reason: str | None = None,
) -> Resolution:
    return Resolution(
        entry_id=entry_id,
        status=status,
        reason_code=reason,
        checked_at="2026-11-19T00:00:00Z",
        sources=sources if sources is not None else (_source(),),
        method="direct_file",
    )


def _context(tmp_path, routes, *, free: int = 10**12, **overrides):
    config = make_config(tmp_path)
    if overrides:
        config = config.replace(**overrides)
    transport = FakeStreamTransport(routes)
    context = acquire.build_context(
        config,
        transport=transport,
        budget=acquire.RunBudget(config, disk_usage=lambda _path: FakeDiskUsage(free)),
        now="2026-11-19T00:00:00Z",
    )
    return context, transport


def _blob_path(context, digest: str):
    return context.store.blob_path(digest)


def test_acquisition_stores_a_content_addressed_blob(tmp_path):
    body = b"name,value\nalpha,1\nbeta,2\n"
    context, transport = _context(
        tmp_path, {URL: FakeStreamResponse(body, headers={"content-length": str(len(body))})}
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)

    assert result.status == "acquired"
    assert result.files_acquired == 1
    assert result.bytes_total == len(body)
    assert result.records_total == 2
    assert result.parse_status == "parsed"
    assert transport.urls() == [URL]

    digest = hashlib.sha256(body).hexdigest()
    blob = _blob_path(context, digest)
    assert blob.read_bytes() == body
    assert blob.stat().st_mode & 0o777 == 0o600
    item = result.files[0]
    assert item.sha256 == digest
    assert item.blob == f"cache/blobs/{digest[:2]}/{digest}"
    assert item.parse["adapter"] == "csv"
    assert item.parse["record_count"] == 2
    assert json.dumps(result.to_artifact())


def test_declared_oversize_is_refused_before_any_request(tmp_path):
    context, transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"x")}, max_file_bytes=10
    )
    result = acquire.acquire_entry(
        _row(), _resolution(sources=(_source(size_bytes=1000),)), context
    )
    assert result.status == "deferred"
    assert result.reason_code == "oversized_declared"
    assert transport.calls == []


def test_stream_overrun_aborts_and_leaves_no_blob(tmp_path):
    body = b"a" * 4096
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(body)}, max_file_bytes=128
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "oversized_stream"
    assert list((context.config.data_root / "cache" / "blobs").glob("**/*")) == []
    assert not context.store.part_paths(URL)[0].exists()


def test_sha256_checksum_mismatch_is_refused(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"a,b\n1,2\n")}
    )
    result = acquire.acquire_entry(
        _row(), _resolution(sources=(_source(checksum="sha256:" + "0" * 64),)), context
    )
    assert result.reason_code == "checksum_mismatch"
    assert result.retryable is True
    assert list((context.config.data_root / "cache" / "blobs").glob("**/*")) == []


def test_md5_checksum_from_the_resolver_is_verified(tmp_path):
    body = b"a,b\n1,2\n"
    digest = hashlib.md5(body).hexdigest()
    context, _transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    result = acquire.acquire_entry(
        _row(), _resolution(sources=(_source(checksum=f"md5:{digest}"),)), context
    )
    assert result.status == "acquired"
    assert result.files[0].checksum_verified is True
    assert result.files[0].checksum_declared == f"md5:{digest}"


def test_declared_sha256_hit_in_the_store_skips_the_network(tmp_path):
    body = b"a,b\n1,2\n"
    digest = hashlib.sha256(body).hexdigest()
    context, transport = _context(tmp_path, {})
    context.store.blob_path(digest).parent.mkdir(parents=True)
    context.store.blob_path(digest).write_bytes(body)
    result = acquire.acquire_entry(
        _row(), _resolution(sources=(_source(checksum="sha256:" + digest),)), context
    )
    assert result.status == "acquired"
    assert result.files[0].cached is True
    assert result.files[0].sha256 == digest
    assert transport.calls == []


def test_resume_appends_to_a_partial_file(tmp_path):
    body = b"name,value\nalpha,1\nbeta,2\n"
    prefix = body[:9]
    context, transport = _context(tmp_path, {})
    part, _sidecar = context.store.part_paths(URL)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(prefix)
    context.store.write_sidecar(
        URL,
        {"url": URL, "bytes": len(prefix), "etag": '"v1"'},
    )

    def route(headers):
        assert headers.get("Range") == f"bytes={len(prefix)}-"
        assert headers.get("If-Range") == '"v1"'
        return FakeStreamResponse(
            body[len(prefix) :],
            status=206,
            headers={
                "content-range": f"bytes {len(prefix)}-{len(body) - 1}/{len(body)}",
                "content-length": str(len(body) - len(prefix)),
                "etag": '"v1"',
            },
        )

    transport.add(URL, route)
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.status == "acquired", result.detail
    assert result.files[0].resumed is True
    assert result.files[0].sha256 == hashlib.sha256(body).hexdigest()
    assert result.files[0].attempts == 1
    assert _blob_path(context, hashlib.sha256(body).hexdigest()).read_bytes() == body
    assert not part.exists()


def test_server_ignoring_the_range_restarts_from_zero(tmp_path):
    body = b"a,b\n1,2\n"
    context, transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    part, _sidecar = context.store.part_paths(URL)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a,b\n")
    context.store.write_sidecar(URL, {"url": URL, "bytes": 4})
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.status == "acquired", result.detail
    # A 200 while a range was requested restarts in the same attempt.
    assert result.files[0].resumed is False
    assert result.files[0].attempts == 1
    assert _blob_path(context, hashlib.sha256(body).hexdigest()).read_bytes() == body


def test_unsatisfiable_range_restarts(tmp_path):
    body = b"a,b\n1,2\n"
    context, transport = _context(tmp_path, {})
    part, _sidecar = context.store.part_paths(URL)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a,b\n")
    context.store.write_sidecar(URL, {"url": URL, "bytes": 4})
    calls = {"count": 0}

    def route(headers):
        calls["count"] += 1
        if "Range" in headers:
            return FakeStreamResponse(b"", status=416)
        return FakeStreamResponse(body)

    transport.add(URL, route)
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.status == "acquired", result.detail
    assert calls["count"] == 2
    assert result.files[0].attempts == 2


def test_content_length_mismatch_keeps_the_partial_for_resume(tmp_path):
    context, _transport = _context(
        tmp_path,
        {URL: FakeStreamResponse(b"abcde", headers={"content-length": "100"})},
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "partial_download"
    part, sidecar = context.store.part_paths(URL)
    assert part.read_bytes() == b"abcde"
    assert json.loads(sidecar.read_text())["bytes"] == 5
    assert list((context.config.data_root / "cache" / "blobs").glob("**/*")) == []


def test_html_served_as_csv_is_a_content_type_mismatch(tmp_path):
    body = b"<!DOCTYPE html><html><body>Sign in</body></html>"
    context, _transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "content_type_mismatch"
    assert not context.store.part_paths(URL)[0].exists()


def test_binary_served_as_csv_is_a_content_type_mismatch(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"\x00\x01\x02binary")}
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "content_type_mismatch"


def test_http_error_maps_to_a_reason_code(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"", status=404)}
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "not_found"
    assert result.status == "deferred"
    assert result.retryable is True


def test_netguard_error_is_mapped(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: NetguardError("ssrf_blocked", "non-public address")}
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.reason_code == "ssrf_blocked"
    assert result.status == "failed"


def test_disk_guard_defers_the_entry(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"a,b\n1,2\n")}, free=0
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.status == "deferred"
    assert result.reason_code == "disk_guard"


def test_run_byte_budget_exhaustion(tmp_path):
    context, _transport = _context(
        tmp_path, {URL: FakeStreamResponse(b"a,b\n1,2\n")}, max_run_bytes=1
    )
    result = acquire.acquire_entry(_row(), _resolution(), context)
    assert result.status == "deferred"
    assert result.reason_code == "run_budget_exhausted"


def test_run_dataset_cap(tmp_path):
    context, transport = _context(
        tmp_path,
        {
            "https://a.example.org/a.csv": FakeStreamResponse(b"a\n1\n"),
            "https://b.example.org/b.csv": FakeStreamResponse(b"b\n2\n"),
        },
        max_run_datasets=1,
    )
    first = acquire.acquire_entry(
        _row("apd:Test/a.yml"),
        _resolution("apd:Test/a.yml", (_source("https://a.example.org/a.csv"),)),
        context,
    )
    second = acquire.acquire_entry(
        _row("apd:Test/b.yml"),
        _resolution("apd:Test/b.yml", (_source("https://b.example.org/b.csv"),)),
        context,
    )
    assert first.status == "acquired"
    assert second.status == "deferred"
    assert second.reason_code == "run_budget_exhausted"
    assert transport.urls() == ["https://a.example.org/a.csv"]


def test_dataset_byte_cap_stops_extra_files(tmp_path):
    context, transport = _context(
        tmp_path,
        {
            "https://a.example.org/a.csv": FakeStreamResponse(b"a\n" * 10),
            "https://b.example.org/b.csv": FakeStreamResponse(b"b\n" * 10),
        },
        max_dataset_bytes=30,
    )
    result = acquire.acquire_entry(
        _row(),
        _resolution(
            sources=(
                _source("https://a.example.org/a.csv"),
                _source("https://b.example.org/b.csv"),
            )
        ),
        context,
    )
    # A partial dataset keeps a retryable reason; the stored file stays recorded.
    assert result.status == "deferred"
    assert result.reason_code == "oversized_stream"
    assert result.files_acquired == 1
    assert result.files_failed == 1
    # The second file is attempted but refused once the dataset cap is spent.
    assert transport.urls() == [
        "https://a.example.org/a.csv",
        "https://b.example.org/b.csv",
    ]
    failed = [item for item in result.files if item.status == "failed"]
    assert failed[0].reason_code == "oversized_stream"


def test_remote_filename_is_never_used_as_a_path(tmp_path):
    url = "https://data.example.org/../../etc/evil.csv"
    context, _transport = _context(tmp_path, {url: FakeStreamResponse(b"a,b\n1,2\n")})
    result = acquire.acquire_entry(
        _row(), _resolution(sources=(_source(url),)), context
    )
    assert result.status == "acquired"
    for path in (context.config.data_root / "cache" / "blobs").rglob("*"):
        assert "evil" not in path.name
    assert not (tmp_path / "evil.csv").exists()


def test_non_index_content_is_skipped_with_a_reason(tmp_path):
    context, transport = _context(tmp_path, {URL: FakeStreamResponse(b"a,b\n1,2\n")})
    row = _row(policy="metadata_only_review", reason="license_unknown_metadata_only")
    result = acquire.acquire_entry(row, _resolution(), context)
    assert result.status == "skipped"
    assert result.reason_code == "license_unknown_metadata_only"
    assert transport.calls == []


def test_unresolved_resolution_reason_is_inherited(tmp_path):
    context, transport = _context(tmp_path, {})
    resolution = _resolution(status="unresolved", reason="landing_page_only")
    result = acquire.acquire_entry(_row(), resolution, context)
    assert result.status == "deferred"
    assert result.reason_code == "landing_page_only"
    assert transport.calls == []


def test_missing_resolution_is_explicit(tmp_path):
    context, transport = _context(tmp_path, {})
    result = acquire.acquire_entry(_row(), None, context)
    assert result.status == "skipped"
    assert result.reason_code == "not_resolved"
    assert result.next_action == "resolve"
    assert transport.calls == []


def test_archived_content_is_inspected_and_parsed(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("data/table.csv", "a,b\n1,2\n3,4\n")
        handle.writestr("__MACOSX/._table.csv", b"junk")
    body = buffer.getvalue()
    url = "https://data.example.org/bundle.zip"
    context, _transport = _context(tmp_path, {url: FakeStreamResponse(body)})
    result = acquire.acquire_entry(
        _row(),
        _resolution(sources=(_source(url, kind="archive", media_type="application/zip"),)),
        context,
    )
    assert result.status == "acquired"
    assert result.records_total == 2
    item = result.files[0]
    assert item.archive["format"] == "zip"
    assert item.archive["status"] == "parsed"
    assert item.archive["junk"] == ["__MACOSX/._table.csv"]
    assert item.archive["members"][0]["parse"]["record_count"] == 2


def test_hostile_archive_is_acquired_but_not_parsed(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("../escape.csv", "a,b\n1,2\n")
    url = "https://data.example.org/evil.zip"
    context, _transport = _context(tmp_path, {url: FakeStreamResponse(buffer.getvalue())})
    result = acquire.acquire_entry(
        _row(),
        _resolution(sources=(_source(url, kind="archive", media_type="application/zip"),)),
        context,
    )
    assert result.status == "acquired"
    assert result.records_total == 0
    assert result.files[0].archive["status"] == "failed"
    assert result.files[0].archive["reason_code"] == "archive_unsafe_path"


def test_partial_entry_retries_only_the_failed_source(tmp_path):
    ok_url = "https://data.example.org/ok.csv"
    bad_url = "https://data.example.org/missing.csv"
    context, transport = _context(
        tmp_path,
        {
            ok_url: FakeStreamResponse(b"a,b\n1,2\n"),
            bad_url: FakeStreamResponse(b"", status=404),
        },
    )
    resolution = _resolution(sources=(_source(ok_url), _source(bad_url)))
    first = acquire.acquire_entry(_row(), resolution, context)
    assert first.status == "deferred"
    assert first.reason_code == "not_found"
    assert first.files_acquired == 1
    assert transport.urls() == [ok_url, bad_url]

    second_context, second_transport = _context(
        tmp_path, {bad_url: FakeStreamResponse(b"", status=404)}
    )
    second = acquire.acquire_entry(_row(), resolution, second_context)
    assert second.status == "deferred"
    assert second.reason_code == "not_found"
    assert second.files_acquired == 1
    # The stored sibling is reused; only the failure is retried.
    assert second_transport.urls() == [bad_url]


def test_repository_source_is_refused_without_downloading_html(tmp_path):
    url = "https://github.com/owner/repo"
    context, transport = _context(tmp_path, {})
    result = acquire.acquire_entry(
        _row(),
        _resolution(sources=(_source(url, kind="repository", media_type=None),)),
        context,
    )
    assert result.status == "skipped"
    assert result.reason_code == "repository_requires_clone"
    assert result.files[0].kind == "repository"
    assert transport.calls == []


def test_a_parser_bump_reparses_the_stored_blob_without_a_refetch(tmp_path):
    body = b"a,b\n1,2\n3,4\n"
    context, transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    first = acquire.acquire_entry(_row(), _resolution(), context)
    assert first.records_total == 2

    cached = next((context.config.data_root / "cache" / "acquisitions").glob("*.json"))
    payload = json.loads(cached.read_text())
    payload["files"][0]["parse"]["adapter_version"] = 0
    payload["files"][0]["parse"]["record_count"] = 999
    cached.write_text(json.dumps(payload))

    second_context, second_transport = _context(tmp_path, {})
    second = acquire.acquire_entry(_row(), _resolution(), second_context)
    assert second.status == "unchanged"
    assert second_transport.calls == []
    assert second.records_total == 2
    assert second.files[0].parse["adapter_version"] == adapters.ADAPTER_VERSION
    assert second.files[0].parse["record_count"] == 2
    assert transport.urls() == [URL]


def test_acquisition_cache_makes_a_second_run_offline(tmp_path):
    body = b"a,b\n1,2\n"
    context, transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    first = acquire.acquire_entry(_row(), _resolution(), context)
    assert first.status == "acquired"

    # A fresh context (new transport and cache reader) over the same data root.
    second_context, second_transport = _context(tmp_path, {})
    second = acquire.acquire_entry(_row(), _resolution(), second_context)
    assert second.status == "unchanged"
    assert second.cached is True
    assert second.bytes_total == first.bytes_total
    assert second_transport.calls == []
    assert transport.urls() == [URL]


def test_no_reuse_forces_a_redownload(tmp_path):
    body = b"a,b\n1,2\n"
    context, _transport = _context(tmp_path, {URL: FakeStreamResponse(body)})
    acquire.acquire_entry(_row(), _resolution(), context)
    second_context, second_transport = _context(
        tmp_path, {URL: FakeStreamResponse(body)}, acquire_reuse_enabled=False
    )
    second = acquire.acquire_entry(_row(), _resolution(), second_context)
    assert second.status == "acquired"
    assert second_transport.urls() == [URL]


def test_select_entries_only_takes_index_content(tmp_path):
    rows = [
        _row("apd:Test/a.yml"),
        _row("apd:Test/b.yml", policy="metadata_only_review"),
        _row("apd:Test/c.yml", policy="blocked"),
        _row("apd:Test/d.yml", status="failed"),
    ]
    selected, truncated = acquire.select_entries(rows)
    assert selected == {"apd:Test/a.yml"}
    assert truncated is False
    limited, truncated = acquire.select_entries(rows, limit=0)
    assert limited == set()
    assert truncated is True


def test_select_named_takes_explicit_ids_only():
    rows = [
        _row("apd:Test/a.yml"),
        _row("apd:Test/b.yml", policy="metadata_only_review"),
    ]
    accepted, unknown = acquire.select_named(
        rows, ["apd:Test/a.yml", "apd:Test/b.yml", "apd:Test/missing.yml"]
    )
    assert accepted == {"apd:Test/a.yml"}
    assert unknown == ["apd:Test/b.yml", "apd:Test/missing.yml"]


def test_complete_acquisitions_gives_every_row_one_status(tmp_path):
    rows = [
        _row("apd:Test/a.yml"),
        _row("apd:Test/b.yml"),
        _row("apd:Test/c.yml", policy="metadata_only_review", reason="license_unknown_metadata_only"),
        _row("apd:Test/d.yml", status="failed"),
    ]
    rows[3]["reason_code"] = "catalog_parse_error"
    acquired = [acquire.Acquisition(entry_id="apd:Test/a.yml", status="acquired")]
    completed = acquire.complete_acquisitions(
        rows, acquired, now="2026-11-19T00:00:00Z", config=make_config(tmp_path)
    )
    by_id = {item.entry_id: item for item in completed}
    assert by_id["apd:Test/a.yml"].status == "acquired"
    assert by_id["apd:Test/b.yml"].reason_code == "run_budget_exhausted"
    assert by_id["apd:Test/c.yml"].reason_code == "license_unknown_metadata_only"
    assert by_id["apd:Test/d.yml"].reason_code == "catalog_parse_error"

    accounting = acquire.check_acquisition_accounting(completed, len(rows))
    assert accounting.ok is True
    assert accounting.status_counts == {"acquired": 1, "deferred": 1, "failed": 1, "skipped": 1}


def test_accounting_flags_a_manufactured_mismatch():
    completed = [acquire.Acquisition(entry_id="apd:Test/a.yml", status="deferred")]
    accounting = acquire.check_acquisition_accounting(completed, 2)
    assert accounting.ok is False
    assert any(problem.startswith("row_count:") for problem in accounting.problems)
    assert accounting.missing_reason == ["apd:Test/a.yml"]


#: The design's outcome taxonomy for the acquisition, archive and parse stages.
PR3_REASON_CODES = frozenset(
    {
        # acquisition
        "oversized_declared",
        "oversized_stream",
        "content_type_mismatch",
        "checksum_mismatch",
        "partial_download",
        "disk_guard",
        "run_budget_exhausted",
        # network (inherited from resolution)
        "not_found",
        "gone",
        "server_error",
        "rate_limited",
        "timeout",
        "dns_failure",
        "tls_failure",
        "connection_failed",
        "http_error",
        "too_many_redirects",
        "unsafe_redirect",
        "ssrf_blocked",
        "gated_authentication",
        "blocked",
        "requires_javascript",
        "requires_interaction",
        "robots_disallowed",
        "invalid_url",
        "unsupported_scheme",
        "no_source_url",
        "landing_page_only",
        "resolver_not_implemented",
        "doi_unresolved",
        # archive
        "archive_unsafe_path",
        "archive_link_or_device",
        "archive_bomb",
        "archive_too_many_entries",
        "archive_encrypted",
        "archive_nested",
        # parse
        "unsupported_format",
        "adapter_unavailable",
        "parse_error",
        "empty_content",
        "encoding_error",
        "xml_unsafe",
        "limits_exceeded",
        # license / selection
        "license_unknown_metadata_only",
        "license_restricted",
        "license_review_required",
        "not_resolved",
        "not_selected",
        "repository_requires_clone",
        # catalog (inherited from PR 1)
        "catalog_parse_error",
        "catalog_entry_invalid",
        "catalog_duplicate_id",
        "catalog_snapshot_failed",
    }
)


def test_every_retryable_reason_is_in_the_taxonomy():
    assert acquire.RETRYABLE_REASONS <= PR3_REASON_CODES


def test_outcome_pairs_are_in_the_taxonomy():
    pairs = [
        ("acquired", None),
        ("unchanged", None),
        ("skipped", "license_unknown_metadata_only"),
        ("skipped", "not_resolved"),
        ("skipped", "repository_requires_clone"),
        ("deferred", "oversized_declared"),
        ("deferred", "checksum_mismatch"),
        ("deferred", "partial_download"),
        ("deferred", "disk_guard"),
        ("deferred", "run_budget_exhausted"),
        ("deferred", "rate_limited"),
        ("failed", "content_type_mismatch"),
        ("failed", "ssrf_blocked"),
        ("failed", "archive_unsafe_path"),
        ("failed", "xml_unsafe"),
    ]
    for status, code in pairs:
        assert status in acquire.VALID_ACQUISITION_STATUSES
        if code is not None:
            assert code in PR3_REASON_CODES


def test_archive_and_adapter_codes_are_in_the_taxonomy(tmp_path):
    from ingestion.datasets import archive

    assert archive.ArchiveError("archive_bomb").code in PR3_REASON_CODES
    for code in (
        "archive_unsafe_path",
        "archive_link_or_device",
        "archive_bomb",
        "archive_too_many_entries",
        "archive_encrypted",
        "archive_nested",
    ):
        assert code in PR3_REASON_CODES
    for code in (
        "unsupported_format",
        "adapter_unavailable",
        "parse_error",
        "empty_content",
        "encoding_error",
        "xml_unsafe",
        "limits_exceeded",
    ):
        assert code in PR3_REASON_CODES


def test_run_time_budget(tmp_path):
    ticks = {"value": 0.0}

    def clock() -> float:
        ticks["value"] += 100.0
        return ticks["value"]

    config = make_config(tmp_path).replace(max_run_seconds=150)
    budget = acquire.RunBudget(config, clock=clock, disk_usage=lambda _path: FakeDiskUsage(10**12))
    budget.check_run()
    with pytest.raises(acquire.BudgetExceeded) as error:
        budget.check_run()
    assert error.value.code == "run_budget_exhausted"


def test_checksum_parsing():
    assert acquire.parse_checksum("sha256:" + "a" * 64) == ("sha256", "a" * 64)
    assert acquire.parse_checksum("md5:" + "b" * 32) == ("md5", "b" * 32)
    assert acquire.parse_checksum("c" * 40) == ("sha1", "c" * 40)
    assert acquire.parse_checksum("not-a-digest") is None
    assert acquire.parse_checksum(None) is None
