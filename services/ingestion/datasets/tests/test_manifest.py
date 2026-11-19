"""Manifest rows, accounting invariant and append-only outcome idempotency."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import catalog, manifest, model

from ingestion.datasets.tests.helpers import make_catalog, make_config


def _build(tmp_path, **config_overrides):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root, **config_overrides)
    snapshot = catalog.snapshot_from_directory(root, config)
    return config, snapshot, manifest.build_rows(snapshot, config)


def _row(rows, title):
    return next(row for row in rows if row.get("title") == title)


def test_build_rows_cover_every_discovered_file(tmp_path):
    config, snapshot, build = _build(tmp_path)
    assert build.discovered == len(snapshot.files) == 12
    assert len(build.rows) == 12
    assert build.unhandled == 0
    assert build.truncated is False
    accounting = manifest.check_accounting(build.rows, build.discovered)
    assert accounting.ok is True
    assert accounting.missing_reason == []


def test_full_row_shape_and_sanitization(tmp_path):
    config, snapshot, build = _build(tmp_path)
    row = _row(build.rows, "Full Entry")
    assert row["entry_id"] == "apd:Test/full.yml"
    assert row["catalog"]["path"] == "Test/full.yml"
    assert row["catalog"]["readme_status"] == "OK"
    assert row["license"]["catalog"] == "MIT"
    assert row["license"]["metadata_repo"] == "GPL-3.0"
    assert row["license"]["dataset"]["family"] == "cc-by"
    assert row["license"]["dataset"]["policy"] == "index_content"
    assert row["license"]["dataset"]["declared_raw"] == (
        "https://creativecommons.org/licenses/by/4.0/"
    )
    assert row["disposition"] == "index_content"
    assert row["status"] == "deferred"
    assert row["reason_code"] == "resolver_not_implemented"
    assert row["probe"] is None
    assert row["classes"]["size"] == {
        "class": "unknown",
        "bytes": None,
        "source": "metadata",
    }
    assert "csv" in row["classes"]["format"]
    # Query strings must not survive into the manifest.
    encoded = json.dumps(row)
    assert "SECRET" not in encoded
    assert row["sources"][1]["access_url"] == "https://example.org/api/full"
    # URLs nested in `extra` are sanitized too.
    assert row["extra"]["image"] == "https://example.org/img.png"
    assert row["extra"]["references"][0]["reference"] == "https://example.org/ref"


def test_license_driven_statuses(tmp_path):
    config, snapshot, build = _build(tmp_path)
    sparse = _row(build.rows, "Sparse Entry")
    assert sparse["disposition"] == "metadata_only_review"
    assert sparse["reason_code"] == "license_unknown_metadata_only"
    restricted = _row(build.rows, "Restricted License")
    assert restricted["disposition"] == "metadata_only_review"
    assert restricted["reason_code"] == "license_restricted"
    proprietary = _row(build.rows, "Proprietary License")
    assert proprietary["disposition"] == "blocked"
    assert proprietary["status"] == "skipped"
    assert proprietary["reason_code"] == "license_restricted"
    ambiguous = _row(build.rows, "License List")
    assert ambiguous["disposition"] == "index_content"
    assert ambiguous["status"] == "deferred"
    assert ambiguous["reason_code"] == "license_review_required"


def test_scheme_anomalies_are_recorded(tmp_path):
    config, snapshot, build = _build(tmp_path)
    hhttps = _row(build.rows, "Hhttps Scheme")
    assert hhttps["classes"]["scheme_unsupported"] == ["hhttps"]
    assert hhttps["classes"]["access"] == "inaccessible"
    assert hhttps["classes"]["access_reason"] == "unsupported_scheme"
    ftp = _row(build.rows, "FTP Scheme")
    assert ftp["classes"]["scheme_unsupported"] == ["ftp"]


def test_limit_marks_remaining_entries_as_budget_deferred(tmp_path):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    build = manifest.build_rows(snapshot, config, limit=3)
    assert build.truncated is True
    assert len(build.rows) == 12
    assert sum(1 for row in build.rows if row["reason_code"] == "run_budget_exhausted") == 9
    accounting = manifest.check_accounting(build.rows, build.discovered)
    assert accounting.ok is True


def test_malformed_entry_is_failed_but_others_are_processed(tmp_path):
    root = make_catalog(tmp_path)
    (root / "core" / "Test" / "bad.yml").write_text("[unclosed\n", encoding="utf-8")
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    build = manifest.build_rows(snapshot, config)
    assert build.discovered == 13
    assert len(build.rows) == 13
    failed = next(row for row in build.rows if row["catalog"]["path"] == "Test/bad.yml")
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "catalog_parse_error"
    assert failed["next_action"] == "fix_metadata"
    assert _row(build.rows, "Full Entry")["status"] == "deferred"
    assert manifest.check_accounting(build.rows, build.discovered).ok is True


def test_duplicate_ids_are_failed(tmp_path, monkeypatch):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    monkeypatch.setattr(model, "entry_id_for", lambda category, filename: "apd:Test/same")
    build = manifest.build_rows(snapshot, config)
    duplicates = [row for row in build.rows if row["reason_code"] == "catalog_duplicate_id"]
    assert len(duplicates) == 11
    assert manifest.check_accounting(build.rows, build.discovered).ok is True


def test_manifest_is_byte_identical_on_re_run(tmp_path):
    config, snapshot, build = _build(tmp_path)
    path = config.data_root / "manifest" / "catalog.jsonl"
    changed, path = manifest.write_manifest(build.rows, path)
    first = path.read_bytes()
    assert changed is True
    changed_again, path = manifest.write_manifest(build.rows, path)
    assert changed_again is False
    assert path.read_bytes() == first
    assert all(line for line in first.splitlines())


def test_changing_one_entry_changes_exactly_one_metadata_hash(tmp_path):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root)
    first_snapshot = catalog.snapshot_from_directory(root, config)
    first = manifest.build_rows(first_snapshot, config)
    first_hashes = {row["entry_id"]: row["metadata_sha256"] for row in first.rows}

    (root / "core" / "Test" / "full.yml").write_text(
        (root / "core" / "Test" / "full.yml")
        .read_text(encoding="utf-8")
        .replace("title: Full Entry", "title: Full Entry Changed"),
        encoding="utf-8",
    )
    second_snapshot = catalog.snapshot_from_directory(root, config)
    second = manifest.build_rows(second_snapshot, config)
    second_hashes = {row["entry_id"]: row["metadata_sha256"] for row in second.rows}

    changed = [
        entry_id
        for entry_id in first_hashes
        if first_hashes[entry_id] != second_hashes[entry_id]
    ]
    assert changed == ["apd:Test/full.yml"]
    assert second_snapshot.snapshot_sha256 != first_snapshot.snapshot_sha256


def test_atomic_manifest_write_leaves_no_temporary_files(tmp_path):
    config, snapshot, build = _build(tmp_path)
    directory = config.data_root / "manifest"
    path = directory / "catalog.jsonl"
    manifest.write_manifest(build.rows, path)
    leftovers = [item.name for item in directory.iterdir() if ".tmp-" in item.name]
    assert leftovers == []


def test_outcomes_are_append_once(tmp_path):
    config, snapshot, build = _build(tmp_path)
    path = config.data_root / "outcomes.jsonl"
    added = manifest.append_outcomes(
        build.rows,
        path,
        run_id="run1",
        catalog_commit=snapshot.commit,
        retrieved_at="2026-01-01T00:00:00Z",
    )
    assert added == 12
    again = manifest.append_outcomes(
        build.rows,
        path,
        run_id="run1",
        catalog_commit=snapshot.commit,
        retrieved_at="2026-01-02T00:00:00Z",
    )
    assert again == 0
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 12
    assert len({line["event_id"] for line in lines}) == 12
    assert all("token=" not in json.dumps(line) for line in lines)


def test_accounting_detects_missing_rows_and_reasons(tmp_path):
    config, snapshot, build = _build(tmp_path)
    missing = manifest.check_accounting(build.rows[:-1], build.discovered)
    assert missing.ok is False
    assert any(item.startswith("row_count:") for item in missing.problems)

    bad = [dict(row) for row in build.rows]
    bad[0]["status"] = "deferred"
    bad[0]["reason_code"] = None
    result = manifest.check_accounting(bad, build.discovered)
    assert result.ok is False
    assert result.missing_reason == [bad[0]["entry_id"]]

    invalid = [dict(row) for row in build.rows]
    invalid[0]["status"] = "exploded"
    result = manifest.check_accounting(invalid, build.discovered)
    assert result.ok is False
    assert any(item.startswith("invalid_status:") for item in result.problems)


def test_run_id_is_deterministic(tmp_path):
    config, snapshot, _build_result = _build(tmp_path)
    first = manifest.run_id_for(config, snapshot)
    second = manifest.run_id_for(config, snapshot)
    assert first == second
    assert len(first) == 16


def test_sanitize_url_strips_secrets():
    assert manifest.sanitize_url("https://user:pass@example.org/x?token=s#frag") == (
        "https://example.org/x"
    )
    assert manifest.sanitize_url("https://example.org:8443/x?q=1") == (
        "https://example.org:8443/x"
    )
    assert manifest.sanitize_url(None) is None
