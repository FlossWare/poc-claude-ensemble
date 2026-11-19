"""End-to-end ``acquire`` CLI: offline transport, manifest, resume, dry run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import dataset_ingest
from ingestion.datasets import acquire, catalog, manifest, resolver, report
from ingestion.datasets.config import Config
from ingestion.datasets.resolver import Resolution, ResolvedSource

from ingestion.datasets.tests.helpers import (
    FakeStreamResponse,
    FakeStreamTransport,
    make_catalog,
)

CSV_URL = "https://data.example.org/table.csv"
JSON_URL = "https://data.example.org/table.json"
JSONL_URL = "https://data.example.org/rows.jsonl"

CSV_BODY = b"name,value\nalpha,1\nbeta,2\n"
JSON_BODY = json.dumps([{"name": "alpha"}, {"name": "beta"}]).encode()
JSONL_BODY = b'{"name": "alpha"}\n{"name": "beta"}\n'


def _setup(tmp_path):
    root = make_catalog(tmp_path, names=("full.yml", "restricted.yml"))
    data_root = tmp_path / "data"
    config = Config.load(catalog_dir=root, data_root=data_root, ledger_enabled=False)
    snapshot = catalog.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)
    return root, data_root, config, build


def _seed(config, rows, entry_id: str, sources: list[ResolvedSource]):
    row = next(item for item in rows if item["entry_id"] == entry_id)
    resolver.ResolutionCache(config).put(
        Resolution(
            entry_id=entry_id,
            status="resolved",
            reason_code=None,
            checked_at=resolver.utc_now(),
            sources=tuple(sources),
            method="direct_file",
        ),
        str(row["metadata_sha256"]),
    )


def _transport() -> FakeStreamTransport:
    return FakeStreamTransport(
        {
            CSV_URL: FakeStreamResponse(CSV_BODY, headers={"content-length": str(len(CSV_BODY))}),
            JSON_URL: FakeStreamResponse(JSON_BODY),
            JSONL_URL: FakeStreamResponse(JSONL_BODY),
        }
    )


def _run(root: Path, data_root: Path, extra: list[str] | None = None) -> int:
    argv = [
        "acquire",
        "--catalog-dir",
        str(root),
        "--data-root",
        str(data_root),
        "--no-ledger",
        *(extra or ["--limit", "1"]),
    ]
    return dataset_ingest.main(argv)


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_acquire_cli_end_to_end(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    other = [row["entry_id"] for row in build.rows if row["entry_id"] != "apd:Test/full.yml"]
    assert other, "the blocked fixture must still be present"
    _seed(
        config,
        build.rows,
        "apd:Test/full.yml",
        [
            ResolvedSource(uri=CSV_URL, kind="direct_file", method="direct_file", media_type="text/csv", member="table.csv"),
            ResolvedSource(uri=JSON_URL, kind="direct_file", method="direct_file", media_type="application/json", member="table.json"),
            ResolvedSource(uri=JSONL_URL, kind="direct_file", method="direct_file", media_type="application/x-ndjson", member="rows.jsonl"),
        ],
    )
    transport = _transport()
    monkeypatch.setattr(acquire, "build_transport", lambda config: transport)

    assert _run(root, data_root) == 0
    out = capsys.readouterr().out
    assert "acquire: discovered=2" in out
    assert "'acquired': 1" in out
    assert "'skipped': 1" in out
    assert transport.urls() == [CSV_URL, JSON_URL, JSONL_URL]

    acquisition_manifest = next((data_root / "manifest").glob("*.acquisition.jsonl"))
    rows = _rows(acquisition_manifest)
    assert len(rows) == 2
    by_id = {str(row["entry_id"]): row for row in rows}
    full = by_id["apd:Test/full.yml"]["acquisition"]
    assert full["status"] == "acquired"
    assert full["files_acquired"] == 3
    assert full["records_total"] == 6
    assert {item["parse"]["adapter"] for item in full["files"]} == {"csv", "json", "jsonl"}
    for item in full["files"]:
        assert len(item["sha256"]) == 64
        assert item["bytes"] > 0
    skipped = by_id[other[0]]["acquisition"]
    assert skipped["status"] == "skipped"
    assert skipped["reason_code"] == "license_restricted"

    outcomes = _rows(data_root / "outcomes.jsonl")
    assert len(outcomes) == 2
    assert {row["stage"] for row in outcomes} == {"acquire"}
    assert all(row["reason_code"] or row["status"] == "acquired" for row in outcomes)

    run_report = json.loads(next((data_root / "runs").glob("*.json")).read_text())
    assert run_report["report_type"] == "acquisition"
    assert run_report["invariant"]["ok"] is True
    assert run_report["funnel"]["files"] == 3
    assert run_report["funnel"]["records"] == 6
    assert run_report["counts"]["adapter"] == {"csv": 1, "json": 1, "jsonl": 1}
    assert run_report["budget"]["bytes_used"] == len(CSV_BODY) + len(JSON_BODY) + len(JSONL_BODY)

    # A second run reuses the stored content: no network at all.
    before = len(outcomes)
    assert _run(root, data_root) == 0
    second = capsys.readouterr().out
    assert "'unchanged': 1" in second
    assert transport.urls() == [CSV_URL, JSON_URL, JSONL_URL]
    # Outcomes are append-once by deterministic event id: the unchanged entry
    # adds one row ("unchanged") and the skipped row is a duplicate, so it is
    # not appended again.
    after = _rows(data_root / "outcomes.jsonl")
    assert len(after) == before + 1
    assert after[-1]["status"] == "unchanged"


def test_acquire_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _seed(
        config,
        build.rows,
        "apd:Test/full.yml",
        [ResolvedSource(uri=CSV_URL, kind="direct_file", method="direct_file", member="table.csv")],
    )
    transport = _transport()
    monkeypatch.setattr(acquire, "build_transport", lambda config: transport)

    assert _run(root, data_root, extra=["--limit", "1", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["selected"] == 1
    planned = [item for item in payload["plan"] if item["status"] == "planned"]
    assert len(planned) == 1
    assert planned[0]["sources"][0]["uri"] == CSV_URL
    assert transport.calls == []
    assert not (data_root / "manifest").exists()
    assert not (data_root / "outcomes.jsonl").exists()
    assert not (data_root / "runs").exists()


def test_acquire_without_resolution_evidence_is_explicit(tmp_path, monkeypatch, capsys):
    root, data_root, _config, _build = _setup(tmp_path)
    transport = _transport()
    monkeypatch.setattr(acquire, "build_transport", lambda config: transport)

    assert _run(root, data_root, extra=["--limit", "1", "--no-resolve"]) == 0
    acquisition_manifest = next((data_root / "manifest").glob("*.acquisition.jsonl"))
    rows = _rows(acquisition_manifest)
    full = next(row for row in rows if row["entry_id"] == "apd:Test/full.yml")
    assert full["acquisition"]["status"] == "skipped"
    assert full["acquisition"]["reason_code"] == "not_resolved"
    assert full["acquisition"]["next_action"] == "resolve"
    assert transport.calls == []


def test_report_cli_renders_the_acquisition_report(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _seed(
        config,
        build.rows,
        "apd:Test/full.yml",
        [ResolvedSource(uri=CSV_URL, kind="direct_file", method="direct_file", member="table.csv")],
    )
    monkeypatch.setattr(acquire, "build_transport", lambda config: FakeStreamTransport({
        CSV_URL: FakeStreamResponse(CSV_BODY)
    }))
    assert _run(root, data_root) == 0
    capsys.readouterr()

    assert dataset_ingest.main(["report", "--format", "md", "--data-root", str(data_root)]) == 0
    markdown = capsys.readouterr().out
    assert "# Dataset acquisition report" in markdown
    assert "apd:Test/full.yml" in markdown
    assert "sha" in markdown.lower()


def test_render_acquisition_markdown_handles_an_empty_run(tmp_path):
    root, data_root, config, build = _setup(tmp_path)
    payload = report.build_acquisition_report(
        build.rows,
        [],
        run_id="r",
        snapshot=catalog.load_catalog(config),
        accounting=acquire.check_acquisition_accounting([], len(build.rows)),
        generated_at="2026-11-19T00:00:00Z",
        duration_ms=1,
        exit_code=2,
        manifest_path="m",
        manifest_changed=False,
        outcomes_added=0,
        ledger_result={"enabled": False, "posted": False},
        acquire_version=1,
        adapter_version=1,
        archive_version=1,
        sample=None,
        limit=None,
        unhandled=0,
        truncated=False,
        cache_hits=0,
        cache_misses=0,
    )
    text = report.render_markdown(payload)
    assert "# Dataset acquisition report" in text
    assert "Holds: **False**" in text
