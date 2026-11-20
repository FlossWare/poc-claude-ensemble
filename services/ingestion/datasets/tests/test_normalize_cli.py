"""End-to-end ``normalize`` CLI: acquire -> parse -> normalize -> index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import dataset_ingest
from ingestion.datasets import acquire, catalog, indexing, manifest, resolver
from ingestion.datasets.config import Config
from ingestion.datasets.resolver import Resolution, ResolvedSource

from ingestion.datasets.tests.helpers import (
    FakeIntegrationBackend,
    FakeStreamResponse,
    FakeStreamTransport,
    make_catalog,
)

CSV_URL = "https://data.example.org/table.csv"
JSONL_URL = "https://data.example.org/rows.jsonl"
CSV_BODY = b"name,value\nalpha,1\nbeta,2\n"
JSONL_BODY = b'{"name": "alpha"}\n{"name": "beta"}\n{"name": "gamma"}\n'


def _setup(tmp_path):
    root = make_catalog(tmp_path, names=("full.yml", "sparse.yml"))
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
            JSONL_URL: FakeStreamResponse(JSONL_BODY),
        }
    )


def _acquire(root: Path, data_root: Path, config, rows, monkeypatch) -> None:
    _seed(
        config,
        rows,
        "apd:Test/full.yml",
        [
            ResolvedSource(
                uri=CSV_URL,
                kind="direct_file",
                method="direct_file",
                media_type="text/csv",
                member="table.csv",
            ),
            ResolvedSource(
                uri=JSONL_URL,
                kind="direct_file",
                method="direct_file",
                media_type="application/x-ndjson",
                member="rows.jsonl",
            ),
        ],
    )
    transport = _transport()
    monkeypatch.setattr(acquire, "build_transport", lambda config: transport)
    assert (
        dataset_ingest.main(
            [
                "acquire",
                "--catalog-dir",
                str(root),
                "--data-root",
                str(data_root),
                "--no-ledger",
                "--entry",
                "apd:Test/full.yml",
            ]
        )
        == 0
    )


def _normalize(root: Path, data_root: Path, extra: list[str] | None = None) -> int:
    return dataset_ingest.main(
        [
            "normalize",
            "--catalog-dir",
            str(root),
            "--data-root",
            str(data_root),
            "--no-ledger",
            *(extra or []),
        ]
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wire(monkeypatch, backend: FakeIntegrationBackend) -> None:
    monkeypatch.setattr(indexing, "post_json", backend.post)
    monkeypatch.setattr(indexing, "source_state", backend.state)


def test_normalize_cli_indexes_acquired_content_end_to_end(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _acquire(root, data_root, config, build.rows, monkeypatch)
    backend = FakeIntegrationBackend()
    _wire(monkeypatch, backend)

    assert _normalize(root, data_root) == 0
    out = capsys.readouterr().out
    assert "normalize: discovered=2" in out
    assert "'indexed': 1" in out, (out[-1200:], backend.paths(), list(backend.index))
    assert "'deferred': 1" in out

    integration_manifest = next((data_root / "manifest").glob("*.integration.jsonl"))
    rows = {str(row["entry_id"]): row for row in _rows(integration_manifest)}
    full = rows["apd:Test/full.yml"]["integration"]
    assert full["status"] == "indexed"
    assert full["records_total"] == 5
    assert full["shards"] == 2
    assert full["metadata"]["status"] == "indexed"
    assert full["content"]["indexed"] == 2
    assert full["chunks"] == 3 and full["vectors"] == 3
    source_ids = [item["source_id"] for item in full["content"]["records"]]
    assert all(source_id.startswith("apd:Test/full.yml/src/") for source_id in source_ids)
    assert all(source_id.endswith("/shard/0000") for source_id in source_ids)
    sparse = rows["apd:Test/sparse.yml"]["integration"]
    assert sparse["status"] == "deferred"
    assert sparse["reason_code"] == "not_selected"

    # Knowledge holds the metadata record and both content shards.
    assert set(backend.knowledge) == {
        "apd:Test/full.yml",
        *source_ids,
    }
    metadata_record = backend.knowledge["apd:Test/full.yml"]
    assert metadata_record["provenance"]["origin"].startswith("apd-core@")
    assert "title: Full Entry" in metadata_record["content"]
    assert all(record["content"] for record in backend.knowledge.values())

    run_report = json.loads(next((data_root / "runs").glob("*.json")).read_text())
    assert run_report["report_type"] == "integration"
    assert run_report["invariant"]["ok"] is True
    assert run_report["funnel"]["discovered"] == 2
    assert run_report["funnel"]["indexed"] == 1
    assert run_report["funnel"]["metadata_indexed"] == 0
    assert run_report["funnel"]["deferred"] == 1
    assert run_report["funnel"]["knowledge_records"] == 3
    assert run_report["counts"]["knowledge_decision"] == {"accepted": 3}

    outcomes = _rows(data_root / "outcomes.jsonl")
    integrate_rows = [row for row in outcomes if row["stage"] == "integrate"]
    assert len(integrate_rows) == 4  # three records plus the unselected entry
    for row in integrate_rows:
        if row["status"] not in {"indexed", "metadata_indexed"}:
            assert row["reason_code"], row
    assert {row["knowledge_source_id"] for row in integrate_rows} >= set(source_ids)

    # Second run: content is unchanged, so nothing is written to a service.
    backend.calls.clear()
    assert _normalize(root, data_root) == 0
    second = capsys.readouterr().out
    assert "'unchanged': 1" in second
    assert backend.calls == []
    assert set(backend.knowledge) == {"apd:Test/full.yml", *source_ids}

    markdown = dataset_ingest.main(
        ["report", "--format", "md", "--data-root", str(data_root)]
    )
    assert markdown == 0
    rendered = capsys.readouterr().out
    assert "# Dataset normalization and indexing report" in rendered
    assert "Non-indexed records" in rendered


def test_normalize_cli_includes_metadata_only_entries(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _acquire(root, data_root, config, build.rows, monkeypatch)
    backend = FakeIntegrationBackend()
    _wire(monkeypatch, backend)

    assert _normalize(root, data_root, ["--include-metadata-only"]) == 0
    capsys.readouterr()
    assert "apd:Test/sparse.yml" in backend.knowledge
    metadata = backend.knowledge["apd:Test/sparse.yml"]
    assert "title:" in metadata["content"]
    integration_manifest = next((data_root / "manifest").glob("*.integration.jsonl"))
    rows = {str(row["entry_id"]): row for row in _rows(integration_manifest)}
    sparse = rows["apd:Test/sparse.yml"]["integration"]
    assert sparse["status"] == "metadata_indexed"
    assert sparse["reason_code"] == "license_unknown_metadata_only"
    assert sparse["content"]["shards"] == 0


def test_normalize_cli_reports_partial_failure_per_record(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _acquire(root, data_root, config, build.rows, monkeypatch)
    backend = FakeIntegrationBackend()
    _wire(monkeypatch, backend)
    backend.fail_paths = {"/embeddings"}

    assert _normalize(root, data_root) == 0
    capsys.readouterr()
    integration_manifest = next((data_root / "manifest").glob("*.integration.jsonl"))
    rows = {str(row["entry_id"]): row for row in _rows(integration_manifest)}
    full = rows["apd:Test/full.yml"]["integration"]
    # Knowledge stored every revision, but the retryable downstream failure
    # leaves the entry deferred so the next run completes the indexes.
    assert full["status"] == "deferred"
    assert full["reason_code"] == "dependency_unavailable"
    assert full["metadata"]["status"] == "failed"
    assert full["content"]["failed"] == 2
    assert full["chunks"] == 0
    for record in full["content"]["records"]:
        assert record["status"] == "failed"
        assert record["reason_code"] == "dependency_unavailable"

    # The failure heals on the next run without a second Knowledge revision.
    backend.fail_paths = set()
    backend.calls.clear()
    assert _normalize(root, data_root) == 0
    third = capsys.readouterr().out
    assert "'indexed': 1" in third
    assert "/records" not in backend.paths()


def test_normalize_cli_dry_run_makes_no_service_calls(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _acquire(root, data_root, config, build.rows, monkeypatch)
    backend = FakeIntegrationBackend()
    _wire(monkeypatch, backend)

    capsys.readouterr()  # discard the acquire output
    before = (data_root / "outcomes.jsonl").stat().st_size
    assert _normalize(root, data_root, ["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["selected"] == 1
    plan = payload["plan"][0]
    assert plan["entry_id"] == "apd:Test/full.yml"
    assert plan["metadata_record"]["kind"] == "metadata"
    assert len(plan["content_records"]) == 2
    assert backend.calls == []
    assert (data_root / "outcomes.jsonl").stat().st_size == before
    assert not list((data_root / "manifest").glob("*.integration.jsonl"))


def test_normalize_cli_requires_an_acquisition_manifest(tmp_path, capsys):
    root, data_root, _config, _build = _setup(tmp_path)
    assert _normalize(root, data_root) == 1
    assert "run the acquire stage first" in capsys.readouterr().err


def test_normalize_cli_limit_defers_the_rest_and_exits_3(tmp_path, monkeypatch, capsys):
    root, data_root, config, build = _setup(tmp_path)
    _acquire(root, data_root, config, build.rows, monkeypatch)
    _wire(monkeypatch, FakeIntegrationBackend())

    assert _normalize(root, data_root, ["--limit", "1", "--include-metadata-only"]) == 3
    capsys.readouterr()
    integration_manifest = next((data_root / "manifest").glob("*.integration.jsonl"))
    rows = {str(row["entry_id"]): row for row in _rows(integration_manifest)}
    statuses = {row["integration"]["status"] for row in rows.values()}
    assert statuses == {"indexed", "deferred"}
    assert rows["apd:Test/sparse.yml"]["integration"]["reason_code"] == "not_selected"


def test_normalize_cli_rejects_a_manifest_from_another_commit(tmp_path, capsys):
    root, data_root, _config, _build = _setup(tmp_path)
    manifest_dir = data_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    config = Config.load(catalog_dir=root, data_root=data_root, ledger_enabled=False)
    snapshot = catalog.load_catalog(config)
    snapshot_name = dataset_ingest._manifest_name(snapshot).replace(
        ".jsonl", ".acquisition.jsonl"
    )
    (manifest_dir / snapshot_name).write_text(
        json.dumps(
            {
                "entry_id": "apd:Test/full.yml",
                "catalog": {"commit": "f" * 40},
                "acquisition": {"status": "acquired", "files": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _normalize(root, data_root) == 1
    assert "acquisition manifest is for commit" in capsys.readouterr().err
