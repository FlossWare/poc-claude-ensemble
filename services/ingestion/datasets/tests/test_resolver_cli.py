"""End-to-end ``resolve`` CLI: accounting, sampling, resume and ledger."""
from __future__ import annotations

import json

import pytest

from ingestion import dataset_ingest
from ingestion.datasets import resolver

from ingestion.datasets.tests.helpers import FakeProbeTransport, make_catalog, memory_server


@pytest.fixture(autouse=True)
def _fast_politeness(monkeypatch):
    """The fake transport is instant; do not pay the real per-host 1 req/s."""
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_DATASET_PER_HOST_RPS", "10000")
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_DATASET_CONCURRENCY", "4")


def _fake(default=None):
    return FakeProbeTransport(
        default=default
        or {
            "status": 200,
            "headers": {"content-type": "text/csv", "content-length": "10"},
        }
    )


def _run(tmp_path, *extra, catalog=None, data=None):
    root = catalog or make_catalog(tmp_path)
    data_root = data or (tmp_path / "data")
    argv = [
        "resolve",
        "--catalog-dir",
        str(root),
        "--data-root",
        str(data_root),
        *extra,
    ]
    return dataset_ingest.main(argv), data_root


def _manifest(data_root):
    files = sorted((data_root / "manifest").glob("*.resolution.jsonl"))
    assert files
    return files[-1]


def _latest_report(data_root):
    files = sorted((data_root / "runs").glob("*.json"), key=lambda item: item.stat().st_mtime)
    assert files
    return json.loads(files[-1].read_text(encoding="utf-8"))


def test_resolve_and_report_round_trip(tmp_path, monkeypatch, capsys):
    fake = _fake()
    monkeypatch.setattr(resolver, "build_transport", lambda config: fake)
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 0
    output = capsys.readouterr().out
    assert "discovered=12" in output
    assert "invariant=ok" in output

    rows = [
        json.loads(line)
        for line in _manifest(data_root).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 12
    assert len((data_root / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()) == 12
    assert all("resolution" in row for row in rows)
    assert all(
        row["resolution"]["reason_code"]
        for row in rows
        if row["resolution"]["status"] != "resolved"
    )

    assert dataset_ingest.main(
        ["report", "--format", "json", "--data-root", str(data_root)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_type"] == "resolution"
    assert payload["invariant"]["ok"] is True
    assert payload["counts"]["discovered"] == 12

    assert dataset_ingest.main(
        ["report", "--format", "md", "--data-root", str(data_root)]
    ) == 0
    assert "# Dataset resolution report" in capsys.readouterr().out


def test_limit_exits_3_and_preserves_accounting(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(resolver, "build_transport", lambda config: _fake())
    code, data_root = _run(tmp_path, "--limit", "3", "--no-ledger")
    assert code == 3
    payload = _latest_report(data_root)
    assert payload["truncated"] is True
    assert payload["invariant"]["ok"] is True
    assert payload["counts"]["discovered"] == 12
    rows = [
        json.loads(line)
        for line in _manifest(data_root).read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        1 for row in rows if row["resolution"]["reason_code"] == "run_budget_exhausted"
    ) == 9


def test_sample_spans_disposition_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "build_transport", lambda config: _fake())
    code, data_root = _run(tmp_path, "--sample", "6", "--no-ledger")
    assert code == 3
    payload = _latest_report(data_root)
    assert payload["truncated"] is True
    dispositions = {
        entry["disposition"] for entry in payload["entries"] if isinstance(entry, dict)
    }
    assert {"blocked", "index_content", "metadata_only_review"} <= dispositions
    assert payload["invariant"]["ok"] is True


def test_re_running_resolve_is_idempotent_and_cached(tmp_path, monkeypatch):
    first = _fake()
    monkeypatch.setattr(resolver, "build_transport", lambda config: first)
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 0
    assert first.calls
    manifest = _manifest(data_root)
    first_manifest = manifest.read_bytes()
    first_outcomes = (data_root / "outcomes.jsonl").read_bytes()

    second = FakeProbeTransport(default=None)
    monkeypatch.setattr(resolver, "build_transport", lambda config: second)
    code, data_root = _run(tmp_path, "--no-ledger", data=data_root)
    assert code == 0
    assert second.calls == []
    assert manifest.read_bytes() == first_manifest
    assert (data_root / "outcomes.jsonl").read_bytes() == first_outcomes
    payload = _latest_report(data_root)
    assert payload["cache"]["hits"] == 12
    assert payload["cache"]["misses"] == 0
    assert payload["outcomes_added"] == 0


def test_no_cache_forces_the_network_again(tmp_path, monkeypatch):
    first = _fake()
    monkeypatch.setattr(resolver, "build_transport", lambda config: first)
    code, data_root = _run(tmp_path, "--no-cache", "--no-ledger")
    assert code == 0
    assert first.calls

    second = _fake()
    monkeypatch.setattr(resolver, "build_transport", lambda config: second)
    code, _data_root = _run(tmp_path, "--no-cache", "--no-ledger", data=data_root)
    assert code == 0
    assert second.calls


def test_partial_failure_exits_1_but_keeps_accounting(tmp_path, monkeypatch):
    original = resolver.resolve_entry

    def flaky(row, context):
        if str(row.get("entry_id")) == "apd:Test/full.yml":
            raise RuntimeError("synthetic resolver failure")
        return original(row, context)

    monkeypatch.setattr(resolver, "resolve_entry", flaky)
    monkeypatch.setattr(resolver, "build_transport", lambda config: _fake())
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 1
    payload = _latest_report(data_root)
    assert payload["unhandled"] == 1
    assert payload["invariant"]["ok"] is True
    failed = [
        entry
        for entry in payload["entries"]
        if entry.get("entry_id") == "apd:Test/full.yml"
    ]
    assert failed and failed[0]["status"] == "failed"
    assert failed[0]["reason_code"] == "resolver_error"


def test_ledger_event_is_append_once_and_numeric_only(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "build_transport", lambda config: _fake())
    with memory_server() as (url, handler):
        monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_MEMORY_URL", url)
        code, data_root = _run(tmp_path)
        assert code == 0
        code, data_root = _run(tmp_path, data=data_root)
        assert code == 0

        assert len(handler.posts) == 2
        first, second = handler.posts
        assert first["event_id"] == second["event_id"]
        entry = first["entry"]
        assert entry["event_name"] == "dataset.resolve.completed"
        assert entry["discovered"] == 12
        assert entry["invariant_ok"] is True
        assert isinstance(entry["status_counts"], dict)
        assert isinstance(entry["method_counts"], dict)
        serialized = json.dumps(first)
        assert "http://" not in serialized
        assert "https://" not in serialized
        assert "?" not in serialized

        payload = _latest_report(data_root)
        assert payload["ledger"]["posted"] is True
        assert payload["ledger"]["duplicate"] is True


def test_invariant_violation_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "build_transport", lambda config: _fake())
    complete = resolver.complete_resolutions

    def short(rows, resolutions, **kwargs):
        return complete(rows, resolutions, **kwargs)[:-1]

    monkeypatch.setattr(resolver, "complete_resolutions", short)
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 2
    payload = _latest_report(data_root)
    assert payload["invariant"]["ok"] is False
    assert any(
        problem.startswith("row_count:") for problem in payload["invariant"]["problems"]
    )


def test_limit_and_sample_are_mutually_exclusive(tmp_path):
    code, _data_root = _run(tmp_path, "--limit", "1", "--sample", "1")
    assert code == 1


def test_negative_sample_is_rejected(tmp_path):
    code, _data_root = _run(tmp_path, "--sample", "-1")
    assert code == 1
