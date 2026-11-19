"""End-to-end CLI behaviour, exit codes, ledger idempotency and --probe."""
from __future__ import annotations

import json

from ingestion import dataset_ingest
from ingestion.datasets import probe as probe_module

from ingestion.datasets.tests.helpers import FakeProbeTransport, make_catalog, memory_server


def _run(tmp_path, *extra, catalog=None, data=None):
    root = catalog or make_catalog(tmp_path)
    data_root = data or (tmp_path / "data")
    argv = [
        "inventory",
        "--catalog-dir",
        str(root),
        "--data-root",
        str(data_root),
        *extra,
    ]
    return dataset_ingest.main(argv), data_root


def _manifest_files(data_root):
    return sorted((data_root / "manifest").glob("*.jsonl"))


def _latest_report(data_root):
    files = sorted((data_root / "runs").glob("*.json"), key=lambda item: item.stat().st_mtime)
    assert files
    return json.loads(files[-1].read_text(encoding="utf-8"))


def test_inventory_and_report_round_trip(tmp_path, capsys):
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 0
    output = capsys.readouterr().out
    assert "discovered=12" in output
    assert "invariant=ok" in output
    manifests = _manifest_files(data_root)
    assert len(manifests) == 1
    assert len(manifests[0].read_text(encoding="utf-8").splitlines()) == 12
    assert len((data_root / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()) == 12

    assert dataset_ingest.main(["report", "--format", "json", "--data-root", str(data_root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["invariant"]["ok"] is True
    assert payload["counts"]["discovered"] == 12

    assert dataset_ingest.main(["report", "--format", "md", "--data-root", str(data_root)]) == 0
    assert "# Dataset inventory report" in capsys.readouterr().out


def test_re_running_inventory_is_idempotent(tmp_path):
    code, data_root = _run(tmp_path, "--no-ledger")
    assert code == 0
    manifest = _manifest_files(data_root)[0]
    first = manifest.read_bytes()
    outcomes_first = (data_root / "outcomes.jsonl").read_bytes()

    code, data_root = _run(tmp_path, "--no-ledger", data=data_root)
    assert code == 0
    assert manifest.read_bytes() == first
    assert (data_root / "outcomes.jsonl").read_bytes() == outcomes_first


def test_limit_exits_3_and_preserves_accounting(tmp_path):
    code, data_root = _run(tmp_path, "--limit", "2", "--no-ledger")
    assert code == 3
    payload = _latest_report(data_root)
    assert payload["truncated"] is True
    assert payload["invariant"]["ok"] is True
    assert payload["counts"]["discovered"] == 12
    assert payload["counts"]["status"]["deferred"] == 12
    lines = [
        json.loads(line)
        for line in _manifest_files(data_root)[0].read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for row in lines if row["reason_code"] == "run_budget_exhausted") == 10


def test_no_ledger_posts_nothing(tmp_path, monkeypatch):
    with memory_server() as (url, handler):
        monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_MEMORY_URL", url)
        code, _data_root = _run(tmp_path, "--no-ledger")
        assert code == 0
        assert handler.posts == []


def test_summary_ledger_event_is_append_once_and_secret_free(tmp_path, monkeypatch):
    with memory_server() as (url, handler):
        monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_MEMORY_URL", url)
        code, data_root = _run(tmp_path)
        assert code == 0
        code, data_root = _run(tmp_path, data=data_root)
        assert code == 0

        assert len(handler.posts) == 2
        first, second = handler.posts
        assert first["name"] == "session_events"
        assert first["event_id"] == second["event_id"]
        entry = first["entry"]
        assert entry["discovered"] == 12
        assert entry["status_counts"] == {"deferred": 11, "skipped": 1}
        assert entry["invariant_ok"] is True
        assert entry["exit_code"] == 0
        serialized = json.dumps(first)
        assert "http://" not in serialized
        assert "https://" not in serialized
        assert "SECRET" not in serialized
        assert "?" not in serialized

        payload = _latest_report(data_root)
        assert payload["ledger"]["posted"] is True
        assert payload["ledger"]["duplicate"] is True


def test_probe_is_bounded_cached_and_measured(tmp_path, monkeypatch):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_DATASET_PER_HOST_RPS", "1000")
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_DATASET_CONCURRENCY", "4")
    fake = FakeProbeTransport(default={"status": 200, "headers": {"content-length": "10"}})
    monkeypatch.setattr(probe_module, "build_transport", lambda config: fake)
    code, data_root = _run(tmp_path, "--probe", "--no-ledger")
    assert code == 0
    assert len(fake.calls) == 12
    assert all(call["method"] == "HEAD" for call in fake.calls)
    assert all(call["store_body"] is False for call in fake.calls)
    payload = _latest_report(data_root)
    assert payload["probe_enabled"] is True
    assert payload["probe_checked"] == 12
    assert payload["counts"]["access"]["public"] == 12
    rows = [
        json.loads(line)
        for line in _manifest_files(data_root)[0].read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["classes"]["size"]["source"] == "probe" for row in rows)
    assert all(row["probe"]["method"] == "HEAD" for row in rows)

    # A re-run on the same snapshot reuses the probe cache and performs no I/O.
    manifest_before = _manifest_files(data_root)[0].read_bytes()
    silent = FakeProbeTransport(default=AssertionError("network used"))
    monkeypatch.setattr(probe_module, "build_transport", lambda config: silent)
    code, data_root = _run(tmp_path, "--probe", "--no-ledger", data=data_root)
    assert code == 0
    assert silent.calls == []
    assert _manifest_files(data_root)[0].read_bytes() == manifest_before


def test_unhandled_per_entry_error_exits_1(tmp_path, monkeypatch):
    from ingestion.datasets import manifest as manifest_module

    original = manifest_module.build_rows

    def with_unhandled(snapshot, config, **kwargs):
        result = original(snapshot, config, **kwargs)
        result.unhandled = 1
        return result

    monkeypatch.setattr(manifest_module, "build_rows", with_unhandled)
    code, _data_root = _run(tmp_path, "--no-ledger")
    assert code == 1


def test_invariant_violation_exits_2(tmp_path, monkeypatch):
    from ingestion.datasets import manifest as manifest_module

    original = manifest_module.check_accounting

    def manufactured(rows, discovered):
        result = original(rows, discovered)
        result.ok = False
        result.problems.append("manufactured mismatch")
        return result

    monkeypatch.setattr(manifest_module, "check_accounting", manufactured)
    code, _data_root = _run(tmp_path, "--no-ledger")
    assert code == 2


def test_report_without_a_run_exits_1(tmp_path):
    assert dataset_ingest.main(["report", "--data-root", str(tmp_path / "empty")]) == 1


def test_negative_limit_exits_1(tmp_path):
    code, _data_root = _run(tmp_path, "--limit", "-1", "--no-ledger")
    assert code == 1
