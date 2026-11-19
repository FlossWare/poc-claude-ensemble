"""Funnel, counts and Markdown/JSON rendering."""
from __future__ import annotations

import json

from ingestion.datasets import catalog, manifest, report

from ingestion.datasets.tests.helpers import make_catalog, make_config


def _report(tmp_path):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    build = manifest.build_rows(snapshot, config)
    accounting = manifest.check_accounting(build.rows, build.discovered)
    payload = report.build_report(
        build.rows,
        run_id="run123",
        snapshot=snapshot,
        accounting=accounting,
        generated_at="2026-01-01T00:00:00Z",
        duration_ms=5,
        exit_code=0,
        manifest_path="/tmp/manifest.jsonl",
        manifest_changed=True,
        outcomes_added=12,
        ledger_result={"enabled": False, "posted": False},
        probe_checked=0,
        probe_enabled=False,
        limit=None,
        unhandled=0,
        truncated=False,
    )
    return payload


def test_report_counts_and_funnel(tmp_path):
    payload = _report(tmp_path)
    assert payload["run_id"] == "run123"
    assert payload["invariant"]["ok"] is True
    assert payload["funnel"] == {
        "discovered": 12,
        "resolved": 0,
        "acquired": 0,
        "parsed": 0,
        "indexed": 0,
        "unchanged": 0,
        "skipped": 1,
        "failed": 0,
        "deferred": 11,
    }
    assert payload["counts"]["status"] == {"deferred": 11, "skipped": 1}
    assert payload["counts"]["disposition"]["blocked"] == 1
    assert payload["counts"]["license_family"]["unknown"] == 1
    assert payload["counts"]["license_family"]["cc-by"] == 3
    assert payload["counts"]["format"]["csv"] >= 1
    assert payload["counts"]["scheme_unsupported"] == {"ftp": 1, "hhttps": 1}
    assert payload["readme"]["meta_links"] == 12
    assert payload["readme"]["entries_missing_status"] == 0
    assert len(payload["entries"]) == 12


def test_report_renders_json_and_markdown(tmp_path):
    payload = _report(tmp_path)
    as_json = report.render_json(payload)
    assert json.loads(as_json)["run_id"] == "run123"
    text = report.render_markdown(payload)
    assert "# Dataset inventory report" in text
    assert "## Accounting invariant" in text
    assert "## Funnel" in text
    assert "apd:Test/full.yml" in text
    assert "| discovered | 12 |" in text


def test_probe_counts_are_empty_without_probe(tmp_path):
    payload = _report(tmp_path)
    assert payload["counts"]["probe_access"] == {}
