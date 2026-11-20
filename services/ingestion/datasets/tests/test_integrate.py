"""PR 4 entry-level accounting: terminal statuses, outcomes and invariants."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import indexing, integrate, normalize


def _outcome(kind: str, status: str, reason: str | None = None, **overrides) -> indexing.RecordOutcome:
    values = dict(
        source_id="apd:Test/a.yml" + ("" if kind == "metadata" else "/src/x/shard/0000"),
        entry_id="apd:Test/a.yml",
        kind=kind,
        content_sha256="1" * 64,
        revision_id="2" * 64,
        status=status,
        reason_code=reason,
    )
    values.update(overrides)
    return indexing.RecordOutcome(**values)


def _normalization(content_reason: str | None = None) -> normalize.EntryNormalization:
    return normalize.EntryNormalization(
        entry_id="apd:Test/a.yml",
        metadata=None,
        metadata_reason=None,
        content_reason=content_reason,
    )


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ([_outcome("metadata", "indexed"), _outcome("content", "indexed")], "indexed"),
        ([_outcome("metadata", "indexed")], "metadata_indexed"),
        ([_outcome("metadata", "unchanged")], "unchanged"),
        ([_outcome("metadata", "indexed"), _outcome("content", "failed", "parse_error")], "failed"),
        (
            [_outcome("metadata", "indexed"), _outcome("content", "failed", "dependency_unavailable")],
            "deferred",
        ),
        ([_outcome("metadata", "skipped", "source_tombstoned")], "skipped"),
    ],
)
def test_entry_status_collapses_record_outcomes(outcomes, expected):
    status, reason, detail, retryable = integrate._entry_status(_normalization(), outcomes)
    assert status == expected
    if expected not in {"indexed", "metadata_indexed", "unchanged"}:
        assert reason
    if expected == "deferred":
        assert retryable is True


def test_content_failure_is_never_reported_as_metadata_success():
    status, reason, _, _ = integrate._entry_status(
        _normalization("adapter_unavailable"),
        [_outcome("metadata", "indexed"), _outcome("content", "failed", "adapter_unavailable")],
    )
    assert status == "deferred"
    assert reason == "adapter_unavailable"


def test_accounting_requires_a_reason_for_every_non_success_status():
    row = {"entry_id": "apd:Test/a.yml"}
    integrations = [
        integrate.EntryIntegration(
            entry_id="apd:Test/a.yml", status="skipped", reason_code=None
        )
    ]
    accounting = integrate.check_accounting([row], integrations, 1)
    assert accounting.ok is False
    assert accounting.missing_reason == ["apd:Test/a.yml"]
    assert any(problem.startswith("missing_reason") for problem in accounting.problems)


def test_accounting_rejects_a_status_count_that_does_not_match_discovery():
    rows = [{"entry_id": "apd:Test/a.yml"}, {"entry_id": "apd:Test/b.yml"}]
    integrations = [
        integrate.EntryIntegration(entry_id="apd:Test/a.yml", status="indexed"),
    ]
    accounting = integrate.check_accounting(rows, integrations, 2)
    assert accounting.ok is False
    assert accounting.status_total == 1
    assert any("missing_integration" in problem for problem in accounting.problems)


def test_outcome_rows_are_deterministic_and_cover_every_record():
    row = {
        "entry_id": "apd:Test/a.yml",
        "homepage": "https://example.org/x?token=SECRET",
        "metadata_sha256": "3" * 64,
        "catalog": {"commit": "c" * 40, "meta_url": "https://github.com/x/y"},
    }
    integration = integrate.EntryIntegration(
        entry_id="apd:Test/a.yml",
        status="indexed",
        reason_code=None,
        metadata={"kind": "metadata", "source_id": "apd:Test/a.yml", "status": "indexed",
                  "revision_id": "4" * 64, "chunks": 1, "vectors": 1},
        record_outcomes=(
            {"kind": "content", "source_id": "apd:Test/a.yml/src/x/shard/0000",
             "status": "failed", "reason_code": "dependency_unavailable",
             "revision_id": "5" * 64, "chunks": 0, "vectors": 0},
        ),
    )
    first = integrate.integration_outcome_rows(
        [row], [integration], run_id="r", catalog_commit="c" * 40, retrieved_at="t"
    )
    second = integrate.integration_outcome_rows(
        [row], [integration], run_id="r", catalog_commit="c" * 40, retrieved_at="t"
    )
    assert [item["event_id"] for item in first] == [item["event_id"] for item in second]
    assert len(first) == 2  # metadata plus one content shard
    content = next(item for item in first if item["record_kind"] == "content")
    assert content["reason_code"] == "dependency_unavailable"
    assert content["retryable"] is True
    assert content["knowledge_source_id"].endswith("/shard/0000")
    # Provenance URLs never carry query strings.
    assert "SECRET" not in json.dumps(first)


def test_record_less_entries_still_produce_an_outcome_row():
    row = {"entry_id": "apd:Test/a.yml", "catalog": {"commit": "c" * 40}}
    integration = integrate.EntryIntegration(
        entry_id="apd:Test/a.yml",
        status="deferred",
        reason_code="not_selected",
        retryable=True,
    )
    outcomes = integrate.integration_outcome_rows(
        [row], [integration], run_id="r", catalog_commit="c" * 40, retrieved_at="t"
    )
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "deferred"
    assert outcomes[0]["reason_code"] == "not_selected"


def test_load_acquisitions_ignores_malformed_lines(tmp_path):
    path = tmp_path / "acq.jsonl"
    path.write_text(
        "\n".join(
            [
                "not json",
                json.dumps({"entry_id": "apd:Test/a.yml",
                            "catalog": {"commit": "c" * 40},
                            "acquisition": {"status": "acquired", "files": []}}),
                json.dumps({"entry_id": 5, "acquisition": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    acquisitions, commit = integrate.load_acquisitions(path)
    assert list(acquisitions) == ["apd:Test/a.yml"]
    assert commit == "c" * 40


def test_selection_defaults_to_acquired_entries_and_can_include_metadata_only():
    rows = [
        {"entry_id": "apd:Test/a.yml", "license": {"dataset": {"policy": "index_content"}}},
        {"entry_id": "apd:Test/b.yml", "license": {"dataset": {"policy": "metadata_only_review"}}},
        {"entry_id": "apd:Test/c.yml", "license": {"dataset": {"policy": "blocked"}}},
    ]
    acquisitions = {
        "apd:Test/a.yml": {"status": "acquired"},
        "apd:Test/b.yml": {"status": "skipped", "reason_code": "license_unknown_metadata_only"},
        "apd:Test/c.yml": {"status": "skipped", "reason_code": "license_restricted"},
    }
    selected, unknown, truncated = integrate.select_entries(rows, acquisitions)
    assert selected == {"apd:Test/a.yml"}
    assert unknown == [] and truncated is False

    selected, _, _ = integrate.select_entries(
        rows, acquisitions, include_metadata_only=True
    )
    assert selected == {"apd:Test/a.yml", "apd:Test/b.yml"}

    selected, unknown, _ = integrate.select_entries(
        rows, acquisitions, entries=["apd:Test/a.yml", "apd:Test/zzz.yml"]
    )
    assert selected == {"apd:Test/a.yml"}
    assert unknown == ["apd:Test/zzz.yml"]
