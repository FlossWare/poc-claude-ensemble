"""CLI tests for ``python -m learning.meta``."""

from __future__ import annotations

import json

import pytest

from learning.meta.__main__ import main


@pytest.fixture
def env(tmp_path, monkeypatch):
    meta = tmp_path / "meta"
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_META_LEDGER_DIR", str(meta))
    return meta


def test_status_lists_all_domains(env, capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    for name in ("crew_size", "review_depth", "batch_size", "rag_scope",
                 "prompt_strategy", "chunking", "tool_selection", "graph_traversal"):
        assert name in out
    assert "* = instrumentation domain" in out


def test_default_command_is_status(env, capsys):
    assert main([]) == 0
    assert "meta-learning domains" in capsys.readouterr().out


def test_record_then_posterior(env, capsys):
    assert main(["record", "crew_size", "--arm", "4", "--outcome", "0.75",
                 "--cost", "0.25", "--task-type", "code-review",
                 "--outcome-id", "cli-1", "--json"]) == 0
    json.loads(capsys.readouterr().out)  # record prints the row as JSON
    assert main(["posterior", "crew_size", "--task-type", "code-review", "--json"]) == 0
    posterior = json.loads(capsys.readouterr().out)
    assert posterior["arms"]["4"]["successes"] == pytest.approx(0.75)
    assert posterior["arms"]["4"]["calls"] == 1


def test_record_rag_scope_accepts_a_config_mapping(env, capsys):
    arm = json.dumps({"memory.max_results": 10, "knowledge.max_queries": 2,
                      "graph.enabled": True})
    assert main(["record", "rag_scope", "--arm", arm, "--outcome", "1", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["arm"] == "m10-k2-g1"
    assert row["recorded"] is True


def test_select_prints_a_choice_in_range(env, capsys):
    assert main(["select", "crew_size", "--task-type", "code-review", "--json"]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert 1 <= detail["choice"] <= 5
    assert detail["arm"] in {"1", "2", "3", "4", "5"}
    assert detail["source"] in ("thompson", "posterior_fallback", "default_fallback")


def test_select_rag_scope_returns_knobs(env, capsys):
    assert main(["select", "rag_scope", "--json"]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert set(detail["choice"]) == {"memory.max_results", "knowledge.max_queries",
                                     "graph.enabled"}


def test_backfill_crew_size_is_idempotent(env, tmp_path, monkeypatch, capsys):
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in [
        {"review_id": "r1", "subject_sha256": "s", "kind": "worker_result", "model": "m1"},
        {"review_id": "r1", "subject_sha256": "s", "kind": "worker_result", "model": "m2"},
        {"review_id": "r1", "subject_sha256": "s", "kind": "arbiter_decision",
         "model": "arb", "text": "VERDICT: approve"},
    ]) + "\n", encoding="utf-8")
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", str(ledger))
    assert main(["backfill", "crew_size", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)[0]
    assert (first["recorded"], first["duplicates"]) == (1, 0)
    assert main(["backfill", "crew_size", "--json"]) == 0
    second = json.loads(capsys.readouterr().out)[0]
    assert (second["recorded"], second["duplicates"]) == (0, 1)


def test_readiness_reports_missing_journal(env, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLOSSWARE_CE_RAG_JOURNAL", str(tmp_path / "missing.jsonl"))
    assert main(["readiness", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["observations"] == 0
    assert report["ready"] is False
