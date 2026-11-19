"""Security controls (design section 3, test plan 15-17)."""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx
from learning.ga import cli, corpus as corpus_mod, journal
from shared import thompson_policy as policy

GA_PACKAGE = Path(__file__).resolve().parent
FORBIDDEN_IMPORT_ROOTS = {
    "urllib",
    "requests",
    "http",
    "socket",
    "ssl",
    "ftplib",
    "thompson_client",
    "pickle",
    "marshal",
    "eval",
    "exec",
}


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def test_symlink_outside_the_allowlist_is_a_config_error(isolated):
    link = isolated / "outcomes" / "outcomes.jsonl"
    link.symlink_to("/etc/hostname")
    with pytest.raises(corpus_mod.CorpusConfigError):
        corpus_mod.ingest_turn_corpus()
    result = cli.execute_run("thompson_policy", seed=1)
    assert result.outcome == "config_error"
    assert result.exit_code == cli.EXIT_CONFIG


def test_non_jsonl_and_symlinked_policy_are_refused(isolated):
    with pytest.raises(corpus_mod.CorpusConfigError):
        corpus_mod.ingest_turn_corpus(isolated / "outcomes" / "outcomes.jsonl.gz")
    policy_path = Path(policy.policy_file_path())
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.symlink_to("/etc/hostname")
    policy.clear_cache()
    assert policy.load_policy_file()["default"] == policy.default_params()


def test_oversize_line_and_non_finite_cost_are_counted(isolated):
    outcomes = isolated / "outcomes" / "outcomes.jsonl"
    huge = json.dumps(
        {
            "kind": "turn_outcome",
            "status": "pending",
            "arm": "haiku",
            "success": True,
            "cost": 1.0,
            "timestamp": fx.iso(fx.EPOCH),
            "text": "x" * (corpus_mod.MAX_LINE_BYTES + 64),
        }
    )
    nan = json.dumps({**fx.outcome_row("nan", "haiku", True, 1.0, fx.EPOCH), "cost": float("nan")})
    fx.write_jsonl(outcomes, [huge, nan, fx.outcome_row("ok", "haiku", True, 0.1, fx.EPOCH)])
    corpus = corpus_mod.ingest_turn_corpus()
    assert corpus.counts["rejected_oversize_line"] == 1
    assert corpus.counts["rejected_non_finite"] == 1
    assert corpus.counts["accepted"] == 1


def test_no_network_or_dynamic_execution_imports_anywhere_in_the_ga():
    offenders = []
    for path in GA_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            roots = set()
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".")[0]}
            if roots & FORBIDDEN_IMPORT_ROOTS:
                offenders.append((path.name, roots & FORBIDDEN_IMPORT_ROOTS))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile"}:
                    offenders.append((path.name, node.func.id))
    assert not offenders


def test_ga_outputs_never_contain_ledger_text(isolated, tmp_path):
    marker = "UNIQUE-REVIEW-BODY-MARKER-4f19"
    rows = []
    for index in range(3):
        row = fx.review_row(f"r{index}", "alpha", True, 0.1, fx.EPOCH + index)
        row["text"] = f"{marker} {index}"
        rows.append(row)
        rows.append({**fx.review_row(f"r{index}", "beta", True, 0.1, fx.EPOCH + index), "text": marker})
    fx.write_jsonl(isolated / "reviews" / "reviews.jsonl", rows)
    fx.write_jsonl(
        isolated / "outcomes" / "outcomes.jsonl",
        [{**fx.outcome_row(f"e{i}", "alpha", True, 0.1, fx.EPOCH + i), "text": marker} for i in range(10)],
    )

    result = cli.execute_run("thompson_policy", seed=2)
    assert result.exit_code == 0

    for path in Path(journal.ga_root()).rglob("*"):
        if path.is_file():
            assert marker not in path.read_text(errors="ignore"), path
    if journal.journal_path().exists():
        for entry in journal.read_entries():
            assert "text" not in entry


def test_report_only_run_never_writes_live_state(isolated, monkeypatch):
    """Test 15: ledgers, selection ledger, and Thompson state are untouched."""
    live = {
        "outcomes": isolated / "outcomes" / "outcomes.jsonl",
        "reviews": isolated / "reviews" / "reviews.jsonl",
        "selections": isolated / "selections" / "selections.jsonl",
        "state": Path(os.environ["POC_CLAUDE_ENSEMBLE_THOMPSON_STATE_FILE"]),
    }
    fx.write_jsonl(live["outcomes"], fx.passing_turn_rows(days=10, per_day=6))
    fx.write_jsonl(live["reviews"], fx.passing_review_rows(rounds=5))
    fx.write_jsonl(live["selections"], [{"kind": "selection_shadow"}])
    live["state"].parent.mkdir(parents=True, exist_ok=True)
    live["state"].write_text(json.dumps({"models": {"haiku": {"successes": 1}}}))
    before = fx.snapshot_files(live.values())

    result = cli.execute_run("thompson_policy", seed=4)
    assert result.exit_code == 0

    after = fx.snapshot_files(live.values())
    assert before == after
    assert not Path(policy.policy_file_path()).exists()  # report-only writes no policy


def test_journal_allowlist_and_error_redaction(isolated):
    with pytest.raises(journal.JournalError):
        journal.append({"kind": "apply", "text": "secret"})
    with pytest.raises(journal.JournalError):
        journal.append({"kind": "apply", "query": "secret"})
    journal.append(journal.error_entry("apply", RuntimeError("file content: secret")))
    entry = journal.read_entries()[-1]
    assert "error_message" in entry and "traceback" not in entry


def test_gene_bounds_are_the_documented_ones():
    bounds = {name: (spec.low, spec.high, spec.default) for name, spec in policy.PARAM_BOUNDS.items()}
    assert bounds == {
        "prior_strength": (0.1, 5.0, 1.0),
        "cost_weight": (0.0, 1.0, 0.0),
        "exploration_epsilon": (0.0, 0.2, 0.0),
        "capability_margin": (-1.0, 1.0, -1.0),
    }
    assert math.isclose(policy.PARAM_BOUNDS["cost_weight"].high, 1.0)
