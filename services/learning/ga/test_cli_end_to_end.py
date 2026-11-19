"""End-to-end CLI behaviour: refusal, promotion, idempotency, rollback, report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx
from learning.ga import cli, journal
from shared import thompson_policy as policy
from shared.thompson_policy import default_params


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def _write_prior_policy() -> bytes:
    path = Path(policy.policy_file_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(policy.policy_document({"default": default_params()}), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    policy.clear_cache()
    return encoded


def _run_dir(result) -> Path:
    return journal.run_dir(result.run_id)


def test_real_world_sized_corpus_refuses_and_writes_nothing_live(isolated):
    fx.write_tiny_corpora(isolated)
    result = cli.execute_run("thompson_policy", seed=1)
    assert result.exit_code == 0
    assert result.outcome == "not_promoted:insufficient_evidence"
    assert not Path(policy.policy_file_path()).exists()
    assert (_run_dir(result) / "results.json").exists()
    assert (_run_dir(result) / "manifest.json").exists()
    assert not journal.journal_path().exists()  # report-only refusals journal nothing

    results = json.loads((_run_dir(result) / "results.json").read_text())
    assert results["outcome"] == "not_promoted:insufficient_evidence"
    assert results["candidate"] is None
    review_gate = {gate["name"]: gate for gate in results["gates"]}["review_rounds"]
    assert review_gate["passed"] is False


def test_results_are_byte_identical_for_the_same_corpus_and_seed(isolated, tmp_path, monkeypatch):
    fx.write_passing_corpora(isolated)
    first = cli.execute_run("thompson_policy", seed=7, max_evals=24)
    first_bytes = (_run_dir(first) / "results.json").read_bytes()

    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_GA_DIR", str(tmp_path / "ga-second"))
    second = cli.execute_run("thompson_policy", seed=7, max_evals=24)
    second_bytes = (_run_dir(second) / "results.json").read_bytes()
    assert first.outcome == second.outcome
    assert first_bytes == second_bytes


def test_promotion_apply_idempotency_and_rollback(isolated, monkeypatch):
    fx.write_passing_corpora(isolated)
    prior = _write_prior_policy()
    monkeypatch.setenv(cli.AUTO_APPLY_ENV, "1")

    result = cli.execute_run("thompson_policy", seed=7, apply=True)
    assert result.exit_code == 0, result.message
    assert result.outcome == "promoted", [g for g in result.results["gates"] if not g["passed"]]
    assert result.results["delta"] >= 0.01
    assert result.results["ci95"][0] > 0

    policy_path = Path(policy.policy_file_path())
    written = json.loads(policy_path.read_text())
    assert written["schema_version"] == 1
    assert written["policies"]["default"]["cost_weight"] == pytest.approx(
        result.results["candidate"]["params"]["cost_weight"]
    )
    assert written["provenance"]["change_id"]
    assert oct(policy_path.stat().st_mode & 0o777) == "0o600"

    entries = journal.read_entries()
    applies = [entry for entry in entries if entry["kind"] == "apply"]
    assert len(applies) == 1
    entry = applies[0]
    assert entry["target"] == "thompson_policy"
    assert entry["old_params"] == default_params()  # the live policy before promotion
    assert entry["new_params"] == result.results["candidate"]["params"]
    assert entry["previous_file_existed"] is True
    assert entry["ci95"][0] > 0

    # Same corpus + code on the same day: nothing new to evaluate.
    again = cli.execute_run("thompson_policy", seed=7, apply=True)
    assert again.outcome == "no_new_evidence"
    assert again.exit_code == 0
    assert len([e for e in journal.read_entries() if e["kind"] == "apply"]) == 1

    # Rollback restores the prior file byte for byte.
    rolled = cli.execute_rollback(None, last=True, applied_by="tester")
    assert rolled.exit_code == 0
    assert policy_path.read_bytes() == prior
    assert [entry["kind"] for entry in journal.read_entries()].count("rollback") == 1

    second_rollback = cli.execute_rollback(None, last=True, applied_by="tester")
    assert second_rollback.exit_code == 0
    assert "already rolled back" in second_rollback.message


def test_corpus_changed_aborts_the_run(isolated, monkeypatch):
    """Test 13: a ledger that grows mid-run aborts before anything is applied."""
    fx.write_passing_corpora(isolated)
    outcomes = isolated / "outcomes" / "outcomes.jsonl"
    real_search = cli.search_turn

    def tampering_search(*args, **kwargs):
        outcome = real_search(*args, **kwargs)
        fx.write_jsonl(
            outcomes,
            [
                *[json.loads(line) for line in outcomes.read_text().splitlines() if line],
                fx.outcome_row("late-row", "cheap", True, 0.001, fx.EPOCH + 999 * fx.DAY),
            ],
        )
        return outcome

    monkeypatch.setattr(cli, "search_turn", tampering_search)
    monkeypatch.setenv(cli.AUTO_APPLY_ENV, "1")
    result = cli.execute_run("thompson_policy", seed=7, max_evals=24, apply=True)
    assert result.outcome == "corpus_changed"
    assert result.exit_code == cli.EXIT_INFRA
    assert not Path(policy.policy_file_path()).exists()
    assert not any(entry["kind"] == "apply" for entry in journal.read_entries())


def test_budget_exhaustion_is_an_infrastructure_error(isolated, monkeypatch):
    from learning.ga.engine import BudgetExceeded

    fx.write_passing_corpora(isolated)

    def exhausted(*_args, **_kwargs):
        raise BudgetExceeded("wall clock")

    monkeypatch.setattr(cli, "search_turn", exhausted)
    result = cli.execute_run("thompson_policy", seed=7)
    assert result.outcome == "error"
    assert result.exit_code == cli.EXIT_INFRA
    assert "wall clock" in result.message


def test_apply_requires_the_auto_apply_env(isolated, monkeypatch):
    fx.write_passing_corpora(isolated)
    monkeypatch.delenv(cli.AUTO_APPLY_ENV, raising=False)
    result = cli.execute_run("thompson_policy", seed=7, apply=True)
    assert result.exit_code == cli.EXIT_CONFIG
    assert result.outcome == "config_error"
    assert not Path(policy.policy_file_path()).exists()


def test_non_promotable_targets_are_config_errors(isolated):
    for name in ("rag_config", "review_pipeline"):
        result = cli.execute_run(name)
        assert result.exit_code == cli.EXIT_CONFIG
        assert result.outcome == "config_error"
        assert "not promotable" in result.message


def test_already_running_is_a_no_op(isolated):
    fx.write_tiny_corpora(isolated)
    lock_path = journal.locks_dir() / "ga.lock"
    journal.ensure_private_dir(journal.locks_dir())
    import fcntl

    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = cli.execute_run("thompson_policy", seed=1)
        assert result.outcome == "already_running"
        assert result.exit_code == 0
    finally:
        os.close(handle)


def test_report_and_cli_exit_codes(isolated, capsys):
    fx.write_tiny_corpora(isolated)
    assert cli.main(["run", "--target", "thompson_policy", "--seed", "3"]) == 0
    captured = capsys.readouterr().out
    assert "outcome: not_promoted:insufficient_evidence" in captured

    assert cli.main(["report"]) == 0
    assert "outcome: not_promoted:insufficient_evidence" in capsys.readouterr().out

    assert cli.main(["rollback", "--last"]) == 0
    assert "no apply entry" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--target", "not-a-target"])
    assert excinfo.value.code == 2


def test_module_entry_point_refuses_on_tiny_corpus(isolated):
    fx.write_tiny_corpora(isolated)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-m", "learning.ga", "run", "--target", "thompson_policy"],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "outcome: not_promoted:insufficient_evidence" in completed.stdout
