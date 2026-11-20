#!/usr/bin/env python3
"""Tests for tools/learn.py and its integration points.

Every test redirects external caches, the prior registry, the Thompson state,
the review ledger, and the crew ledger into a private temp root before the
modules under test read them.  No test touches the live Thompson service or the
operator's ~/.FlossWare state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
SERVICES = TOOLS.parent / "services"
for candidate in (str(TOOLS), str(SERVICES), str(SERVICES / "shared")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="learn-tests-"))
os.environ["POC_CLAUDE_ENSEMBLE_EXTERNAL_DIR"] = str(_TEST_ROOT / "external")
os.environ["POC_CLAUDE_ENSEMBLE_THOMPSON_STATE_FILE"] = str(_TEST_ROOT / "thompson-state.json")
os.environ["POC_CLAUDE_ENSEMBLE_META_LEDGER_DIR"] = str(_TEST_ROOT / "meta")
os.environ["POC_CLAUDE_ENSEMBLE_GA_DIR"] = str(_TEST_ROOT / "ga")
os.environ["POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"] = str(_TEST_ROOT / "reviews.jsonl")
os.environ["POC_CLAUDE_ENSEMBLE_TRAIN_DIR"] = str(_TEST_ROOT / "train")
os.environ["POC_CLAUDE_ENSEMBLE_PREDICTOR_DIR"] = str(_TEST_ROOT / "predictors")
os.environ["POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR"] = str(_TEST_ROOT / "crews")

import learn  # noqa: E402
import crew_select  # noqa: E402
from learning.external import registry  # noqa: E402
from learning.ga.cli import RunResult  # noqa: E402

FIXTURE_STATE = {
    "last_updated": "2026-11-19T00:00:00Z",
    "models": {
        "glm-5-3": {"successes": 51, "failures": 13, "calls": 64,
                    "total_cost": 70.1, "total_tokens": 1},
        "ollama/qwen2.5-coder:3b": {"successes": 0.0, "failures": 4.0, "calls": 0,
                                    "total_cost": 0.0, "total_tokens": 0,
                                    "prior": {"source": "lmsys_arena_elo", "alpha": 1.0,
                                              "beta": 5.0, "match_type": "local_surrogate"}},
    },
}


class FakeClient:
    def __init__(self, *, ping=True, draws=(), seed=True):
        self._ping = ping
        self.draws = list(draws)
        self.seed = seed
        self.seeded = []

    def ping(self):
        return self._ping

    def get_state(self):
        return FIXTURE_STATE

    def select_model(self, *args, **kwargs):
        return self.draws.pop(0) if self.draws else "glm-5-3"

    def seed_prior(self, arm, alpha, beta, **kwargs):
        self.seeded.append((arm, alpha, beta))
        return self.seed


# ---------------------------------------------------------------------------
# status / models
# ---------------------------------------------------------------------------


def test_collect_status_and_render(monkeypatch):
    monkeypatch.setattr(learn, "thompson_state", lambda timeout=2.0: {
        "origin": "state_file", "last_updated": "2026-11-19T00:00:00Z",
        "models": FIXTURE_STATE["models"], "state_path": "/tmp/state.json"})
    monkeypatch.setattr(learn, "trained_models_status", lambda: {
        "artifact_dir": "/tmp/train", "models": {
            "quality_predictor": {"present": False, "artifact": "/tmp/q.joblib"}}})
    monkeypatch.setattr(learn, "routing_predictors_status", lambda: {
        "cost": {"kind": "cost", "available": False, "reason": "artifact missing"}})
    monkeypatch.setattr(learn, "meta_status", lambda: {
        "domains": [{"domain": "crew_size", "arms": ["1", "2", "3"], "observations": 7,
                     "top_arm": "3", "top_posterior_mean": 0.8, "outcome_source": "ledger"}]})
    monkeypatch.setattr(learn, "ga_status", lambda: {
        "journal_path": "/tmp/journal.jsonl", "entries": 0, "last_run": None,
        "last_results": None, "last_promotion": None, "last_rollback": None,
        "auto_reverts": 0, "policy_file": "/tmp/policy.json", "policy_file_present": False})
    monkeypatch.setattr(learn, "ollama_models", lambda runner=None: {
        "available": True, "reason": "", "models": ["qwen2.5-coder:3b"]})
    monkeypatch.setattr(learn, "external_priors_status", lambda state=None: [{
        "arm": "ollama/qwen2.5-coder:3b", "alpha": 5.5, "beta": 1.5, "source": "lmsys_arena_elo",
        "match_type": "local_surrogate", "in_state": True, "applied_via": "service", "calls": 0}])

    report = learn.collect_status()
    text = learn.render_status(report)
    assert "Thompson arms" in text
    assert "glm-5-3" in text
    assert "quality_predictor: absent" in text
    assert "crew_size" in text
    assert "auto-reverts: 0" in text
    assert "qwen2.5-coder:3b" in text
    assert "lmsys_arena_elo" in text


def test_status_cli_json(monkeypatch, capsys):
    monkeypatch.setattr(learn, "collect_status", lambda: {"ok": True, "thompson": {"origin": "x"}})
    code = learn.main(["status", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_models_include_unregistered_ollama(monkeypatch):
    monkeypatch.setattr(learn, "thompson_state", lambda timeout=2.0: {
        "origin": "state_file", "models": {"glm-5-3": FIXTURE_STATE["models"]["glm-5-3"]}})
    monkeypatch.setattr(learn, "ollama_models", lambda runner=None: {
        "available": True, "reason": "", "models": ["qwen2.5-coder:3b"]})
    rows = learn.collect_models()
    arms = {row["arm"]: row for row in rows}
    assert "glm-5-3" in arms
    assert arms["ollama/qwen2.5-coder:3b"]["unregistered"] is True
    text = learn.render_models(rows)
    assert "unregistered" in text


# ---------------------------------------------------------------------------
# add-model
# ---------------------------------------------------------------------------


def _isolate_prior_paths(monkeypatch, tmp_path):
    priors = tmp_path / "priors.json"
    state = tmp_path / "state.json"
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_EXTERNAL_DIR", str(tmp_path / "external"))
    original_list = registry.list_priors
    monkeypatch.setattr(registry, "registry_path", lambda path=None: priors)
    monkeypatch.setattr(registry, "list_priors", lambda path=None: original_list(path=priors))
    monkeypatch.setattr(registry, "thompson_state_path", lambda: state)
    return priors, state


def test_add_model_uses_elo_prior_and_state_file_fallback(tmp_path, monkeypatch):
    priors, state = _isolate_prior_paths(monkeypatch, tmp_path)
    from learning.external import elo_priors

    monkeypatch.setattr(elo_priors, "prior_for_model", lambda model, **kwargs: elo_priors.EloPrior(
        model=model, matched_model="qwen2.5-coder-32b-instruct", rating=1235.0,
        alpha=5.5, beta=1.5, base_alpha=10.0, base_beta=2.0,
        match_type="local_surrogate", confidence=0.5,
        snapshot="elo_results_20250829.pkl", note="surrogate"))

    result = learn.add_model(
        "ollama/qwen2.5-coder:3b", use_elo=True,
        client=FakeClient(ping=False), apply_to_service=True,
    )
    assert result["alpha"] == pytest.approx(5.5)
    assert result["beta"] == pytest.approx(1.5)
    assert result["applied_via"] == "state_file"
    assert priors.exists() and state.exists()
    stats = json.loads(state.read_text())["models"]["ollama/qwen2.5-coder:3b"]
    assert stats["successes"] + 1.0 == pytest.approx(5.5)
    assert stats["calls"] == 0
    stored = registry.get_prior("ollama/qwen2.5-coder:3b", path=priors)
    assert stored is not None and stored.applied_via == "state_file"


def test_add_model_prefers_live_service(tmp_path, monkeypatch):
    priors, state = _isolate_prior_paths(monkeypatch, tmp_path)
    from learning.external import elo_priors

    monkeypatch.setattr(elo_priors, "prior_for_model", lambda model, **kwargs: elo_priors.EloPrior(
        model=model, matched_model="glm-4.5", rating=1429.0,
        alpha=13.2, beta=1.3, base_alpha=13.2, base_beta=1.3,
        match_type="alias", confidence=1.0, snapshot="elo_results_20250829.pkl"))
    client = FakeClient(ping=True, seed=True)
    result = learn.add_model("glm-5-3", use_elo=True, client=client)
    assert result["applied_via"] == "service"
    assert client.seeded and client.seeded[0][0] == "glm-5-3"
    assert registry.get_prior("glm-5-3", path=priors).applied_via == "service"


# ---------------------------------------------------------------------------
# test-model replay
# ---------------------------------------------------------------------------


def _write_ledger(tmp_path):
    subjects = []
    rows = []
    for index, name in enumerate(("alpha.py", "beta.py", "missing.py")):
        path = tmp_path / name
        if name != "missing.py":
            path.write_text("def f():\n    return 1\n", encoding="utf-8")
        review_id = f"review-{index}"
        subjects.append(path)
        rows.append({"kind": "worker_result", "review_id": review_id,
                     "subject": str(path), "model": "rits/zai-org/glm-5-3"})
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return ledger


def test_test_model_with_mock_gateway(tmp_path):
    ledger = _write_ledger(tmp_path)
    calls = []

    def caller(model, prompt):
        calls.append((model, prompt))
        return {"text": "a finding", "latency_ms": 12, "stop_reason": "end_turn",
                "input_tokens": 10, "output_tokens": 5}

    result = learn.test_model("ollama/qwen2.5-coder:3b", reviews=10,
                              ledger=ledger, caller=caller)
    assert result["reviews_found"] == 3
    assert result["reviews_attempted"] == 2  # missing.py is not called
    assert result["usable"] == 2
    assert len(calls) == 2
    assert "usable_output_proxy" in result["label_source"]


def test_test_model_records_gateway_errors(tmp_path):
    ledger = _write_ledger(tmp_path)

    def caller(model, prompt):
        raise RuntimeError("gateway down")

    result = learn.test_model("m", reviews=3, ledger=ledger, caller=caller)
    assert result["usable"] == 0
    assert result["reviews_attempted"] == 2
    errors = [row for row in result["results"] if row["status"] == "error"]
    assert len(errors) == 2
    assert any(row["status"] == "missing_subject" for row in result["results"])


def test_test_model_dry_run_never_calls_gateway(tmp_path, monkeypatch):
    ledger = _write_ledger(tmp_path)

    def explode(model, prompt):
        raise AssertionError("dry run must not call the gateway")

    monkeypatch.setattr(learn, "_artifact_quality_prior", lambda model, artifact=None: 0.66)
    result = learn.test_model("m", reviews=3, ledger=ledger, caller=explode, dry_run=True)
    assert result["dry_run"] is True
    assert result["reviews_attempted"] == 0
    estimates = [row.get("quality_estimate") for row in result["results"]
                 if row["status"] == "dry_run"]
    assert estimates and all(value == 0.66 for value in estimates)


# ---------------------------------------------------------------------------
# crew / tune / journal / ollama
# ---------------------------------------------------------------------------


def test_crew_draws_ollama_arms():
    client = FakeClient(draws=["ollama/qwen2.5-coder:3b", "glm-5-3"])
    record = learn.crew("code-review", 2, client=client, record_selection=False)
    assert record["status"] == "thompson"
    assert [worker["arm"] for worker in record["workers"]] == [
        "ollama/qwen2.5-coder:3b", "glm-5-3"
    ]
    assert record["models"][0] == "ollama/qwen2.5-coder:3b"


def test_crew_can_use_external_priors(monkeypatch, tmp_path):
    priors, state = _isolate_prior_paths(monkeypatch, tmp_path)
    registry.register_prior(registry.PriorRecord(
        arm="ollama/qwen2.5-coder:3b", model_id="ollama/qwen2.5-coder:3b",
        alpha=5.5, beta=1.5, source="lmsys_arena_elo"))
    client = FakeClient(draws=["ollama/qwen2.5-coder:3b"])
    client.get_state = lambda: {"models": {"glm-5-3": FIXTURE_STATE["models"]["glm-5-3"]}}
    record = learn.crew("code-review", 1, client=client, record_selection=False,
                        use_external_priors=True)
    assert record["external_priors"]["seeded"] == ["ollama/qwen2.5-coder:3b"]
    assert client.seeded


def test_ollama_models_parses_table():
    class Completed:
        returncode = 0
        stdout = "NAME ID SIZE MODIFIED\nqwen2.5-coder:3b abc 1.9 GB now\nphi4-mini:latest def 2.5 GB now\n"

    report = learn.ollama_models(runner=lambda *a, **k: Completed())
    assert report["available"] is True
    assert report["models"] == ["qwen2.5-coder:3b", "phi4-mini:latest"]


def test_ollama_models_missing_binary():
    def runner(*args, **kwargs):
        raise FileNotFoundError

    report = learn.ollama_models(runner=runner)
    assert report["available"] is False
    assert "not installed" in report["reason"]


def test_tune_delegates_to_ga(monkeypatch):
    calls = {}

    def fake_run(target, **kwargs):
        calls["target"] = target
        calls.update(kwargs)
        return RunResult("not_promoted:insufficient_evidence", 0, run_id="ga-1")

    from learning.ga import cli as ga_cli

    monkeypatch.setattr(ga_cli, "execute_run", fake_run)
    result = learn.tune(apply=False, seed=7, max_evals=3)
    assert result["outcome"] == "not_promoted:insufficient_evidence"
    assert calls["target"] == "thompson_policy"
    assert calls["seed"] == 7 and calls["max_evals"] == 3
    assert calls["apply"] is False


def test_journal_merges_ga_meta_and_external(monkeypatch):
    from learning.ga import journal as ga_journal

    monkeypatch.setattr(ga_journal, "read_entries", lambda: [{
        "kind": "apply", "recorded_at": "2026-11-19T10:00:00Z",
        "target": "thompson_policy", "change_id": "abc", "delta": 0.02,
        "reason": "passed all promotion gates"}])
    monkeypatch.setattr(learn, "_meta_journal_events", lambda limit=200: [{
        "at": "2026-11-19T11:00:00Z", "kind": "meta_outcome",
        "detail": "crew_size=3 outcome=1.0 cost=0.1"}])
    priors, _ = _isolate_prior_paths(monkeypatch, Path(tempfile.mkdtemp()))
    registry.register_prior(registry.PriorRecord(
        arm="a", model_id="a", alpha=2.0, beta=2.0, source="lmsys_arena_elo",
        registered_at="2026-11-19T09:00:00Z"))
    events = learn.journal_events(10)
    kinds = [event["kind"] for event in events]
    assert "ga:apply" in kinds
    assert "meta_outcome" in kinds
    assert "external_prior" in kinds
    assert events[0]["at"] == "2026-11-19T11:00:00Z"


def test_add_model_cli_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(learn, "add_model", lambda model, **kwargs: {
        "arm": model, "model_id": model, "alpha": 2.0, "beta": 2.0, "mean": 0.5,
        "source": "neutral", "match_type": "none", "snapshot": "", "note": "",
        "applied_via": None, "seed_error": None, "ollama": None})
    code = learn.main(["add-model", "ollama/qwen2.5-coder:3b", "--use-elo", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["arm"] == "ollama/qwen2.5-coder:3b"
