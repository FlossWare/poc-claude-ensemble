"""Prior registry and application to Thompson state."""

from __future__ import annotations

import json

import pytest

from learning.external import registry


def _record(arm="ollama/qwen2.5-coder:3b", alpha=5.5, beta=1.5):
    return registry.PriorRecord(
        arm=arm,
        model_id=arm,
        alpha=alpha,
        beta=beta,
        source="lmsys_arena_elo",
        match_type="local_surrogate",
        confidence=0.5,
        snapshot="elo_results_20250829.pkl",
        detail={"rating": 1235.1},
    )


def test_register_list_and_get(tmp_path):
    path = tmp_path / "priors.json"
    registry.register_prior(_record(), path=path)
    records = registry.list_priors(path=path)
    assert len(records) == 1
    assert records[0].arm == "ollama/qwen2.5-coder:3b"
    assert records[0].mean == pytest.approx(5.5 / 7.0)
    assert registry.get_prior("missing", path=path) is None


def test_apply_priors_to_state_uses_pseudo_counts(tmp_path, monkeypatch):
    priors = tmp_path / "priors.json"
    state = tmp_path / "state.json"
    registry.register_prior(_record(), path=priors)
    original_list = registry.list_priors
    monkeypatch.setattr(registry, "list_priors", lambda path=None: original_list(path=priors))
    monkeypatch.setattr(registry, "registry_path", lambda path=None: priors)
    monkeypatch.setattr(registry, "mark_applied", lambda arm, via, path=None: None)

    outcome = registry.apply_priors_to_state(state_path=state, prior_strength=1.0)
    assert outcome["applied"][0]["arm"] == "ollama/qwen2.5-coder:3b"
    document = json.loads(state.read_text())
    stats = document["models"]["ollama/qwen2.5-coder:3b"]
    assert stats["successes"] == pytest.approx(4.5)
    assert stats["failures"] == pytest.approx(0.5)
    assert stats["calls"] == 0
    assert stats["prior"]["alpha"] == pytest.approx(5.5)
    # Effective posterior is exactly the external prior.
    assert stats["successes"] + 1.0 == pytest.approx(5.5)
    assert stats["failures"] + 1.0 == pytest.approx(1.5)


def test_apply_never_overwrites_observed_calls(tmp_path, monkeypatch):
    priors = tmp_path / "priors.json"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "last_updated": "2026-01-01T00:00:00Z",
        "models": {
            "ollama/qwen2.5-coder:3b": {
                "model_name": "ollama/qwen2.5-coder:3b",
                "successes": 9, "failures": 1, "calls": 10,
                "total_cost": 0.0, "total_tokens": 0,
            }
        },
    }))
    registry.register_prior(_record(), path=priors)
    original_list = registry.list_priors
    monkeypatch.setattr(registry, "list_priors", lambda path=None: original_list(path=priors))
    monkeypatch.setattr(registry, "registry_path", lambda path=None: priors)
    monkeypatch.setattr(registry, "mark_applied", lambda arm, via, path=None: None)

    outcome = registry.apply_priors_to_state(state_path=state, prior_strength=1.0)
    assert outcome["applied"] == []
    assert outcome["skipped"][0]["reason"] == "observed_calls"
    stats = json.loads(state.read_text())["models"]["ollama/qwen2.5-coder:3b"]
    assert stats["successes"] == 9


def test_apply_priors_via_client_marks_service_path(tmp_path, monkeypatch):
    priors = tmp_path / "priors.json"
    registry.register_prior(_record(), path=priors)
    original_list = registry.list_priors
    monkeypatch.setattr(registry, "list_priors", lambda path=None: original_list(path=priors))
    monkeypatch.setattr(registry, "registry_path", lambda path=None: priors)
    marked = {}

    def fake_mark(arm, via, path=None):
        marked[arm] = via

    monkeypatch.setattr(registry, "mark_applied", fake_mark)

    class Client:
        def seed_prior(self, *args, **kwargs):
            return True

    outcome = registry.apply_priors_via_client(Client())
    assert outcome["applied"]
    assert marked["ollama/qwen2.5-coder:3b"] == "service"


def test_unapplied_skips_seeded_and_observed(tmp_path):
    records = [_record("new"), _record("seeded"), _record("observed")]
    state = {
        "models": {
            "seeded": {"calls": 0, "prior": {"alpha": 2.0}},
            "observed": {"calls": 3},
        }
    }
    pending = registry.unapplied(records, state)
    assert [record.arm for record in pending] == ["new"]
