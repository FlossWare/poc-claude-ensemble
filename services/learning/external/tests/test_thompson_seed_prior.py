"""The Thompson service's seed_prior action and request validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[3] / "thompson-service"


def _service(monkeypatch, tmp_path):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_LOG_DIR", str(tmp_path / "logs"))
    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    import thompson_service

    return thompson_service


def test_seed_prior_initialises_untested_arm(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    state = service.ThompsonState(tmp_path / "state.json")
    assert state.seed_prior("glm-5-3", 10.0, 2.0, prior_strength=1.0,
                            source="lmsys_arena_elo", detail={"rating": 1429.0})
    stats = state.models["glm-5-3"]
    assert stats.successes == pytest.approx(9.0)
    assert stats.failures == pytest.approx(1.0)
    assert stats.calls == 0
    assert stats.prior["source"] == "lmsys_arena_elo"

    reloaded = service.ThompsonState(tmp_path / "state.json")
    assert reloaded.models["glm-5-3"].prior["alpha"] == pytest.approx(10.0)
    # Effective posterior is the external prior.
    assert reloaded.models["glm-5-3"].successes + 1.0 == pytest.approx(10.0)
    assert reloaded.models["glm-5-3"].failures + 1.0 == pytest.approx(2.0)


def test_seed_prior_does_not_overwrite_real_evidence(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    state = service.ThompsonState(tmp_path / "state.json")
    state.record_outcome("glm-5-3", "code-review", True, 0.0, 10)
    assert not state.seed_prior("glm-5-3", 10.0, 2.0)
    assert state.models["glm-5-3"].successes == 1


def test_seed_prior_rejects_invalid_values(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    state = service.ThompsonState(tmp_path / "state.json")
    assert not state.seed_prior("x", 0.0, 2.0)
    assert not state.seed_prior("x", 2.0, -1.0)
    assert not state.seed_prior("", 2.0, 2.0)


def test_seed_prior_request_roundtrip(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    instance = service.ThompsonService(tmp_path / "sock", tmp_path / "state.json")
    response = json.loads(instance._process_request(
        json.dumps({
            "action": "seed_prior",
            "model": "ollama/qwen2.5-coder:3b",
            "alpha": 4.5,
            "beta": 1.5,
            "prior_strength": 1.0,
            "source": "lmsys_arena_elo",
        }),
        service.RequestContext(caller="test", method="http"),
    ))
    assert response["ok"] is True
    assert response["seeded"] is True
    stats = instance.state.models["ollama/qwen2.5-coder:3b"]
    assert stats.successes == pytest.approx(3.5)
    assert stats.failures == pytest.approx(0.5)

    # A structurally invalid seed is rejected by validators, not the handler.
    bad = json.loads(instance._process_request(
        json.dumps({"action": "seed_prior", "model": "x", "alpha": -1, "beta": 1}),
        service.RequestContext(caller="test", method="http"),
    ))
    assert bad["ok"] is False


def test_validator_accepts_and_rejects_seed_prior():
    from shared.validators import Validators

    ok, error = Validators.validate_thompson_request(
        {"action": "seed_prior", "model": "m", "alpha": 2.0, "beta": 3.0}, "seed_prior"
    )
    assert ok, error
    for payload in (
        {"model": "m", "alpha": 0.0, "beta": 3.0},
        {"model": "m", "alpha": 2.0},
        {"model": 3, "alpha": 2.0, "beta": 3.0},
        {"model": "m", "alpha": 2.0, "beta": 3.0, "detail": "not-json"},
    ):
        ok, _ = Validators.validate_thompson_request(payload, "seed_prior")
        assert not ok
