"""ThompsonClient.seed_prior request shape and failure handling."""

from __future__ import annotations

from shared.thompson_client import ThompsonClient


def test_seed_prior_sends_expected_request(monkeypatch):
    client = ThompsonClient(enable_circuit_breaker=False)
    captured = {}

    def fake_send(request):
        captured.update(request)
        return {"ok": True, "seeded": True}

    monkeypatch.setattr(client, "_send_request", fake_send)
    assert client.seed_prior("glm-5-3", 10.0, 2.0, prior_strength=1.0,
                             source="lmsys_arena_elo", detail={"rating": 1429.0})
    assert captured["action"] == "seed_prior"
    assert captured["model"] == "glm-5-3"
    assert captured["alpha"] == 10.0
    assert captured["beta"] == 2.0
    assert captured["detail"] == {"rating": 1429.0}


def test_seed_prior_reports_old_daemon_as_false(monkeypatch):
    client = ThompsonClient(enable_circuit_breaker=False)
    monkeypatch.setattr(
        client, "_send_request",
        lambda request: {"ok": False, "error": "Unknown action: seed_prior"},
    )
    assert client.seed_prior("x", 2.0, 2.0) is False


def test_seed_prior_reports_declined_seed(monkeypatch):
    client = ThompsonClient(enable_circuit_breaker=False)
    monkeypatch.setattr(
        client, "_send_request", lambda request: {"ok": True, "seeded": False}
    )
    assert client.seed_prior("x", 2.0, 2.0) is False
