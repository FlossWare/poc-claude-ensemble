"""Endpoint tests for model capabilities and health."""
from __future__ import annotations

from types import SimpleNamespace

import server.ensemble_server as ensemble_server


class FakeCapabilityHealth:
    def capabilities(self):
        return {"models": [{"model": "sonnet", "provider": "anthropic", "credential_available": True}]}

    def health(self):
        return {"models": [{"model": "sonnet", "state": "available", "calls": 3}]}


def test_capabilities_endpoint(monkeypatch) -> None:
    application = ensemble_server.EnsembleApplication.__new__(ensemble_server.EnsembleApplication)
    application.capability_health = FakeCapabilityHealth()
    sent = {}
    monkeypatch.setattr(ensemble_server, "_send", lambda _handler, status, payload: sent.update(status=status, payload=payload))
    application._handle_capabilities(SimpleNamespace(command="GET"), "/")
    assert sent["status"] == ensemble_server.HTTPStatus.OK
    assert sent["payload"]["ok"] is True
    assert sent["payload"]["models"][0]["provider"] == "anthropic"
    assert sent["payload"]["models"][0]["credential_available"] is True


def test_model_health_endpoint(monkeypatch) -> None:
    application = ensemble_server.EnsembleApplication.__new__(ensemble_server.EnsembleApplication)
    application.capability_health = FakeCapabilityHealth()
    sent = {}
    monkeypatch.setattr(ensemble_server, "_send", lambda _handler, status, payload: sent.update(status=status, payload=payload))
    application._handle_capabilities(SimpleNamespace(command="GET"), "/health")
    assert sent["status"] == ensemble_server.HTTPStatus.OK
    assert sent["payload"]["models"][0]["state"] == "available"
