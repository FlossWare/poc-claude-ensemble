"""Tests for the local OpenVINO serving-layer provider (Arc iGPU MoE arms)."""

from __future__ import annotations

import pytest

from providers.local_serving import (
    LOCAL_RUNTIME,
    LocalServingProvider,
    endpoint_map,
)
from providers.model_provider import ModelRequest
from providers.registry import ProviderRegistry


def _provider(**kwargs) -> LocalServingProvider:
    return LocalServingProvider(
        endpoints={"qwen3-coder-30b-a3b": "http://127.0.0.1:8931"}, **kwargs
    )


def test_label_resolves_to_local_runtime_and_strips_prefix() -> None:
    registry = ProviderRegistry(
        {LOCAL_RUNTIME: _provider()},
    )
    assert registry.resolve_model("local/qwen3-coder-30b-a3b") == (
        "local",
        "qwen3-coder-30b-a3b",
    )
    assert isinstance(registry.resolve("local/qwen3-coder-30b-a3b"), LocalServingProvider)


def test_empty_local_label_is_rejected() -> None:
    registry = ProviderRegistry({LOCAL_RUNTIME: _provider()})
    with pytest.raises(ValueError, match="must include a model name"):
        registry.resolve_model("local/")


def test_unconfigured_local_model_fails_at_resolve_time() -> None:
    registry = ProviderRegistry({LOCAL_RUNTIME: _provider()})
    with pytest.raises(RuntimeError, match="no local serving endpoint"):
        registry.resolve_model("local/not-served")


def test_build_payload_matches_the_anthropic_shim() -> None:
    provider = _provider()
    payload = provider.build_payload(
        ModelRequest(
            prompt="hello",
            model="qwen3-coder-30b-a3b",
            system_prompt="be brief",
            messages=({"role": "assistant", "content": "hi"},),
            max_tokens=64,
        )
    )
    assert payload == {
        "model": "qwen3-coder-30b-a3b",
        "max_tokens": 64,
        "messages": [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ],
        "system": "be brief",
    }


def test_build_payload_clamps_max_tokens_to_the_shim_limit() -> None:
    provider = _provider()
    payload = provider.build_payload(
        ModelRequest(prompt="x", model="qwen3-coder-30b-a3b", max_tokens=99999)
    )
    assert payload["max_tokens"] == 4096


def test_parse_response_reads_text_and_usage() -> None:
    response = LocalServingProvider.parse_response(
        {
            "id": "msg_local_0_1",
            "model": "qwen3-coder-30b-a3b",
            "content": [{"type": "text", "text": "def f(): pass"}],
            "usage": {"input_tokens": 12, "output_tokens": 5},
        },
        requested_model="qwen3-coder-30b-a3b",
        latency_ms=10.0,
    )
    assert response.provider == "local"
    assert response.text == "def f(): pass"
    assert (response.input_tokens, response.output_tokens) == (12, 5)
    assert response.cost_usd == 0.0
    assert response.request_id == "msg_local_0_1"


def test_parse_response_rejects_a_non_anthropic_body() -> None:
    with pytest.raises(RuntimeError, match="content"):
        LocalServingProvider.parse_response(
            {"error": "boom"}, requested_model="m"
        )


def test_temperature_override_is_not_advertised() -> None:
    # The shim generates greedily and ignores temperature; advertising support
    # would make callers believe a knob works when it does not.
    assert _provider().capabilities.supports_temperature_override is False


def test_endpoint_map_reads_json_override(monkeypatch) -> None:
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_URLS",
        '{"qwen3-coder-30b-a3b": "http://127.0.0.1:9999/"}',
    )
    endpoints = endpoint_map()
    assert endpoints["qwen3-coder-30b-a3b"] == "http://127.0.0.1:9999"


def test_endpoint_map_rejects_non_object_override(monkeypatch) -> None:
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_URLS", "[]")
    with pytest.raises(ValueError, match="JSON object"):
        endpoint_map()
