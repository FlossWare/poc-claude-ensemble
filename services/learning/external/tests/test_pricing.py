"""Pricing catalogue parsing, matching, caching, cost estimates, predictor sync."""

from __future__ import annotations

import json

import pytest

from learning.external import net, pricing, pricing_cli

CATALOGUE = {
    "data": [
        {
            "id": "anthropic/claude-opus-5.5",
            "name": "Anthropic: Claude Opus 5.5",
            "pricing": {"prompt": "0.000004", "completion": "0.00002"},
            "context_length": 200000,
        },
        {
            "id": "anthropic/claude-opus-5.5:batch",
            "name": "Anthropic: Claude Opus 5.5 (batch)",
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        },
        {
            "id": "z-ai/glm-5.3",
            "name": "Z.ai: GLM 5.3",
            "pricing": {"prompt": "0.0000000699", "completion": "0.0000014"},
        },
        {"id": "no-price/model", "name": "No price", "pricing": {}},
    ]
}


def _fake_fetch(monkeypatch, document=CATALOGUE):
    monkeypatch.setattr(pricing.net, "fetch_json", lambda url, **kwargs: document)


def test_parse_catalogue_skips_models_without_prices():
    models = pricing.parse_catalogue(CATALOGUE)
    assert "anthropic/claude-opus-5.5" in models
    assert "no-price/model" not in models
    assert models["anthropic/claude-opus-5.5"]["prompt_per_token"] == pytest.approx(0.000004)


def test_match_price_handles_dots_dashes_and_prefers_plain_variant():
    models = pricing.parse_catalogue(CATALOGUE)
    assert pricing.match_price("rits/zai-org/glm-5-3", models) == ("z-ai/glm-5.3", "exact")
    assert pricing.match_price("claude-opus-5-5", models) == (
        "anthropic/claude-opus-5.5", "exact"
    )
    assert pricing.match_price("unmatched", models) == (None, "none")


def test_match_price_family_suffix():
    models = {"meta/muse-spark-open": {"prompt_per_token": 1.25e-6,
                                       "completion_per_token": 4.25e-6}}
    assert pricing.match_price(
        "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4", models
    ) == ("meta/muse-spark-open", "family")


def test_price_for_model_caches_and_respects_freshness(tmp_path, monkeypatch):
    cache = tmp_path / "pricing.json"
    _fake_fetch(monkeypatch)
    first = pricing.price_for_model("claude-opus-5-5", cache_path=cache)
    assert first.openrouter_id == "anthropic/claude-opus-5.5"
    assert first.estimate(1000, 2000) == pytest.approx(0.044)
    assert cache.exists()

    def explode(url, **kwargs):
        raise AssertionError("network used while the cache is fresh")

    monkeypatch.setattr(pricing.net, "fetch_json", explode)
    second = pricing.price_for_model("claude-opus-5-5", cache_path=cache)
    assert second.prompt_per_token == first.prompt_per_token


def test_local_ollama_is_free_without_network(monkeypatch):
    def explode(url, **kwargs):
        raise AssertionError("Ollama price lookup must not touch the network")

    monkeypatch.setattr(pricing.net, "fetch_json", explode)
    quote = pricing.price_for_model("ollama/qwen2.5-coder:3b")
    assert quote.match_type == "local"
    assert quote.estimate(1_000_000, 1_000_000) == 0.0


def test_estimate_cost_falls_back_to_unavailable_without_predictor(tmp_path, monkeypatch):
    cache = tmp_path / "pricing.json"
    _fake_fetch(monkeypatch)
    result = pricing.estimate_cost(
        "mystery-model", 1000, 2000, cache_path=cache, predictor_dir=tmp_path / "predictors"
    )
    assert result["cost_usd"] is None
    assert result["source"] == "unavailable"


def test_sync_cost_predictor_updates_sidecar(tmp_path, monkeypatch):
    _fake_fetch(monkeypatch)
    predictor_dir = tmp_path / "predictors"
    predictor_dir.mkdir()
    (predictor_dir / "cost.json").write_text(json.dumps({"kind": "cost", "metrics": {}}))
    report = pricing.sync_cost_predictor(
        cache_path=tmp_path / "pricing.json", predictor_dir=predictor_dir
    )
    assert report["sidecar_updated"]
    sidecar = json.loads((predictor_dir / "cost.json").read_text())
    assert sidecar["pricing"]["models"] == 3
    assert (predictor_dir / "pricing.json").exists()


def test_pricing_cli_json(tmp_path, capsys, monkeypatch):
    _fake_fetch(monkeypatch)
    code = pricing_cli.main([
        "--model", "glm-5-3",
        "--cache", str(tmp_path / "pricing.json"),
        "--input-tokens", "1000",
        "--output-tokens", "2000",
        "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["openrouter_id"] == "z-ai/glm-5.3"
    assert payload["estimate"] == pytest.approx(0.0028699)
