"""ELO -> Beta mapping, matching, cache, CLI, and restricted unpickling."""

from __future__ import annotations

import json
import math
import pickle

import pytest

from learning.external import elo, elo_priors, net, paths


def test_mapping_boundaries_are_exact():
    assert elo_priors.elo_to_beta(800) == pytest.approx((1.0, 5.0))
    assert elo_priors.elo_to_beta(1000) == pytest.approx((3.0, 3.0))
    assert elo_priors.elo_to_beta(1300) == pytest.approx((10.0, 2.0))


def test_mapping_clamps_and_stays_monotonic():
    assert elo_priors.elo_to_beta(400) == pytest.approx(elo_priors.elo_to_beta(800))
    assert elo_priors.elo_to_beta(2000) == pytest.approx(elo_priors.elo_to_beta(1500))
    previous_mean = -1.0
    previous_alpha = -1.0
    for rating in range(800, 1501, 25):
        alpha, beta = elo_priors.elo_to_beta(rating)
        mean = alpha / (alpha + beta)
        assert alpha > 0 and beta > 0
        assert mean >= previous_mean
        assert alpha >= previous_alpha
        previous_mean, previous_alpha = mean, alpha


def test_mapping_rejects_non_numbers():
    with pytest.raises(ValueError):
        elo_priors.elo_to_beta("high")
    with pytest.raises(ValueError):
        elo_priors.elo_to_beta(math.nan)


def test_shrink_prior_moves_toward_neutral():
    alpha, beta = elo_priors.shrink_prior(10.0, 2.0, 0.5)
    assert (alpha, beta) == pytest.approx((5.5, 1.5))
    assert elo_priors.shrink_prior(10.0, 2.0, 0.0) == pytest.approx((1.0, 1.0))


def test_normalize_and_match():
    assert elo_priors.normalize_name("rits/zai-org/glm-5-3") == "glm-5-3"
    assert elo_priors.normalize_name("ollama/qwen2.5-coder:3b") == "qwen2-5-coder-3b"
    ratings = {
        "glm-4.5": {"rating": 1429.0},
        "qwen2.5-coder-32b-instruct": {"rating": 1235.0},
        "claude-opus-4-1-20250805": {"rating": 1406.0},
    }
    assert elo_priors.match_model("glm-5-3", ratings)[0] == "glm-4.5"
    assert elo_priors.match_model("opus", ratings)[0] == "claude-opus-4-1-20250805"
    assert elo_priors.match_model("ollama/qwen2.5-coder:3b", ratings)[0] == (
        "qwen2.5-coder-32b-instruct"
    )
    assert elo_priors.match_model("unknown-model", ratings) == (None, "none")


def _write_cache(tmp_path, models):
    document = {
        "schema_version": 1,
        "source": elo_priors.SOURCE_PAGE,
        "snapshot": "elo_results_20250829.pkl",
        "snapshot_date": "2025-08-29",
        "fetched_at": "2026-01-01T00:00:00Z",
        "models": models,
    }
    path = tmp_path / "elo.json"
    net.write_cache(path, document)
    return path


def test_prior_for_model_uses_cache_without_network(tmp_path):
    cache = _write_cache(tmp_path, {
        "glm-4.5": {"rating": 1429.0},
        "qwen2.5-coder-32b-instruct": {"rating": 1235.0},
    })
    prior = elo_priors.prior_for_model("glm-5-3", cache_path=cache)
    assert prior.matched_model == "glm-4.5"
    assert prior.snapshot == "elo_results_20250829.pkl"
    expected_alpha, expected_beta = elo_priors.elo_to_beta(1429.0)
    assert (prior.alpha, prior.beta) == pytest.approx((expected_alpha, expected_beta))

    surrogate = elo_priors.prior_for_model("ollama/qwen2.5-coder:3b", cache_path=cache)
    assert surrogate.match_type in ("alias", "local_surrogate", "family_surrogate")
    assert surrogate.confidence == pytest.approx(0.5)
    base_beta = elo_priors.elo_to_beta(1235.0)[1]
    assert abs(surrogate.beta - 1.0) < abs(base_beta - 1.0)  # shrunk toward 1


def test_prior_for_model_no_match_is_neutral(tmp_path):
    cache = _write_cache(tmp_path, {"glm-4.5": {"rating": 1429.0}})
    prior = elo_priors.prior_for_model("totally-unknown", cache_path=cache)
    assert (prior.alpha, prior.beta) == (1.0, 1.0)
    assert prior.match_type == "none"


def test_latest_snapshot_and_parse_pickle_roundtrip(tmp_path):
    tree = [
        {"path": "elo_results_20250101.pkl"},
        {"path": "elo_results_20250829.pkl"},
        {"path": "leaderboard_table_20250804.csv"},
    ]
    assert elo_priors.latest_snapshot_name(tree) == "elo_results_20250829.pkl"
    with pytest.raises(elo_priors.EloSourceError):
        elo_priors.latest_snapshot_name([])


def test_restricted_unpickler_refuses_code_execution(tmp_path):
    marker = tmp_path / "pwned"
    payload = pickle.dumps((__import__("os").system, (f"touch {marker}",)))
    with pytest.raises(pickle.UnpicklingError):
        elo_priors.parse_elo_pickle(payload)
    assert not marker.exists()


def test_elo_cli_prints_prior(tmp_path, capsys, monkeypatch):
    cache = _write_cache(tmp_path, {"glm-4.5": {"rating": 1429.0}})
    code = elo.main(["--model", "glm-5-3", "--cache", str(cache), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matched_model"] == "glm-4.5"
    assert payload["alpha"] > 1.0


def test_elo_cli_list(tmp_path, capsys):
    cache = _write_cache(tmp_path, {
        "glm-4.5": {"rating": 1429.0},
        "deepseek-v3.1": {"rating": 1417.0},
    })
    code = elo.main(["--list", "--limit", "1", "--cache", str(cache), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0]["model"] == "glm-4.5"


def test_paths_honours_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.EXTERNAL_DIR_ENV, str(tmp_path / "external"))
    assert paths.elo_cache_path() == tmp_path / "external" / "elo.json"
    assert paths.ensure_dir().is_dir()
