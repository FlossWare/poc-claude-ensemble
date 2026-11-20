"""Benchmark index/parse/estimate, cached fetch, and cold-start pre-training."""

from __future__ import annotations

import json

import pytest

from learning.external import benchmark_priors as bp, benchmarks, net

SAMPLE_RESULT = {
    "results": {
        "leaderboard_mmlu_pro": {"acc,none": 0.44},
        "leaderboard_bbh": {"acc_norm,none": 0.66},
        "leaderboard_math_hard": {"exact_match,none": 0.50},
        "leaderboard_gpqa": {"acc_norm,none": 0.35},
        "leaderboard_musr": {"acc_norm,none": 0.44},
        "leaderboard_ifeval": {"prompt_level_strict_acc,none": 0.68},
        "leaderboard_humaneval": {"pass@1,none": 0.72},
    }
}


def test_build_index_keeps_newest_result():
    siblings = [
        {"rfilename": "Qwen/Qwen2.5-Coder-32B-Instruct/results_2024-01-01T00-00-00.json"},
        {"rfilename": "Qwen/Qwen2.5-Coder-32B-Instruct/results_2025-02-13T18-27-04.json"},
        {"rfilename": "org/model/config.json"},
    ]
    index = bp.build_index(siblings)
    assert index == {
        "Qwen/Qwen2.5-Coder-32B-Instruct":
            "Qwen/Qwen2.5-Coder-32B-Instruct/results_2025-02-13T18-27-04.json"
    }


def test_parse_result_document_reads_v2_and_optional_humaneval():
    scores = bp.parse_result_document(SAMPLE_RESULT)
    assert scores["mmlu_pro"] == pytest.approx(0.44)
    assert scores["humaneval"] == pytest.approx(0.72)


def test_estimate_quality_renormalizes_missing_metrics():
    full = bp.estimate_quality(bp.parse_result_document(SAMPLE_RESULT))
    partial = bp.estimate_quality({"mmlu_pro": 0.44, "bbh": 0.66})
    assert full is not None and 0.0 <= full <= 1.0
    assert partial == pytest.approx((0.44 * 0.25 + 0.66 * 0.15) / 0.40)
    assert bp.estimate_quality({}) is None


def test_match_repo_prefers_canonical_alias():
    repos = [
        "Etherll/Qwen2.5-Coder-7B-Instruct-Ties",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "google/gemma-2-27b-it",
        "google/gemma-3-27b-it",
    ]
    assert bp.match_repo("ollama/qwen2.5-coder:3b", repos) == (
        "Qwen/Qwen2.5-Coder-32B-Instruct", "alias"
    )
    assert bp.match_repo("ollama/gemma3:4b", repos)[0] == "google/gemma-3-27b-it"
    assert bp.match_repo("nothing-here", repos) == (None, "none")


def _fake_fetch_json(monkeypatch, index_document, result_document=SAMPLE_RESULT):
    def fake(url, **kwargs):
        if "api/datasets" in url:
            return index_document
        return result_document

    monkeypatch.setattr(bp.net, "fetch_json", fake)


def test_fetch_model_benchmarks_caches_result(tmp_path, monkeypatch):
    cache = tmp_path / "benchmarks.json"
    _fake_fetch_json(monkeypatch, {
        "siblings": [
            {"rfilename": "Qwen/Qwen2.5-Coder-32B-Instruct/results_2025-02-13T18-27-04.json"}
        ]
    })
    first = bp.fetch_model_benchmarks("Qwen/Qwen2.5-Coder-32B-Instruct", cache_path=cache)
    assert first.repo == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert first.match_type == "exact"
    assert first.estimated_quality is not None
    assert cache.exists()

    # Second call reads the per-model cache and must not fetch anything.
    def explode(url, **kwargs):
        raise AssertionError("network used on a cached lookup")

    monkeypatch.setattr(bp.net, "fetch_json", explode)
    second = bp.fetch_model_benchmarks("Qwen/Qwen2.5-Coder-32B-Instruct", cache_path=cache)
    assert second.estimated_quality == pytest.approx(first.estimated_quality)


def test_cached_benchmark_never_fetches(tmp_path, monkeypatch):
    cache = tmp_path / "benchmarks.json"
    net.write_cache(cache, {
        "schema_version": 1,
        "index": {},
        "results": {
            "m": {
                "model": "m",
                "repo": "org/m",
                "scores": {"mmlu_pro": 0.5},
                "estimated_quality": 0.5,
                "match_type": "exact",
            }
        },
    })
    record = bp.cached_benchmark("m", cache_path=cache)
    assert record is not None and record.estimated_quality == 0.5
    assert bp.cached_benchmark("other", cache_path=cache) is None


def test_quality_prior_shrinks_surrogates():
    exact = bp.BenchmarkPrior(
        model="Qwen/Qwen2.5-Coder-32B-Instruct", repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        result_file=None, scores={}, estimated_quality=0.8, match_type="exact",
    )
    surrogate = bp.BenchmarkPrior(
        model="ollama/qwen2.5-coder:3b", repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        result_file=None, scores={}, estimated_quality=0.8, match_type="alias",
    )
    exact_alpha, exact_beta = exact.quality_prior()
    surrogate_alpha, surrogate_beta = surrogate.quality_prior()
    assert exact_alpha > surrogate_alpha
    assert abs(surrogate_alpha - 1.0) < abs(exact_alpha - 1.0)


def test_pretrain_quality_predictor_writes_separate_artifact(tmp_path, monkeypatch):
    qualities = [0.25, 0.75, 0.30, 0.80, 0.20, 0.70, 0.35, 0.85]

    def fake_fetch(model, **kwargs):
        quality = qualities[bp.KNOWN_MODELS.index(model) % len(qualities)]
        return bp.BenchmarkPrior(
            model=model, repo=f"org/{model}", result_file=None,
            scores={"mmlu_pro": quality}, estimated_quality=quality,
            match_type="exact",
        )

    monkeypatch.setattr(bp, "fetch_model_benchmarks", fake_fetch)
    artifact = tmp_path / "quality_predictor_benchmark.joblib"
    report = bp.pretrain_quality_predictor(artifact_path=artifact)
    assert artifact.exists()
    assert report["rows"] == len(bp.KNOWN_MODELS)
    assert set(report["class_balance"]) == {"0", "1"}
    assert "benchmark" in report["label_source"]

    from learning.train.quality_predictor import QualityPredictor, default_worker_features

    predictor = QualityPredictor.load(str(artifact))
    score = predictor.cold_start_prior(default_worker_features("ollama/qwen3:4b"))
    assert 0.0 <= score <= 1.0


def test_benchmarks_cli_reads_cache(tmp_path, capsys, monkeypatch):
    cache = tmp_path / "benchmarks.json"
    net.write_cache(cache, {
        "schema_version": 1,
        "index": {},
        "results": {
            "m": {
                "model": "m", "repo": "org/m", "result_file": "org/m/results_2025-01-01T00-00-00.json",
                "scores": {"mmlu_pro": 0.5, "bbh": 0.6},
                "estimated_quality": 0.54, "match_type": "exact",
                "snapshot": "2025-01-01", "fetched_at": "2026-01-01T00:00:00Z",
            }
        },
    })
    code = benchmarks.main(["--model", "m", "--cache", str(cache), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["estimated_quality"] == pytest.approx(0.54)
    assert "quality_beta_prior" in payload
