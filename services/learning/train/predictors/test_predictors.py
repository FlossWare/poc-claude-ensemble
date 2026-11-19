"""Fit/predict/evaluate/save-load tests for the three routing predictors."""

from __future__ import annotations

import json

import numpy as np
import pytest

from learning.train.predictors import _fixtures
from learning.train.predictors.base import InsufficientDataError, NotTrainedError
from learning.train.predictors.cost_predictor import CostPredictor
from learning.train.predictors.failure_predictor import FailurePredictor
from learning.train.predictors.features import extract_features
from learning.train.predictors.latency_predictor import LatencyPredictor


@pytest.fixture(scope="module")
def rows():
    return extract_features(_fixtures.synthetic_ledger_rows(count=180, seed=7))


def test_cost_predictor_fits_predicts_and_evaluates(rows):
    predictor = CostPredictor()
    metadata = predictor.fit(rows)
    metrics = metadata["metrics"]

    assert metadata["samples"] == 180
    assert metrics["evaluation"].endswith("cross-validation")
    assert metrics["r2"] > 0.5, metrics
    assert metrics["mae"] < metrics["baseline_mae"], metrics

    cheap = predictor.predict_cost("gamma-1", {"task_type": "code", "input_tokens": 500})
    expensive = predictor.predict_cost("delta-1", {"task_type": "code", "input_tokens": 500})
    assert 0.0 <= cheap < expensive
    assert predictor.predict("gamma-1", {"task_type": "code", "input_tokens": 500}) == pytest.approx(
        cheap, rel=1e-12
    )


def test_latency_predictor_fits_predicts_and_evaluates(rows):
    predictor = LatencyPredictor()
    metadata = predictor.fit(rows)
    metrics = metadata["metrics"]

    assert metrics["r2"] > 0.5, metrics
    assert metrics["mae"] < metrics["baseline_mae"], metrics

    fast = predictor.predict_latency("gamma-1", {"task_type": "code", "input_tokens": 500})
    slow = predictor.predict_latency("delta-1", {"task_type": "code", "input_tokens": 500})
    assert 0.0 <= fast < slow


def test_failure_predictor_learns_synthetic_signal(rows):
    predictor = FailurePredictor()
    metadata = predictor.fit(rows)
    metrics = metadata["metrics"]

    assert metrics["roc_auc"] is not None
    assert metrics["roc_auc"] > 0.65, metrics
    assert 0 <= metrics["confusion"]["tp"] <= metadata["failures"]

    risky = predictor.predict_failure_risk(
        "delta-1", {"task_type": "config", "input_tokens": 1500}
    )
    safe = predictor.predict_failure_risk(
        "gamma-1", {"task_type": "docs", "input_tokens": 1500}
    )
    assert 0.0 <= safe < risky <= 1.0


@pytest.mark.parametrize(
    "predictor_class,target_call",
    [
        (CostPredictor, lambda predictor: predictor.predict_cost("alpha-1", {"task_type": "code"})),
        (FailurePredictor, lambda predictor: predictor.predict_failure_risk("alpha-1", {"task_type": "code"})),
        (LatencyPredictor, lambda predictor: predictor.predict_latency("alpha-1", {"task_type": "code"})),
    ],
)
def test_save_load_round_trip(rows, tmp_path, predictor_class, target_call):
    predictor = predictor_class()
    predictor.fit(rows)
    before = target_call(predictor)

    artifact = predictor.save(tmp_path / f"{predictor_class.KIND}.pkl")
    assert artifact.exists()
    assert artifact.with_suffix(".json").exists()

    loaded = predictor_class.load(artifact)
    after = target_call(loaded)
    assert after == pytest.approx(before, rel=1e-12)
    assert loaded.metadata["feature_names"] == predictor.metadata["feature_names"]

    status = predictor_class.status(artifact)
    assert status["available"] is True
    assert status["samples"] == 180
    assert status["metrics"] == predictor.metadata["metrics"]


def test_sidecar_metadata_is_valid_json(rows, tmp_path):
    predictor = CostPredictor()
    predictor.fit(rows)
    artifact = predictor.save(tmp_path / "cost.pkl")
    document = json.loads(artifact.with_suffix(".json").read_text())
    assert document["kind"] == "cost"
    assert document["target"] == "cost_usd"
    assert document["samples"] == 180
    assert isinstance(document["warnings"], list)


def test_classifier_handles_single_class_and_round_trips(tmp_path):
    documents = _fixtures.synthetic_ledger_rows(count=40, seed=3)
    for document in documents:
        document["text"] = "always good"
        document["stop_reason"] = "end_turn"
        document.pop("error", None)
    rows = extract_features(documents)

    predictor = FailurePredictor()
    metadata = predictor.fit(rows)
    assert metadata["failures"] == 0
    assert predictor.predict_failure_risk("alpha-1", {"task_type": "code"}) == 0.0
    assert metadata["metrics"]["evaluation"].startswith("in-sample")

    artifact = predictor.save(tmp_path / "failure.pkl")
    loaded = FailurePredictor.load(artifact)
    assert loaded.predict_failure_risk("alpha-1", {"task_type": "code"}) == 0.0


def test_unseen_model_is_encoded_and_finite(rows, tmp_path):
    predictor = LatencyPredictor()
    predictor.fit(rows)
    value = predictor.predict_latency(
        "brand-new-model", {"task_type": "never_seen", "input_tokens": 100}
    )
    assert np.isfinite(value)
    assert value >= 0

    artifact = predictor.save(tmp_path / "latency.pkl")
    loaded = LatencyPredictor.load(artifact)
    assert loaded.predict_latency("brand-new-model", {"task_type": "never_seen"}) == pytest.approx(
        value, rel=1e-12
    )


def test_insufficient_data_and_untrained_errors():
    with pytest.raises(InsufficientDataError):
        CostPredictor().fit([])

    with pytest.raises(NotTrainedError):
        CostPredictor().predict_cost("alpha-1", {})

    with pytest.raises(NotTrainedError):
        CostPredictor.load("/nonexistent/path/cost.pkl")


def test_artifact_kind_mismatch_is_rejected(rows, tmp_path):
    predictor = CostPredictor()
    predictor.fit(rows)
    artifact = predictor.save(tmp_path / "cost.pkl")
    with pytest.raises(Exception, match="expected 'latency'"):
        LatencyPredictor.load(artifact)


def test_feature_usage_reports_ignored_outcome_fields(rows):
    predictor = CostPredictor()
    predictor.fit(rows)
    usage = predictor.feature_usage(
        {
            "model": "alpha-1",
            "task_type": "code",
            "input_tokens": 900,
            "text_length": 4000,
            "output_tokens": 1200,
        }
    )
    assert usage["used"] == {"input_tokens": 900, "task_type": "code"}
    assert usage["ignored"] == ["output_tokens", "text_length"]


def test_outcome_fields_are_excluded_from_model_inputs(rows):
    predictor = CostPredictor()
    predictor.fit(rows)
    names = predictor.feature_names
    for outcome in (
        "output_tokens",
        "text_length",
        "stop_reason",
        "latency_ms",
        "cost_usd",
        "tokens_per_second",
        "cost_per_token",
    ):
        assert outcome not in names
    assert names[-1] == "input_tokens"


def test_small_sample_warnings_are_present():
    # The real ledger is 50 rows / 19 episodes; warn loudly at that scale.
    rows = extract_features(_fixtures.synthetic_ledger_rows(count=20, seed=1))
    metadata = CostPredictor().fit(rows)
    warnings = " ".join(metadata["warnings"])
    assert "small sample" in warnings
    assert "not independent" in warnings
    assert "production accuracy" in warnings
