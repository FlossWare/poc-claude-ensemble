"""Quality predictor: SELECTED labels, proxy fallback, fit/evaluate/persist."""

from __future__ import annotations

import json

import pytest

from learning.train import _fixtures as fx
from learning.train import quality_predictor as qp


def test_parse_selected_and_label_matching():
    assert qp.parse_selected("blah\nVERDICT: approve\nSELECTED: GLM 5.3") == "GLM 5.3"
    assert qp.parse_selected("no selection here") is None
    assert qp.label_matches("GLM 5.3", "rits/zai-org/glm-5-3")
    assert qp.label_matches("DeepSeek V4.1 Flash", "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017")
    assert not qp.label_matches("Muse Spark Open", "rits/zai-org/glm-5-3")
    assert not qp.label_matches(None, "rits/zai-org/glm-5-3")


def test_extract_authoritative_labels_from_selected_lines():
    rows = fx.review_rows(reviews=8, workers=3, selected=True)
    data, info = qp.extract_training_data(rows)
    assert info.reviews == 8
    assert info.worker_rows == 24
    assert info.reviews_with_selection == 8
    assert info.authoritative_labeled == 24
    assert info.authoritative_positive == 8  # exactly one winner per review
    assert info.used_proxy is False
    assert data.label_source == "arbiter_selected"
    assert data.class_balance() == {"1": 8, "0": 16}


def test_extract_falls_back_to_usable_output_proxy_without_selection():
    rows = fx.review_rows(reviews=8, workers=3, selected=False, empty_worker=True)
    data, info = qp.extract_training_data(rows)
    assert info.reviews_with_selection == 0
    assert info.authoritative_labeled == 0
    assert info.used_proxy is True
    assert data.label_source == "usable_output_proxy"
    assert "no arbiter SELECTED line" in info.note
    # Reviews 0 and 4 have one empty worker: 2 negatives, 22 positives.
    assert data.class_balance() == {"0": 2, "1": 22}


def test_extract_without_proxy_yields_no_labels_when_selection_absent():
    rows = fx.review_rows(reviews=4, workers=3, selected=False)
    data, info = qp.extract_training_data(rows, allow_proxy=False)
    assert len(data) == 0
    assert info.authoritative_labeled == 0


def test_fit_predict_evaluate_and_feature_importances(tmp_path):
    rows = fx.review_rows(reviews=12, workers=3, selected=True)
    data, _ = qp.extract_training_data(rows)
    predictor = qp.QualityPredictor.fit(data, n_estimators=50)
    assert predictor.model.n_train == pytest.approx(int(len(data) * 0.7), abs=0)
    assert predictor.model.n_train + predictor.model.n_test == len(data)
    assert predictor.model.metrics["train"]["n"] == predictor.model.n_train
    assert predictor.model.metrics["test"]["n"] == predictor.model.n_test
    assert predictor.model.metrics["train"]["accuracy"] is not None

    winner = qp.worker_features(rows[0])
    loser = qp.worker_features(rows[1])
    assert predictor.predict_quality(winner) > predictor.predict_quality(loser)

    importances = predictor.feature_importances()
    assert importances and importances[0][1] >= importances[-1][1]
    assert any(name.startswith("stop_reason=") for name, _ in importances)

    path = tmp_path / "quality.joblib"
    predictor.save(str(path))
    loaded = qp.QualityPredictor.load(str(path))
    assert loaded.predict_quality(winner) == pytest.approx(predictor.predict_quality(winner))
    stamp = loaded.model.version_stamp()
    assert stamp["training_data_hash"] == data.data_hash()
    assert stamp["label_source"] == "arbiter_selected"
    assert stamp["metrics"]["test"]["n"] == predictor.model.n_test

    # The public integration entry point accepts a path, a model, or a predictor.
    assert qp.predict_quality(str(path), winner) == pytest.approx(
        predictor.predict_quality(winner)
    )
    assert qp.predict_quality(predictor.model, winner) == pytest.approx(
        predictor.predict_quality(winner)
    )


def test_cold_start_prior_is_clipped():
    rows = fx.review_rows(reviews=12, workers=3, selected=True)
    data, _ = qp.extract_training_data(rows)
    predictor = qp.QualityPredictor.fit(data, n_estimators=50)
    prior = predictor.cold_start_prior(qp.worker_features(rows[0]))
    assert 0.05 <= prior <= 0.95
    # A brand-new arm (no review history) still gets a bounded prior.
    seeded = qp.cold_start_prior(predictor, "provider/new-model")
    assert 0.05 <= seeded <= 0.95


def test_insufficient_data_raises():
    data, _ = qp.extract_training_data(fx.review_rows(reviews=1, workers=1, selected=True))
    with pytest.raises(qp.InsufficientTrainingData):
        qp.QualityPredictor.fit(data)


def test_real_schema_sample_from_jsonl(tmp_path):
    """Integration: read a schema-faithful JSONL sample and report metrics."""
    path = tmp_path / "reviews.jsonl"
    fx.write_jsonl(path, fx.review_rows(reviews=12, workers=3, selected=True))
    rows = qp.load_rows(str(path))
    data, info = qp.extract_training_data(rows)
    predictor = qp.QualityPredictor.fit(data, n_estimators=50)
    metrics = predictor.evaluate(data)
    assert metrics["n"] == len(data)
    assert metrics["accuracy"] is not None
    print(json.dumps({"info": info.as_dict(), "metrics": metrics}, indent=2))
