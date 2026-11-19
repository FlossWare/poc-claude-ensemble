"""Retrieval relevance: nullable labels, journal-schema mapping, logistic model."""

from __future__ import annotations

import pytest

from learning.train import _fixtures as fx
from learning.train import retrieval_relevance as rr


def test_injection_features_map_hook_sections():
    row = fx.injection_rows(labeled=True, count=2)[1]
    features = rr.injection_features(row)
    assert features["memory_count"] == row["sections"]["memory"]["results"]
    assert features["knowledge_count"] == row["sections"]["knowledge"]["results"]
    assert features["memory_chars"] == row["sections"]["memory"]["chars"]
    assert features["knowledge_chars"] == row["sections"]["knowledge"]["chars"]
    assert features["graph_count"] == 0
    assert features["graph_chars"] == 0
    assert features["task_type"] == row["sections"]["intent"]["label"]


def test_all_null_labels_report_insufficient_and_fit_refuses():
    rows = fx.injection_rows(labeled=False)
    data, info = rr.extract_training_data(rows)
    assert len(data) == 0
    assert info.total_rows == len(rows)
    assert info.labeled_rows == 0
    assert info.null_label_rows == len(rows)
    assert info.sufficient is False
    assert "insufficient labeled data" in rr.insufficient_message(info)
    with pytest.raises(rr.InsufficientLabeledData):
        rr.RetrievalRelevanceModel.fit(data)


def test_labeled_rows_train_predict_roundtrip(tmp_path):
    rows = fx.injection_rows(labeled=True, count=40)
    data, info = rr.extract_training_data(rows)
    assert info.labeled_rows == 40
    assert set(data.labels) == {0, 1}

    model = rr.RetrievalRelevanceModel.fit(data)
    assert model.model.n_train + model.model.n_test == 40
    assert model.model.metrics["test"]["accuracy"] is not None
    coefficients = model.coefficients()
    assert set(coefficients) == {"0", "1"}
    assert "memory_count" in coefficients["1"]
    # Binary linear coefficients are the positive class and its negation.
    assert coefficients["0"]["memory_count"] == pytest.approx(
        -coefficients["1"]["memory_count"]
    )

    helpful = rr.injection_features(rows[0])   # even index -> helped=True
    unhelpful = rr.injection_features(rows[1])  # odd -> helped=False
    assert model.predict_relevance(helpful) > model.predict_relevance(unhelpful)

    path = tmp_path / "retrieval.joblib"
    model.save(str(path))
    loaded = rr.RetrievalRelevanceModel.load(str(path))
    assert loaded.predict_relevance(helpful) == pytest.approx(
        model.predict_relevance(helpful)
    )
    stamp = loaded.model.version_stamp()
    assert stamp["label_name"] == "helped"
    assert stamp["training_data_hash"] == data.data_hash()
    assert rr.predict_relevance(str(path), helpful) == pytest.approx(
        model.predict_relevance(helpful)
    )


def test_mixed_null_labels_are_skipped_not_invented():
    rows = fx.injection_rows(labeled=True, count=20)
    for row in rows[::3]:
        row["helped"] = None
    data, info = rr.extract_training_data(rows)
    assert info.null_label_rows == len(rows[::3])
    assert len(data) == len(rows) - len(rows[::3])
