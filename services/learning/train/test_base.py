"""Shared infrastructure: chronological splits, imbalance, artifacts, trigger."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from learning.train.base import (
    FeatureEncoder,
    RetrainTrigger,
    TrainingData,
    TrainedModel,
    evaluate_predictions,
)


class _TwoFeatureEncoder(FeatureEncoder):
    feature_names = ["x", "y"]

    def fit(self, features):
        return self

    def transform(self, features):
        return np.asarray([[float(f["x"]), float(f["y"])] for f in features])


def _data(labels=(0, 1, 0, 1, 0, 1), timestamps=None):
    features = [{"x": index, "y": index % 2} for index in range(len(labels))]
    if timestamps is None:
        timestamps = [f"2026-01-{index + 1:02d}T00:00:00Z" for index in range(len(labels))]
    return TrainingData(features=features, labels=list(labels), timestamps=list(timestamps))


def test_split_is_chronological_with_no_lookahead():
    # Deliberately shuffled input order; split must sort by timestamp first.
    timestamps = [
        "2026-01-03T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-05T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-09T00:00:00Z",
        "2026-01-04T00:00:00Z",
        "2026-01-07T00:00:00Z",
        "2026-01-06T00:00:00Z",
        "2026-01-08T00:00:00Z",
        "2026-01-10T00:00:00Z",
    ]
    data = _data(labels=[index % 2 for index in range(10)], timestamps=timestamps)
    train, test = data.split(0.7)
    assert len(train) == 7 and len(test) == 3
    assert max(train.timestamps) <= min(test.timestamps)
    # The chronological order is preserved in each side.
    assert train.timestamps == sorted(train.timestamps)
    assert test.timestamps == sorted(test.timestamps)


def test_empty_data_has_empty_split_and_no_weights():
    data = TrainingData(features=[], labels=[])
    assert data.is_empty
    train, test = data.split()
    assert train.is_empty and test.is_empty
    assert data.class_balance() == {}
    assert data.class_balance_report()["is_imbalanced"] is False
    assert data.sample_weights().size == 0
    assert data.data_hash()  # stable hash of the empty set


def test_class_balance_and_balanced_sample_weights():
    data = _data(labels=[1] * 8 + [0] * 2)
    report = data.class_balance_report()
    assert report["counts"] == {"1": 8, "0": 2}
    assert report["is_imbalanced"] is True
    assert report["imbalance_ratio"] == pytest.approx(4.0)
    weights = data.sample_weights()
    # Minority rows weigh more; the weights average to 1.
    assert weights[8:].mean() > weights[:8].mean()
    assert weights.mean() == pytest.approx(1.0)


def test_data_hash_is_order_stable_and_content_sensitive():
    original = _data(labels=[0, 1, 0, 1])
    shuffled = TrainingData(
        features=list(reversed(original.features)),
        labels=list(reversed(original.labels)),
        timestamps=list(reversed(original.timestamps)),
    )
    assert original.data_hash() == shuffled.data_hash()
    changed = TrainingData(
        features=original.features, labels=[0, 1, 1, 1], timestamps=original.timestamps
    )
    assert changed.data_hash() != original.data_hash()


def test_metrics_binary_and_empty():
    metrics = evaluate_predictions([0, 1, 1, 0], [0, 1, 0, 0], np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]]))
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["auc"] == pytest.approx(0.75)
    empty = evaluate_predictions([], [])
    assert empty["n"] == 0 and empty["accuracy"] is None


def test_trained_model_roundtrip_version_stamp_and_permissions(tmp_path):
    data = _data()
    train, test = data.split(0.7)
    encoder = _TwoFeatureEncoder().fit(train.features)
    estimator = LogisticRegression().fit(encoder.transform(train.features), train.labels)
    model = TrainedModel(
        name="demo",
        estimator=estimator,
        encoder=encoder,
        label_name="label",
        label_source="unit-test",
        trained_at="2026-01-01T00:00:00Z",
        data_hash=data.data_hash(),
        n_train=len(train),
        n_test=len(test),
    )
    model.metrics = model.evaluate(test)
    path = model.save(tmp_path / "nested" / "demo.joblib")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    loaded = TrainedModel.load(path)
    stamp = loaded.version_stamp()
    import sklearn

    assert stamp["model"] == "demo"
    assert stamp["sklearn_version"] == sklearn.__version__
    assert stamp["training_data_hash"] == data.data_hash()
    assert stamp["label_source"] == "unit-test"
    assert stamp["n_train"] == len(train)
    assert loaded.predict([{"x": 1, "y": 1}]) == model.predict([{"x": 1, "y": 1}])
    assert loaded.metrics["n"] == len(test)


def test_retrain_trigger_min_rows_and_idempotence(tmp_path):
    trigger = RetrainTrigger("demo", min_new_rows=10, state_dir=tmp_path)
    data = _data(labels=[index % 2 for index in range(20)])

    first = trigger.decide(data)
    assert first.should_retrain is True  # first run with ample data

    trigger.mark_trained(data)
    unchanged = trigger.decide(data)
    assert unchanged.should_retrain is False
    assert unchanged.reason == "data hash unchanged"

    # Same row count but different rows still counts as changed data *and*
    # zero new rows, so the min-new-rows gate holds it back.
    changed = TrainingData(
        features=[{"x": index, "y": 1} for index in range(20)],
        labels=[1] * 10 + [0] * 10,
        timestamps=data.timestamps,
    )
    gated = trigger.decide(changed)
    assert gated.should_retrain is False
    assert "need 10 more" in gated.reason

    larger = TrainingData(
        features=changed.features + [{"x": 99, "y": 0}] * 10,
        labels=changed.labels + [0] * 10,
        timestamps=changed.timestamps + ["2026-02-01T00:00:00Z"] * 10,
    )
    allowed = trigger.decide(larger)
    assert allowed.should_retrain is True
    assert allowed.new_rows == 10


def test_retrain_trigger_force_and_empty(tmp_path):
    trigger = RetrainTrigger("demo", min_new_rows=10, state_dir=tmp_path)
    empty = TrainingData(features=[], labels=[])
    assert trigger.decide(empty).should_retrain is False
    assert trigger.decide(empty, force=True).should_retrain is True
    trigger.mark(data_hash="abc", rows=3)
    assert json.loads((tmp_path / "retrain-demo.json").read_text())["rows"] == 3
