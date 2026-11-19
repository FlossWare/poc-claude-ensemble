"""Task classifier: weak-label rules, TF-IDF features, fit/classify/persist."""

from __future__ import annotations

import pytest

from learning.train import _fixtures as fx
from learning.train import task_classifier as tc


def test_heuristic_labels_cover_every_task_type():
    assert tc.heuristic_label("please review this pull request") == "code-review"
    assert tc.heuristic_label("run the failing tests") == "qa"
    assert tc.heuristic_label("write documentation for the API") == "documentation"
    assert tc.heuristic_label("implement a retry loop") == "implementation"
    assert tc.heuristic_label("investigate why latency increased") == "research"
    assert tc.heuristic_label("hello there") == "other"


def test_extract_deduplicates_and_reports_counts():
    prompts = fx.prompt_rows()
    prompts.append(dict(prompts[0]))  # exact duplicate
    data, info = tc.extract_training_data(prompts, fx.review_episode_rows())
    assert info.duplicates_skipped == 1
    assert info.descriptions == len(data)
    assert info.label_counts is not None
    assert set(info.label_counts) <= set(tc.TASK_TYPES)
    assert data.label_source == tc.LABEL_SOURCE


def test_fit_classify_confidence_and_roundtrip(tmp_path):
    data, _ = tc.extract_training_data(fx.prompt_rows(), fx.review_episode_rows())
    model = tc.TaskClassifier.fit(data)

    assert model.model.n_train + model.model.n_test == len(data)
    assert model.model.metrics["train"]["accuracy"] is not None
    assert model.model.metrics["test"]["accuracy"] is not None

    task_type, confidence = model.classify("investigate why the gateway is slow")
    assert task_type in tc.TASK_TYPES
    assert 0.0 <= confidence <= 1.0
    batch = model.classify_batch(["write the readme", "run the tests"])
    assert len(batch) == 2
    assert all(label in tc.TASK_TYPES for label, _ in batch)

    top = model.top_features(limit=5)
    assert len(top) <= 5

    path = tmp_path / "task.joblib"
    model.save(str(path))
    loaded = tc.TaskClassifier.load(str(path))
    loaded_type, loaded_confidence = loaded.classify("investigate why the gateway is slow")
    assert loaded_type == task_type
    assert loaded_confidence == pytest.approx(confidence)
    stamp = loaded.model.version_stamp()
    assert stamp["label_source"] == "heuristic_keyword_rules"
    assert stamp["training_data_hash"] == data.data_hash()
    assert tc.classify_task(str(path), "run the tests")[0] in tc.TASK_TYPES


def test_insufficient_data_raises():
    data, _ = tc.extract_training_data([{"prompt": "hello there"}])
    with pytest.raises(tc.InsufficientTrainingData):
        tc.TaskClassifier.fit(data)


def test_real_format_prompt_stream(tmp_path):
    """Integration: train on a schema-faithful prompt stream and report metrics."""
    path = fx.write_jsonl(tmp_path / "prompts.jsonl", fx.prompt_rows())
    rows = tc.load_rows(str(path))
    data, info = tc.extract_training_data(rows)
    model = tc.TaskClassifier.fit(data)
    print({"info": info.as_dict(), "metrics": model.model.metrics})
    assert info.descriptions == 18
    assert model.model.metrics["test"]["n"] > 0
