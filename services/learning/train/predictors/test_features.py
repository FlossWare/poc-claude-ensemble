"""Feature extraction and encoding tests against a fixture review ledger."""

from __future__ import annotations

import json

import numpy as np
import pytest

from learning.train.predictors.features import (
    FeatureEncoder,
    UNKNOWN_CATEGORY,
    derive_task_type,
    extract_features,
    is_success,
    summarize,
    worker_rows,
)


@pytest.fixture()
def ledger(tmp_path):
    rows = [
        {
            "kind": "worker_result",
            "review_id": "r1",
            "model": "model-a",
            "subject": "/repo/service.py",
            "recorded_at": "2026-11-01T10:00:00Z",
            "text": "hello",
            "stop_reason": "end_turn",
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_ms": 1000,
            "cost_usd": 0.01,
        },
        {
            "kind": "worker_result",
            "review_id": "r1",
            "model": "model-b",
            "subject": "/repo/config.yml",
            "recorded_at": "2026-11-01T10:00:01Z",
            "text": "",
            "stop_reason": "max_tokens",
            "input_tokens": 200,
            "output_tokens": 4000,
            "latency_ms": 2000,
            "cost_usd": 0.02,
        },
        {
            "kind": "worker_result",
            "review_id": "r2",
            "model": "model-a",
            "subject": "/repo/template.j2",
            "recorded_at": "2026-11-01T10:00:02Z",
            "text": "   ",
            "stop_reason": "end_turn",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0,
        },
        {
            "kind": "worker_result",
            "review_id": "r2",
            "model": "model-c",
            "subject": "/repo/run.sh",
            "recorded_at": "2026-11-01T10:00:03Z",
            "text": "ok",
            "stop_reason": "end_turn",
            "error": "timeout",
            "input_tokens": 300,
            "output_tokens": 100,
            "latency_ms": 3000,
            "cost_usd": 0.03,
        },
        {
            "kind": "worker_result",
            "review_id": "r3",
            "model": "model-a",
            "subject": "/repo/truncated.md",
            "recorded_at": "2026-11-01T10:00:04Z",
            "text": "abc",
            "truncated": True,
            "original_length": 500,
            "stop_reason": "end_turn",
            "input_tokens": 50,
            "output_tokens": 60,
            "latency_ms": 500,
            "cost_usd": 0.005,
        },
        {
            "kind": "arbiter_decision",
            "review_id": "r3",
            "model": "arbiter-z",
            "role": "arbiter",
            "latency_ms": 9999,
        },
    ]
    path = tmp_path / "reviews.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.write("{this is not json\n")
        handle.write("\n")
    return path


def test_worker_rows_ignores_arbiter_and_malformed(ledger):
    rows = worker_rows(ledger)
    assert len(rows) == 5
    assert {row["kind"] for row in rows} == {"worker_result"}


def test_extract_features_fields_and_derived_values(ledger):
    features = extract_features(path=ledger)
    assert len(features) == 5

    first = features[0]
    assert first.review_id == "r1"
    assert first.model_id == "model-a"
    assert first.task_type == "code"
    assert first.input_tokens == 100
    assert first.output_tokens == 50
    assert first.text_length == len("hello")
    assert first.stop_reason == "end_turn"
    assert first.latency_ms == 1000.0
    assert first.cost_usd == 0.01
    assert first.tokens_per_second == pytest.approx(50.0)
    assert first.cost_per_token == pytest.approx(0.01 / 150.0)
    assert first.success is True

    truncated = next(row for row in features if row.subject.endswith(".md"))
    assert truncated.text_length == 500  # pre-truncation length wins
    assert truncated.success is True


def test_task_type_derivation_and_success_rules(ledger):
    features = extract_features(path=ledger)
    by_model = {(row.model_id, row.stop_reason): row for row in features}
    assert derive_task_type("x.yaml") == "config"
    assert derive_task_type("x.j2") == "template"
    assert derive_task_type("x.sh") == "shell"
    assert derive_task_type("x.unknown_ext") == "other"
    assert derive_task_type(None, explicit="Code-Review") == "code_review"

    empty_text = features[2]
    assert empty_text.task_type == "template"
    assert empty_text.success is False  # whitespace-only output

    errored = features[3]
    assert errored.task_type == "shell"
    assert errored.success is False  # explicit error field

    assert is_success({"stop_reason": "end_turn", "text": "x", "error": None}) is True
    assert is_success({"stop_reason": "end_turn", "text": "x", "error": "boom"}) is False
    assert is_success({"stop_reason": "max_tokens", "text": "x"}) is False


def test_summary_counts_and_rates(ledger):
    features = extract_features(path=ledger)
    summary = summarize(features)
    assert summary["samples"] == 5
    assert summary["review_episodes"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 3
    assert summary["positive_rate"] == pytest.approx(0.6)
    assert summary["models"] == ["model-a", "model-b", "model-c"]
    assert summary["task_types"] == ["code", "config", "docs", "shell", "template"]


def test_encoder_one_hot_unknowns_and_round_trip(ledger):
    features = extract_features(path=ledger)
    encoder = FeatureEncoder.fit(features)
    names = encoder.feature_names
    assert names[-1] == "input_tokens"
    assert f"model={UNKNOWN_CATEGORY}" in names
    assert f"task_type={UNKNOWN_CATEGORY}" in names

    vector = encoder.encode("model-a", {"task_type": "code", "input_tokens": 123})
    assert vector.dtype == np.float64
    assert vector[names.index("model=model-a")] == 1.0
    assert vector[names.index("task_type=code")] == 1.0
    assert vector[names.index("input_tokens")] == 123.0
    assert vector[names.index(f"model={UNKNOWN_CATEGORY}")] == 0.0

    unseen = encoder.encode("never-seen", {"subject": "file.weird", "input_tokens": 5})
    assert unseen[names.index(f"model={UNKNOWN_CATEGORY}")] == 1.0
    assert unseen[names.index(f"task_type={UNKNOWN_CATEGORY}")] == 1.0

    restored = FeatureEncoder.from_dict(encoder.to_dict())
    assert restored.feature_names == names
    assert np.array_equal(restored.encode("model-a", {"task_type": "code"}), encoder.encode("model-a", {"task_type": "code"}))


def test_encoder_rejects_bad_numeric_and_non_mapping(ledger):
    encoder = FeatureEncoder.fit(extract_features(path=ledger))
    with pytest.raises(ValueError, match="must be a number"):
        encoder.encode("model-a", {"input_tokens": "lots"})
    with pytest.raises(ValueError, match="JSON object"):
        encoder.encode("model-a", ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id is required"):
        encoder.encode("", {})
