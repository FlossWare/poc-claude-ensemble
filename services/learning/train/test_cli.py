"""CLI: status, retrain gating, artifacts, predictions, honest skips."""

from __future__ import annotations

import json

import pytest

from learning.train import _fixtures as fx
from learning.train import cli, paths


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    review = fx.write_jsonl(tmp_path / "reviews.jsonl", fx.review_rows(reviews=12))
    prompts = fx.write_jsonl(tmp_path / "prompts.jsonl", fx.prompt_rows())
    journal = fx.write_jsonl(tmp_path / "injections.jsonl", fx.injection_rows(labeled=True, count=40))
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_TRAIN_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", str(review))
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_PROMPT_LEDGER", str(prompts))
    monkeypatch.setenv("FLOSSWARE_CE_RAG_JOURNAL", str(journal))
    return tmp_path


def test_status_reports_absent_models_before_training(isolated, capsys):
    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    for name in cli.MODELS:
        assert name in output
    assert "artifact: absent" in output
    assert "training data:" in output


def test_status_json_is_machine_readable(isolated, capsys):
    assert cli.main(["status", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["artifact_dir"] == str(isolated / "artifacts")
    assert set(report["models"]) == set(cli.MODELS)
    assert report["models"][cli.MODEL_RETRIEVAL]["data"]["labeled_rows"] == 40


def test_train_all_then_skip_on_unchanged_hash(isolated, capsys):
    assert cli.main(["train"]) == 0
    first = capsys.readouterr().out
    for name in cli.MODELS:
        assert paths.artifact_path(name).exists(), name
    assert "trained:" in first
    assert "version stamp:" in first

    assert cli.main(["train"]) == 0
    second = capsys.readouterr().out
    assert second.count("data hash unchanged") >= 2  # quality + task
    assert "trained:" not in second


def test_train_skips_retrieval_when_labels_are_null(isolated, capsys, monkeypatch):
    journal = fx.write_jsonl(isolated / "null-injections.jsonl", fx.injection_rows(labeled=False))
    monkeypatch.setenv("FLOSSWARE_CE_RAG_JOURNAL", str(journal))
    assert cli.main(["train", "--model", "retrieval"]) == 0
    output = capsys.readouterr().out
    assert "insufficient labeled data" in output
    assert not paths.artifact_path(cli.MODEL_RETRIEVAL).exists()


def test_predict_quality_and_task(isolated, capsys):
    assert cli.main(["train", "--model", "quality"]) == 0
    assert cli.main(["train", "--model", "task"]) == 0
    capsys.readouterr()

    features = json.dumps(
        {
            "model_id": "rits/zai-org/glm-5-3",
            "input_tokens": 500,
            "output_tokens": 800,
            "latency_ms": 1000,
            "cost_usd": 0.01,
            "stop_reason": "end_turn",
            "text_length": 4000,
        }
    )
    assert cli.main(["predict", "quality", features]) == 0
    quality = json.loads(capsys.readouterr().out)
    assert 0.0 <= quality["quality"] <= 1.0

    assert cli.main(["predict", "task", "investigate why the gateway is slow"]) == 0
    task = json.loads(capsys.readouterr().out)
    assert task["task_type"]
    assert 0.0 <= task["confidence"] <= 1.0


def test_predict_without_artifact_is_usage_error(isolated, capsys):
    assert cli.main(["predict", "retrieval", "{}"]) == 2
    assert "no trained artifact" in capsys.readouterr().err


def test_unknown_model_is_usage_error(isolated):
    assert cli.main(["train", "--model", "nope"]) == 2
    assert cli.main(["predict", "nope", "{}"]) == 2
