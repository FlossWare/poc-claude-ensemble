"""End-to-end CLI tests through ``python -m learning.train``'s dispatcher."""

from __future__ import annotations

import json

import pytest

from learning.train.cli import main
from learning.train.predictors import _fixtures


@pytest.fixture()
def environment(tmp_path):
    ledger = _fixtures.write_ledger(
        tmp_path / "reviews.jsonl",
        _fixtures.synthetic_ledger_rows(count=160, seed=5),
    )
    ops = tmp_path / "ops"
    ops.mkdir()
    documents, outlier_timestamps = _fixtures.synthetic_telemetry_documents(
        gateway_rows=120, embedding_rows=60, with_outliers=True
    )
    for source in ("gateway_forward", "embedding_batch"):
        with open(ops / f"{source}.jsonl", "w", encoding="utf-8") as handle:
            for document in documents:
                if document["op"] == source:
                    handle.write(json.dumps(document) + "\n")
    return {
        "ledger": ledger,
        "ops": ops,
        "model_dir": tmp_path / "models",
        "outlier_timestamps": outlier_timestamps,
    }


def test_train_status_and_predict_round_trip(environment, capsys):
    model_dir = environment["model_dir"]
    assert (
        main(
            [
                "predictors",
                "train",
                "--ledger",
                str(environment["ledger"]),
                "--ops-dir",
                str(environment["ops"]),
                "--model-dir",
                str(model_dir),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Routing predictor training" in output
    assert "worker results: 160" in output
    for kind in ("cost", "failure", "latency", "anomaly"):
        assert (model_dir / f"{kind}.pkl").exists()

    assert main(["predictors", "status", "--model-dir", str(model_dir)]) == 0
    output = capsys.readouterr().out
    assert output.count("READY") == 4
    assert "MISSING" not in output

    payload = json.dumps(
        {"model": "alpha-1", "task_type": "code", "input_tokens": 800, "text_length": 5000}
    )
    assert (
        main(
            [
                "predictors",
                "predict",
                "cost",
                payload,
                "--model-dir",
                str(model_dir),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "cost predictor" in output
    assert "ignored" in output and "text_length" in output

    assert (
        main(
            [
                "predictors",
                "predict",
                "latency",
                payload,
                "--model-dir",
                str(model_dir),
                "--json",
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["kind"] == "latency"
    assert document["unit"] == "ms"
    assert document["prediction"] > 0
    assert document["ignored_features"] == ["text_length"]


def test_status_reports_missing_artifacts(tmp_path, capsys):
    assert main(["predictors", "status", "--model-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert output.count("MISSING") == 4


def test_predict_without_artifact_exits_data_error(tmp_path, capsys):
    payload = json.dumps({"model": "alpha-1", "task_type": "code"})
    assert (
        main(
            [
                "predictors",
                "predict",
                "cost",
                payload,
                "--model-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "artifact not found" in error
    assert "predictors train" in error


def test_invalid_json_and_missing_model_exit_config_error(environment, capsys):
    model_dir = environment["model_dir"]
    main(
        [
            "predictors",
            "train",
            "--ledger",
            str(environment["ledger"]),
            "--ops-dir",
            str(environment["ops"]),
            "--model-dir",
            str(model_dir),
        ]
    )
    capsys.readouterr()

    assert main(["predictors", "predict", "cost", "{not json", "--model-dir", str(model_dir)]) == 2
    assert "not valid JSON" in capsys.readouterr().err

    assert (
        main(
            [
                "predictors",
                "predict",
                "cost",
                json.dumps({"task_type": "code"}),
                "--model-dir",
                str(model_dir),
            ]
        )
        == 2
    )
    assert "non-empty 'model'" in capsys.readouterr().err


def test_anomaly_check_json_and_exit_code(environment, capsys):
    code = main(
        [
            "anomaly",
            "check",
            "--ops-dir",
            str(environment["ops"]),
            "--window",
            "200",
            "--contamination",
            "0.02",
            "--json",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert code in (0, 3)
    assert document["rows_loaded"] == 183
    assert document["checked_window"] == 183
    assert document["source_counts"] == {"embedding_batch": 61, "gateway_forward": 122}
    flagged_timestamps = {finding["timestamp"] for finding in document["findings"]}
    for timestamp in environment["outlier_timestamps"]:
        assert timestamp in flagged_timestamps
    assert code == 3


def test_anomaly_check_with_saved_model(environment, capsys):
    model_dir = environment["model_dir"]
    main(
        [
            "predictors",
            "train",
            "--ledger",
            str(environment["ledger"]),
            "--ops-dir",
            str(environment["ops"]),
            "--model-dir",
            str(model_dir),
        ]
    )
    capsys.readouterr()

    code = main(
        [
            "anomaly",
            "check",
            "--ops-dir",
            str(environment["ops"]),
            "--use-model",
            "--model-dir",
            str(model_dir),
            "--json",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert document["mode"] == "saved-model"
    assert code in (0, 3)


def test_anomaly_check_without_telemetry_exits_data_error(tmp_path, capsys):
    assert main(["anomaly", "check", "--ops-dir", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert "need at least" in error


def test_predictors_train_without_ledger_exits_data_error(tmp_path, capsys):
    model_dir = tmp_path / "models"
    assert (
        main(
            [
                "predictors",
                "train",
                "--ledger",
                str(tmp_path / "missing.jsonl"),
                "--model-dir",
                str(model_dir),
                "--skip-anomaly",
            ]
        )
        == 1
    )
    assert "0 usable worker rows" in capsys.readouterr().err


def test_argument_validation():
    with pytest.raises(SystemExit):
        main(["anomaly", "check", "--window", "0"])
    with pytest.raises(SystemExit):
        main(["anomaly", "check", "--contamination", "0.9"])
