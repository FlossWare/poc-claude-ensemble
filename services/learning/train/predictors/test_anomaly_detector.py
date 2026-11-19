"""Anomaly detector tests: normal traffic, injected outliers, persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from learning.train.predictors import _fixtures
from learning.train.predictors.anomaly_detector import (
    AnomalyDetector,
    AnomalyFinding,
    MIN_ROWS,
    TelemetryRow,
    featurize,
    load_telemetry,
)
from learning.train.predictors.base import InsufficientDataError


def _telemetry_rows(documents):
    rows = []
    for document in documents:
        rows.append(
            TelemetryRow(
                timestamp=document["ts"],
                source=document["op"],
                latency_ms=float(document["latency_ms"]),
                status=int(document["status"]),
                raw=document,
            )
        )
    return rows


def test_load_telemetry_sorts_skips_malformed_and_handles_missing(tmp_path):
    documents, _ = _fixtures.synthetic_telemetry_documents(gateway_rows=12, embedding_rows=8)
    gateway = tmp_path / "gateway_forward.jsonl"
    embedding = tmp_path / "embedding_batch.jsonl"
    with open(gateway, "w", encoding="utf-8") as handle:
        for document in documents:
            if document["op"] == "gateway_forward":
                handle.write(json.dumps(document) + "\n")
        handle.write("not json\n")
        handle.write('{"ts": "garbage", "op": "gateway_forward"}\n')
    with open(embedding, "w", encoding="utf-8") as handle:
        for document in documents:
            if document["op"] == "embedding_batch":
                handle.write(json.dumps(document) + "\n")

    # A source file that does not exist must not fail the load.
    rows = load_telemetry(tmp_path, sources=("gateway_forward", "embedding_batch", "missing"))
    assert len(rows) == 20
    timestamps = [row.timestamp for row in rows]
    assert timestamps == sorted(timestamps)
    assert {row.source for row in rows} == {"gateway_forward", "embedding_batch"}

    limited = load_telemetry(tmp_path, sources=("gateway_forward",), per_source_limit=5)
    assert len(limited) == 5


def test_featurize_keeps_per_source_rolling_scales():
    documents, _ = _fixtures.synthetic_telemetry_documents(gateway_rows=30, embedding_rows=30)
    rows = sorted(_telemetry_rows(documents), key=lambda row: row.timestamp)
    window = featurize(rows, sources=("gateway_forward", "embedding_batch"), window=10)
    columns = {name: index for index, name in enumerate(window.feature_names)}
    gateway_index = next(
        index for index, row in enumerate(window.rows) if row.source == "gateway_forward"
    )
    embedding_index = next(
        index for index, row in enumerate(window.rows) if row.source == "embedding_batch"
    )
    gateway_mean = window.matrix[gateway_index, columns["latency_roll_mean"]]
    embedding_mean = window.matrix[embedding_index, columns["latency_roll_mean"]]
    assert gateway_mean < 100
    assert embedding_mean > 100


def test_normal_traffic_produces_few_findings():
    documents, _ = _fixtures.synthetic_telemetry_documents(gateway_rows=240, embedding_rows=120)
    detector = AnomalyDetector(contamination=0.01)
    detector.fit(_telemetry_rows(documents))

    findings = detector.detect_anomalies(120)
    assert len(findings) <= 0.05 * 120, findings
    for finding in findings:
        assert isinstance(finding, AnomalyFinding)
        assert isinstance(finding.timestamp, str)
        assert isinstance(finding.metric, str)
        assert isinstance(finding.score, float)


def test_injected_outliers_are_flagged_with_interpretable_metrics():
    documents, outlier_timestamps = _fixtures.synthetic_telemetry_documents(
        gateway_rows=240, embedding_rows=120, with_outliers=True
    )
    detector = AnomalyDetector(contamination=0.02)
    detector.fit(_telemetry_rows(documents))

    details = detector.detect_details(500)
    found = {detail.timestamp: detail for detail in details}
    for timestamp in outlier_timestamps:
        assert timestamp in found, (timestamp, sorted(found))

    status_outlier = found[outlier_timestamps[1]]
    assert status_outlier.metric in {"status", "is_error", "error_rate_roll"}
    assert status_outlier.status == 503

    latency_outlier = found[outlier_timestamps[0]]
    assert latency_outlier.metric in {"latency_ms", "latency_roll_mean", "latency_roll_std"}
    assert latency_outlier.latency_ms == 6000.0

    # Public contract: a list of (timestamp, metric, score) tuples.
    findings = detector.detect_anomalies(500)
    assert findings == sorted(findings, key=lambda finding: finding.score, reverse=True)
    assert all(len(finding) == 3 for finding in findings)


def test_save_load_round_trip_preserves_findings(tmp_path):
    documents, _ = _fixtures.synthetic_telemetry_documents(
        gateway_rows=120, embedding_rows=60, with_outliers=True
    )
    detector = AnomalyDetector(contamination=0.02)
    detector.fit(_telemetry_rows(documents))
    expected = detector.detect_anomalies(100)

    artifact = detector.save(tmp_path / "anomaly.pkl")
    assert artifact.with_suffix(".json").exists()
    loaded = AnomalyDetector.load(artifact)
    actual = loaded.detect_anomalies(100)
    assert actual == expected

    status = AnomalyDetector.status(artifact)
    assert status["available"] is True
    assert status["samples"] == 183
    assert status["metrics"]["flagged_in_training"] == detector.metadata_["flagged_in_training"]


def test_explicit_window_rows_are_scored(tmp_path):
    documents, outlier_timestamps = _fixtures.synthetic_telemetry_documents(
        gateway_rows=120, embedding_rows=60
    )
    detector = AnomalyDetector(contamination=0.02)
    detector.fit(_telemetry_rows(documents))

    outlier_document = {
        "ts": "2026-11-20T00:00:00Z",
        "op": "gateway_forward",
        "latency_ms": 9000.0,
        "status": 500,
    }
    findings = detector.detect_anomalies([outlier_document])
    assert any(finding.timestamp == outlier_document["ts"] for finding in findings)


def test_too_few_rows_is_an_error():
    documents, _ = _fixtures.synthetic_telemetry_documents(gateway_rows=4, embedding_rows=2)
    detector = AnomalyDetector()
    with pytest.raises(InsufficientDataError):
        detector.fit(_telemetry_rows(documents))
    assert MIN_ROWS == 10


def test_untrained_detector_raises():
    detector = AnomalyDetector()
    with pytest.raises(Exception, match="not trained"):
        detector.detect_anomalies()
