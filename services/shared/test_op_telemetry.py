#!/usr/bin/env python3
"""Tests for the best-effort ops telemetry recorder."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from shared.op_telemetry import record


@pytest.fixture()
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR", str(tmp_path / "ops"))
    return tmp_path / "ops"


def _rows(ledger_dir, name):
    path = Path(ledger_dir) / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_record_appends_jsonl_row(ledger_dir):
    assert record("gateway_forward", path="/api/v1/memory/search", method="POST",
                  status=200, latency_ms=12.5, hops=1)
    rows = _rows(ledger_dir, "gateway_forward")
    assert len(rows) == 1
    row = rows[0]
    assert row["op"] == "gateway_forward"
    assert row["path"] == "/api/v1/memory/search"
    assert row["status"] == 200 and row["latency_ms"] == 12.5
    assert "ts" in row and row["ts"].endswith("Z")


def test_ledger_file_permissions(ledger_dir):
    record("embedding_batch", n_texts=8, mode="document", status=200, latency_ms=42.0)
    path = Path(ledger_dir) / "embedding_batch.jsonl"
    assert path.exists()
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_oversize_row_is_dropped_not_truncated(ledger_dir):
    huge = "x" * 20_000
    assert record("gateway_forward", path=huge) is False
    assert _rows(ledger_dir, "gateway_forward") == []


def test_non_serializable_value_never_raises(ledger_dir):
    assert record("gateway_forward", weird=object()) is False  # not JSON-serializable


def test_nan_rejected(ledger_dir):
    assert record("gateway_forward", latency_ms=float("nan")) is False


def test_short_name_cap(ledger_dir):
    record("n" * 100, status=200)
    rows = _rows(ledger_dir, "n" * 64)
    assert len(rows) == 1 and rows[0]["op"] == "n" * 64
