"""Deterministic synthetic data builders for predictor and detector tests.

These mirror the real ledger shapes closely enough that tests exercise the same
production extraction and featurisation paths, including the failure-label
definition (stop reason, empty text, error field).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODELS: Tuple[str, ...] = ("alpha-1", "beta-1", "gamma-1", "delta-1")
TASK_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("code", "subject.py"),
    ("config", "subject.yml"),
    ("docs", "subject.md"),
)
BASE_COST = {"alpha-1": 0.02, "beta-1": 0.04, "gamma-1": 0.01, "delta-1": 0.06}
BASE_LATENCY = {"alpha-1": 2000.0, "beta-1": 5000.0, "gamma-1": 800.0, "delta-1": 9000.0}


def synthetic_ledger_rows(count: int = 180, seed: int = 7) -> List[Dict[str, Any]]:
    """Worker-result rows with learnable cost/latency/failure signals."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        model = MODELS[index % len(MODELS)]
        task_type, subject = TASK_SUFFIXES[(index // len(MODELS)) % len(TASK_SUFFIXES)]
        input_tokens = int(rng.integers(100, 2500))
        output_tokens = int(rng.integers(200, 1200))
        cost = max(
            0.0,
            BASE_COST[model] + input_tokens * 1e-5 + output_tokens * 2e-5 + rng.normal(0, 0.0002),
        )
        latency = max(
            1.0,
            BASE_LATENCY[model] + input_tokens * 0.5 + output_tokens * 0.2 + rng.normal(0, 100),
        )

        risk = 0.02 + (0.55 if model == "delta-1" else 0.0) + (0.15 if task_type == "config" else 0.0)
        failed = bool(rng.random() < risk * 0.5)
        rows.append(
            {
                "schema_version": 1,
                "producer": "predictor-test-fixture",
                "kind": "worker_result",
                "review_id": f"review-{index // len(MODELS):04d}",
                "subject": subject,
                "role": "worker",
                "recorded_at": (
                    datetime(2026, 11, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model": model,
                "text": "" if failed else "usable review text",
                "stop_reason": "max_tokens" if failed else "end_turn",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(float(latency), 3),
                "cost_usd": round(float(cost), 6),
            }
        )
    return rows


def write_ledger(path: Any, rows: Sequence[Dict[str, Any]]) -> Any:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def synthetic_telemetry_documents(
    gateway_rows: int = 240,
    embedding_rows: int = 120,
    seed: int = 11,
    with_outliers: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Interleaved normal telemetry plus optional obvious outliers.

    Returns ``(documents, outlier_timestamps)``.  Normal gateway latency is
    ~12 ms, normal embedding latency ~160 ms; per-source rolling features must
    keep those scales separate.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc)
    documents: List[Dict[str, Any]] = []
    for index in range(gateway_rows):
        timestamp = base + timedelta(seconds=5 * index)
        documents.append(
            {
                "ts": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "op": "gateway_forward",
                "path": "/memory/search",
                "method": "POST",
                "status": 200,
                "latency_ms": round(max(1.0, float(rng.normal(12, 2))), 1),
                "hops": 1,
            }
        )
    for index in range(embedding_rows):
        timestamp = base + timedelta(seconds=5 * index + 2)
        documents.append(
            {
                "ts": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "op": "embedding_batch",
                "n_texts": int(rng.integers(1, 8)),
                "mode": "document",
                "status": 200,
                "latency_ms": round(max(1.0, float(rng.normal(160, 10))), 1),
            }
        )

    outlier_timestamps: List[str] = []
    if with_outliers:
        tail = base + timedelta(hours=2)
        outliers = [
            {"op": "gateway_forward", "latency_ms": 6000.0, "status": 200},
            {"op": "gateway_forward", "latency_ms": 0.0, "status": 503},
            {"op": "embedding_batch", "latency_ms": 9000.0, "status": 200},
        ]
        for offset, outlier in enumerate(outliers):
            timestamp = tail + timedelta(seconds=offset)
            document = {
                "ts": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "path": "/memory/search",
                "method": "POST",
                "hops": 1,
                "n_texts": 2,
                "mode": "document",
            }
            document.update(outlier)
            documents.append(document)
            outlier_timestamps.append(document["ts"])
    return documents, outlier_timestamps
