"""batch_size: items per ingest/embedding batch (arms 25, 50, 100, 250, 500).

Evidence comes from the operational telemetry that already exists:

* ``ops/embedding_batch.jsonl`` rows carry ``n_texts``, status, and latency;
* ``ops/gateway_forward.jsonl`` ``/ingestion/ingest`` rows are joined by
  timestamp to the bulk-ingest ledger, whose rows carry the chunk/vector count
  for that request.

The reward for a row is throughput relative to the observed median latency,
zeroed on a failed request.  Cost is wall-clock seconds, which is the resource
a larger batch spends; the cost is per-domain and never compared to dollar
costs from another domain.

``tools/bulk_ingest.py --learn-batch`` consumes the draw as this run's item
budget and records the run's own throughput/failure outcome afterwards.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_int_label


class BatchSize(Domain):
    name = "batch_size"
    arms = ("25", "50", "100", "250", "500")
    default_arm = "50"
    description = "Items per ingest/embedding batch"
    instrumentation = False
    outcome_source = "ops telemetry: embedding_batch rows and gateway ingest joined to bulk-ingest rows"
    value_kind = "int"

    @classmethod
    def normalize(cls, value: Any) -> str:
        return coerce_int_label(value, cls.arms, what="batch size")

    @classmethod
    def value(cls, arm: str) -> int:
        return int(arm)

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        return {"items": int(arm)}


def _latency_reward(latency_ms: Any, median_ms: float) -> float:
    """Throughput score in [0, 1] relative to the median observation."""
    try:
        latency = float(latency_ms)
    except (TypeError, ValueError):
        return 0.5  # unknown latency: neutral, not a win and not a failure
    if latency <= 0.0:
        return 1.0
    if median_ms <= 0.0:
        return 0.5
    return round(min(1.0, median_ms / latency), 6)


def derive_outcomes(ops_path=None, bulk_dir=None) -> List[Dict[str, Any]]:
    ops_root = Path(ops_path) if ops_path is not None else _ledger.ops_dir()
    embedding_rows = _ledger.read_jsonl(ops_root / "embedding_batch.jsonl")
    latencies = [float(row["latency_ms"]) for row in embedding_rows
                 if isinstance(row.get("latency_ms"), (int, float))
                 and not isinstance(row.get("latency_ms"), bool)]
    median_ms = statistics.median(latencies) if latencies else 0.0
    records: List[Dict[str, Any]] = []

    for index, row in enumerate(embedding_rows):
        size = row.get("n_texts")
        if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
            continue
        success = row.get("status") == 200
        reward = _latency_reward(row.get("latency_ms"), median_ms) if success else 0.0
        records.append({
            "arm": _ledger.nearest_arm(size, BatchSize.arms, floor=True),
            "outcome": reward,
            "cost": float(row.get("latency_ms") or 0.0) / 1000.0,
            "context": {"op": "embedding_batch"},
            "metadata": {
                "outcome_id": _ledger.stable_id("batch_size", "embedding_batch", index,
                                                row.get("ts"), size),
                "source": "embedding_batch",
                "n_texts": size,
                "status": row.get("status"),
                "latency_ms": row.get("latency_ms"),
                "median_latency_ms": median_ms,
            },
        })

    # Gateway /ingestion/ingest rows have no batch size of their own; the
    # bulk-ingest ledger row for the same moment carries chunks/vectors.
    bulk_root = Path(bulk_dir) if bulk_dir is not None else _ledger.bulk_ingest_dir()
    bulk_by_ts: Dict[int, List[Dict[str, Any]]] = {}
    for path in sorted(bulk_root.glob("*.jsonl")) if bulk_root.is_dir() else []:
        for row in _ledger.read_jsonl(path):
            stamp = _ledger.parse_ts(row.get("timestamp"))
            if stamp is not None:
                bulk_by_ts.setdefault(int(stamp), []).append(row)
    gateway_rows = _ledger.read_jsonl(ops_root / "gateway_forward.jsonl")
    for index, row in enumerate(gateway_rows):
        if row.get("path") != "/ingestion/ingest":
            continue
        stamp = _ledger.parse_ts(row.get("ts"))
        if stamp is None:
            continue
        joined = None
        for offset in (0, 1, -1, 2, -2):
            candidates = bulk_by_ts.get(int(stamp) + offset)
            if candidates:
                joined = candidates[0]
                break
        if joined is None:
            continue
        size = joined.get("chunks") or joined.get("vectors")
        if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
            continue
        success = row.get("status") == 200
        reward = _latency_reward(row.get("latency_ms"), median_ms) if success else 0.0
        records.append({
            "arm": _ledger.nearest_arm(size, BatchSize.arms, floor=True),
            "outcome": reward,
            "cost": float(row.get("latency_ms") or 0.0) / 1000.0,
            "context": {"op": "gateway_ingest"},
            "metadata": {
                "outcome_id": _ledger.stable_id("batch_size", "gateway_ingest", index,
                                                row.get("ts"), joined.get("content_sha256")),
                "source": "gateway_forward+bulk-ingest",
                "items": size,
                "status": row.get("status"),
                "latency_ms": row.get("latency_ms"),
                "median_latency_ms": median_ms,
            },
        })
    return records


def backfill(learner: Any, *, ops_path=None, bulk_dir=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, BatchSize.name, derive_outcomes(ops_path, bulk_dir), limit=limit)


__all__ = ["BatchSize", "backfill", "derive_outcomes"]
