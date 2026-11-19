"""graph_traversal: traversal depth/limits per query (signal capture).

The graph service records one ``ops/graph_traverse.jsonl`` row per traversal:
requested depth, direction, result/node/edge counts, latency, and whether the
request timed out or failed.  This domain turns those rows into per-depth
quality evidence.  Instrumentation-only: no traversal parameter is drawn from
it yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_int_label


class GraphTraversal(Domain):
    name = "graph_traversal"
    arms = ("1", "2", "3", "5")
    default_arm = "2"
    description = "Graph traversal depth per query (instrumented, not yet tuned)"
    instrumentation = True
    outcome_source = "ops graph_traverse telemetry: depth, result counts, latency, timeouts"
    value_kind = "int"

    @classmethod
    def normalize(cls, value: Any) -> str:
        return coerce_int_label(value, cls.arms, what="traversal depth")

    @classmethod
    def value(cls, arm: str) -> int:
        return int(arm)

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        return {"max_depth": int(arm)}


def derive_outcomes(ops_path=None) -> List[Dict[str, Any]]:
    root = Path(ops_path) if ops_path is not None else _ledger.ops_dir()
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(_ledger.read_jsonl(root / "graph_traverse.jsonl")):
        depth = row.get("depth")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            continue
        status = row.get("status")
        timed_out = bool(row.get("timeout"))
        results = row.get("result_count")
        if timed_out or status != "ok":
            reward = 0.0
        elif isinstance(results, (int, float)) and not isinstance(results, bool) and results > 0:
            reward = 1.0
        else:
            reward = 0.25  # completed, but the traversal found nothing
        records.append({
            "arm": _ledger.nearest_arm(depth, GraphTraversal.arms, floor=True),
            "outcome": reward,
            "cost": float(row.get("latency_ms") or 0.0) / 1000.0,
            "context": {"direction": str(row.get("direction") or "out")},
            "metadata": {
                "outcome_id": _ledger.stable_id("graph_traversal", index, row.get("ts"),
                                                row.get("start")),
                "depth": depth,
                "status": status,
                "timeout": timed_out,
                "result_count": results,
                "node_count": row.get("node_count"),
                "edge_count": row.get("edge_count"),
                "latency_ms": row.get("latency_ms"),
            },
        })
    return records


def backfill(learner: Any, *, ops_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, GraphTraversal.name, derive_outcomes(ops_path), limit=limit)


__all__ = ["GraphTraversal", "backfill", "derive_outcomes"]
