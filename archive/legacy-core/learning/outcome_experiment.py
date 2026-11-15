"""Small, dependency-free evaluation of learning-enabled outcomes vs baseline."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OutcomeComparison:
    baseline_count: int
    learning_count: int
    paired_count: int
    baseline_quality: float
    learning_quality: float
    quality_delta: float
    baseline_cost: float
    learning_cost: float
    cost_delta: float
    baseline_latency_ms: float
    learning_latency_ms: float
    latency_delta_ms: float
    regressions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_count": self.baseline_count,
            "learning_count": self.learning_count,
            "paired_count": self.paired_count,
            "baseline_quality": self.baseline_quality,
            "learning_quality": self.learning_quality,
            "quality_delta": self.quality_delta,
            "baseline_cost": self.baseline_cost,
            "learning_cost": self.learning_cost,
            "cost_delta": self.cost_delta,
            "baseline_latency_ms": self.baseline_latency_ms,
            "learning_latency_ms": self.learning_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
            "regressions": self.regressions,
        }


def _index_by_task_id(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        task_id = str(record["task_id"])
        if task_id in indexed:
            raise ValueError(f"duplicate task_id: {task_id}")
        indexed[task_id] = record
    return indexed


def compare_outcomes(
    baseline: Iterable[Mapping[str, Any]],
    learning: Iterable[Mapping[str, Any]],
    *,
    regression_threshold: float = 0.10,
) -> OutcomeComparison:
    """Compare repeatable outcomes by task_id.

    Quality regressions are the only regressions counted. Cost and latency
    changes remain explicit deltas so callers can apply their own policy.
    Missing metrics are excluded from their respective averages.
    """
    baseline_by_id = _index_by_task_id(baseline)
    learning_by_id = _index_by_task_id(learning)
    paired_ids = sorted(baseline_by_id.keys() & learning_by_id.keys())

    def avg(records: list[Mapping[str, Any]], key: str) -> float:
        values = [
            float(record[key])
            for record in records
            if key in record and record[key] is not None
        ]
        return mean(values) if values else 0.0

    paired_baseline = [baseline_by_id[key] for key in paired_ids]
    paired_learning = [learning_by_id[key] for key in paired_ids]

    regressions = sum(
        1
        for base, learned in zip(paired_baseline, paired_learning)
        if "quality_score" in base
        and "quality_score" in learned
        and base["quality_score"] is not None
        and learned["quality_score"] is not None
        and float(learned["quality_score"])
        < float(base["quality_score"]) - regression_threshold
    )

    baseline_quality = avg(paired_baseline, "quality_score")
    learning_quality = avg(paired_learning, "quality_score")
    baseline_cost = avg(paired_baseline, "cost")
    learning_cost = avg(paired_learning, "cost")
    baseline_latency = avg(paired_baseline, "latency_ms")
    learning_latency = avg(paired_learning, "latency_ms")

    return OutcomeComparison(
        baseline_count=len(baseline_by_id),
        learning_count=len(learning_by_id),
        paired_count=len(paired_ids),
        baseline_quality=baseline_quality,
        learning_quality=learning_quality,
        quality_delta=learning_quality - baseline_quality,
        baseline_cost=baseline_cost,
        learning_cost=learning_cost,
        cost_delta=learning_cost - baseline_cost,
        baseline_latency_ms=baseline_latency,
        learning_latency_ms=learning_latency,
        latency_delta_ms=learning_latency - baseline_latency,
        regressions=regressions,
    )
