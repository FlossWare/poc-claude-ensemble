"""Promotion gates for Target A.

Two groups, deliberately separated:

* **Evidence gates** depend only on corpus facts (how many rounds, outcomes,
  and holdout decisions exist).  They are checked *before* the search, so a
  corpus that cannot possibly justify a promotion never spends the budget.
* **Statistical gates** compare the candidate with the baseline on the holdout
  window: delta, block-bootstrap CI, cost, per-arm support and regression,
  unsupported fraction, coverage, and cooldown.

Every failed gate carries one of the design's run-outcome reasons; the first
failure becomes the run outcome.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from learning.ga.engine import EvalResult

MIN_HOLDOUT_DECISIONS = 300
MIN_TOTAL_OUTCOMES = 1000
MIN_ARM_OUTCOMES = 30
MIN_REVIEW_ROUNDS = 50
MIN_DELTA = 0.01
MIN_COVERAGE = 0.5
MAX_UNSUPPORTED_FRACTION = 0.01
MAX_COST_INCREASE = 0.10
MAX_ARM_SUCCESS_DROP = 0.05
COOLDOWN_DAYS = 7
COOLDOWN_NEW_OUTCOMES = 100
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260101


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "reason": self.reason}


def bootstrap_ci(
    deltas: Sequence[float],
    block_keys: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """95% block bootstrap CI of the mean paired delta, resampling by block."""
    paired = list(zip(deltas, block_keys))
    if not paired:
        return (0.0, 0.0)
    blocks: Dict[str, List[float]] = {}
    for delta, key in paired:
        blocks.setdefault(key, []).append(delta)
    keys = sorted(blocks)
    sums = [sum(blocks[key]) for key in keys]
    counts = [len(blocks[key]) for key in keys]
    rng = random.Random(seed)
    n = len(keys)
    means: List[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in range(n):
            index = rng.randrange(n)
            total += sums[index]
            count += counts[index]
        means.append(total / count if count else 0.0)
    means.sort()
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(0.975 * (resamples - 1))]
    return (low, high)


def evidence_gates(
    *,
    review_rounds: int,
    total_outcomes: int,
    holdout_decisions: int,
) -> List[GateResult]:
    return [
        GateResult(
            "review_rounds",
            review_rounds >= MIN_REVIEW_ROUNDS,
            f"{review_rounds} complete review rounds (need >= {MIN_REVIEW_ROUNDS})",
            "insufficient_evidence",
        ),
        GateResult(
            "total_outcomes",
            total_outcomes >= MIN_TOTAL_OUTCOMES,
            f"{total_outcomes} turn outcomes (need >= {MIN_TOTAL_OUTCOMES})",
            "insufficient_evidence",
        ),
        GateResult(
            "holdout_decisions",
            holdout_decisions >= MIN_HOLDOUT_DECISIONS,
            f"{holdout_decisions} holdout decisions (need >= {MIN_HOLDOUT_DECISIONS})",
            "insufficient_evidence",
        ),
    ]


def statistical_gates(
    baseline: EvalResult,
    candidate: EvalResult,
    *,
    min_support: int,
    cooldown_ok: bool = True,
    cooldown_detail: str = "no recent promotion",
) -> Tuple[List[GateResult], Tuple[float, float]]:
    """Compare candidate with baseline on the holdout window."""
    gates: List[GateResult] = []

    unsupported_arms = [
        arm
        for arm, stats in candidate.per_arm.items()
        if stats["decisions"] > 0 and stats["logged_rows"] < MIN_ARM_OUTCOMES
    ]
    gates.append(
        GateResult(
            "per_arm_support",
            not unsupported_arms,
            "selected arms without enough holdout outcomes: "
            + (", ".join(sorted(unsupported_arms)) if unsupported_arms else "none"),
            "insufficient_support",
        )
    )

    gates.append(
        GateResult(
            "unsupported_fraction",
            candidate.unsupported_fraction <= MAX_UNSUPPORTED_FRACTION,
            f"unsupported fraction {candidate.unsupported_fraction:.4f} "
            f"(max {MAX_UNSUPPORTED_FRACTION})",
            "insufficient_support",
        )
    )

    gates.append(
        GateResult(
            "realized_coverage",
            candidate.coverage >= MIN_COVERAGE,
            f"realized coverage {candidate.coverage:.4f} (min {MIN_COVERAGE})",
            "low_coverage",
        )
    )

    delta = candidate.fitness - baseline.fitness
    gates.append(
        GateResult(
            "delta",
            delta >= MIN_DELTA,
            f"holdout delta {delta:.6f} (min {MIN_DELTA})",
            "delta_below_threshold",
        )
    )

    length = min(len(candidate.contributions), len(baseline.contributions))
    diffs = [
        candidate.contributions[index] - baseline.contributions[index] for index in range(length)
    ]
    keys = list(candidate.block_keys[:length]) or ["all"]
    low, high = bootstrap_ci(diffs, keys) if diffs else (0.0, 0.0)
    gates.append(
        GateResult(
            "bootstrap_ci",
            bool(diffs) and low > 0.0,
            f"95% block-bootstrap CI [{low:.6f}, {high:.6f}] (need lower > 0)",
            "ci_not_positive",
        )
    )

    if baseline.cost_hat > 0.0:
        cost_ok = candidate.cost_hat <= baseline.cost_hat * (1.0 + MAX_COST_INCREASE)
        cost_detail = (
            f"holdout cost {candidate.cost_hat:.6f} vs baseline {baseline.cost_hat:.6f} "
            f"(max +{MAX_COST_INCREASE:.0%})"
        )
    else:
        cost_ok = candidate.cost_hat <= 0.0
        cost_detail = f"baseline cost is zero; candidate cost {candidate.cost_hat:.6f}"
    gates.append(GateResult("cost_regression", cost_ok, cost_detail, "cost_regression"))

    floor = baseline.succ_hat - MAX_ARM_SUCCESS_DROP
    regressed = [
        arm
        for arm, stats in candidate.per_arm.items()
        if stats["decisions"] > 0
        and stats["logged_success"] is not None
        and stats["logged_success"] < floor
    ]
    gates.append(
        GateResult(
            "arm_regression",
            not regressed,
            "arms more than "
            f"{MAX_ARM_SUCCESS_DROP} below baseline success: "
            + (", ".join(sorted(regressed)) if regressed else "none"),
            "arm_regression",
        )
    )

    gates.append(GateResult("cooldown", cooldown_ok, cooldown_detail, "cooldown"))
    return gates, (low, high)


def first_failure(gates: Iterable[GateResult]) -> Optional[GateResult]:
    for gate in gates:
        if not gate.passed:
            return gate
    return None


def outcome_for(gates: Iterable[GateResult], *, report_only: bool) -> str:
    failure = first_failure(gates)
    if failure is not None:
        return f"not_promoted:{failure.reason}"
    return "not_promoted:report_only" if report_only else "promoted"


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "COOLDOWN_DAYS",
    "COOLDOWN_NEW_OUTCOMES",
    "GateResult",
    "MAX_ARM_SUCCESS_DROP",
    "MAX_COST_INCREASE",
    "MAX_UNSUPPORTED_FRACTION",
    "MIN_ARM_OUTCOMES",
    "MIN_COVERAGE",
    "MIN_DELTA",
    "MIN_HOLDOUT_DECISIONS",
    "MIN_REVIEW_ROUNDS",
    "MIN_TOTAL_OUTCOMES",
    "bootstrap_ci",
    "evidence_gates",
    "first_failure",
    "outcome_for",
    "statistical_gates",
]
