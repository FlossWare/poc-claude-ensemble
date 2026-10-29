#!/usr/bin/env python3
"""Ground-truth gating for autonomous learning signals.

Operational telemetry describes whether a workflow ran as intended.
It is deliberately not evidence that the result was correct.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GroundTruthSource(str, Enum):
    """Sources that can establish a correctness outcome."""

    HUMAN_ACCEPTANCE = "human_acceptance"
    HUMAN_DISMISSAL = "human_dismissal"
    POST_MERGE_BUG = "post_merge_bug"
    REVERT = "revert"
    ATTRIBUTED_TEST_FAILURE = "attributed_test_failure"
    CONFIRMED_SECURITY_FINDING = "confirmed_security_finding"
    NONE = "none"


@dataclass(frozen=True)
class GroundTruth:
    """Externally observable evidence about correctness."""

    source: GroundTruthSource
    correct: bool
    evidence_id: Optional[str] = None
    evidence: Optional[str] = None


@dataclass(frozen=True)
class OperationalMetrics:
    """Telemetry about workflow execution, never a correctness label."""

    completed: bool = False
    findings_processed: int = 0
    target_findings_processed: int = 0
    execution_succeeded: bool = False


@dataclass(frozen=True)
class LearningSignal:
    """The only representation permitted to update correctness learning."""

    ground_truth: GroundTruth

    @property
    def correctness(self) -> bool:
        return self.ground_truth.correct


def build_learning_signal(
    ground_truth: Optional[GroundTruth],
    operational: OperationalMetrics,
) -> Optional[LearningSignal]:
    """Return a correctness signal only when external ground truth exists.

    Operational success, completion, or finding counts cannot manufacture a
    positive correctness label.
    """
    del operational  # Explicitly not part of correctness determination.
    if ground_truth is None or ground_truth.source is GroundTruthSource.NONE:
        return None
    return LearningSignal(ground_truth=ground_truth)
