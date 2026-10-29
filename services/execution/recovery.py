"""Deterministic, bounded recovery decisions for execution failures.

Recovery is deliberately small: retry the same execution, try explicitly
configured fallback candidates, then either escalate or abstain. The module
does not perform retries, call providers, or introduce orchestration state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecoveryPolicy:
    """Bounds and actions for one execution."""

    max_retries: int = 0
    allow_fallback: bool = True
    allow_escalation: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not isinstance(self.allow_fallback, bool):
            raise ValueError("allow_fallback must be a boolean")
        if not isinstance(self.allow_escalation, bool):
            raise ValueError("allow_escalation must be a boolean")


@dataclass(frozen=True)
class RecoveryCandidate:
    """An explicitly configured alternate execution target."""

    model: str | None = None
    worker_id: str | None = None
    escalation: bool = False

    def __post_init__(self) -> None:
        if self.model is None and self.worker_id is None:
            raise ValueError("recovery candidate requires model or worker_id")
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise ValueError("model must be a non-empty string")
        if self.worker_id is not None and (
            not isinstance(self.worker_id, str) or not self.worker_id
        ):
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(self.escalation, bool):
            raise ValueError("escalation must be a boolean")


@dataclass(frozen=True)
class RecoveryDecision:
    """One observable recovery decision."""

    action: str
    attempt: int
    reason: str
    candidate: RecoveryCandidate | None = None

    def __post_init__(self) -> None:
        if self.action not in {"retry", "fallback", "escalate", "abstain"}:
            raise ValueError("unsupported recovery action")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise ValueError("attempt must be a non-negative integer")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if self.action in {"fallback", "escalate"} and self.candidate is None:
            raise ValueError(f"{self.action} requires a candidate")
        if self.action in {"retry", "abstain"} and self.candidate is not None:
            raise ValueError(f"{self.action} cannot have a candidate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "attempt": self.attempt,
            "reason": self.reason,
            "candidate": (
                {
                    "model": self.candidate.model,
                    "worker_id": self.candidate.worker_id,
                    "escalation": self.candidate.escalation,
                }
                if self.candidate is not None
                else None
            ),
        }


def next_decision(
    *,
    attempt: int,
    policy: RecoveryPolicy,
    candidates: tuple[RecoveryCandidate, ...] = (),
    tried_candidates: tuple[RecoveryCandidate, ...] = (),
    error: str = "execution failed",
) -> RecoveryDecision:
    """Select the next bounded recovery action deterministically."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    if not isinstance(error, str) or not error:
        raise ValueError("error must be a non-empty string")

    if attempt < policy.max_retries:
        return RecoveryDecision("retry", attempt + 1, error)

    tried = set(tried_candidates)
    if policy.allow_fallback:
        for candidate in candidates:
            if candidate in tried:
                continue
            if candidate.escalation:
                if policy.allow_escalation:
                    return RecoveryDecision("escalate", attempt + 1, error, candidate)
                continue
            return RecoveryDecision("fallback", attempt + 1, error, candidate)

    if policy.allow_escalation:
        for candidate in candidates:
            if candidate not in tried and candidate.escalation:
                return RecoveryDecision("escalate", attempt + 1, error, candidate)

    return RecoveryDecision("abstain", attempt + 1, error)
