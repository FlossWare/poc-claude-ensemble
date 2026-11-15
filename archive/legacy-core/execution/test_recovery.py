from __future__ import annotations

import pytest

from execution.recovery import RecoveryCandidate, RecoveryPolicy, next_decision


def test_retry_is_bounded_and_precedes_fallback() -> None:
    policy = RecoveryPolicy(max_retries=2)
    candidate = RecoveryCandidate(model="sonnet")

    first = next_decision(attempt=0, policy=policy, candidates=(candidate,))
    second = next_decision(attempt=1, policy=policy, candidates=(candidate,))
    third = next_decision(attempt=2, policy=policy, candidates=(candidate,))

    assert first.to_dict()["action"] == "retry"
    assert second.to_dict()["action"] == "retry"
    assert third.to_dict()["action"] == "fallback"
    assert third.candidate == candidate


def test_fallback_candidates_are_deterministic_and_not_reused() -> None:
    first = RecoveryCandidate(model="sonnet")
    second = RecoveryCandidate(model="opus")
    policy = RecoveryPolicy()

    decision = next_decision(
        attempt=0,
        policy=policy,
        candidates=(first, second),
        tried_candidates=(first,),
    )

    assert decision.action == "fallback"
    assert decision.candidate == second


def test_escalation_requires_policy_and_candidate() -> None:
    candidate = RecoveryCandidate(model="opus", escalation=True)

    allowed = next_decision(
        attempt=0,
        policy=RecoveryPolicy(allow_fallback=True, allow_escalation=True),
        candidates=(candidate,),
    )
    denied = next_decision(
        attempt=0,
        policy=RecoveryPolicy(allow_fallback=True, allow_escalation=False),
        candidates=(candidate,),
    )

    assert allowed.action == "escalate"
    assert denied.action == "abstain"


def test_abstention_is_explicit_when_no_recovery_is_allowed() -> None:
    decision = next_decision(
        attempt=0,
        policy=RecoveryPolicy(allow_fallback=False, allow_escalation=False),
    )

    assert decision.action == "abstain"
    assert decision.attempt == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"max_retries": True},
        {"allow_fallback": 1},
        {"allow_escalation": 1},
    ],
)
def test_policy_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RecoveryPolicy(**kwargs)


def test_candidate_requires_target() -> None:
    with pytest.raises(ValueError):
        RecoveryCandidate()


def test_decision_serializes_observable_recovery() -> None:
    candidate = RecoveryCandidate(model="sonnet", worker_id="worker-2")
    decision = next_decision(
        attempt=1,
        policy=RecoveryPolicy(),
        candidates=(candidate,),
        error="provider unavailable",
    )

    assert decision.to_dict() == {
        "action": "fallback",
        "attempt": 2,
        "reason": "provider unavailable",
        "candidate": {
            "model": "sonnet",
            "worker_id": "worker-2",
            "escalation": False,
        },
    }
