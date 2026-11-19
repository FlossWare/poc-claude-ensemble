"""Evidence and statistical gates, bootstrap CI, and run-outcome mapping."""

from __future__ import annotations

import pytest

from learning.ga import gates
from learning.ga.engine import EvalResult


def _eval(fitness=0.5, *, succ=0.9, cost=1.0, cost_ref=1.0, unsupported=0.0, coverage=1.0,
          contributions=None, keys=None, per_arm=None, decisions=400):
    return EvalResult(
        window="holdout",
        estimator="logged_bandit_direct_method",
        decisions=decisions,
        succ_hat=succ,
        cost_hat=cost,
        cost_ref=cost_ref,
        unsupported_fraction=unsupported,
        coverage=coverage,
        fitness=fitness,
        contributions=contributions if contributions is not None else [fitness] * decisions,
        block_keys=keys if keys is not None else ["2026-01-01"] * decisions,
        per_arm=per_arm
        if per_arm is not None
        else {"haiku": {"decisions": decisions, "logged_rows": 400, "logged_success": succ,
                        "logged_cost": cost}},
    )


def test_evidence_gate_refuses_the_real_world_snapshot():
    """Test 11: a 19-review fixture refuses with insufficient_evidence."""
    results = gates.evidence_gates(review_rounds=19, total_outcomes=3555, holdout_decisions=1066)
    assert gates.outcome_for(results, report_only=True) == "not_promoted:insufficient_evidence"
    failed = [gate.name for gate in results if not gate.passed]
    assert failed == ["review_rounds"]


def test_evidence_gates_pass_with_enough_data():
    results = gates.evidence_gates(review_rounds=60, total_outcomes=1200, holdout_decisions=400)
    assert all(gate.passed for gate in results)


def test_bootstrap_ci_is_deterministic_and_directional():
    deltas = [0.2] * 300
    keys = [f"2026-01-{index % 20 + 1:02d}" for index in range(300)]
    first = gates.bootstrap_ci(deltas, keys)
    second = gates.bootstrap_ci(deltas, keys)
    assert first == second
    assert first[0] > 0.0
    assert gates.bootstrap_ci([0.0] * 100, ["d"] * 100)[0] == 0.0
    assert gates.bootstrap_ci([], []) == (0.0, 0.0)


def test_all_statistical_gates_pass_for_a_clear_improvement():
    baseline = _eval(0.50, succ=0.90, cost=1.0)
    candidate = _eval(0.75, succ=0.95, cost=1.02)
    results, ci = gates.statistical_gates(baseline, candidate, min_support=30)
    assert all(gate.passed for gate in results), [gate.to_dict() for gate in results]
    assert ci[0] > 0
    assert gates.outcome_for(results, report_only=True) == "not_promoted:report_only"
    assert gates.outcome_for(results, report_only=False) == "promoted"


def test_delta_gate_blocks_a_tie():
    baseline = _eval(0.50)
    candidate = _eval(0.505)
    results, _ci = gates.statistical_gates(baseline, candidate, min_support=30)
    assert gates.outcome_for(results, report_only=False) == "not_promoted:delta_below_threshold"


def test_cost_regression_gate():
    baseline = _eval(0.50, cost=1.0)
    candidate = _eval(0.90, cost=1.20)
    results, _ci = gates.statistical_gates(baseline, candidate, min_support=30)
    reasons = {gate.name: gate for gate in results}
    assert not reasons["cost_regression"].passed


def test_arm_regression_gate():
    baseline = _eval(0.50, succ=0.90)
    candidate = _eval(
        0.75,
        succ=0.95,
        per_arm={"bad": {"decisions": 400, "logged_rows": 400, "logged_success": 0.5,
                         "logged_cost": 1.0}},
    )
    results, _ci = gates.statistical_gates(baseline, candidate, min_support=30)
    reasons = {gate.name: gate for gate in results}
    assert not reasons["arm_regression"].passed


def test_support_and_coverage_gates():
    baseline = _eval(0.50)
    sparse = _eval(
        0.75,
        unsupported=0.02,
        coverage=0.4,
        per_arm={"rare": {"decisions": 400, "logged_rows": 5, "logged_success": 0.9,
                          "logged_cost": 1.0}},
    )
    results, _ci = gates.statistical_gates(baseline, sparse, min_support=30)
    reasons = {gate.name: gate for gate in results}
    assert not reasons["per_arm_support"].passed
    assert not reasons["unsupported_fraction"].passed
    assert not reasons["realized_coverage"].passed
    assert gates.outcome_for(results, report_only=False) == "not_promoted:insufficient_support"


def test_cooldown_gate():
    results, _ci = gates.statistical_gates(
        _eval(0.50),
        _eval(0.75),
        min_support=30,
        cooldown_ok=False,
        cooldown_detail="too soon",
    )
    assert not {gate.name: gate for gate in results}["cooldown"].passed
