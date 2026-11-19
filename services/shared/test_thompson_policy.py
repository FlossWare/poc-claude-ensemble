"""Tests for the shared Thompson selection policy (design test plan 1-5)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

SERVICES = Path(__file__).resolve().parents[1]
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from shared import thompson_policy as policy
from shared.thompson_policy import Arm, ParamValidationError, default_params, load_policy_file, select

INF = float("inf")


class FrozenStats:
    """Frozen copy of the pre-GA ModelStats surface used by the old selector."""

    def __init__(self, successes: int, failures: int, cost: float, calls: int):
        self.successes = successes
        self.failures = failures
        self.total_cost = cost * calls
        self.calls = calls

    @property
    def avg_cost(self) -> float:
        return 0.0 if self.calls == 0 else self.total_cost / self.calls


def frozen_select(models, max_cost=INF):
    """The pre-GA algorithm, copied verbatim so drift is detectable."""
    candidates = {
        name: stats for name, stats in models.items() if stats.avg_cost <= max_cost or stats.calls == 0
    }
    if not candidates:
        cheapest = min(models.items(), key=lambda item: item[1].avg_cost)
        candidates = {cheapest[0]: cheapest[1]}
    best_model = None
    best_sample = -1
    for name, stats in candidates.items():
        sample = np.random.beta(stats.successes + 1, stats.failures + 1)
        if sample > best_sample:
            best_sample = sample
            best_model = name
    return best_model


def arms_of(models) -> list:
    return [Arm(n, s.successes, s.failures, s.avg_cost, s.calls) for n, s in models.items()]


def rng_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@pytest.fixture()
def models():
    return {
        "haiku": FrozenStats(30, 3, 0.01, 33),
        "sonnet": FrozenStats(5, 1, 0.05, 6),
        "opus": FrozenStats(1, 2, 0.2, 3),
        "untested": FrozenStats(0, 0, 0.0, 0),
    }


def test_default_equivalence_sequence_and_draws(models):
    """Test 1: default params reproduce the old algorithm with no extra RNG draws."""
    np.random.seed(4242)
    frozen = [frozen_select(models) for _ in range(60)]
    np.random.seed(4242)
    gated = [
        select(default_params(), arms_of(models), "task", 0.5, INF, np.random) for _ in range(60)
    ]
    assert frozen == gated

    np.random.seed(99)
    frozen_select(models)
    expected_state = np.random.get_state()
    np.random.seed(99)
    select(default_params(), arms_of(models), "task", 0.5, INF, np.random)
    assert rng_state_equal(expected_state, np.random.get_state())


def test_default_equivalence_with_cost_cap(models):
    for cap in (0.0, 0.02, 0.06, 0.25):
        np.random.seed(7)
        frozen = [frozen_select(models, cap) for _ in range(30)]
        np.random.seed(7)
        gated = [select(default_params(), arms_of(models), "t", 0.5, cap, np.random) for _ in range(30)]
        assert frozen == gated, cap


BAD_VALUES = [
    pytest.param({"prior_strength": float("nan")}, id="nan"),
    pytest.param({"prior_strength": float("inf")}, id="inf"),
    pytest.param({"cost_weight": True}, id="bool-as-int"),
    pytest.param({"cost_weight": 1.5}, id="out-of-range"),
    pytest.param({"capability_margin": -2.0}, id="out-of-range-low"),
    pytest.param({"not_a_gene": 0.5}, id="unknown-key"),
    pytest.param({"prior_strength": "1.0"}, id="string"),
    pytest.param({"prior_strength": None}, id="none"),
]


def _validated(bad):
    params = default_params()
    params.update(bad)
    return params


def _validator_select(bad):
    return select(_validated(bad), [Arm("a", 1, 0, 0.1, 1)], None, 0.0, INF, random.Random(1))


def _validator_policy_document(bad):
    return policy.policy_document({"default": _validated(bad)})


def _validator_prepare_apply(bad):
    from learning.ga.targets import thompson_policy as adapter

    return adapter.prepare_apply({"default": _validated(bad)}, {})


def _validator_make_candidate(bad):
    from learning.ga import schema

    return schema.make_candidate(_validated(bad))


def _validator_apply(bad):
    from learning.ga.targets import thompson_policy as adapter

    return adapter.apply_policies({"default": _validated(bad)}, {})


VALIDATION_POINTS = [
    pytest.param(policy.validate_params, id="shared.validate_params"),
    pytest.param(_validator_make_candidate, id="schema.make_candidate"),
    pytest.param(_validator_select, id="shared.select"),
    pytest.param(_validator_policy_document, id="shared.policy_document"),
    pytest.param(_validator_prepare_apply, id="adapter.prepare_apply"),
    pytest.param(_validator_apply, id="adapter.apply_policies"),
]


@pytest.mark.parametrize("validator", VALIDATION_POINTS)
@pytest.mark.parametrize("bad", BAD_VALUES)
def test_bounds_rejected_at_every_validation_point(validator, bad):
    """Test 2: NaN, inf, bool, unknown keys and out-of-range fail at all points."""
    with pytest.raises(ValueError):
        validator(bad)


def test_cost_weight_is_scale_free():
    """Test 3: multiplying every cost by a constant cannot change the choice."""
    left = [Arm("cheap", 9, 1, 1.0, 10), Arm("pricey", 10, 0, 10.0, 10)]
    right = [Arm("cheap", 9, 1, 2.0, 10), Arm("pricey", 10, 0, 20.0, 10)]
    params = dict(default_params(), cost_weight=0.6)
    left_choices = [select(params, left, None, 0.0, INF, random.Random(seed)) for seed in range(40)]
    right_choices = [select(params, right, None, 0.0, INF, random.Random(seed)) for seed in range(40)]
    assert left_choices == right_choices
    assert "cheap" in left_choices


def test_capability_margin_filters_and_falls_back():
    """Test 4: the floor removes low-success arms; an empty result falls back."""
    arms = [Arm("strong", 10, 0, 1.0, 10), Arm("weak", 1, 9, 1.0, 10)]
    params = dict(default_params(), capability_margin=0.9)
    picks = {select(params, arms, None, 0.0, INF, random.Random(seed)) for seed in range(30)}
    assert picks == {"strong"}

    # Everything below the floor: fall back to the unfiltered eligible set.
    everything_weak = [Arm("a", 1, 9, 1.0, 10), Arm("b", 2, 8, 1.0, 10)]
    fallback = {select(params, everything_weak, None, 0.0, INF, random.Random(seed)) for seed in range(40)}
    assert fallback == {"a", "b"}

    # required_capability never loosens the gene.
    strict = dict(default_params(), capability_margin=0.9)
    picks_required = {
        select(strict, arms, None, 0.0, INF, random.Random(seed)) for seed in range(20)
    }
    assert picks_required == {"strong"}


def test_exploration_epsilon_stays_in_bounds_and_changes_choice():
    arms = [Arm("a", 10, 0, 1.0, 10), Arm("b", 10, 0, 1.0, 10)]
    params = dict(default_params(), exploration_epsilon=0.9)  # over the bound on purpose
    with pytest.raises(ParamValidationError):
        select(params, arms, None, 0.0, INF, random.Random(1))
    params = dict(default_params(), exploration_epsilon=0.2)
    picks = {select(params, arms, None, 0.0, INF, random.Random(seed)) for seed in range(200)}
    assert picks == {"a", "b"}


def _write_policy(path: Path, params: dict, *, mtime_ns: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "policies": {"default": params}, "provenance": {"test": True}}
    path.write_text(json.dumps(document), encoding="utf-8")
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_load_policy_file_valid_invalid_and_mtime(tmp_path, monkeypatch):
    """Test 5a: valid files load, invalid files fall back, mtime edits are picked up."""
    path = tmp_path / "thompson-policy.json"
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE", str(path))

    assert load_policy_file() ["default"] == default_params()

    _write_policy(path, dict(default_params(), cost_weight=0.5), mtime_ns=1_000_000_000)
    loaded = load_policy_file()
    assert loaded["default"]["cost_weight"] == 0.5

    _write_policy(path, dict(default_params(), cost_weight=0.9), mtime_ns=1_000_000_001)
    loaded = load_policy_file()
    assert loaded["default"]["cost_weight"] == 0.9

    # Invalid JSON / bad bounds / unknown keys never raise and never leak through.
    for document in ("{not json", json.dumps({"schema_version": 1, "policies": {"default": {"cost_weight": 9}}}),
                     json.dumps({"schema_version": 1, "policies": {"default": {}}, "extra": 1})):
        path.write_text(document, encoding="utf-8")
        policy.clear_cache()
        assert load_policy_file()["default"] == default_params()

    # A symlinked policy file is refused (defaults), not followed.
    target = tmp_path / "elsewhere.json"
    _write_policy(target, dict(default_params(), cost_weight=0.4))
    link = tmp_path / "link-policy.json"
    link.symlink_to(target)
    policy.clear_cache()
    assert load_policy_file(link)["default"] == default_params()


def test_service_delegates_to_shared_policy(tmp_path, monkeypatch):
    """Test 5b: the live service loads a valid file, ignores an invalid one, and reloads."""
    sys.path.insert(0, str(SERVICES / "thompson-service"))
    import thompson_service

    policy_file = tmp_path / "policy.json"
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE", str(policy_file))
    policy.clear_cache()

    service = thompson_service.ThompsonService(
        tmp_path / "thompson.sock", tmp_path / "state.json"
    )
    service.state.record_outcome("cheap", "t", True, 0.001, 10)
    service.state.record_outcome("pricey", "t", True, 10.0, 10)

    _write_policy(policy_file, dict(default_params(), cost_weight=1.0), mtime_ns=2_000_000_000)
    assert {service.state.select_model("t") for _ in range(20)} == {"cheap"}

    # Invalid file: ignored, defaults used, selection still returns a real arm.
    policy_file.write_text(json.dumps({"schema_version": 1, "policies": {"default": {"cost_weight": 7}}}))
    policy.clear_cache()
    assert service.state.select_model("t") in {"cheap", "pricey"}

    # mtime change is picked up without rebuilding the service.
    _write_policy(policy_file, dict(default_params(), cost_weight=1.0), mtime_ns=3_000_000_000)
    assert {service.state.select_model("t") for _ in range(20)} == {"cheap"}
    _write_policy(policy_file, dict(default_params(), prior_strength=1.0), mtime_ns=3_000_000_001)
    assert service.state.select_model("t") in {"cheap", "pricey"}


def test_default_params_and_bounds_shape():
    assert set(policy.PARAM_BOUNDS) == {
        "prior_strength",
        "cost_weight",
        "exploration_epsilon",
        "capability_margin",
    }
    assert policy.PARAM_BOUNDS["prior_strength"].low == 0.1
    assert policy.PARAM_BOUNDS["prior_strength"].high == 5.0
    assert policy.PARAM_BOUNDS["cost_weight"].default == 0.0
    assert policy.PARAM_BOUNDS["exploration_epsilon"].high == 0.2
    assert policy.PARAM_BOUNDS["capability_margin"].default == -1.0
    assert math.isclose(default_params()["prior_strength"], 1.0)
