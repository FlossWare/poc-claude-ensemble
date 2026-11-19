"""Core MetaLearner tests: draws, exclusion, fallback, posterior, state."""

from __future__ import annotations

import json
import os
import random
import stat
from collections import deque

import pytest

from learning.meta import MetaLearner
from learning.meta.domains.base import DomainError
from shared import thompson_policy


class FixedBetaRng:
    """RNG whose Beta draws come from a fixed queue (one per eligible arm)."""

    def __init__(self, betas):
        self.betas = deque(betas)

    def beta(self, alpha, beta):
        return self.betas.popleft()

    def random(self):
        return 1.0  # never below exploration_epsilon


class EpsilonRng(FixedBetaRng):
    """Beta draws fixed; random() forces exploration and randrange picks last."""

    def random(self):
        return 0.0

    def randrange(self, n):
        return n - 1


class BrokenClient:
    def select_arm(self, *args, **kwargs):
        raise RuntimeError("draw unavailable")


@pytest.fixture
def learner(tmp_path):
    return MetaLearner(tmp_path / "meta", rng=random.Random(7))


def write_policy(path, **overrides):
    params = {"prior_strength": 1.0, "cost_weight": 0.0,
              "exploration_epsilon": 0.0, "capability_margin": -1.0}
    params.update(overrides)
    path.write_text(json.dumps({"schema_version": 1, "policies": {"default": params}}),
                    encoding="utf-8")
    thompson_policy.clear_cache()
    return path


def test_seeded_draws_are_deterministic(tmp_path):
    first = MetaLearner(tmp_path / "a", rng=random.Random(19))
    second = MetaLearner(tmp_path / "b", rng=random.Random(19))
    assert first.select("crew_size") == second.select("crew_size")
    assert first.select("rag_scope") == second.select("rag_scope")


def test_fixed_beta_queue_picks_the_highest_draw(learner):
    learner.rng = FixedBetaRng([0.1, 0.2, 0.3, 0.9, 0.1])
    detail = learner.decide("crew_size", {"task_type": "code-review"})
    assert detail["arm"] == "4"
    assert detail["choice"] == 4
    assert detail["source"] == "thompson"


def test_ga_policy_file_changes_every_domain_draw(tmp_path, monkeypatch):
    policy = write_policy(tmp_path / "thompson-policy.json", exploration_epsilon=0.2)
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE", str(policy))
    try:
        learner = MetaLearner(tmp_path / "meta")
        learner.rng = FixedBetaRng([0.9, 0.1, 0.1, 0.1, 0.1])
        assert learner.decide("crew_size")["arm"] == "1"  # highest Beta draw
        learner.rng = EpsilonRng([0.9, 0.1, 0.1, 0.1, 0.1])
        # exploration_epsilon=0.2 plus random()==0 forces a uniform pick.
        assert learner.decide("crew_size")["arm"] == "5"
    finally:
        thompson_policy.clear_cache()


def test_exclusion_is_respected(learner):
    learner.rng = FixedBetaRng([0.9, 0.1, 0.1, 0.1])
    detail = learner.decide("crew_size", exclude=[1, 2, 3, 4])
    assert detail["arm"] == "5"
    assert detail["excluded"] == ["1", "2", "3", "4"]


def test_all_arms_excluded_falls_back_to_the_domain_default(learner):
    detail = learner.decide("crew_size", exclude=["1", "2", "3", "4", "5"])
    assert detail["source"] == "default_fallback"
    assert detail["choice"] == 3


def test_draw_failure_falls_back_to_highest_posterior_mean(tmp_path):
    learner = MetaLearner(tmp_path / "meta", client=BrokenClient())
    for _ in range(5):
        learner.record("crew_size", 2, True, 0.01)
    detail = learner.decide("crew_size")
    assert detail["source"] == "posterior_fallback"
    assert detail["arm"] == "2"


def test_posterior_reflects_graded_outcomes(learner):
    learner.record("crew_size", 3, 0.75, 0.5)
    posterior = learner.posterior("crew_size")
    arm = posterior["arms"]["3"]
    assert arm["successes"] == pytest.approx(0.75)
    assert arm["failures"] == pytest.approx(0.25)
    assert arm["alpha"] == pytest.approx(1.75)
    assert arm["beta"] == pytest.approx(1.25)
    assert arm["posterior_mean"] == pytest.approx(0.583333, abs=1e-6)
    assert arm["avg_cost"] == pytest.approx(0.5)


def test_record_is_idempotent_by_outcome_id(learner):
    first = learner.record("crew_size", 3, 1.0, 0.1, {"outcome_id": "review-1"})
    second = learner.record("crew_size", 3, 1.0, 0.1, {"outcome_id": "review-1"})
    assert first["recorded"] is True
    assert second["duplicate"] is True
    assert learner.posterior("crew_size")["arms"]["3"]["calls"] == 1


def test_state_persists_across_instances_and_is_0600(tmp_path):
    meta_dir = tmp_path / "meta"
    learner = MetaLearner(meta_dir)
    learner.record("crew_size", 5, True, 0.2, {"outcome_id": "persisted"})
    reopened = MetaLearner(meta_dir)
    assert reopened.posterior("crew_size")["arms"]["5"]["calls"] == 1
    mode = stat.S_IMODE(os.stat(meta_dir / "crew_size.json").st_mode)
    assert mode == 0o600


def test_context_bucket_requires_minimum_observations(learner):
    for _ in range(3):
        learner.record("crew_size", 2, True, 0.0, {"task_type": "code-review"},
                       context={"task_type": "code-review"})
    learner.record("crew_size", 5, True, 0.0, {"task_type": "testing"},
                   context={"task_type": "testing"})

    bucketed = learner.posterior("crew_size", {"task_type": "code-review"})
    assert bucketed["context_scope"] == "context"
    assert bucketed["arms"]["2"]["calls"] == 3
    assert bucketed["arms"]["5"]["calls"] == 0

    fallback = learner.posterior("crew_size", {"task_type": "testing"})
    assert fallback["context_scope"] == "global"
    assert fallback["arms"]["5"]["calls"] == 1


def test_corrupt_state_is_quarantined_not_fatal(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "crew_size.json").write_text("{not json", encoding="utf-8")
    learner = MetaLearner(meta_dir)
    assert learner.posterior("crew_size")["total_observations"] == 0
    assert list(meta_dir.glob("crew_size.json.corrupt-*"))


def test_unknown_domain_and_invalid_inputs_raise(learner):
    with pytest.raises(DomainError):
        learner.select("no_such_domain")
    with pytest.raises(ValueError):
        learner.record("crew_size", 3, 1.5)
    with pytest.raises(ValueError):
        learner.record("crew_size", 3, True, -0.1)
    with pytest.raises(ValueError):
        learner.record("crew_size", 9, True)
    with pytest.raises(ValueError):
        learner.record("crew_size", 3, True, 0.0, {"blob": "x" * (17 * 1024)})
    with pytest.raises(ValueError):
        learner.record("crew_size", 3, True, metadata=[("not", "a mapping")])


def test_status_reports_every_domain_with_default_top_arm(learner):
    status = learner.status()
    names = [row["domain"] for row in status["domains"]]
    assert names == ["crew_size", "review_depth", "batch_size", "rag_scope",
                     "prompt_strategy", "chunking", "tool_selection", "graph_traversal"]
    by_name = {row["domain"]: row for row in status["domains"]}
    assert by_name["crew_size"]["top_arm"] == "3"
    assert by_name["crew_size"]["top_posterior_mean"] is None
    assert by_name["prompt_strategy"]["instrumentation"] is True
    assert by_name["crew_size"]["instrumentation"] is False
