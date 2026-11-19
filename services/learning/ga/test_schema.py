"""Candidate schema: gene specs, canonical JSON, validation, and hashing."""

from __future__ import annotations

import json
import math

import pytest

from learning.ga import schema
from shared.thompson_policy import PARAM_BOUNDS, default_params


def test_gene_specs_match_the_shared_policy_bounds():
    specs = schema.gene_specs()
    assert set(specs) == set(PARAM_BOUNDS)
    for name, spec in PARAM_BOUNDS.items():
        assert specs[name] == {"low": spec.low, "high": spec.high, "default": spec.default}
    assert "cost_lambda" not in specs
    assert "unsupported_mu" not in specs


def test_default_candidate_is_the_default_params():
    assert schema.default_candidate() == default_params()
    assert schema.make_candidate(default_params()) == default_params()


@pytest.mark.parametrize(
    "params",
    [
        {**default_params(), "prior_strength": float("nan")},
        {**default_params(), "prior_strength": float("inf")},
        {**default_params(), "cost_weight": True},
        {**default_params(), "cost_weight": -0.1},
        {**default_params(), "exploration_epsilon": 0.5},
        {**default_params(), "capability_margin": 2.0},
        {**default_params(), "unknown_gene": 1.0},
        {"prior_strength": 1.0},  # incomplete
        "not a mapping",
    ],
)
def test_make_candidate_rejects_bad_input(params):
    with pytest.raises(ValueError):
        schema.make_candidate(params)


def test_validate_params_can_fill_defaults_for_policy_files():
    assert schema.validate_params({"cost_weight": 0.5}, require_all=False) == dict(
        default_params(), cost_weight=0.5
    )


def test_canonical_json_is_order_independent_and_rejects_non_finite():
    left = schema.canonical_json({"b": 1, "a": 2})
    right = schema.canonical_json({"a": 2, "b": 1})
    assert left == right == '{"a":2,"b":1}'
    with pytest.raises(ValueError):
        schema.canonical_json({"a": float("nan")})
    with pytest.raises(ValueError):
        schema.canonical_json({"a": float("inf")})


def test_candidate_hash_is_stable_and_sensitive():
    base = schema.default_candidate()
    digest = schema.candidate_hash(base)
    assert len(digest) == 64
    assert digest == schema.candidate_hash(dict(reversed(list(base.items()))))
    assert digest != schema.candidate_hash(dict(base, cost_weight=0.5))


def test_clamp_stays_inside_bounds():
    spec = PARAM_BOUNDS["cost_weight"]
    assert schema.clamp("cost_weight", -5.0) == spec.low
    assert schema.clamp("cost_weight", 5.0) == spec.high
    assert schema.clamp("cost_weight", 0.25) == 0.25
    assert schema.clamp("cost_weight", float("nan")) == spec.default


def test_policy_document_shape():
    document = schema.policy_document({"default": default_params()}, {"run_id": "ga-x"})
    assert document["schema_version"] == 1
    assert document["policies"]["default"] == default_params()
    assert document["provenance"] == {"run_id": "ga-x"}
    json.dumps(document)  # must be JSON-serialisable
    with pytest.raises(ValueError):
        schema.policy_document({"default": dict(default_params(), cost_weight=9.0)})


def test_target_names():
    assert set(schema.TARGETS) == {"thompson_policy", "rag_config", "review_pipeline"}
    assert math.isclose(PARAM_BOUNDS["prior_strength"].default, 1.0)
