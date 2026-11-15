"""Contract fixtures for learning scoring semantics.

This module intentionally contains no scoring formula. It provides stable semantic
vectors that the eventual scoring implementation must satisfy.
"""

from __future__ import annotations

SCORE_COMPONENTS = (
    "base_score",
    "complexity_adjustment",
    "domain_adjustment",
)

SIGN_CONTRACT = (
    ("complexity_adjustment", "positive", 1),
    ("complexity_adjustment", "negative", -1),
    ("domain_adjustment", "positive", 1),
    ("domain_adjustment", "negative", -1),
)

CONFIDENCE_FIELDS = (
    "sample_count",
    "agreement_rate",
    "empirical_success_rate",
    "recommendation_confidence",
)

SEPARATION_RULES = (
    ("sample_count", "must_not_be_treated_as_score"),
    ("agreement_rate", "must_not_be_treated_as_correctness"),
    ("empirical_success_rate", "must_not_be_treated_as_future_probability"),
    ("recommendation_confidence", "must_not_be_treated_as_model_correctness"),
)
