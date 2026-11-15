"""Regression contract tests for learning score/confidence semantics.

These tests deliberately do not implement the learning formula. They lock down the
semantic vectors that a later formula implementation must consume.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.scoring_semantics_contract import (  # noqa: E402
    CONFIDENCE_FIELDS,
    SCORE_COMPONENTS,
    SEPARATION_RULES,
    SIGN_CONTRACT,
)


def test_score_components_have_explicit_semantics() -> None:
    assert SCORE_COMPONENTS == (
        "base_score",
        "complexity_adjustment",
        "domain_adjustment",
    )


def test_positive_and_negative_adjustment_signs_are_explicit() -> None:
    assert SIGN_CONTRACT == (
        ("complexity_adjustment", "positive", 1),
        ("complexity_adjustment", "negative", -1),
        ("domain_adjustment", "positive", 1),
        ("domain_adjustment", "negative", -1),
    )


def test_confidence_dimensions_are_distinct() -> None:
    assert len(CONFIDENCE_FIELDS) == len(set(CONFIDENCE_FIELDS))
    assert set(CONFIDENCE_FIELDS) == {
        "sample_count",
        "agreement_rate",
        "empirical_success_rate",
        "recommendation_confidence",
    }


def test_confidence_dimensions_cannot_be_collapsed_semantically() -> None:
    assert all(field in CONFIDENCE_FIELDS for field, _ in SEPARATION_RULES)
    assert len(SEPARATION_RULES) == len(CONFIDENCE_FIELDS)
    assert len({rule for _, rule in SEPARATION_RULES}) == len(SEPARATION_RULES)


def test_specification_does_not_restore_abandoned_scoring_formula() -> None:
    documentation = (
        Path(__file__).parent.parent / "docs" / "LEARNING_SCORING_SEMANTICS.md"
    ).read_text(encoding="utf-8")

    assert "not a source of truth" in documentation
    assert "must not be restored wholesale" in documentation
    assert "No unsupported probabilistic terminology" in documentation
