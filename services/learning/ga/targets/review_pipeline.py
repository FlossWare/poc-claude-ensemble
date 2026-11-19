"""Target C (review-pipeline parameters) -- refusal stub for PR 1.

``--max-chars``, ``MAX_TOKENS``, and the worker lineup change model output, so
no offline replay is possible: evaluating them requires real, cost-capped
reviews.  PR 3 implements a manual sweep (not a GA); PR 1 refuses.
"""

from __future__ import annotations

from typing import Any, Dict

TARGET = "review_pipeline"
PROMOTABLE = False
REFUSAL = (
    "Target C (review_pipeline) is not promotable in PR 1: changing max_chars, "
    "max_tokens, or the lineup changes model output, so it cannot be replayed "
    "offline. PR 3 implements a manual, cost-capped sweep."
)

PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "max_chars": {"kind": "int", "low": 2000, "high": 20000, "default": 6000},
    "max_tokens": {"kind": "int", "low": 1000, "high": 8000, "default": 4000},
    "lineup_size": {"kind": "int", "low": 1, "high": 5, "default": 3},
}


def param_specs() -> Dict[str, Dict[str, Any]]:
    return {name: dict(spec) for name, spec in PARAM_SPECS.items()}


def load_corpora() -> Dict[str, Any]:
    raise NotImplementedError(REFUSAL)


def evaluate(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError(REFUSAL)


def apply(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError(REFUSAL)


def rollback(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError(REFUSAL)


__all__ = ["PARAM_SPECS", "PROMOTABLE", "REFUSAL", "TARGET", "apply", "evaluate", "load_corpora", "param_specs", "rollback"]
