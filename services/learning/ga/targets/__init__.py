"""Target adapters.

Only Target A (``thompson_policy``) is promotable in PR 1.  Targets B and C
ship as explicit refusal stubs: they expose their parameter specs so the shape
of the eventual work is visible, but ``evaluate`` raises ``NotImplementedError``
and the CLI reports a configuration error instead of running them.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict

from learning.ga.targets import rag_config, review_pipeline, thompson_policy

_REGISTRY: Dict[str, ModuleType] = {
    thompson_policy.TARGET: thompson_policy,
    rag_config.TARGET: rag_config,
    review_pipeline.TARGET: review_pipeline,
}


def get_target(name: str) -> ModuleType:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown GA target {name!r}; expected one of {sorted(_REGISTRY)}") from exc


def target_names() -> list:
    return sorted(_REGISTRY)


__all__ = ["get_target", "rag_config", "review_pipeline", "target_names", "thompson_policy"]
