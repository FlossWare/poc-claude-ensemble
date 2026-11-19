"""Target B (prompt-time RAG configuration) -- refusal stub for PR 1.

There is no automatic fitness signal for retrieval usefulness in the recorded
data, so this target must not run on a schedule.  PR 1 therefore ships the
parameter surface only; ``evaluate`` and ``apply`` refuse explicitly.
"""

from __future__ import annotations

from typing import Any, Dict

TARGET = "rag_config"
PROMOTABLE = False
REFUSAL = (
    "Target B (rag_config) is not promotable: there is no automatic fitness "
    "signal for retrieval usefulness in the recorded ledgers. A live consumer "
    "and an explicit rating signal must exist first."
)

# Documented knob ranges, recorded here so the future implementation has one
# place to start.  None of these are evolvable today.
PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "memory.enabled": {"kind": "bool", "default": True},
    "memory.max_results": {"kind": "int", "low": 1, "high": 50, "default": 5},
    "memory.timeout_ms": {"kind": "int", "low": 50, "high": 5000, "default": 500},
    "knowledge.enabled": {"kind": "bool", "default": True},
    "knowledge.max_queries": {"kind": "int", "low": 1, "high": 10, "default": 3},
    "knowledge.max_results": {"kind": "int", "low": 1, "high": 50, "default": 5},
    "knowledge.timeout_ms": {"kind": "int", "low": 50, "high": 5000, "default": 800},
    "graph.enabled": {"kind": "bool", "default": False},
    "graph.max_depth": {"kind": "int", "low": 1, "high": 5, "default": 2},
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
