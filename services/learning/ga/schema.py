"""Candidate representation, gene specs, canonical JSON, and hashing.

The gene bounds live in :mod:`shared.thompson_policy` and are re-exported here
so the live service and the tuner cannot drift apart.  The fitness weights
(lambda for cost, mu for unsupported selections) deliberately live in
:mod:`learning.ga.engine` and are *not* genes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping

from shared.thompson_policy import (
    GENES,
    PARAM_BOUNDS,
    ParamValidationError,
    default_params,
    validate_params as _validate_params,
)

TARGET_THOMPSON_POLICY = "thompson_policy"
TARGET_RAG_CONFIG = "rag_config"
TARGET_REVIEW_PIPELINE = "review_pipeline"
TARGETS = (TARGET_THOMPSON_POLICY, TARGET_RAG_CONFIG, TARGET_REVIEW_PIPELINE)


def gene_specs() -> Dict[str, Dict[str, float]]:
    """JSON-friendly view of the evolvable genes for one target."""
    return {
        name: {"low": spec.low, "high": spec.high, "default": spec.default}
        for name, spec in PARAM_BOUNDS.items()
    }


def validate_params(
    params: Mapping[str, Any],
    *,
    require_all: bool = True,
    source: str = "params",
) -> Dict[str, float]:
    """Validate a candidate/parameter mapping (validation point)."""
    return _validate_params(params, require_all=require_all, source=source)


def canonical_json(value: Any) -> str:
    """Stable JSON encoding.  Rejects NaN/inf, which never enter a candidate."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_hash(params: Mapping[str, Any]) -> str:
    """Stable hash of a fully specified candidate."""
    normalised = validate_params(params, require_all=True, source="candidate")
    return hashlib.sha256(canonical_json(normalised).encode("utf-8")).hexdigest()


def make_candidate(params: Mapping[str, Any]) -> Dict[str, float]:
    """Validate and return a canonical candidate dict in gene order."""
    normalised = validate_params(params, require_all=True, source="candidate")
    return {name: normalised[name] for name in GENES}


def default_candidate() -> Dict[str, float]:
    return make_candidate(default_params())


def clamp(name: str, value: float) -> float:
    spec = PARAM_BOUNDS[name]
    if value != value:  # NaN
        return spec.default
    return min(spec.high, max(spec.low, float(value)))


def policy_document(
    policies: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Policy-file document (schema_version 1) for the given policies."""
    from shared.thompson_policy import policy_document as _document

    return _document(policies, provenance)


__all__ = [
    "GENES",
    "PARAM_BOUNDS",
    "ParamValidationError",
    "TARGETS",
    "TARGET_RAG_CONFIG",
    "TARGET_REVIEW_PIPELINE",
    "TARGET_THOMPSON_POLICY",
    "candidate_hash",
    "canonical_json",
    "clamp",
    "default_candidate",
    "default_params",
    "gene_specs",
    "make_candidate",
    "policy_document",
    "validate_params",
]
