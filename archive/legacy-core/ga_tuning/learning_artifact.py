#!/usr/bin/env python3
"""Build a portable, provenance-preserving artifact from a GA run."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from typing import Any, Mapping

from learning.portable_artifacts import LearningArtifact
from ga_tuning.parameter_schema import PARAMETER_SCHEMA, validate_parameter


OPTIMIZER_VERSION = "0.1"


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GA summary timestamp is required")
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(value, fmt)
            # The GA filename timestamp has no timezone; preserve that fact.
            return parsed.isoformat()
        except ValueError:
            continue
    raise ValueError(f"unsupported GA summary timestamp: {value!r}")


def _candidate_parameters(system: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every parameter included in the persisted candidate population."""
    if system not in PARAMETER_SCHEMA:
        raise ValueError(f"Unsupported GA parameter system: {system}")
    nested = candidate.get("parameters")
    raw = nested if isinstance(nested, Mapping) else candidate
    parameters: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "fitness":
            continue
        parameters[str(key)] = validate_parameter(system, str(key), value)
    return parameters


def candidate_digest(artifact: LearningArtifact) -> str:
    """Digest immutable candidate evidence while excluding mutable runtime fallbacks."""
    # LearningArtifact exposes immutable mapping proxies; normalize through its
    # canonical JSON representation before hashing the candidate evidence.
    payload = json.loads(artifact.to_json())["payload"]
    identity = {
        "run_id": payload.get("run_id"),
        "candidate_population": payload.get("candidate_population"),
        "selected_parameters": payload.get("selected_parameters"),
        "objectives": payload.get("objectives"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def scored_candidates(candidates: Any, limit: int = 3) -> list[tuple[Mapping[str, Any], float]]:
    """Return the top source candidates with finite, numeric fitness in source order."""
    if not isinstance(candidates, list):
        return []
    scored: list[tuple[Mapping[str, Any], float]] = []
    for candidate in candidates[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        fitness = candidate.get("fitness")
        if (
            isinstance(fitness, bool)
            or not isinstance(fitness, (int, float))
            or not math.isfinite(float(fitness))
        ):
            continue
        scored.append((candidate, float(fitness)))
    return scored


def build_ga_learning_artifact(
    summary: Mapping[str, Any],
    best_by_system: Mapping[str, Any],
    fallback_parameters: Mapping[str, Any],
    *,
    best_parameters_source: str,
    summary_source: str,
) -> LearningArtifact:
    """Build a deterministic artifact; synthetic fitness is not ground truth."""
    timestamp = summary.get("timestamp")
    # Do not fabricate provenance or collapse unrelated malformed runs into
    # the same "ga-unknown" idempotency key.
    created_at = _timestamp(timestamp)
    timestamp_text = timestamp.strip()
    # Keep the timestamp human-readable, but distinguish different optimizer
    # outputs that happen to share a second. Exclude fallback settings because
    # they change after application and must not change the identity of a run.
    try:
        identity_source = json.dumps(
            {
                "summary": summary,
                "best_by_system": best_by_system,
                "source_files": {
                    "summary": summary_source,
                    "best_parameters": best_parameters_source,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("GA summary and candidate values must be finite JSON numbers") from exc
    result_digest = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()[:12]
    run_id = f"ga-{timestamp_text}-{result_digest}"
    selected_parameters: dict[str, dict[str, Any]] = {}
    objectives: dict[str, float] = {}
    top_candidates: dict[str, list[dict[str, Any]]] = {}

    for system, candidates in best_by_system.items():
        if not isinstance(candidates, list) or not candidates:
            continue
        normalized = []
        for candidate, fitness in scored_candidates(candidates):
            parameters = _candidate_parameters(str(system), candidate)
            normalized.append({
                "rank": len(normalized) + 1,
                "fitness": fitness,
                "parameters": parameters,
            })
        if not normalized:
            continue
        top_candidates[str(system)] = normalized
        selected_parameters[str(system)] = dict(normalized[0]["parameters"])
        objectives[str(system)] = normalized[0]["fitness"]

    payload = {
        "run_id": run_id,
        "run_timestamp": timestamp_text,
        "optimizer": {"name": "genetic-algorithm", "version": OPTIMIZER_VERSION},
        "population_size": summary.get("population_size"),
        "generations": summary.get("generations"),
        "total_evaluations": summary.get("total_evaluations"),
        "evaluation_scope": {
            "systems_evaluated": sorted(selected_parameters),
            "evaluation_mode": "synthetic-local-evaluators",
            "api_calls": 0,
        },
        "candidate_population": {
            "kind": "top_candidates_per_system",
            "candidates": top_candidates,
        },
        "objectives": objectives,
        "selected_parameters": selected_parameters,
        "fallback_parameters": dict(fallback_parameters),
        "evidence": {
            "kind": "synthetic-fitness",
            "externally_validated": False,
            "confidence": None,
        },
        "configuration_status": "candidate",
        "knowledge_promotion": {
            "eligible": False,
            "reason": "synthetic GA fitness alone is not operational evidence",
        },
    }
    return LearningArtifact.create(
        "ga.tuning.result",
        payload,
        source="claude-ensemble.ga-tuning",
        provenance={
            "run_id": run_id,
            "optimizer": "genetic-algorithm",
            "optimizer_version": OPTIMIZER_VERSION,
            "source_files": {
                "summary": summary_source,
                "best_parameters": best_parameters_source,
            },
            "candidate_count_by_system": {
                system: len(candidates) for system, candidates in top_candidates.items()
            },
        },
        created_at=created_at,
    )
