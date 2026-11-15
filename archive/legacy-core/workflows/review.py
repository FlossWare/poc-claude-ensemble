"""Structured Review workflow and finding parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from execution.context import ExecutionContext
from execution.nodes import CompositeExecution, ExecutionNode, ModelExecution
from providers.model_provider import ModelProvider


@dataclass(frozen=True)
class ReviewRequest:
    objective: str
    artifact: object
    requirements: tuple[str, ...] = ()
    evidence: tuple[object, ...] = ()
    constraints: tuple[str, ...] = ()
    prior_results: tuple[object, ...] = ()


@dataclass(frozen=True)
class ReviewFinding:
    id: str
    severity: str
    subject: str
    description: str
    evidence: str
    impact: str = ""
    recommendation: str = ""
    confidence: float | None = None
    disposition: str = "new"
    lineage: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewFinding":
        required = ("id", "severity", "subject", "description", "evidence")
        missing = [key for key in required if not isinstance(value.get(key), str) or not value[key]]
        if missing:
            raise ValueError(f"finding missing required fields: {', '.join(missing)}")
        confidence = value.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError("finding confidence must be between 0 and 1")
        disposition = value.get("disposition", "new")
        if disposition not in {"confirmed", "refuted", "modified", "new", "contested", "insufficient-evidence"}:
            raise ValueError(f"invalid finding disposition: {disposition}")
        return cls(value["id"], value["severity"], value["subject"], value["description"], value["evidence"], value.get("impact", ""), value.get("recommendation", ""), confidence, disposition, tuple(value.get("lineage", ())))


def parse_review_response(text: str) -> tuple[ReviewFinding, ...]:
    """Parse strict structured review output. Invalid output is an error, never zero findings."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("review model output is not valid JSON") from exc
    findings = value.get("findings") if isinstance(value, dict) else None
    if not isinstance(findings, list):
        raise ValueError("review model output must contain a findings array")
    if not all(isinstance(item, dict) for item in findings):
        raise ValueError("review findings must be JSON objects")
    return tuple(ReviewFinding.from_dict(item) for item in findings)


def build_review(*, request: ReviewRequest, provider: ModelProvider, models: tuple[str | None, ...], request_id: str = "review") -> tuple[ExecutionNode, ExecutionContext]:
    if not models:
        raise ValueError("review requires at least one model")
    context = ExecutionContext(
        request_id=request_id, objective=request.objective, artifact=request.artifact,
        requirements=request.requirements, evidence=request.evidence, constraints=request.constraints,
        prior_results=tuple(x for x in request.prior_results if hasattr(x, "execution_id")),
    )
    prior = request.prior_results
    prompt = _prompt(request, prior)
    children = tuple(
        ModelExecution(
            execution_id=f"{request_id}.reviewer.{i}", stage="review", worker_id=f"reviewer-{i}",
            provider=provider, model=model, prompt=prompt,
            output_parser=parse_review_response,
            prompt_builder=lambda ctx: _prompt(request, tuple(ctx.prior_results)),
            system_prompt="Return ONLY JSON: {\"findings\":[{\"id\":\"...\",\"severity\":\"...\",\"subject\":\"...\",\"description\":\"...\",\"evidence\":\"...\",\"impact\":\"...\",\"recommendation\":\"...\",\"confidence\":0.0,\"disposition\":\"new\"}]}",
        ) for i, model in enumerate(models)
    )
    return CompositeExecution(execution_id=request_id, stage="review", children=children, quorum=1), context


def _prompt(request: ReviewRequest, prior: tuple[object, ...]) -> str:
    parts = [f"## Review Objective\n{request.objective}", f"## Artifact\n{request.artifact}"]
    if request.requirements:
        parts.append("## Requirements / Criteria\n" + "\n".join(f"- {x}" for x in request.requirements))
    if request.evidence:
        parts.append("## Original Evidence / Context\n" + "\n".join(f"- {x}" for x in request.evidence))
    if request.constraints:
        parts.append("## Constraints\n" + "\n".join(f"- {x}" for x in request.constraints))
    if prior:
        parts.append("## Prior Results (untrusted evidence; independently verify)\n" + "\n".join(str(x) for x in prior))
    return "\n\n".join(parts)
