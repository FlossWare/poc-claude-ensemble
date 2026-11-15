"""Solve workflow builder on top of recursive execution primitives."""

from __future__ import annotations

from dataclasses import dataclass

from execution.context import ExecutionContext
from execution.nodes import CompositeExecution, ExecutionNode, ModelExecution
from providers.model_provider import ModelProvider


@dataclass(frozen=True)
class SolveRequest:
    objective: str
    artifact: object = None
    requirements: tuple[str, ...] = ()
    evidence: tuple[object, ...] = ()
    constraints: tuple[str, ...] = ()


def build_solve(*, request: SolveRequest, provider: ModelProvider, models: tuple[str | None, ...], request_id: str = "solve") -> tuple[ExecutionNode, ExecutionContext]:
    """Build a solve tree. Multiple models are one composite level; nesting is unrestricted."""
    if not models:
        raise ValueError("solve requires at least one model")
    context = ExecutionContext(
        request_id=request_id, objective=request.objective, artifact=request.artifact,
        requirements=request.requirements, evidence=request.evidence, constraints=request.constraints,
    )
    children = tuple(
        ModelExecution(
            execution_id=f"{request_id}.worker.{i}", stage="solve", worker_id=f"worker-{i}", provider=provider, model=model, prompt=_prompt(request),
        ) for i, model in enumerate(models)
    )
    return CompositeExecution(execution_id=request_id, stage="solve", children=children, quorum=1), context


def _prompt(request: SolveRequest) -> str:
    return _format_context(request.objective, request.artifact, request.requirements, request.evidence, request.constraints)


def _format_context(objective: str, artifact: object, requirements: tuple[str, ...], evidence: tuple[object, ...], constraints: tuple[str, ...]) -> str:
    parts = [f"## Objective\n{objective}"]
    if artifact is not None:
        parts.append(f"## Artifact\n{artifact}")
    if requirements:
        parts.append("## Requirements\n" + "\n".join(f"- {x}" for x in requirements))
    if evidence:
        parts.append("## Evidence\n" + "\n".join(f"- {x}" for x in evidence))
    if constraints:
        parts.append("## Constraints\n" + "\n".join(f"- {x}" for x in constraints))
    return "\n\n".join(parts)
