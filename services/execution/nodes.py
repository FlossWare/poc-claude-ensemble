"""Generic recursive execution nodes.

Solve and Review are workflows built on these primitives, not separate levels
of execution. A CompositeExecution may contain any ExecutionNode, recursively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from providers.model_provider import ModelProvider


@dataclass(frozen=True)
class ExecutionNode:
    execution_id: str
    stage: str


@dataclass(frozen=True)
class ModelExecution(ExecutionNode):
    provider: ModelProvider = field(compare=False, repr=False)
    prompt: str = ""
    model: str | None = None
    system_prompt: str | None = None
    timeout: float = 300.0
    worker_id: str | None = None
    output_parser: Callable[[str], Any] | None = field(default=None, compare=False, repr=False)
    prompt_builder: Callable[[Any], str] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt and self.prompt_builder is None:
            raise ValueError("model execution requires prompt or prompt_builder")


@dataclass(frozen=True)
class PipelineExecution(ExecutionNode):
    """Sequential composition; each child receives all prior child results."""

    children: tuple[ExecutionNode, ...] = ()

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("pipeline execution requires at least one child")


@dataclass(frozen=True)
class CompositeExecution(ExecutionNode):
    children: tuple[ExecutionNode, ...] = ()
    quorum: int = 1
    aggregation: str = "collect"

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("composite execution requires at least one child")
        if self.quorum < 1 or self.quorum > len(self.children):
            raise ValueError("quorum must be between 1 and the number of children")
        if self.aggregation not in {"collect", "first-success"}:
            raise ValueError("unsupported aggregation strategy")
