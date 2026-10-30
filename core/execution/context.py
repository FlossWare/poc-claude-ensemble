"""Execution context and result contracts shared by nested workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass(frozen=True)
class ExecutionSerializationLimits:
    """Bounds applied when an execution context crosses a serialization boundary."""

    max_serialized_bytes: int = 64 * 1024
    max_prior_result_depth: int = 16
    max_artifact_bytes: int = 16 * 1024
    max_evidence_item_bytes: int = 16 * 1024

    def validate(self) -> None:
        for name, value in (
            ("max_serialized_bytes", self.max_serialized_bytes),
            ("max_prior_result_depth", self.max_prior_result_depth),
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("max_evidence_item_bytes", self.max_evidence_item_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if name == "max_prior_result_depth":
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif value < 1:
                raise ValueError(f"{name} must be positive")


DEFAULT_EXECUTION_SERIALIZATION_LIMITS = ExecutionSerializationLimits()

# Version of the persisted ExecutionContext shape used by Memory Service.
# Missing versions are treated as legacy version 0 and normalized to this version.
EXECUTION_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExecutionLimits:
    max_depth: int = 16
    max_children: int = 64
    max_total_executions: int = 256
    max_concurrent_executions: int = 16

    def validate(self) -> None:
        if self.max_depth < 0 or self.max_children < 1 or self.max_total_executions < 1 or self.max_concurrent_executions < 1:
            raise ValueError("execution limits must be positive (max_depth may be zero)")


def _json_default(value: Any) -> Any:
    """Convert supported dataclass payloads for deterministic JSON accounting."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for serialization accounting."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"execution context contains a value that cannot be serialized: {exc}"
        ) from exc


def _validate_payload_size(value: Any, max_bytes: int, label: str) -> None:
    """Reject an individual artifact/evidence payload before full serialization."""
    encoded = _canonical_json(value)
    if len(encoded) > max_bytes:
        raise ValueError(
            f"{label} exceeds maximum serialized size: "
            f"{len(encoded)} > {max_bytes} bytes"
        )


def _execution_result_from_dict(
    value: dict[str, Any],
    *,
    depth: int,
    limits: ExecutionSerializationLimits,
) -> "ExecutionResult":
    """Restore nested results without allowing unbounded recursive traversal."""
    if depth > limits.max_prior_result_depth:
        raise ValueError(
            "prior execution result nesting exceeds maximum depth: "
            f"{depth} > {limits.max_prior_result_depth}"
        )
    return ExecutionResult(
        execution_id=value["execution_id"],
        node_type=value["node_type"],
        status=ExecutionStatus(value["status"]),
        output=value.get("output"),
        error=value.get("error"),
        children=tuple(
            _execution_result_from_dict(child, depth=depth + 1, limits=limits)
            for child in value.get("children", ())
        ),
        provider=value.get("provider"),
        model=value.get("model"),
        input_tokens=value.get("input_tokens", 0),
        output_tokens=value.get("output_tokens", 0),
        cost_usd=value.get("cost_usd"),
        metadata=dict(value.get("metadata", {})),
    )


def _execution_result_to_dict(
    result: "ExecutionResult",
    *,
    depth: int,
    limits: ExecutionSerializationLimits,
) -> dict[str, Any]:
    """Serialize nested results without allowing unbounded recursive traversal."""
    if depth > limits.max_prior_result_depth:
        raise ValueError(
            "prior execution result nesting exceeds maximum depth: "
            f"{depth} > {limits.max_prior_result_depth}"
        )
    return {
        "execution_id": result.execution_id,
        "node_type": result.node_type,
        "status": result.status.value,
        "output": result.output,
        "error": result.error,
        "children": tuple(
            _execution_result_to_dict(child, depth=depth + 1, limits=limits)
            for child in result.children
        ),
        "provider": result.provider,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "metadata": result.metadata,
    }


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable context propagated through every nested execution."""

    request_id: str
    objective: str
    execution_id: str | None = None
    parent_execution_id: str | None = None
    artifact: Any = None
    requirements: tuple[str, ...] = ()
    evidence: tuple[Any, ...] = ()
    constraints: tuple[str, ...] = ()
    prior_results: tuple["ExecutionResult", ...] = ()
    lineage: tuple[str, ...] = ()
    stage: str = "root"
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_context: tuple[dict[str, Any], ...] = ()

    def child(self, *, execution_id: str, stage: str, worker_id: str | None = None, prior_results: tuple["ExecutionResult", ...] | None = None) -> "ExecutionContext":
        return replace(
            self,
            execution_id=execution_id,
            parent_execution_id=self.execution_id,
            lineage=(*self.lineage, execution_id),
            stage=stage,
            worker_id=worker_id,
            prior_results=self.prior_results if prior_results is None else prior_results,
        )

    def to_dict(
        self,
        *,
        limits: ExecutionSerializationLimits | None = None,
    ) -> dict[str, Any]:
        """Serialize the canonical context while enforcing serialization limits."""
        limits = limits or DEFAULT_EXECUTION_SERIALIZATION_LIMITS
        limits.validate()

        _validate_payload_size(
            self.artifact,
            limits.max_artifact_bytes,
            "artifact",
        )
        for index, evidence in enumerate(self.evidence):
            _validate_payload_size(
                evidence,
                limits.max_evidence_item_bytes,
                f"evidence[{index}]",
            )

        serialized = {
            "schema_version": EXECUTION_CONTEXT_SCHEMA_VERSION,
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "parent_execution_id": self.parent_execution_id,
            "lineage": self.lineage,
            "stage": self.stage,
            "worker_id": self.worker_id,
            "objective": self.objective,
            "artifact": self.artifact,
            "requirements": self.requirements,
            "evidence": self.evidence,
            "constraints": self.constraints,
            "prior_results": tuple(
                _execution_result_to_dict(result, depth=1, limits=limits)
                for result in self.prior_results
            ),
            "metadata": self.metadata,
            "memory_context": self.memory_context,
        }

        encoded = _canonical_json(serialized)
        if len(encoded) > limits.max_serialized_bytes:
            raise ValueError(
                "serialized execution context exceeds maximum size: "
                f"{len(encoded)} > {limits.max_serialized_bytes} bytes"
            )
        return serialized

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        limits: ExecutionSerializationLimits | None = None,
    ) -> "ExecutionContext":
        """Restore a canonical context while enforcing serialization limits."""
        limits = limits or DEFAULT_EXECUTION_SERIALIZATION_LIMITS
        limits.validate()
        if not isinstance(value, dict):
            raise ValueError("execution context must be a mapping")
        schema_version = value.get("schema_version", 0)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("execution context schema_version must be an integer")
        if schema_version < 0:
            raise ValueError("execution context schema_version must be non-negative")
        if schema_version > EXECUTION_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                "execution context schema version is newer than supported: "
                f"{schema_version} > {EXECUTION_CONTEXT_SCHEMA_VERSION}"
            )
        encoded = _canonical_json(value)
        if len(encoded) > limits.max_serialized_bytes:
            raise ValueError(
                "serialized execution context exceeds maximum size: "
                f"{len(encoded)} > {limits.max_serialized_bytes} bytes"
            )
        required = ("request_id", "objective")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"execution context missing required fields: {', '.join(missing)}")
        context = cls(
            request_id=value["request_id"],
            execution_id=value.get("execution_id"),
            parent_execution_id=value.get("parent_execution_id"),
            objective=value["objective"],
            artifact=value.get("artifact"),
            requirements=tuple(value.get("requirements", ())),
            evidence=tuple(value.get("evidence", ())),
            constraints=tuple(value.get("constraints", ())),
            prior_results=tuple(
                _execution_result_from_dict(item, depth=1, limits=limits)
                for item in value.get("prior_results", ())
            ),
            lineage=tuple(value.get("lineage", ())),
            stage=value.get("stage", "root"),
            worker_id=value.get("worker_id"),
            metadata=dict(value.get("metadata", {})),
            memory_context=tuple(dict(item) for item in value.get("memory_context", ())),
        )
        # Re-run the canonical serializer so restored payloads cannot bypass
        # artifact, evidence, depth, or total-size limits.
        context.to_dict(limits=limits)
        return context


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    node_type: str
    status: ExecutionStatus
    output: Any = None
    error: str | None = None
    children: tuple["ExecutionResult", ...] = ()
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        limits: ExecutionSerializationLimits | None = None,
    ) -> dict[str, Any]:
        """Serialize an execution result while enforcing serialization limits."""
        limits = limits or DEFAULT_EXECUTION_SERIALIZATION_LIMITS
        limits.validate()
        serialized = _execution_result_to_dict(self, depth=1, limits=limits)
        encoded = _canonical_json(serialized)
        if len(encoded) > limits.max_serialized_bytes:
            raise ValueError(
                "serialized execution result exceeds maximum size: "
                f"{len(encoded)} > {limits.max_serialized_bytes} bytes"
            )
        return serialized

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        limits: ExecutionSerializationLimits | None = None,
    ) -> "ExecutionResult":
        limits = limits or DEFAULT_EXECUTION_SERIALIZATION_LIMITS
        limits.validate()
        encoded = _canonical_json(value)
        if len(encoded) > limits.max_serialized_bytes:
            raise ValueError(
                "serialized execution result exceeds maximum size: "
                f"{len(encoded)} > {limits.max_serialized_bytes} bytes"
            )
        return _execution_result_from_dict(value, depth=1, limits=limits)

    @property
    def successful(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status is ExecutionStatus.FAILURE
