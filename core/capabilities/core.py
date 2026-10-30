"""Transport-neutral capability contracts and registry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapabilityError("capability arguments must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CapabilityError("capability argument mapping keys must be strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise CapabilityError(
        f"unsupported capability argument type: {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    """Return an independent transport-ready copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise CapabilityError("capability name must be a non-empty string")
    return name.strip().lower()


@dataclass(frozen=True)
class CapabilityRequest:
    """Immutable request presented to a capability implementation."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        if not isinstance(self.arguments, Mapping):
            raise CapabilityError("capability arguments must be a mapping")
        object.__setattr__(self, "arguments", _freeze(self.arguments))


@dataclass(frozen=True)
class CapabilityResult:
    """Transport-neutral result returned by a capability."""

    capability: str
    success: bool
    data: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        capability: str,
        data: Any = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapabilityResult:
        return cls(
            capability=capability,
            success=True,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        capability: str,
        error: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapabilityResult:
        return cls(
            capability=capability,
            success=False,
            error=error,
            metadata=metadata or {},
        )


class Capability(Protocol):
    """Protocol implemented by native or transport-backed capabilities."""

    name: str

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        """Execute a capability request."""


class CapabilityError(RuntimeError):
    """Raised for invalid capability registration or resolution."""


CapabilityHandler = Callable[[CapabilityRequest], CapabilityResult]


class CapabilityRegistry:
    """Resolve named capabilities without exposing their transport."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityHandler] = {}

    def register(self, name: str, handler: CapabilityHandler) -> None:
        normalized = _normalize_name(name)
        if normalized in self._capabilities:
            raise CapabilityError(f"capability already registered: {normalized}")
        self._capabilities[normalized] = handler

    def replace(self, name: str, handler: CapabilityHandler) -> None:
        self._capabilities[_normalize_name(name)] = handler

    def resolve(self, name: str) -> CapabilityHandler:
        normalized = _normalize_name(name)
        try:
            return self._capabilities[normalized]
        except KeyError as exc:
            raise CapabilityError(f"unknown capability: {normalized}") from exc

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        handler = self.resolve(request.name)
        try:
            result = handler(request)
        except Exception as exc:
            return CapabilityResult.failed(
                request.name,
                f"capability failed: {type(exc).__name__}",
            )
        if not isinstance(result, CapabilityResult):
            raise CapabilityError(
                f"capability {request.name!r} returned an invalid result"
            )
        return result

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))
