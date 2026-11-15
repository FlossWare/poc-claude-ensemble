"""MCP adapter for the transport-neutral capability layer.

This module deliberately has no MCP SDK dependency. The orchestration layer
knows only about capabilities; an MCP client is supplied as a small adapter
with a call_tool method.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .core import (
    CapabilityError,
    CapabilityRequest,
    CapabilityResult,
    _normalize_name,
    _thaw,
)


class MCPClient(Protocol):
    """Minimal MCP transport contract required by CE."""

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        """Invoke one MCP tool and return its transport result."""


def _mcp_result_is_error(result: Any) -> bool:
    if isinstance(result, Mapping):
        return result.get("isError") is True
    return getattr(result, "isError", False) is True


class MCPAdapter:
    """Expose selected MCP tools as named CE capabilities."""

    def __init__(
        self,
        client: MCPClient,
        bindings: Mapping[str, str],
    ) -> None:
        self._client = client
        self._bindings: dict[str, str] = {}
        for capability, tool in bindings.items():
            normalized = _normalize_name(capability)
            if normalized in self._bindings:
                raise CapabilityError(
                    f"capability binding already defined: {normalized}"
                )
            if not isinstance(tool, str) or not tool.strip():
                raise CapabilityError(
                    f"MCP tool name must be a non-empty string: {normalized}"
                )
            self._bindings[normalized] = tool.strip()

    def has_capability(self, name: str) -> bool:
        return _normalize_name(name) in self._bindings

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        capability = _normalize_name(request.name)
        tool_name = self._bindings.get(capability)
        if tool_name is None:
            return CapabilityResult.failed(
                capability,
                "capability is not bound to an MCP tool",
            )

        try:
            data = self._client.call_tool(
                tool_name,
                _thaw(request.arguments),
            )
        except Exception as exc:
            return CapabilityResult.failed(
                capability,
                f"MCP capability failed: {type(exc).__name__}",
                metadata={"transport": "mcp", "tool": tool_name},
            )

        if _mcp_result_is_error(data):
            return CapabilityResult.failed(
                capability,
                "MCP capability reported tool failure",
                metadata={"transport": "mcp", "tool": tool_name},
            )

        return CapabilityResult.ok(
            capability,
            data,
            metadata={"transport": "mcp", "tool": tool_name},
        )

    def register_with(self, registry) -> None:
        """Register each bound MCP capability with a CE registry."""
        for capability in self._bindings:
            registry.register(capability, self.execute)
