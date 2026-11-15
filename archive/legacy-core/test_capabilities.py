from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from capabilities import (
    CapabilityError,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
    MCPAdapter,
)


def test_registry_resolves_and_executes_capability() -> None:
    registry = CapabilityRegistry()
    registry.register(
        "resource.fetch",
        lambda request: CapabilityResult.ok(
            request.name,
            {"uri": request.arguments["uri"]},
        ),
    )

    result = registry.execute(
        CapabilityRequest("RESOURCE.FETCH", {"uri": "file:///tmp/a"})
    )

    assert result.success is True
    assert result.data == {"uri": "file:///tmp/a"}
    assert registry.names() == ("resource.fetch",)


def test_registry_rejects_duplicate_and_unknown_capabilities() -> None:
    registry = CapabilityRegistry()
    handler = lambda request: CapabilityResult.ok(request.name)

    registry.register("repository.read", handler)

    with pytest.raises(CapabilityError, match="already registered"):
        registry.register("REPOSITORY.READ", handler)

    with pytest.raises(CapabilityError, match="unknown capability"):
        registry.resolve("knowledge.search")


def test_registry_isolates_handler_failure_without_leaking_exception_text() -> None:
    registry = CapabilityRegistry()

    def failing_handler(request):
        raise RuntimeError("secret token should not escape")

    registry.register("knowledge.search", failing_handler)

    result = registry.execute(CapabilityRequest("knowledge.search"))

    assert result.success is False
    assert result.error == "capability failed: RuntimeError"
    assert "secret token" not in (result.error or "")


def test_capability_request_snapshots_nested_arguments() -> None:
    arguments = {"options": {"limit": 5}, "tags": ["one", "two"]}
    request = CapabilityRequest("knowledge.search", arguments)

    arguments["options"]["limit"] = 100
    arguments["extra"] = "changed"
    arguments["tags"].append("three")

    assert request.arguments["options"]["limit"] == 5
    assert "extra" not in request.arguments
    assert request.arguments["tags"] == ("one", "two")

    with pytest.raises(TypeError):
        request.arguments["new"] = "value"

    with pytest.raises(TypeError):
        request.arguments["options"]["limit"] = 10


def test_capability_request_rejects_non_mapping_arguments() -> None:
    with pytest.raises(CapabilityError, match="arguments must be a mapping"):
        CapabilityRequest("example", ["not", "a", "mapping"])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (bytearray(b"mutable"), "unsupported capability argument type"),
        ({"one", "two"}, "unsupported capability argument type"),
        (float("nan"), "finite numbers"),
        (float("inf"), "finite numbers"),
        ({1: "non-string key"}, "mapping keys must be strings"),
    ],
)
def test_capability_request_rejects_unsupported_argument_values(value, message) -> None:
    class FailingClient:
        def call_tool(self, tool_name, arguments):
            raise AssertionError("MCP client must not be called")

    with pytest.raises(CapabilityError, match=message):
        request = CapabilityRequest("example", {"value": value})
        MCPAdapter(FailingClient(), {"example": "example_tool"}).execute(request)


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, tool_name: str, arguments) -> object:
        self.calls.append((tool_name, dict(arguments)))
        return {"ok": True}


def test_mcp_adapter_delegates_without_exposing_transport() -> None:
    client = FakeMCPClient()
    adapter = MCPAdapter(
        client,
        {"repository.read": "github_get_pull_request"},
    )
    registry = CapabilityRegistry()
    adapter.register_with(registry)

    result = registry.execute(
        CapabilityRequest(
            "repository.read",
            {"repository": "FlossWare/claude-ensemble"},
        )
    )

    assert result.success is True
    assert result.data == {"ok": True}
    assert result.metadata == {
        "transport": "mcp",
        "tool": "github_get_pull_request",
    }
    assert client.calls == [
        (
            "github_get_pull_request",
            {"repository": "FlossWare/claude-ensemble"},
        )
    ]


def test_mcp_adapter_thaws_nested_arguments_for_transport() -> None:
    class JsonClient:
        def call_tool(self, tool_name, arguments):
            json.dumps(arguments)
            arguments["options"]["limit"] = 99
            arguments["tags"].append("three")
            return {"ok": True}

    request = CapabilityRequest(
        "knowledge.search",
        {"options": {"limit": 5}, "tags": ["one", "two"]},
    )

    result = MCPAdapter(
        JsonClient(),
        {"knowledge.search": "search_knowledge"},
    ).execute(request)

    assert result.success is True
    assert request.arguments["options"]["limit"] == 5
    assert request.arguments["tags"] == ("one", "two")


def test_mcp_adapter_reports_mapped_tool_failure() -> None:
    class FailingResultClient:
        def call_tool(self, tool_name, arguments):
            return {
                "content": [{"type": "text", "text": "secret upstream details"}],
                "isError": True,
            }

    result = MCPAdapter(
        FailingResultClient(),
        {"knowledge.search": "search_knowledge"},
    ).execute(CapabilityRequest("knowledge.search"))

    assert result.success is False
    assert result.error == "MCP capability reported tool failure"
    assert "secret upstream details" not in (result.error or "")


def test_mcp_adapter_reports_object_tool_failure() -> None:
    class FailingResultClient:
        def call_tool(self, tool_name, arguments):
            return SimpleNamespace(
                content=[{"type": "text", "text": "secret upstream details"}],
                isError=True,
            )

    result = MCPAdapter(
        FailingResultClient(),
        {"knowledge.search": "search_knowledge"},
    ).execute(CapabilityRequest("knowledge.search"))

    assert result.success is False
    assert result.error == "MCP capability reported tool failure"


def test_mcp_adapter_reports_transport_failure_without_leaking_exception_text() -> None:
    class FailingClient:
        def call_tool(self, tool_name, arguments):
            raise RuntimeError("secret token should not escape")

    adapter = MCPAdapter(
        FailingClient(),
        {"knowledge.search": "search_knowledge"},
    )

    result = adapter.execute(CapabilityRequest("knowledge.search"))

    assert result.success is False
    assert result.error == "MCP capability failed: RuntimeError"
    assert result.metadata == {
        "transport": "mcp",
        "tool": "search_knowledge",
    }
    assert "secret token" not in (result.error or "")


def test_mcp_adapter_rejects_normalized_binding_collision() -> None:
    with pytest.raises(CapabilityError, match="binding already defined"):
        MCPAdapter(
            FakeMCPClient(),
            {
                "repository.read": "read_repository",
                " REPOSITORY.READ ": "different_tool",
            },
        )


def test_mcp_adapter_rejects_empty_tool_name() -> None:
    with pytest.raises(CapabilityError, match="tool name must be a non-empty string"):
        MCPAdapter(FakeMCPClient(), {"repository.read": "  "})


def test_mcp_adapter_reports_unbound_capability() -> None:
    adapter = MCPAdapter(FakeMCPClient(), {"resource.fetch": "fetch_resource"})

    result = adapter.execute(CapabilityRequest("repository.read"))

    assert result.success is False
    assert result.error == "capability is not bound to an MCP tool"
