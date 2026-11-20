"""Claude Ensemble MCP server (stdio transport).

Run from the repository root::

    python3 -m mcp

The process speaks the Model Context Protocol on stdin/stdout.  Logs go to
stderr only; stdout is reserved for protocol messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from mcp import __version__, load_impl

# Load the local implementation modules under mcp.ce_* before importing them.
load_impl("client")
load_impl("tools")

from mcp.ce_client import GatewayClient, GatewayError  # noqa: E402
from mcp.ce_tools import ToolSpec, build_tool_specs  # noqa: E402
from mcp.server import Server  # noqa: E402  (official SDK)
from mcp.server.stdio import stdio_server  # noqa: E402  (official SDK)
from mcp.types import CallToolResult, TextContent  # noqa: E402  (official SDK)
from pydantic import ValidationError  # noqa: E402

LOG = logging.getLogger("claude_ensemble_mcp")
SERVER_NAME = "claude-ensemble"

INSTRUCTIONS = (
    "Claude Ensemble exposes its local knowledge corpus, knowledge graph, "
    "Thompson router and learning stack. All calls go through the loopback "
    "gateway; no provider credentials are held by this server."
)


def _json_default(value: object) -> str:
    return str(value)


def _error_result(tool: str, error: GatewayError | dict[str, object]) -> CallToolResult:
    payload = error.to_dict() if isinstance(error, GatewayError) else dict(error)
    envelope = {"ok": False, "tool": tool, "error": payload}
    return CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text",
                text=json.dumps(envelope, indent=2, default=_json_default),
            )
        ],
    )


def _invalid_arguments(tool: str, message: str, details: object = None) -> CallToolResult:
    error: dict[str, object] = {"code": "invalid_arguments", "message": message}
    if details is not None:
        error["details"] = details
    return _error_result(tool, error)


def create_server(
    client: GatewayClient | None = None,
    specs: list[ToolSpec] | None = None,
) -> Server:
    """Build an MCP ``Server`` bound to ``client`` (injectable for tests)."""
    client = client if client is not None else GatewayClient()
    specs = specs if specs is not None else build_tool_specs(client)
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("tool names must be unique")

    server: Server = Server(
        SERVER_NAME,
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.list_tools()
    async def list_tools():
        return [spec.to_tool() for spec in specs]

    # Input is validated by the per-tool Pydantic model so validation failures
    # use the same structured error envelope as gateway failures.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, object]):
        spec = by_name.get(name)
        if spec is None:
            return _error_result(
                name, {"code": "unknown_tool", "message": f"no such tool: {name}"}
            )
        try:
            parsed = spec.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return _invalid_arguments(
                name,
                "arguments do not match the tool schema",
                exc.errors(include_url=False),
            )
        try:
            raw = await asyncio.to_thread(spec.handler, parsed)
            result = spec.output_model.model_validate(raw).model_dump(
                mode="json", exclude_none=True
            )
        except GatewayError as exc:
            LOG.warning("%s failed: %s: %s", name, exc.code, exc.message)
            return _error_result(name, exc)
        except ValueError as exc:
            return _invalid_arguments(name, str(exc))
        except ValidationError as exc:
            LOG.error("%s returned a result that violates its output schema", name)
            return _error_result(
                name,
                {
                    "code": "invalid_tool_output",
                    "message": "tool result did not match its declared output schema",
                    "details": exc.errors(include_url=False),
                },
            )
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
            LOG.exception("%s failed unexpectedly", name)
            return _error_result(
                name,
                {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
        return CallToolResult(
            isError=False,
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=_json_default),
                )
            ],
            structuredContent=result,
        )

    return server


async def _serve(client: GatewayClient) -> None:
    server = create_server(client)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m mcp",
        description="Claude Ensemble MCP server (stdio transport).",
    )
    parser.add_argument(
        "--gateway",
        default=None,
        help=(
            "Gateway base URL. Defaults to POC_CLAUDE_ENSEMBLE_GATEWAY_URL or "
            "http://127.0.0.1:8080. Loopback only unless explicitly overridden."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("POC_CLAUDE_ENSEMBLE_MCP_LOG_LEVEL", "WARNING"),
        help="stderr log level (default: WARNING).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        client = GatewayClient(base_url=args.gateway)
    except ValueError as exc:
        LOG.error("invalid gateway configuration: %s", exc)
        return 2
    LOG.info("starting %s MCP server; gateway=%s", SERVER_NAME, client.base_url)
    try:
        asyncio.run(_serve(client))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
