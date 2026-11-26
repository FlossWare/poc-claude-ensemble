"""Tests for the Claude Ensemble MCP server.

The tests start a real loopback HTTP server that mimics the gateway routes and
then drive the MCP server through the SDK's in-memory transport, so tool
listing, schemas, structured results and structured errors are all exercised
over the actual protocol rather than by calling handlers directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp import load_impl  # noqa: E402

load_impl("client")
load_impl("tools")
load_impl("server")

from mcp.ce_client import GatewayClient, GatewayError, RateLimiter  # noqa: E402
from mcp.ce_server import create_server  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

EXPECTED_TOOLS = {
    "search_knowledge",
    "search_semantic",
    "search_hybrid",
    "get_record",
    "list_origins",
    "graph_query",
    "graph_traverse",
    "get_model_status",
    "draw_crew",
    "get_learning_status",
    "health_check",
}

THOMPSON_MODELS = {
    "sonnet": {"calls": 4, "successes": 3, "failures": 1, "total_cost": 2.0,
               "total_tokens": 100, "last_updated": "2026-11-20T00:00:00Z"},
    "glm-5-3": {"calls": 2, "successes": 2, "failures": 0, "total_cost": 0.0,
                "total_tokens": 50, "last_updated": "2026-11-20T00:00:00Z"},
    "muse-spark-open": {"calls": 1, "successes": 1, "failures": 0, "total_cost": 0.0,
                        "total_tokens": 10, "last_updated": "2026-11-20T00:00:00Z"},
    "haiku": {"calls": 1, "successes": 0, "failures": 1, "total_cost": 0.1,
              "total_tokens": 10, "last_updated": "2026-11-20T00:00:00Z"},
    "opus": {"calls": 1, "successes": 1, "failures": 0, "total_cost": 1.0,
             "total_tokens": 10, "last_updated": "2026-11-20T00:00:00Z"},
}
DRAW_ORDER = ["sonnet", "glm-5-3", "muse-spark-open", "haiku", "opus"]


class MockGateway:
    """Loopback HTTP server implementing the gateway routes used by the tools."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._draw_index = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MockGateway/1"

            def log_message(self, fmt, *args):  # noqa: D401 - keep test output quiet
                return

            def do_GET(self):
                outer._dispatch(self, "GET")

            def do_POST(self):
                outer._dispatch(self, "POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    # -- request handling --------------------------------------------------

    def _dispatch(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        parsed = urllib.parse.urlsplit(handler.path)
        length = int(handler.headers.get("Content-Length", "0") or 0)
        body = json.loads(handler.rfile.read(length).decode("utf-8")) if length else None
        with self._lock:
            self.requests.append(
                {
                    "method": method,
                    "path": parsed.path,
                    "query": urllib.parse.parse_qs(parsed.query),
                    "body": body,
                    "authorization": handler.headers.get("Authorization"),
                }
            )
        status, payload = self._route(method, parsed.path, parsed.query, body)
        raw = (json.dumps(payload) + "\n").encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def _route(self, method: str, path: str, query: str, body: Any):
        if method == "GET" and path == "/api/v1/health":
            return 200, {"ok": True, "service": "claude-ensemble"}
        if method == "GET" and path == "/api/v1/ready":
            return 200, {"ok": True, "ready": True, "dependencies": {"graph": {"ok": True}}}
        if method == "GET" and path == "/api/v1/services":
            return 200, {
                "ok": False,
                "counts": {"healthy": 2, "degraded": 1, "unavailable": 0, "total": 3},
                "services": [
                    {"id": "graph", "status": "healthy", "configured": True},
                    {"id": "memory", "status": "healthy", "configured": True},
                    {"id": "lexical", "status": "degraded", "configured": True},
                ],
            }
        if method == "POST" and path == "/api/v1/lexical/knowledge/search":
            if (body or {}).get("query") == "trigger-error":
                return 503, {
                    "ok": False,
                    "error": {"code": "service_unavailable", "message": "unable to open database file"},
                }
            return 200, {
                "ok": True,
                "results": [
                    {
                        "vector_id": "k1",
                        "chunk_id": "c1",
                        "url": "https://example.test/rfc",
                        "source": "rfc",
                        "title": "Example RFC",
                        "text": "thompson sampling for model routing",
                        "key": "k1",
                    }
                ],
            }
        if method == "POST" and path == "/api/v1/embedding/embeddings":
            return 200, {"ok": True, "model": "mock", "dimensions": 4,
                         "embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        if method == "POST" and path == "/api/v1/vector/collections/knowledge_chunks/query":
            assert body and len(body["embedding"]) == 4
            return 200, {
                "ok": True,
                "results": [
                    {"id": "v1", "document": "semantic hit", "score": 0.91,
                     "metadata": {"source": "rfc", "title": "Example RFC"}}
                ],
            }
        if method == "POST" and path == "/api/v1/hybrid/search":
            return 200, {
                "ok": True,
                "results": [{"key": "t1", "text": "transcript hit", "source": "vector"}],
                "knowledge": [{"key": "k1", "text": "knowledge hit", "source": "knowledge"}],
                "degraded": ["lexical"],
                "reranked": True,
                "related_sessions": [{"session_id": "s1"}],
                "intent": {"intent": "code"},
            }
        if method == "GET" and path == "/api/v1/corpus/records":
            parameters = urllib.parse.parse_qs(query)
            source_id = (parameters.get("source_id") or [""])[0]
            if source_id == "missing":
                return 200, {"ok": True, "records": [], "next_cursor": None}
            if (parameters.get("cursor") or [""])[0]:
                # page 2: the NEWEST revision lives here, not on page 1 —
                # regression coverage for get_record pagination
                return 200, {
                    "ok": True,
                    "next_cursor": None,
                    "records": [
                        {"source_id": source_id, "revision_sequence": 3,
                         "title": "newest",
                         "origin": "inline-text", "locator": "three",
                         "content": "newest content",
                         "content_length": 14, "content_truncated": False},
                    ],
                }
            return 200, {
                "ok": True,
                "next_cursor": "page2",
                "records": [
                    {"source_id": source_id, "revision_sequence": 1, "title": "old",
                     "origin": "inline-text", "locator": "one", "content": "old content",
                     "content_length": 11, "content_truncated": False},
                    {"source_id": source_id, "revision_sequence": 2, "title": "new",
                     "origin": "inline-text", "locator": "two", "content": "newest content",
                     "content_length": 14, "content_truncated": False},
                ],
            }
        if method == "GET" and path == "/api/v1/corpus/origins":
            return 200, {"ok": True, "origins": [{"origin": "inline-text", "count": 42}]}
        if method == "POST" and path == "/api/v1/graph/query":
            if (body or {}).get("type") == "edges":
                return 200, {"ok": True, "edges": [
                    {"id": "e1", "from_id": "model:sonnet", "to_id": "task:review",
                     "relationship": "used_for", "properties": {}}
                ]}
            return 200, {"ok": True, "nodes": [
                {"id": "model:sonnet", "type": "model", "properties": {"family": "claude"}}
            ]}
        if method == "POST" and path == "/api/v1/graph/traverse":
            if (body or {}).get("start") == "missing":
                return 400, {"ok": False, "error": "node not found: missing"}
            return 200, {"ok": True, "results": [
                {"depth": 1, "node": {"id": "task:review"}},
                {"depth": 1, "node": {"id": "task:code"}},
                {"depth": 2, "node": {"id": "model:sonnet"}},
            ]}
        if method == "POST" and path == "/api/v1/thompson":
            action = (body or {}).get("action")
            if action == "ping":
                return 200, {"ok": True, "message": "pong"}
            if action == "get_state":
                return 200, {
                    "ok": True,
                    "state": {"last_updated": "2026-11-20T00:00:00Z", "models": THOMPSON_MODELS},
                }
            if action == "select_model":
                with self._lock:
                    model = DRAW_ORDER[self._draw_index % len(DRAW_ORDER)]
                    self._draw_index += 1
                return 200, {"ok": True, "model": model}
            return 200, {"ok": False, "error": f"Unknown action: {action}"}
        if method == "POST" and path == "/api/v1/learning":
            operation = (body or {}).get("op")
            if operation == "get_report":
                return 200, {"ok": True, "report": "2 outcomes recorded", "total_outcomes": 2}
            if operation == "get_recent_outcomes":
                return 200, {"ok": True, "outcomes": [{"task_id": "t1", "rating": 1.0}]}
            return 200, {"ok": False, "error": f"Unknown operation: {operation}"}
        return 404, {"ok": False, "error": {"code": "not_found", "message": "unknown path"}}


@pytest.fixture()
def gateway():
    mock = MockGateway()
    mock.start()
    try:
        yield mock
    finally:
        mock.stop()


def make_client(gateway: MockGateway) -> GatewayClient:
    return GatewayClient(
        base_url=gateway.url,
        timeout=5.0,
        rate_limiter=RateLimiter(0),
        auth_token="",
    )


def call_tool(gateway: MockGateway, name: str, arguments: dict[str, Any] | None = None):
    async def scenario():
        server = create_server(make_client(gateway))
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(name, arguments or {})

    return asyncio.run(scenario())


def structured(result) -> dict[str, Any]:
    assert result.structuredContent is not None, result.content
    return result.structuredContent


def error_envelope(result) -> dict[str, Any]:
    assert result.isError is True
    assert result.structuredContent is None
    return json.loads(result.content[0].text)


# -- protocol and schema ---------------------------------------------------


def test_tool_listing_exposes_all_tools_with_schemas(gateway):
    async def scenario():
        server = create_server(make_client(gateway))
        async with create_connected_server_and_client_session(server) as session:
            return await session.list_tools()

    listing = asyncio.run(scenario())
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == EXPECTED_TOOLS
    for name, tool in tools.items():
        assert tool.description, name
        assert tool.inputSchema["type"] == "object", name
        assert tool.outputSchema is not None, name
        assert tool.outputSchema["type"] == "object", name
        assert tool.annotations is not None, name
        assert tool.annotations.readOnlyHint is not None, name
    # draw_crew is the one tool with a local side effect (the exposure ledger).
    assert tools["draw_crew"].annotations.readOnlyHint is False
    assert tools["search_knowledge"].annotations.readOnlyHint is True
    assert "limit" in tools["search_knowledge"].inputSchema["properties"]
    assert "source_id" in tools["get_record"].inputSchema["properties"]
    assert "query_type" in tools["graph_query"].inputSchema["properties"]
    assert "limits" in tools["graph_traverse"].inputSchema["properties"]


# -- retrieval -------------------------------------------------------------


def test_search_knowledge_calls_knowledge_lexical_route(gateway):
    result = call_tool(gateway, "search_knowledge", {"query": "thompson", "limit": 5})
    payload = structured(result)
    assert payload["ok"] is True and payload["count"] == 1
    assert payload["results"][0]["source"] == "rfc"
    request = gateway.requests[-1]
    assert request["path"] == "/api/v1/lexical/knowledge/search"
    assert request["body"] == {"query": "thompson", "top_k": 5}


def test_search_semantic_embeds_then_queries_knowledge_collection(gateway):
    result = call_tool(gateway, "search_semantic", {"query": "meaning", "limit": 3})
    payload = structured(result)
    assert payload["method"] == "embedding+vector_query"
    assert payload["results"][0]["text"] == "semantic hit"
    assert payload["results"][0]["score"] == 0.91
    paths = [request["path"] for request in gateway.requests]
    assert paths == [
        "/api/v1/embedding/embeddings",
        "/api/v1/vector/collections/knowledge_chunks/query",
    ]
    assert gateway.requests[0]["body"]["mode"] == "query"


def test_search_hybrid_preserves_degraded_and_knowledge(gateway):
    result = call_tool(gateway, "search_hybrid", {"query": "hybrid", "limit": 4})
    payload = structured(result)
    assert payload["degraded"] == ["lexical"]
    assert payload["reranked"] is True
    assert payload["knowledge"][0]["key"] == "k1"
    assert payload["related_sessions"] == [{"session_id": "s1"}]


def test_get_record_returns_newest_revision(gateway):
    result = call_tool(gateway, "get_record", {"source_id": "text://sha256/abc"})
    payload = structured(result)
    # both pages are fetched now: 3 records, newest on page 2
    assert payload["revision_count"] == 3
    assert payload["record"]["revision_sequence"] == 3
    assert payload["record"]["content"] == "newest content"
    assert gateway.requests[-1]["query"]["source_id"] == ["text://sha256/abc"]


def test_list_origins(gateway):
    payload = structured(call_tool(gateway, "list_origins", {}))
    assert payload["count"] == 1
    assert payload["origins"] == [{"origin": "inline-text", "count": 42}]


# -- graph -----------------------------------------------------------------


def test_graph_query_nodes_and_edges(gateway):
    nodes = structured(call_tool(gateway, "graph_query", {"query_type": "model", "params": {"family": "claude"}}))
    assert nodes["count"] == 1 and nodes["nodes"][0]["id"] == "model:sonnet"
    assert gateway.requests[-1]["body"] == {"type": "model", "properties": {"family": "claude"}}

    edges = structured(call_tool(gateway, "graph_query", {"query_type": "edges", "params": {"relationship": "used_for"}}))
    assert edges["count"] == 1 and edges["edges"][0]["relationship"] == "used_for"
    assert gateway.requests[-1]["body"] == {"type": "edges", "relationship": "used_for"}


def test_graph_traverse_enforces_max_results(gateway):
    payload = structured(
        call_tool(
            gateway,
            "graph_traverse",
            {"start_node": "model:sonnet", "depth": 2, "limits": {"direction": "both", "max_results": 2}},
        )
    )
    assert payload["count"] == 2
    assert payload["truncated"] is True
    assert payload["direction"] == "both"
    assert gateway.requests[-1]["body"] == {
        "start": "model:sonnet",
        "direction": "both",
        "max_depth": 2,
        "edge_type": None,
    }


# -- learning --------------------------------------------------------------


def test_get_model_status_derives_beta_posteriors(gateway):
    payload = structured(call_tool(gateway, "get_model_status", {}))
    assert payload["arm_count"] == len(THOMPSON_MODELS)
    sonnet = payload["arms"]["sonnet"]
    assert sonnet["alpha"] == 4.0
    assert sonnet["beta"] == 2.0
    assert sonnet["posterior_mean"] == pytest.approx(0.666667, abs=1e-6)
    assert sonnet["avg_cost"] == 0.5


def test_draw_crew_uses_crew_select_and_excludes_arbiter(gateway, tmp_path, monkeypatch):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR", str(tmp_path))
    payload = structured(
        call_tool(gateway, "draw_crew", {"task_type": "code-review", "workers": 2, "arbiter": "opus"})
    )
    selection = payload["selection"]
    assert selection["task_type"] == "code-review"
    assert selection["status"] == "thompson"
    assert len(selection["workers"]) == 2
    assert all(worker["arm"] != "opus" for worker in selection["workers"])
    assert len({worker["arm"] for worker in selection["workers"]}) == 2
    ledger_files = list(tmp_path.glob("*.jsonl"))
    assert len(ledger_files) == 1
    assert json.loads(ledger_files[0].read_text())["selection_id"] == selection["selection_id"]


def test_get_learning_status(gateway):
    payload = structured(call_tool(gateway, "get_learning_status", {}))
    assert payload["report"] == "2 outcomes recorded"
    assert payload["total_outcomes"] == 2
    assert payload["recent_outcomes"] == [{"task_id": "t1", "rating": 1.0}]
    assert payload["errors"] == []


def test_health_check_reports_degraded_service(gateway):
    payload = structured(call_tool(gateway, "health_check", {}))
    assert payload["gateway"]["service"] == "claude-ensemble"
    assert payload["ready"]["ready"] is True
    assert payload["degraded"] == [{"id": "lexical", "status": "degraded"}]
    assert payload["counts"]["degraded"] == 1


# -- structured errors -----------------------------------------------------


def test_gateway_error_is_structured(gateway):
    envelope = error_envelope(call_tool(gateway, "search_knowledge", {"query": "trigger-error"}))
    assert envelope["ok"] is False
    assert envelope["tool"] == "search_knowledge"
    assert envelope["error"]["code"] == "service_unavailable"
    assert envelope["error"]["status"] == 503
    assert "unable to open database file" in envelope["error"]["message"]


def test_missing_record_is_not_found(gateway):
    envelope = error_envelope(call_tool(gateway, "get_record", {"source_id": "missing"}))
    assert envelope["error"]["code"] == "not_found"
    assert envelope["error"]["status"] == 404


def test_invalid_arguments_use_structured_error(gateway):
    envelope = error_envelope(call_tool(gateway, "search_knowledge", {"query": "x", "limit": 0}))
    assert envelope["error"]["code"] == "invalid_arguments"
    assert isinstance(envelope["error"]["details"], list)


def test_unknown_tool_is_structured(gateway):
    envelope = error_envelope(call_tool(gateway, "not_a_tool", {}))
    assert envelope["error"]["code"] == "unknown_tool"


def test_graph_traverse_missing_node_surfaces_upstream_code(gateway):
    envelope = error_envelope(
        call_tool(gateway, "graph_traverse", {"start_node": "missing", "depth": 1})
    )
    assert envelope["error"]["status"] == 400
    assert envelope["error"]["message"] == "node not found: missing"


# -- client safety ---------------------------------------------------------


def test_non_loopback_gateway_is_rejected(monkeypatch):
    monkeypatch.delenv("POC_CLAUDE_ENSEMBLE_MCP_ALLOW_REMOTE_GATEWAY", raising=False)
    with pytest.raises(ValueError, match="loopback"):
        GatewayClient(base_url="http://example.com:8080", rate_limiter=RateLimiter(0))


def test_client_rejects_absolute_and_traversal_paths(gateway):
    client = make_client(gateway)
    with pytest.raises(ValueError):
        client.get("http://example.com/api/v1/health")
    with pytest.raises(ValueError):
        client.get("/api/v1/../secret")


def test_rate_limiter_delays_after_burst():
    limiter = RateLimiter(rate=20.0, burst=1.0)
    limiter.acquire()
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started >= 0.03


def test_gateway_error_round_trips_details():
    error = GatewayError("service_unavailable", "nope", status=503, details={"ok": False})
    assert error.to_dict() == {
        "code": "service_unavailable",
        "message": "nope",
        "status": 503,
        "details": {"ok": False},
    }


def test_get_record_paginates_to_true_newest(gateway):
    """Regression: the newest revision sat on page 2; the old single-page
    max() returned the page-1 maximum instead."""
    client = make_client(gateway)
    result = client.get_record("text://sha256/abc")
    assert result["record"]["revision_sequence"] == 3
    assert result["record"]["locator"] == "three"


def test_client_refuses_redirects():
    """Regression: urlopen followed 3xx and re-sent the Authorization
    header to the redirect target, which the loopback check never
    validated. Redirects must surface as gateway errors instead."""
    class Redirector(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.send_response(302)
            self.send_header("Location", "http://example.com/api/v1/health")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = GatewayClient(
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            timeout=5.0,
            rate_limiter=RateLimiter(0),
            auth_token="secret-token",
        )
        with pytest.raises(GatewayError):
            client.get("/api/v1/health")
    finally:
        server.shutdown()


def test_call_models_parallel_empty_roster_is_noop():
    """Regression: ThreadPoolExecutor(max_workers=0) raised ValueError on
    an empty model list; an empty roster is a valid no-op."""
    import sys as _sys
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(_file_))) if False else _REPO_ROOT
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from services.arbitration.api_client import MultiModelClient

    client = MultiModelClient()
    assert client.call_models_parallel([], "hello") == {}
