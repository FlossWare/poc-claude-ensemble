"""HTTP client for the Claude Ensemble gateway.

All tools go through the loopback gateway (default ``http://127.0.0.1:8080``);
the MCP server never holds provider credentials and never talks to a backing
service directly.  Request/response handling mirrors the patterns used by the
existing ``services/shared`` clients:

* ``urllib`` only, no extra HTTP dependency;
* a bounded timeout on every request;
* service error envelopes normalised into :class:`GatewayError`;
* a token-bucket rate limiter so one MCP client cannot spam the gateway.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_RATE_PER_SECOND = 8.0
DEFAULT_RATE_BURST = 8.0
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
API_PREFIX = "/api/v1/"
MAX_QUERY_CHARS = 2_000
MAX_LIMIT = 100
MAX_SOURCE_ID_CHARS = 512
DEFAULT_KNOWLEDGE_COLLECTION = "knowledge_chunks"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GatewayError(Exception):
    """A structured, client-visible gateway failure.

    ``code`` is stable enough for an MCP client to branch on; ``message`` is
    human-readable; ``status`` is the upstream HTTP status when one was seen;
    ``details`` is the parsed upstream body (never a traceback).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.status is not None:
            payload["status"] = self.status
        if self.details is not None:
            payload["details"] = self.details
        return payload


class RateLimiter:
    """Thread-safe token bucket.

    ``rate`` tokens are added per second up to ``burst``.  ``rate <= 0``
    disables limiting (useful for tests and for a single-user local run).
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        self.rate = float(rate)
        self.burst = float(burst if burst is not None else max(rate, 1.0))
        self._tokens = self.burst
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._updated)
                self._updated = now
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(min(wait, 1.0))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _validate_loopback(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("gateway URL must be an http(s) URL with a host")
    allow_remote = os.environ.get("POC_CLAUDE_ENSEMBLE_MCP_ALLOW_REMOTE_GATEWAY", "").lower()
    if parsed.hostname not in LOOPBACK_HOSTS and allow_remote not in {"1", "true", "yes"}:
        raise ValueError(
            "gateway URL must be loopback (127.0.0.1, ::1, localhost); set "
            "POC_CLAUDE_ENSEMBLE_MCP_ALLOW_REMOTE_GATEWAY=1 to override"
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("gateway URL must not include a path, query, or fragment")
    return base_url.rstrip("/")


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query is limited to {MAX_QUERY_CHARS} characters")
    return query


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return limit


def _error_from(payload: Any, status: int | None) -> GatewayError:
    """Normalise every CE error envelope into one exception shape."""
    code: str | None = None
    message: str | None = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else None
            message = error.get("message") if isinstance(error.get("message"), str) else None
        elif isinstance(error, str):
            message = error
        error_code = payload.get("error_code")
        if isinstance(error_code, str) and error_code:
            code = error_code
        if message is None and isinstance(payload.get("message"), str):
            message = payload["message"]
    if code is None:
        if status == 404:
            code = "not_found"
        elif status == 400:
            code = "invalid_request"
        elif status == 429:
            code = "rate_limited"
        elif status is not None and status >= 500:
            code = "service_unavailable"
        else:
            code = "gateway_error"
    return GatewayError(
        code,
        message or (f"gateway returned HTTP {status}" if status else "gateway request failed"),
        status=status,
        details=payload,
    )


class GatewayClient:
    """Small, bounded client for the Claude Ensemble gateway REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        rate_limiter: RateLimiter | None = None,
        auth_token: str | None = None,
        user_agent: str = "claude-ensemble-mcp/1.0",
    ) -> None:
        self.base_url = _validate_loopback(
            base_url or os.environ.get("POC_CLAUDE_ENSEMBLE_GATEWAY_URL", DEFAULT_GATEWAY_URL)
        )
        self.timeout = (
            float(timeout)
            if timeout is not None
            else _env_float("POC_CLAUDE_ENSEMBLE_MCP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = RateLimiter(
                _env_float("POC_CLAUDE_ENSEMBLE_MCP_RATE_PER_SECOND", DEFAULT_RATE_PER_SECOND),
                _env_float("POC_CLAUDE_ENSEMBLE_MCP_RATE_BURST", DEFAULT_RATE_BURST),
            )
        self.auth_token = (
            auth_token
            if auth_token is not None
            else os.environ.get("POC_CLAUDE_ENSEMBLE_COLLABORATION_AUTH_TOKEN", "")
        )
        self.user_agent = user_agent
        self.knowledge_collection = os.environ.get(
            "POC_CLAUDE_ENSEMBLE_MCP_KNOWLEDGE_COLLECTION", DEFAULT_KNOWLEDGE_COLLECTION
        )

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(path, str) or not path.startswith(API_PREFIX):
            raise ValueError(f"gateway path must start with {API_PREFIX}")
        # Reject absolute URLs and path traversal before they reach urllib.
        if "://" in path or ".." in path.split("?", 1)[0].split("/"):
            raise ValueError("gateway path must be a relative /api/v1/ path")

        url = self.base_url + path
        if params:
            query = urllib.parse.urlencode(
                {key: value for key, value in params.items() if value is not None}, doseq=True
            )
            if query:
                url = f"{url}?{query}"

        data = None
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self.rate_limiter.acquire()
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(response.status)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayError(
                "gateway_unavailable",
                f"cannot reach the Claude Ensemble gateway at {self.base_url}: {exc}",
            ) from exc

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayError(
                "invalid_gateway_response",
                "gateway returned a non-JSON response",
                status=status,
                details={"body_prefix": raw[:256].decode("utf-8", "replace")},
            ) from exc
        if not isinstance(payload, dict):
            raise GatewayError(
                "invalid_gateway_response",
                "gateway returned a non-object JSON response",
                status=status,
                details={"type": type(payload).__name__},
            )
        # /api/v1/services deliberately returns HTTP 200 with ok=false when a
        # backing service is degraded; callers that want that payload set
        # allow_not_ok=True.  HTTP failures are always errors.
        if status >= 400 or (payload.get("ok") is False and not allow_not_ok):
            raise _error_from(payload, status)
        return payload

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        return self._request("GET", path, params=params, allow_not_ok=allow_not_ok)

    def post(self, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, body=body)

    # -- knowledge & retrieval --------------------------------------------

    def search_knowledge(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Exact-term search over the knowledge corpus.

        The gateway route is ``/api/v1/lexical/knowledge/search``.  The shorter
        ``/api/v1/lexical/search`` route targets transcript messages at the
        lexical service, not knowledge chunks, so it is not used here.
        """
        query, limit = _validate_query(query), _validate_limit(limit)
        payload = self.post(
            "/api/v1/lexical/knowledge/search", {"query": query, "top_k": limit}
        )
        results = payload.get("results") or []
        return {
            "ok": True,
            "tool": "search_knowledge",
            "query": query,
            "limit": limit,
            "count": len(results),
            "results": results,
        }

    def search_semantic(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Dense-vector search over the knowledge collection.

        No gateway ``/api/v1/vector/search`` route exists in this deployment:
        the vector service exposes only collection operations.  Semantic search
        is therefore two gateway calls -- embed the query, then query the
        knowledge collection -- which is the same path the hybrid service uses.
        Set ``POC_CLAUDE_ENSEMBLE_MCP_SEMANTIC_PATH`` to use a single deployed
        route instead if one is added later.
        """
        query, limit = _validate_query(query), _validate_limit(limit)
        direct_path = os.environ.get("POC_CLAUDE_ENSEMBLE_MCP_SEMANTIC_PATH", "").strip()
        if direct_path:
            payload = self.post(direct_path, {"query": query, "top_k": limit})
            results = payload.get("results") or []
            method = "direct"
        else:
            embedded = self.post(
                "/api/v1/embedding/embeddings", {"texts": [query], "mode": "query"}
            )
            vectors = embedded.get("embeddings")
            if not isinstance(vectors, list) or not vectors or not isinstance(vectors[0], list):
                raise GatewayError(
                    "invalid_gateway_response",
                    "embedding service returned no query vector",
                    details={"keys": sorted(embedded)},
                )
            payload = self.post(
                f"/api/v1/vector/collections/{self.knowledge_collection}/query",
                {"embedding": vectors[0], "top_k": limit},
            )
            results = [
                {
                    "id": item.get("id"),
                    "text": item.get("document"),
                    "score": item.get("score"),
                    "metadata": item.get("metadata") or {},
                    "source": "knowledge",
                }
                for item in (payload.get("results") or [])
            ]
            method = "embedding+vector_query"
        return {
            "ok": True,
            "tool": "search_semantic",
            "query": query,
            "limit": limit,
            "count": len(results),
            "method": method,
            "results": results,
        }

    def search_hybrid(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Hybrid transcript + knowledge retrieval through the hybrid service."""
        query, limit = _validate_query(query), _validate_limit(limit)
        payload = self.post("/api/v1/hybrid/search", {"query": query, "top_k": limit})
        results = payload.get("results") or []
        knowledge = payload.get("knowledge") or []
        return {
            "ok": True,
            "tool": "search_hybrid",
            "query": query,
            "limit": limit,
            "count": len(results),
            "results": results,
            "knowledge": knowledge,
            "degraded": payload.get("degraded") or [],
            "reranked": payload.get("reranked"),
            "related_sessions": payload.get("related_sessions") or [],
            "intent": payload.get("intent"),
        }

    def get_record(self, source_id: str) -> dict[str, Any]:
        """Return the newest ingested revision for one corpus ``source_id``."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if len(source_id) > MAX_SOURCE_ID_CHARS:
            raise ValueError(f"source_id is limited to {MAX_SOURCE_ID_CHARS} characters")
        payload = self.get(
            "/api/v1/corpus/records",
            {"source_id": source_id, "limit": 1000},
        )
        records = payload.get("records") or []
        if not records:
            raise GatewayError(
                "not_found",
                f"no corpus record for source_id {source_id!r}",
                status=404,
                details={"source_id": source_id},
            )
        latest = max(records, key=lambda item: int(item.get("revision_sequence") or 0))
        return {
            "ok": True,
            "tool": "get_record",
            "source_id": source_id,
            "record": latest,
            "revision_count": len(records),
            "page_truncated": bool(payload.get("next_cursor")),
        }

    def list_origins(self) -> dict[str, Any]:
        payload = self.get("/api/v1/corpus/origins")
        origins = payload.get("origins") or []
        return {
            "ok": True,
            "tool": "list_origins",
            "count": len(origins),
            "origins": origins,
        }

    # -- graph -------------------------------------------------------------

    def graph_query(self, query_type: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Query graph nodes by type/properties, or edges by endpoint/relationship.

        ``query_type="edges"`` treats ``params`` as ``from_type``/``to_type``/
        ``relationship``.  Any other value is treated as a node type and
        ``params`` as exact property filters (the graph service semantics).
        """
        if not isinstance(query_type, str) or not query_type.strip():
            raise ValueError("query_type must be a non-empty string")
        params = dict(params or {})
        if query_type == "edges":
            body = {"type": "edges", **params}
            payload = self.post("/api/v1/graph/query", body)
            edges = payload.get("edges") or []
            return {
                "ok": True,
                "tool": "graph_query",
                "query_type": query_type,
                "count": len(edges),
                "nodes": [],
                "edges": edges,
            }
        payload = self.post(
            "/api/v1/graph/query", {"type": query_type, "properties": params}
        )
        nodes = payload.get("nodes") or []
        return {
            "ok": True,
            "tool": "graph_query",
            "query_type": query_type,
            "count": len(nodes),
            "nodes": nodes,
            "edges": [],
        }

    def graph_traverse(
        self,
        start_node: str,
        depth: int = 1,
        limits: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bounded traversal from ``start_node``.

        ``limits`` may contain ``direction`` (out/in/both), ``edge_type`` and
        ``max_results``.  The graph service has no result cap, so
        ``max_results`` is enforced here after the response arrives.
        """
        if not isinstance(start_node, str) or not start_node.strip():
            raise ValueError("start_node must be a non-empty string")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 100:
            raise ValueError("depth must be an integer between 1 and 100")
        limits = dict(limits or {})
        direction = limits.get("direction", "out")
        if direction not in {"out", "in", "both"}:
            raise ValueError("limits.direction must be out, in, or both")
        edge_type = limits.get("edge_type")
        if edge_type is not None and not isinstance(edge_type, str):
            raise ValueError("limits.edge_type must be a string")
        max_results = limits.get("max_results")
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
                raise ValueError("limits.max_results must be a positive integer")
            max_results = min(max_results, 10_000)

        payload = self.post(
            "/api/v1/graph/traverse",
            {
                "start": start_node,
                "direction": direction,
                "max_depth": depth,
                "edge_type": edge_type,
            },
        )
        results = payload.get("results") or []
        truncated = False
        if max_results is not None and len(results) > max_results:
            results = results[:max_results]
            truncated = True
        return {
            "ok": True,
            "tool": "graph_traverse",
            "start": start_node,
            "depth": depth,
            "direction": direction,
            "count": len(results),
            "truncated": truncated,
            "results": results,
        }

    # -- learning ----------------------------------------------------------

    def _thompson(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return self.post("/api/v1/thompson", body)

    def get_model_status(self) -> dict[str, Any]:
        """Thompson arms with Beta posteriors derived from observed outcomes."""
        payload = self._thompson({"action": "get_state"})
        state = payload.get("state") or {}
        models = state.get("models") or {}
        try:
            prior = float(os.environ.get("POC_CLAUDE_ENSEMBLE_MCP_PRIOR_STRENGTH", "1.0"))
        except ValueError:
            prior = 1.0
        if prior <= 0:
            prior = 1.0

        arms: dict[str, Any] = {}
        for name, stats in sorted(models.items()):
            if not isinstance(stats, dict):
                continue
            successes = int(stats.get("successes") or 0)
            failures = int(stats.get("failures") or 0)
            calls = int(stats.get("calls") or 0)
            alpha = successes + prior
            beta = failures + prior
            arms[name] = {
                "successes": successes,
                "failures": failures,
                "calls": calls,
                "total_cost": float(stats.get("total_cost") or 0.0),
                "avg_cost": round(float(stats.get("total_cost") or 0.0) / calls, 8) if calls else 0.0,
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "posterior_mean": round(alpha / (alpha + beta), 6) if alpha + beta else None,
                "last_updated": stats.get("last_updated"),
            }
        return {
            "ok": True,
            "tool": "get_model_status",
            "state_last_updated": state.get("last_updated"),
            "prior_strength": prior,
            "arm_count": len(arms),
            "arms": arms,
        }

    def _crew_module(self):
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import importlib

        return importlib.import_module("tools.crew_select")

    def draw_crew(
        self,
        task_type: str,
        workers: int = 3,
        arbiter: str | None = None,
    ) -> dict[str, Any]:
        """Draw a distinct-arm crew with the existing ``crew_select`` workflow.

        The injected ``ThompsonClient`` points at the gateway's
        ``/api/v1/thompson`` route, and its requests pass through this client's
        rate limiter.  Like the CLI, this writes an exposure row to the crew
        ledger (``~/.FlossWare/ledger/crews`` by default).  That is a local
        side effect, not a gateway mutation.
        """
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a non-empty string")
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 10:
            raise ValueError("workers must be an integer between 1 and 10")
        if arbiter is not None and (not isinstance(arbiter, str) or not arbiter.strip()):
            raise ValueError("arbiter must be a non-empty string when provided")

        crew = self._crew_module()
        thompson = crew.ThompsonClient(
            timeout=self.timeout, service_url=self.base_url + "/api/v1"
        )
        original_send = thompson._send_request

        def rate_limited_send(request: Mapping[str, Any]) -> dict[str, Any]:
            self.rate_limiter.acquire()
            return original_send(request)

        thompson._send_request = rate_limited_send  # type: ignore[method-assign]
        selection = crew.select_crew(
            task_type=task_type,
            count=workers,
            arbiter=arbiter or crew.DEFAULT_ARBITER,
            client=thompson,
        )
        return {
            "ok": True,
            "tool": "draw_crew",
            "selection": selection,
        }

    def get_learning_status(self) -> dict[str, Any]:
        """Learning report plus recent outcomes, with partial-failure reporting."""
        report = self.post("/api/v1/learning", {"op": "get_report"})
        errors: list[str] = []
        recent: list[Any] = []
        try:
            outcomes = self.post(
                "/api/v1/learning", {"op": "get_recent_outcomes", "days": 7}
            )
            recent = outcomes.get("outcomes") or []
        except GatewayError as exc:
            errors.append(f"recent_outcomes: {exc.code}: {exc.message}")
        return {
            "ok": True,
            "tool": "get_learning_status",
            "report": report.get("report"),
            "total_outcomes": report.get("total_outcomes"),
            "recent_outcomes": recent,
            "errors": errors,
        }

    # -- health ------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Gateway health, readiness and per-service status in one call."""
        gateway = self.get("/api/v1/health")
        errors: list[str] = []
        ready: dict[str, Any] | None = None
        services: list[dict[str, Any]] = []
        counts: dict[str, Any] = {}
        try:
            ready = self.get("/api/v1/ready")
        except GatewayError as exc:
            errors.append(f"ready: {exc.code}: {exc.message}")
        try:
            overview = self.get("/api/v1/services", allow_not_ok=True)
            services = overview.get("services") or []
            counts = overview.get("counts") or {}
        except GatewayError as exc:
            errors.append(f"services: {exc.code}: {exc.message}")
        degraded = [
            {"id": service.get("id"), "status": service.get("status")}
            for service in services
            if service.get("status") != "healthy"
        ]
        return {
            "ok": True,
            "tool": "health_check",
            "gateway": gateway,
            "ready": ready,
            "counts": counts,
            "degraded": degraded,
            "services": services,
            "errors": errors,
        }
