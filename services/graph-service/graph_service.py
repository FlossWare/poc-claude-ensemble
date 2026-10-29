#!/usr/bin/env python3
"""Claude Ensemble graph service with a real HTTP boundary.

The graph is held in a NetworkX MultiDiGraph. It provides durable, thread-safe
graph mutations and read operations over HTTP. The service is local by default;
federation belongs to a later recovery issue.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import networkx as nx


MAX_REQUEST_BYTES = 1_048_576
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


class GraphStore:
    """Thread-safe durable graph store backed by a NetworkX MultiDiGraph.

    Nodes are keyed by their stable ID. Edges are keyed by their stable ID
    as well, so repeated relationships between the same two nodes stay
    separate and each can be looked up or traversed individually.
    The graph is persisted as node-link JSON through an atomic replace.
    """

    FORMAT_VERSION = 2

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.lock = threading.RLock()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._edge_index: dict[str, tuple[str, str]] = {}
        self._load()

    def _load(self) -> None:
        with self.lock:
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid graph store: {self.path}")
            if data.get("version") != self.FORMAT_VERSION or "graph" not in data:
                raise ValueError(f"Unsupported graph store format: {self.path}")
            graph = nx.node_link_graph(data["graph"], directed=True, multigraph=True, edges="edges")
            self.graph = graph
            self._edge_index = {
                attrs["id"]: (u, v) for u, v, attrs in graph.edges(data=True)
            }

    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": self.FORMAT_VERSION,
            "graph": nx.node_link_data(self.graph, edges="edges"),
        }

    def _persist(self, graph: nx.MultiDiGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.FORMAT_VERSION,
            "graph": nx.node_link_data(graph, edges="edges"),
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".graph-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def add_node(self, node_type: str, properties: dict[str, Any], node_id: str | None = None) -> dict[str, Any]:
        if not isinstance(node_type, str) or not node_type:
            raise ValueError("node_type must be a non-empty string")
        if not isinstance(properties, dict):
            raise ValueError("properties must be an object")
        with self.lock:
            identity = {"type": node_type, "properties": properties}
            node_id = node_id or _stable_id("node", identity)
            node = {"id": node_id, "type": node_type, "properties": properties}
            if self.graph.has_node(node_id):
                existing = self._node_record(node_id)
                if existing != node:
                    raise ValueError(f"node id already exists with different content: {node_id}")
                return dict(node)
            proposed = self.graph.copy()
            proposed.add_node(node_id, type=node_type, properties=properties)
            self._persist(proposed)
            self.graph = proposed
            return dict(node)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        properties: dict[str, Any],
        edge_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        if not isinstance(edge_type, str) or not edge_type:
            raise ValueError("edge_type must be a non-empty string")
        if not isinstance(properties, dict):
            raise ValueError("properties must be an object")
        with self.lock:
            if not self.graph.has_node(source) or not self.graph.has_node(target):
                raise KeyError("source and target nodes must exist")
            identity = {
                "source": source,
                "target": target,
                "type": edge_type,
                "properties": properties,
            }
            edge_id = edge_id or _stable_id("edge", identity)
            edge = {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "properties": properties,
            }
            if edge_id in self._edge_index:
                existing = self._edge_record(edge_id)
                if existing != edge:
                    raise ValueError(f"edge id already exists with different content: {edge_id}")
                return dict(edge)
            proposed = self.graph.copy()
            proposed.add_edge(source, target, key=edge_id, id=edge_id, type=edge_type, properties=properties)
            self._persist(proposed)
            self.graph = proposed
            self._edge_index[edge_id] = (source, target)
            return dict(edge)

    def _node_record(self, node_id: str) -> dict[str, Any]:
        attrs = self.graph.nodes[node_id]
        return {"id": node_id, "type": attrs["type"], "properties": attrs["properties"]}

    def _edge_record(self, edge_id: str) -> dict[str, Any]:
        source, target = self._edge_index[edge_id]
        attrs = self.graph.edges[source, target, edge_id]
        return {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": attrs["type"],
            "properties": attrs["properties"],
        }

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self.lock:
            if not self.graph.has_node(node_id):
                return None
            return self._node_record(node_id)

    def query(self, node_type: str | None, properties: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(properties, dict):
            raise ValueError("properties must be an object")
        with self.lock:
            result = []
            for node_id, attrs in self.graph.nodes(data=True):
                if node_type is not None and attrs["type"] != node_type:
                    continue
                if all(attrs["properties"].get(key) == value for key, value in properties.items()):
                    result.append(self._node_record(node_id))
            return sorted(result, key=lambda node: node["id"])

    def query_edges(self, from_type: str | None = None, to_type: str | None = None, relationship: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            result = []
            for source, target, edge_id in self.graph.edges(keys=True):
                if from_type is not None and self.graph.nodes[source]["type"] != from_type:
                    continue
                if to_type is not None and self.graph.nodes[target]["type"] != to_type:
                    continue
                attrs = self.graph.edges[source, target, edge_id]
                if relationship is not None and attrs["type"] != relationship:
                    continue
                result.append({"id": edge_id, "from_id": source, "to_id": target,
                               "relationship": attrs["type"], "properties": dict(attrs["properties"])})
            return sorted(result, key=lambda item: item["id"])

    def traverse(
        self,
        start: str,
        direction: str = "out",
        max_depth: int = 1,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in, or both")
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 100:
            raise ValueError("max_depth must be an integer from 1 to 100")
        with self.lock:
            if not self.graph.has_node(start):
                raise KeyError(f"node not found: {start}")
            visited = {start}
            frontier = {start}
            result = []
            for depth in range(1, max_depth + 1):
                next_frontier: set[str] = set()
                for source, target, edge_id, attrs in self.graph.edges(keys=True, data=True):
                    if edge_type is not None and attrs["type"] != edge_type:
                        continue
                    neighbors: list[str] = []
                    if direction in {"out", "both"} and source in frontier:
                        neighbors.append(target)
                    if direction in {"in", "both"} and target in frontier:
                        neighbors.append(source)
                    for neighbor in neighbors:
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        via = self._edge_record(edge_id)
                        result.append({"depth": depth, "node": self._node_record(neighbor), "via": via})
                frontier = next_frontier
                if not frontier:
                    break
            return result


class GraphRequestHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleGraph/1"

    @property
    def store(self) -> GraphStore:
        return self.server.graph_store  # type: ignore[attr-defined]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = (_canonical(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body must be between 1 byte and 1 MiB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/health":
                self._send(200, {"ok": True, "service": "graph"})
                return
            if path.startswith("/graph/node/"):
                node_id = unquote(path[len("/graph/node/"):])
                node = self.store.get_node(node_id)
                if node is None:
                    self._send(404, {"ok": False, "error": "node not found"})
                else:
                    self._send(200, {"ok": True, "node": node})
                return
            self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/graph/add-node":
                node = self.store.add_node(body.get("type"), body.get("properties", {}), body.get("id"))
                self._send(200, {"ok": True, "node": node})
                return
            if path == "/graph/add-edge":
                edge = self.store.add_edge(
                    body.get("source"),
                    body.get("target"),
                    body.get("type"),
                    body.get("properties", {}),
                    body.get("id"),
                )
                self._send(200, {"ok": True, "edge": edge})
                return
            if path == "/graph/query":
                if body.get("type") == "edges":
                    edges = self.store.query_edges(body.get("from_type"), body.get("to_type"), body.get("relationship"))
                    self._send(200, {"ok": True, "edges": edges})
                else:
                    nodes = self.store.query(body.get("type"), body.get("properties", {}))
                    self._send(200, {"ok": True, "nodes": nodes})
                return
            if path == "/graph/traverse":
                result = self.store.traverse(
                    body.get("start"),
                    body.get("direction", "out"),
                    body.get("max_depth", 1),
                    body.get("edge_type"),
                )
                self._send(200, {"ok": True, "results": result})
                return
            self._send(404, {"ok": False, "error": "not found"})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def _validate_loopback_host(host: str) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Graph service only accepts loopback binds; remote access belongs to federation")


def _configured_host(environ: dict[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    host = values.get("ENSEMBLE_GRAPH_HOST", DEFAULT_HOST)
    _validate_loopback_host(host)
    return host


def create_server(host: str, port: int, store_path: str | Path) -> ThreadingHTTPServer:
    _validate_loopback_host(host)
    server = ThreadingHTTPServer((host, port), GraphRequestHandler)
    server.graph_store = GraphStore(store_path)  # type: ignore[attr-defined]
    return server


def main() -> None:
    try:
        host = _configured_host()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    port = int(os.environ.get("ENSEMBLE_GRAPH_PORT", str(DEFAULT_PORT)))
    store_path = os.environ.get(
        "ENSEMBLE_GRAPH_STORE",
        str(Path.home() / ".local" / "share" / "claude-ensemble" / "graph.json"),
    )
    server = create_server(host, port, store_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
