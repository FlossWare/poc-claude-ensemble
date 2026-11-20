"""Shared offline test helpers and fakes."""
from __future__ import annotations

import io
import json
import shutil
import socket
import tarfile
import threading
from urllib.parse import unquote, urlparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ingestion.datasets.config import Config
from ingestion.datasets.netguard import NetResponse, RawResponse, Transport

FIXTURES = Path(__file__).with_name("fixtures")

ENTRY_FIXTURES = (
    "full.yml",
    "sparse.yml",
    "license-list.yml",
    "typo-name.yml",
    "hhttps.yml",
    "ftp.yml",
    "San Diego, CA.yml",
    "OpenNEURO",
    "LocationData.wml",
    "duplicate-keys.yml",
    "restricted.yml",
    "proprietary.yml",
)

GPL_LICENSE = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
MIT_LICENSE = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy.\n"


def build_readme(names, statuses: dict[str, str] | None = None, category: str = "Test") -> str:
    statuses = statuses or {}
    lines = [
        "Mini Awesome Public Datasets",
        "============================",
        "",
        ".. |OK_ICON| image:: https://example.org/ok.png",
        ".. |FIXME_ICON| image:: https://example.org/fixme.png",
        "",
    ]
    for name in names:
        status = statuses.get(name, "OK")
        icon = "OK_ICON" if status == "OK" else "FIXME_ICON"
        title = name.rsplit(".", 1)[0]
        lines.append(
            f"* |{icon}| `{title} <https://example.org/>`_ "
            f"[`Meta <https://github.com/awesomedata/apd-core/tree/master/core//{category}/{name}>`_]"
        )
        lines.append("")
    return "\n".join(lines)


def make_catalog(
    tmp_path: Path,
    *,
    names=ENTRY_FIXTURES,
    readme: str | None = None,
    statuses: dict[str, str] | None = None,
    include_readme_in_root: bool = True,
) -> Path:
    root = tmp_path / "apd-core"
    core = root / "core" / "Test"
    core.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(FIXTURES / name, core / name)
    (root / "LICENSE").write_text(GPL_LICENSE, encoding="utf-8")
    if include_readme_in_root:
        # The real apd-core README has no Meta links; discovery must prefer the
        # generated awesome-public-datasets README instead.
        (root / "README.md").write_text("# apd-core\n", encoding="utf-8")
    sibling = tmp_path / "awesome-public-datasets"
    sibling.mkdir(exist_ok=True)
    (sibling / "LICENSE").write_text(MIT_LICENSE, encoding="utf-8")
    (sibling / "README.rst").write_text(
        readme if readme is not None else build_readme(list(names), statuses),
        encoding="utf-8",
    )
    return root


def make_config(tmp_path: Path, root: Path | None = None, **overrides) -> Config:
    if root is None:
        root = make_catalog(tmp_path)
    config = Config.load(
        catalog_dir=root,
        data_root=tmp_path / "data",
        ledger_enabled=False,
    )
    changes = {"ledger_enabled": False, **overrides}
    return config.replace(**changes)


def tar_bytes(entries, *, mode: str = "w:gz") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as archive:
        for info, data in entries:
            archive.addfile(info, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


def regular(name: str, data: bytes, *, mtime: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    info.size = len(data)
    info.mtime = mtime
    return info


def directory(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = 0
    return info


def symlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    info.mtime = 0
    return info


def hardlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    info.mtime = 0
    return info


def device(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.CHRTYPE
    info.devmajor = 1
    info.devminor = 3
    info.mtime = 0
    return info


def stub_resolver(mapping):
    def resolve(host, port, *args, **kwargs):
        value = mapping(host) if callable(mapping) else mapping[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (value, 0))]

    return resolve


def _no_connector(*args, **kwargs):
    raise AssertionError("tests must override _request_once instead of connecting")


class StubTransport(Transport):
    """Transport whose per-hop request function is canned."""

    def __init__(self, config: Config, resolver, responses: dict[str, object]):
        super().__init__(config, resolver=resolver, connector=_no_connector)
        self.responses = responses
        self.calls: list[str] = []

    def _request_once(self, validated, *, method, headers, max_bytes, store_body, deadline):
        self.calls.append(validated.url)
        value = self.responses.get(validated.url)
        if value is None:
            raise AssertionError(f"unexpected request to {validated.url}")
        if isinstance(value, Exception):
            raise value
        if isinstance(value, RawResponse):
            return value
        raise AssertionError(f"bad canned response for {validated.url}")


class FakeProbeTransport:
    """Probe-level fake returning NetResponse objects."""

    def __init__(self, responses: dict[object, object] | None = None, default=None):
        self.responses = responses or {}
        self.default = default
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def request(self, url, *, method="GET", headers=None, max_bytes=0, store_body=False):
        with self._lock:
            self.calls.append(
                {
                    "url": url,
                    "method": method,
                    "headers": dict(headers or {}),
                    "max_bytes": max_bytes,
                    "store_body": store_body,
                }
            )
        value = self.responses.get((url, method))
        if value is None:
            value = self.responses.get(url, self.default)
        if value is None:
            raise AssertionError(f"unexpected probe request {method} {url}")
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(url, method, headers)
        if isinstance(value, dict):
            body = value.get("body")
            if isinstance(body, str):
                body = body.encode("utf-8")
            return NetResponse(
                status=int(value.get("status", 200)),
                headers=dict(value.get("headers") or {}),
                final_url=str(value.get("final_url") or url),
                redirect_chain=list(value.get("redirect_chain") or []),
                body=body if isinstance(body, (bytes, bytearray)) else None,
            )
        return value


class FakeStreamResponse:
    """A canned live response for :class:`FakeStreamTransport`."""

    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        final_url: str = "",
        redirect_chain: list[str] | None = None,
    ) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self.final_url = final_url
        self.redirect_chain = list(redirect_chain or [])
        self.reads: list[int] = []

    @property
    def has_body(self) -> bool:
        return self.status in {200, 206}

    def read(self, size: int = 64 * 1024) -> bytes:
        self.reads.append(size)
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class FakeStreamTransport:
    """Transport double for acquisition: no sockets, no DNS, canned bodies.

    A route value may be a :class:`FakeStreamResponse`, an exception instance,
    or a callable that receives the request headers and returns either.
    """

    def __init__(self, routes: dict[str, object] | None = None) -> None:
        self.routes = dict(routes or {})
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def add(self, url: str, value: object) -> None:
        self.routes[url] = value

    @contextmanager
    def open_stream(self, url, *, method="GET", headers=None):
        with self._lock:
            self.calls.append(
                {"url": url, "method": method, "headers": dict(headers or {})}
            )
        value = self.routes.get(url)
        if value is None:
            raise AssertionError(f"unexpected stream request {url}")
        if callable(value):
            value = value(dict(headers or {}))
        if isinstance(value, BaseException):
            raise value
        if not isinstance(value, FakeStreamResponse):
            raise AssertionError(f"bad canned stream for {url}")
        yield value

    def urls(self) -> list[str]:
        return [str(call["url"]) for call in self.calls]


class FakeDiskUsage:
    """A ``shutil.disk_usage`` stand-in with a controllable free byte count."""

    def __init__(self, free: int) -> None:
        self.free = free
        self.total = free * 4
        self.used = self.total - free


class MemoryHandler(BaseHTTPRequestHandler):
    seen: set[str] = set()
    posts: list[dict[str, object]] = []
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        event_id = str(payload.get("event_id") or "")
        with self.lock:
            duplicate = event_id in self.seen
            self.seen.add(event_id)
            self.posts.append(payload)
        response = json.dumps({"ok": True, "duplicate": duplicate}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):  # noqa: D401 - silence test server
        return


@contextmanager
def memory_server():
    MemoryHandler.seen = set()
    MemoryHandler.posts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MemoryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", MemoryHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class FakeIntegrationBackend:
    """In-memory stand-in for Knowledge, Index and the retrieval services.

    Records requests exactly like the real services understand them, so a test
    can assert the integration sequence without a socket.
    """

    WINDOW = 1800
    OVERLAP = 200

    def __init__(self) -> None:
        self.index: dict[str, dict[str, object]] = {}
        self.knowledge: dict[str, dict[str, object]] = {}
        self.chunks: dict[str, dict[str, object]] = {}
        self.vectors: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_paths: set[str] = set()

    # -- service doubles ---------------------------------------------------
    def state(self, base, source_id, timeout=10):
        current = self.index.get(source_id)
        if current is None:
            return None
        return dict(current, source_id=source_id)

    def post(self, base, path, payload, timeout=60):
        self.calls.append((path, payload))
        for prefix in self.fail_paths:
            if path.startswith(prefix):
                from ingestion.datasets.indexing import IndexingError

                raise IndexingError("a required ingestion capability is unavailable", 503, "dependency_unavailable")
        if path == "/records":
            return self._records(payload)
        if path.endswith("/tombstone"):
            from urllib.parse import unquote

            source_id = unquote(path[len("/records/") : -len("/tombstone")])
            current = self.index.get(source_id)
            sequence = int(payload.get("revision_sequence") or 1)
            if current is None:
                self.index[source_id] = {
                    "high_sequence": sequence,
                    "revision_id": "tombstone",
                    "tombstoned": True,
                }
                return {"ok": True, "decision": "tombstoned"}
            if sequence >= int(current["high_sequence"]):
                current["tombstoned"] = True
                current["high_sequence"] = max(sequence, int(current["high_sequence"]))
                return {"ok": True, "decision": "tombstoned"}
            return {"ok": True, "decision": "superseded"}
        if path == "/chunk":
            return {"ok": True, "chunks": self._chunk(payload)}
        if path == "/embeddings":
            texts = payload.get("texts") or []
            return {"ok": True, "embeddings": [[float(len(text) % 11), 0.5, 0.25, 0.125] for text in texts]}
        if path == "/knowledge/documents":
            for row in payload.get("chunks") or []:
                self.chunks.setdefault(row["chunk_id"], row)
            return {"ok": True, "inserted": len(payload.get("chunks") or [])}
        if path == "/collections/knowledge_chunks/upsert":
            for vector_id, meta in zip(payload.get("ids") or [], payload.get("metadatas") or []):
                self.vectors[vector_id] = meta
            return {"ok": True, "upserted": len(payload.get("ids") or [])}
        if path == "/collections/knowledge_chunks/delete-superseded":
            keep = payload.get("keep_sha256")
            url = payload.get("url")
            stale = [
                vector_id
                for vector_id, meta in self.vectors.items()
                if meta.get("url") == url and meta.get("content_sha256") != keep
            ]
            for vector_id in stale:
                del self.vectors[vector_id]
            return {"ok": True, "deleted": len(stale)}
        return {"ok": True}

    # -- behaviour ---------------------------------------------------------
    def _records(self, payload):
        source_id = payload["source_id"]
        sequence = int(payload["revision_sequence"])
        revision_id = payload["revision_id"]
        current = self.index.get(source_id)
        if current is not None and current.get("tombstoned"):
            return {"ok": True, "decision": "tombstoned", "stored": False}
        if current is None:
            self.index[source_id] = {
                "high_sequence": sequence,
                "revision_id": revision_id,
                "tombstoned": False,
            }
            self.knowledge[source_id] = dict(payload)
            return {"ok": True, "decision": "accepted", "stored": True}
        if sequence < int(current["high_sequence"]):
            return {"ok": True, "decision": "superseded", "stored": False}
        if sequence == int(current["high_sequence"]):
            if current["revision_id"] == revision_id:
                return {"ok": True, "decision": "duplicate", "stored": False}
            return {"ok": True, "decision": "conflict", "stored": False}
        current["high_sequence"] = sequence
        current["revision_id"] = revision_id
        self.knowledge[source_id] = dict(payload)
        return {"ok": True, "decision": "accepted", "stored": True}

    def _chunk(self, payload):
        text = str(payload.get("text") or "")
        source_id = str(payload.get("source_id") or "")
        profile = str(payload.get("profile") or "window-v1")
        spans = []
        start = 0
        while start < len(text):
            end = min(start + self.WINDOW, len(text))
            spans.append((start, end))
            if end >= len(text):
                break
            start += self.WINDOW - self.OVERLAP
        chunks = []
        for index, (start, end) in enumerate(spans):
            key = f"{profile}|1|{source_id}|{start}|{end}"
            chunks.append(
                {
                    "chunk_id": __import__("hashlib").sha256(key.encode()).hexdigest(),
                    "index": index,
                    "start": start,
                    "end": end,
                    "header": "",
                    "text": text[start:end],
                }
            )
        return chunks

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


# ---------------------------------------------------------------------------
# PR 5: a real HTTP fake for the graph and retrieval services
# ---------------------------------------------------------------------------


class FakeServicesBackend:
    """In-memory state shared by :class:`FakeServicesHandler` routes.

    The graph half reproduces the real service's immutability contract: a node
    or edge id may be asserted repeatedly, but a *different* payload under the
    same id is a 400.  That is what makes the projection idempotency tests
    meaningful rather than tautological.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[str, dict[str, object]] = {}
        self.index: dict[str, dict[str, object]] = {}
        self.knowledge: dict[str, dict[str, object]] = {}
        self.chunks: dict[str, dict[str, object]] = {}
        self.vectors: dict[str, dict[str, object]] = {}
        self.requests: list[tuple[str, str]] = []
        #: (source_id, decision) for every Knowledge /records response.
        self.decisions: list[tuple[str, str]] = []
        #: Writes that asserted an already-present record with identical content.
        self.repeated_node_writes = 0
        self.repeated_edge_writes = 0
        self.lock = threading.RLock()
        #: Fail these paths with the mapped HTTP status.
        self.fail_paths: dict[str, int] = {}
        #: Fail these paths N times, then succeed (transient-failure tests).
        self.fail_once: dict[str, int] = {}
        #: Block (after writing a marker) when this path is hit this many times.
        self.block_path: str | None = None
        self.block_after = 0
        self.block_marker: Path | None = None
        self.release = threading.Event()
        self.block_hits = 0

    # -- recording ---------------------------------------------------------
    def note(self, method: str, path: str) -> None:
        with self.lock:
            self.requests.append((method, path))

    def count(self, prefix: str) -> int:
        with self.lock:
            return sum(1 for _method, path in self.requests if path.startswith(prefix))

    def reset(self) -> None:
        with self.lock:
            self.requests = []

    # -- graph -------------------------------------------------------------
    def add_node(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        node_id = payload.get("id")
        node = {
            "id": node_id,
            "type": payload.get("type"),
            "properties": payload.get("properties") or {},
        }
        if not isinstance(node_id, str) or not node_id:
            return 400, {"ok": False, "error": "node id is required"}
        with self.lock:
            existing = self.nodes.get(node_id)
            if existing is not None:
                if existing != node:
                    return 400, {
                        "ok": False,
                        "error": f"node id already exists with different content: {node_id}",
                    }
                self.repeated_node_writes += 1
                return 200, {"ok": True, "node": existing}
            self.nodes[node_id] = node
            return 200, {"ok": True, "node": node}

    def add_edge(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        edge_id = payload.get("id")
        edge = {
            "id": edge_id,
            "source": payload.get("source"),
            "target": payload.get("target"),
            "type": payload.get("type"),
            "properties": payload.get("properties") or {},
        }
        if not isinstance(edge_id, str) or not edge_id:
            return 400, {"ok": False, "error": "edge id is required"}
        with self.lock:
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes:
                return 400, {"ok": False, "error": "source and target nodes must exist"}
            existing = self.edges.get(edge_id)
            if existing is not None:
                if existing != edge:
                    return 400, {
                        "ok": False,
                        "error": f"edge id already exists with different content: {edge_id}",
                    }
                self.repeated_edge_writes += 1
                return 200, {"ok": True, "edge": existing}
            self.edges[edge_id] = edge
            return 200, {"ok": True, "edge": edge}

    def query(self, payload: dict[str, object]) -> dict[str, object]:
        properties = payload.get("properties") or {}
        if payload.get("type") == "edges":
            from_type = payload.get("from_type")
            result = []
            for edge in self.edges.values():
                source = self.nodes.get(str(edge["source"]), {})
                if from_type is not None and source.get("type") != from_type:
                    continue
                result.append(
                    {
                        "id": edge["id"],
                        "from_id": edge["source"],
                        "to_id": edge["target"],
                        "relationship": edge["type"],
                        "properties": edge["properties"],
                    }
                )
            return {"ok": True, "edges": result}
        node_type = payload.get("type")
        nodes = [
            node
            for node in self.nodes.values()
            if (node_type is None or node.get("type") == node_type)
            and all(
                (node.get("properties") or {}).get(key) == value
                for key, value in properties.items()
            )
        ]
        return {"ok": True, "nodes": sorted(nodes, key=lambda item: str(item["id"]))}

    def traverse(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        start = payload.get("start")
        if start not in self.nodes:
            return 400, {"ok": False, "error": f"node not found: {start}"}
        direction = payload.get("direction", "out")
        max_depth = int(payload.get("max_depth", 1))
        edge_type = payload.get("edge_type")
        visited = {start}
        frontier = {start}
        results = []
        for depth in range(1, max_depth + 1):
            next_frontier: set[str] = set()
            for edge in list(self.edges.values()):
                if edge_type is not None and edge["type"] != edge_type:
                    continue
                neighbours = []
                if direction in {"out", "both"} and edge["source"] in frontier:
                    neighbours.append(edge["target"])
                if direction in {"in", "both"} and edge["target"] in frontier:
                    neighbours.append(edge["source"])
                for neighbour in neighbours:
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    next_frontier.add(str(neighbour))
                    results.append(
                        {
                            "depth": depth,
                            "node": self.nodes[str(neighbour)],
                            "via": edge,
                        }
                    )
            frontier = next_frontier
            if not frontier:
                break
        return 200, {"ok": True, "results": results}

    def remove_nodes(self, payload: dict[str, object]) -> dict[str, object]:
        ids = payload.get("ids") or []
        dry_run = bool(payload.get("dry_run"))
        present = [node_id for node_id in ids if node_id in self.nodes]
        missing = [node_id for node_id in ids if node_id not in self.nodes]
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "would_remove_nodes": len(present),
                "would_remove_edges": sum(
                    1
                    for edge in self.edges.values()
                    if edge["source"] in present or edge["target"] in present
                ),
                "missing": missing,
                "sample_ids": present[:10],
            }
        with self.lock:
            for node_id in present:
                self.nodes.pop(node_id, None)
            for edge_id, edge in list(self.edges.items()):
                if edge["source"] in present or edge["target"] in present:
                    self.edges.pop(edge_id, None)
        return {
            "ok": True,
            "dry_run": False,
            "removed_nodes": len(present),
            "removed_edges": 0,
            "missing": missing,
        }

    # -- knowledge / index -------------------------------------------------
    def records(self, payload: dict[str, object]) -> dict[str, object]:
        result = self._records(payload)
        with self.lock:
            self.decisions.append((str(payload["source_id"]), str(result.get("decision"))))
        return result

    def _records(self, payload: dict[str, object]) -> dict[str, object]:
        source_id = str(payload["source_id"])
        sequence = int(payload["revision_sequence"])
        revision_id = str(payload["revision_id"])
        with self.lock:
            current = self.index.get(source_id)
            if current is not None and current.get("tombstoned"):
                return {"ok": True, "decision": "tombstoned", "stored": False}
            if current is None:
                self.index[source_id] = {
                    "high_sequence": sequence,
                    "revision_id": revision_id,
                    "tombstoned": False,
                }
                self.knowledge[source_id] = dict(payload)
                return {"ok": True, "decision": "accepted", "stored": True}
            if current.get("revision_id") == revision_id and int(current["high_sequence"]) == sequence:
                return {"ok": True, "decision": "duplicate", "stored": False}
            current["high_sequence"] = sequence
            current["revision_id"] = revision_id
            self.knowledge[source_id] = dict(payload)
            return {"ok": True, "decision": "accepted", "stored": True}

    def chunk(self, payload: dict[str, object]) -> dict[str, object]:
        text = str(payload.get("text") or "")
        source_id = str(payload.get("source_id") or "")
        size = 1800
        chunks = []
        for index, start in enumerate(range(0, max(1, len(text)), size)):
            end = min(start + size, len(text))
            key = f"window-v1|1|{source_id}|{start}|{end}"
            chunks.append(
                {
                    "chunk_id": __import__("hashlib").sha256(key.encode()).hexdigest(),
                    "index": index,
                    "start": start,
                    "end": end,
                    "header": "",
                    "text": text[start:end],
                }
            )
        return {"ok": True, "chunks": chunks}

    def lexical_search(self, payload: dict[str, object]) -> dict[str, object]:
        query = str(payload.get("query") or "").lower()
        top_k = int(payload.get("top_k") or 10)
        tokens = [token for token in query.split() if len(token) > 3]
        rows = list(self.chunks.values())
        if tokens:
            rows = [
                row
                for row in rows
                if any(token in str(row.get("text", "")).lower() for token in tokens)
            ]
        return {"ok": True, "results": rows[:top_k]}

    def vector_query(self, payload: dict[str, object]) -> dict[str, object]:
        top_k = int(payload.get("top_k") or 10)
        rows = [
            {"id": vector_id, "document": meta.get("text"), "metadata": meta, "score": 0.9}
            for vector_id, meta in self.vectors.items()
        ]
        return {"ok": True, "results": rows[:top_k]}


class FakeServicesHandler(BaseHTTPRequestHandler):
    backend: FakeServicesBackend  # set by the context manager

    def log_message(self, *args):  # noqa: D401 - silence test server
        return

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            # The client was killed mid-request (the interruption test).
            return

    def _payload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def _maybe_block(self, path: str) -> None:
        backend = self.backend
        if backend.block_path is None or not path.startswith(backend.block_path):
            return
        with backend.lock:
            backend.block_hits += 1
            hit = backend.block_hits
        if hit < backend.block_after:
            return
        if backend.block_marker is not None:
            backend.block_marker.parent.mkdir(parents=True, exist_ok=True)
            backend.block_marker.write_text(path, encoding="utf-8")
        backend.release.wait(timeout=30)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        backend = self.backend
        backend.note("GET", path)
        if path == "/health":
            self._send(200, {"ok": True, "service": "fake", "backend": "fake"})
            return
        if path == "/collections":
            self._send(
                200,
                {
                    "ok": True,
                    "collections": [
                        {"name": "knowledge_chunks", "count": len(backend.vectors)}
                    ],
                },
            )
            return
        if path.startswith("/sources/"):
            source_id = unquote(path[len("/sources/") :])
            state = backend.index.get(source_id)
            if state is None:
                self._send(404, {"ok": False, "error": "unknown source"})
                return
            self._send(200, {"ok": True, "source": dict(state, source_id=source_id)})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        backend = self.backend
        backend.note("POST", path)
        payload = self._payload()
        self._maybe_block(path)
        status = backend.fail_paths.get(path)
        if status is None:
            remaining = backend.fail_once.get(path, 0)
            if remaining > 0:
                backend.fail_once[path] = remaining - 1
                status = 503
        if status is not None:
            self._send(status, {"ok": False, "error": "forced failure"})
            return
        try:
            if path == "/graph/add-node":
                code, body = backend.add_node(payload)
                self._send(code, body)
                return
            if path == "/graph/add-edge":
                code, body = backend.add_edge(payload)
                self._send(code, body)
                return
            if path == "/graph/query":
                self._send(200, backend.query(payload))
                return
            if path == "/graph/traverse":
                code, body = backend.traverse(payload)
                self._send(code, body)
                return
            if path == "/graph/remove-nodes":
                self._send(200, backend.remove_nodes(payload))
                return
            if path == "/records":
                self._send(200, backend.records(payload))
                return
            if path.endswith("/tombstone"):
                self._send(200, {"ok": True, "decision": "tombstoned"})
                return
            if path == "/chunk":
                self._send(200, backend.chunk(payload))
                return
            if path == "/embeddings":
                texts = payload.get("texts") or []
                self._send(
                    200,
                    {
                        "ok": True,
                        "embeddings": [
                            [float(len(text) % 11), 0.5, 0.25, 0.125] for text in texts
                        ],
                    },
                )
                return
            if path == "/knowledge/documents":
                for row in payload.get("chunks") or []:
                    backend.chunks.setdefault(row["chunk_id"], row)
                self._send(200, {"ok": True, "inserted": len(payload.get("chunks") or [])})
                return
            if path == "/collections/knowledge_chunks/upsert":
                for vector_id, meta in zip(
                    payload.get("ids") or [], payload.get("metadatas") or []
                ):
                    backend.vectors[vector_id] = meta
                self._send(200, {"ok": True, "upserted": len(payload.get("ids") or [])})
                return
            if path == "/collections/knowledge_chunks/delete-superseded":
                keep = payload.get("keep_sha256")
                url = payload.get("url")
                stale = [
                    vector_id
                    for vector_id, meta in backend.vectors.items()
                    if meta.get("url") == url and meta.get("content_sha256") != keep
                ]
                for vector_id in stale:
                    backend.vectors.pop(vector_id, None)
                self._send(200, {"ok": True, "deleted": len(stale)})
                return
            if path == "/collections/knowledge_chunks/query":
                self._send(200, backend.vector_query(payload))
                return
            if path == "/knowledge/search":
                self._send(200, backend.lexical_search(payload))
                return
        except Exception as exc:  # noqa: BLE001 - a test server must answer
            self._send(500, {"ok": False, "error": type(exc).__name__})
            return
        self._send(404, {"ok": False, "error": "not found"})


@contextmanager
def fake_services_server(backend: FakeServicesBackend | None = None):
    """Serve :class:`FakeServicesBackend` over real HTTP on a loopback port."""
    backend = backend or FakeServicesBackend()
    handler = type("BoundFakeServicesHandler", (FakeServicesHandler,), {"backend": backend})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", backend
    finally:
        backend.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def service_env(base: str) -> dict[str, str]:
    """Environment variables that point the CLI at a fake service base URL."""
    return {
        "POC_CLAUDE_ENSEMBLE_GRAPH_URL": base,
        "POC_CLAUDE_ENSEMBLE_KNOWLEDGE_URL": base,
        "POC_CLAUDE_ENSEMBLE_CHUNKING_URL": base,
        "POC_CLAUDE_ENSEMBLE_EMBEDDING_URL": base,
        "POC_CLAUDE_ENSEMBLE_LEXICAL_URL": base,
        "POC_CLAUDE_ENSEMBLE_VECTOR_URL": base,
        "POC_CLAUDE_ENSEMBLE_INDEX_URL": base,
        "POC_CLAUDE_ENSEMBLE_HYBRID_URL": base,
        "POC_CLAUDE_ENSEMBLE_MEMORY_URL": base,
    }
