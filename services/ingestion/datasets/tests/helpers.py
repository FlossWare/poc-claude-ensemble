"""Shared offline test helpers and fakes."""
from __future__ import annotations

import io
import json
import shutil
import socket
import tarfile
import threading
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
