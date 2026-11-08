#!/usr/bin/env python3
"""POC source ingestion: URI resolution, content extraction, and Knowledge indexing.

The public source contract is URI-agnostic. This POC currently registers HTTP(S)
fetchers and PDF, HTML, plain-text, and Markdown extractors. Adding a transport
does not change the request contract.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlencode, urlsplit
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8782
MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 4_000_000
FETCH_TIMEOUT = 15
MAX_REDIRECTS = 3
CHUNK_PROFILE = "window-v1"
PROFILES = {
    "chunking": "window-v1@1",
    "embedding": "nomic-embed-text-v1.5@fp32",
    "vector": "knowledge_chunks@1",
}
EMBED_BATCH = 64

KNOWLEDGE_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_KNOWLEDGE_URL", "http://127.0.0.1:8781")
CHUNKING_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_CHUNKING_URL", "http://127.0.0.1:8775")
EMBEDDING_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
VECTOR_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
LEXICAL_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_LEXICAL_URL", "http://127.0.0.1:8772")
MEMORY_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_MEMORY_URL", "http://127.0.0.1:8767")


class IngestionError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _global_address(host: str) -> str:
    try:
        results = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise IngestionError("URI host could not be resolved", 400, "unresolvable_host") from exc
    addresses = sorted({item[4][0] for item in results})
    if not addresses:
        raise IngestionError("URI host resolved to no addresses", 400, "unresolvable_host")
    parsed = [ipaddress.ip_address(value.split("%", 1)[0]) for value in addresses]
    if any(not address.is_global for address in parsed):
        raise IngestionError("URI resolves to a non-public network address", 403, "unsafe_uri")
    return addresses[0]


def _fetch_http(uri: str) -> tuple[bytes, str, str]:
    current = uri
    for redirect in range(MAX_REDIRECTS + 1):
        parsed = urllib.parse.urlsplit(current)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            raise IngestionError(f"unsupported URI scheme: {scheme or '(missing)'}", 415, "unsupported_scheme")
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise IngestionError("URI must have a host and must not contain credentials", 400, "invalid_uri")
        if parsed.fragment:
            current = urllib.parse.urlunsplit(parsed._replace(fragment=""))
            parsed = urllib.parse.urlsplit(current)
        port = parsed.port or (443 if scheme == "https" else 80)
        if port != (443 if scheme == "https" else 80):
            raise IngestionError("only the standard HTTP and HTTPS ports are enabled", 403, "unsafe_uri")
        host = parsed.hostname.encode("idna").decode("ascii")
        address = _global_address(host)
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        sock = socket.create_connection((address, port), timeout=FETCH_TIMEOUT)
        try:
            if scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            connection = http.client.HTTPConnection(host, port, timeout=FETCH_TIMEOUT)
            connection.sock = sock
            connection.request("GET", path, headers={
                "Host": host,
                "User-Agent": "FlossWare-POC-Ingestion/1",
                "Accept": "application/pdf,text/html,text/plain,text/markdown,application/octet-stream;q=0.5",
                "Connection": "close",
            })
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                response.read(64 * 1024)
                connection.close()
                if not location or redirect >= MAX_REDIRECTS:
                    raise IngestionError("remote URI redirect limit exceeded", 400, "redirect_limit")
                current = urllib.parse.urljoin(current, location)
                continue
            if response.status < 200 or response.status >= 300:
                raise IngestionError(f"remote server returned HTTP {response.status}", 502, "remote_http_error")
            declared = response.getheader("Content-Length")
            if declared and (not declared.isdigit() or int(declared) > MAX_SOURCE_BYTES):
                raise IngestionError("remote resource exceeds the 8 MiB limit", 413, "source_too_large")
            data = response.read(MAX_SOURCE_BYTES + 1)
            if len(data) > MAX_SOURCE_BYTES:
                raise IngestionError("remote resource exceeds the 8 MiB limit", 413, "source_too_large")
            media_type = (response.getheader("Content-Type") or "application/octet-stream").split(";", 1)[0].strip().lower()
            final_uri = current
            connection.close()
            return data, media_type, final_uri
        finally:
            try:
                sock.close()
            except OSError:
                pass
    raise IngestionError("remote URI redirect limit exceeded", 400, "redirect_limit")


FETCHERS = {"http": _fetch_http, "https": _fetch_http}


def _extract(data: bytes, media_type: str, filename: str = "") -> tuple[str, str]:
    if len(data) > MAX_SOURCE_BYTES:
        raise IngestionError("source exceeds the 8 MiB limit", 413, "source_too_large")
    suffix = PurePosixPath(filename).suffix.lower()
    if media_type == "application/pdf" or suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(data)
            handle.flush()
            try:
                result = subprocess.run(
                    ["pdftotext", "-q", handle.name, "-"],
                    capture_output=True, timeout=60, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise IngestionError("PDF extraction tool is unavailable or timed out", 503, "extractor_unavailable") from exc
        if result.returncode != 0:
            raise IngestionError("PDF text extraction failed", 422, "extraction_failed")
        text = result.stdout.decode("utf-8", errors="replace")
        resolved_type = "application/pdf"
    elif media_type in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
        parser = _VisibleText()
        parser.feed(data.decode("utf-8", errors="replace"))
        text = "\n".join(parser.parts)
        resolved_type = "text/html"
    elif media_type.startswith("text/") or suffix in {".md", ".markdown", ".txt", ".rst"}:
        text = data.decode("utf-8-sig", errors="replace")
        resolved_type = "text/markdown" if suffix in {".md", ".markdown"} or media_type == "text/markdown" else "text/plain"
    else:
        raise IngestionError(f"unsupported content type: {media_type}", 415, "unsupported_media_type")
    text = re.sub(r"\n[ \t]+", "\n", text).strip()
    if not text:
        raise IngestionError("source contains no extractable text", 422, "empty_extraction")
    if len(text) > MAX_TEXT_CHARS:
        raise IngestionError("extracted text exceeds the 4,000,000 character limit", 413, "text_too_large")
    return text, resolved_type


def resolve_source(body: dict[str, Any]) -> dict[str, Any]:
    supplied = [key for key in ("uri", "content_base64", "text") if key in body]
    if len(supplied) != 1:
        raise IngestionError("provide exactly one of uri, content_base64, or text")
    if "uri" in body:
        uri = body.get("uri")
        if not isinstance(uri, str) or not uri.strip() or len(uri) > 4096:
            raise IngestionError("uri must be a non-empty string of at most 4096 characters")
        parsed = urllib.parse.urlsplit(uri.strip())
        fetcher = FETCHERS.get(parsed.scheme.lower())
        if fetcher is None:
            raise IngestionError(f"unsupported URI scheme: {parsed.scheme or '(missing)'}", 415, "unsupported_scheme")
        data, media_type, final_uri = fetcher(uri.strip())
        text, extracted_type = _extract(data, media_type, PurePosixPath(parsed.path).name)
        title = PurePosixPath(urllib.parse.urlsplit(final_uri).path).name or host_title(final_uri)
        return {"source_id": final_uri, "locator": final_uri, "origin": "remote-uri",
                "title": title[:300], "media_type": extracted_type, "text": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if "content_base64" in body:
        encoded = body.get("content_base64")
        filename = body.get("filename")
        media_type = body.get("media_type", "application/octet-stream")
        if not isinstance(encoded, str) or not isinstance(filename, str) or not filename.strip():
            raise IngestionError("filename and content_base64 are required for file uploads")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise IngestionError("content_base64 is not valid base64") from exc
        text, extracted_type = _extract(data, str(media_type).lower(), filename)
        digest = hashlib.sha256(data).hexdigest()
        safe_name = PurePosixPath(filename.replace("\\", "/")).name
        locator = f"upload://sha256/{digest}/{safe_name}"
        return {"source_id": locator, "locator": locator, "origin": "upload",
                "title": safe_name[:300], "media_type": extracted_type, "text": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if "text" in body:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise IngestionError("text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise IngestionError("text exceeds the 4,000,000 character limit", 413, "text_too_large")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        title = body.get("title", "Pasted text")
        if not isinstance(title, str) or not title.strip():
            raise IngestionError("title must be a non-empty string")
        locator = f"text://sha256/{digest}"
        return {"source_id": locator, "locator": locator, "origin": "inline-text",
                "title": title[:300], "media_type": "text/plain", "text": text.strip(),
                "content_sha256": digest}
    raise IngestionError("provide exactly one of uri, content_base64, or text")


def host_title(uri: str) -> str:
    return urllib.parse.urlsplit(uri).hostname or "Remote document"


def _post_json(base: str, path: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise IngestionError(f"downstream service returned HTTP {exc.code}", 502, "downstream_failure") from exc
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise IngestionError("a required ingestion capability is unavailable", 503, "dependency_unavailable") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise IngestionError("a downstream ingestion capability rejected the request", 502, "downstream_failure")
    return result


def preview(body: dict[str, Any]) -> dict[str, Any]:
    source = resolve_source(body)
    return {
        "ok": True,
        "source": {key: source[key] for key in ("source_id", "locator", "origin", "title", "media_type", "content_sha256")},
        "characters": len(source["text"]),
        "preview": source["text"][:3000],
        "planned_stages": ["knowledge", "chunking", "embedding", "lexical-index", "vector-index"],
        "persisted": False,
    }


def ingest(body: dict[str, Any]) -> dict[str, Any]:
    source = resolve_source(body)
    ingestion_id = str(uuid4())
    _record_ingestion_event(source, ingestion_id, "ingestion.started", status="running")
    content = source["text"]
    revision_id = source["content_sha256"]
    try:
        knowledge = _post_json(KNOWLEDGE_URL, "/records", {
            "source_id": source["source_id"],
            "revision_sequence": time.time_ns(),
            "revision_id": revision_id,
            "title": source["title"],
            "content": content,
            "provenance": {"origin": source["origin"], "locator": source["locator"]},
            "profiles": PROFILES,
        })
        if knowledge.get("decision") not in {"accepted", "duplicate"}:
            _record_ingestion_event(source, ingestion_id, "ingestion.rejected", status="rejected")
            return {"ok": True, "status": "rejected", "ingestion_id": ingestion_id,
                    "source_id": source["source_id"], "decision": knowledge.get("decision"),
                    "chunks": 0, "vectors": 0}
        chunked = _post_json(CHUNKING_URL, "/chunk", {
            "profile": CHUNK_PROFILE, "source_id": source["source_id"],
            "media_type": "text/plain", "text": content,
        })
        chunks = chunked.get("chunks")
        if not isinstance(chunks, list):
            raise IngestionError("chunking service returned an invalid result", 502, "invalid_downstream_response")
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stored = 0
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            texts = [item["text"] for item in batch]
            embedded = _post_json(EMBEDDING_URL, "/embeddings", {"texts": texts, "mode": "document"})
            vectors = embedded.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise IngestionError("embedding service returned an invalid vector count", 502, "invalid_downstream_response")
            rows = []
            ids = []
            metadatas = []
            for item in batch:
                chunk_id = item["chunk_id"]
                vector_id = hashlib.sha256(f"{chunk_id}|{content_sha}".encode("utf-8")).hexdigest()
                meta = {"kind": "knowledge_chunk", "source": source["origin"], "url": source["locator"],
                        "title": source["title"], "chunk": item.get("index"),
                        "chunks": len(chunks), "chunk_id": chunk_id, "content_sha256": content_sha}
                ids.append(vector_id)
                metadatas.append(meta)
                rows.append({"chunk_id": chunk_id, "vector_id": vector_id, "url": source["locator"],
                             "source": source["origin"], "title": source["title"], "text": item["text"],
                             "chunk_index": item.get("index"), "chunk_count": len(chunks),
                             "content_sha256": content_sha})
            _post_json(LEXICAL_URL, "/knowledge/documents", {"chunks": rows})
            _post_json(VECTOR_URL, "/collections/knowledge_chunks/upsert", {
                "ids": ids, "embeddings": vectors, "documents": texts, "metadatas": metadatas,
            })
            stored += len(batch)
        # Remove superseded vectors only after the new revision has been indexed successfully.
        _post_json(VECTOR_URL, "/collections/knowledge_chunks/delete-superseded", {
            "url": source["locator"], "keep_sha256": content_sha,
        })
        _record_ingestion_event(
            source, ingestion_id, "ingestion.completed", status="completed",
            chunks=len(chunks), vectors=stored, content_sha256=content_sha,
        )
        return {"ok": True, "status": "completed", "ingestion_id": ingestion_id,
                "source_id": source["source_id"], "decision": knowledge.get("decision"),
                "chunks": len(chunks), "vectors": stored, "content_sha256": content_sha}
    except Exception as exc:
        _record_ingestion_event(
            source, ingestion_id, "ingestion.failed", status="failed",
            error_type=type(exc).__name__,
        )
        raise


def _record_ingestion_event(
    source: dict[str, Any],
    ingestion_id: str,
    event_name: str,
    *,
    status: str,
    chunks: int | None = None,
    vectors: int | None = None,
    content_sha256: str | None = None,
    error_type: str | None = None,
) -> None:
    """Best-effort metadata-only event in the existing append-only POC ledger."""
    event_id = str(uuid4())
    entry = {
        "event_id": event_id,
        "event_type": "ingestion",
        "event_name": event_name,
        "source": "poc-ingestion",
        "producer": "poc-claude-ensemble",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "correlation_id": ingestion_id,
        "execution_id": ingestion_id,
        "source_id": source["source_id"],
        "title": source["title"],
        "origin": source["origin"],
        "media_type": source["media_type"],
        "status": status,
    }
    if chunks is not None:
        entry["chunk_count"] = chunks
    if vectors is not None:
        entry["vector_count"] = vectors
    if content_sha256 is not None:
        entry["content_sha256"] = content_sha256
    if error_type is not None:
        entry["error_type"] = error_type
    try:
        _post_json(MEMORY_URL, "/memory/append-once", {
            "name": "session_events", "event_id": event_id, "entry": entry,
        }, timeout=5)
    except IngestionError:
        # Ingestion success/failure is not hidden or rolled back because optional observability is down.
        pass


def _body(handler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "")
    if not raw_length.isdigit():
        raise IngestionError("Content-Length is required")
    length = int(raw_length)
    if length <= 0 or length > MAX_BODY_BYTES:
        raise IngestionError("request body must be between 1 byte and 12 MiB", 413, "payload_too_large")
    value = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(value, dict):
        raise IngestionError("request body must be a JSON object")
    return value


class IngestionHandler(BaseHTTPRequestHandler):
    server_version = "FlossWarePOCIngestion/1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._send(200, {"ok": True, "service": "ingestion", "schemes": ["http", "https"],
                             "media_types": ["application/pdf", "text/html", "text/plain", "text/markdown"]})
            return
        if parsed.path == "/history":
            try:
                params = parse_qs(parsed.query, strict_parsing=True) if parsed.query else {}
                if set(params) - {"limit", "cursor"} or any(len(values) != 1 for values in params.values()):
                    raise IngestionError("history supports only limit and cursor")
                limit = int(params.get("limit", ["50"])[0])
                raw_cursor = params.get("cursor", [None])[0]
                if limit < 1 or limit > 100:
                    raise IngestionError("limit must be 1..100 and cursor must be non-negative")
                if raw_cursor is None:
                    first_query = urlencode({"event_type": "ingestion", "limit": "1", "cursor": "0"})
                    first_request = urllib.request.Request(MEMORY_URL.rstrip("/") + "/history?" + first_query, method="GET")
                    with urllib.request.urlopen(first_request, timeout=5) as first_response:
                        first_result = json.loads(first_response.read().decode("utf-8"))
                    cursor = max(0, int(first_result.get("pagination", {}).get("total", 0)) - limit)
                else:
                    cursor = int(raw_cursor)
                if cursor < 0:
                    raise IngestionError("limit must be 1..100 and cursor must be non-negative")
                query = urlencode({"event_type": "ingestion", "limit": str(limit), "cursor": str(cursor)})
                request = urllib.request.Request(MEMORY_URL.rstrip("/") + "/history?" + query, method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.loads(response.read().decode("utf-8"))
                self._send(200, result)
            except IngestionError as exc:
                self._send(exc.status, {"ok": False, "error_code": exc.code, "error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"ok": False, "error_code": "invalid_request", "error": str(exc)})
            except (urllib.error.URLError, OSError):
                self._send(503, {"ok": False, "error_code": "history_unavailable", "error": "ingestion history unavailable"})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            body = _body(self)
            if self.path == "/preview":
                self._send(200, preview(body))
            elif self.path == "/ingest":
                self._send(200, ingest(body))
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except IngestionError as exc:
            self._send(exc.status, {"ok": False, "error_code": exc.code, "error": str(exc)})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error_code": "invalid_request", "error": str(exc)})
        except Exception:
            self._send(502, {"ok": False, "error_code": "ingestion_failed", "error": "ingestion failed"})


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("ingestion service only accepts loopback binds")
    return ThreadingHTTPServer((host, port), IngestionHandler)


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("POC_CLAUDE_ENSEMBLE_INGESTION_HOST", DEFAULT_HOST)
    port = int(os.environ.get("POC_CLAUDE_ENSEMBLE_INGESTION_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
