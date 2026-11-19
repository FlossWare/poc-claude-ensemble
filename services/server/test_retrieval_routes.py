"""HTTP contract tests for the retrieval-stack gateway routes.

The embedding, vector, lexical, hybrid, reranking, and extraction services keep
their compact API at the service root. The gateway prefix (/api/v1/lexical, ...)
is a public namespace only, so every one of these routes must reach the upstream
service with the prefix stripped.
"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVICES_ROOT = Path(__file__).resolve().parents[1]
for _path in (SERVICES_ROOT, SERVICES_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class EchoUpstreamHandler(BaseHTTPRequestHandler):
    """Records the path the gateway forwards and echoes it back."""

    def _respond(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.startswith("/unknown"):
            self._send(404, {"ok": False, "error": {"code": "not_found", "message": "unknown path"}})
            return
        payload = {"ok": True, "received": self.path, "body": json.loads(raw) if raw else None}
        self._send(200, payload)

    do_GET = _respond
    do_POST = _respond

    def _send(self, status, payload):
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


class RetrievalGatewayRouteTests(unittest.TestCase):
    def setUp(self):
        from server.ensemble_server import DEFAULT_SERVICE_URLS, ROOT_SCOPED_SERVICES, create_server

        self.defaults = DEFAULT_SERVICE_URLS
        self.root_scoped = ROOT_SCOPED_SERVICES
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), EchoUpstreamHandler)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        base = f"http://127.0.0.1:{self.upstream.server_port}"
        self.gateway = create_server("127.0.0.1", 0)
        for service in self.root_scoped:
            self.gateway.application.service_urls[service] = base
        self.gateway_thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gateway_thread.start()

    def tearDown(self):
        self.gateway.shutdown()
        self.gateway.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.gateway_thread.join(2)
        self.upstream_thread.join(2)

    def request(self, path, method="GET", payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway.server_port}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_retrieval_services_are_registered_with_their_loopback_ports(self):
        expected = {
            "embedding": "http://127.0.0.1:8770",
            "vector": "http://127.0.0.1:8771",
            "lexical": "http://127.0.0.1:8772",
            "hybrid": "http://127.0.0.1:8773",
            "reranking": "http://127.0.0.1:8774",
            "extraction": "http://127.0.0.1:8778",
        }
        self.assertEqual(set(self.root_scoped), set(expected))
        for service, url in expected.items():
            self.assertEqual(self.defaults[service], url)

    def test_each_public_route_reaches_the_service_root(self):
        cases = [
            ("/api/v1/embedding/model", "GET", None, "/model"),
            ("/api/v1/vector/health", "GET", None, "/health"),
            ("/api/v1/vector/collections", "GET", None, "/collections"),
            ("/api/v1/vector/collections/knowledge_chunks/records", "GET", None,
             "/collections/knowledge_chunks/records"),
            ("/api/v1/lexical/health", "GET", None, "/health"),
            ("/api/v1/lexical/search", "POST", {"query": "gateway routing"}, "/search"),
            ("/api/v1/lexical/knowledge/search", "POST", {"query": "gateway routing"}, "/knowledge/search"),
            ("/api/v1/hybrid/search", "POST", {"query": "gateway routing"}, "/search"),
            ("/api/v1/hybrid/knowledge/search", "POST", {"query": "gateway routing"}, "/knowledge/search"),
            ("/api/v1/reranking/rerank", "POST", {"query": "q", "documents": ["d"]}, "/rerank"),
            ("/api/v1/extraction/extract", "POST", {"text": "see https://example.org"}, "/extract"),
        ]
        for public, method, payload, upstream_path in cases:
            with self.subTest(public=public):
                status, body = self.request(public, method, payload)
                self.assertEqual(status, 200, body)
                self.assertTrue(body["ok"], body)
                self.assertEqual(body["received"], upstream_path)
                if payload is not None:
                    self.assertEqual(body["body"], payload)

    def test_query_string_is_preserved_after_prefix_stripping(self):
        status, body = self.request("/api/v1/vector/collections/knowledge_chunks/records?offset=5&limit=2")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["received"], "/collections/knowledge_chunks/records?offset=5&limit=2")

    def test_upstream_not_found_envelope_passes_through(self):
        status, body = self.request("/api/v1/vector/unknown")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"ok": False, "error": {"code": "not_found", "message": "unknown path"}})

    def test_unreachable_service_returns_service_unavailable_envelope(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        status, body = self.request("/api/v1/extraction/health")
        self.assertEqual(status, 503)
        self.assertFalse(body["ok"])
        self.assertIn("service unavailable", body["error"])

    def test_unknown_service_keeps_the_service_not_found_envelope(self):
        status, body = self.request("/api/v1/not-a-service/health")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"ok": False, "error": "service not found"})


if __name__ == "__main__":
    unittest.main()
