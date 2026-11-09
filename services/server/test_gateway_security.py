"""Security boundary tests for the POC HTTP gateway."""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import unittest

from server.ensemble_server import create_server


class GatewaySecurityTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            name: os.environ.get(name)
            for name in (
                "POC_CLAUDE_ENSEMBLE_ALLOWED_NETWORKS",
                "POC_CLAUDE_ENSEMBLE_AUTH_TOKEN",
                "POC_CLAUDE_ENSEMBLE_ALLOWED_HOSTS",
            )
        }
        os.environ["POC_CLAUDE_ENSEMBLE_ALLOWED_NETWORKS"] = "192.168.1.0/24,192.168.2.0/24"
        os.environ["POC_CLAUDE_ENSEMBLE_AUTH_TOKEN"] = "gateway-test-token"
        os.environ.pop("POC_CLAUDE_ENSEMBLE_ALLOWED_HOSTS", None)
        self.server = create_server("0.0.0.0", 0, graph_url="", memory_url="")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def request(self, method, path, *, headers=None, body=None, content_type="application/json"):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            data=None if body is None else body,
            method=method,
            headers={"Content-Type": content_type, **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_is_public_but_dashboard_requires_bearer_token(self):
        status, payload = self.request("GET", "/api/v1/health")
        self.assertEqual(status, 200, payload)
        status, payload = self.request("GET", "/dashboard")
        self.assertEqual(status, 401, payload)
        status, payload = self.request(
            "GET", "/dashboard",
            headers={"Authorization": "Bearer gateway-test-token"},
        )
        self.assertEqual(status, 200, payload)

    def test_rejects_untrusted_host(self):
        status, _ = self.request(
            "GET", "/api/v1/health",
            headers={"Host": f"rebind.attacker.example:{self.server.server_port}"},
        )
        self.assertEqual(status, 403)

    def test_rejects_foreign_origin(self):
        status, _ = self.request(
            "POST", "/api/v1/memory/write",
            headers={
                "Authorization": "Bearer gateway-test-token",
                "Origin": "https://attacker.example",
            },
            body=b'{"name":"test","content":"test"}',
        )
        self.assertEqual(status, 403)

    def test_rejects_non_json_mutation(self):
        status, _ = self.request(
            "POST", "/api/v1/memory/write",
            headers={"Authorization": "Bearer gateway-test-token"},
            body=b"name=test",
            content_type="text/plain",
        )
        self.assertEqual(status, 403)

    def test_non_loopback_bind_requires_token_at_startup(self):
        original = os.environ.pop("POC_CLAUDE_ENSEMBLE_AUTH_TOKEN", None)
        try:
            with self.assertRaisesRegex(ValueError, "POC_CLAUDE_ENSEMBLE_AUTH_TOKEN"):
                create_server("0.0.0.0", 0, graph_url="", memory_url="")
        finally:
            if original is not None:
                os.environ["POC_CLAUDE_ENSEMBLE_AUTH_TOKEN"] = original


if __name__ == "__main__":
    unittest.main()
