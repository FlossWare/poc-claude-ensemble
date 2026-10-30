import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import http_server


class HttpServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), http_server.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:{}/mcp".format(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, request, headers=None):
        data = json.dumps(request).encode()
        request = urllib.request.Request(self.url, data=data, method="POST", headers=headers or {})
        return urllib.request.urlopen(request, timeout=2)

    def test_requires_bearer_token_when_configured(self):
        with patch.dict(os.environ, {"MCP_AUTH_TOKEN": "secret"}, clear=True):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(raised.exception.code, 401)

            response = self.post(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                {"Authorization": "Bearer secret"},
            )
            self.assertEqual(response.status, 200)

    def test_non_loopback_host_requires_token(self):
        with patch.dict(os.environ, {"MCP_HOST": "0.0.0.0"}, clear=True):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(raised.exception.code, 401)

    def test_notification_returns_202(self):
        with patch.dict(os.environ, {"MCP_HOST": "127.0.0.1"}, clear=True):
            response = self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self.assertEqual(response.status, 202)
            self.assertEqual(response.read(), b"")

    def test_tools_list_returns_200_json(self):
        with patch.dict(os.environ, {"MCP_HOST": "127.0.0.1"}, clear=True), \
             patch.object(http_server, "handle", return_value={
                 "jsonrpc": "2.0", "id": 1, "result": {"tools": []}
             }):
            response = self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["id"], 1)


if __name__ == "__main__":
    unittest.main()
