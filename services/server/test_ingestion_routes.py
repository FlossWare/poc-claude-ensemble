import json
import os
import threading
import urllib.request
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch


class IngestionUpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200, {"ok": True, "service": "ingestion", "schemes": ["http", "https"]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self._send(200, {"ok": True, "preview": payload.get("uri")})

    def _send(self, status, payload):
        data = (json.dumps(payload) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


class IngestionGatewayRouteTests(unittest.TestCase):
    def test_gateway_forwards_ingestion_routes_to_service_root_paths(self):
        import sys
        from pathlib import Path
        services_root = Path(__file__).resolve().parents[1]
        if str(services_root) not in sys.path:
            sys.path.insert(0, str(services_root))
        from server.ensemble_server import create_server

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), IngestionUpstreamHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        gateway = create_server("127.0.0.1", 0)
        gateway.application.service_urls["ingestion"] = f"http://127.0.0.1:{upstream.server_port}"
        gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gateway_thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{gateway.server_port}/api/v1/ingestion/health", timeout=3
            ) as response:
                health = json.loads(response.read())
            self.assertTrue(health["ok"])
            request = urllib.request.Request(
                f"http://127.0.0.1:{gateway.server_port}/api/v1/ingestion/preview",
                data=json.dumps({"uri": "https://example.org/report.pdf"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                preview = json.loads(response.read())
            self.assertEqual(preview["preview"], "https://example.org/report.pdf")
        finally:
            gateway.shutdown()
            gateway.server_close()
            upstream.shutdown()
            upstream.server_close()
            gateway_thread.join(2)
            upstream_thread.join(2)


if __name__ == "__main__":
    unittest.main()
