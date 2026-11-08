import json
import os
import threading
import time
import urllib.error
import urllib.request
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch


class IngestionUpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "service": "ingestion", "schemes": ["http", "https"]})
        elif self.path.startswith("/history?"):
            self._send(200, {"ok": True, "entries": [{"event_name": "ingestion.completed"}]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

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
            with urllib.request.urlopen(
                f"http://127.0.0.1:{gateway.server_port}/api/v1/ingestion/history?limit=10", timeout=3
            ) as response:
                history = json.loads(response.read())
            self.assertEqual(history["entries"][0]["event_name"], "ingestion.completed")
        finally:
            gateway.shutdown()
            gateway.server_close()
            upstream.shutdown()
            upstream.server_close()
            gateway_thread.join(2)
            upstream_thread.join(2)


if __name__ == "__main__":
    unittest.main()


class SlowUpstreamHandler(BaseHTTPRequestHandler):
    """Answers after a delay, like an ingestion run that chunks, embeds, and indexes before replying."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        time.sleep(0.8)
        self._send(200, {"ok": True, "status": "completed"})

    def do_GET(self):
        time.sleep(0.8)
        self._send(200, {"ok": True})

    def _send(self, status, payload):
        data = (json.dumps(payload) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


class IngestionForwardTimeoutTests(unittest.TestCase):
    def test_ingestion_route_gets_longer_budget_than_other_services(self):
        import sys
        from pathlib import Path
        services_root = Path(__file__).resolve().parents[1]
        for path in (services_root, services_root / "server"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from server import ensemble_server

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), SlowUpstreamHandler)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        gateway = ensemble_server.create_server("127.0.0.1", 0)
        base = f"http://127.0.0.1:{upstream.server_port}"
        gateway.application.service_urls.update({"ingestion": base, "graph": base})
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        try:
            with patch.object(ensemble_server, "FORWARD_TIMEOUT_SECONDS", 0.2), \
                    patch.object(ensemble_server, "INGESTION_FORWARD_TIMEOUT_SECONDS", 3.0):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{gateway.server_port}/api/v1/ingestion/ingest",
                    data=json.dumps({"preview_id": "p"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read())["status"], "completed")
                # Control: the same delay on another service still exceeds the default budget.
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{gateway.server_port}/api/v1/graph/slow", timeout=10)
                self.assertEqual(caught.exception.code, 503)
        finally:
            gateway.shutdown()
            gateway.server_close()
            upstream.shutdown()
            upstream.server_close()
