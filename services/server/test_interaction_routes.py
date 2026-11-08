import json
import threading
import urllib.error
import urllib.request
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MemoryHistoryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/v1/health":
            body = {"ok": True, "service": "memory"}
            status = 200
        elif self.path == "/history?event_type=model_interaction&limit=10&cursor=0":
            body = {
                "ok": True,
                "entries": [{
                    "event_id": "event-1",
                    "event_type": "model_interaction",
                    "event_name": "worker.started",
                    "execution_id": "execution-1",
                }],
                "pagination": {"limit": 10, "next_cursor": None, "returned": 1, "total": 1},
            }
            status = 200
        else:
            body = {"ok": False, "error": "not found"}
            status = 404
        data = (json.dumps(body) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


class InteractionRouteTests(unittest.TestCase):
    def test_gateway_exposes_historical_model_interactions(self):
        import sys
        from pathlib import Path
        services_root = Path(__file__).resolve().parents[1]
        if str(services_root) not in sys.path:
            sys.path.insert(0, str(services_root))
        from server.ensemble_server import create_server

        memory = ThreadingHTTPServer(("127.0.0.1", 0), MemoryHistoryHandler)
        memory_thread = threading.Thread(target=memory.serve_forever, daemon=True)
        memory_thread.start()
        gateway = create_server(
            "127.0.0.1", 0,
            memory_url=f"http://127.0.0.1:{memory.server_port}",
        )
        gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gateway_thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{gateway.server_port}/api/v1/interactions?limit=10&cursor=0",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["entries"][0]["event_name"], "worker.started")
            self.assertEqual(payload["entries"][0]["execution_id"], "execution-1")
        finally:
            gateway.shutdown()
            gateway.server_close()
            memory.shutdown()
            memory.server_close()
            gateway_thread.join(2)
            memory_thread.join(2)


if __name__ == "__main__":
    unittest.main()
