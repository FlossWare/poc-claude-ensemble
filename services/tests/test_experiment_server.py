"""REST tests for lightweight experiment evaluation."""
from __future__ import annotations
import json
import threading
import urllib.error
import urllib.request

from server.ensemble_server import create_server

def request(server, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())

def test_experiment_evaluation_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ENSEMBLE_EXPERIMENT_FILE", str(tmp_path / "experiments.jsonl"))
    server = create_server("127.0.0.1", 0, graph_url="", memory_url="")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        experiment = {
            "experiment_id": "rest-1",
            "hypothesis": "variant improves quality",
            "baseline": {"strategy": "current"},
            "variant": {"strategy": "candidate"},
            "inputs": {"dataset": "replay"},
            "measurements": {
                "quality": {"baseline": 0.7, "variant": 0.9, "direction": "higher"}
            },
        }
        status, body = request(server, "POST", "/api/v1/experiments/evaluate", experiment)
        assert status == 200
        assert body["ok"] and body["result"]["winner"] == "variant"

        status, body = request(server, "GET", "/api/v1/experiments/rest-1")
        assert status == 200
        assert body["result"]["experiment_id"] == "rest-1"

        status, body = request(server, "POST", "/api/v1/experiments/evaluate",
                               {**experiment, "measurements": {}})
        assert status == 400
        assert body["error_code"] == "invalid_request"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
