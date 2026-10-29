#!/usr/bin/env python3
"""Integration coverage for the shipped Claude Ensemble architecture."""

from __future__ import annotations

import importlib
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_modules_import() -> None:
    modules = (
        "arbitration.orchestrator",
        "arbitration.api_client",
        "learning.learning_client",
        "learning.autonomous_learning",
        "graph_service",
    )

    import sys

    sys.path.insert(0, str(ROOT / "graph-service"))
    sys.path.insert(0, str(ROOT / "learning"))
    try:
        for module in modules:
            importlib.import_module(module)
    finally:
        sys.path.remove(str(ROOT / "learning"))
        sys.path.remove(str(ROOT / "graph-service"))


def request(base_url: str, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_graph_http_end_to_end() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "graph-service"))
    try:
        from graph_service import create_server
    finally:
        sys.path.remove(str(ROOT / "graph-service"))

    with tempfile.TemporaryDirectory() as tmp:
        server = create_server("127.0.0.1", 0, Path(tmp) / "graph.json")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        deadline = time.monotonic() + 3
        while True:
            try:
                status, health = request(f"http://{host}:{port}", "GET", "/health")
                if status == 200 and health == {"ok": True, "service": "graph"}:
                    break
            except urllib.error.URLError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("graph service did not become ready")
            time.sleep(0.01)
        base_url = f"http://{host}:{port}"

        try:
            status, health = request(base_url, "GET", "/health")
            assert status == 200
            assert health == {"ok": True, "service": "graph"}

            status, left = request(
                base_url,
                "POST",
                "/graph/add-node",
                {"type": "person", "properties": {"name": "Ada"}},
            )
            assert status == 200
            assert left["ok"] is True

            status, right = request(
                base_url,
                "POST",
                "/graph/add-node",
                {"type": "project", "properties": {"name": "Ensemble"}},
            )
            assert status == 200
            assert right["ok"] is True

            status, edge = request(
                base_url,
                "POST",
                "/graph/add-edge",
                {
                    "source": left["node"]["id"],
                    "target": right["node"]["id"],
                    "type": "works_on",
                    "properties": {},
                },
            )
            assert status == 200
            assert edge["ok"] is True

            status, query = request(
                base_url,
                "POST",
                "/graph/query",
                {"type": "person", "properties": {"name": "Ada"}},
            )
            assert status == 200
            assert [node["id"] for node in query["nodes"]] == [left["node"]["id"]]

            status, traversal = request(
                base_url,
                "POST",
                "/graph/traverse",
                {"start": left["node"]["id"], "direction": "out", "max_depth": 1},
            )
            assert status == 200
            assert traversal["results"][0]["node"]["id"] == right["node"]["id"]

            status, persisted = request(
                base_url,
                "GET",
                f"/graph/node/{right['node']['id']}",
            )
            assert status == 200
            assert persisted["node"]["properties"]["name"] == "Ensemble"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
