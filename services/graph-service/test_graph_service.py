#!/usr/bin/env python3
"""End-to-end tests for the Graph HTTP boundary and durable store."""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from graph_service import GraphStore, create_server


def request(server, method, path, payload=None):
    url = f"http://127.0.0.1:{server.server_port}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "graph.json"
        server = create_server("127.0.0.1", 0, store_path)
        thread = start_server(server)
        try:
            status, body = request(server, "GET", "/health")
            assert status == 200 and body["ok"] is True

            status, body = request(
                server, "POST", "/graph/add-node",
                {"type": "person", "properties": {"name": "Alice"}},
            )
            assert status == 200
            alice = body["node"]

            status, body = request(
                server, "POST", "/graph/add-node",
                {"type": "person", "properties": {"name": "Bob"}},
            )
            assert status == 200
            bob = body["node"]

            status, body = request(
                server, "POST", "/graph/add-edge",
                {
                    "source": alice["id"],
                    "target": bob["id"],
                    "type": "knows",
                    "properties": {},
                },
            )
            assert status == 200
            edge = body["edge"]
            assert edge["source"] == alice["id"]
            assert edge["target"] == bob["id"]

            status, body = request(server, "GET", f"/graph/node/{alice['id']}")
            assert status == 200, body
            assert body["node"] == alice

            status, body = request(
                server, "POST", "/graph/query",
                {"type": "person", "properties": {"name": "Bob"}},
            )
            assert status == 200
            assert body["nodes"] == [bob]

            status, body = request(
                server, "POST", "/graph/traverse",
                {"start": alice["id"], "direction": "out", "max_depth": 1},
            )
            assert status == 200
            assert [item["node"] for item in body["results"]] == [bob]

            status, body = request(server, "GET", "/graph/node/missing")
            assert status == 404
            assert body["ok"] is False

            status, body = request(
                server, "POST", "/graph/add-edge",
                {"source": alice["id"], "target": "missing", "type": "knows", "properties": {}},
            )
            assert status == 400
            assert body["ok"] is False

            status, body = request(
                server, "POST", "/graph/add-node",
                {"id": alice["id"], "type": "person", "properties": {"name": "Different"}},
            )
            assert status == 400
            assert body["ok"] is False

            original_persist = server.graph_store._persist
            server.graph_store._persist = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected persistence failure"))
            try:
                status, body = request(
                    server, "POST", "/graph/add-node",
                    {"type": "person", "properties": {"name": "Failure"}},
                )
                assert status == 500
                failed_id = body["error"]
                assert "injected persistence failure" in failed_id
            finally:
                server.graph_store._persist = original_persist

            status, body = request(
                server, "POST", "/graph/query",
                {"type": "person", "properties": {"name": "Failure"}},
            )
            assert status == 200
            assert body["nodes"] == []

            status, body = request(
                server, "POST", "/graph/add-node",
                {"type": "person", "properties": {"name": "Failure"}},
            )
            assert status == 200
            failed_node = body["node"]

            server.shutdown()
            server.server_close()

            restarted = create_server("127.0.0.1", 0, store_path)
            restarted_thread = start_server(restarted)
            try:
                status, body = request(restarted, "GET", f"/graph/node/{alice['id']}")
                assert status == 200, body
                assert body["node"] == alice

                status, body = request(restarted, "GET", f"/graph/node/{urllib.parse.quote(failed_node['id'], safe='')}")
                assert status == 200
                assert body["node"] == failed_node

                status, body = request(
                    restarted, "POST", "/graph/add-node",
                    {"type": "person", "properties": {"name": "Alice"}},
                )
                assert status == 200
                assert body["node"]["id"] == alice["id"]
            finally:
                restarted.shutdown()
                restarted.server_close()
                restarted_thread.join(timeout=3)

            assert store_path.exists()
        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            server.server_close()
            thread.join(timeout=3)

    try:
        create_server("0.0.0.0", 0, store_path)
    except ValueError:
        pass
    else:
        raise AssertionError("non-loopback Graph bind was accepted")

    original_host = os.environ.get("POC_CLAUDE_ENSEMBLE_GRAPH_HOST")
    try:
        os.environ["POC_CLAUDE_ENSEMBLE_GRAPH_HOST"] = "0.0.0.0"
        try:
            from graph_service import _configured_host
            _configured_host()
        except ValueError:
            pass
        else:
            raise AssertionError("non-loopback environment override was accepted")
    finally:
        if original_host is None:
            os.environ.pop("POC_CLAUDE_ENSEMBLE_GRAPH_HOST", None)
        else:
            os.environ["POC_CLAUDE_ENSEMBLE_GRAPH_HOST"] = original_host

    print("graph service tests passed")


if __name__ == "__main__":
    main()


def test_remove_nodes_by_ids_and_prefix(tmp_path):
    import graph_service as gs
    store = gs.GraphStore(tmp_path / "g.json")
    store.add_node("file", {"path": "/old/repo/a.py"}, "file:a")
    store.add_node("file", {"path": "/new/repo/b.py"}, "file:b")
    store.add_node("file", {"path": "/old/repo/c.py"}, "file:c")
    store.add_edge("file:a", "file:b", "relates", {})

    # dry run reports without mutating
    dry = store.remove_nodes(path_prefix="/old/repo", dry_run=True)
    assert dry["would_remove_nodes"] == 2 and dry["dry_run"] is True
    assert store.graph.has_node("file:a")

    # real removal drops nodes and their edges, persists
    result = store.remove_nodes(path_prefix="/old/repo")
    assert result["removed_nodes"] == 2 and result["removed_edges"] >= 1
    assert not store.graph.has_node("file:a") and store.graph.has_node("file:b")

    # reloaded from disk: removal is durable
    store2 = gs.GraphStore(tmp_path / "g.json")
    assert not store2.graph.has_node("file:a") and store2.graph.has_node("file:b")

    # explicit ids + missing reporting
    result = store.remove_nodes(ids=["file:b", "file:ghost"])
    assert result["removed_nodes"] == 1 and result["missing"] == ["file:ghost"]

    # bad input rejected
    try:
        store.remove_nodes()
        assert False, "expected ValueError"
    except ValueError:
        pass
