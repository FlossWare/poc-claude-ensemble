"""Integration tests for acceptance.py using controlled local services.

Runs the real ingestion HTTP server and the real REST gateway, with in-process stub
Knowledge, Chunking, Embedding, Lexical, Vector, Memory, and Graph services. No
external network, no live services, and no writes outside the stubs.
"""
import contextlib
import hashlib
import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
SERVICES_ROOT = HERE.parent
for _path in (SERVICES_ROOT, SERVICES_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import acceptance  # noqa: E402
import ingestion_service as ingestion  # noqa: E402


class StubDownstream(BaseHTTPRequestHandler):
    """One loopback server standing in for every downstream POC service."""

    def log_message(self, *_args):
        return

    def _send(self, status, payload):
        data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        return json.loads(self.rfile.read(length).decode("utf-8") or "null")

    @property
    def store(self):
        return self.server.store

    def do_GET(self):
        parsed = urlsplit(self.path)
        # Graph and Memory answer readiness with their own service name, as the gateway requires.
        if parsed.path == "/health":
            return self._send(200, {"ok": True, "service": "graph"})
        if parsed.path == "/ready":
            return self._send(200, {"ok": True, "ready": True, "service": "memory"})
        if parsed.path.startswith("/records/"):
            record = self.store["records"].get(unquote(parsed.path[len("/records/"):]))
            # Matches the live Knowledge API: the record is wrapped as {"ok": true, "record": {...}}.
            return self._send(200, {"ok": True, "record": record}) if record \
                else self._send(404, {"ok": False, "error_code": "not_found"})
        if parsed.path == "/history":
            query = parse_qs(parsed.query)
            limit, cursor = int(query["limit"][0]), int(query["cursor"][0])
            selected = [e for e in self.store["memory"] if e.get("event_type") == query["event_type"][0]]
            page = selected[cursor:cursor + limit]
            return self._send(200, {"ok": True, "entries": page, "pagination": {
                "limit": limit, "returned": len(page), "total": len(selected), "next_cursor": None}})
        return self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = urlsplit(self.path).path
        body = self._body()
        store = self.store
        if path == "/records":
            store["records"][body["source_id"]] = {
                "source_id": body["source_id"], "content": body["content"],
                "content_sha256": hashlib.sha256(body["content"].encode("utf-8")).hexdigest(),
                "provenance": body["provenance"], "tombstoned": False,
                "revision_sequence": body["revision_sequence"]}
            return self._send(200, {"ok": True, "decision": "accepted"})
        if path.startswith("/records/") and path.endswith("/tombstone"):
            source_id = unquote(path.split("/")[2])
            if source_id not in store["records"]:
                return self._send(404, {"ok": False, "error_code": "not_found"})
            store["records"][source_id]["tombstoned"] = True
            return self._send(200, {"ok": True, "decision": "tombstoned", "source_id": source_id})
        if path == "/chunk":
            text = body["text"]
            chunks = [{"chunk_id": f"{body['source_id']}#{i}", "index": i, "text": text[start:start + 200]}
                      for i, start in enumerate(range(0, len(text), 200))]
            return self._send(200, {"ok": True, "chunks": chunks})
        if path == "/embeddings":
            vectors = [[float(sum(map(ord, t)) % 997), float(len(t)), 1.0] for t in body["texts"]]
            return self._send(200, {"ok": True, "embeddings": vectors})
        if path == "/knowledge/documents":
            store["lexical"].extend(body["chunks"])
            return self._send(200, {"ok": True})
        if path == "/knowledge/search":
            needle = body["query"].lower()
            hits = [{"url": row["url"], "vector_id": row["vector_id"], "text": row["text"]}
                    for row in store["lexical"] if needle in row["text"].lower()]
            return self._send(200, {"ok": True, "results": hits[: body.get("top_k", 10)]})
        if path == "/collections/knowledge_chunks/upsert":
            for vector_id, embedding, document, metadata in zip(
                    body["ids"], body["embeddings"], body["documents"], body["metadatas"]):
                store["vectors"][vector_id] = (embedding, document, metadata)
            return self._send(200, {"ok": True})
        if path == "/collections/knowledge_chunks/delete-superseded":
            return self._send(200, {"ok": True, "removed": 0})
        if path == "/collections/knowledge_chunks/query":
            query = body["embedding"]
            scored = sorted(
                ({"id": vid, "document": doc, "score": sum(a * b for a, b in zip(query, emb)), "metadata": meta}
                 for vid, (emb, doc, meta) in store["vectors"].items()),
                key=lambda item: item["score"], reverse=True)
            return self._send(200, {"ok": True, "results": scored[: body.get("top_k", 10)]})
        if path == "/memory/append-once":
            entry = dict(body["entry"])
            if store["inject_leak"]:
                entry["headers"] = {"Authorization": "Bearer stub-credential"}
            store["memory"].append(entry)
            return self._send(200, {"ok": True, "appended": True})
        return self._send(404, {"ok": False, "error": "not found"})


class AcceptanceHarnessIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.ensemble_server import create_server as create_gateway

        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubDownstream)
        cls.stub.store = {"records": {}, "lexical": [], "vectors": {}, "memory": [], "inject_leak": False}
        cls.stub_base = f"http://127.0.0.1:{cls.stub.server_port}"
        cls.ingestion_server = ingestion.create_server("127.0.0.1", 0)
        cls.ingestion_base = f"http://127.0.0.1:{cls.ingestion_server.server_port}"
        cls.gateway = create_gateway("127.0.0.1", 0)
        cls.gateway.application.service_urls.update({
            "ingestion": cls.ingestion_base, "memory": cls.stub_base, "graph": cls.stub_base})
        cls.gateway_base = f"http://127.0.0.1:{cls.gateway.server_port}/api/v1"
        cls.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                       for server in (cls.stub, cls.ingestion_server, cls.gateway)]
        for thread in cls.threads:
            thread.start()
        cls.patches = patch.multiple(
            ingestion, KNOWLEDGE_URL=cls.stub_base, CHUNKING_URL=cls.stub_base, EMBEDDING_URL=cls.stub_base,
            LEXICAL_URL=cls.stub_base, VECTOR_URL=cls.stub_base, MEMORY_URL=cls.stub_base)
        cls.patches.start()
        cls.quiet = patch.object(ingestion.IngestionHandler, "log_message", lambda self, *args: None)
        cls.quiet.start()

    @classmethod
    def tearDownClass(cls):
        cls.quiet.stop()
        cls.patches.stop()
        for server in (cls.gateway, cls.ingestion_server, cls.stub):
            server.shutdown()
            server.server_close()

    def setUp(self):
        self.stub.store["inject_leak"] = False

    def config(self, *, confirm):
        return acceptance.Config(gateway=self.gateway_base, knowledge=self.stub_base, lexical=self.stub_base,
                                 vector=self.stub_base, embedding=self.stub_base, confirm=confirm)

    def test_preview_only_run_stops_before_any_write(self):
        records_before = len(self.stub.store["records"])
        report = acceptance.run(self.config(confirm=False))
        self.assertEqual(report["outcome"], "awaiting_confirmation", report)
        self.assertEqual(len(self.stub.store["records"]), records_before)
        self.assertIsNone(report["cleanup"])

    def test_confirmed_run_passes_every_step_and_cleans_up(self):
        report = acceptance.run(self.config(confirm=True))
        self.assertEqual(report["outcome"], "passed", report)
        self.assertIsNone(report["failed_step"])
        self.assertTrue(all(step["ok"] for step in report["steps"]), report["steps"])
        names = [step["name"] for step in report["steps"]]
        for expected in ("ingestion completed", "replay of consumed preview refused",
                         "lexical search finds fixture", "vector search finds fixture",
                         "ledger records lifecycle events", "ledger has no forbidden keys"):
            self.assertIn(expected, names)
        self.assertTrue(report["cleanup"]["knowledge_tombstoned"])
        self.assertEqual(report["cleanup"]["residual_indexes"], ["lexical", "vector"])

    def test_failed_ingestion_reports_failed_step_and_still_cleans_up(self):
        dead_port = 1  # nothing listens here; the ingestion service sees a dependency outage
        with patch.object(ingestion, "EMBEDDING_URL", f"http://127.0.0.1:{dead_port}"):
            report = acceptance.run(self.config(confirm=True))
        self.assertEqual(report["outcome"], "failed")
        self.assertEqual(report["failed_step"], "ingestion completed")
        self.assertTrue(report["cleanup"]["knowledge_tombstoned"], report["cleanup"])

    def test_ledger_leak_is_detected_by_the_harness(self):
        self.stub.store["inject_leak"] = True
        report = acceptance.run(self.config(confirm=True))
        self.assertEqual(report["outcome"], "failed")
        self.assertEqual(report["failed_step"], "ledger has no forbidden keys")


class AcceptanceHarnessUnitTests(unittest.TestCase):
    def test_forbidden_key_walker_finds_nested_credentials(self):
        keys = acceptance._walk_keys([{"entry": {"headers": {"Authorization": "x"}}}])
        self.assertIn("authorization", keys & acceptance.FORBIDDEN_LEDGER_KEYS)

    def test_cli_exit_code_for_preview_only_is_two(self):
        with patch.object(acceptance, "run", return_value={"outcome": "awaiting_confirmation", "steps": []}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(acceptance.main([]), 2)

    def test_cli_exit_code_for_failure_is_one(self):
        with patch.object(acceptance, "run", return_value={"outcome": "failed", "steps": []}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(acceptance.main(["--confirm-ingest"]), 1)


if __name__ == "__main__":
    unittest.main()
