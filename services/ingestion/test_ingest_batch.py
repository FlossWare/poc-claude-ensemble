"""Tests for cross-item batched ingestion (index_texts_batch + /ingest/batch)."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import ingestion_service as svc
from ingestion.datasets import indexing


class RecordingPost:
    """Fake transport recording calls; serves canned downstream responses."""

    def __init__(self, chunks_per_source):
        self.calls = []
        self._chunks_per_source = chunks_per_source

    def __call__(self, base, path, payload, timeout=60):
        self.calls.append((base, path, payload))
        if path == "/chunk/batch":
            results = []
            for item in payload["items"]:
                source_id = item["source_id"]
                n = self._chunks_per_source.get(source_id, 2)
                results.append({"ok": True, "chunks": [
                    {"chunk_id": f"{source_id}#c{i}", "index": i, "text": f"chunk {i} of {source_id}"}
                    for i in range(n)
                ]})
            return {"ok": True, "results": results}
        if path == "/embeddings":
            texts = payload["texts"]
            return {"embeddings": [[0.1, 0.2] for _ in texts]}
        return {"ok": True}


def _sources(count, prefix="s"):
    return [
        {"source_id": f"{prefix}{i}", "title": f"t{i}", "origin": "inline-text",
         "locator": f"text://sha256/{prefix}{i}", "text": f"text number {i} with content"}
        for i in range(count)
    ]


class IndexTextsBatchTests(unittest.TestCase):
    def test_per_item_results_in_order(self):
        post = RecordingPost({"s0": 1, "s1": 3, "s2": 2})
        endpoints = indexing.Endpoints(
            knowledge="http://k", chunking="http://c", embedding="http://e",
            lexical="http://l", vector="http://v", index="http://i")
        results = indexing.index_texts_batch(
            endpoints=endpoints, sources=_sources(3), post=post)
        self.assertEqual(len(results), 3)
        self.assertEqual([r["chunks"] for r in results], [1, 3, 2])
        for r in results:
            self.assertEqual(r["vectors"], r["chunks"])

    def test_embedding_batches_span_items(self):
        """Chunks from multiple sources share embedding requests."""
        post = RecordingPost({"s0": 2, "s1": 2, "s2": 2})
        endpoints = indexing.Endpoints(
            knowledge="http://k", chunking="http://c", embedding="http://e",
            lexical="http://l", vector="http://v", index="http://i")
        indexing.index_texts_batch(endpoints=endpoints, sources=_sources(3), post=post)
        embed_calls = [c for c in post.calls if c[1] == "/embeddings"]
        # 6 chunks total must be embedded in ONE cross-item request
        # (EMBED_BATCH is 64), not three per-item requests.
        self.assertEqual(len(embed_calls), 1)
        self.assertEqual(len(embed_calls[0][2]["texts"]), 6)

    def test_writes_aggregated_across_batch(self):
        """Lexical + vector writes are ONE call for the whole batch."""
        post = RecordingPost({})
        endpoints = indexing.Endpoints(
            knowledge="http://k", chunking="http://c", embedding="http://e",
            lexical="http://l", vector="http://v", index="http://i")
        indexing.index_texts_batch(endpoints=endpoints, sources=_sources(3), post=post)
        lexical_calls = [c for c in post.calls if c[1] == "/knowledge/documents"]
        upsert_calls = [c for c in post.calls if c[1].endswith("/upsert")]
        self.assertEqual(len(lexical_calls), 1)
        self.assertEqual(len(upsert_calls), 1)
        # 3 sources x 2 chunks each = 6 rows in the single lexical call.
        self.assertEqual(len(lexical_calls[0][2]["chunks"]), 6)
        self.assertEqual(len(upsert_calls[0][2]["ids"]), 6)
        # Superseded cleanup is one batched call covering every source.
        superseded_calls = [c for c in post.calls if c[1].endswith("/delete-superseded-batch")]
        self.assertEqual(len(superseded_calls), 1)
        self.assertEqual(len(superseded_calls[0][2]["deletions"]), 3)

    def test_invalid_chunk_response_names_source(self):
        class BadChunkPost:
            def __call__(self, base, path, payload, timeout=60):
                if path == "/chunk/batch":
                    # per-item failure: the source must be named
                    return {"ok": True, "results": [
                        {"ok": False, "error": "chunk profile rejected"}]}
                return {"ok": True}
        endpoints = indexing.Endpoints(
            knowledge="http://k", chunking="http://c", embedding="http://e",
            lexical="http://l", vector="http://v", index="http://i")
        with self.assertRaises(indexing.IndexingError) as ctx:
            indexing.index_texts_batch(
                endpoints=endpoints, sources=_sources(1), post=BadChunkPost())
        self.assertIn("s0", str(ctx.exception))

    def test_invalid_vector_count_rejected(self):
        class BadEmbedPost:
            def __call__(self, base, path, payload, timeout=60):
                if path == "/chunk":
                    return {"chunks": [{"chunk_id": "c0", "index": 0, "text": "x"}]}
                if path == "/embeddings":
                    return {"embeddings": [[0.1]] * (len(payload["texts"]) + 1)}
                return {"ok": True}
        endpoints = indexing.Endpoints(
            knowledge="http://k", chunking="http://c", embedding="http://e",
            lexical="http://l", vector="http://v", index="http://i")
        with self.assertRaises(indexing.IndexingError):
            indexing.index_texts_batch(
                endpoints=endpoints, sources=_sources(1), post=BadEmbedPost())


class BatchEndpointTests(unittest.TestCase):
    """HTTP-level tests against the real handler with stubbed downstreams."""

    @classmethod
    def setUpClass(cls):
        cls.server = svc.create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_validation(self):
        status, body = self._post("/ingest/batch", {"items": "nope"})
        self.assertEqual(status, 400)
        status, body = self._post("/ingest/batch", {"items": []})
        self.assertEqual(status, 400)
        status, body = self._post("/ingest/batch", {"items": [{"text": "x"}] * 51})
        self.assertEqual(status, 413)

    def test_invalid_item_reported_not_fatal(self):
        status, body = self._post("/ingest/batch", {
            "items": [
                {"text": "", "title": "empty text is invalid"},   # invalid: empty
                {"no_text": True},                                 # invalid: no fields
            ]})
        # Both fail resolve_source validation -> reported per item, ok envelope.
        self.assertEqual(status, 200)
        self.assertEqual(body["invalid"], 2)
        self.assertEqual(body["completed"], 0)

    def test_text_budget(self):
        # Point downstreams at an unreachable port: the assertion is only
        # "inside the text budget" (not 413), and with within-batch
        # dedupe these 50 identical items would otherwise do REAL
        # indexing work against live services (slow, environment-
        # dependent, and contends with whatever else is ingesting).
        url_keys = ("KNOWLEDGE_URL", "CHUNKING_URL", "EMBEDDING_URL",
                    "LEXICAL_URL", "VECTOR_URL")
        saved = {k: getattr(svc, k) for k in url_keys}
        for k in url_keys:
            setattr(svc, k, "http://127.0.0.1:9")
        try:
            status, body = self._post("/ingest/batch", {
                "items": [{"text": "a" * 100, "title": "t"}] * 50})
        finally:
            for k, v in saved.items():
                setattr(svc, k, v)
        # Inside the character budget; downstreams unreachable -> 503, not 413.
        self.assertNotEqual(status, 413)

    def test_duplicate_texts_index_each_unique_source_once(self):
        """Two identical texts in one batch share a source_id and vector
        ids; the vector store rejects duplicate ids in a single upsert
        (previously an opaque downstream 500 for the whole batch). The
        service must index each unique source once and complete both
        items."""
        calls = []

        class StubIndexing:
            Endpoints = indexing.Endpoints
            IndexingError = indexing.IndexingError
            # the service's _post_json adapter routes ledger/event writes
            # through these; keep the test fully offline
            post_json = staticmethod(lambda base, path, payload, timeout=60: {"ok": True})

            @staticmethod
            def submit_records_batch(knowledge_base, *, records, timeout=120, post=None):
                calls.append(("knowledge", len(records)))
                return [{"decision": "accepted", "source_id": r["source_id"],
                         "stored": True, "content_sha256": "x"} for r in records]

            @staticmethod
            def index_texts_batch(*, endpoints, sources, profiles, chunk_profile,
                                  media_type="text/plain", post=None, timeout=300):
                calls.append(("index", [s["source_id"] for s in sources]))
                return [{"chunks": 1, "vectors": 1, "content_sha256": "x"}
                        for _ in sources]

        real_indexing = svc.indexing
        svc.indexing = StubIndexing()
        try:
            body = svc.ingest_batch({"items": [
                {"text": "identical duplicate text", "title": "a"},
                {"text": "identical duplicate text", "title": "b"},
                {"text": "a different text entirely", "title": "c"},
            ]})
        finally:
            svc.indexing = real_indexing
        self.assertTrue(body["ok"])
        self.assertEqual(body["completed"], 3)
        # knowledge still gets one revision per item...
        self.assertEqual(calls[0], ("knowledge", 3))
        # ...but indexing runs once per UNIQUE source_id (2, not 3)
        self.assertEqual(calls[1][0], "index")
        self.assertEqual(len(calls[1][1]), 2)
        # and no duplicate vector ids reach the aggregate write
        for result in body["results"]:
            self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
