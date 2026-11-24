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
        if path == "/chunk":
            source_id = payload["source_id"]
            n = self._chunks_per_source.get(source_id, 2)
            return {"chunks": [
                {"chunk_id": f"{source_id}#c{i}", "index": i, "text": f"chunk {i} of {source_id}"}
                for i in range(n)
            ]}
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
        # Superseded cleanup stays per source (per-locator semantics).
        self.assertEqual(
            len([c for c in post.calls if c[1].endswith("/delete-superseded")]), 3)

    def test_invalid_chunk_response_names_source(self):
        class BadChunkPost:
            def __call__(self, base, path, payload, timeout=60):
                if path == "/chunk":
                    return {"chunks": "not-a-list"}
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
        status, body = self._post("/ingest/batch", {
            "items": [{"text": "a" * 100, "title": "t"}] * 50})
        # 50 * 100 = 5000 chars, inside budget; fails downstream instead.
        self.assertNotEqual(status, 413)


if __name__ == "__main__":
    unittest.main()
