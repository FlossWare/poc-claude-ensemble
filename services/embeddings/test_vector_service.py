"""Tests for the vector service: exact search over a real ChromaDB store, persistence, validation, and HTTP."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

import vector_service as vs
import vector_store
from embed_lib import DIMENSIONS

COLLECTION = "unit_test_vectors"


def basis(index):
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[index] = 1.0
    return vector.tolist()


def meta(**extra):
    return dict({"kind": "prompt", "project": "proj"}, **extra)


class VectorTestCase(unittest.TestCase):
    def setUp(self):
        self._saved_dir = vector_store.STORE_DIR
        vector_store.STORE_DIR = tempfile.mkdtemp()
        vs._cache.clear()

    def tearDown(self):
        vs._cache.clear()
        vector_store.STORE_DIR = self._saved_dir

    def upsert(self, ids, vectors, documents=None, metadatas=None, name=COLLECTION):
        body = {"ids": ids, "embeddings": vectors,
                "documents": documents or [f"doc {i}" for i in ids],
                "metadatas": metadatas or [meta() for _ in ids]}
        with vs._lock:
            vector_store.collection(name).upsert(ids=ids, embeddings=vectors,
                                                 documents=body["documents"], metadatas=body["metadatas"])
            matrix = np.asarray(vectors, dtype=np.float32)
            vs._apply_upsert(name, ids, body["documents"], matrix, body["metadatas"])


class SearchTests(VectorTestCase):
    def test_nearest_vector_ranks_first_with_exact_score(self):
        self.upsert(["a", "b", "c"], [basis(0), basis(1), basis(2)])
        entry = vs._ensure(COLLECTION)
        scores = entry["buf"][:entry["rows"]] @ np.asarray(basis(1), dtype=np.float32)
        self.assertEqual(entry["ids"][int(np.argmax(scores))], "b")
        self.assertAlmostEqual(float(scores.max()), 1.0, places=5)

    def test_query_result_shape_and_order(self):
        self.upsert(["a", "b"], [basis(0), basis(5)], documents=["alpha doc", "beta doc"])
        results = self._query(basis(0), top_k=2)
        self.assertEqual([r["id"] for r in results], ["a", "b"])
        self.assertEqual(results[0]["document"], "alpha doc")
        self.assertEqual(results[0]["metadata"]["project"], "proj")

    def test_top_k_limits_results(self):
        self.upsert([f"r{i}" for i in range(6)], [basis(i) for i in range(6)])
        self.assertEqual(len(self._query(basis(0), top_k=3)), 3)

    def test_empty_collection_returns_no_results(self):
        vector_store.collection(COLLECTION)
        self.assertEqual(self._query(basis(0), top_k=5), [])

    def test_growth_beyond_initial_capacity_keeps_every_row(self):
        count = 1100  # the initial buffer holds 1024 rows
        vectors = [basis(i % DIMENSIONS) for i in range(count)]
        for start in range(0, count, vs.MAX_UPSERT):
            end = min(start + vs.MAX_UPSERT, count)
            self.upsert([f"g{i}" for i in range(start, end)], vectors[start:end])
        self.assertEqual(vs._ensure(COLLECTION)["rows"], count)
        self.assertEqual(len(vs._ensure(COLLECTION)["ids"]), count)

    def test_reupserting_an_id_replaces_its_vector_and_document(self):
        self.upsert(["x"], [basis(0)], documents=["first"])
        self.upsert(["x"], [basis(7)], documents=["second"])
        entry = vs._ensure(COLLECTION)
        self.assertEqual(entry["rows"], 1)
        self.assertEqual(entry["documents"][0], "second")
        self.assertEqual(self._query(basis(7), top_k=1)[0]["document"], "second")

    def test_results_survive_cache_reload_from_disk(self):
        self.upsert(["p", "q"], [basis(3), basis(4)], documents=["persisted p", "persisted q"])
        vs._cache.clear()  # forces a reload from ChromaDB
        results = self._query(basis(4), top_k=1)
        self.assertEqual((results[0]["id"], results[0]["document"]), ("q", "persisted q"))

    def test_metadata_update_is_visible_in_cache_and_store(self):
        self.upsert(["m"], [basis(2)])
        with vs._lock:
            col = vector_store.collection(COLLECTION)
            col.update(ids=["m"], metadatas=[meta(project="renamed")])
            entry = vs._cache[COLLECTION]
            entry["metadatas"][entry["index"]["m"]] = meta(project="renamed")
        self.assertEqual(self._query(basis(2), top_k=1)[0]["metadata"]["project"], "renamed")
        vs._cache.clear()
        self.assertEqual(self._query(basis(2), top_k=1)[0]["metadata"]["project"], "renamed")

    def test_collection_built_with_another_model_is_refused(self):
        vector_store.client().get_or_create_collection(
            "other_model", metadata={"hnsw:space": "cosine", "model": "someone/else", "dimensions": DIMENSIONS})
        with self.assertRaises(RuntimeError):
            vector_store.collection("other_model")

    def _query(self, vector, top_k):
        entry = vs._ensure(COLLECTION)
        rows = entry["rows"]
        if rows == 0:
            return []
        scores = entry["buf"][:rows] @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [{"id": entry["ids"][i], "document": entry["documents"][i], "score": float(scores[i]),
                 "metadata": entry["metadatas"][i]} for i in order]


class ValidationTests(VectorTestCase):
    """Exercise the request checks through the same handler code path the HTTP server uses."""

    def check(self, payload, path="upsert"):
        handler = vs.VectorHandler.__new__(vs.VectorHandler)
        if path == "upsert":
            with self.assertRaises(ValueError) as ctx:
                handler._upsert(COLLECTION, payload)
        elif path == "metadata":
            with self.assertRaises(ValueError) as ctx:
                handler._metadata(COLLECTION, payload)
        else:
            with self.assertRaises(ValueError) as ctx:
                handler._query(COLLECTION, payload)
        return str(ctx.exception)

    def test_upsert_requires_equal_length_lists(self):
        self.assertIn("same length", self.check({"ids": ["a"], "embeddings": [basis(0), basis(1)],
                                                  "documents": ["d"]}))

    def test_upsert_rejects_empty_batch(self):
        self.assertIn("non-empty", self.check({"ids": [], "embeddings": [], "documents": []}))

    def test_upsert_rejects_batches_over_limit(self):
        n = vs.MAX_UPSERT + 1
        self.assertIn("at most", self.check({"ids": [str(i) for i in range(n)],
                                             "embeddings": [basis(0)] * n, "documents": ["d"] * n}))

    def test_upsert_rejects_wrong_dimensions(self):
        self.assertIn("dimensions", self.check({"ids": ["a"], "embeddings": [[1.0, 0.0]], "documents": ["d"]}))

    def test_upsert_rejects_non_finite_values(self):
        vector = basis(0)
        vector[1] = float("nan")
        self.assertIn("finite", self.check({"ids": ["a"], "embeddings": [vector], "documents": ["d"]}))

    def test_upsert_requires_list_fields(self):
        self.assertIn("must be lists", self.check({"ids": "a", "embeddings": [basis(0)], "documents": ["d"]}))

    def test_missing_or_empty_metadata_is_rejected(self):
        self.assertIn("metadatas must be a list", self.check({"ids": ["a"], "embeddings": [basis(0)],
                                                              "documents": ["d"]}))
        self.assertIn("non-empty metadata", self.check({"ids": ["a"], "embeddings": [basis(0)],
                                                        "documents": ["d"], "metadatas": [{}]}))

    def test_metadata_length_must_match(self):
        self.assertIn("equal length", self.check({"ids": ["a"], "metadatas": []}, path="metadata"))

    def test_query_rejects_bad_top_k_and_vectors(self):
        self.assertIn("top_k", self.check({"embedding": basis(0), "top_k": 0}, path="query"))
        self.assertIn("top_k", self.check({"embedding": basis(0), "top_k": vs.MAX_TOP_K + 1}, path="query"))
        self.assertIn("embedding must be a list", self.check({"embedding": "abc"}, path="query"))
        self.assertIn("dimensions", self.check({"embedding": [0.1] * 3}, path="query"))


class HttpTests(VectorTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = vs.create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body=None, method="POST"):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                         headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_is_ok_with_chromadb_backend(self):
        status, body = self.call("/health", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "chromadb")

    def test_upsert_then_query_over_http(self):
        status, body = self.call(f"/collections/{COLLECTION}/upsert", {
            "ids": ["h1", "h2"], "embeddings": [basis(10), basis(11)],
            "documents": ["http one", "http two"], "metadatas": [meta(), meta()]})
        self.assertEqual((status, body["upserted"]), (200, 2))
        status, body = self.call(f"/collections/{COLLECTION}/query", {"embedding": basis(11), "top_k": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["id"], "h2")

    def test_existing_reports_only_stored_ids(self):
        self.call(f"/collections/{COLLECTION}/upsert", {
            "ids": ["e1"], "embeddings": [basis(20)], "documents": ["exists"], "metadatas": [meta()]})
        status, body = self.call(f"/collections/{COLLECTION}/existing", {"ids": ["e1", "missing"]})
        self.assertEqual((status, body["found"]), (200, ["e1"]))

    def test_metadata_endpoint_updates_only_present_ids(self):
        self.call(f"/collections/{COLLECTION}/upsert", {
            "ids": ["mu"], "embeddings": [basis(30)], "documents": ["meta doc"], "metadatas": [meta()]})
        status, body = self.call(f"/collections/{COLLECTION}/metadata",
                                 {"ids": ["mu", "absent"], "metadatas": [meta(project="new"), meta()]})
        self.assertEqual((status, body["updated"]), (200, 1))

    def test_upsert_without_metadata_is_400_not_500(self):
        status, body = self.call(f"/collections/{COLLECTION}/upsert", {
            "ids": ["nometa"], "embeddings": [basis(40)], "documents": ["no metadata"]})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_bad_requests_are_400_and_unknown_paths_404(self):
        self.assertEqual(self.call(f"/collections/{COLLECTION}/upsert", {"ids": "x"})[0], 400)
        self.assertEqual(self.call(f"/collections/{COLLECTION}/query", {"embedding": basis(0), "top_k": 0})[0], 400)
        self.assertEqual(self.call("/nowhere", {})[0], 404)
        self.assertEqual(self.call(f"/collections/{COLLECTION}/unknown", {})[0], 404)

    def test_collection_mismatch_is_409(self):
        vector_store.client().get_or_create_collection(
            "mismatch_http", metadata={"hnsw:space": "cosine", "model": "other", "dimensions": DIMENSIONS})
        status, body = self.call("/collections/mismatch_http/query", {"embedding": basis(0), "top_k": 1})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "collection_mismatch")

    def test_non_loopback_bind_is_refused(self):
        with self.assertRaises(ValueError):
            vs.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
