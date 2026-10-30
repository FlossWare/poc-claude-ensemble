"""Tests for the embedding service: request validation, serialized model calls, and real-model behaviour."""
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

import embedding_service as es
from embed_lib import DIMENSIONS, MODEL_DIR


class StubEmbedder:
    """Stands in for the ONNX model in validation tests. Records how many calls overlap."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.profile = "nomic-embed-text-v1.5@fp32"
        self.info = {"backend": "stub", "device": "CPU", "precision": "FP32",
                     "variant": "model.onnx", "profile": self.profile, "fallback_reason": None}
        self.in_flight = 0
        self.max_in_flight = 0
        self._guard = threading.Lock()

    def _run(self, texts):
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.delay)
        with self._guard:
            self.in_flight -= 1
        return np.zeros((len(texts), DIMENSIONS), dtype=np.float32)

    def embed_documents(self, texts):
        return self._run(texts)

    def embed_queries(self, texts):
        return self._run(texts)


def post(port, body=None, raw=None, path="/embeddings"):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class HttpValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = es.create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self._saved = es._embedder
        self.stub = StubEmbedder()
        es._embedder = self.stub

    def tearDown(self):
        es._embedder = self._saved

    def test_model_endpoint_reports_identity_and_limits(self):
        status, body = get(self.port, "/model")
        self.assertEqual(status, 200)
        self.assertEqual((body["dimensions"], body["max_texts"]), (DIMENSIONS, es.MAX_TEXTS))
        self.assertEqual(body["modes"], ["document", "query"])
        self.assertEqual(body["profile"], self.stub.profile)
        self.assertEqual((body["backend"], body["device"], body["precision"]),
                         ("stub", "CPU", "FP32"))

    def test_health_is_503_until_model_loads(self):
        es._embedder = None
        es._load_error = "model files missing"
        try:
            status, body = get(self.port, "/health")
            self.assertEqual(status, 503)
            self.assertEqual(body["error"], "model files missing")
            self.assertEqual(post(self.port, {"texts": ["hello there"]})[0], 503)
        finally:
            es._load_error = None

    def test_successful_embedding_response_shape(self):
        status, body = post(self.port, {"texts": ["one text", "two text"], "mode": "query"})
        self.assertEqual(status, 200)
        self.assertEqual((body["dimensions"], len(body["embeddings"])), (DIMENSIONS, 2))
        self.assertEqual(len(body["embeddings"][0]), DIMENSIONS)
        self.assertEqual(body["profile"], self.stub.profile)

    def test_mode_defaults_to_document_and_unknown_mode_is_400(self):
        self.assertEqual(post(self.port, {"texts": ["a sample text"]})[0], 200)
        self.assertEqual(post(self.port, {"texts": ["a sample text"], "mode": "passage"})[0], 400)

    def test_request_validation(self):
        cases = [
            ({"texts": []}, 400),
            ({"texts": "not a list"}, 400),
            ({"texts": ["   "]}, 400),
            ({"texts": [42]}, 400),
            ({"texts": ["x" * (es.MAX_TEXT_CHARS + 1)]}, 413),
            ({"texts": ["hello world"] * (es.MAX_TEXTS + 1)}, 413),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:50]):
                self.assertEqual(post(self.port, body)[0], expected)

    def test_non_object_and_malformed_bodies_are_400(self):
        self.assertEqual(post(self.port, raw=b"[]")[0], 400)
        self.assertEqual(post(self.port, raw=b"{oops")[0], 400)
        self.assertEqual(post(self.port, raw=b"\xff")[0], 400)

    def test_empty_body_and_unknown_path(self):
        self.assertEqual(post(self.port, raw=b"")[0], 400)
        self.assertEqual(post(self.port, {"texts": ["x"]}, path="/nope")[0], 404)

    def test_oversized_body_is_refused(self):
        oversized = b"{" + b" " * es.MAX_BODY_BYTES + b"}"
        try:
            self.assertEqual(post(self.port, raw=oversized)[0], 413)
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # the server may close the socket before the client finishes sending
        self.assertEqual(get(self.port, "/health")[0], 200)

    def test_concurrent_requests_never_call_the_model_at_once(self):
        self.stub.delay = 0.05
        threads = [threading.Thread(target=post, args=(self.port, {"texts": [f"concurrent {i}"]}))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(self.stub.max_in_flight, 1)


class RealModelTests(unittest.TestCase):
    """Runs the deployed nomic model. Skipped only when the model files are absent."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(MODEL_DIR, "onnx", "model.onnx")):
            raise unittest.SkipTest(f"model not installed at {MODEL_DIR}")
        from embed_lib import Embedder
        cls.embedder = Embedder(threads=2)

    def test_vectors_are_unit_length_with_model_dimensions(self):
        vectors = self.embedder.embed_documents(["first sentence about parsers", "second about deployment"])
        self.assertEqual(vectors.shape, (2, DIMENSIONS))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)

    def test_document_and_query_prefixes_change_the_vector(self):
        doc = self.embedder.embed_documents(["how do I rotate the signing key"])[0]
        query = self.embedder.embed_queries(["how do I rotate the signing key"])[0]
        self.assertLess(float(doc @ query), 0.9999)

    def test_related_text_scores_higher_than_unrelated_text(self):
        anchor = self.embedder.embed_queries(["rotating the TLS certificate on the load balancer"])[0]
        docs = self.embedder.embed_documents([
            "Steps for renewing the load balancer TLS certificate before it expires",
            "Quarterly tax filing deadlines for contractors",
        ])
        related, unrelated = float(docs[0] @ anchor), float(docs[1] @ anchor)
        self.assertGreater(related, unrelated)

    def test_repeated_embedding_is_deterministic(self):
        first = self.embedder.embed_documents(["determinism check text"])
        second = self.embedder.embed_documents(["determinism check text"])
        np.testing.assert_allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
