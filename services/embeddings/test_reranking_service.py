"""Tests for the reranking service: request validation, serialized scoring, and real cross-encoder behaviour."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

import reranking_service as rs


class StubModel:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self._guard = threading.Lock()

    def predict(self, pairs):
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.delay)
        with self._guard:
            self.in_flight -= 1
        return [0.5 for _ in pairs]


def post(port, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/rerank", data=data, method="POST",
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
        cls.server = rs.create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self._saved = rs._model
        self.stub = StubModel()
        rs._model = self.stub

    def tearDown(self):
        rs._model = self._saved

    def test_scores_align_with_input_order(self):
        status, body = post(self.port, {"query": "tls renewal", "documents": ["a", "b", "c"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["scores"]), 3)
        self.assertTrue(all(isinstance(s, float) for s in body["scores"]))

    def test_health_is_503_when_model_missing(self):
        rs._model = None
        rs._load_error = "model directory missing"
        try:
            status, body = get(self.port, "/health")
            self.assertEqual(status, 503)
            self.assertEqual(body["error"], "model directory missing")
            self.assertEqual(post(self.port, {"query": "q", "documents": ["d"]})[0], 503)
        finally:
            rs._load_error = None

    def test_request_validation(self):
        cases = [
            ({"query": "", "documents": ["d"]}, 400),
            ({"query": "   ", "documents": ["d"]}, 400),
            ({"query": "q" * (rs.MAX_QUERY_CHARS + 1), "documents": ["d"]}, 400),
            ({"query": "q", "documents": []}, 400),
            ({"query": "q", "documents": "d"}, 400),
            ({"query": "q", "documents": ["d"] * (rs.MAX_DOCUMENTS + 1)}, 400),
            ({"query": "q", "documents": [123]}, 400),
            ({"query": "q", "documents": ["d" * (rs.MAX_DOCUMENT_CHARS + 1)]}, 400),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:50]):
                self.assertEqual(post(self.port, body)[0], expected)

    def test_malformed_and_non_object_bodies_are_400(self):
        self.assertEqual(post(self.port, raw=b"{bad")[0], 400)
        self.assertEqual(post(self.port, raw=b"[1]")[0], 400)
        self.assertEqual(post(self.port, raw=b"")[0], 400)

    def test_unknown_path_is_404(self):
        self.assertEqual(get(self.port, "/nope")[0], 404)

    def test_concurrent_requests_never_score_at_once(self):
        self.stub.delay = 0.05
        threads = [threading.Thread(target=post, args=(self.port, {"query": f"q{i}", "documents": ["d"]}))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(self.stub.max_in_flight, 1)

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            rs.create_server("0.0.0.0", 0)


class RealModelTests(unittest.TestCase):
    """Runs the deployed cross-encoder. Skipped only when the model cannot load."""

    @classmethod
    def setUpClass(cls):
        rs._load()
        if rs._model is None:
            raise unittest.SkipTest(f"reranker not loadable: {rs._load_error}")
        cls.model = rs._model

    def test_relevant_document_outscores_unrelated_document(self):
        scores = self.model.predict([
            ("how do I renew the TLS certificate", "Renew the TLS certificate on the load balancer before expiry."),
            ("how do I renew the TLS certificate", "Quarterly tax filing deadlines for contractors."),
        ])
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_scores_are_stable_for_identical_input(self):
        pair = [("parser error", "The parser rejects trailing commas in JSON.")]
        first = float(self.model.predict(pair)[0])
        second = float(self.model.predict(pair)[0])
        self.assertAlmostEqual(first, second, places=5)


if __name__ == "__main__":
    unittest.main()
