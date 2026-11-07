"""Hybrid search runs its independent legs concurrently, and the shared pool does not deadlock."""

import concurrent.futures
import time
import unittest
from unittest import mock

import hybrid_service as hs

DELAY = 0.25


def fake_call(name, url, payload=None, method="POST", timeout=None):
    """Every REST call takes DELAY seconds, then returns an empty but valid response for its endpoint."""
    time.sleep(DELAY)
    if url.endswith("/embeddings"):
        return {"ok": True, "embeddings": [[0.1] * 768]}
    if url.endswith("/classify"):
        return {"ok": True, "label": "other", "confidence": 0.5}
    if url.endswith("/extract"):
        return {"ok": True, "entities": []}
    if url.endswith("/rank"):
        return {"ok": True, "results": []}
    return {"ok": True, "results": []}


class ConcurrentLegsTests(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(hs, "_call", fake_call), mock.patch.object(hs, "RERANK_ENABLED", False)]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in self.patches])

    def test_search_takes_about_the_longest_path_not_the_sum_of_legs(self):
        # Sequentially this path makes about eight calls (embedding, transcript vector, ranking, extraction,
        # knowledge vector, knowledge exact, intent, lexical). Concurrently the longest chain is about three.
        started = time.monotonic()
        result = hs._hybrid_search_uncached("zanzibar eviction", 5)
        elapsed = time.monotonic() - started
        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["intent"], {"label": "other", "confidence": 0.5})
        self.assertLess(elapsed, 5 * DELAY, f"search took {elapsed:.2f}s, expected under {5 * DELAY:.2f}s")

    def test_knowledge_endpoint_runs_its_legs_concurrently(self):
        started = time.monotonic()
        result = hs.knowledge_search("RFC 8259", 5, rerank=False)
        elapsed = time.monotonic() - started
        self.assertEqual(result["retrieval"], "knowledge-only")
        self.assertLess(elapsed, 3 * DELAY, f"knowledge search took {elapsed:.2f}s")

    def test_many_concurrent_searches_complete_without_deadlock(self):
        queries = [f"query {i}" for i in range(24)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as callers:
            futures = [callers.submit(hs._hybrid_search_uncached, q, 5) for q in queries]
            results = [f.result(timeout=60) for f in futures]
        self.assertEqual(len(results), len(queries))
        self.assertTrue(all(r["ok"] for r in results))

    def test_a_failing_leg_is_reported_without_losing_the_others(self):
        def failing_knowledge(name, url, payload=None, method="POST", timeout=None):
            if "/collections/knowledge_chunks/" in url:
                raise hs.DependencyError("knowledge-vector", "down")
            return fake_call(name, url, payload, method, timeout)
        with mock.patch.object(hs, "_call", failing_knowledge):
            result = hs._hybrid_search_uncached("zanzibar eviction", 5)
        self.assertIn("knowledge:knowledge-vector", result["degraded"])
        self.assertEqual(result["intent"], {"label": "other", "confidence": 0.5})


if __name__ == "__main__":
    unittest.main()
