"""Tests for the tokenization service: pure analyzer tests and the real HTTP server."""
import json
import threading
import unittest
import urllib.error
import urllib.request

import tokenization_service as ts


def terms(text):
    return [t["term"] for t in ts.tokenize("lexical-v1", text)]


class AnalyzerTests(unittest.TestCase):
    def test_case_folding_and_simple_words(self):
        self.assertEqual(terms("Hello WORLD"), ["hello", "world"])

    def test_short_tokens_dropped(self):
        self.assertEqual(terms("a is it the"), ["is", "it", "the"])

    def test_compound_identifier_kept_whole_and_split(self):
        tokens = ts.tokenize("lexical-v1", "see memory-service/tokenization_service.py")
        whole = {t["term"] for t in tokens if t["kind"] == "whole"}
        parts = {t["term"] for t in tokens if t["kind"] == "part"}
        self.assertIn("memory-service/tokenization_service.py", whole)
        self.assertIn("memory", parts)
        self.assertIn("service", parts)

    def test_uuid_kept_whole(self):
        uid = "12073e51-80ad-4eae-8963-e15686cde922"
        self.assertIn(uid, terms(f"session {uid} ended"))

    def test_offsets_point_into_original_text(self):
        text = "Héllo 🙂 Wörld memory-service"
        for t in ts.tokenize("lexical-v1", text):
            self.assertEqual(text[t["start"]:t["end"]].casefold(), t["term"])

    def test_unicode_letters_and_digits(self):
        self.assertIn("café", terms("café au lait 2026"))
        self.assertIn("2026", terms("café au lait 2026"))

    def test_empty_and_punctuation_only(self):
        self.assertEqual(ts.tokenize("lexical-v1", ""), [])
        self.assertEqual(ts.tokenize("lexical-v1", "!!! ... ---"), [])

    def test_deterministic(self):
        text = "same input, same output: memory-service v1"
        self.assertEqual(ts.tokenize("lexical-v1", text), ts.tokenize("lexical-v1", text))

    def test_rejections(self):
        with self.assertRaises(ts.TokenizationError) as ctx:
            ts.tokenize("unknown", "x")
        self.assertEqual((ctx.exception.code, ctx.exception.status), ("unsupported_profile", 422))
        with self.assertRaises(ts.TokenizationError) as ctx:
            ts.tokenize("lexical-v1", 5)
        self.assertEqual(ctx.exception.code, "invalid_request")
        with self.assertRaises(ts.TokenizationError) as ctx:
            ts.tokenize("lexical-v1", "x" * (ts.MAX_TEXT_CHARS + 1))
        self.assertEqual(ctx.exception.status, 413)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ts.create_server("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_health_ready_profiles(self):
        self.assertEqual(self.call("GET", "/health")[0], 200)
        self.assertTrue(self.call("GET", "/ready")[1]["ready"])
        self.assertEqual(self.call("GET", "/profiles")[1]["profiles"][0]["name"], "lexical-v1")

    def test_tokenize_round_trip(self):
        status, body = self.call("POST", "/tokenize", {"profile": "lexical-v1", "text": "Memory service"})
        self.assertEqual(status, 200)
        self.assertEqual([t["term"] for t in body["tokens"]], ["memory", "service"])

    def test_error_envelopes(self):
        status, body = self.call("POST", "/tokenize", {"profile": "nope", "text": "x"})
        self.assertEqual((status, body["error"]["code"]), (422, "unsupported_profile"))
        status, body = self.call("POST", "/tokenize", {"profile": "lexical-v1", "text": None})
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            ts.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
