import os
import re
import shutil
import tempfile
import unittest

import knowledge_lexical as kl


def analyze(text):
    # Stand-in for the Tokenization service: each compound identifier and its parts are both terms.
    terms = []
    for token in re.findall(r"[\w\-._:/]+", text.lower()):
        terms.append(token)
        terms.extend(part for part in re.split(r"[\-._:/]+", token) if part and part != token)
    return terms


def chunk(chunk_id, text, url="https://example.test/doc", title="Doc"):
    return {"chunk_id": chunk_id, "vector_id": "v-" + chunk_id, "url": url, "source": "rfc",
            "title": title, "text": text}


class KnowledgeLexicalTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db = os.path.join(self.dir, "knowledge-lexical.sqlite3")

    def test_ingest_inserts_then_skips_existing_chunks(self):
        first = kl.ingest([chunk("c1", "RFC-8259 JSON text format")], analyze, path=self.db)
        again = kl.ingest([chunk("c1", "RFC-8259 JSON text format")], analyze, path=self.db)
        self.assertEqual(first, {"inserted": 1, "already_present": 0})
        self.assertEqual(again, {"inserted": 0, "already_present": 1})
        self.assertEqual(kl.count(path=self.db), 1)

    def test_exact_identifier_is_found_and_unrelated_text_is_not(self):
        kl.ingest([chunk("c1", "RFC-8259 defines the JSON text format"),
                   chunk("c2", "RFC-8285 header extensions for RTP"),
                   chunk("c3", "CBOR web token claims")], analyze, path=self.db)
        hits = [h["chunk_id"] for h in kl.search("RFC-8259", 5, analyze, path=self.db)]
        # Terms are OR-combined, so other RFC chunks can match on the shared "rfc" term.
        # The exact identifier must still rank first, and unrelated text must not appear.
        self.assertEqual(hits[0], "c1")
        self.assertNotIn("c3", hits)

    def test_results_carry_vector_id_for_fusion(self):
        kl.ingest([chunk("c1", "RFC-7519 JSON web token")], analyze, path=self.db)
        hits = kl.search("7519", 3, analyze, path=self.db)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["key"], "v-c1")
        self.assertEqual(hits[0]["vector_id"], "v-c1")

    def test_top_k_limits_results(self):
        kl.ingest([chunk(f"c{i}", f"shared term number {i}") for i in range(6)], analyze, path=self.db)
        self.assertEqual(len(kl.search("shared", 3, analyze, path=self.db)), 3)

    def test_empty_query_returns_nothing(self):
        kl.ingest([chunk("c1", "anything")], analyze, path=self.db)
        self.assertEqual(kl.search("   ", 5, lambda t: [], path=self.db), [])

    def test_quotes_in_query_do_not_break_matching(self):
        kl.ingest([chunk("c1", 'say "hello" world')], analyze, path=self.db)
        self.assertEqual(kl.search('"hello"', 5, analyze, path=self.db)[0]["chunk_id"], "c1")

    def test_invalid_chunks_are_rejected_and_nothing_is_written(self):
        bad_batches = [
            [],
            [chunk("c1", "")],
            [chunk("c1", "ok"), {"chunk_id": "c2"}],
            [{**chunk("c1", "ok"), "url": 5}],
            [{**chunk("c1", "ok"), "chunk_id": "  "}],
        ]
        for batch in bad_batches:
            with self.assertRaises(kl.ChunkValidationError):
                kl.ingest(batch, analyze, path=self.db)
        self.assertEqual(kl.count(path=self.db), 0)

    def test_oversized_text_is_rejected(self):
        with self.assertRaises(kl.ChunkValidationError):
            kl.ingest([chunk("c1", "x" * (kl.MAX_TEXT_CHARS + 1))], analyze, path=self.db)

    def test_too_many_chunks_in_one_request_is_rejected(self):
        batch = [chunk(f"c{i}", "x") for i in range(kl.MAX_CHUNKS_PER_REQUEST + 1)]
        with self.assertRaises(kl.ChunkValidationError):
            kl.ingest(batch, analyze, path=self.db)

    def test_tokenizer_settings_match_the_transcript_index(self):
        import lexical_service
        self.assertEqual(kl.FTS_TOKENIZE, lexical_service.FTS_TOKENIZE)


class KnowledgeRoutesTests(unittest.TestCase):
    """The lexical service's /knowledge routes over real HTTP, with the stub analyzer and a temp database."""

    def setUp(self):
        import json
        import threading
        from unittest import mock
        import lexical_service as lx
        self.json = json
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        db = os.path.join(self.dir, "knowledge-lexical.sqlite3")
        patches = [mock.patch.object(lx, "KNOWLEDGE_DB_PATH", db), mock.patch.object(lx, "analyze", analyze)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.server = lx.create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, body):
        import urllib.error
        import urllib.request
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                         data=self.json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, self.json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, self.json.loads(exc.read())

    def test_ingest_then_search_over_http(self):
        status, body = self.post("/knowledge/documents", {"chunks": [chunk("c1", "RFC-8259 JSON text format")]})
        self.assertEqual((status, body["inserted"]), (200, 1))
        status, body = self.post("/knowledge/search", {"query": "RFC-8259", "top_k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["chunk_id"], "c1")

    def test_invalid_chunks_return_400(self):
        status, body = self.post("/knowledge/documents", {"chunks": [{"chunk_id": "x"}]})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])

    def test_search_validation_matches_transcript_search(self):
        status, _ = self.post("/knowledge/search", {"query": "  ", "top_k": 3})
        self.assertEqual(status, 400)
        status, _ = self.post("/knowledge/search", {"query": "x", "top_k": 0})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
