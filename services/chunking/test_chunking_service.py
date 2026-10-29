"""Tests for the chunking service: unit (pure function) and HTTP (the real server on an ephemeral port)."""
import json
import threading
import unittest
import urllib.error
import urllib.request

import chunking_service as cs


class UnitTests(unittest.TestCase):
    def test_same_input_gives_same_chunks_and_ids(self):
        text = "alpha beta gamma " * 300
        self.assertEqual(cs.chunk("window-v1", "src-1", text, "text/plain"),
                         cs.chunk("window-v1", "src-1", text, "text/plain"))

    def test_chunk_id_depends_on_profile_and_source(self):
        text = "one two three " * 200
        a = cs.chunk("window-v1", "src-1", text, "text/plain")[0]["chunk_id"]
        b = cs.chunk("window-v1", "src-2", text, "text/plain")[0]["chunk_id"]
        c = cs.chunk("message-v1", "src-1", text, "text/plain")[0]["chunk_id"]
        self.assertEqual(len({a, b, c}), 3)

    def test_windows_cover_text_with_overlap_and_exact_offsets(self):
        text = "".join(chr(0x61 + (i % 26)) for i in range(5000))
        records = cs.chunk("window-v1", "s", text, "text/plain")
        self.assertEqual(records[0]["start"], 0)
        self.assertEqual(records[-1]["end"], len(text))
        for r in records:
            self.assertEqual(text[r["start"]:r["end"]], r["text"])
            self.assertLessEqual(len(r["text"]), 1800)
        for prev, nxt in zip(records, records[1:]):
            self.assertEqual(prev["end"] - nxt["start"], 200)

    def test_unicode_offsets_are_code_point_exact(self):
        text = "héllo 🙂 wörld " * 200
        for r in cs.chunk("window-v1", "s", text, "text/plain"):
            self.assertEqual(text[r["start"]:r["end"]], r["text"])

    def test_empty_and_short_input(self):
        self.assertEqual(cs.chunk("window-v1", "s", "", "text/plain"), [])
        records = cs.chunk("window-v1", "s", "short", "text/plain")
        self.assertEqual([(r["start"], r["end"]) for r in records], [(0, 5)])

    def test_markdown_headings_give_header_provenance(self):
        text = "intro line here\n# First\nbody one\n## Second\nbody two\n"
        records = cs.chunk("markdown-v1", "doc", text, "text/markdown")
        self.assertEqual([r["header"] for r in records], ["", "First", "Second"])
        self.assertTrue(records[1]["text"].startswith("# First"))

    def test_markdown_treats_hash_lines_as_plain_text_for_plain_media_type(self):
        text = "# not a heading for plain text\nbody\n"
        records = cs.chunk("markdown-v1", "doc", text, "text/plain")
        self.assertEqual([r["header"] for r in records], [""])

    def test_oversized_section_is_windowed_with_its_header(self):
        text = "# Big\n" + ("word " * 2000)
        records = cs.chunk("markdown-v1", "doc", text, "text/markdown")
        self.assertGreater(len(records), 1)
        self.assertTrue(all(r["header"] == "Big" for r in records))
        self.assertTrue(all(len(r["text"]) <= 2000 for r in records))

    def test_message_profile_is_one_chunk_and_clipped(self):
        records = cs.chunk("message-v1", "m", "x" * 25000, "text/plain")
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["text"]), 20000)

    def test_rejections(self):
        with self.assertRaises(cs.ChunkingError) as ctx:
            cs.chunk("no-such-profile", "s", "t", "text/plain")
        self.assertEqual((ctx.exception.code, ctx.exception.status), ("unsupported_profile", 422))
        with self.assertRaises(cs.ChunkingError) as ctx:
            cs.chunk("window-v1", "", "t", "text/plain")
        self.assertEqual(ctx.exception.code, "invalid_request")
        with self.assertRaises(cs.ChunkingError) as ctx:
            cs.chunk("window-v1", "s", "t", "application/pdf")
        self.assertEqual(ctx.exception.status, 415)
        with self.assertRaises(cs.ChunkingError) as ctx:
            cs.chunk("window-v1", "s", "t" * (cs.MAX_TEXT_CHARS + 1), "text/plain")
        self.assertEqual(ctx.exception.status, 413)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = cs.create_server("127.0.0.1", 0)
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
        status, body = self.call("GET", "/ready")
        self.assertTrue(body["ready"])
        status, body = self.call("GET", "/profiles")
        self.assertEqual({p["name"] for p in body["profiles"]}, set(cs.PROFILES))

    def test_chunk_endpoint_round_trip(self):
        status, body = self.call("POST", "/chunk", {"profile": "markdown-v1", "source_id": "doc-1",
                                                    "media_type": "text/markdown", "text": "# A\nbody\n"})
        self.assertEqual(status, 200)
        self.assertEqual(body["profile"], {"name": "markdown-v1", "version": "1"})
        self.assertEqual(body["chunks"][0]["header"], "A")

    def test_error_envelopes(self):
        status, body = self.call("POST", "/chunk", {"profile": "bogus", "source_id": "s", "text": "t"})
        self.assertEqual((status, body["error"]["code"]), (422, "unsupported_profile"))
        status, body = self.call("POST", "/chunk", {"profile": "window-v1", "source_id": "s", "text": 5})
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))
        status, body = self.call("GET", "/nope")
        self.assertEqual(status, 404)

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            cs.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
