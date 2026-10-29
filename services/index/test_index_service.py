"""Tests for the index manifest: revision decisions, tombstones, persistence, and the HTTP boundary."""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import index_service as ix

PROFILES = {"chunking": "window-v1@1", "tokenization": "lexical-v1@1"}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        ix.DB_PATH = os.path.join(self.dir, "manifest.sqlite3")

    def test_first_revision_accepted(self):
        self.assertEqual(ix.record_revision("doc-1", 1, "rev-a", PROFILES)["decision"], "accepted")

    def test_newer_sequence_accepted_and_becomes_high_water(self):
        ix.record_revision("doc-1", 1, "rev-a", PROFILES)
        result = ix.record_revision("doc-1", 2, "rev-b", PROFILES)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual((result["source"]["high_sequence"], result["source"]["revision_id"]), (2, "rev-b"))

    def test_older_sequence_superseded_and_state_unchanged(self):
        ix.record_revision("doc-1", 5, "rev-e", PROFILES)
        result = ix.record_revision("doc-1", 3, "rev-c", PROFILES)
        self.assertEqual(result["decision"], "superseded")
        self.assertEqual(result["source"]["high_sequence"], 5)

    def test_same_sequence_same_revision_is_duplicate(self):
        ix.record_revision("doc-1", 2, "rev-b", PROFILES)
        self.assertEqual(ix.record_revision("doc-1", 2, "rev-b", PROFILES)["decision"], "duplicate")

    def test_same_sequence_different_revision_is_conflict(self):
        ix.record_revision("doc-1", 2, "rev-b", PROFILES)
        self.assertEqual(ix.record_revision("doc-1", 2, "rev-x", PROFILES)["decision"], "conflict")

    def test_profile_change_at_same_sequence_is_conflict(self):
        ix.record_revision("doc-1", 2, "rev-b", PROFILES)
        other = dict(PROFILES, chunking="window-v2@2")
        self.assertEqual(ix.record_revision("doc-1", 2, "rev-b", other)["decision"], "conflict")

    def test_tombstone_blocks_later_revisions(self):
        ix.record_revision("doc-1", 2, "rev-b", PROFILES)
        self.assertEqual(ix.record_tombstone("doc-1", 3)["decision"], "tombstoned")
        self.assertEqual(ix.record_revision("doc-1", 4, "rev-d", PROFILES)["decision"], "tombstoned")

    def test_stale_tombstone_superseded(self):
        ix.record_revision("doc-1", 5, "rev-e", PROFILES)
        self.assertEqual(ix.record_tombstone("doc-1", 2)["decision"], "superseded")
        self.assertFalse(ix.get_source("doc-1")["tombstoned"])

    def test_state_survives_reopen(self):
        ix.record_revision("doc-1", 7, "rev-g", PROFILES)
        self.assertEqual(ix.get_source("doc-1")["high_sequence"], 7)
        self.assertEqual(ix.record_revision("doc-1", 6, "rev-f", PROFILES)["decision"], "superseded")

    def test_manifest_listing_and_unknown_source(self):
        ix.record_revision("a", 1, "r", PROFILES)
        ix.record_revision("b", 1, "r", PROFILES)
        self.assertEqual([s["source_id"] for s in ix.list_sources(10)], ["a", "b"])
        self.assertIsNone(ix.get_source("missing"))

    def test_validation_rejections(self):
        for args in [("", 1, "r", PROFILES), ("s", 0, "r", PROFILES), ("s", True, "r", PROFILES),
                     ("s", 1, "", PROFILES), ("s", 1, "r", {})]:
            with self.assertRaises(ix.IndexError_) as ctx:
                ix.record_revision(*args)
            self.assertEqual(ctx.exception.code, "invalid_request")


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ix.DB_PATH = os.path.join(tempfile.mkdtemp(), "http.sqlite3")
        cls.server = ix.create_server("127.0.0.1", 0)
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

    def test_revision_then_read_back(self):
        status, body = self.call("POST", "/revisions", {"source_id": "h1", "revision_sequence": 1,
                                                       "revision_id": "r1", "profiles": PROFILES})
        self.assertEqual((status, body["decision"]), (200, "accepted"))
        status, body = self.call("GET", "/sources/h1")
        self.assertEqual(body["source"]["revision_id"], "r1")

    def test_tombstone_over_http(self):
        self.call("POST", "/revisions", {"source_id": "h2", "revision_sequence": 1, "revision_id": "r", "profiles": PROFILES})
        status, body = self.call("POST", "/tombstones", {"source_id": "h2", "revision_sequence": 2})
        self.assertEqual(body["decision"], "tombstoned")

    def test_error_envelopes(self):
        status, body = self.call("POST", "/revisions", {"source_id": "", "revision_sequence": 1,
                                                       "revision_id": "r", "profiles": PROFILES})
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))
        self.assertEqual(self.call("GET", "/sources/nope")[0], 404)
        self.assertEqual(self.call("GET", "/manifest?limit=0")[0], 400)
        self.assertTrue(self.call("GET", "/ready")[1]["ready"])

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            ix.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
