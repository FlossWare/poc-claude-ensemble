"""Knowledge tests. They run against a real Index server on an ephemeral port, so the REST boundary is exercised."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

INDEX_DIR = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/shared/index")
sys.path.insert(0, INDEX_DIR)
import index_service  # noqa: E402
import knowledge_service as ks  # noqa: E402

PROFILES = {"chunking": "window-v1@1", "tokenization": "lexical-v1@1"}


def record(seq, rev, content="the body", source="rfc-1"):
    return {"source_id": source, "revision_sequence": seq, "revision_id": rev, "title": "RFC 1",
            "content": content, "provenance": {"origin": "rfc-editor", "locator": "https://example.org/rfc1"},
            "profiles": PROFILES}


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp()
        index_service.DB_PATH = os.path.join(tmp, "index.sqlite3")
        cls.index_server = index_service.create_server("127.0.0.1", 0)
        threading.Thread(target=cls.index_server.serve_forever, daemon=True).start()
        ks.INDEX_URL = f"http://127.0.0.1:{cls.index_server.server_address[1]}"
        ks.DB_PATH = os.path.join(tmp, "knowledge.sqlite3")
        cls.server = ks.create_server("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.server, cls.index_server):
            s.shutdown()
            s.server_close()

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

    def test_store_then_read_back_with_provenance(self):
        status, body = self.call("POST", "/records", record(1, "rev-1", "first body", source="rfc-fresh"))
        self.assertEqual((status, body["decision"], body["stored"]), (200, "accepted", True))
        status, body = self.call("GET", "/records/rfc-fresh")
        self.assertEqual(body["record"]["content"], "first body")
        self.assertEqual(body["record"]["provenance"]["locator"], "https://example.org/rfc1")

    def test_newer_revision_replaces_older(self):
        self.call("POST", "/records", record(2, "rev-2", "second body"))
        self.assertEqual(self.call("GET", "/records/rfc-1")[1]["record"]["revision_sequence"], 2)

    def test_older_revision_is_not_stored(self):
        status, body = self.call("POST", "/records", record(1, "rev-old", "stale body"))
        self.assertEqual((body["decision"], body["stored"]), ("superseded", False))
        self.assertEqual(self.call("GET", "/records/rfc-1")[1]["record"]["content"], "second body")

    def test_tombstone_removes_from_current_and_blocks_revisions(self):
        status, body = self.call("POST", "/records/rfc-1/tombstone", {"revision_sequence": 3})
        self.assertEqual(body["decision"], "tombstoned")
        self.assertTrue(self.call("GET", "/records/rfc-1")[1]["record"]["tombstoned"])
        status, body = self.call("POST", "/records", record(4, "rev-4", "after delete"))
        self.assertEqual((body["decision"], body["stored"]), ("tombstoned", False))

    def test_index_unreachable_is_explicit_not_a_silent_accept(self):
        original = ks.INDEX_URL
        ks.INDEX_URL = "http://127.0.0.1:1"
        try:
            status, body = self.call("POST", "/records", record(9, "rev-9"))
            self.assertEqual((status, body["error"]["code"]), (503, "index_unavailable"))
        finally:
            ks.INDEX_URL = original

    def test_validation(self):
        status, body = self.call("POST", "/records", {"source_id": "x"})
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))
        bad = record(1, "r")
        bad["provenance"] = {"origin": "x"}
        self.assertEqual(self.call("POST", "/records", bad)[0], 400)
        self.assertEqual(self.call("GET", "/records/nope")[0], 404)

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            ks.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
