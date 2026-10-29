"""Integration tests for the knowledge ingestion client, against real Index, Knowledge, Chunking,
embedding, and vector services running in-process on ephemeral loopback ports.

Only the scrape ledger and raw records are fixtures; everything the client calls is real.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request

_SERVICES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _name in ("index", "knowledge", "chunking"):
    sys.path.insert(0, os.path.join(_SERVICES, _name))

import build_knowledge_chunks as bk  # noqa: E402
import chunking_service as cs  # noqa: E402
import embedding_service as es  # noqa: E402
import index_service as ix  # noqa: E402
import knowledge_service as kn  # noqa: E402
import vector_service as vs  # noqa: E402
import vector_store  # noqa: E402
from embed_lib import MODEL_DIR  # noqa: E402

PROFILES = bk.PROFILES


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def body_for(paragraphs):
    return ("\n\n".join(paragraphs)).strip()


PARAGRAPHS = [
    "The Transport Layer Security protocol secures communication between a client and a server.",
    "Implementations must reject certificates whose validity period has expired or is malformed.",
    "Extensions negotiate optional behaviour, and unknown extensions must be ignored by the receiver.",
    "Session resumption allows a previously established secret to be reused for a new connection.",
    "Error handling follows the alert protocol, and a fatal alert terminates the connection at once.",
]


class KnowledgeIngestTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(MODEL_DIR, "onnx", "model.onnx")):
            raise unittest.SkipTest(f"embedding model not installed at {MODEL_DIR}")
        from embed_lib import Embedder
        es._embedder = Embedder(threads=2)
        cls.index_server = serve(ix.create_server("127.0.0.1", 0))
        cls.chunking_server = serve(cs.create_server("127.0.0.1", 0))
        cls.embedding_server = serve(es.create_server("127.0.0.1", 0))

    @classmethod
    def tearDownClass(cls):
        for server in (cls.index_server, cls.chunking_server, cls.embedding_server):
            server.shutdown()
            server.server_close()
        es._embedder = None

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.raw = os.path.join(self.root, "raw")
        os.makedirs(self.raw)
        self.ledger = os.path.join(self.root, "ledger.jsonl")
        ix.DB_PATH = os.path.join(self.root, "index.sqlite3")
        kn.DB_PATH = os.path.join(self.root, "knowledge.sqlite3")
        vector_store.STORE_DIR = os.path.join(self.root, "vectors")
        vs._cache.clear()
        self.index_url = f"http://127.0.0.1:{self.index_server.server_address[1]}"
        kn.INDEX_URL = self.index_url
        self.knowledge_server = serve(kn.create_server("127.0.0.1", 0))
        self.vector_server = serve(vs.create_server("127.0.0.1", 0))
        self._saved = {k: getattr(bk, k) for k in ("LEDGER", "RAW", "LOG", "EMBEDDING_URL", "VECTOR_URL",
                                                   "CHUNKING_URL", "KNOWLEDGE_URL")}
        bk.LEDGER, bk.RAW = self.ledger, self.raw
        bk.LOG = os.path.join(self.root, "build.log")
        bk.EMBEDDING_URL = f"http://127.0.0.1:{self.embedding_server.server_address[1]}"
        bk.VECTOR_URL = f"http://127.0.0.1:{self.vector_server.server_address[1]}"
        bk.CHUNKING_URL = f"http://127.0.0.1:{self.chunking_server.server_address[1]}"
        bk.KNOWLEDGE_URL = f"http://127.0.0.1:{self.knowledge_server.server_address[1]}"

    def tearDown(self):
        for server in (self.knowledge_server, self.vector_server):
            server.shutdown()
            server.server_close()
        for key, value in self._saved.items():
            setattr(bk, key, value)
        shutil.rmtree(self.root, ignore_errors=True)

    def add_document(self, name, source, url, content, fetched_at="2026-07-11T19:35:07.000000", title=None):
        record_path = f"{source}/{name}.json"
        os.makedirs(os.path.join(self.raw, source), exist_ok=True)
        with open(os.path.join(self.raw, record_path), "w", encoding="utf-8") as handle:
            json.dump({"title": title or f"Doc {name}", "content": content}, handle)
        with open(self.ledger, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"url": url, "source": source, "fetched_at": fetched_at,
                                     "record": record_path}) + "\n")

    def vector_count(self):
        return vs._ensure(bk.COLLECTION)["rows"]

    def expected_chunks(self, url, content):
        return cs.chunk("window-v1", url, content, "text/plain")


class SelectionTests(KnowledgeIngestTestCase):
    def test_only_selected_sources_are_ingested(self):
        self.add_document("r1", "rfc", "https://rfc.example/1", body_for(PARAGRAPHS))
        self.add_document("p1", "patents", "https://patents.example/1", body_for(PARAGRAPHS))
        self.add_document("n1", "news", "https://news.example/1", body_for(PARAGRAPHS))
        summary = bk.run(["rfc"])
        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["accepted"], 1)

    def test_empty_content_is_not_ingested(self):
        self.add_document("empty", "rfc", "https://rfc.example/empty", "   ")
        self.assertEqual(bk.run(["rfc"])["documents"], 0)

    def test_limit_stops_after_n_documents(self):
        for n in range(3):
            self.add_document(f"l{n}", "rfc", f"https://rfc.example/l{n}", body_for(PARAGRAPHS))
        self.assertEqual(bk.run(["rfc"], limit=2)["documents"], 2)


class IngestTests(KnowledgeIngestTestCase):
    def test_document_is_stored_as_a_knowledge_record(self):
        content = body_for(PARAGRAPHS)
        url = "https://rfc.example/8446"
        self.add_document("doc", "rfc", url, content, title="The TLS Protocol")
        bk.run(["rfc"])
        record = get_json(f"{bk.KNOWLEDGE_URL}/records/{urllib_quote(url)}")["record"]
        self.assertEqual(record["content"], content)
        self.assertEqual(record["provenance"], {"origin": "rfc", "locator": url})
        self.assertEqual(record["title"], "The TLS Protocol")

    def test_chunks_are_the_chunking_service_windows(self):
        content = " ".join(PARAGRAPHS) * 12   # several 1800-character windows
        url = "https://rfc.example/long"
        self.add_document("long", "rfc", url, content)
        summary = bk.run(["rfc"])
        self.assertEqual(summary["chunks_stored"], len(self.expected_chunks(url, content)))
        self.assertGreater(summary["chunks_stored"], 1)

    def test_vector_ids_and_metadata_trace_back_to_the_source(self):
        content = " ".join(PARAGRAPHS) * 12
        url = "https://rfc.example/traced"
        self.add_document("traced", "rfc", url, content)
        bk.run(["rfc"])
        content_sha = bk.hashlib.sha256(content.encode("utf-8")).hexdigest()
        expected = {bk.vector_id(c["chunk_id"], content_sha): c for c in self.expected_chunks(url, content)}
        entry = vs._ensure(bk.COLLECTION)
        self.assertEqual(set(entry["ids"]), set(expected))
        for vid, metadata in zip(entry["ids"], entry["metadatas"]):
            self.assertEqual((metadata["source"], metadata["url"]), ("rfc", url))
            self.assertEqual(metadata["chunk"], expected[vid]["index"])
            self.assertEqual(metadata["content_sha256"], content_sha)

    def test_rerun_stores_nothing_new(self):
        self.add_document("again", "rfc", "https://rfc.example/again", " ".join(PARAGRAPHS) * 2)
        first = bk.run(["rfc"])
        before = self.vector_count()
        second = bk.run(["rfc"])
        self.assertGreater(first["chunks_stored"], 0)
        self.assertEqual(second["chunks_stored"], 0)
        self.assertEqual(self.vector_count(), before)

    def test_changed_content_is_a_new_revision_with_new_vectors(self):
        url = "https://rfc.example/revised"
        self.add_document("v1", "rfc", url, body_for(PARAGRAPHS), fetched_at="2026-07-11T10:00:00")
        bk.run(["rfc"])
        before = self.vector_count()
        revised = body_for(PARAGRAPHS[:3] + ["A new paragraph describing a revised rule for padding."])
        self.add_document("v2", "rfc", url, revised, fetched_at="2026-08-01T10:00:00")
        summary = bk.run(["rfc"])
        # The first ledger row is the old fetch, now a duplicate; the second is the newer revision.
        self.assertEqual((summary["documents"], summary["accepted"]), (2, 2))
        self.assertGreater(summary["chunks_stored"], 0)
        self.assertGreater(self.vector_count(), before)
        self.assertEqual(get_json(f"{bk.KNOWLEDGE_URL}/records/{urllib_quote(url)}")["record"]["content"], revised)

    def test_index_refusal_means_no_chunks_are_written(self):
        url = "https://rfc.example/refused"
        self.add_document("refused", "rfc", url, body_for(PARAGRAPHS), fetched_at="2026-07-11T10:00:00")
        ix.record_revision(url, 9_999_999_999, "newer-elsewhere", PROFILES)
        summary = bk.run(["rfc"])
        self.assertEqual((summary["documents"], summary["accepted"], summary["chunks_stored"]), (1, 0, 0))
        self.assertEqual(self.vector_count(), 0)

    def test_dependency_failure_stops_the_run_and_a_rerun_completes_it(self):
        url = "https://rfc.example/outage"
        self.add_document("outage", "rfc", url, " ".join(PARAGRAPHS) * 2)
        saved = bk.CHUNKING_URL
        with socket_free_url() as dead:
            bk.CHUNKING_URL = dead
            with self.assertRaises(OSError):
                bk.run(["rfc"])
        self.assertEqual(self.vector_count(), 0)
        bk.CHUNKING_URL = saved
        summary = bk.run(["rfc"])
        self.assertGreater(summary["chunks_stored"], 0)


class RevisionSequenceTests(unittest.TestCase):
    def test_sequence_follows_fetch_time_and_falls_back_to_one(self):
        later = bk.revision_sequence("2026-08-01T10:00:00")
        earlier = bk.revision_sequence("2026-07-01T10:00:00")
        self.assertGreater(later, earlier)
        self.assertEqual(bk.revision_sequence(None), 1)
        self.assertEqual(bk.revision_sequence("not a date"), 1)


def urllib_quote(value):
    from urllib.parse import quote
    return quote(value, safe="")


class socket_free_url:
    """A loopback URL with nothing listening on it."""

    def __enter__(self):
        import socket
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{self._sock.getsockname()[1]}"

    def __exit__(self, *exc):
        self._sock.close()
        return False


if __name__ == "__main__":
    unittest.main()
