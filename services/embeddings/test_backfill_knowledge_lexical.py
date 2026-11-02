"""Backfill tests: vector records are copied into the exact-match index, idempotently, over real HTTP."""
import os
import re
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import backfill_knowledge_lexical as bf
import knowledge_lexical as kl
import lexical_service as lx
import vector_service as vs
import vector_store


def basis(i):
    vector = [0.0] * 768
    vector[i] = 1.0
    return vector


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self._saved_store = vector_store.STORE_DIR
        vector_store.STORE_DIR = os.path.join(self.root, "vectors")
        vs._cache.clear()
        self.addCleanup(setattr, vector_store, "STORE_DIR", self._saved_store)
        self.addCleanup(vs._cache.clear)
        self.db = os.path.join(self.root, "knowledge-lexical.sqlite3")
        for patch in (mock.patch.object(lx, "KNOWLEDGE_DB_PATH", self.db),
                      mock.patch.object(lx, "analyze", lambda text: re.findall(r"[\w\-._:/]+", text.lower()))):
            patch.start()
            self.addCleanup(patch.stop)
        self.vector_server = serve(vs.create_server("127.0.0.1", 0))
        self.lexical_server = serve(lx.create_server("127.0.0.1", 0))
        self.addCleanup(self.stop_servers)
        self.vector_url = f"http://127.0.0.1:{self.vector_server.server_address[1]}"
        self.lexical_url = f"http://127.0.0.1:{self.lexical_server.server_address[1]}"

    def stop_servers(self):
        for server in (self.vector_server, self.lexical_server):
            server.shutdown()
            server.server_close()

    def store(self, ids, documents, metas):
        with vs._lock:
            vector_store.collection(bf.COLLECTION).upsert(
                ids=ids, embeddings=[basis(i) for i in range(len(ids))], documents=documents, metadatas=metas)
            vs._apply_upsert(bf.COLLECTION, ids, documents, __import__("numpy").asarray(
                [basis(i) for i in range(len(ids))], dtype="float32"), metas)

    def test_copies_every_stored_chunk_into_the_exact_match_index(self):
        self.store(["v1", "v2"], ["RFC-8259 JSON", "CBOR tokens"],
                   [{"chunk_id": "c1", "url": "https://u/1", "source": "rfc", "title": "RFC 8259"},
                    {"chunk_id": "c2", "url": "https://u/2", "source": "rfc", "title": "CBOR"}])
        totals = bf.backfill(self.vector_url, self.lexical_url)
        self.assertEqual((totals["read"], totals["inserted"], totals["skipped"]), (2, 2, 0))
        self.assertEqual(kl.count(path=self.db), 2)
        hit = kl.search("RFC-8259", 3, lambda t: re.findall(r"[\w\-._:/]+", t.lower()), path=self.db)[0]
        self.assertEqual((hit["vector_id"], hit["chunk_id"]), ("v1", "c1"))

    def test_rerun_inserts_nothing_new(self):
        self.store(["v1"], ["text"], [{"chunk_id": "c1", "url": "u", "source": "rfc", "title": "T"}])
        bf.backfill(self.vector_url, self.lexical_url)
        again = bf.backfill(self.vector_url, self.lexical_url)
        self.assertEqual((again["inserted"], again["already_present"]), (0, 1))
        self.assertEqual(kl.count(path=self.db), 1)

    def test_records_without_chunk_identity_are_skipped(self):
        self.store(["v1", "v2"], ["no metadata", "with metadata"],
                   [{"source": "rfc"}, {"chunk_id": "c2", "url": "u", "source": "rfc", "title": "T"}])
        totals = bf.backfill(self.vector_url, self.lexical_url)
        self.assertEqual((totals["skipped"], totals["inserted"]), (1, 1))

    def test_limit_stops_early(self):
        self.store(["v1", "v2", "v3"], ["a", "b", "c"],
                   [{"chunk_id": f"c{i}", "url": "u", "source": "rfc", "title": "T"} for i in (1, 2, 3)])
        totals = bf.backfill(self.vector_url, self.lexical_url, limit=2)
        self.assertEqual(totals["read"], 2)
        self.assertEqual(kl.count(path=self.db), 2)


if __name__ == "__main__":
    unittest.main()
