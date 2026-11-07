"""Provenance fields must reach the exact-match index from the builder and the backfill."""

import unittest
from unittest import mock

import build_knowledge_chunks as bk
import backfill_knowledge_lexical as backfill

SHA = "9f" * 32


class KnowledgeRowsTests(unittest.TestCase):
    def test_builder_rows_carry_chunk_position_count_and_hash(self):
        meta = {"chunk_id": "c1", "url": "https://example.test/doc", "source": "rfc", "title": "Doc",
                "chunk": 3, "chunks": 7, "content_sha256": SHA}
        row = bk.knowledge_rows([("v1", "text", meta)])[0]
        self.assertEqual((row["chunk_index"], row["chunk_count"], row["content_sha256"]), (3, 7, SHA))

    def test_builder_rows_without_metadata_fields_carry_nulls(self):
        meta = {"chunk_id": "c1", "url": "https://example.test/doc", "source": "rfc", "title": "Doc"}
        row = bk.knowledge_rows([("v1", "text", meta)])[0]
        self.assertEqual((row["chunk_index"], row["chunk_count"], row["content_sha256"]), (None, None, None))


class BackfillRowTests(unittest.TestCase):
    def test_backfill_row_maps_stored_metadata_to_provenance(self):
        record = {"id": "v1", "document": "text", "metadata": {
            "chunk_id": "c1", "url": "https://example.test/doc", "source": "rfc", "title": "Doc",
            "chunk": 3, "chunks": 7, "content_sha256": SHA}}
        row = backfill.chunk_row(record)
        self.assertEqual((row["chunk_index"], row["chunk_count"], row["content_sha256"]), (3, 7, SHA))
        self.assertEqual(row["vector_id"], "v1")

    def test_backfill_row_skips_records_that_cannot_identify_the_chunk(self):
        self.assertIsNone(backfill.chunk_row({"id": "v1", "document": "t", "metadata": {"url": "u"}}))


class BackfillRunTests(unittest.TestCase):
    """The backfill against fake services: totals, offsets, and failure handling with parallel posts."""

    def setUp(self):
        self.records = [{"id": f"v{i}", "document": f"text {i}", "metadata": {
            "chunk_id": f"c{i}", "url": "https://example.test/doc", "source": "rfc", "title": "Doc",
            "chunk": i, "chunks": 1000, "content_sha256": SHA}} for i in range(1234)]
        self.records.append({"id": "bad", "document": "x", "metadata": {"url": "u"}})  # cannot identify its chunk
        self.posted = []

    def fake_get(self, url):
        query = dict(part.split("=") for part in url.split("?", 1)[1].split("&"))
        offset, limit = int(query["offset"]), int(query["limit"])
        page = self.records[offset:offset + limit]
        return {"records": page, "total": len(self.records)}

    def fake_post(self, url, payload):
        self.posted.append(len(payload["chunks"]))
        return {"ok": True, "inserted": len(payload["chunks"]), "already_present": 0, "repaired": 0}

    def run_backfill(self, **kwargs):
        with mock.patch.object(backfill, "get", self.fake_get), mock.patch.object(backfill, "post", self.fake_post):
            return backfill.backfill(**kwargs)

    def test_parallel_run_reads_every_record_and_skips_unidentified_ones(self):
        totals = self.run_backfill(workers=4)
        self.assertEqual(totals["read"], len(self.records))
        self.assertEqual(totals["skipped"], 1)
        self.assertEqual(totals["inserted"], len(self.records) - 1)
        self.assertEqual(sum(self.posted), len(self.records) - 1)

    def test_start_offset_resumes_from_that_record(self):
        totals = self.run_backfill(workers=2, start_offset=1000)
        self.assertEqual(totals["read"], len(self.records) - 1000)
        self.assertEqual(totals["inserted"], len(self.records) - 1000 - 1)

    def test_failed_post_stops_the_run(self):
        def failing_post(url, payload):
            raise RuntimeError("lexical service unavailable")
        with mock.patch.object(backfill, "get", self.fake_get), mock.patch.object(backfill, "post", failing_post):
            with self.assertRaises(RuntimeError):
                backfill.backfill(workers=3)


if __name__ == "__main__":
    unittest.main()
