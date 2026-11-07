"""Provenance fields must reach the exact-match index from the builder and the backfill."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
