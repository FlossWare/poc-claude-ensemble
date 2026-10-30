"""Scrape ledger: JSON records keep their URL identity; books are identified by file path."""
import json
import os
import tempfile
import unittest

import scrape_ledger as sl


class BooksTests(unittest.TestCase):
    def setUp(self):
        self.raw = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.raw, "books"))
        self.ledger = os.path.join(tempfile.mkdtemp(), "fetches.jsonl")
        self._saved = (sl.RAW, sl.LEDGER, sl.SOURCES)
        sl.RAW, sl.LEDGER, sl.SOURCES = self.raw, self.ledger, ["books"]
        with open(os.path.join(self.raw, "books", "Guide.pdf"), "wb") as handle:
            handle.write(b"%PDF-1.4 fake bytes for hashing")

    def tearDown(self):
        sl.RAW, sl.LEDGER, sl.SOURCES = self._saved

    def rows(self):
        with open(self.ledger, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_pdf_becomes_a_books_row_with_file_identity_and_hash(self):
        self.assertEqual(sl.build()["added"], 1)
        row = self.rows()[0]
        self.assertEqual((row["source"], row["record"]), ("books", "books/Guide.pdf"))
        self.assertTrue(row["url"].startswith("file://") and row["url"].endswith("Guide.pdf"))
        self.assertEqual(len(row["sha256"]), 64)
        self.assertEqual((row["producer"], row["kind"]), ("scrape", "fetch"))

    def test_rebuild_does_not_add_books_twice(self):
        sl.build()
        self.assertEqual(sl.build()["added"], 0)
        self.assertEqual(len(self.rows()), 1)


if __name__ == "__main__":
    unittest.main()
