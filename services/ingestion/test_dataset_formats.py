import tempfile
import unittest
import zipfile
from pathlib import Path

import dataset_formats as formats


class DatasetFormatTests(unittest.TestCase):
    def test_csv_is_emitted_in_bounded_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            path.write_text("id,name\n1,Ada\n2,Grace\n3,Linus\n")
            docs = list(formats.iter_documents(path, rows_per_document=2))
        self.assertEqual(len(docs), 2)
        self.assertIn('"name": "Ada"', docs[0]["text"])
        self.assertIn('"name": "Linus"', docs[1]["text"])

    def test_json_array_is_batched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text('[{"id":1},{"id":2},{"id":3}]')
            docs = list(formats.iter_documents(path, rows_per_document=2))
        self.assertEqual(len(docs), 2)
        self.assertIn('"id": 3', docs[1]["text"])

    def test_jsonl_tolerates_malformed_line_without_dropping_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"id":1}\nnot-json\n')
            docs = list(formats.iter_documents(path))
        self.assertEqual(len(docs), 1)
        self.assertIn('"_raw": "not-json"', docs[0]["text"])

    def test_zip_extracts_regular_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("folder/data.csv", "id\n1\n")
            extracted = formats.extract_archive(archive, root / "out")
            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0].read_text(), "id\n1\n")

    def test_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "no")
            with self.assertRaises(formats.DatasetFormatError):
                formats.extract_archive(archive, root / "out")
            self.assertFalse((root.parent / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
