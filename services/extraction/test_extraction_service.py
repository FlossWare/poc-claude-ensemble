import unittest
import extraction_service as es


def kinds(text):
    return {(e["type"], e["text"]) for e in es.extract(text)}


class ExtractionTests(unittest.TestCase):
    def test_exact_identifiers_are_found_with_full_confidence(self):
        text = "see CPSEARCH-10177 in memory-service/tokenization_service.py at v2.4.1 and https://example.com/x"
        found = kinds(text)
        self.assertIn(("issue_key", "CPSEARCH-10177"), found)
        self.assertIn(("path", "memory-service/tokenization_service.py"), found)
        self.assertIn(("version", "v2.4.1"), found)
        self.assertIn(("url", "https://example.com/x"), found)
        for e in es.extract(text):
            if e["exact"]:
                self.assertEqual(e["confidence"], 1.0)

    def test_uuid_and_file_names(self):
        uid = "12073e51-80ad-4eae-8963-e15686cde922"
        self.assertIn(("uuid", uid), kinds(f"session {uid}"))
        self.assertIn(("file", "hybrid_service.py"), kinds("edit hybrid_service.py now"))

    def test_spelling_preserved_exactly(self):
        text = "Thompson_Router and Memory-Service"
        for e in es.extract(text):
            self.assertEqual(text[e["start"]:e["end"]], e["text"])

    def test_uncertain_names_are_flagged_not_exact(self):
        entities = es.extract("Talk to Greg Thompson about it")
        names = [e for e in entities if e["type"] == "capitalised_name"]
        self.assertTrue(names)
        self.assertTrue(all(not e["exact"] and e["confidence"] < 1.0 for e in names))

    def test_no_overlapping_spans(self):
        entities = es.extract("https://example.com/path/to/file.py and memory-service/x.py")
        spans = [(e["start"], e["end"]) for e in entities if e["exact"]]
        for i, a in enumerate(spans):
            for b in spans[i + 1:]:
                self.assertTrue(a[1] <= b[0] or b[1] <= a[0])

    def test_empty_and_rejections(self):
        self.assertEqual(es.extract(""), [])
        with self.assertRaises(ValueError):
            es.extract(5)
        with self.assertRaises(OverflowError):
            es.extract("x" * (es.MAX_TEXT_CHARS + 1))

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            es.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
