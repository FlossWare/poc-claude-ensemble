import unittest
import intent_service as its


class IntentTests(unittest.TestCase):
    def test_labels(self):
        cases = {
            "what did we decide about the cache TTL last time": "history",
            "why does the hook time out": "debug",
            "how do I register a Stop hook": "howto",
            "where is the graph store file": "lookup",
            "implement a retry for the embedding client": "change",
        }
        for text, label in cases.items():
            self.assertEqual(its.classify(text)["label"], label, text)

    def test_evidence_lists_fired_rules(self):
        result = its.classify("why is the service broken and why does it crash")
        self.assertEqual(result["label"], "debug")
        self.assertTrue(result["evidence"])
        self.assertTrue(all(e["rule"] == "debug" for e in result["evidence"]))

    def test_no_evidence_is_other_with_zero_confidence(self):
        result = its.classify("zxq flumox")
        self.assertEqual((result["label"], result["confidence"], result["evidence"]), ("other", 0.0, []))

    def test_confidence_bounded_and_deterministic(self):
        text = "how do I fix the error, how does it work"
        first, second = its.classify(text), its.classify(text)
        self.assertEqual(first, second)
        self.assertGreater(first["confidence"], 0.0)
        self.assertLessEqual(first["confidence"], 1.0)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            its.classify(None)
        with self.assertRaises(OverflowError):
            its.classify("x" * (its.MAX_TEXT_CHARS + 1))

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            its.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
