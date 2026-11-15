"""The arbitration model pool records Thompson's shadow proposal and returns exactly the same models."""
import json
import os
import tempfile
import unittest

from arbitration.orchestrator import ModelPool
from shared import thompson_shadow as ts
from shared.test_thompson_shadow import FakeThompson, closed_url


class ShadowPoolTests(unittest.TestCase):
    def setUp(self):
        self._saved = (ts.THOMPSON_URL, ts.LEDGER)
        self.ledger = os.path.join(tempfile.mkdtemp(), "selections.jsonl")
        ts.LEDGER = self.ledger
        self.fake = FakeThompson("deepseek-v4.1-flash")
        ts.THOMPSON_URL = self.fake.url

    def tearDown(self):
        self.fake.close()
        ts.THOMPSON_URL, ts.LEDGER = self._saved

    def rows(self):
        with open(self.ledger, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_selection_is_unchanged_and_shadow_rows_are_written(self):
        pool = ModelPool()
        self.assertEqual(pool.get_workers(2, "balanced"), ["sonnet", "rits/zai-org/glm-5-3"])
        self.assertEqual(pool.get_arbiter("expensive"), "opus")
        roles = [r["role"] for r in self.rows()]
        self.assertEqual(roles, ["worker", "arbiter"])
        self.assertEqual(self.rows()[0]["thompson_proposal"], "deepseek-v4.1-flash")

    def test_selection_is_unchanged_when_thompson_is_down(self):
        ts.THOMPSON_URL = closed_url()
        pool = ModelPool()
        self.assertEqual(pool.get_workers(1, "cheap"), ["haiku"])
        self.assertIsNone(self.rows()[0]["thompson_proposal"])


if __name__ == "__main__":
    unittest.main()
