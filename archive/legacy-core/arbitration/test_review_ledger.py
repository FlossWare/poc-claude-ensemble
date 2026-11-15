"""Worker results and arbiter decisions are written to the reviews ledger, and a failed write never stops a review."""
import json
import os
import tempfile
import unittest

from arbitration import review_ledger as rl
from arbitration.orchestrator import ArbiterResult, WorkerResult


class ReviewLedgerTests(unittest.TestCase):
    def setUp(self):
        self._saved = rl.LEDGER
        self.ledger = os.path.join(tempfile.mkdtemp(), "reviews.jsonl")
        rl.LEDGER = self.ledger

    def tearDown(self):
        rl.LEDGER = self._saved

    def rows(self):
        with open(self.ledger, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_worker_and_arbiter_rows_share_a_review_id(self):
        workers = [WorkerResult(model="sonnet", phase=1, analysis="found a bug"),
                   WorkerResult(model="haiku", phase=1, error="timeout")]
        arbiter = ArbiterResult(model="opus", phase=1, adjudicated_result="approve with fixes", selected_worker="sonnet")
        rl.record_phase("rid1", "code_review", 1, workers, arbiter)
        rows = self.rows()
        self.assertEqual([r["kind"] for r in rows], ["worker_result", "worker_result", "arbiter_decision"])
        self.assertEqual({r["review_id"] for r in rows}, {"rid1"})
        self.assertEqual((rows[1]["succeeded"], rows[1]["error"]), (False, "timeout"))
        self.assertEqual(rows[2]["selected_worker"], "sonnet")
        self.assertEqual(rows[0]["producer"], "arbitration")

    def test_unwritable_ledger_does_not_raise(self):
        rl.LEDGER = "/proc/ce-no-such-dir/reviews.jsonl"
        rl.record_phase("rid2", "code_review", 1, [WorkerResult(model="haiku", phase=1, analysis="x")], None)


if __name__ == "__main__":
    unittest.main()
