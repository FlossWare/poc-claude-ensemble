#!/usr/bin/env python3
"""Tests for the multipass ingest queue (services/ingestion/ingest_queue.py)."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_queue import (IngestQueue, QueueError, PENDING, IN_FLIGHT,
                          DONE, FAILED)


def _queue(td):
    return IngestQueue(Path(td) / "q.sqlite3")


class EnqueueClaimTests(unittest.TestCase):
    def test_enqueue_claim_complete_lifecycle(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            n = q.enqueue([{"path": "a.json"}, {"path": "b.json"}])
            self.assertEqual(n, 2)
            claimed = q.claim(pass_=1, limit=10)
            self.assertEqual(len(claimed), 2)
            self.assertEqual(claimed[0]["payload"]["path"], "a.json")  # FIFO
            token = claimed[0]["claim_token"]
            self.assertEqual(q.stats(pass_=1), {IN_FLIGHT: 2})
            done = q.complete(ids=[c["id"] for c in claimed],
                              claim_token=token, result={"ok": True})
            self.assertEqual(done, 2)
            self.assertEqual(q.stats(pass_=1), {DONE: 2})

    def test_wrong_token_cannot_complete(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": 1}])
            claimed = q.claim(pass_=1)
            n = q.complete(ids=[claimed[0]["id"]],
                           claim_token="not-the-token")
            self.assertEqual(n, 0)  # claim still held
            self.assertEqual(q.stats(pass_=1), {IN_FLIGHT: 1})

    def test_priority_orders_drain(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": "low"}], priority=100)
            q.enqueue([{"p": "high"}], priority=1)
            claimed = q.claim(pass_=1)
            self.assertEqual(claimed[0]["payload"]["p"], "high")

    def test_passes_are_independent(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": 1}])
            q.enqueue([{"p": 2}], pass_=2)
            claimed1 = q.claim(pass_=1)
            claimed2 = q.claim(pass_=2)
            self.assertEqual(claimed1[0]["payload"]["p"], 1)
            self.assertEqual(claimed2[0]["payload"]["p"], 2)

    def test_fail_requeues_under_max_attempts(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": 1}], max_attempts=2)
            claimed = q.claim(pass_=1)
            q.fail(ids=[claimed[0]["id"]], claim_token=claimed[0]["claim_token"],
                   reason="transient")
            self.assertEqual(q.stats(pass_=1), {PENDING: 1})
            # second claim exhausts attempts
            claimed2 = q.claim(pass_=1)
            self.assertEqual(claimed2[0]["attempts"], 2)
            q.fail(ids=[claimed2[0]["id"]],
                   claim_token=claimed2[0]["claim_token"], reason="again")
            self.assertEqual(q.stats(pass_=1), {FAILED: 1})

    def test_stale_claim_is_requeued_on_next_claim(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": 1}])
            claimed = q.claim(pass_=1)
            self.assertEqual(q.stats(pass_=1), {IN_FLIGHT: 1})
            # simulate a stale claim: backdate claimed_at beyond timeout
            with q._conn:
                q._conn.execute("UPDATE queue SET claimed_at=?",
                                (time.time() - 7200,))
            reclaimed = q.claim(pass_=1)
            self.assertEqual(len(reclaimed), 1)
            self.assertEqual(reclaimed[0]["attempts"], 2)

    def test_validation(self):
        with TemporaryDirectory() as td:
            q = _queue(td)
            with self.assertRaises(QueueError):
                q.enqueue([])
            with self.assertRaises(QueueError):
                q.enqueue(["not-a-dict"])
            with self.assertRaises(QueueError):
                q.claim(pass_=1, limit=0)

    def test_durability_across_reopen(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "q.sqlite3"
            q = IngestQueue(path)
            q.enqueue([{"p": 1}, {"p": 2}])
            claimed = q.claim(pass_=1, limit=1)
            q2 = IngestQueue(path)  # reopen: state must survive
            self.assertEqual(q2.stats(), {PENDING: 1, IN_FLIGHT: 1})
            # and the other session can still complete with the token
            n = q2.complete(ids=[claimed[0]["id"]],
                            claim_token=claimed[0]["claim_token"])
            self.assertEqual(n, 1)


    def test_stale_exhausted_claim_fails_not_requeues(self):
        """A poison item whose worker crashed (stale claim) and whose
        attempts are exhausted must go to failed, not be reclaimed
        forever (review finding: stale requeue ignored the retry cap)."""
        with TemporaryDirectory() as td:
            q = _queue(td)
            q.enqueue([{"p": "poison"}], max_attempts=1)
            claimed = q.claim(pass_=1)  # attempts -> 1 (at max)
            self.assertEqual(claimed[0]["attempts"], 1)
            # simulate a crashed worker: backdate the claim past timeout
            with q._conn:
                q._conn.execute("UPDATE queue SET claimed_at=?",
                                (time.time() - 7200,))
            reclaimed = q.claim(pass_=1)
            self.assertEqual(reclaimed, [])  # NOT reclaimed
            self.assertEqual(q.stats(pass_=1), {FAILED: 1})



if __name__ == "__main__":
    unittest.main()
