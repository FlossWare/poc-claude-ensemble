import sys
import unittest
from pathlib import Path

services_root = Path(__file__).resolve().parents[1]
if str(services_root) not in sys.path:
    sys.path.insert(0, str(services_root))

from collaboration import CollaborationOrchestrator


class InteractionEventTests(unittest.TestCase):
    def test_event_sink_receives_stable_execution_correlation_and_metadata(self):
        events = []
        orchestrator = CollaborationOrchestrator(
            "inspect this task",
            solvers={"worker-a": object()},
            arbiter=object(),
            reviewers={},
            execution_id="execution-123",
            parent_event_id="parent-1",
            event_sink=events.append,
        )

        orchestrator._emit_interaction(
            "worker.started", worker_id="worker-a", role="solver", round=1
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_type"], "model_interaction")
        self.assertEqual(event["event_name"], "worker.started")
        self.assertEqual(event["execution_id"], "execution-123")
        self.assertEqual(event["correlation_id"], "execution-123")
        self.assertEqual(event["worker_id"], "worker-a")
        self.assertEqual(event["parent_event_id"], "parent-1")
        self.assertTrue(event["event_id"])
        self.assertTrue(event["timestamp"])

    def test_event_sink_failure_does_not_break_execution(self):
        def failing_sink(_event):
            raise OSError("ledger unavailable")

        orchestrator = CollaborationOrchestrator(
            "inspect this task",
            solvers={"worker-a": object()},
            arbiter=object(),
            reviewers={},
            event_sink=failing_sink,
        )

        orchestrator._emit_interaction("worker.started", worker_id="worker-a")


if __name__ == "__main__":
    unittest.main()
