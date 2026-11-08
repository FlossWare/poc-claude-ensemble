import unittest

from orchestrator import CollaborationOrchestrator


class InteractionEventTests(unittest.TestCase):
    def test_event_sink_receives_stable_execution_correlation_and_metadata(self):
        events = []
        orchestrator = CollaborationOrchestrator(
            "inspect this task",
            solvers={"worker-a": object()},
            arbiter=object(),
            reviewers={},
            execution_id="execution-123",
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
