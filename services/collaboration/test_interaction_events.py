import sys
import unittest
from pathlib import Path

services_root = Path(__file__).resolve().parents[1]
if str(services_root) not in sys.path:
    sys.path.insert(0, str(services_root))

from collaboration import CollaborationOrchestrator
from providers import ModelCapabilities, ModelProvider, ModelRequest, ModelResponse


class FakeProvider(ModelProvider):
    def __init__(self, text):
        self.text = text

    @property
    def capabilities(self):
        return ModelCapabilities(
            provider="fake",
            supports_system_prompt=True,
            supports_messages=True,
            supports_temperature_override=True,
            supports_max_tokens_override=True,
            supports_timeout_override=True,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider="fake",
            model=request.model or "fake-model",
            text=self.text,
            input_tokens=12,
            output_tokens=7,
            cost_usd=0.001,
            request_id="request-1",
        )


class InteractionEventTests(unittest.TestCase):
    def test_real_collaboration_emits_worker_and_arbiter_lifecycle_events(self):
        import json

        events = []
        arbiter_text = json.dumps({
            "selected_candidate": "r1-sonnet",
            "decision": "accept",
            "rationale": "supported",
            "supporting_evidence": ["worker proposal"],
            "rejected_alternatives": [],
            "blocking_concerns": [],
            "follow_ups": [],
            "complete": True,
            "human_decision_required": False,
        })
        loop = CollaborationOrchestrator(
            "inspect a task",
            solvers={"sonnet": FakeProvider("proposal")},
            arbiter=FakeProvider(arbiter_text),
            reviewers={},
            execution_id="execution-real",
            event_sink=events.append,
        )

        result = loop.run()

        self.assertEqual(result.status, "accepted")
        names = [event["event_name"] for event in events]
        self.assertIn("worker.started", names)
        self.assertIn("worker.responded", names)
        self.assertIn("arbiter.started", names)
        self.assertIn("arbiter.responded", names)
        self.assertIn("arbiter.decided", names)
        worker = next(event for event in events if event["event_name"] == "worker.responded")
        self.assertEqual(worker["provider"], "fake")
        self.assertEqual(worker["model"], "sonnet")
        self.assertEqual(worker["input_tokens"], 12)
        self.assertEqual(worker["output_tokens"], 7)
        self.assertNotIn("prompt", worker)
        self.assertNotIn("response", worker)

    def test_empty_worker_response_is_recorded_as_failure(self):
        events = []
        loop = CollaborationOrchestrator(
            "inspect a task",
            solvers={"sonnet": FakeProvider("")},
            arbiter=FakeProvider("{}"),
            reviewers={},
            event_sink=events.append,
        )
        with self.assertRaises(RuntimeError):
            loop.run()
        failure = next(event for event in events if event["event_name"] == "worker.failed")
        self.assertEqual(failure["error_type"], "EmptyModelResponse")

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
