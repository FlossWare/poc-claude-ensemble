"""Contract tests for GA artifact ingestion through Learning and Memory."""
import json
import tempfile
from pathlib import Path

from learning.portable_artifacts import LearningArtifact
from learning_service import LearningService


class RecordingMemory:
    def __init__(self, result=True):
        self.result = result
        self.events = []

    def write_event(self, **event):
        self.events.append(event)
        return self.result


def artifact(parameters=None):
    return LearningArtifact.create(
        "ga.tuning.result",
        {
            "run_id": "ga-20261008_160000",
            "selected_parameters": parameters or {"compression": {"compression_level": 4.0}},
            "knowledge_promotion": {"eligible": False},
        },
        source="claude-ensemble.ga-tuning",
        provenance={"optimizer": "genetic-algorithm"},
        created_at="2026-10-08T16:00:00+00:00",
    ).to_dict()


def request(service, value):
    return json.loads(service._process_request(json.dumps({"op": "record_artifact", "artifact": value})))


def make_service(root, memory):
    service = LearningService(Path(root) / "learning.sock", Path(root) / "learning", operational_memory=memory)
    service.thompson_client = None
    return service


def test_ga_artifact_is_persisted_to_learning_and_memory_idempotently():
    with tempfile.TemporaryDirectory() as root:
        memory = RecordingMemory()
        service = make_service(root, memory)
        first = request(service, artifact())
        duplicate = request(service, artifact())
        assert first["ok"] is True
        assert first["memory"] is True
        assert first["artifact_status"] == "stored"
        assert duplicate["ok"] is True
        assert duplicate["artifact_status"] == "duplicate"
        stored = (Path(root) / "learning" / "portable_artifacts.jsonl").read_text()
        assert len([line for line in stored.splitlines() if line.strip()]) == 1
        assert len(memory.events) == 2
        assert memory.events[0]["event_type"] == "learning.artifact"
        assert memory.events[0]["event_id"] == memory.events[1]["event_id"]
        assert first["knowledge_promotion"] == "not_attempted"


def test_ga_artifact_memory_failure_is_not_reported_as_success():
    with tempfile.TemporaryDirectory() as root:
        service = make_service(root, RecordingMemory(result=False))
        response = request(service, artifact())
        assert response["ok"] is False
        assert response["memory"] is False
        assert response["knowledge_promotion"] == "not_attempted"



def test_memory_failure_can_retry_after_local_artifact_was_stored():
    class RetryMemory:
        def __init__(self):
            self.events = []
            self.results = [False, True]

        def write_event(self, **event):
            self.events.append(event)
            return self.results.pop(0)

    with tempfile.TemporaryDirectory() as root:
        memory = RetryMemory()
        service = make_service(root, memory)
        first = request(service, artifact())
        retry = request(service, artifact())
        assert first["ok"] is False
        assert first["artifact_status"] == "stored"
        assert retry["ok"] is True
        assert retry["artifact_status"] == "duplicate"
        assert len(memory.events) == 2
        assert memory.events[0]["event_id"] == memory.events[1]["event_id"]

def test_ga_artifact_rejects_conflicting_run_id():
    with tempfile.TemporaryDirectory() as root:
        memory = RecordingMemory()
        service = make_service(root, memory)
        assert request(service, artifact())["ok"] is True
        changed = artifact({"compression": {"compression_level": 1.0}})
        response = request(service, changed)
        assert response["ok"] is False
        assert "different artifact content" in response["error"]
        assert len(memory.events) == 1
