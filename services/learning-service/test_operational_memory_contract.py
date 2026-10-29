import json
import tempfile
from pathlib import Path

from learning_service import LearningService


class SnapshotMemory:
    def __init__(self, snapshot_result=True, snapshot_exception=None):
        self.events = []
        self.snapshot_result = snapshot_result
        self.snapshot_exception = snapshot_exception

    def write_event(self, event_id, event_type, source, payload):
        self.events.append((event_id, event_type, source, payload))
        if event_type == "thompson.state":
            if self.snapshot_exception is not None:
                raise self.snapshot_exception
            return self.snapshot_result
        return True


class SuccessfulThompson:
    def get_circuit_breaker_state(self):
        return {"state": "closed"}

    def record_outcome(self, **kwargs):
        return True

    def get_state(self):
        return {"models": {"haiku": {"successes": 1}}}


class FailingGetStateThompson(SuccessfulThompson):
    def get_state(self):
        raise RuntimeError("Thompson state unavailable")


def _process(service, task_id):
    return json.loads(service._process_request(json.dumps({
        "op": "process_outcome",
        "task_id": task_id,
        "task_type": "testing",
        "model": "haiku",
        "rating": 4,
        "tokens": 1000,
        "cost": 0.005,
    })))


def _service(temp_dir, memory, thompson):
    service = LearningService(
        Path(temp_dir) / "learning.sock",
        Path(temp_dir) / "learning",
        operational_memory=memory,
    )
    service.thompson_client = thompson
    return service


def test_thompson_snapshot_failure_blocks_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        memory = SnapshotMemory(snapshot_exception=RuntimeError("required Thompson snapshot unavailable"))
        service = _service(temp_dir, memory, SuccessfulThompson())

        response = _process(service, "snapshot_failure_001")

        assert response["ok"] is False
        assert response["checkpoint_advanced"] is False
        assert response["thompson_state_memory"] is False
        assert not service.system.is_processed("snapshot_failure_001")
        assert [event[1] for event in memory.events] == [
            "learning.outcome",
            "thompson.state",
        ]


def test_thompson_snapshot_soft_failure_blocks_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        memory = SnapshotMemory(snapshot_result=False)
        service = _service(temp_dir, memory, SuccessfulThompson())

        response = _process(service, "snapshot_failure_002")

        assert response["ok"] is False
        assert response["checkpoint_advanced"] is False
        assert response["thompson_state_memory"] is False
        assert not service.system.is_processed("snapshot_failure_002")
        assert [event[1] for event in memory.events] == [
            "learning.outcome",
            "thompson.state",
        ]


def test_thompson_get_state_failure_blocks_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        memory = SnapshotMemory()
        service = _service(temp_dir, memory, FailingGetStateThompson())

        response = _process(service, "snapshot_failure_003")

        assert response["ok"] is False
        assert response["checkpoint_advanced"] is False
        assert response["thompson_state_memory"] is False
        assert not service.system.is_processed("snapshot_failure_003")
        assert [event[1] for event in memory.events] == ["learning.outcome"]
