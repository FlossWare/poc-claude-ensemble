import importlib.util
import json
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("learning_service.py")


def load_learning_service():
    spec = importlib.util.spec_from_file_location("learning_service_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(task_id="retry-contract-001"):
    return {
        "op": "process_outcome",
        "task_id": task_id,
        "task_type": "testing",
        "model": "haiku",
        "rating": 4,
        "tokens": 1000,
        "cost": 0.005,
    }


class RecordingThompson:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def get_circuit_breaker_state(self):
        return {"state": "closed"}

    def record_outcome(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def get_state(self):
        return {"models": {"haiku": {"successes": 1}}}


class RecordingMemory:
    def __init__(self, result=True):
        self.result = result
        self.events = []

    def write_event(self, event_id, event_type, source, payload):
        self.events.append((event_id, event_type, source, payload))
        return self.result


def make_service(module, root, memory=None):
    return module.LearningService(
        Path(root) / "learning.sock",
        Path(root) / "learning",
        operational_memory=memory,
    )


def test_failed_downstream_learning_leaves_durable_retry_payload():
    module = load_learning_service()
    with tempfile.TemporaryDirectory() as root:
        service = make_service(module, root)
        service.thompson_client = RecordingThompson(result=False)
        response = json.loads(service._process_request(json.dumps(payload())))
        assert response["ok"] is False
        assert response["checkpoint_advanced"] is False
        assert service.system.get_outcome("retry-contract-001") is not None
        assert not service.system.is_processed("retry-contract-001")


def test_retry_after_restart_reuses_persisted_payload_and_completes():
    module = load_learning_service()
    with tempfile.TemporaryDirectory() as root:
        first = make_service(module, root)
        first.thompson_client = RecordingThompson(result=False)
        failed = json.loads(
            first._process_request(json.dumps(payload("restart-retry-001")))
        )
        assert failed["ok"] is False
        assert failed["checkpoint_advanced"] is False

        restarted = make_service(module, root)
        successful = RecordingThompson(result=True)
        restarted.thompson_client = successful
        retry = dict(payload("restart-retry-001"), rating=1, tokens=999)
        conflict = json.loads(restarted._process_request(json.dumps(retry)))
        assert conflict["ok"] is False
        assert conflict["conflict"] is True
        assert not restarted.system.is_processed("restart-retry-001")

        completed = json.loads(
            restarted._process_request(json.dumps(payload("restart-retry-001")))
        )
        assert completed["ok"] is True
        assert completed["checkpoint_advanced"] is True
        assert len(successful.calls) == 1
        assert successful.calls[0]["success"] is True
        assert restarted.system.is_processed("restart-retry-001")


def test_memory_failure_blocks_checkpoint_and_retry_can_succeed():
    module = load_learning_service()
    with tempfile.TemporaryDirectory() as root:
        memory = RecordingMemory(result=False)
        service = make_service(module, root, memory=memory)
        service.thompson_client = RecordingThompson(result=True)
        failed = json.loads(
            service._process_request(json.dumps(payload("memory-retry-001")))
        )
        assert failed["ok"] is False
        assert failed["checkpoint_advanced"] is False
        assert not service.system.is_processed("memory-retry-001")
        memory.result = True
        completed = json.loads(
            service._process_request(json.dumps(payload("memory-retry-001")))
        )
        assert completed["ok"] is True
        assert completed["checkpoint_advanced"] is True
        assert service.system.is_processed("memory-retry-001")


def test_checkpointed_task_is_not_reprocessed():
    module = load_learning_service()
    with tempfile.TemporaryDirectory() as root:
        service = make_service(module, root)
        thompson = RecordingThompson(result=True)
        service.thompson_client = thompson
        first = json.loads(
            service._process_request(json.dumps(payload("duplicate-001")))
        )
        second = json.loads(
            service._process_request(json.dumps(payload("duplicate-001")))
        )
        assert first["ok"] is True
        assert second["ok"] is True
        assert second["duplicate"] is True
        assert len(thompson.calls) == 1
