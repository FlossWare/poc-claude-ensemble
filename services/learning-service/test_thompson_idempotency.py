#!/usr/bin/env python3
"""Regression: Thompson must not double-count an outcome on retry.

Scenario (confirmed by two-group multi-model review 2026-11-25): a
process_outcome delivers to Thompson, then fails at a later step (state
snapshot, checkpoint write, or a crash). The client retries. The retry
finds the persisted outcome, passes payload checks, and must NOT
deliver to Thompson again — the durable thompson_applied marker
prevents the double count.

Also covers: the marker is cleared by reset_learning, and the per-task
lock map is bounded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # services/
SERVICE_SCRIPT = Path(__file__).with_name("learning_service.py")

sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("learning_service_under_test",
                                                SERVICE_SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


class CountingThompson:
    """Thompson stand-in that counts deliveries and can fail snapshots."""

    def __init__(self, fail_state: bool = False):
        self.calls = 0
        self.fail_state = fail_state

    def record_outcome(self, **_kwargs) -> bool:
        self.calls += 1
        return True

    def get_state(self):
        if self.fail_state:
            raise RuntimeError("simulated snapshot failure")
        return {"models": {}}

    def get_circuit_breaker_state(self):
        return None


class MemoryOk:
    """Operational memory stand-in; the post-Thompson state snapshot path
    only runs when operational_memory is configured."""

    def write_event(self, **_kwargs) -> bool:
        return True


def _outcome_request(task_id: str) -> str:
    return json.dumps({
        "op": "process_outcome",
        "task_id": task_id,
        "task_type": "code_review",
        "model": "haiku",
        "rating": 4,
        "tokens": 100,
        "cost": 0.01,
    })


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ce-learning-idem-") as temp:
        learning_dir = Path(temp) / "learning"
        service = mod.LearningService(Path(temp) / "unused.sock", learning_dir)
        thompson = CountingThompson(fail_state=True)
        service.thompson_client = thompson
        service.operational_memory = MemoryOk()

        # 1. first attempt: Thompson delivered (call 1), then the state
        #    snapshot fails -> ok False, checkpoint NOT advanced
        r1 = json.loads(service._process_request(_outcome_request("task-idem-1")))
        assert r1.get("thompson") is True, r1
        assert r1.get("checkpoint_advanced") is False, r1
        assert thompson.calls == 1, thompson.calls
        assert service.system.thompson_was_applied("task-idem-1")

        # 2. retry with the snapshot still failing: the outcome is
        #    already delivered — Thompson must NOT be called again
        r2 = json.loads(service._process_request(_outcome_request("task-idem-1")))
        assert r2.get("thompson") is True, r2
        assert thompson.calls == 1, (
            f"double-delivery: Thompson called {thompson.calls} times")
        assert r2.get("checkpoint_advanced") is False, r2

        # 3. snapshot heals: the retry completes WITHOUT a new delivery
        thompson.fail_state = False
        r3 = json.loads(service._process_request(_outcome_request("task-idem-1")))
        assert r3.get("ok") is True, r3
        assert thompson.calls == 1, thompson.calls

        # 4. full replay of a processed task: duplicate, no delivery
        r4 = json.loads(service._process_request(_outcome_request("task-idem-1")))
        assert r4.get("ok") is True and r4.get("duplicate") is True, r4
        assert thompson.calls == 1, thompson.calls

        # 5. reset_learning clears the thompson_applied markers too
        assert service.system.reset_learning() is True
        assert not service.system.thompson_was_applied("task-idem-1")

        # 6. the per-task lock structure is bounded by construction
        #    (fixed stripe array, no per-task entries to leak)
        stripes = mod.LearningService.TASK_LOCK_STRIPES
        assert len(service._task_lock_stripes) == stripes
        for i in range(stripes + 500):
            lock = service._task_ingestion_lock(f"lock-probe-{i}")
            assert lock in service._task_lock_stripes
        assert len(service._task_lock_stripes) == stripes

        # 7. reset waits for in-flight ingestion: it must block while an
        #    ingestion stripe is held and complete only after release
        import threading
        stripe = service._task_ingestion_lock("task-idem-2")
        stripe.acquire()
        reset_done = threading.Event()

        def _do_reset():
            r = json.loads(service._process_request(
                json.dumps({"op": "reset_learning"})))
            reset_done.set()

        worker = threading.Thread(target=_do_reset)
        worker.start()
        assert not reset_done.wait(timeout=1.0), \
            "reset completed while an ingestion stripe was held"
        stripe.release()
        assert reset_done.wait(timeout=10.0), \
            "reset did not complete after stripe release"
        worker.join(timeout=10.0)

        print("PASS: Thompson idempotent redelivery, reset clears markers, "
              "lock map bounded")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: Thompson idempotency test: {exc}", file=sys.stderr)
        raise
