"""PR 5 resume/retry primitives: checkpoints, fingerprints and bounded retries."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import resume


def _checkpoint(entry_id: str = "apd:Test/a.yml", **overrides) -> resume.Checkpoint:
    values = dict(
        stage="integrate",
        entry_id=entry_id,
        fingerprint="f" * 64,
        status="indexed",
        reason_code=None,
        retryable=False,
        attempts=1,
        updated_at="2026-01-01T00:00:00Z",
        result={"status": "indexed"},
    )
    values.update(overrides)
    return resume.Checkpoint(**values)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_order_independent():
    first = resume.fingerprint("integrate", entry_id="a", inputs={"x": 1, "y": [2, 3]})
    second = resume.fingerprint("integrate", entry_id="a", inputs={"y": [2, 3], "x": 1})
    assert first == second
    assert len(first) == 64


def test_fingerprint_changes_with_any_input():
    base = resume.fingerprint("integrate", entry_id="a", inputs={"sha": "1"})
    assert base != resume.fingerprint("integrate", entry_id="a", inputs={"sha": "2"})
    assert base != resume.fingerprint("integrate", entry_id="b", inputs={"sha": "1"})
    assert base != resume.fingerprint("graph", entry_id="a", inputs={"sha": "1"})


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_retry_policy_bounds_attempts_and_backoff():
    policy = resume.RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=0.75)
    assert policy.max_attempts == 3
    assert policy.delay_for(1) == 0.5
    assert policy.delay_for(2) == 0.75  # capped
    assert policy.should_retry(retryable=True, attempts=1) is True
    assert policy.should_retry(retryable=True, attempts=3) is False
    assert policy.should_retry(retryable=False, attempts=1) is False


def test_retry_policy_clamps_absurd_values():
    policy = resume.RetryPolicy(max_attempts=0, base_delay_seconds=-1, max_delay_seconds=10_000)
    assert policy.max_attempts == 1
    assert policy.base_delay_seconds == 0.0
    assert policy.max_delay_seconds == 300.0
    assert policy.max_total_attempts >= policy.max_attempts


def test_next_retry_at_is_in_the_future():
    policy = resume.RetryPolicy(base_delay_seconds=1, max_delay_seconds=1)
    assert policy.next_retry_at(1) > resume.utc_now()


class _Result:
    def __init__(self, status: str, retryable: bool) -> None:
        self.status = status
        self.retryable = retryable


def test_with_retries_returns_first_success():
    calls = []
    attempt = resume.with_retries(
        lambda: calls.append(1) or _Result("indexed", False),
        policy=resume.RetryPolicy(max_attempts=3),
        retryable=lambda result: result.retryable,
        sleep=lambda delay: pytest.fail("must not sleep on success"),
    )
    assert attempt.attempts == 1
    assert attempt.retried is False
    assert len(calls) == 1


def test_with_retries_stops_at_the_budget():
    delays: list[float] = []
    attempt = resume.with_retries(
        lambda: _Result("deferred", True),
        policy=resume.RetryPolicy(max_attempts=3, base_delay_seconds=0.1, max_delay_seconds=0.2),
        retryable=lambda result: result.retryable,
        sleep=delays.append,
    )
    assert attempt.attempts == 3
    assert attempt.retried is True
    assert delays == [0.1, 0.2]
    assert attempt.value.status == "deferred"


def test_with_retries_does_not_retry_a_terminal_failure():
    attempt = resume.with_retries(
        lambda: _Result("failed", False),
        policy=resume.RetryPolicy(max_attempts=5),
        retryable=lambda result: result.retryable,
        sleep=lambda delay: pytest.fail("must not sleep for a terminal failure"),
    )
    assert attempt.attempts == 1


# ---------------------------------------------------------------------------
# Checkpoint log
# ---------------------------------------------------------------------------


def test_checkpoint_is_durable_immediately(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record(_checkpoint())
    # A brand-new instance (as a restarted process would create) sees the row
    # without any explicit flush: record() appends and fsyncs.
    reloaded = resume.CheckpointLog(tmp_path, "integrate")
    assert reloaded.load() == 1
    stored = reloaded.get("apd:Test/a.yml")
    assert stored is not None
    assert stored.status == "indexed"


def test_later_rows_win(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record(_checkpoint(status="deferred", retryable=True, attempts=1))
    log.record(_checkpoint(status="indexed", attempts=2))
    reloaded = resume.CheckpointLog(tmp_path, "integrate")
    reloaded.load()
    stored = reloaded.get("apd:Test/a.yml")
    assert stored is not None
    assert stored.status == "indexed"
    assert stored.attempts == 2


def test_torn_final_line_is_ignored(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record(_checkpoint())
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"stage":"integrate","entry_id":"apd:Test/b.yml"')  # killed mid-write
    reloaded = resume.CheckpointLog(tmp_path, "integrate")
    assert reloaded.load() == 1
    assert reloaded.stats()["ignored_lines"] == 1
    assert reloaded.get("apd:Test/a.yml") is not None


def test_rows_from_another_stage_are_ignored(tmp_path):
    graph_log = resume.CheckpointLog(tmp_path, "graph")
    graph_log.record(_checkpoint(stage="graph"))
    integrate_log = resume.CheckpointLog(tmp_path, "integrate")
    assert integrate_log.load() == 0


def test_disabled_log_writes_nothing(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate", enabled=False)
    log.record(_checkpoint())
    assert not log.path.exists()
    assert log.decide("apd:Test/a.yml", "f" * 64, policy=resume.RetryPolicy()).reuse is False


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def _decide(tmp_path, checkpoint, digest="f" * 64, policy=None):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record(checkpoint)
    log.load()
    return log.decide(checkpoint.entry_id, digest, policy=policy or resume.RetryPolicy())


def test_fresh_entry_is_not_reused(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.load()
    decision = log.decide("apd:Test/a.yml", "f" * 64, policy=resume.RetryPolicy())
    assert decision.reuse is False
    assert decision.reason == "fresh"


def test_recorded_success_is_reused_only_for_the_same_inputs(tmp_path):
    assert _decide(tmp_path, _checkpoint()).reuse is True
    changed = _decide(tmp_path / "other", _checkpoint(), digest="e" * 64)
    assert changed.reuse is False
    assert changed.reason == "fingerprint_changed"


def test_terminal_failure_is_not_retried(tmp_path):
    decision = _decide(
        tmp_path, _checkpoint(status="failed", retryable=False, reason_code="parse_error")
    )
    assert decision.reuse is True
    assert decision.reason == "resume_not_retryable"


def test_retryable_failure_is_retried_until_the_lifetime_budget(tmp_path):
    retryable = _checkpoint(
        status="deferred", retryable=True, reason_code="dependency_unavailable", attempts=3
    )
    decision = _decide(tmp_path, retryable)
    assert decision.reuse is False
    assert decision.reason == "retry"

    spent = _decide(
        tmp_path / "spent",
        _checkpoint(
            status="deferred",
            retryable=True,
            reason_code="dependency_unavailable",
            attempts=12,
        ),
    )
    assert spent.reuse is True
    assert spent.exhausted is True
    assert spent.reason == resume.RETRY_EXHAUSTED_REASON


def test_attempts_accumulate_across_runs(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record_result(
        "apd:Test/a.yml",
        "f" * 64,
        status="deferred",
        reason_code="dependency_unavailable",
        retryable=True,
        attempts=2,
    )
    stored = log.record_result(
        "apd:Test/a.yml",
        "f" * 64,
        status="deferred",
        reason_code="dependency_unavailable",
        retryable=True,
        attempts=1,
        prior_attempts=2,
    )
    assert stored.attempts == 3
    reloaded = resume.CheckpointLog(tmp_path, "integrate")
    reloaded.load()
    assert reloaded.get("apd:Test/a.yml").attempts == 3


def test_stats_report_the_recorded_statuses(tmp_path):
    log = resume.CheckpointLog(tmp_path, "integrate")
    log.record(_checkpoint("apd:Test/a.yml"))
    log.record(_checkpoint("apd:Test/b.yml", status="skipped", reason_code="source_tombstoned"))
    stats = log.stats()
    assert stats["entries"] == 2
    assert stats["by_status"] == {"indexed": 1, "skipped": 1}
    assert json.loads(json.dumps(stats))["stage"] == "integrate"
