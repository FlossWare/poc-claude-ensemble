"""Event-driven GA trigger in tools/ledger_to_thompson.py (user-approved amendment)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "ledger_to_thompson.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("ledger_to_thompson_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_is_four_rows(tool):
    assert tool.GA_TRIGGER_MIN_ROWS == 4


def test_below_threshold_does_not_spawn(tool):
    calls = []

    def fake_popen(*args, **kwargs):  # pragma: no cover - must not be called
        calls.append((args, kwargs))

    assert tool._maybe_trigger_tuning(0, popen=fake_popen) is None
    assert tool._maybe_trigger_tuning(3, popen=fake_popen) is None
    assert calls == []


def test_threshold_spawns_a_detached_fixed_argv_run(tool):
    calls = []

    class FakeProcess:
        pid = 4321

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeProcess()

    process = tool._maybe_trigger_tuning(4, popen=fake_popen)
    assert process is not None and process.pid == 4321
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        sys.executable,
        "-m",
        "learning.ga",
        "run",
        "--target",
        "thompson_policy",
        "--report-only",
    ]
    assert kwargs["start_new_session"] is True
    assert "shell" not in kwargs
    assert kwargs["cwd"] == tool.REPO
    pythonpath = kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == os.path.join(tool.REPO, "services")


def test_spawn_failure_is_swallowed(tool):
    def exploding_popen(*_args, **_kwargs):
        raise OSError("no processes today")

    assert tool._maybe_trigger_tuning(9, popen=exploding_popen) is None


def test_sync_triggers_only_for_live_runs(tool, tmp_path, monkeypatch):
    ledger = tmp_path / "reviews.jsonl"
    rows = [
        fx.review_row("review-1", f"model-{index}", True, 0.1, fx.EPOCH)
        for index in range(4)
    ]
    fx.write_jsonl(ledger, rows)
    monkeypatch.setattr(tool, "LEDGER", str(ledger))
    monkeypatch.setattr(tool, "STATE", str(tmp_path / "thompson-sync-state.json"))

    class FakeClient:
        def record_outcome(self, **_kwargs):
            return True

    monkeypatch.setattr(tool, "ThompsonClient", FakeClient)
    triggered = []
    monkeypatch.setattr(tool, "_maybe_trigger_tuning", lambda recorded: triggered.append(recorded))

    assert tool.sync() == 4
    assert triggered == [4]

    # Dry runs never spawn, and neither does a sync with nothing new.
    second = fx.review_row("review-2", "model-x", True, 0.1, fx.EPOCH + 1)
    fx.write_jsonl(ledger, rows + [second])
    assert tool.sync(dry_run=True) == 1
    assert triggered == [4]
    assert tool.sync() == 1
    assert triggered == [4, 1]
