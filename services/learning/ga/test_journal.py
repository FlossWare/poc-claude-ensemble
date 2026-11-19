"""Journal: flocked appends, field allowlist, change ids, cooldown, retention."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx
from learning.ga import journal


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def test_append_and_read_roundtrip(isolated):
    stored = journal.append({"kind": "apply", "change_id": "abc123", "target": "thompson_policy"})
    assert stored["schema_version"] == 1
    assert stored["recorded_at"].endswith("Z")
    entries = journal.read_entries()
    assert len(entries) == 1
    assert entries[0]["change_id"] == "abc123"
    assert journal.find_change("abc123")["kind"] == "apply"
    assert journal.already_applied("abc123")
    assert not journal.rolled_back("abc123")


def test_field_allowlist_rejects_untrusted_content(isolated):
    with pytest.raises(journal.JournalError):
        journal.append({"kind": "apply", "text": "model output must never be journaled"})
    with pytest.raises(journal.JournalError):
        journal.append({"kind": "apply", "subject": "src/secret.py"})
    with pytest.raises(journal.JournalError):
        journal.append({"change_id": "no-kind"})
    assert journal.read_entries() == []


def test_error_entry_carries_no_traceback(isolated):
    entry = journal.error_entry("run", ValueError("bad input"))
    assert entry["error_type"] == "ValueError"
    assert entry["error_message"] == "bad input"
    journal.append(entry)
    raw = journal.journal_path().read_text(encoding="utf-8")
    assert "Traceback" not in raw


def test_change_id_is_stable():
    first = journal.change_id("ga-run-1", "deadbeef", "thompson_policy")
    assert first == journal.change_id("ga-run-1", "deadbeef", "thompson_policy")
    assert first != journal.change_id("ga-run-2", "deadbeef", "thompson_policy")
    assert len(first) == 16
    assert all(ch in "0123456789abcdef" for ch in first)


def test_concurrent_appends_are_serialised(isolated):
    errors = []

    def writer(worker: int) -> None:
        try:
            for index in range(10):
                journal.append({"kind": "error", "reason": f"w{worker}-{index}"})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    lines = [line for line in journal.journal_path().read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 40
    for line in lines:
        json.loads(line)


def test_previous_file_encoding_roundtrip(isolated):
    payload = b'{"schema_version": 1}\n'
    fields = journal.encode_previous_file(payload)
    assert fields["previous_file_existed"] is True
    assert journal.read_previous_file(fields) == payload
    assert journal.read_previous_file({"previous_file_b64": None}) is None
    with pytest.raises(journal.JournalError):
        journal.encode_previous_file(b"x" * (journal.MAX_PREVIOUS_FILE_BYTES + 1))


def _apply_entry(recorded_at: str, accepted: int) -> dict:
    return {
        "kind": "apply",
        "change_id": "prev-change",
        "corpus": {"turn": {"accepted": accepted}},
        "recorded_at": recorded_at,
    }


def test_cooldown_needs_both_time_and_new_outcomes(isolated):
    now = time.time()
    ok, detail = journal.cooldown_status(total_outcomes=5000, now=now)
    assert ok and "no previous promotion" in detail

    journal.append(_apply_entry("2026-01-01T00:00:00Z", 1000))
    ok, detail = journal.cooldown_status(total_outcomes=1050, now=now)
    assert not ok

    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    journal.append(
        _apply_entry(fresh.strftime("%Y-%m-%dT%H:%M:%SZ"), 1000)  # newest apply wins
    )
    ok, _detail = journal.cooldown_status(total_outcomes=5000, now=now)
    assert not ok

    old = datetime.now(timezone.utc) - timedelta(days=30)
    journal.append(_apply_entry(old.strftime("%Y-%m-%dT%H:%M:%SZ"), 1000))
    ok, _detail = journal.cooldown_status(total_outcomes=5000, now=now)
    assert ok
    ok, _detail = journal.cooldown_status(total_outcomes=1050, now=now)
    assert not ok


def test_atomic_write_and_prune(isolated):
    target = journal.ga_root() / "file.json"
    journal.atomic_write(target, b"first")
    assert target.read_bytes() == b"first"
    journal.atomic_write(target, b"second")
    assert target.read_bytes() == b"second"
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert oct(journal.ga_root().stat().st_mode & 0o777) == "0o700"

    for index in range(35):
        run = journal.runs_dir() / f"ga-thompson_policy-202601{index:02d}"
        run.mkdir(parents=True)
        (run / "results.json").write_text("{}", encoding="utf-8")
    removed = journal.prune_runs(30)
    assert len(removed) == 5
    assert len([entry for entry in journal.runs_dir().iterdir() if entry.is_dir()]) == 30
    assert not (journal.runs_dir() / "ga-thompson_policy-20260100").exists()
    assert (journal.runs_dir() / "ga-thompson_policy-20260134").exists()


def test_run_id_validation(isolated):
    with pytest.raises(journal.JournalError):
        journal.run_dir("../../escape")
    assert journal.run_dir("ga-thompson_policy-20260101").name == "ga-thompson_policy-20260101"
