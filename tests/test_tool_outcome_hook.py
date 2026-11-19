#!/usr/bin/env python3
"""Tests for tools/tool_outcome_hook.py (metadata-only tool completion journal)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "tools" / "tool_outcome_hook.py"


def run_hook(payload, ledger, *extra):
    return subprocess.run(
        [sys.executable, str(HOOK), "--ledger", str(ledger), "--quiet", *extra],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True, text=True, check=False, timeout=30,
    )


def read_rows(ledger):
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]


def test_successful_tool_completion_is_journalled(tmp_path):
    ledger = tmp_path / "tools" / "tools.jsonl"
    result = run_hook({
        "hook_event_name": "PostToolUse", "session_id": "s1", "tool_name": "Bash",
        "tool_use_id": "t1", "tool_input": {"command": "secret"}, "tool_response": {"ok": True},
    }, ledger, "--task-type", "testing")
    assert result.returncode == 0, result.stderr
    rows = read_rows(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["tool"] == "Bash"
    assert row["success"] is True
    assert row["task_type"] == "testing"
    assert row["tool_use_id"] == "t1"
    # Metadata only: never the tool input or output.
    assert "secret" not in ledger.read_text(encoding="utf-8")
    assert "command" not in row
    assert stat.S_IMODE(os.stat(ledger).st_mode) == 0o600


def test_error_response_and_failure_event_mark_failure(tmp_path):
    ledger = tmp_path / "tools.jsonl"
    assert run_hook({"tool_name": "Read", "tool_response": {"is_error": True}}, ledger).returncode == 0
    assert run_hook({"tool_name": "Edit", "hook_event_name": "PostToolUseFailure"}, ledger).returncode == 0
    rows = read_rows(ledger)
    assert [row["success"] for row in rows] == [False, False]


def test_unreadable_input_is_ignored_without_writing(tmp_path):
    ledger = tmp_path / "tools.jsonl"
    result = run_hook("not json", ledger)
    assert result.returncode == 0
    assert not ledger.exists()


def test_json_output_has_no_free_text_fields(tmp_path):
    ledger = tmp_path / "tools.jsonl"
    run_hook({"tool_name": "Grep", "tool_response": {"stdout": "very secret output"}}, ledger)
    row = read_rows(ledger)[0]
    assert set(row) <= {"schema_version", "producer", "kind", "timestamp", "session_id",
                        "tool", "tool_use_id", "task_type", "success", "duration_ms"}
    assert row["success"] is True
