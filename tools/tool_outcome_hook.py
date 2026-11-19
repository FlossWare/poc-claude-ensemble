#!/usr/bin/env python3
"""PostToolUse hook: journal one metadata-only tool-completion row.

Claude Code invokes this with the hook payload on stdin.  The row records the
tool name, whether it succeeded, the task type, and the session/tool ids --
never the tool input, output, or any file content.  It appends to
``~/.FlossWare/ledger/tools/tools.jsonl`` (override with
``POC_CLAUDE_ENSEMBLE_TOOL_LEDGER``), mode 0600.

The meta-learning ``tool_selection`` domain reads this ledger to compare tool
success rates per task type.  Instrumentation only: nothing is tuned from it
yet.

A hook must never block Claude Code: every failure exits 0 with a short
message on stderr.

Install (user settings, not done by this repo):

  {"hooks": {"PostToolUse": [{"matcher": "*",
     "hooks": [{"type": "command",
                "command": "python3 /path/to/tools/tool_outcome_hook.py"}]}]}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_LEDGER = "~/.FlossWare/ledger/tools/tools.jsonl"
LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_TOOL_LEDGER"
TASK_TYPE_ENV = "POC_CLAUDE_ENSEMBLE_TASK_TYPE"
MAX_ROW_BYTES = 16 * 1024


def _success(payload: dict) -> bool:
    """PostToolUseFailure is a failure; otherwise inspect the response shape."""
    if str(payload.get("hook_event_name") or "") == "PostToolUseFailure":
        return False
    response = payload.get("tool_response")
    if isinstance(response, dict):
        if response.get("is_error") is True:
            return False
        if response.get("success") is False:
            return False
        if response.get("error"):
            return False
        if response.get("interrupted") is True:
            return False
    return True


def build_row(payload: dict, task_type: str) -> dict:
    tool = payload.get("tool_name") or payload.get("tool") or "unknown"
    row = {
        "schema_version": 1,
        "producer": "tool-outcome-hook",
        "kind": "tool_completion",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": str(payload.get("session_id") or "")[:128],
        "tool": str(tool)[:64],
        "tool_use_id": str(payload.get("tool_use_id") or "")[:128],
        "task_type": str(task_type or "unknown")[:64],
        "success": _success(payload),
    }
    duration = payload.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
        row["duration_ms"] = round(float(duration), 3)
    return row


def append_row(row: dict, ledger: Path) -> bool:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if len(line.encode("utf-8")) > MAX_ROW_BYTES:
        return False
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(ledger.parent, 0o700)
    except OSError:
        pass
    fd = os.open(ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (line + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Journal one tool completion (metadata only).")
    parser.add_argument("--ledger", default=os.environ.get(LEDGER_ENV, DEFAULT_LEDGER))
    parser.add_argument("--task-type", default=os.environ.get(TASK_TYPE_ENV, "unknown"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - a hook must never block the tool call
        if not args.quiet:
            print(f"tool outcome hook: ignored unreadable input ({type(exc).__name__})",
                  file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        if not args.quiet:
            print("tool outcome hook: ignored non-object input", file=sys.stderr)
        return 0

    try:
        row = build_row(payload, args.task_type)
        if not append_row(row, Path(args.ledger).expanduser()):
            if not args.quiet:
                print("tool outcome hook: row dropped (oversize)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        if not args.quiet:
            print(f"tool outcome hook: journal write failed ({type(exc).__name__})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
