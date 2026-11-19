"""tool_selection: tool completions per tool and task type (signal capture).

``tools/tool_outcome_hook.py`` is a Claude Code ``PostToolUse`` hook that
appends one metadata-only row per tool completion to
``~/.FlossWare/ledger/tools/tools.jsonl``.  This domain turns those rows into
per-tool success/failure evidence per task type.  Instrumentation-only: no
tool choice is drawn from it yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_label

# Claude Code tool names -> stable arm labels.
TOOL_ALIASES = {
    "bash": "bash",
    "read": "read",
    "edit": "edit",
    "multiedit": "edit",
    "write": "write",
    "notebookedit": "write",
    "grep": "grep",
    "glob": "glob",
    "task": "task",
    "agent": "task",
    "webfetch": "webfetch",
    "websearch": "webfetch",
}


class ToolSelection(Domain):
    name = "tool_selection"
    arms = ("bash", "read", "edit", "write", "grep", "glob", "task", "webfetch", "other")
    default_arm = "other"
    description = "Tool used for a task type (instrumented, not yet tuned)"
    instrumentation = True
    outcome_source = "tool outcome hook ledger: success/failure per tool per task type"
    value_kind = "label"

    @classmethod
    def normalize(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "other"
        key = value.strip().lower()
        return coerce_label(TOOL_ALIASES.get(key, "other"), cls.arms, what="tool")


def derive_outcomes(ledger_path=None) -> List[Dict[str, Any]]:
    path = ledger_path or _ledger.tool_ledger()
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(_ledger.read_jsonl(path)):
        arm = ToolSelection.normalize(row.get("tool") or row.get("tool_name"))
        success = row.get("success")
        if not isinstance(success, bool):
            continue
        task_type = row.get("task_type") if isinstance(row.get("task_type"), str) else "unknown"
        records.append({
            "arm": arm,
            "outcome": success,
            "cost": 0.0,
            "context": {"task_type": task_type},
            "metadata": {
                "outcome_id": row.get("outcome_id") or _ledger.stable_id(
                    "tool_selection", index, row.get("timestamp"), row.get("tool_use_id")),
                "tool": row.get("tool") or row.get("tool_name"),
                "task_type": task_type,
                "session_id": row.get("session_id"),
            },
        })
    return records


def backfill(learner: Any, *, ledger_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, ToolSelection.name, derive_outcomes(ledger_path), limit=limit)


__all__ = ["TOOL_ALIASES", "ToolSelection", "backfill", "derive_outcomes"]
