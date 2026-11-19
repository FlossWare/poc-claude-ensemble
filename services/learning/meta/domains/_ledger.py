"""Read-only ledger helpers shared by the meta-learning domain adapters.

Everything here tolerates a missing or partly corrupt ledger: a backfill must
be able to report "no evidence yet" rather than crash.  Only identifier,
numeric, and enum fields are read -- never prompt or review text, except the
arbiter's machine-readable ``VERDICT:``/``SELECTED:`` lines that the existing
tools already parse.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

VERDICTS = ("approve with fixes", "approve", "reject")

#: A backfill never loads more than this per ledger file.  Ledgers are local
#: and append-only; the cap only protects against a runaway file.
MAX_LEDGER_BYTES = 64 * 1024 * 1024


def expand(value: Optional[str], default: str) -> Path:
    return Path(os.path.expanduser(value or default))


def reviews_ledger() -> Path:
    return expand(os.environ.get("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"),
                  "~/.FlossWare/ledger/reviews/reviews.jsonl")


def crews_dir() -> Path:
    return expand(os.environ.get("POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR"),
                  "~/.FlossWare/ledger/crews")


def ops_dir() -> Path:
    return expand(os.environ.get("POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR"),
                  "~/.FlossWare/ledger/ops")


def bulk_ingest_dir() -> Path:
    return expand(os.environ.get("POC_CLAUDE_ENSEMBLE_BULK_LEDGER_DIR"),
                  "~/.FlossWare/ledger/bulk-ingest")


def rag_journal() -> Optional[Path]:
    configured = os.environ.get("FLOSSWARE_CE_RAG_JOURNAL")
    if configured and configured.strip().lower() == "off":
        return None
    return expand(configured, "~/.FlossWare/ledger/rag/injections.jsonl")


def tool_ledger() -> Path:
    return expand(os.environ.get("POC_CLAUDE_ENSEMBLE_TOOL_LEDGER"),
                  "~/.FlossWare/ledger/tools/tools.jsonl")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL file, silently skipping blank and corrupt lines."""
    rows: List[Dict[str, Any]] = []
    consumed = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                consumed += len(line)
                if consumed > MAX_LEDGER_BYTES:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def review_groups(rows: Iterable[Dict[str, Any]]) -> "Dict[str, List[Dict[str, Any]]]":
    """Group review-ledger rows by review_id, preserving ledger order."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        review_id = row.get("review_id")
        if isinstance(review_id, str) and review_id:
            groups.setdefault(review_id, []).append(row)
    return groups


def subject_of(group: List[Dict[str, Any]]) -> Optional[str]:
    for row in group:
        subject = row.get("subject_sha256")
        if isinstance(subject, str) and subject:
            return subject
    return None


def subjects(groups: "Dict[str, List[Dict[str, Any]]]") -> Dict[str, List[str]]:
    """subject_sha256 -> review_ids in first-seen ledger order."""
    mapping: Dict[str, List[str]] = {}
    for review_id, rows in groups.items():
        subject = subject_of(rows)
        if subject:
            mapping.setdefault(subject, []).append(review_id)
    return mapping


def verdict_of(text: str) -> str:
    """The verdict on the last ``VERDICT:`` line, or ``unclear``."""
    for line in reversed((text or "").strip().splitlines()):
        if line.strip().upper().startswith("VERDICT:"):
            value = line.split(":", 1)[1].strip().lower().rstrip(".")
            return value if value in VERDICTS else "unclear"
    return "unclear"


def selected_of(text: str) -> Optional[str]:
    """The label on the last ``SELECTED:`` line, or None."""
    for line in reversed((text or "").strip().splitlines()):
        if line.strip().upper().startswith("SELECTED:"):
            return line.split(":", 1)[1].strip() or None
    return None


def rows_of_kind(group: Iterable[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [row for row in group if row.get("kind") == kind]


def total_cost(group: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for row in group:
        value = row.get("cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return round(total, 8)


def worker_models(group: Iterable[Dict[str, Any]]) -> List[str]:
    """Distinct worker models that actually ran in a review, in ledger order."""
    seen: List[str] = []
    for row in group:
        if row.get("kind") != "worker_result":
            continue
        model = row.get("model")
        if isinstance(model, str) and model and model not in seen:
            seen.append(model)
    return seen


def arbiter_rows(group: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = rows_of_kind(group, "arbiter_decision")
    rows += rows_of_kind(group, "arbiter_verification")
    return rows


def review_depth(group: Iterable[Dict[str, Any]]) -> int:
    """Phases executed for a review.

    New rows carry ``review_depth``.  Historical rows are inferred: no arbiter
    decision is depth 1; an arbiter decision is depth 2; an arbiter
    verification pass makes depth 3.
    """
    rows = list(group)
    for row in rows:
        depth = row.get("review_depth")
        if isinstance(depth, int) and not isinstance(depth, bool) and depth in (1, 2, 3):
            return depth
    if rows_of_kind(rows, "arbiter_verification"):
        return 3
    if rows_of_kind(rows, "arbiter_decision"):
        return 2
    return 1


def final_arbiter_row(group: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = arbiter_rows(group)
    return rows[-1] if rows else None


def parse_ts(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def stable_id(*parts: Any) -> str:
    """Deterministic outcome id for idempotent backfills."""
    digest = hashlib.sha256("\u0000".join(str(part) for part in parts).encode("utf-8"))
    return digest.hexdigest()[:32]


def nearest_arm(value: float, arms: Iterable[str], *, floor: bool = False) -> str:
    """Nearest (or largest arm not above) a numeric observation."""
    numeric = sorted((float(arm), arm) for arm in arms)
    if floor:
        eligible = [item for item in numeric if item[0] <= value]
        if eligible:
            return eligible[-1][1]
        return numeric[0][1]
    return min(numeric, key=lambda item: (abs(item[0] - value), item[0]))[1]
