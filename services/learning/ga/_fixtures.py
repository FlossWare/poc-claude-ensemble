"""Shared synthetic ledgers and path isolation for the GA tests.

Not collected by pytest (leading underscore).  Everything here writes only
inside the test's ``tmp_path`` and redirects every runtime path so a test can
never touch the live ledger, Thompson state, or policy file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DAY = 86400.0
EPOCH = 1_767_225_600.0  # 2026-01-01T00:00:00Z, a fixed base for deterministic fixtures


def configure_paths(monkeypatch, tmp_path: Path) -> Path:
    """Redirect every GA input/output into ``tmp_path``.  Returns the ledger root."""
    ledger = Path(tmp_path) / "ledger"
    for sub in ("outcomes", "reviews", "selections"):
        (ledger / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_OUTCOME_LEDGER", str(ledger / "outcomes" / "outcomes.jsonl")
    )
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", str(ledger / "reviews" / "reviews.jsonl")
    )
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_SELECTION_LEDGER", str(ledger / "selections" / "selections.jsonl")
    )
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_GA_DIR", str(Path(tmp_path) / "ga"))
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE",
        str(Path(tmp_path) / "policy" / "thompson-policy.json"),
    )
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_THOMPSON_STATE_FILE",
        str(Path(tmp_path) / "state" / "thompson-sampling-state.json"),
    )
    monkeypatch.setenv(
        "POC_CLAUDE_ENSEMBLE_DEPLOYED_SERVICES_DIR", str(Path(tmp_path) / "deployed-services")
    )
    allowed = [str(tmp_path)]
    existing = os.environ.get("POC_CLAUDE_ENSEMBLE_GA_ALLOWED_ROOTS")
    if existing:
        allowed.extend(part for part in existing.split(os.pathsep) if part)
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_GA_ALLOWED_ROOTS", os.pathsep.join(allowed))
    return ledger


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            if isinstance(row, str):
                stream.write(row + "\n")
            else:
                stream.write(json.dumps(row, allow_nan=True) + "\n")
    return path


def iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def outcome_row(
    event_id: str,
    arm: str,
    success: bool,
    cost: float,
    ts: float,
    *,
    status: str = "pending",
    tokens: int = 100,
    kind: str = "turn_outcome",
) -> Dict[str, Any]:
    row = {
        "schema_version": 1,
        "producer": "claude-code",
        "kind": kind,
        "event_id": event_id,
        "status": status,
        "uuid": event_id,
        "arm": arm,
        "model": arm,
        "success": success,
        "signal": "none",
        "cost": cost,
        "tokens": tokens,
        "timestamp": iso(ts),
    }
    if status != "pending":
        for key in ("arm", "success", "cost", "tokens", "timestamp"):
            row.pop(key, None)
    return row


def review_row(
    review_id: str,
    arm: str,
    success: bool,
    cost: float,
    ts: float,
    *,
    producer: str = "multi-model-review",
    kind: str = "worker_result",
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "producer": producer,
        "review_id": review_id,
        "subject": "src/example.py",
        "kind": kind,
        "role": "worker",
        "recorded_at": iso(ts),
        "model": arm,
        "latency_ms": 1000,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": cost,
        "text": "a plausible review body" if success else "",
    }


def tiny_corpora() -> tuple:
    """A 19-round review corpus and a few turn outcomes: refuses for lack of evidence."""
    reviews: List[Dict[str, Any]] = []
    for index in range(19):
        ts = EPOCH + index * DAY
        reviews.append(review_row(f"r{index:02d}", "alpha", True, 0.10, ts))
        reviews.append(review_row(f"r{index:02d}", "beta", index % 3 != 0, 0.05, ts))
    outcomes = [
        outcome_row(f"e{index}", "alpha", True, 0.01, EPOCH + index * DAY)
        for index in range(40)
    ]
    return outcomes, reviews


def passing_turn_rows(days: int = 40, per_day: int = 30) -> List[Dict[str, Any]]:
    """A corpus where the default policy over-picks an excellent, expensive arm.

    ``pricey`` always succeeds and costs 10; ``cheap`` fails one turn in twenty
    and costs 0.001.  Thompson's default (cost_weight 0) chases ``pricey``;
    any candidate with a real cost weight wins on fitness.
    """
    rows: List[Dict[str, Any]] = []
    counter = 0
    for day in range(days):
        for slot in range(per_day):
            arm = "pricey" if counter % 2 == 0 else "cheap"
            if arm == "pricey":
                success, cost = True, 10.0
            else:
                success, cost = (counter % 20 != 0), 0.001
            rows.append(
                outcome_row(
                    f"turn-{counter:05d}",
                    arm,
                    success,
                    cost,
                    EPOCH + day * DAY + slot * 60.0,
                )
            )
            counter += 1
    return rows


def passing_review_rows(rounds: int = 60) -> List[Dict[str, Any]]:
    """A complete-round review corpus (every arm in every round)."""
    rows: List[Dict[str, Any]] = []
    for index in range(rounds):
        ts = EPOCH + index * DAY
        rows.append(review_row(f"review-{index:03d}", "alpha", True, 0.20, ts))
        rows.append(review_row(f"review-{index:03d}", "beta", index % 4 != 0, 0.10, ts))
    return rows


def write_passing_corpora(ledger: Path) -> None:
    write_jsonl(Path(ledger) / "outcomes" / "outcomes.jsonl", passing_turn_rows())
    write_jsonl(Path(ledger) / "reviews" / "reviews.jsonl", passing_review_rows())


def write_tiny_corpora(ledger: Path) -> None:
    outcomes, reviews = tiny_corpora()
    write_jsonl(Path(ledger) / "outcomes" / "outcomes.jsonl", outcomes)
    write_jsonl(Path(ledger) / "reviews" / "reviews.jsonl", reviews)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return "<missing>"
    return digest.hexdigest()


def snapshot_files(paths: Iterable[Path]) -> Dict[str, tuple]:
    result = {}
    for path in paths:
        path = Path(path)
        try:
            info = path.stat()
            result[str(path)] = (file_sha256(path), info.st_size, info.st_mtime_ns)
        except FileNotFoundError:
            result[str(path)] = ("<missing>", None, None)
    return result
