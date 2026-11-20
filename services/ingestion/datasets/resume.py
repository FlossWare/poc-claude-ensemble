"""PR 5: crash-safe progress checkpoints and bounded retries.

The pipeline stages before this one already keep durable *content*: PR 3 has a
content-addressed blob store with resumable ``.part`` files, and PR 4 has a
completion journal keyed by revision id.  What was missing is a durable record
of **per-entry completion**.  Manifests are written at the end of a run, so a
process killed mid-run used to lose every entry it had already finished even
though the remote work was done.

This module adds that record:

* :class:`CheckpointLog` is an append-only, fsynced JSONL file per stage.  One
  line is appended the moment an entry reaches a terminal state, so a SIGKILL
  (or a power loss) loses at most the entry that was in flight.
* Every line carries a *fingerprint* of the inputs that produced it.  A resume
  reuses a recorded result only when the fingerprint still matches, so changed
  metadata, changed acquisition evidence or changed stage versions force
  re-processing instead of silently replaying a stale result.
* :class:`RetryPolicy` bounds how many times an entry is attempted.  Only
  outcomes explicitly marked retryable are retried, with exponential backoff
  capped by the policy, and the number of attempts is carried across runs so a
  repeatedly failing entry cannot be retried forever.

Checkpoints are rebuildable artifacts, exactly like the manifests: deleting one
never loses ingested data, it only costs redundant (idempotent) work.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Generic, TypeVar

from . import model

CHECKPOINT_VERSION = 1
DEFAULT_MAX_ATTEMPTS = 3
#: Lifetime attempt budget: four runs at the default per-run budget.
DEFAULT_MAX_TOTAL_ATTEMPTS = 12

#: Statuses that mean this stage is finished successfully for the entry.
SUCCESS_STATUSES = frozenset({"indexed", "metadata_indexed", "unchanged"})
#: Statuses that are finished and should not be retried without new input.
TERMINAL_FAILURE_STATUSES = frozenset({"skipped", "failed"})

#: Reason recorded when a retryable entry has used its whole attempt budget.
RETRY_EXHAUSTED_REASON = "retry_budget_exhausted"

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fingerprint(stage: str, *, entry_id: str, inputs: dict[str, object]) -> str:
    """Stable hash of everything that would change this stage's result."""
    return model.sha256_hex(
        model.canonical_json(
            {
                "schema_version": CHECKPOINT_VERSION,
                "stage": stage,
                "entry_id": entry_id,
                "inputs": inputs,
            }
        )
    )


# ---------------------------------------------------------------------------
# Bounded retries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts an entry gets, and how long to wait between them.

    ``max_attempts`` bounds one run: a transient dependency failure heals on the
    next run, which is what the existing pipeline guarantees.  ``max_total_attempts``
    bounds the *lifetime* of the entry across runs, so a permanently broken
    entry cannot be retried forever; when it is reached the entry is reported
    ``deferred`` with ``retry_budget_exhausted``.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0
    max_total_attempts: int = DEFAULT_MAX_TOTAL_ATTEMPTS

    def __post_init__(self) -> None:
        # Bounded on both ends: at least one attempt, and a sane ceiling so a
        # misconfigured environment cannot stall a run.
        object.__setattr__(self, "max_attempts", max(1, min(int(self.max_attempts), 10)))
        object.__setattr__(
            self, "base_delay_seconds", max(0.0, min(float(self.base_delay_seconds), 60.0))
        )
        object.__setattr__(
            self, "max_delay_seconds", max(0.0, min(float(self.max_delay_seconds), 300.0))
        )
        total = max(int(self.max_total_attempts), int(self.max_attempts))
        object.__setattr__(self, "max_total_attempts", max(1, min(total, 100)))

    def delay_for(self, attempt: int) -> float:
        """Exponential backoff after ``attempt`` failed attempts."""
        if attempt < 1:
            return 0.0
        delay = self.base_delay_seconds * (2 ** (attempt - 1))
        return min(delay, self.max_delay_seconds)

    def should_retry(self, *, retryable: bool, attempts: int) -> bool:
        return bool(retryable) and attempts < self.max_attempts

    def next_retry_at(self, attempts: int, *, now: datetime | None = None) -> str | None:
        delay = self.delay_for(max(1, attempts))
        moment = (now or datetime.now(timezone.utc)) + timedelta(seconds=delay)
        return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AttemptResult(Generic[T]):
    """The winning result plus what it cost to get it."""

    value: T
    attempts: int
    delays: tuple[float, ...] = ()

    @property
    def retried(self) -> bool:
        return self.attempts > 1


def with_retries(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    retryable: Callable[[T], bool],
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, T, float], None] | None = None,
) -> AttemptResult[T]:
    """Run ``operation`` until it succeeds or the bounded budget is spent.

    ``operation`` is required to be idempotent: the caller's operations are
    Knowledge revision writes, deterministic file writes and deterministic
    graph ids, all of which are no-ops when repeated.  Raised exceptions are
    *not* retried here; the stage converts transient failures into results with
    an explicit ``retryable`` flag so the reason code stays meaningful.
    """
    delays: list[float] = []
    attempts = 0
    while True:
        attempts += 1
        result = operation()
        if not policy.should_retry(retryable=retryable(result), attempts=attempts):
            return AttemptResult(value=result, attempts=attempts, delays=tuple(delays))
        delay = policy.delay_for(attempts)
        if on_retry is not None:
            on_retry(attempts, result, delay)
        delays.append(delay)
        if delay > 0:
            sleep(delay)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """One entry's terminal state for one stage, with its input fingerprint."""

    stage: str
    entry_id: str
    fingerprint: str
    status: str
    reason_code: str | None
    retryable: bool
    attempts: int
    updated_at: str
    detail: str | None = None
    result: dict[str, object] | None = None

    def to_row(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_VERSION,
            "stage": self.stage,
            "entry_id": self.entry_id,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "reason_code": self.reason_code,
            "retryable": bool(self.retryable),
            "attempts": int(self.attempts),
            "updated_at": self.updated_at,
            "detail": self.detail[:300] if self.detail else None,
            "result": self.result,
        }

    @classmethod
    def from_row(cls, row: object) -> "Checkpoint | None":
        if not isinstance(row, dict):
            return None
        stage = row.get("stage")
        entry_id = row.get("entry_id")
        digest = row.get("fingerprint")
        status = row.get("status")
        if not all(isinstance(value, str) and value for value in (stage, entry_id, digest, status)):
            return None
        attempts = row.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            attempts = 0
        result = row.get("result")
        detail = row.get("detail")
        return cls(
            stage=str(stage),
            entry_id=str(entry_id),
            fingerprint=str(digest),
            status=str(status),
            reason_code=row.get("reason_code") if isinstance(row.get("reason_code"), str) else None,
            retryable=bool(row.get("retryable")),
            attempts=attempts,
            updated_at=str(row.get("updated_at") or ""),
            detail=detail if isinstance(detail, str) else None,
            result=result if isinstance(result, dict) else None,
        )


@dataclass(frozen=True)
class ResumeDecision:
    """Whether a checkpoint may be trusted for the current inputs."""

    reuse: bool
    reason: str
    checkpoint: Checkpoint | None = None

    @property
    def exhausted(self) -> bool:
        return self.reason == RETRY_EXHAUSTED_REASON


class CheckpointLog:
    """Append-only, fsynced per-entry progress for one stage.

    The file lives under ``cache/checkpoints`` in the dataset data root.  Later
    rows win: a re-processed entry simply appends a new line.  A torn final line
    (a kill during ``write``) is ignored rather than treated as corruption.
    """

    def __init__(
        self,
        root: Path | str,
        stage: str,
        *,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.stage = stage
        self.enabled = enabled
        self._records: dict[str, Checkpoint] = {}
        self._loaded = False
        self._skipped_lines = 0
        self._writes = 0

    @property
    def path(self) -> Path:
        return self.root / f"{self.stage}.jsonl"

    # -- reading -----------------------------------------------------------
    def load(self) -> int:
        """Read every row; malformed or torn lines are counted and ignored."""
        self._records = {}
        self._skipped_lines = 0
        self._loaded = True
        try:
            handle = self.path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            return 0
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    self._skipped_lines += 1
                    continue
                checkpoint = Checkpoint.from_row(row)
                if checkpoint is None or checkpoint.stage != self.stage:
                    self._skipped_lines += 1
                    continue
                self._records[checkpoint.entry_id] = checkpoint
        return len(self._records)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, entry_id: str) -> Checkpoint | None:
        self._ensure_loaded()
        return self._records.get(entry_id)

    def entry_ids(self) -> set[str]:
        self._ensure_loaded()
        return set(self._records)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._records)

    def stats(self) -> dict[str, object]:
        self._ensure_loaded()
        by_status: dict[str, int] = {}
        for checkpoint in self._records.values():
            by_status[checkpoint.status] = by_status.get(checkpoint.status, 0) + 1
        return {
            "enabled": self.enabled,
            "stage": self.stage,
            "path": str(self.path),
            "entries": len(self._records),
            "by_status": dict(sorted(by_status.items())),
            "ignored_lines": self._skipped_lines,
            "appended": self._writes,
        }

    # -- deciding ----------------------------------------------------------
    def decide(
        self,
        entry_id: str,
        digest: str,
        *,
        policy: RetryPolicy,
    ) -> ResumeDecision:
        """Decide whether the recorded result still applies."""
        if not self.enabled:
            return ResumeDecision(reuse=False, reason="disabled")
        checkpoint = self.get(entry_id)
        if checkpoint is None:
            return ResumeDecision(reuse=False, reason="fresh")
        if checkpoint.fingerprint != digest:
            # Inputs changed: the old result describes different content.
            return ResumeDecision(reuse=False, reason="fingerprint_changed", checkpoint=checkpoint)
        if checkpoint.status in SUCCESS_STATUSES:
            return ResumeDecision(reuse=True, reason="resume", checkpoint=checkpoint)
        if checkpoint.status in TERMINAL_FAILURE_STATUSES and not checkpoint.retryable:
            return ResumeDecision(reuse=True, reason="resume_not_retryable", checkpoint=checkpoint)
        if checkpoint.attempts >= policy.max_total_attempts:
            # The lifetime budget is spent: stop retrying and say so.
            return ResumeDecision(
                reuse=True, reason=RETRY_EXHAUSTED_REASON, checkpoint=checkpoint
            )
        return ResumeDecision(reuse=False, reason="retry", checkpoint=checkpoint)

    # -- writing -----------------------------------------------------------
    def record(self, checkpoint: Checkpoint) -> None:
        """Append one line and fsync it before returning.

        The fsync is deliberate: the whole point of this file is to survive a
        kill immediately after an entry finishes.
        """
        self.record_row(checkpoint.to_row(), stage=checkpoint.stage)
        self._ensure_loaded()
        self._records[checkpoint.entry_id] = checkpoint

    def record_row(self, row: dict[str, object], *, stage: str | None = None) -> None:
        if not self.enabled:
            return
        if stage is not None and stage != self.stage:
            raise ValueError("checkpoint row belongs to a different stage")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = (model.canonical_json(row) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._writes += 1

    def record_result(
        self,
        entry_id: str,
        digest: str,
        *,
        status: str,
        reason_code: str | None,
        retryable: bool,
        attempts: int,
        detail: str | None = None,
        result: dict[str, object] | None = None,
        prior_attempts: int = 0,
        updated_at: str | None = None,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            stage=self.stage,
            entry_id=entry_id,
            fingerprint=digest,
            status=status,
            reason_code=reason_code,
            retryable=retryable,
            attempts=max(0, prior_attempts) + max(0, attempts),
            updated_at=updated_at or utc_now(),
            detail=detail,
            result=result,
        )
        self.record(checkpoint)
        return checkpoint


def checkpoint_root(config, *, stage_dir: str = "checkpoints") -> Path:
    """Where checkpoints live for this data root."""
    return Path(config.data_root) / "cache" / stage_dir
