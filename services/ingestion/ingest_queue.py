#!/usr/bin/env python3
"""Durable multipass work queue for corpus ingestion.

SQLite-backed work spine coordinating ingestion passes:
  pass 1 (content)  — items → knowledge/vector/lexical stores
  pass 2 (entities) — completed items → entity extraction
  pass 3 (enrichment)
Nightly syncs append incremental changes to the same queue, so the
initial campaign and steady-state increments share one mechanism.

Design constraints:
- Durable: SQLite, WAL; a crash mid-item leaves it in_flight with a
  claimed_at timestamp — stale claims are requeueable by age.
- At-least-once: workers claim, process, then complete/fail. A worker
  crash means the claim expires and the item is requeued (the ingestion
  pipeline itself is idempotent by content hash, so redelivery is safe).
- Priority-ordered: lower number = higher priority; within a priority,
  FIFO by enqueued_at.
- Bounded: claim sizes and total queue rows are capped to keep the
  table small; completed rows older than a retention window are pruned.

This module is deliberately dependency-free (stdlib only).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB = os.environ.get(
    "POC_CLAUDE_ENSEMBLE_INGEST_QUEUE_DB",
    os.path.expanduser("~/.FlossWare/poc-claude-ensemble/data/sqlite/ingest-queue.sqlite3"),
)

# states
PENDING = "pending"
IN_FLIGHT = "in_flight"
DONE = "done"
FAILED = "failed"

MAX_CLAIM = 200          # max items per claim
CLAIM_TIMEOUT_S = 3600    # a claim older than this is stale → requeueable
DONE_RETENTION_S = 7 * 24 * 3600
MAX_QUEUE_ROWS = 2_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pass INTEGER NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    state TEXT NOT NULL DEFAULT 'pending',
    payload TEXT NOT NULL,             -- JSON: {source, path, origin, ...}
    result TEXT,                        -- JSON: completion details
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    enqueued_at REAL NOT NULL,
    claimed_at REAL,
    completed_at REAL,
    claim_token TEXT                    -- opaque token held by the worker
);
CREATE INDEX IF NOT EXISTS idx_queue_drain
    ON queue(state, pass, priority, enqueued_at);
CREATE INDEX IF NOT EXISTS idx_queue_stale
    ON queue(state, claimed_at);
"""


class QueueError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


class IngestQueue:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- api
    def enqueue(self, items: list[dict], *, pass_: int = 1,
                priority: int = 100, max_attempts: int = 3) -> int:
        """Add work items. Each item is a JSON-serializable payload dict.
        Returns the number enqueued."""
        if not isinstance(items, list) or not items:
            raise QueueError("items must be a non-empty list")
        if len(items) > MAX_CLAIM:
            raise QueueError(f"at most {MAX_CLAIM} items per enqueue",
                             413, "batch_too_large")
        now = time.time()
        rows = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise QueueError(f"items[{index}] must be an object")
            try:
                payload = json.dumps(item, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise QueueError(f"items[{index}] is not JSON-serializable: {exc}")
            rows.append((pass_, priority, PENDING, payload, 0, max_attempts, now))
        with self._conn:
            self._conn.executemany(
                "INSERT INTO queue (pass, priority, state, payload, attempts,"
                " max_attempts, enqueued_at) VALUES (?,?,?,?,?,?,?)", rows)
        self._prune()
        return len(rows)

    def claim(self, *, pass_: int, limit: int = 50) -> list[dict]:
        """Atomically claim up to `limit` pending items for a pass.
        Returns [{id, payload, attempts, claim_token}]."""
        if not isinstance(limit, int) or not 1 <= limit <= MAX_CLAIM:
            raise QueueError(f"limit must be 1..{MAX_CLAIM}")
        now = time.time()
        token = f"{os.getpid()}-{now}-{id(object())}"
        with self._conn:
            # requeue stale claims first (same transaction as the claim).
            # NB: a stale claim whose attempts are exhausted goes to
            # failed, NOT back to pending — otherwise a poison item
            # that crashes its worker is reclaimed forever.
            cutoff = now - CLAIM_TIMEOUT_S
            self._conn.execute(
                "UPDATE queue SET state=?, completed_at=?, result=?,"
                " claim_token=NULL WHERE state=? AND claimed_at < ?"
                " AND attempts >= max_attempts",
                (FAILED, now, json.dumps({"error": "stale claim; attempts exhausted"}),
                 IN_FLIGHT, cutoff))
            self._conn.execute(
                "UPDATE queue SET state=?, claimed_at=NULL, claim_token=NULL"
                " WHERE state=? AND claimed_at < ?",
                (PENDING, IN_FLIGHT, cutoff))
            rows = self._conn.execute(
                "SELECT id, payload, attempts FROM queue"
                " WHERE state=? AND pass=? ORDER BY priority, enqueued_at, id"
                " LIMIT ?",
                (PENDING, pass_, limit)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                marks = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE queue SET state=?, claimed_at=?, claim_token=?,"
                    f" attempts=attempts+1 WHERE id IN ({marks})",
                    [IN_FLIGHT, now, token] + ids)
        return [{"id": r["id"], "payload": json.loads(r["payload"]),
                 "attempts": r["attempts"] + 1, "claim_token": token}
                for r in rows]

    def complete(self, *, ids: list[int], claim_token: str,
                 result: Optional[dict] = None) -> int:
        """Mark claimed items done. Only the holding token may complete."""
        if not isinstance(ids, list) or not ids or len(ids) > MAX_CLAIM:
            raise QueueError("ids must be a non-empty list within the claim cap")
        with self._conn:
            marks = ",".join("?" * len(ids))
            cur = self._conn.execute(
                f"UPDATE queue SET state=?, completed_at=?, result=?,"
                f" claim_token=NULL WHERE id IN ({marks}) AND claim_token=?"
                f" AND state=?",
                [DONE, time.time(),
                 json.dumps(result, sort_keys=True) if result else None]
                + ids + [claim_token, IN_FLIGHT])
        return cur.rowcount

    def fail(self, *, ids: list[int], claim_token: str,
             reason: str = "") -> int:
        """Release a claim. Items under max_attempts return to pending;
        exhausted items are marked failed."""
        if not isinstance(ids, list) or not ids:
            raise QueueError("ids must be a non-empty list")
        now = time.time()
        with self._conn:
            marks = ",".join("?" * len(ids))
            # exhausted attempts → failed
            cur1 = self._conn.execute(
                f"UPDATE queue SET state=?, completed_at=?, result=?,"
                f" claim_token=NULL WHERE id IN ({marks}) AND claim_token=?"
                f" AND state=? AND attempts >= max_attempts",
                [FAILED, now, json.dumps({"error": reason})] + ids
                + [claim_token, IN_FLIGHT])
            # otherwise → back to pending for retry
            cur2 = self._conn.execute(
                f"UPDATE queue SET state=?, claimed_at=NULL, claim_token=NULL"
                f" WHERE id IN ({marks}) AND claim_token=? AND state=?",
                [PENDING] + ids + [claim_token, IN_FLIGHT])
        return cur1.rowcount + cur2.rowcount

    def stats(self, *, pass_: Optional[int] = None) -> dict:
        """Counts by state, optionally per pass."""
        where, args = "", []
        if pass_ is not None:
            where, args = "WHERE pass=?", [pass_]
        rows = self._conn.execute(
            f"SELECT state, COUNT(*) AS n FROM queue {where}"
            f" GROUP BY state", args).fetchall()
        return {r["state"]: r["n"] for r in rows}

    def pending_since(self, *, pass_: int) -> Optional[float]:
        """enqueued_at of the oldest pending item for a pass (None if empty)."""
        row = self._conn.execute(
            "SELECT MIN(enqueued_at) AS t FROM queue WHERE state=? AND pass=?",
            (PENDING, pass_)).fetchone()
        return row["t"] if row else None

    # ------------------------------------------------------------ internal
    def _prune(self) -> None:
        """Keep the table bounded: drop done rows past retention."""
        try:
            self._conn.execute(
                "DELETE FROM queue WHERE state=? AND completed_at < ?",
                (DONE, time.time() - DONE_RETENTION_S))
        except sqlite3.Error:
            pass  # pruning is best-effort
