#!/usr/bin/env python3
"""Durable multipass work queue for corpus ingestion — ledger-backed.

Event-sourced on an append-only JSONL ledger (the FlossWare framework):
every state change is an event; queue state is derived by replaying the
log at startup. The log is the permanent evidence stream (what was
enqueued, claimed, completed, failed, when, and why); in-memory state is
a rebuildable projection. Claim atomicity comes from the single
ingestion-service process serializing all operations under a lock —
SQLite's multi-writer atomicity is unnecessary in the single-process
deployment and would introduce a second storage framework.

Semantics (unchanged from the SQLite predecessor):
- enqueue/claim/complete/fail/stats with priority-ordered drain
- at-least-once delivery: stale claims (CLAIM_TIMEOUT_S) requeue;
  the ingestion pipeline is idempotent by content hash
- bounded retries: a stale or failed claim with exhausted attempts
  goes to failed, never reclaimed forever
- memory bound: done items older than DONE_RETENTION_S are dropped
  from the in-memory projection (counted, not held); the LOG retains
  full detail forever

Module is deliberately dependency-free (stdlib only).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_LEDGER = os.environ.get(
    "POC_CLAUDE_ENSEMBLE_INGEST_QUEUE_LEDGER",
    os.path.expanduser("~/.FlossWare/ledger/ingest-queue/events.jsonl"),
)

# states
PENDING = "pending"
IN_FLIGHT = "in_flight"
DONE = "done"
FAILED = "failed"

MAX_CLAIM = 200          # max items per claim
CLAIM_TIMEOUT_S = 3600    # a claim older than this is stale → requeueable
DONE_RETENTION_S = 7 * 24 * 3600


class QueueError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


class IngestQueue:
    def __init__(self, ledger_path: str | Path = DEFAULT_LEDGER):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_id = 1
        # id -> {pass, priority, state, payload, attempts, max_attempts,
        #        enqueued_at, claimed_at, claim_token}
        self._items: dict[int, dict[str, Any]] = {}
        self._done_counts: dict[int, int] = {}   # pass -> done count (all time)
        self._replay()
        self._fh = open(self.ledger_path, "a", encoding="utf-8")

    # ------------------------------------------------------------ replay
    def _replay(self) -> None:
        """Rebuild in-memory state by replaying the event log."""
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                is_final = line_number == len(lines)
                ends_newline = line.endswith("\n")
                if is_final and not ends_newline:
                    # torn TAIL from a crash mid-write: the append
                    # protocol writes whole batches with a single
                    # fsync, so only the final line can be torn —
                    # truncate it and continue
                    with self.ledger_path.open("r+b") as fix:
                        fix.seek(-(len(line)), 2)
                        fix.truncate()
                    break
                # corruption in the MIDDLE is unrecoverable: fail closed
                raise RuntimeError(
                    f"ingest-queue ledger is corrupt at line {line_number}")
            self._apply(event, from_log=True)

    def _apply(self, event: dict, *, from_log: bool,
               done_cutoff: float = 0.0) -> None:
        """Fold one event into in-memory state. from_log=True during
        replay (no append); False during live operations (append first,
        then fold — caller holds the lock)."""
        kind = event.get("event")
        item_id = event.get("id")
        ts = event.get("ts", 0.0)

        if kind == "enqueued":
            item_id = item_id if item_id is not None else self._next_id
            self._next_id = max(self._next_id, item_id + 1)
            self._items[item_id] = {
                "pass": event.get("pass", 1),
                "priority": event.get("priority", 100),
                "state": PENDING,
                "payload": event.get("payload", {}),
                "attempts": 0,
                "max_attempts": event.get("max_attempts", 3),
                "enqueued_at": ts,
                "claimed_at": None,
                "claim_token": None,
            }
        elif kind == "claimed":
            item = self._items.get(item_id)
            if item is not None:
                item["state"] = IN_FLIGHT
                item["claimed_at"] = ts
                item["claim_token"] = event.get("token")
                item["attempts"] = event.get("attempts", item["attempts"] + 1)
        elif kind == "completed":
            item = self._items.pop(item_id, None)
            pass_ = item["pass"] if item else event.get("pass", 1)
            self._done_counts[pass_] = self._done_counts.get(pass_, 0) + 1
            # done items leave the in-memory projection immediately
            # (memory bound); the log retains full detail forever
        elif kind == "failed":
            item = self._items.get(item_id)
            if item is not None:
                item["state"] = FAILED
                item["claim_token"] = None
        elif kind == "requeued":
            item = self._items.get(item_id)
            if item is not None:
                item["state"] = PENDING
                item["claimed_at"] = None
                item["claim_token"] = None

    # ------------------------------------------------------------- events
    def _append_batch(self, events: list[dict]) -> None:
        """Write a whole batch as ONE buffer + single fsync, so a mid-
        batch failure cannot leave a torn line in the middle of the
        ledger (a torn TAIL from a crash is recoverable at replay; a
        torn MIDDLE is not). Raises on failure — nothing partially
        applied, because callers fold state only after this returns."""
        if not events:
            return
        buf = "".join(json.dumps(e, sort_keys=True) + "\n" for e in events)
        self._fh.write(buf)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    # ---------------------------------------------------------------- api
    def enqueue(self, items: list[dict], *, pass_: int = 1,
                priority: int = 100, max_attempts: int = 3) -> int:
        if not isinstance(items, list) or not items:
            raise QueueError("items must be a non-empty list")
        if len(items) > MAX_CLAIM:
            raise QueueError(f"at most {MAX_CLAIM} items per enqueue",
                             413, "batch_too_large")
        now = time.time()
        with self._lock:
            events = []
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise QueueError(f"items[{index}] must be an object")
                try:
                    payload = json.dumps(item, sort_keys=True)
                    json.loads(payload)  # round-trip: must be loadable
                except (TypeError, ValueError) as exc:
                    raise QueueError(
                        f"items[{index}] is not JSON-serializable: {exc}")
                events.append({"event": "enqueued", "id": self._next_id,
                               "pass": pass_, "priority": priority,
                               "max_attempts": max_attempts,
                               "payload": item, "ts": now})
                self._next_id += 1
            self._append_batch(events)
            for event in events:
                self._apply(event, from_log=False)
        return len(events)

    def claim(self, *, pass_: int, limit: int = 50) -> list[dict]:
        if not isinstance(limit, int) or not 1 <= limit <= MAX_CLAIM:
            raise QueueError(f"limit must be 1..{MAX_CLAIM}")
        now = time.time()
        cutoff = now - CLAIM_TIMEOUT_S
        import uuid
        token = uuid.uuid4().hex  # reliably unique; id(object()) reuses
        with self._lock:
            # stale-claim recovery: requeue or fail by attempt budget
            recovery = []
            for item_id, item in list(self._items.items()):
                if item["state"] != IN_FLIGHT:
                    continue
                if (item.get("claimed_at") or 0) >= cutoff:
                    continue
                if item["attempts"] >= item["max_attempts"]:
                    recovery.append({"event": "failed", "id": item_id,
                                     "reason": "stale claim; attempts exhausted",
                                     "ts": now})
                else:
                    recovery.append({"event": "requeued", "id": item_id,
                                     "reason": "stale claim", "ts": now})
            if recovery:
                self._append_batch(recovery)
                for event in recovery:
                    self._apply(event, from_log=False)
            # priority-ordered drain of pending items for this pass
            pending = sorted(
                ((i, it) for i, it in self._items.items()
                 if it["state"] == PENDING and it["pass"] == pass_),
                key=lambda pair: (pair[1]["priority"],
                                  pair[1]["enqueued_at"], pair[0]))
            claim_events = []
            claimed = []
            for item_id, item in pending[:limit]:
                attempts = item["attempts"] + 1
                claim_events.append({"event": "claimed", "id": item_id,
                                     "token": token, "attempts": attempts,
                                     "ts": now})
                claimed.append({"id": item_id, "payload": item["payload"],
                                "attempts": attempts, "claim_token": token})
            if claim_events:
                self._append_batch(claim_events)
                for event in claim_events:
                    self._apply(event, from_log=False)
            return claimed

    def complete(self, *, ids: list[int], claim_token: str,
                 result: Optional[dict] = None) -> int:
        if not isinstance(ids, list) or not ids or len(ids) > MAX_CLAIM:
            raise QueueError("ids must be a non-empty list within the claim cap")
        # serialize-check and size-cap BEFORE any state change: a
        # TypeError mid-loop would commit earlier ids and drop the rest
        result_json = None
        if result is not None:
            try:
                result_json = json.dumps(result, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise QueueError(f"result is not JSON-serializable: {exc}")
            if len(result_json) > 65_536:
                raise QueueError("result exceeds the 64 KiB cap", 413,
                                 "payload_too_large")
        now = time.time()
        with self._lock:
            events = []
            for item_id in ids:
                item = self._items.get(item_id)
                if (item is None or item["state"] != IN_FLIGHT
                        or item.get("claim_token") != claim_token):
                    continue
                events.append({"event": "completed", "id": item_id,
                               "result": result, "ts": now})
            if events:
                self._append_batch(events)
                for event in events:
                    self._apply(event, from_log=False)
            return len(events)

    def fail(self, *, ids: list[int], claim_token: str,
             reason: str = "") -> int:
        if not isinstance(ids, list) or not ids:
            raise QueueError("ids must be a non-empty list")
        if not isinstance(reason, str) or len(reason) > 4096:
            raise QueueError("reason must be a string of at most 4096 characters")
        now = time.time()
        with self._lock:
            events = []
            for item_id in ids:
                item = self._items.get(item_id)
                if (item is None or item["state"] != IN_FLIGHT
                        or item.get("claim_token") != claim_token):
                    continue
                if item["attempts"] >= item["max_attempts"]:
                    events.append({"event": "failed", "id": item_id,
                                   "reason": reason, "ts": now})
                else:
                    events.append({"event": "requeued", "id": item_id,
                                   "reason": reason, "ts": now})
            if events:
                self._append_batch(events)
                for event in events:
                    self._apply(event, from_log=False)
            return len(events)

    def stats(self, *, pass_: Optional[int] = None) -> dict:
        with self._lock:
            out: dict[str, int] = {}
            for item in self._items.values():
                if pass_ is not None and item["pass"] != pass_:
                    continue
                out[item["state"]] = out.get(item["state"], 0) + 1
            if pass_ is not None:
                total_done = self._done_counts.get(pass_, 0)
            else:
                total_done = sum(self._done_counts.values())
            if total_done:
                out[DONE] = total_done
            return out
