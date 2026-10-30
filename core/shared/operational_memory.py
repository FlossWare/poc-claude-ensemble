#!/usr/bin/env python3
"""REST client for durable operational events in the canonical Memory Service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_GATEWAY_URL = "http://127.0.0.1:8767/memory"
_EVENT_TYPE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class OperationalMemoryPersistenceError(RuntimeError):
    """Raised when a required operational event cannot be persisted."""


class OperationalMemoryWriter:
    """Write operational events through the canonical Ensemble REST boundary."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 2.0) -> None:
        self.base_url = (
            base_url
            or os.environ.get("ENSEMBLE_MEMORY_GATEWAY_URL")
            or DEFAULT_MEMORY_GATEWAY_URL
        ).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _memory_name(event_type: str, event_id: str) -> str:
        digest = hashlib.sha256(f"{event_type}:{event_id}".encode("utf-8")).hexdigest()
        safe_type = _EVENT_TYPE_PATTERN.sub("-", event_type).strip("-") or "event"
        return f"operational-{safe_type}-{digest[:24]}"

    @staticmethod
    def _document(
        event_id: str,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
    ) -> str:
        record = {
            "event_id": event_id,
            "event_type": event_type,
            "source": source,
            "persisted_at": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        return (
            f"# Operational Event: {event_type}\n\n"
            f"**Event ID:** {event_id}\n\n"
            f"**Source:** {source}\n\n"
            "```json\n"
            f"{json.dumps(record, indent=2, sort_keys=True)}\n"
            "```\n"
        )

    def write_event(
        self,
        event_id: str,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
    ) -> bool:
        """Persist one complete operational event through Memory REST."""
        if not event_id:
            raise ValueError("event_id is required")
        if not event_type:
            raise ValueError("event_type is required")
        if not source:
            raise ValueError("source is required")

        body = json.dumps(
            {
                "name": self._memory_name(event_type, event_id),
                "content": self._document(event_id, event_type, source, payload),
            },
            sort_keys=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/write",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not result.get("ok"):
                logger.warning(
                    "Memory rejected operational event %s: %s",
                    event_id,
                    result.get("error", "unknown error"),
                )
                if event_type == "thompson.state":
                    raise OperationalMemoryPersistenceError(
                        f"required Thompson state persistence failed: {result.get('error', 'unknown error')}"
                    )
                return False
            return True
        except OperationalMemoryPersistenceError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("Memory write failed for operational event %s: %s", event_id, exc)
            if event_type == "thompson.state":
                raise OperationalMemoryPersistenceError(
                    f"required Thompson state persistence failed: {exc}"
                ) from exc
            return False
