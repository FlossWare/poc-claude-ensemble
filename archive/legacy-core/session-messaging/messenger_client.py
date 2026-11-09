#!/usr/bin/env python3
"""Client for the Claude Ensemble session messaging service."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterator

MAX_MESSAGE_BYTES = 1024 * 1024


def socket_path() -> Path:
    configured = os.environ.get("CLAUDE_MESSENGER_SOCKET")
    if configured:
        return Path(configured)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "claude-messenger" / "claude-messenger.sock"
    return Path(f"/run/user/{os.getuid()}/claude-messenger/claude-messenger.sock")


class MessengerClient:
    def __init__(
        self,
        path: Path | None = None,
        *,
        connect_timeout=2.0,
        reconnect_delay=0.5,
    ):
        self.path = path or socket_path()
        self.connect_timeout = connect_timeout
        self.reconnect_delay = reconnect_delay

    def publish(self, topic: str, data: Any, *, event_type: str | None = None) -> int:
        request = {"op": "publish", "topic": topic, "data": data}
        if event_type is not None:
            request["event_type"] = event_type
        with self._connect() as client:
            self._send(client, request)
            response = self._receive(client)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "publish failed"))
            return int(response.get("delivered", 0))

    def subscribe(self, topic: str, *, reconnect: bool = True) -> Iterator[dict]:
        return self.subscribe_filter({"topics": [topic]}, reconnect=reconnect)

    def subscribe_events(self, event_types: list[str], *, reconnect: bool = True) -> Iterator[dict]:
        return self.subscribe_filter({"event_types": event_types}, reconnect=reconnect)

    def subscribe_filter(
        self,
        filter: dict[str, list[str]],
        *,
        subscription_id: str | None = None,
        reconnect: bool = True,
    ) -> Iterator[dict]:
        if not isinstance(filter, dict) or not filter:
            raise ValueError("subscription filter must be a non-empty object")
        if subscription_id is None:
            subscription_id = self._subscription_id(filter)

        while True:
            try:
                with self._connect() as client:
                    self._send(
                        client,
                        {
                            "op": "subscribe",
                            "subscription_id": subscription_id,
                            "filter": filter,
                        },
                    )
                    response = self._receive(client)
                    if not response.get("ok"):
                        raise RuntimeError(response.get("error", "subscribe failed"))

                    reader = client.makefile("r", encoding="utf-8")
                    try:
                        for line in reader:
                            if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                                raise RuntimeError("message too large")
                            message = json.loads(line)
                            if isinstance(message, dict) and self._matches(filter, message):
                                yield message
                    finally:
                        reader.close()
            except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError):
                if not reconnect:
                    raise
                time.sleep(self.reconnect_delay)

    @staticmethod
    def _matches(filter: dict[str, list[str]], message: dict) -> bool:
        topics = filter.get("topics")
        event_types = filter.get("event_types")
        return (topics is None or message.get("topic") in topics) and (
            event_types is None or message.get("event_type") in event_types
        )

    @staticmethod
    def _subscription_id(filter: dict[str, list[str]]) -> str:
        canonical = json.dumps(filter, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return f"subscription-{digest}"

    def _connect(self) -> socket.socket:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.connect_timeout)
        client.connect(str(self.path))
        client.settimeout(None)
        return client

    @staticmethod
    def _send(client: socket.socket, request: dict) -> None:
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        client.sendall(payload)

    @staticmethod
    def _receive(client: socket.socket) -> dict:
        reader = client.makefile("r", encoding="utf-8")
        try:
            line = reader.readline()
        finally:
            reader.close()
        if not line:
            raise ConnectionError("messenger closed connection")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("invalid messenger response")
        return response
