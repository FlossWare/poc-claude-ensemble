#!/usr/bin/env python3
"""Small Unix-socket pub/sub daemon for Claude Ensemble sessions."""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
from pathlib import Path
from typing import Any

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_FILTER_VALUES = 64
MAX_FILTER_VALUE_LENGTH = 128


def socket_path() -> Path:
    configured = os.environ.get("CLAUDE_MESSENGER_SOCKET")
    if configured:
        return Path(configured)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "claude-messenger" / "claude-messenger.sock"
    return Path(f"/run/user/{os.getuid()}/claude-messenger/claude-messenger.sock")


class MessengerServer:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or socket_path()
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._subscriptions: dict[socket.socket, dict[str, dict[str, frozenset[str]]]] = {}

    def serve_forever(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._remove_stale_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen()
        server.settimeout(1.0)
        self._server = server

        try:
            while not self._stop.is_set():
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                ).start()
        finally:
            server.close()
            self._close_all_subscribers()
            self._remove_stale_socket()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass

    def _remove_stale_socket(self) -> None:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return
        if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
            raise RuntimeError(f"refusing to remove unexpected socket path: {self.path}")
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _handle_client(self, client: socket.socket) -> None:
        reader = client.makefile("r", encoding="utf-8")
        try:
            for line in reader:
                if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    self._send(client, {"ok": False, "error": "message too large"})
                    break
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    response = self._dispatch(client, request)
                except (ValueError, json.JSONDecodeError) as exc:
                    response = {"ok": False, "error": str(exc)}
                self._send(client, response)
        finally:
            reader.close()
            self._remove_client(client)
            try:
                client.close()
            except OSError:
                pass

    def _dispatch(self, client: socket.socket, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op not in {"subscribe", "unsubscribe", "publish"}:
            return {"ok": False, "error": "unsupported operation"}

        if op == "subscribe":
            try:
                subscription_id, filters = self._parse_subscription(request)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            with self._lock:
                subscriptions = self._subscriptions.setdefault(client, {})
                subscriptions[subscription_id] = filters
            return {"ok": True, "subscription_id": subscription_id}

        if op == "unsubscribe":
            subscription_id = request.get("subscription_id")
            if not isinstance(subscription_id, str) or not subscription_id:
                return {"ok": False, "error": "unsubscribe requires subscription_id"}
            with self._lock:
                subscriptions = self._subscriptions.get(client)
                if subscriptions is not None:
                    subscriptions.pop(subscription_id, None)
            return {"ok": True, "subscription_id": subscription_id}

        topic = request.get("topic")
        if not isinstance(topic, str) or not topic or len(topic) > MAX_FILTER_VALUE_LENGTH:
            return {"ok": False, "error": "invalid topic"}
        if "data" not in request:
            return {"ok": False, "error": "publish requires data"}

        event_type = request.get("event_type", topic)
        if not isinstance(event_type, str) or not event_type or len(event_type) > MAX_FILTER_VALUE_LENGTH:
            return {"ok": False, "error": "invalid event_type"}

        message = {"topic": topic, "event_type": event_type, "data": request["data"]}
        return {"ok": True, "delivered": self._publish(topic, event_type, message)}

    def _parse_subscription(self, request: dict[str, Any]) -> tuple[str, dict[str, frozenset[str]]]:
        subscription_id = request.get("subscription_id")
        if subscription_id is None:
            subscription_id = request.get("topic")
        if not isinstance(subscription_id, str) or not subscription_id or len(subscription_id) > MAX_FILTER_VALUE_LENGTH:
            raise ValueError("subscription requires subscription_id")

        raw_filter = request.get("filter")
        if raw_filter is None:
            topic = request.get("topic")
            if not isinstance(topic, str) or not topic or len(topic) > MAX_FILTER_VALUE_LENGTH:
                raise ValueError("subscription requires filter")
            raw_filter = {"topics": [topic]}

        if not isinstance(raw_filter, dict) or not raw_filter:
            raise ValueError("subscription filter must be a non-empty object")

        allowed = {"topics", "event_types"}
        if set(raw_filter) - allowed:
            raise ValueError("unsupported subscription filter field")

        parsed: dict[str, frozenset[str]] = {}
        for field in allowed:
            values = raw_filter.get(field)
            if values is None:
                continue
            if not isinstance(values, list) or not values or len(values) > MAX_FILTER_VALUES:
                raise ValueError(f"{field} must be a non-empty list")
            if not all(isinstance(value, str) and 0 < len(value) <= MAX_FILTER_VALUE_LENGTH for value in values):
                raise ValueError(f"{field} contains an invalid value")
            parsed[field] = frozenset(values)

        if not parsed:
            raise ValueError("subscription filter must contain topics or event_types")
        return subscription_id, parsed

    def _publish(self, topic: str, event_type: str, message: dict[str, Any]) -> int:
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_MESSAGE_BYTES:
            return 0

        with self._lock:
            recipients = [
                client
                for client, subscriptions in self._subscriptions.items()
                if any(self._matches(filters, topic, event_type) for filters in subscriptions.values())
            ]

        delivered = 0
        stale = []
        for client in recipients:
            try:
                client.sendall(payload)
                delivered += 1
            except OSError:
                stale.append(client)
        for client in stale:
            with self._lock:
                self._remove_client_locked(client)
        return delivered

    @staticmethod
    def _matches(filters: dict[str, frozenset[str]], topic: str, event_type: str) -> bool:
        topics = filters.get("topics")
        event_types = filters.get("event_types")
        return (topics is None or topic in topics) and (
            event_types is None or event_type in event_types
        )

    @staticmethod
    def _send(client: socket.socket, response: dict[str, Any]) -> None:
        payload = (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            client.sendall(payload)
        except OSError:
            pass

    def _remove_client(self, client: socket.socket) -> None:
        with self._lock:
            self._remove_client_locked(client)

    def _remove_client_locked(self, client: socket.socket) -> None:
        self._subscriptions.pop(client, None)

    def _close_all_subscribers(self) -> None:
        with self._lock:
            clients = set(self._subscriptions)
            self._subscriptions.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass


if __name__ == "__main__":
    MessengerServer().serve_forever()
