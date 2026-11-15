#!/usr/bin/env python3

import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from messenger_client import MessengerClient
from messenger_client import socket_path as client_socket_path
from messenger_service import MessengerServer, socket_path as service_socket_path


def _recv_line(sock: socket.socket, timeout: float = 2.0) -> dict:
    sock.settimeout(timeout)
    data = bytearray()
    while b"\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise AssertionError("messenger socket closed before response")
        data.extend(chunk)
    line, _ = bytes(data).split(b"\n", 1)
    return json.loads(line.decode("utf-8"))


class MessengerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "messenger.sock"
        self.server = MessengerServer(self.path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.connect(str(self.path))
            except (ConnectionRefusedError, FileNotFoundError):
                time.sleep(0.01)
            else:
                break
        else:
            self.fail("messenger server did not become ready within 2 seconds")

    def tearDown(self):
        self.server.stop()
        self.thread.join(timeout=2)
        self.tempdir.cleanup()

    def test_idle_server_stops_promptly(self):
        self.assertTrue(self.thread.is_alive())
        self.server.stop()
        self.thread.join(timeout=1)
        self.assertFalse(self.thread.is_alive())

    def test_multiple_subscribers_receive_publish(self):
        clients = [MessengerClient(self.path) for _ in range(2)]
        received = [[], []]

        def listen(index):
            for message in clients[index].subscribe("test", reconnect=False):
                received[index].append(message)
                break

        threads = [threading.Thread(target=listen, args=(i,), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + 2
        while sum(len(items) for items in received) < 2 and time.monotonic() < deadline:
            clients[0].publish("test", {"value": 42})
            time.sleep(0.02)

        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(received[0][0]["data"]["value"], 42)
        self.assertEqual(received[1][0]["data"]["value"], 42)

    def test_event_type_filter_prevents_unrelated_delivery(self):
        subscriber = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        subscriber.connect(str(self.path))
        subscriber.sendall(
            (
                json.dumps(
                    {
                        "op": "subscribe",
                        "subscription_id": "credentials-reload",
                        "filter": {"event_types": ["credentials.reload"]},
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        self.assertTrue(_recv_line(subscriber)["ok"])

        publisher = MessengerClient(self.path)
        publisher.publish("credentials", {"action": "ignored"}, event_type="credentials.rotate")

        readable, _, _ = select.select([subscriber], [], [], 0.1)
        self.assertFalse(readable)

        publisher.publish("credentials", {"action": "reload"}, event_type="credentials.reload")
        message = _recv_line(subscriber)
        self.assertEqual(message["event_type"], "credentials.reload")
        subscriber.close()

    def test_topic_and_event_type_filters_are_conjunctive(self):
        subscriber = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        subscriber.connect(str(self.path))
        subscriber.sendall(
            (
                json.dumps(
                    {
                        "op": "subscribe",
                        "subscription_id": "credentials-reload",
                        "filter": {
                            "topics": ["credentials"],
                            "event_types": ["credentials.reload"],
                        },
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        self.assertTrue(_recv_line(subscriber)["ok"])

        publisher = MessengerClient(self.path)
        publisher.publish("other", {}, event_type="credentials.reload")
        publisher.publish("credentials", {}, event_type="credentials.rotate")

        readable, _, _ = select.select([subscriber], [], [], 0.1)
        self.assertFalse(readable)

        publisher.publish("credentials", {}, event_type="credentials.reload")
        message = _recv_line(subscriber)
        self.assertEqual(message["topic"], "credentials")
        self.assertEqual(message["event_type"], "credentials.reload")
        subscriber.close()

    def test_subscription_id_is_stable_across_processes(self):
        filter_value = {"event_types": ["credentials.reload"], "topics": ["credentials"]}
        code = (
            "import json; "
            "from messenger_client import MessengerClient; "
            "print(MessengerClient._subscription_id(json.loads(__import__('sys').argv[1])))"
        )
        ids = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", code, json.dumps(filter_value)],
                cwd=Path(__file__).parent,
                check=True,
                capture_output=True,
                text=True,
            )
            ids.append(result.stdout.strip())

        self.assertEqual(ids[0], ids[1])
        self.assertTrue(ids[0].startswith("subscription-"))
        self.assertEqual(len(ids[0]), len("subscription-") + 24)

    def test_invalid_subscription_filter_is_rejected(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(self.path))
            client.sendall(
                b'{"op":"subscribe","subscription_id":"bad","filter":{}}\n'
            )
            response = _recv_line(client)
        self.assertFalse(response["ok"])

    def test_publish_without_subscribers_is_successful(self):
        self.assertEqual(MessengerClient(self.path).publish("empty", {"value": 1}), 0)

    def test_regular_file_at_socket_path_is_not_deleted(self):
        self.server.stop()
        self.thread.join(timeout=2)
        self.path.write_text("do not delete")
        server = MessengerServer(self.path)
        with self.assertRaises(RuntimeError):
            server.serve_forever()
        self.assertEqual(self.path.read_text(), "do not delete")

    def test_default_socket_path_uses_private_runtime_subdirectory(self):
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1234"}, clear=False):
            os.environ.pop("CLAUDE_MESSENGER_SOCKET", None)
            self.assertEqual(
                service_socket_path(),
                Path("/run/user/1234/claude-messenger/claude-messenger.sock"),
            )
            self.assertEqual(service_socket_path(), client_socket_path())

    def test_socket_is_private(self):
        mode = self.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_invalid_operation_is_rejected(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(self.path))
            client.sendall(b'{"op":"bogus","topic":"test"}\n')
            response = _recv_line(client)
        self.assertFalse(response["ok"])


if __name__ == "__main__":
    unittest.main()
