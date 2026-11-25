#!/usr/bin/env python3
"""Tests for tools/feedback_capture.py.

Network transport is mocked throughout — these tests must never send a
real outcome to the Learning service (that would mutate Thompson state).
"""

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feedback_capture  # noqa: E402


def _urlopen_ok(payload):
    """Build a mock urlopen returning an HTTP 200 with `payload`."""
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode("utf-8")
    return response


class SendOutcomeTests(unittest.TestCase):
    def _capture_request(self, urlopen_mock):
        """Return the parsed request body the tool sent."""
        req = urlopen_mock.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "http://127.0.0.1:8768/learning")
        return json.loads(req.data.decode("utf-8"))

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_success_wire_format(self, urlopen_mock):
        urlopen_mock.return_value = _urlopen_ok({"ok": True})
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t1", task_type="code_review",
            model="haiku", rating=4, tokens=0, cost=0.0, request_id="r1")
        self.assertEqual(result, {"ok": True})
        body = self._capture_request(urlopen_mock)
        self.assertEqual(body, {
            "op": "process_outcome",
            "task_id": "t1",
            "task_type": "code_review",
            "model": "haiku",
            "rating": 4,
            "tokens": 0,
            "cost": 0.0,
            "request_id": "r1",
        })

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_service_error_is_reported_not_raised(self, urlopen_mock):
        urlopen_mock.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8768/learning", 500, "boom", {},
            io.BytesIO(b'{"ok": false, "error": "internal"}'))
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t2", task_type="docs",
            model="opus", rating=5)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "internal")

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_non_json_error_body_surfaces_http_status(self, urlopen_mock):
        urlopen_mock.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8768/learning", 502, "bad gateway", {},
            io.BytesIO(b"<html>proxy error page</html>"))
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t2b", task_type="docs",
            model="opus", rating=5)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "http_502")

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_json_error_body_without_error_key_surfaces_status(self, urlopen_mock):
        urlopen_mock.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8768/learning", 503, "unavailable", {},
            io.BytesIO(b'{"ok": false}'))
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t2c", task_type="docs",
            model="opus", rating=5)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "http_503")

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_context_travels_in_payload(self, urlopen_mock):
        urlopen_mock.return_value = _urlopen_ok({"ok": True})
        feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t5", task_type="code_review",
            model="haiku", rating=4, context="missed SQL injection")
        body = self._capture_request(urlopen_mock)
        self.assertEqual(body["context"], "missed SQL injection")

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_context_omitted_when_none(self, urlopen_mock):
        urlopen_mock.return_value = _urlopen_ok({"ok": True})
        feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t6", task_type="code_review",
            model="haiku", rating=4)
        body = self._capture_request(urlopen_mock)
        self.assertNotIn("context", body)

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_transport_failure_degrades_gracefully(self, urlopen_mock):
        urlopen_mock.side_effect = ConnectionRefusedError("down")
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:65534", task_id="t3", task_type="testing",
            model="glm", rating=3)
        self.assertFalse(result.get("ok"))
        self.assertIn("down", result.get("error", ""))

    @mock.patch("feedback_capture.urllib.request.urlopen")
    def test_non_dict_response_rejected(self, urlopen_mock):
        response = _urlopen_ok({"ok": True})
        response.read.return_value = b'["not", "a", "dict"]'
        urlopen_mock.return_value = response
        result = feedback_capture.send_outcome(
            "http://127.0.0.1:8768", task_id="t4", task_type="x",
            model="m", rating=1)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "invalid_response")


class CaptureFeedbackTests(unittest.TestCase):
    @mock.patch("feedback_capture.send_outcome")
    def test_success_returns_task_id(self, send_mock):
        send_mock.return_value = {"ok": True}
        task_id = feedback_capture.capture_feedback("haiku", 4, "code_review")
        self.assertTrue(task_id.startswith("feedback_haiku_"))
        sent = send_mock.call_args.kwargs
        self.assertEqual(sent["task_type"], "code_review")
        self.assertEqual(sent["model"], "haiku")
        self.assertEqual(sent["rating"], 4)
        self.assertIsNone(sent.get("context"))

    @mock.patch("feedback_capture.send_outcome")
    def test_context_forwarded(self, send_mock):
        send_mock.return_value = {"ok": True}
        feedback_capture.capture_feedback("haiku", 4, "code_review",
                                          context="fast and accurate")
        self.assertEqual(send_mock.call_args.kwargs["context"],
                         "fast and accurate")

    @mock.patch("feedback_capture.send_outcome")
    def test_invalid_rating_rejected_before_any_request(self, send_mock):
        self.assertIsNone(feedback_capture.capture_feedback("haiku", 9, "x"))
        self.assertIsNone(feedback_capture.capture_feedback("haiku", 0, "x"))
        send_mock.assert_not_called()

    @mock.patch("feedback_capture.send_outcome")
    def test_failure_returns_none(self, send_mock):
        send_mock.return_value = {"ok": False, "error": "nope"}
        self.assertIsNone(feedback_capture.capture_feedback("m", 3, "x"))


class CliTests(unittest.TestCase):
    def test_missing_required_args_exit_nonzero(self):
        with mock.patch("sys.argv", ["feedback_capture.py", "--model", "m"]):
            with self.assertRaises(SystemExit) as ctx:
                feedback_capture.main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_exit_code_reflected(self):
        with mock.patch("sys.argv", ["feedback_capture.py", "--model", "m",
                                     "--rating", "5", "--task", "t"]):
            with mock.patch("feedback_capture.capture_feedback",
                            return_value="feedback_m_1"):
                self.assertEqual(feedback_capture.main(), 0)
        with mock.patch("sys.argv", ["feedback_capture.py", "--model", "m",
                                     "--rating", "5", "--task", "t"]):
            with mock.patch("feedback_capture.capture_feedback",
                            return_value=None):
                self.assertEqual(feedback_capture.main(), 1)


if __name__ == "__main__":
    unittest.main()
