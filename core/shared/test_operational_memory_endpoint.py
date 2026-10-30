"""Regression tests for the operational Memory REST endpoint."""

import json
import os
import unittest
from unittest.mock import patch

from shared.operational_memory import (
    DEFAULT_MEMORY_GATEWAY_URL,
    OperationalMemoryWriter,
)


class OperationalMemoryEndpointTests(unittest.TestCase):
    def test_default_points_to_memory_service_rest_base(self):
        self.assertEqual(
            DEFAULT_MEMORY_GATEWAY_URL,
            "http://127.0.0.1:8767/memory",
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch("shared.operational_memory.urllib.request.urlopen")
    def test_write_posts_to_memory_service_write_route(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = b'{"ok": true}'

        writer = OperationalMemoryWriter()
        self.assertTrue(
            writer.write_event(
                "ga-run-123",
                "ga.tuning.result",
                "ga_tuning",
                {"evaluation_mode": "synthetic-local-evaluators"},
            )
        )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8767/memory/write",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        body = json.loads(request.data.decode("utf-8"))
        self.assertTrue(body["name"].startswith("operational-ga.tuning.result-"))
        self.assertIn("ga-run-123", body["content"])
        self.assertIn("synthetic-local-evaluators", body["content"])

    @patch.dict(
        os.environ,
        {"ENSEMBLE_MEMORY_GATEWAY_URL": "http://127.0.0.1:9876/custom-memory"},
        clear=True,
    )
    def test_environment_override_remains_a_base_url(self):
        writer = OperationalMemoryWriter()
        self.assertEqual(
            writer.base_url,
            "http://127.0.0.1:9876/custom-memory",
        )


if __name__ == "__main__":
    unittest.main()
