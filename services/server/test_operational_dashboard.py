"""Tests for the read-only POC operational dashboard."""
import unittest
from unittest.mock import patch

from operational_dashboard import OperationalDashboard


class OperationalDashboardTests(unittest.TestCase):
    @patch("operational_dashboard._get_json")
    def test_service_reports_health_and_latency(self, get_json):
        get_json.return_value = (200, {"ok": True, "service": "embedding", "token": "must-not-leak"}, 2.5)
        result = OperationalDashboard().service("embedding")
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["health"]["latency_ms"], 2.5)
        self.assertEqual(result["id"], "embedding")
        self.assertNotIn("token", result["health"])

    @patch("operational_dashboard._get_json")
    def test_unreachable_service_is_unavailable(self, get_json):
        get_json.return_value = (0, {"error": "connection refused"}, 1500.0)
        result = OperationalDashboard().service("vector")
        self.assertEqual(result["status"], "unavailable")

    @patch("operational_dashboard._get_json")
    def test_vector_metrics_validate_collection_counts(self, get_json):
        get_json.side_effect = [
            (200, {"ok": True, "service": "vector"}, 1.2),
            (200, {"ok": True, "collections": [
                {"name": "transcript_text", "count": 12},
                {"name": "knowledge_chunks", "count": 88},
                {"name": "bad", "count": True},
                {"name": "negative", "count": -1},
            ]}, 3.4),
        ]
        result = OperationalDashboard().service_metrics("vector")
        self.assertTrue(result["metrics"]["available"])
        self.assertEqual(result["metrics"]["collection_count"], 2)
        self.assertEqual(result["metrics"]["vector_count"], 100)

    @patch("operational_dashboard._get_json")
    def test_dashboard_does_not_invent_ingestion_counts_or_repeat_health_probe(self, get_json):
        def response_for(url):
            if url.endswith("/collections"):
                return (200, {"ok": True, "collections": [
                    {"name": "knowledge_chunks", "count": 7},
                ]}, 2.0)
            return (200, {"ok": True}, 1.0)

        get_json.side_effect = response_for
        result = OperationalDashboard().dashboard()
        self.assertFalse(result["processing"]["available"])
        self.assertEqual(result["storage"]["vectors"], 7)
        self.assertEqual(get_json.call_count, 14)

    def test_unknown_service_returns_none(self):
        dashboard = OperationalDashboard()
        self.assertIsNone(dashboard.service("not-a-service"))
        self.assertIsNone(dashboard.service_metrics("not-a-service"))

    @patch.dict("os.environ", {"POC_CLAUDE_ENSEMBLE_VECTOR_URL": "http://127.0.0.1:9991/"})
    def test_poc_service_url_override_is_supported(self):
        from operational_dashboard import _service_url
        self.assertEqual(_service_url("vector", 8771), "http://127.0.0.1:9991")


if __name__ == "__main__":
    unittest.main()
