"""Tests for the POC operational service inventory and dashboard."""
import unittest
from unittest.mock import patch

from operational_dashboard import OperationalDashboard


class OperationalDashboardTests(unittest.TestCase):
    @patch("operational_dashboard._get_json")
    def test_service_reports_health_and_latency(self, get_json):
        get_json.return_value = (200, {"ok": True, "service": "embedding"}, 2.5)
        result = OperationalDashboard().service("embedding")
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["health"]["latency_ms"], 2.5)
        self.assertEqual(result["id"], "embedding")

    @patch("operational_dashboard._get_json")
    def test_unreachable_service_is_unavailable(self, get_json):
        get_json.return_value = (0, {"error": "connection refused"}, 1500.0)
        result = OperationalDashboard().service("vector")
        self.assertEqual(result["status"], "unavailable")

    @patch("operational_dashboard._get_json")
    def test_vector_metrics_include_collection_and_vector_counts(self, get_json):
        get_json.side_effect = [
            (200, {"ok": True, "service": "vector"}, 1.2),
            (200, {"ok": True, "collections": [
                {"name": "transcript_text", "count": 12},
                {"name": "knowledge_chunks", "count": 88},
            ]}, 3.4),
        ]
        result = OperationalDashboard().service_metrics("vector")
        self.assertEqual(result["metrics"]["collection_count"], 2)
        self.assertEqual(result["metrics"]["vector_count"], 100)
        self.assertEqual(result["metrics"]["collections"][1]["name"], "knowledge_chunks")

    @patch("operational_dashboard._get_json")
    def test_dashboard_does_not_invent_ingestion_counts(self, get_json):
        get_json.side_effect = [
            (200, {"ok": True}, 1.0),
        ] * 13 + [
            (200, {"ok": True}, 1.0),
            (200, {"ok": True, "collections": [{"name": "knowledge_chunks", "count": 7}]}, 2.0),
        ]
        result = OperationalDashboard().dashboard()
        self.assertFalse(result["processing"]["available"])
        self.assertEqual(result["storage"]["vectors"], 7)

    def test_unknown_service_returns_none(self):
        self.assertIsNone(OperationalDashboard().service("not-a-service"))
        self.assertIsNone(OperationalDashboard().service_metrics("not-a-service"))


if __name__ == "__main__":
    unittest.main()
