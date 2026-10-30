"""Read-only operational inventory and dashboard aggregation for the POC gateway."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import json
from typing import Any

SERVICE_SPECS = (
    ("embedding", "Embedding inference", 8770, "Generates normalized 768-dimensional vectors."),
    ("vector", "Vector database", 8771, "ChromaDB collections, vector counts, and exact nearest-neighbor search."),
    ("lexical", "Lexical search", 8772, "SQLite FTS5 keyword indexing and search."),
    ("hybrid", "Hybrid search", 8773, "Coordinates vector, lexical, knowledge, and reranking retrieval."),
    ("reranking", "Reranking", 8774, "Cross-encoder relevance scoring."),
    ("chunking", "Chunking", 8775, "Deterministic, versioned document chunking."),
    ("tokenization", "Tokenization", 8776, "Versioned text analyzers."),
    ("index", "Source index", 8777, "Source revision ordering and tombstones."),
    ("extraction", "Entity extraction", 8778, "Exact identifier extraction."),
    ("intent", "Intent", 8779, "Query intent classification."),
    ("ranking", "Ranking", 8780, "Reciprocal-rank fusion."),
    ("knowledge", "Knowledge", 8781, "Sourced records and revision validation."),
    ("memory", "Memory", 8767, "Durable episodic memory."),
)
TIMEOUT_SECONDS = float(os.environ.get("ENSEMBLE_DASHBOARD_PROBE_TIMEOUT", "1.5"))


def _service_url(service_id: str, port: int) -> str:
    return os.environ.get(
        f"ENSEMBLE_{service_id.upper()}_URL",
        f"http://127.0.0.1:{port}",
    ).rstrip("/")


def _get_json(url: str) -> tuple[int, dict[str, Any], float]:
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(2 * 1024 * 1024)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("service response must be a JSON object")
            return response.status, payload, round((time.monotonic() - started) * 1000, 2)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(64 * 1024)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                payload = {"error": "non-object error response"}
        except (ValueError, UnicodeDecodeError):
            payload = {"error": "unable to decode service error response"}
        return exc.code, payload, round((time.monotonic() - started) * 1000, 2)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return 0, {"error": str(exc)}, round((time.monotonic() - started) * 1000, 2)


class OperationalDashboard:
    """Discover configured POC services and aggregate live, explicitly sourced metrics."""

    def __init__(self, gateway_url: str | None = None):
        self.gateway_url = gateway_url

    def _specs(self):
        for service_id, name, port, role in SERVICE_SPECS:
            yield {
                "id": service_id,
                "name": name,
                "role": role,
                "base_url": _service_url(service_id, port),
                "health_url": _service_url(service_id, port) + "/health",
                "configured": True,
            }

    def service(self, service_id: str) -> dict[str, Any] | None:
        spec = next((item for item in self._specs() if item["id"] == service_id), None)
        if spec is None:
            return None
        status, payload, latency = _get_json(spec["health_url"])
        healthy = status == 200 and payload.get("ok") is True
        return {
            **{key: value for key, value in spec.items() if key not in {"base_url", "health_url"}},
            "endpoint": spec["base_url"],
            "status": "healthy" if healthy else ("unavailable" if status == 0 else "degraded"),
            "health": {"http_status": status or None, "response": payload, "latency_ms": latency},
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def services(self) -> dict[str, Any]:
        items = [self.service(spec["id"]) for spec in self._specs()]
        counts = {
            "total": len(items),
            "healthy": sum(item["status"] == "healthy" for item in items),
            "degraded": sum(item["status"] == "degraded" for item in items),
            "unavailable": sum(item["status"] == "unavailable" for item in items),
        }
        return {"ok": counts["healthy"] == counts["total"], "services": items, "counts": counts}

    def service_metrics(self, service_id: str) -> dict[str, Any] | None:
        item = self.service(service_id)
        if item is None:
            return None
        metrics: dict[str, Any] = {"available": False, "reason": "service does not expose a metrics endpoint"}
        if service_id == "vector":
            status, payload, latency = _get_json(item["endpoint"] + "/collections")
            if status == 200 and payload.get("ok") is True and isinstance(payload.get("collections"), list):
                collections = payload["collections"]
                metrics = {
                    "available": True,
                    "collections": collections,
                    "collection_count": len(collections),
                    "vector_count": sum(
                        value.get("count", 0) for value in collections
                        if isinstance(value, dict) and isinstance(value.get("count", 0), int)
                    ),
                    "latency_ms": latency,
                }
            else:
                metrics = {"available": False, "error": payload.get("error", "collection metrics unavailable")}
        return {"ok": True, "service": item, "metrics": metrics}

    def dashboard(self) -> dict[str, Any]:
        inventory = self.services()
        vector = self.service_metrics("vector")
        vector_metrics = vector["metrics"] if vector else {"available": False}
        collections = vector_metrics.get("collections", [])
        by_name = {item.get("name"): item.get("count") for item in collections if isinstance(item, dict)}
        return {
            "ok": inventory["ok"],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "services": inventory["counts"],
            "storage": {
                "vector_database": "ChromaDB",
                "collections": vector_metrics.get("collection_count"),
                "vectors": vector_metrics.get("vector_count"),
                "collection_counts": by_name,
                "available": vector_metrics.get("available", False),
            },
            "processing": {
                "available": False,
                "reason": "durable ingestion-run accounting is not implemented yet",
            },
            "service_details": inventory["services"],
        }
