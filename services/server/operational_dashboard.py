"""Read-only health inventory and operational snapshot for the POC services."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

SERVICE_SPECS = (
    ("embedding", "Embedding inference", 8770, "Generates normalized 768-dimensional vectors."),
    ("vector", "Vector database", 8771, "ChromaDB collections and vector counts."),
    ("lexical", "Lexical search", 8772, "SQLite FTS5 keyword indexing and search."),
    ("hybrid", "Hybrid search", 8773, "Coordinates transcript and knowledge retrieval."),
    ("reranking", "Reranking", 8774, "Cross-encoder relevance scoring."),
    ("chunking", "Chunking", 8775, "Deterministic, versioned document chunking."),
    ("tokenization", "Tokenization", 8776, "Versioned text analyzers."),
    ("index", "Source index", 8777, "Source revision ordering and tombstones."),
    ("extraction", "Entity extraction", 8778, "Exact identifier extraction."),
    ("intent", "Intent", 8779, "Query intent classification."),
    ("ranking", "Ranking", 8780, "Reciprocal-rank fusion."),
    ("knowledge", "Knowledge", 8781, "Sourced records and revision validation."),
    ("memory", "Memory", 8767, "Durable episodic memory and session history."),
    ("ingestion", "Source ingestion", 8782, "URI, uploaded document, and inline-text ingestion."),
)
DEFAULT_TIMEOUT_SECONDS = 1.5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _timeout_seconds() -> float:
    try:
        return max(0.1, min(float(os.environ.get(
            "POC_CLAUDE_ENSEMBLE_DASHBOARD_TIMEOUT", DEFAULT_TIMEOUT_SECONDS
        )), 10.0))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _service_url(service_id: str, port: int) -> str:
    key = service_id.upper()
    configured = os.environ.get(f"POC_CLAUDE_ENSEMBLE_{key}_URL")
    if not configured:
        # Keep compatibility with the original dashboard branch's setting names.
        configured = os.environ.get(f"ENSEMBLE_{key}_URL")
    return (configured or f"http://127.0.0.1:{port}").rstrip("/")


def _get_json(url: str) -> tuple[int, dict[str, Any], float]:
    started = time.monotonic()
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("service response exceeds the dashboard response limit")
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
    """Read-only inventory. It never infers metrics that a service does not expose."""

    def __init__(self, max_workers: int = 8):
        self.max_workers = max(1, min(max_workers, len(SERVICE_SPECS)))

    def _spec(self, service_id: str) -> dict[str, Any] | None:
        for item_id, name, port, role in SERVICE_SPECS:
            if item_id == service_id:
                base_url = _service_url(item_id, port)
                return {
                    "id": item_id,
                    "name": name,
                    "role": role,
                    "base_url": base_url,
                    "health_url": base_url + "/health",
                    "configured": True,
                }
        return None

    def service(self, service_id: str) -> dict[str, Any] | None:
        spec = self._spec(service_id)
        if spec is None:
            return None
        status, payload, latency = _get_json(spec["health_url"])
        healthy = status == 200 and payload.get("ok") is True
        return {
            "id": spec["id"],
            "name": spec["name"],
            "role": spec["role"],
            "configured": True,
            "status": "healthy" if healthy else ("unavailable" if status == 0 else "degraded"),
            "health": {
                "http_status": status or None,
                "ok": healthy,
                "service": payload.get("service") if isinstance(payload.get("service"), str) else None,
                "latency_ms": latency,
            },
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def services(self) -> dict[str, Any]:
        # Bound fan-out so one slow/unavailable service cannot serialize the
        # inventory into N * timeout seconds of waiting.
        service_ids = [item[0] for item in SERVICE_SPECS]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            items = list(executor.map(self.service, service_ids))
        counts = {
            "total": len(items),
            "healthy": sum(item["status"] == "healthy" for item in items),
            "degraded": sum(item["status"] == "degraded" for item in items),
            "unavailable": sum(item["status"] == "unavailable" for item in items),
        }
        return {
            "ok": counts["healthy"] == counts["total"],
            "services": items,
            "counts": counts,
        }

    @staticmethod
    def _vector_metrics(base_url: str, service: dict[str, Any]) -> dict[str, Any]:
        if service["status"] != "healthy":
            return {"available": False, "reason": "vector service is not healthy"}
        status, payload, latency = _get_json(base_url + "/collections")
        collections = payload.get("collections")
        if status != 200 or payload.get("ok") is not True or not isinstance(collections, list):
            return {
                "available": False,
                "reason": payload.get("error", "collection metrics unavailable"),
            }
        valid_collections = [
            item for item in collections
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("count"), int)
            and not isinstance(item.get("count"), bool)
            and item["count"] >= 0
        ]
        return {
            "available": True,
            "collections": valid_collections,
            "collection_count": len(valid_collections),
            "vector_count": sum(item["count"] for item in valid_collections),
            "latency_ms": latency,
        }

    def service_metrics(self, service_id: str) -> dict[str, Any] | None:
        service = self.service(service_id)
        if service is None:
            return None
        metrics: dict[str, Any] = {
            "available": False,
            "reason": "service does not expose a metrics endpoint",
        }
        if service_id == "vector":
            spec = self._spec("vector")
            assert spec is not None
            metrics = self._vector_metrics(spec["base_url"], service)
        return {"ok": True, "service": service, "metrics": metrics}

    def dashboard(self) -> dict[str, Any]:
        inventory = self.services()
        vector_service = next(
            item for item in inventory["services"] if item["id"] == "vector"
        )
        vector_spec = self._spec("vector")
        assert vector_spec is not None
        vector_metrics = self._vector_metrics(vector_spec["base_url"], vector_service)
        collection_counts = {
            item["name"]: item["count"]
            for item in vector_metrics.get("collections", [])
        }
        return {
            "ok": inventory["ok"],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "services": inventory["counts"],
            "storage": {
                "vector_database": "ChromaDB",
                "collections": vector_metrics.get("collection_count"),
                "vectors": vector_metrics.get("vector_count"),
                "collection_counts": collection_counts,
                "available": vector_metrics.get("available", False),
            },
            "processing": {
                "available": False,
                "reason": "durable ingestion-run accounting is not implemented",
            },
            "service_details": inventory["services"],
        }
