#!/usr/bin/env python3
"""Portable, provider-neutral learning artifacts.

The artifact contract deliberately contains only reusable learning state and
provenance. It does not encode CE execution objects, prompts, credentials,
private reasoning, or source payloads.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import threading

SCHEMA = "learning-artifact"
VERSION = "0.1"
DEFAULT_PATH = Path(__file__).resolve().parent / "portable_artifacts.jsonl"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact values must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported artifact value type: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    """Return an immutable snapshot of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class LearningArtifact:
    """A self-contained learning artifact that can be consumed independently."""

    artifact_type: str
    source: str
    payload: Mapping[str, Any]
    provenance: Mapping[str, Any] | None = None
    created_at: str | None = None
    schema: str = SCHEMA
    version: str = VERSION

    def __post_init__(self) -> None:
        _required_string(self.artifact_type, "artifact_type")
        _required_string(self.source, "source")
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported artifact schema: {self.schema!r}")
        if self.version != VERSION:
            raise ValueError(f"unsupported artifact version: {self.version!r}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping when provided")
        if self.created_at is not None:
            _required_string(self.created_at, "created_at")
        _jsonable(self.payload)
        if self.provenance is not None:
            _jsonable(self.provenance)
        object.__setattr__(self, "payload", _freeze(self.payload))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", _freeze(self.provenance))

    @classmethod
    def create(
        cls,
        artifact_type: str,
        payload: Mapping[str, Any],
        *,
        source: str = "claude-ensemble",
        provenance: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "LearningArtifact":
        return cls(
            artifact_type=artifact_type,
            source=source,
            payload=dict(payload),
            provenance=dict(provenance) if provenance is not None else None,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "artifact_type": self.artifact_type,
            "source": self.source,
            "created_at": self.created_at,
            "provenance": _jsonable(self.provenance) if self.provenance is not None else {},
            "payload": _jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningArtifact":
        if not isinstance(data, Mapping):
            raise TypeError("artifact must be a mapping")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported artifact schema: {data.get('schema')!r}")
        if data.get("version") != VERSION:
            raise ValueError(f"unsupported artifact version: {data.get('version')!r}")
        payload = data.get("payload")
        provenance = data.get("provenance", {})
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a JSON object")
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a JSON object")
        return cls(
            artifact_type=data.get("artifact_type"),
            source=data.get("source"),
            payload=payload,
            provenance=provenance,
            created_at=data.get("created_at"),
            schema=data.get("schema"),
            version=data.get("version"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "LearningArtifact":
        return cls.from_dict(json.loads(value))


class LearningArtifactStore:
    """Append-only JSONL storage for independently portable artifacts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, artifact: LearningArtifact) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(artifact.to_json() + "\n")

    def record_once(self, artifact: LearningArtifact, idempotency_key: str) -> str:
        """Persist an artifact once, rejecting conflicting reuse of its key."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        with self._lock:
            if self.path.exists():
                with self.path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            existing = LearningArtifact.from_json(line)
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            raise RuntimeError(
                                f"cannot verify artifact idempotency: malformed record at line {line_number}"
                            ) from exc
                        key_run_id = (
                            idempotency_key[len(existing.artifact_type) + 1:]
                            if idempotency_key.startswith(existing.artifact_type + ":")
                            else None
                        )
                        if (
                            existing.artifact_type != artifact.artifact_type
                            or existing.payload.get("run_id") != key_run_id
                        ):
                            continue
                        if existing.to_json() == artifact.to_json():
                            # A prior append may have reached the page cache even if its
                            # fsync failed. Re-sync before acknowledging this retry.
                            with self.path.open("ab") as handle:
                                handle.flush()
                                os.fsync(handle.fileno())
                            return "duplicate"
                        raise ValueError("idempotency key already exists with different artifact content")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(artifact.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return "stored"

    def read(self, limit: int | None = None) -> list[LearningArtifact]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        if not self.path.exists():
            return []

        if limit is None:
            artifacts: list[LearningArtifact] = []
        else:
            artifacts = deque(maxlen=limit)  # type: ignore[assignment]

        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    artifact = LearningArtifact.from_json(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                artifacts.append(artifact)

        return list(artifacts)
