"""Unsupervised anomaly detection over operational telemetry.

Data source: the ops ledgers written since the telemetry work landed::

    ~/.FlossWare/ledger/ops/gateway_forward.jsonl
    ~/.FlossWare/ledger/ops/embedding_batch.jsonl
    (directory override: POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR)

Features are the current latency and status, an error indicator, and per-source
rolling means/stds over the preceding ``window`` observations::

    latency_ms, status, is_error,
    latency_roll_mean, latency_roll_std, error_rate_roll,
    source one-hots

An :class:`~sklearn.ensemble.IsolationForest` learns the normal shape of those
features.  ``detect_anomalies`` returns ``(timestamp, metric, score)`` tuples for
rows the forest flags; ``score`` is ``-score_samples`` so larger means more
anomalous, and ``metric`` is the feature with the largest deviation from that
source's training distribution.  The detector is unsupervised: the metric is
an explanation of *why the row looks unusual*, not a ground-truth label.

The CLI's ``anomaly check`` is intentionally cheap enough for a systemd timer:
loading and fitting a few thousand rows takes well under a second on CPU.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from learning.train.predictors import base

SCHEMA_VERSION = 1
ARTIFACT_MAGIC = "claude-ensemble-anomaly-detector"
DEFAULT_OPS_DIR = Path.home() / ".FlossWare" / "ledger" / "ops"
OPS_DIR_ENV = "POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR"
DEFAULT_SOURCES: Tuple[str, ...] = ("gateway_forward", "embedding_batch")
MIN_ROWS = 10
UNKNOWN_SOURCE = "__unseen__"

#: Features whose deviation is meaningful enough to name in a finding.
ATTRIBUTABLE_FEATURES: Tuple[str, ...] = (
    "latency_ms",
    "status",
    "is_error",
    "latency_roll_mean",
    "latency_roll_std",
    "error_rate_roll",
)


class AnomalyFinding(NamedTuple):
    """The public detection result shape: timestamp, metric, score."""

    timestamp: str
    metric: str
    score: float


@dataclass(frozen=True)
class TelemetryRow:
    """One normalised telemetry observation."""

    timestamp: str
    source: str
    latency_ms: float
    status: int
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "status": self.status,
        }


@dataclass
class FeaturizedWindow:
    rows: List[TelemetryRow]
    matrix: np.ndarray
    feature_names: List[str]
    rolling: List[Dict[str, float]]


@dataclass
class AnomalyDetail:
    """A finding plus the context an operator needs to investigate it."""

    timestamp: str
    metric: str
    score: float
    source: str
    latency_ms: float
    status: int
    rolling: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metric": self.metric,
            "score": self.score,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "rolling": self.rolling,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def ops_dir(path: Optional[os.PathLike | str] = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get(OPS_DIR_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_OPS_DIR


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp, treating a missing zone as UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_from_mapping(document: Mapping[str, Any], default_source: str = "unknown") -> Optional[TelemetryRow]:
    timestamp = document.get("ts", document.get("timestamp"))
    if parse_timestamp(timestamp) is None:
        return None
    source = str(document.get("op") or document.get("source") or default_source)
    return TelemetryRow(
        timestamp=str(timestamp),
        source=source,
        latency_ms=max(0.0, _number(document.get("latency_ms"))),
        status=int(_number(document.get("status"))),
        raw=dict(document),
    )


def load_telemetry(
    directory: Optional[os.PathLike | str] = None,
    *,
    sources: Optional[Sequence[str]] = None,
    per_source_limit: Optional[int] = None,
) -> List[TelemetryRow]:
    """Load and chronologically sort telemetry rows from the ops ledgers.

    Missing files are not an error (the gateway may never have run on this
    host); malformed lines are skipped.  ``per_source_limit`` keeps the most
    recent N lines *per file* before the global merge.
    """
    root = ops_dir(directory)
    selected = tuple(sources) if sources is not None else DEFAULT_SOURCES
    rows: List[TelemetryRow] = []
    for source in selected:
        path = root / f"{source}.jsonl"
        try:
            handle = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            continue
        parsed: List[TelemetryRow] = []
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    document = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(document, Mapping):
                    continue
                row = _row_from_mapping(document, default_source=source)
                if row is None:
                    continue
                parsed.append(row)
        if per_source_limit is not None and per_source_limit > 0:
            parsed = parsed[-per_source_limit:]
        rows.extend(parsed)

    def sort_key(item: Tuple[int, TelemetryRow]) -> Tuple[datetime, int]:
        index, row = item
        return (parse_timestamp(row.timestamp) or datetime.min.replace(tzinfo=timezone.utc), index)

    return [row for _, row in sorted(enumerate(rows), key=sort_key)]


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


def _source_columns(sources: Sequence[str]) -> List[str]:
    return [f"source={name}" for name in sources] + [f"source={UNKNOWN_SOURCE}"]


def featurize(
    rows: Sequence[TelemetryRow],
    *,
    sources: Optional[Sequence[str]] = None,
    window: int = 25,
) -> FeaturizedWindow:
    """Build the rolling feature matrix for a chronological row sequence.

    Rolling statistics are kept per source, so a fast in-process gateway call
    is never compared to a slow embedding batch.  The current observation is
    included in its own rolling window; with ``window=1`` the rolling mean
    equals the current latency and the std is zero.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    categories = tuple(sources) if sources is not None else tuple(
        sorted({row.source for row in rows})
    )
    histories: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=window)
    )
    feature_names = list(ATTRIBUTABLE_FEATURES) + _source_columns(categories)
    matrix: List[List[float]] = []
    rolling_rows: List[Dict[str, float]] = []

    for row in rows:
        is_error = 1.0 if row.status >= 400 else 0.0
        history = histories[row.source]
        history.append((row.latency_ms, is_error))
        latencies = np.fromiter((latency for latency, _ in history), dtype=np.float64)
        errors = np.fromiter((error for _, error in history), dtype=np.float64)
        rolling = {
            "latency_roll_mean": float(latencies.mean()),
            "latency_roll_std": float(latencies.std(ddof=0)),
            "error_rate_roll": float(errors.mean()),
        }
        vector = [
            float(row.latency_ms),
            float(row.status),
            is_error,
            rolling["latency_roll_mean"],
            rolling["latency_roll_std"],
            rolling["error_rate_roll"],
        ]
        source_vector = [0.0] * (len(categories) + 1)
        if row.source in categories:
            source_vector[list(categories).index(row.source)] = 1.0
        else:
            source_vector[-1] = 1.0
        matrix.append(vector + source_vector)
        rolling_rows.append(rolling)

    return FeaturizedWindow(
        rows=list(rows),
        matrix=np.asarray(matrix, dtype=np.float64)
        if matrix
        else np.empty((0, len(feature_names)), dtype=np.float64),
        feature_names=feature_names,
        rolling=rolling_rows,
    )


def _per_source_stats(window: FeaturizedWindow) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean/std per source per attributable feature, used for attribution."""
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(window.rows):
        grouped[row.source].append(index)
    grouped["__global__"] = list(range(len(window.rows)))

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    columns = {name: position for position, name in enumerate(window.feature_names)}
    for source, indices in grouped.items():
        source_stats: Dict[str, Dict[str, float]] = {}
        for name in ATTRIBUTABLE_FEATURES:
            column = window.matrix[indices, columns[name]]
            source_stats[name] = {
                "mean": float(column.mean()) if len(column) else 0.0,
                "std": float(column.std(ddof=0)) if len(column) else 0.0,
            }
        stats[source] = source_stats
    return stats


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """IsolationForest over rolling ops-telemetry features."""

    KIND = "anomaly"
    DISPLAY = "anomaly detector"

    def __init__(
        self,
        window: int = 25,
        contamination: str | float = 0.01,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.contamination = contamination
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.model_: Optional[IsolationForest] = None
        self.sources_: Tuple[str, ...] = ()
        self._rows: List[TelemetryRow] = []
        self._features: Optional[FeaturizedWindow] = None
        self._stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.metadata_: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    @property
    def feature_names(self) -> List[str]:
        return list(self._features.feature_names) if self._features else []

    def fit(
        self,
        rows: Optional[Sequence[TelemetryRow]] = None,
        *,
        directory: Optional[os.PathLike | str] = None,
        sources: Optional[Sequence[str]] = None,
        per_source_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Learn normal telemetry from ``rows`` or directly from the ops ledger."""
        if rows is None:
            rows = load_telemetry(
                directory, sources=sources, per_source_limit=per_source_limit
            )
        ordered = _ordered(rows)
        if len(ordered) < MIN_ROWS:
            raise base.InsufficientDataError(
                f"anomaly detector needs at least {MIN_ROWS} telemetry rows, got {len(ordered)}"
            )

        self.sources_ = tuple(sorted({row.source for row in ordered}))
        window = featurize(ordered, sources=self.sources_, window=self.window)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(window.matrix)
        self.model_ = model
        self._rows = ordered
        self._features = window
        self._stats = _per_source_stats(window)

        flags = model.predict(window.matrix) == -1
        flagged = int(np.sum(flags))
        timestamps = [row.timestamp for row in ordered]
        warnings: List[str] = []
        if len(ordered) < 200:
            warnings.append(
                f"small telemetry sample ({len(ordered)} rows); normal patterns and thresholds are unstable"
            )
        warnings.append(
            "unsupervised detector: findings are distributional outliers, not confirmed incidents"
        )
        self.metadata_ = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "display": self.DISPLAY,
            "trained_at": base.utc_now(),
            "rows": len(ordered),
            "sources": list(self.sources_),
            "source_counts": {
                source: sum(1 for row in ordered if row.source == source)
                for source in self.sources_
            },
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
            "window": self.window,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "feature_names": list(window.feature_names),
            "flagged_in_training": flagged,
            "anomaly_rate_in_training": flagged / len(ordered),
            "metrics": {
                "flagged_in_training": flagged,
                "anomaly_rate_in_training": flagged / len(ordered),
            },
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "warnings": warnings,
        }
        return self.metadata_

    def _select(
        self, window: Optional[int | Sequence[TelemetryRow] | Sequence[Mapping[str, Any]]]
    ) -> FeaturizedWindow:
        if not self.is_fitted or self._features is None:
            raise base.NotTrainedError("anomaly detector is not trained")
        if window is None:
            return self._features
        if isinstance(window, int):
            if window < 1:
                raise ValueError("window must be >= 1")
            features = self._features
            start = max(0, len(features.rows) - window)
            return FeaturizedWindow(
                rows=features.rows[start:],
                matrix=features.matrix[start:],
                feature_names=features.feature_names,
                rolling=features.rolling[start:],
            )
        rows: List[TelemetryRow] = []
        for item in window:
            if isinstance(item, TelemetryRow):
                rows.append(item)
            elif isinstance(item, Mapping):
                row = _row_from_mapping(item)
                if row is not None:
                    rows.append(row)
            else:
                raise ValueError("window items must be TelemetryRow or mapping instances")
        if not rows:
            return FeaturizedWindow([], np.empty((0, len(self.feature_names))), self.feature_names, [])
        return featurize(rows, sources=self.sources_, window=self.window)

    def decision_scores(
        self, window: Optional[int | Sequence[TelemetryRow] | Sequence[Mapping[str, Any]]] = None
    ) -> np.ndarray:
        """Anomaly score per row (``-score_samples``: larger is stranger)."""
        selected = self._select(window)
        if len(selected.rows) == 0:
            return np.empty(0, dtype=np.float64)
        assert self.model_ is not None
        return -self.model_.score_samples(selected.matrix)

    def _attribute(self, row: TelemetryRow, rolling: Mapping[str, float]) -> str:
        stats = self._stats.get(row.source) or self._stats.get("__global__") or {}
        values = {
            "latency_ms": float(row.latency_ms),
            "status": float(row.status),
            "is_error": 1.0 if row.status >= 400 else 0.0,
            "latency_roll_mean": float(rolling.get("latency_roll_mean", 0.0)),
            "latency_roll_std": float(rolling.get("latency_roll_std", 0.0)),
            "error_rate_roll": float(rolling.get("error_rate_roll", 0.0)),
        }
        metric = "latency_ms"
        largest = -1.0
        for name in ATTRIBUTABLE_FEATURES:
            mean = float(stats.get(name, {}).get("mean", 0.0))
            std = float(stats.get(name, {}).get("std", 0.0))
            # A zero-variance feature still carries information: any deviation
            # from the learned constant is maximally surprising.
            denominator = std if std > 1e-9 else 1e-9
            deviation = abs(values[name] - mean) / denominator
            if deviation > largest:
                largest = deviation
                metric = name
        return metric

    def detect_details(
        self, window: Optional[int | Sequence[TelemetryRow] | Sequence[Mapping[str, Any]]] = None
    ) -> List[AnomalyDetail]:
        """Flagged rows with source/latency/status context, most anomalous first."""
        selected = self._select(window)
        if len(selected.rows) == 0:
            return []
        assert self.model_ is not None
        scores = -self.model_.score_samples(selected.matrix)
        flags = self.model_.predict(selected.matrix) == -1
        details: List[AnomalyDetail] = []
        for index, flagged in enumerate(flags):
            if not flagged:
                continue
            row = selected.rows[index]
            rolling = selected.rolling[index]
            details.append(
                AnomalyDetail(
                    timestamp=row.timestamp,
                    metric=self._attribute(row, rolling),
                    score=float(scores[index]),
                    source=row.source,
                    latency_ms=row.latency_ms,
                    status=row.status,
                    rolling={
                        "latency_roll_mean": round(rolling.get("latency_roll_mean", 0.0), 3),
                        "latency_roll_std": round(rolling.get("latency_roll_std", 0.0), 3),
                        "error_rate_roll": round(rolling.get("error_rate_roll", 0.0), 3),
                    },
                )
            )
        details.sort(key=lambda item: item.score, reverse=True)
        return details

    def detect_anomalies(
        self, window: Optional[int | Sequence[TelemetryRow] | Sequence[Mapping[str, Any]]] = None
    ) -> List[AnomalyFinding]:
        """Return ``(timestamp, metric, score)`` for each flagged row.

        ``window`` may be ``None`` (all fitted rows), an integer (the most
        recent N fitted rows), or an explicit sequence of telemetry rows.
        """
        return [
            AnomalyFinding(item.timestamp, item.metric, item.score)
            for item in self.detect_details(window)
        ]

    # -- persistence -------------------------------------------------------
    def save(self, path: Optional[os.PathLike | str] = None) -> Path:
        if not self.is_fitted:
            raise base.NotTrainedError("anomaly detector is not trained")
        destination = (
            Path(path).expanduser() if path is not None else base.default_model_path(self.KIND)
        )
        payload = {
            "magic": ARTIFACT_MAGIC,
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "model": self.model_,
            "config": {
                "window": self.window,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
            },
            "sources": list(self.sources_),
            "stats": self._stats,
            "rows": [row.to_dict() for row in self._rows],
            "metadata": self.metadata_,
        }
        base.atomic_write(destination, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        sidecar = destination.with_suffix(".json")
        base.atomic_write(
            sidecar,
            (json.dumps(self.metadata_, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        return destination

    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "AnomalyDetector":
        source = Path(path).expanduser() if path is not None else base.default_model_path(cls.KIND)
        try:
            with open(source, "rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError as exc:
            raise base.NotTrainedError(f"anomaly artifact not found: {source}") from exc
        if not isinstance(payload, dict) or payload.get("magic") != ARTIFACT_MAGIC:
            raise base.PredictorError(f"{source} is not an anomaly-detector artifact")
        config = payload.get("config") or {}
        detector = cls(
            window=int(config.get("window", 25)),
            contamination=config.get("contamination", 0.01),
            n_estimators=int(config.get("n_estimators", 200)),
            random_state=int(config.get("random_state", 42)),
        )
        detector.model_ = payload.get("model")
        detector.sources_ = tuple(payload.get("sources") or ())
        detector._stats = payload.get("stats") or {}
        detector.metadata_ = payload.get("metadata") or {}
        rows: List[TelemetryRow] = []
        for document in payload.get("rows") or []:
            row = _row_from_mapping(document, default_source=str(document.get("source", "unknown")))
            if row is not None:
                rows.append(row)
        detector._rows = _ordered(rows)
        detector._features = featurize(
            detector._rows, sources=detector.sources_, window=detector.window
        )
        return detector

    @classmethod
    def status(cls, path: Optional[os.PathLike | str] = None) -> Dict[str, Any]:
        source = Path(path).expanduser() if path is not None else base.default_model_path(cls.KIND)
        status: Dict[str, Any] = {
            "kind": cls.KIND,
            "display": cls.DISPLAY,
            "available": False,
            "path": str(source),
        }
        if not source.exists():
            status["reason"] = "artifact missing"
            return status
        document = base.read_sidecar(source)
        if document is None:
            status["reason"] = "artifact present but metadata sidecar is missing or unreadable"
            return status
        status.update(
            {
                "available": True,
                "trained_at": document.get("trained_at"),
                "samples": document.get("rows"),
                "sources": document.get("sources"),
                "source_counts": document.get("source_counts"),
                "first_timestamp": document.get("first_timestamp"),
                "last_timestamp": document.get("last_timestamp"),
                "metrics": document.get("metrics") or {},
                "warnings": document.get("warnings") or [],
            }
        )
        return status


def _ordered(rows: Iterable[TelemetryRow]) -> List[TelemetryRow]:
    """Stable chronological order; rows with unparseable timestamps sort first."""
    indexed = list(enumerate(rows))
    floor = datetime.min.replace(tzinfo=timezone.utc)
    indexed.sort(key=lambda item: (parse_timestamp(item[1].timestamp) or floor, item[0]))
    return [row for _, row in indexed]
