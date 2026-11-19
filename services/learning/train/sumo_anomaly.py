"""Anomaly detection on Disseminator search metrics from Sumo Logic.

The pipeline has three stages, matching the CLI subcommands:

``fetch``
    Query the Sumo Search Job API for per-timeslice search metrics: p95 and
    average processing latency, request count, HTTP error count, and the
    zero-result rate. Aggregate queries return records from
    ``/search/jobs/{id}/records`` (raw messages are never requested).

``train``
    Fit an unsupervised :class:`~sklearn.ensemble.IsolationForest` over the
    metric vector of every timeslice. IsolationForest learns the joint shape of
    normal traffic instead of applying a fixed per-metric threshold.

``check``
    Score the stored series, report abnormal timeslices, which metrics drove
    each score (robust median/MAD deviations), and whether a finding is recent.
    A finding is flagged when the forest labels it **or** a metric crosses the
    univariate flag threshold (default 4σ); deviating metrics are named at the
    lower explanation threshold (default 3σ).

Security notes
--------------
* Credentials come from the process environment or the ``env`` block of
  ``~/.claude/settings.json``. They are never hardcoded, printed, or persisted
  in the series/artifact.
* Sumo responses are untrusted data. Only the numeric metric fields are
  extracted; ``_raw`` log content is never stored or logged.
* The series file and the joblib artifact are written with mode ``0600``.

Run ``PYTHONPATH=services python3 -m learning.train.sumo_anomaly --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import requests
from sklearn.ensemble import IsolationForest

from learning.train import paths
from learning.train.base import (
    FeatureEncoder,
    TrainedModel,
    TrainingError,
    canonical_json,
    sha256_text,
    utc_now,
)

ARTIFACT_NAME = "sumo_anomaly"
SERIES_FORMAT = "learning.train.sumo_anomaly.series.v1"
DEFAULT_BASE_URL = "https://api.sumologic.com/api/v1"
DEFAULT_INDEX = "rh_cpav-001"
DEFAULT_DAYS = 7
DEFAULT_TIMESLICE = "5m"
DEFAULT_SETTINGS_PATH = "~/.claude/settings.json"
APP_SOURCE_CATEGORY = "disseminator.log"

#: Free-text token for the search endpoint. The access log carries
#: ``path="/rs/search"`` as a key/value pair, but there is no indexed ``path``
#: field, so a ``path=...`` term matches nothing; quoting the token matches.
SEARCH_PATH_TOKEN = '"/rs/search"'

#: Features fed to the model. ``zero_result_rate`` is only populated when the
#: application log is available; the encoder imputes it otherwise.
FEATURE_NAMES = ("p95_ms", "avg_ms", "request_count", "error_count", "zero_result_rate")
COUNT_FEATURES = frozenset({"request_count", "error_count"})

#: Minimum number of timeslices before a fit is meaningful.
MIN_POINTS = 12
#: IsolationForest cannot rank a feature that never varies; a tiny jitter keeps
#: a constant column numerically harmless and the encoder deterministic.
_CONSTANT_JITTER = 1e-9
#: Deviation cap so an infinite robust z-score is still printable/serialisable.
MAX_DEVIATION = 999.0

_INDEX_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
_TIMESLICE_RE = re.compile(r"^(\d{1,4})([smhd])$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{4,128}$")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_TIMESLICE_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400}


def _timeslice_delta(timeslice: str) -> timedelta:
    match = _TIMESLICE_RE.match(timeslice)
    if not match:
        raise ValueError("timeslice must look like 5m, 1h, or 30s")
    return timedelta(seconds=int(match.group(1)) * _TIMESLICE_SECONDS[match.group(2)])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SumoError(Exception):
    """Base error for the Sumo anomaly pipeline."""


class SumoCredentialError(SumoError):
    """Credentials are missing or unusable (no secret value is ever included)."""


class SumoAPIError(SumoError):
    """The Sumo API rejected a request or a search job failed."""


class SumoDataError(SumoError):
    """The API response was malformed or contained no usable metric rows."""


class InsufficientSeriesError(TrainingError):
    """Too few timeslices to fit an IsolationForest."""


# ---------------------------------------------------------------------------
# Configuration and credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SumoConfig:
    """Validated query configuration. Credentials are deliberately *not* here."""

    index: str = DEFAULT_INDEX
    days: int = DEFAULT_DAYS
    timeslice: str = DEFAULT_TIMESLICE
    base_url: str = DEFAULT_BASE_URL
    zero_source_category: str = APP_SOURCE_CATEGORY
    poll_interval: float = 2.0
    job_timeout: float = 1800.0
    request_timeout: float = 60.0
    max_retries: int = 4
    retry_backoff: float = 5.0
    page_size: int = 1000
    max_records: int = 50_000
    recent_minutes: int = 60

    def __post_init__(self) -> None:
        if not _INDEX_RE.match(self.index):
            raise ValueError("index must match [A-Za-z0-9_.-]{1,128}")
        if not _TIMESLICE_RE.match(self.timeslice):
            raise ValueError("timeslice must look like 5m, 1h, or 30s")
        if not _INDEX_RE.match(self.zero_source_category):
            raise ValueError("zero_source_category must be a plain source category")
        if self.days <= 0:
            raise ValueError("days must be positive")
        if not self.base_url.startswith("https://"):
            raise ValueError("base_url must use https")
        if self.poll_interval <= 0 or self.job_timeout <= 0:
            raise ValueError("poll_interval and job_timeout must be positive")
        if not 0 < self.page_size <= 10_000:
            raise ValueError("page_size must be between 1 and 10000")
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.recent_minutes <= 0:
            raise ValueError("recent_minutes must be positive")

    def window(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        """The query window, floored to a timeslice boundary.

        Ending on a boundary keeps the final bucket complete: a window ending
        mid-bucket makes Sumo return a partial slice whose low counts look like
        an anomaly (e.g. 320 requests instead of ~12,000).
        """
        end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
        step = int(_timeslice_delta(self.timeslice).total_seconds())
        if step <= 0:
            raise ValueError("timeslice must be positive")
        end = datetime.fromtimestamp((int(end.timestamp()) // step) * step, tz=timezone.utc)
        return end - timedelta(days=self.days), end


def load_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    settings_path: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Return ``(access_id, access_key)`` from the environment or settings.json.

    The environment wins so an operator can override without editing the
    settings file. Errors never include the secret values.
    """
    env = os.environ if environ is None else environ
    access_id = str(env.get("SUMO_ACCESS_ID") or "").strip()
    access_key = str(env.get("SUMO_ACCESS_KEY") or "").strip()
    if access_id and access_key:
        return access_id, access_key

    path = Path(os.path.expanduser(settings_path or DEFAULT_SETTINGS_PATH))
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise SumoCredentialError(
            "Sumo credentials not found: export SUMO_ACCESS_ID and SUMO_ACCESS_KEY "
            f"or add them to the env block of {path}"
        ) from None
    except (OSError, json.JSONDecodeError):
        raise SumoCredentialError(
            f"could not read Sumo credentials from {path} (file unreadable or not JSON)"
        ) from None

    block = document.get("env") if isinstance(document, dict) else None
    if not isinstance(block, dict):
        raise SumoCredentialError(f"{path} has no 'env' object with Sumo credentials")
    access_id = str(block.get("SUMO_ACCESS_ID") or "").strip()
    access_key = str(block.get("SUMO_ACCESS_KEY") or "").strip()
    if not access_id or not access_key:
        raise SumoCredentialError(f"{path} env block is missing SUMO_ACCESS_ID or SUMO_ACCESS_KEY")
    return access_id, access_key


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def latency_query(config: SumoConfig) -> str:
    """p95 and average server processing time for search requests."""
    return (
        f'_index={config.index} {SEARCH_PATH_TOKEN} '
        '| parse "processed-time-milliseconds=\\"*\\"" as processed_time '
        "| num(processed_time) "
        f"| timeslice {config.timeslice} "
        "| pct(processed_time, 95) as p95_ms, avg(processed_time) as avg_ms by _timeslice"
    )


def request_query(config: SumoConfig) -> str:
    """Total search requests per timeslice (the series spine)."""
    return (
        f"_index={config.index} {SEARCH_PATH_TOKEN} "
        f"| timeslice {config.timeslice} "
        "| count as request_count by _timeslice"
    )


def error_query(config: SumoConfig) -> str:
    """Search requests whose access-log HTTP status is 4xx/5xx."""
    return (
        f'_index={config.index} {SEARCH_PATH_TOKEN} '
        '| parse "status=\\"*\\"" as status_code '
        "| num(status_code) "
        "| where status_code >= 400 "
        f"| timeslice {config.timeslice} "
        "| count as error_count by _timeslice"
    )


def zero_result_query(config: SumoConfig) -> str:
    """Zero-result rate source: ``resultCount`` from the Disseminator app log."""
    return (
        f"_index={config.index} _sourceCategory={config.zero_source_category} {SEARCH_PATH_TOKEN} "
        "| json auto "
        "| if(resultCount = 0, 1, 0) as is_zero "
        f"| timeslice {config.timeslice} "
        "| sum(is_zero) as zero_result_count, count as app_request_count by _timeslice"
    )


def build_queries(config: SumoConfig) -> dict[str, str]:
    """All aggregate queries, in execution order."""
    return {
        "latency": latency_query(config),
        "requests": request_query(config),
        "errors": error_query(config),
        "zero_results": zero_result_query(config),
    }


# ---------------------------------------------------------------------------
# Sumo Search Job API client
# ---------------------------------------------------------------------------


class SearchClient(Protocol):
    """Minimal interface the fetcher needs (tests inject a fake)."""

    def run_search(
        self, query: str, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:  # pragma: no cover - protocol
        ...


def _json_object(response: requests.Response) -> dict[str, Any]:
    """Parse an API response as a JSON object, without echoing its body."""
    try:
        payload = response.json()
    except ValueError:
        raise SumoDataError(
            f"Sumo API returned non-JSON content (HTTP {response.status_code})"
        ) from None
    if not isinstance(payload, dict):
        raise SumoDataError(f"Sumo API returned {type(payload).__name__}, expected an object")
    return payload


def _api_error_message(response: requests.Response) -> str:
    """Sanitised error: HTTP status plus the API's own code/message metadata.

    The response body is never copied verbatim: an upstream error page could
    contain log content. Only short, scalar error metadata is used.
    """
    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        return f"Sumo API request failed with HTTP {status}"
    if not isinstance(payload, dict):
        return f"Sumo API request failed with HTTP {status}"
    code = payload.get("code")
    message = payload.get("message")
    parts = [f"HTTP {status}"]
    if isinstance(code, str) and code:
        parts.append(code[:120])
    if isinstance(message, str) and message:
        parts.append(message[:200])
    return "Sumo API request failed: " + " - ".join(parts)


def _retry_after(response: requests.Response, fallback: float) -> float:
    value = response.headers.get("Retry-After")
    if not value:
        return fallback
    try:
        return max(0.0, min(float(value), 300.0))
    except (TypeError, ValueError):
        return fallback


class SumoClient:
    """Small Search Job API client with polling, pagination, and 429 backoff.

    Only aggregate queries are used, so results are read from the ``/records``
    endpoint. Raw messages are never requested (``requireRawMessages`` is not
    set), which keeps log content out of the process.
    """

    def __init__(
        self,
        access_id: str,
        access_key: str,
        *,
        config: SumoConfig | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not access_id or not access_key:
            raise SumoCredentialError("Sumo access id and key are required")
        self.config = config or SumoConfig()
        self._session = session or requests.Session()
        self._session.auth = (access_id, access_key)
        self._session.headers.update({"Accept": "application/json"})
        self._sleep = sleep

    # -- HTTP ------------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        last_error: str = ""
        for attempt in range(self.config.max_retries):
            try:
                response = self._session.request(
                    method, url, timeout=self.config.request_timeout, **kwargs
                )
            except requests.RequestException as error:
                last_error = type(error).__name__
                if attempt + 1 >= self.config.max_retries:
                    raise SumoAPIError(f"Sumo API request failed: {last_error}") from None
                self._sleep(self.config.retry_backoff * (2**attempt))
                continue
            if response.status_code == 429:
                last_error = "HTTP 429"
                if attempt + 1 >= self.config.max_retries:
                    break
                self._sleep(
                    _retry_after(response, self.config.retry_backoff * (2**attempt))
                )
                continue
            if response.status_code >= 500 and attempt + 1 < self.config.max_retries:
                last_error = f"HTTP {response.status_code}"
                self._sleep(self.config.retry_backoff * (2**attempt))
                continue
            if response.status_code >= 400:
                raise SumoAPIError(_api_error_message(response))
            return response
        raise SumoAPIError(f"Sumo API request failed after retries ({last_error or 'unknown'})")

    # -- search lifecycle ------------------------------------------------

    def run_search(self, query: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Create a search job, wait for it, and return aggregate record maps."""
        job_id = self._create_job(query, start, end)
        try:
            self._wait_for_job(job_id)
            return self._fetch_records(job_id)
        except Exception:
            self._cancel_job(job_id)
            raise

    def _create_job(self, query: str, start: datetime, end: datetime) -> str:
        payload = {
            "query": query,
            "from": _api_time(start),
            "to": _api_time(end),
            "timeZone": "UTC",
        }
        response = self._request(
            "POST", f"{self.config.base_url}/search/jobs", json=payload
        )
        document = _json_object(response)
        job_id = document.get("id")
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            raise SumoDataError("Sumo API response did not contain a valid job id")
        return job_id

    def _wait_for_job(self, job_id: str) -> None:
        deadline = time.monotonic() + self.config.job_timeout
        while True:
            response = self._request(
                "GET", f"{self.config.base_url}/search/jobs/{job_id}"
            )
            document = _json_object(response)
            state = document.get("state")
            if not isinstance(state, str):
                raise SumoDataError("Sumo search status did not contain a state")
            upper = state.upper()
            if "DONE" in upper:
                return
            if "CANCELLED" in upper or "FAILED" in upper:
                raise SumoAPIError(f"Sumo search job ended in state {state!r}")
            if time.monotonic() >= deadline:
                raise SumoAPIError(
                    f"Sumo search job did not finish within {self.config.job_timeout:.0f}s "
                    f"(last state {state!r})"
                )
            self._sleep(self.config.poll_interval)

    def _fetch_records(self, job_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset = 0
        while offset < self.config.max_records:
            limit = min(self.config.page_size, self.config.max_records - offset)
            response = self._request(
                "GET",
                f"{self.config.base_url}/search/jobs/{job_id}/records",
                params={"offset": offset, "limit": limit},
            )
            document = _json_object(response)
            batch = document.get("records")
            if batch is None:
                batch = []
            if not isinstance(batch, list):
                raise SumoDataError("Sumo records response was not a list")
            for item in batch:
                if not isinstance(item, dict):
                    continue
                mapping = item.get("map")
                if isinstance(mapping, dict):
                    records.append(mapping)
            if len(batch) < limit:
                break
            offset += len(batch)
        return records

    def _cancel_job(self, job_id: str) -> None:
        """Best-effort cancellation; never masks the original failure."""
        try:
            self._session.delete(
                f"{self.config.base_url}/search/jobs/{job_id}",
                timeout=self.config.request_timeout,
            )
        except requests.RequestException:
            pass


def _api_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Metric series
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Coerce untrusted API values to a finite float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not _NUMERIC_RE.match(text):
            return None
        number = float(text)
    else:
        return None
    return number if math.isfinite(number) else None


def _safe_count(value: Any, default: float = 0.0) -> float:
    number = _safe_float(value)
    if number is None or number < 0:
        return default
    return number


def _float_or_none(value: Any) -> float | None:
    number = _safe_float(value)
    return None if number is None else float(number)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _timeslice_iso(value: Any) -> str | None:
    """Normalise Sumo's ``_timeslice`` (epoch milliseconds) to ISO-8601 UTC."""
    milliseconds: float | None = None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        milliseconds = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _NUMERIC_RE.match(text):
            milliseconds = float(text)
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return _iso_seconds(parsed)
    if milliseconds is None or not math.isfinite(milliseconds):
        return None
    # Sumo returns epoch milliseconds; tolerate seconds from a future API.
    seconds = milliseconds / 1000.0 if abs(milliseconds) > 1e11 else milliseconds
    try:
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if not 946684800 <= parsed.timestamp() <= 4102444800:  # 2000..2100
        return None
    return _iso_seconds(parsed)


def _iso_seconds(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class MetricPoint:
    """One timeslice of search metrics. All fields are already validated."""

    timestamp: str
    p95_ms: float | None = None
    avg_ms: float | None = None
    request_count: float = 0.0
    error_count: float = 0.0
    zero_result_count: float | None = None
    app_request_count: float | None = None
    zero_result_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        # Cast every numeric field to float so a save/load round-trip keeps the
        # exact same canonical form (and therefore the same data hash).
        return {
            "timestamp": self.timestamp,
            "p95_ms": _float_or_none(self.p95_ms),
            "avg_ms": _float_or_none(self.avg_ms),
            "request_count": float(self.request_count),
            "error_count": float(self.error_count),
            "zero_result_count": _float_or_none(self.zero_result_count),
            "app_request_count": _float_or_none(self.app_request_count),
            "zero_result_rate": _float_or_none(self.zero_result_rate),
        }

    def as_features(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}


@dataclass
class MetricSeries:
    """A fetched window of metric points plus non-secret fetch metadata."""

    points: list[MetricPoint]
    index: str
    window_start: str
    window_end: str
    timeslice: str
    fetched_at: str = field(default_factory=utc_now)
    query_records: dict[str, int] = field(default_factory=dict)
    dropped_partial: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": SERIES_FORMAT,
            "fetched_at": self.fetched_at,
            "index": self.index,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "timeslice": self.timeslice,
            "query_records": dict(self.query_records),
            "dropped_partial": int(self.dropped_partial),
            "points": [point.as_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricSeries":
        if not isinstance(payload, dict) or payload.get("format") != SERIES_FORMAT:
            raise SumoDataError("not a sumo_anomaly series file")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise SumoDataError("series file has no points list")
        points = [_point_from_dict(item) for item in raw_points if isinstance(item, dict)]
        return cls(
            points=[point for point in points if point is not None],
            index=str(payload.get("index") or DEFAULT_INDEX),
            window_start=str(payload.get("window_start") or ""),
            window_end=str(payload.get("window_end") or ""),
            timeslice=str(payload.get("timeslice") or DEFAULT_TIMESLICE),
            fetched_at=str(payload.get("fetched_at") or ""),
            query_records={
                str(key): int(value)
                for key, value in (payload.get("query_records") or {}).items()
                if isinstance(value, (int, float))
            },
            dropped_partial=_safe_int(payload.get("dropped_partial")),
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        return _write_private_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MetricSeries":
        source = Path(path)
        try:
            with open(source, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise SumoDataError(f"series file not found: {source}") from None
        except (OSError, json.JSONDecodeError):
            raise SumoDataError(f"series file is unreadable or not JSON: {source}") from None
        return cls.from_dict(payload)

    def data_hash(self) -> str:
        return sha256_text(canonical_json([point.as_dict() for point in self.points]))


def _point_from_dict(item: Mapping[str, Any]) -> MetricPoint | None:
    timestamp = _timeslice_iso(item.get("timestamp"))
    if timestamp is None:
        return None
    zero_count = _safe_float(item.get("zero_result_count"))
    app_count = _safe_float(item.get("app_request_count"))
    rate = _safe_float(item.get("zero_result_rate"))
    if rate is not None:
        rate = min(max(rate, 0.0), 1.0)
    return MetricPoint(
        timestamp=timestamp,
        p95_ms=_safe_float(item.get("p95_ms")),
        avg_ms=_safe_float(item.get("avg_ms")),
        request_count=_safe_count(item.get("request_count")),
        error_count=_safe_count(item.get("error_count")),
        zero_result_count=None if zero_count is None else max(zero_count, 0.0),
        app_request_count=None if app_count is None else max(app_count, 0.0),
        zero_result_rate=rate,
    )


def _write_private_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temporary, destination)
    return destination


def merge_records(
    latency_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    error_records: Sequence[Mapping[str, Any]],
    zero_records: Sequence[Mapping[str, Any]],
) -> list[MetricPoint]:
    """Join the four aggregate result sets on ``_timeslice``.

    An absent error row means zero errors in that slice; an absent latency row
    is left as ``None`` and imputed later by the encoder. Only numeric fields
    are read, so unexpected/log-bearing keys in an API response are ignored.
    """
    buckets: dict[str, dict[str, Any]] = {}

    def bucket(timestamp: str) -> dict[str, Any]:
        return buckets.setdefault(timestamp, {})

    for record in latency_records:
        timestamp = _timeslice_iso(record.get("_timeslice"))
        if timestamp is None:
            continue
        p95 = _safe_float(record.get("p95_ms"))
        average = _safe_float(record.get("avg_ms"))
        if p95 is None and average is None:
            continue
        entry = bucket(timestamp)
        entry["p95_ms"] = p95
        entry["avg_ms"] = average

    for record in request_records:
        timestamp = _timeslice_iso(record.get("_timeslice"))
        if timestamp is None:
            continue
        bucket(timestamp)["request_count"] = _safe_count(record.get("request_count"))

    for record in error_records:
        timestamp = _timeslice_iso(record.get("_timeslice"))
        if timestamp is None:
            continue
        bucket(timestamp)["error_count"] = _safe_count(record.get("error_count"))

    for record in zero_records:
        timestamp = _timeslice_iso(record.get("_timeslice"))
        if timestamp is None:
            continue
        zero_count = _safe_count(record.get("zero_result_count"))
        app_count = _safe_count(record.get("app_request_count"))
        entry = bucket(timestamp)
        entry["zero_result_count"] = zero_count
        entry["app_request_count"] = app_count
        entry["zero_result_rate"] = (zero_count / app_count) if app_count > 0 else None

    points: list[MetricPoint] = []
    for timestamp in sorted(buckets):
        entry = buckets[timestamp]
        # A bucket with neither latency nor traffic carries no signal.
        if entry.get("p95_ms") is None and entry.get("avg_ms") is None and not entry.get("request_count"):
            continue
        points.append(
            MetricPoint(
                timestamp=timestamp,
                p95_ms=entry.get("p95_ms"),
                avg_ms=entry.get("avg_ms"),
                request_count=float(entry.get("request_count", 0.0)),
                error_count=float(entry.get("error_count", 0.0)),
                zero_result_count=entry.get("zero_result_count"),
                app_request_count=entry.get("app_request_count"),
                zero_result_rate=entry.get("zero_result_rate"),
            )
        )
    return points


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, int], None]


def fetch_metrics(
    client: SearchClient,
    config: SumoConfig | None = None,
    *,
    now: datetime | None = None,
    progress: ProgressCallback | None = None,
) -> MetricSeries:
    """Run every aggregate query and return a merged, sorted series."""
    config = config or SumoConfig()
    start, end = config.window(now)
    records: dict[str, list[dict[str, Any]]] = {}
    for name, query in build_queries(config).items():
        batch = client.run_search(query, start, end)
        records[name] = list(batch)
        if progress is not None:
            progress(name, len(records[name]))

    points = merge_records(
        records["latency"], records["requests"], records["errors"], records["zero_results"]
    )
    # Defensive: drop buckets that would extend outside the query window. The
    # window is already aligned to a boundary, so this normally removes nothing.
    step = _timeslice_delta(config.timeslice)
    complete: list[MetricPoint] = []
    dropped_partial = 0
    for point in points:
        bucket_start = _parse_iso(point.timestamp)
        if bucket_start is None:
            dropped_partial += 1
            continue
        if bucket_start < start or bucket_start + step > end:
            dropped_partial += 1
            continue
        complete.append(point)
    points = complete
    if not points:
        raise SumoDataError(
            f"Sumo returned no complete metric rows for index {config.index} "
            f"({_iso_seconds(start)}..{_iso_seconds(end)})"
        )
    return MetricSeries(
        points=points,
        index=config.index,
        window_start=_iso_seconds(start),
        window_end=_iso_seconds(end),
        timeslice=config.timeslice,
        fetched_at=utc_now(),
        query_records={name: len(batch) for name, batch in records.items()},
        dropped_partial=dropped_partial,
    )


# ---------------------------------------------------------------------------
# Model: encoder, baseline, detector
# ---------------------------------------------------------------------------


class MetricFeatureEncoder(FeatureEncoder):
    """Fixed-order encoder with median imputation for missing metrics."""

    def __init__(
        self,
        feature_names: Sequence[str] = FEATURE_NAMES,
        imputations: Mapping[str, float] | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.imputations = {str(key): float(value) for key, value in (imputations or {}).items()}

    def fit(self, features: Sequence[Mapping[str, Any]]) -> "MetricFeatureEncoder":
        for name in self.feature_names:
            values = [
                number
                for number in (_safe_float(feature.get(name)) for feature in features)
                if number is not None
            ]
            if name in COUNT_FEATURES and not values:
                self.imputations[name] = 0.0
            else:
                self.imputations[name] = float(np.median(values)) if values else 0.0
        return self

    def transform(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.empty((len(features), len(self.feature_names)), dtype=float)
        for row_index, feature in enumerate(features):
            for column_index, name in enumerate(self.feature_names):
                value = _safe_float(feature.get(name))
                if value is None:
                    value = self.imputations.get(name, 0.0)
                matrix[row_index, column_index] = value
        return matrix


# ``python -m learning.train.sumo_anomaly`` executes this file as ``__main__``.
# Without this, joblib records ``__main__.MetricFeatureEncoder`` and any other
# process (a tool, a test, a later CLI invocation) cannot unpickle the artifact.
MetricFeatureEncoder.__module__ = (
    "learning.train.sumo_anomaly" if __name__ == "__main__" else __name__
)


def compute_baseline(
    features: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Median and robust scale (1.4826 * MAD, std fallback) per feature."""
    baseline: dict[str, dict[str, float]] = {}
    for name in feature_names:
        values = [
            number
            for number in (_safe_float(feature.get(name)) for feature in features)
            if number is not None
        ]
        if values:
            array = np.asarray(values, dtype=float)
            median = float(np.median(array))
            mad = float(np.median(np.abs(array - median)))
        else:
            median = 0.0
            mad = 0.0
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = float(np.std(array)) if values else 0.0
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = 0.0
        baseline[name] = {"median": median, "mad": mad, "scale": scale}
    return baseline


@dataclass(frozen=True)
class MetricDeviation:
    metric: str
    value: float
    baseline: float
    deviation: float
    direction: str
    is_deviating: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": _round(self.value),
            "baseline": _round(self.baseline),
            "deviation": _round(self.deviation),
            "direction": self.direction,
            "is_deviating": self.is_deviating,
        }


@dataclass(frozen=True)
class AnomalyFinding:
    timestamp: str
    anomaly_score: float
    is_anomaly: bool
    recent: bool
    deviations: list[MetricDeviation]
    values: dict[str, float]
    imputed: list[str] = field(default_factory=list)
    model_flagged: bool = False
    metric_flagged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "anomaly_score": _round(self.anomaly_score),
            "is_anomaly": self.is_anomaly,
            "model_flagged": self.model_flagged,
            "metric_flagged": self.metric_flagged,
            "recent": self.recent,
            "deviations": [deviation.as_dict() for deviation in self.deviations],
            "values": {name: _round(value) for name, value in self.values.items()},
            "imputed": list(self.imputed),
        }


def _deviation(
    metric: str, value: float, baseline: Mapping[str, Any], threshold: float
) -> MetricDeviation:
    median = _safe_float(baseline.get("median")) or 0.0
    scale = _safe_float(baseline.get("scale")) or 0.0
    if scale > 1e-9:
        raw = (value - median) / scale
    elif abs(value - median) <= 1e-9:
        raw = 0.0
    else:
        raw = MAX_DEVIATION if value > median else -MAX_DEVIATION
    raw = max(-MAX_DEVIATION, min(MAX_DEVIATION, raw))
    if abs(raw) <= 1e-9:
        direction = "flat"
    else:
        direction = "above" if raw > 0 else "below"
    return MetricDeviation(
        metric=metric,
        value=value,
        baseline=median,
        deviation=raw,
        direction=direction,
        is_deviating=abs(raw) >= threshold,
    )


class SumoAnomalyDetector:
    """IsolationForest wrapper with metric-level explanation and persistence.

    A finding is anomalous when the forest labels it an outlier **or** one of
    its metrics crosses the (higher) univariate flag threshold. The second leg
    catches operationally severe single-metric events (for example traffic
    collapsing to near zero) that normal diurnal variance can hide from a
    multivariate model. Deviating metrics are still *named* at the lower
    ``z_threshold`` so a forest flag remains explainable.
    """

    def __init__(self, trained: TrainedModel) -> None:
        self.trained = trained

    # -- fit -------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        points: Sequence[MetricPoint],
        *,
        series: MetricSeries | None = None,
        config: SumoConfig | None = None,
        contamination: float | str = 0.02,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> "SumoAnomalyDetector":
        if len(points) < MIN_POINTS:
            raise InsufficientSeriesError(
                f"need at least {MIN_POINTS} timeslices to fit, got {len(points)}"
            )
        raw_features = [point.as_features() for point in points]
        encoder = MetricFeatureEncoder().fit(raw_features)
        matrix = encoder.transform(raw_features)
        baseline = compute_baseline(raw_features, encoder.feature_names)

        estimator = IsolationForest(
            n_estimators=max(10, int(n_estimators)),
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        estimator.fit(matrix)

        config = config or SumoConfig()
        data_hash = series.data_hash() if series is not None else sha256_text(
            canonical_json([point.as_dict() for point in points])
        )
        trained = TrainedModel(
            name=ARTIFACT_NAME,
            estimator=estimator,
            encoder=encoder,
            label_name="anomaly",
            label_source="unsupervised:isolation_forest",
            trained_at=utc_now(),
            data_hash=data_hash,
            n_train=len(points),
            n_test=0,
            metrics={},
            extras={
                "index": config.index,
                "window_start": series.window_start if series is not None else "",
                "window_end": series.window_end if series is not None else "",
                "timeslice": config.timeslice,
                "contamination": contamination,
                "n_estimators": int(n_estimators),
                "random_state": random_state,
                "feature_names": list(encoder.feature_names),
                "imputations": dict(encoder.imputations),
                "baseline": baseline,
                "recent_minutes": config.recent_minutes,
            },
        )
        return cls(trained)

    # -- scoring ---------------------------------------------------------

    def score(
        self,
        points: Sequence[MetricPoint],
        *,
        now: datetime | None = None,
        z_threshold: float = 3.0,
        flag_threshold: float = 4.0,
    ) -> list[AnomalyFinding]:
        if not points:
            return []
        raw_features = [point.as_features() for point in points]
        matrix = self.trained.encoder.transform(raw_features)
        estimator = self.trained.estimator
        decisions = np.asarray(estimator.decision_function(matrix), dtype=float)
        predictions = np.asarray(estimator.predict(matrix))
        baseline = self.trained.extras.get("baseline") or {}
        names = list(self.trained.encoder.feature_names)
        recent_minutes = int(self.trained.extras.get("recent_minutes") or 60)
        reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

        findings: list[AnomalyFinding] = []
        for index, point in enumerate(points):
            actual = point.as_features()
            values = {
                name: float(matrix[index, column])
                for column, name in enumerate(names)
            }
            imputed = [
                name for name in names if _safe_float(actual.get(name)) is None
            ]
            deviations = []
            for name in names:
                metric_baseline = baseline.get(name)
                if not isinstance(metric_baseline, dict):
                    metric_baseline = {}
                deviations.append(_deviation(name, values[name], metric_baseline, z_threshold))
            model_flagged = bool(predictions[index] == -1)
            # Hybrid rule: the multivariate forest plus a univariate guard. A
            # single metric can be operationally extreme (e.g. traffic falls to
            # ~0) without isolating as a multivariate outlier, because normal
            # diurnal variance inflates that feature's robust scale. The guard
            # uses a higher threshold than the report's naming threshold: these
            # metrics are heavy-tailed, and 3σ on five features would flag ~20%
            # of real traffic.
            metric_flagged = _metric_flagged(deviations, flag_threshold)
            findings.append(
                AnomalyFinding(
                    timestamp=point.timestamp,
                    # Negate decision_function so positive = more anomalous.
                    anomaly_score=float(-decisions[index]),
                    is_anomaly=model_flagged or metric_flagged,
                    model_flagged=model_flagged,
                    metric_flagged=metric_flagged,
                    recent=_is_recent(point.timestamp, reference, recent_minutes),
                    deviations=deviations,
                    values=values,
                    imputed=imputed,
                )
            )
        return findings

    # -- persistence -----------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> Path:
        return self.trained.save(path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SumoAnomalyDetector":
        return cls(TrainedModel.load(path, name=ARTIFACT_NAME))


def _metric_flagged(
    deviations: Sequence[MetricDeviation], flag_threshold: float
) -> bool:
    """Whether any metric crosses the univariate flag threshold."""
    return any(abs(deviation.deviation) >= flag_threshold for deviation in deviations)


def _is_recent(timestamp: str, now: datetime, minutes: int) -> bool:
    parsed = _parse_iso(timestamp)
    if parsed is None:
        return False
    delta = now - parsed
    # Tolerate a small clock skew into the future, but not a stale series.
    return -timedelta(minutes=5) <= delta <= timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def build_report(
    findings: Sequence[AnomalyFinding],
    series: MetricSeries,
    *,
    detector: SumoAnomalyDetector,
    top: int = 10,
    z_threshold: float = 3.0,
    flag_threshold: float = 4.0,
) -> dict[str, Any]:
    anomalies = [finding for finding in findings if finding.is_anomaly]
    recent = [finding for finding in anomalies if finding.recent]
    ordered = sorted(anomalies, key=lambda finding: finding.anomaly_score, reverse=True)
    metric_counts: dict[str, int] = {}
    imputed_counts: dict[str, int] = {}
    for finding in findings:
        for name in finding.imputed:
            imputed_counts[name] = imputed_counts.get(name, 0) + 1
    for finding in anomalies:
        for deviation in finding.deviations:
            if deviation.is_deviating:
                metric_counts[deviation.metric] = metric_counts.get(deviation.metric, 0) + 1
    extras = detector.trained.extras
    return {
        "index": series.index,
        "timeslice": series.timeslice,
        "window_start": series.window_start,
        "window_end": series.window_end,
        "fetched_at": series.fetched_at,
        "points": len(findings),
        "anomalies": len(anomalies),
        "anomaly_rate": _round(len(anomalies) / len(findings) if findings else 0.0),
        "flagged_by_model": sum(1 for finding in findings if finding.model_flagged),
        "flagged_by_deviation_only": sum(
            1 for finding in findings if finding.metric_flagged and not finding.model_flagged
        ),
        "recent_anomalies": len(recent),
        "recent_minutes": int(extras.get("recent_minutes") or 60),
        "z_threshold": z_threshold,
        "flag_threshold": flag_threshold,
        "features": list(detector.trained.encoder.feature_names),
        "model": {
            "trained_at": detector.trained.trained_at,
            "contamination": extras.get("contamination"),
            "n_estimators": extras.get("n_estimators"),
            "data_hash": detector.trained.data_hash,
        },
        "anomalies_by_metric": metric_counts,
        "imputed_counts": imputed_counts,
        "recent": [finding.as_dict() for finding in recent],
        "top_anomalies": [finding.as_dict() for finding in ordered[: max(0, top)]],
    }


def _print_fetch(series: MetricSeries, path: Path) -> None:
    print(
        f"Sumo anomaly fetch: index={series.index} timeslice={series.timeslice} "
        f"window={series.window_start}..{series.window_end}"
    )
    for name, count in series.query_records.items():
        print(f"  {name}: {count} records")
    print(f"  merged: {len(series.points)} timeslices")
    if series.dropped_partial:
        print(f"  dropped partial edge timeslices: {series.dropped_partial}")
    for name in FEATURE_NAMES:
        values = [
            value
            for value in (_safe_float(point.as_features().get(name)) for point in series.points)
            if value is not None
        ]
        if not values:
            print(f"  {name}: n/a")
            continue
        array = np.asarray(values, dtype=float)
        print(
            f"  {name}: min={_fmt(float(array.min()))} "
            f"median={_fmt(float(np.median(array)))} max={_fmt(float(array.max()))}"
        )
    print(f"  saved series: {path}")


def _print_train(detector: SumoAnomalyDetector, series: MetricSeries, path: Path) -> None:
    extras = detector.trained.extras
    print(
        f"Sumo anomaly train: {len(series.points)} timeslices, "
        f"features={','.join(detector.trained.encoder.feature_names)}"
    )
    print(
        f"  contamination={extras.get('contamination')} "
        f"n_estimators={extras.get('n_estimators')} "
        f"timeslice={extras.get('timeslice')}"
    )
    baseline = extras.get("baseline") or {}
    print(
        "  baseline: "
        + ", ".join(
            f"{name} median={_fmt((baseline.get(name) or {}).get('median'))} "
            f"scale={_fmt((baseline.get(name) or {}).get('scale'))}"
            for name in detector.trained.encoder.feature_names
        )
    )
    findings = detector.score(series.points)
    anomalies = sum(1 for finding in findings if finding.is_anomaly)
    print(
        f"  anomalies in training window: {anomalies}/{len(findings)} "
        f"({(100.0 * anomalies / len(findings)) if findings else 0.0:.1f}%)"
    )
    print(f"  saved model: {path}")


def _deviation_line(deviation: MetricDeviation) -> str:
    return (
        f"{deviation.metric} {deviation.deviation:+.1f}σ "
        f"({_fmt(deviation.value)} vs {_fmt(deviation.baseline)})"
    )


def _print_check(report: Mapping[str, Any], *, series: MetricSeries) -> None:
    print(
        f"Sumo search-metric anomalies (index={report['index']}, "
        f"timeslice={report['timeslice']})"
    )
    print(
        f"  window: {report['window_start']}..{report['window_end']} "
        f"({report['points']} points, fetched {report['fetched_at']})"
    )
    model = report.get("model") or {}
    print(
        f"  model: trained {model.get('trained_at') or 'n/a'} "
        f"contamination={model.get('contamination')} "
        f"z_name={report.get('z_threshold')} flag={report.get('flag_threshold')} "
        f"features={','.join(report.get('features') or [])}"
    )
    print(
        f"  anomalies: {report['anomalies']}/{report['points']} "
        f"({100.0 * float(report.get('anomaly_rate') or 0.0):.1f}%); "
        f"recent (last {report.get('recent_minutes')}m): {report['recent_anomalies']}"
    )
    print(
        f"  flagged by model: {report.get('flagged_by_model')}; "
        f"metric-only: {report.get('flagged_by_deviation_only')}"
    )
    if report.get("anomalies_by_metric"):
        print(f"  deviating metrics: {report['anomalies_by_metric']}")
    if report.get("imputed_counts"):
        print(f"  imputed (metric absent in slice): {report['imputed_counts']}")

    def emit(title: str, findings: Sequence[Mapping[str, Any]]) -> None:
        print(f"\n  {title}")
        if not findings:
            print("    none")
            return
        for finding in findings:
            deviating = [
                deviation for deviation in finding.get("deviations") or []
                if deviation.get("is_deviating")
            ]
            if deviating:
                explanation = "; ".join(
                    _deviation_line(
                        MetricDeviation(
                            metric=str(deviation["metric"]),
                            value=float(deviation["value"]),
                            baseline=float(deviation["baseline"]),
                            deviation=float(deviation["deviation"]),
                            direction=str(deviation["direction"]),
                            is_deviating=True,
                        )
                    )
                    for deviation in deviating
                )
            else:
                explanation = "multivariate outlier (no single metric beyond threshold)"
            tag = (
                "  [metric-only]"
                if finding.get("metric_flagged") and not finding.get("model_flagged")
                else ""
            )
            print(
                f"    {finding['timestamp']}  score={float(finding['anomaly_score']):.4f}  "
                f"{explanation}{tag}"
            )

    emit("recent anomalies", report.get("recent") or [])
    emit("top anomalies", report.get("top_anomalies") or [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SERIES_NAME = "sumo_anomaly_series.json"


def series_path() -> Path:
    return paths.train_dir() / DEFAULT_SERIES_NAME


def model_path() -> Path:
    return paths.artifact_path(ARTIFACT_NAME)


def build_client(
    config: SumoConfig,
    *,
    environ: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
) -> SumoClient:
    access_id, access_key = load_credentials(environ=environ)
    return SumoClient(access_id, access_key, config=config, session=session)


def _config_from_args(args: argparse.Namespace) -> SumoConfig:
    return SumoConfig(
        index=args.index,
        days=args.days,
        timeslice=args.timeslice,
        base_url=args.base_url,
        recent_minutes=args.recent_minutes,
        job_timeout=args.job_timeout,
    )


def cmd_fetch(
    args: argparse.Namespace,
    client_factory: Callable[..., SearchClient] = build_client,
    *,
    emit: bool = True,
) -> int:
    config = _config_from_args(args)
    client = client_factory(config)
    now = _parse_iso(getattr(args, "now", "")) if getattr(args, "now", "") else None
    if getattr(args, "now", "") and now is None:
        raise ValueError("--now must be an ISO timestamp")

    def progress(name: str, count: int) -> None:
        print(f"  {name}: {count} records", file=sys.stderr)

    series = fetch_metrics(client, config, now=now, progress=progress)
    path = series_path()
    series.save(path)
    if not emit:
        return 0
    if args.json:
        print(json.dumps(series.as_dict(), indent=2, sort_keys=True))
    else:
        _print_fetch(series, path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    path = series_path()
    series = MetricSeries.load(path)
    config = _config_from_args(args)
    detector = SumoAnomalyDetector.fit(
        series.points,
        series=series,
        config=config,
        contamination=args.contamination,
        n_estimators=args.estimators,
    )
    # Persist the recent-window used at report time with the model.
    detector.trained.extras["recent_minutes"] = args.recent_minutes
    destination = detector.save(model_path())
    if args.json:
        print(json.dumps(
            {
                "model": destination.as_posix(),
                "series": path.as_posix(),
                "points": len(series.points),
                "features": detector.trained.encoder.feature_names,
                "contamination": args.contamination,
                "n_estimators": args.estimators,
                "data_hash": detector.trained.data_hash,
            },
            indent=2,
            sort_keys=True,
        ))
    else:
        _print_train(detector, series, destination)
    return 0


def cmd_check(
    args: argparse.Namespace,
    client_factory: Callable[..., SearchClient] = build_client,
) -> int:
    path = series_path()
    if args.refresh or not path.exists():
        if not path.exists() and not args.refresh:
            print(f"no series at {path}; fetching from Sumo", file=sys.stderr)
        # Suppress fetch output so `check --json` stays machine-readable.
        cmd_fetch(args, client_factory=client_factory, emit=False)
    series = MetricSeries.load(path)
    artifact = model_path()
    if not artifact.exists():
        print(
            f"no trained artifact at {artifact}; run "
            "`PYTHONPATH=services python3 -m learning.train.sumo_anomaly train`",
            file=sys.stderr,
        )
        return 2
    detector = SumoAnomalyDetector.load(artifact)
    now = _parse_iso(args.now) if args.now else datetime.now(timezone.utc)
    findings = detector.score(
        series.points,
        now=now,
        z_threshold=args.z_threshold,
        flag_threshold=args.flag_threshold,
    )
    report = build_report(
        findings,
        series,
        detector=detector,
        top=args.top,
        z_threshold=args.z_threshold,
        flag_threshold=args.flag_threshold,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_check(report, series=series)
    if args.fail_on_anomaly and report["recent_anomalies"]:
        return 3
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default=DEFAULT_INDEX, help=f"Sumo index (default {DEFAULT_INDEX})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="lookback window in days")
    parser.add_argument("--timeslice", default=DEFAULT_TIMESLICE, help="Sumo timeslice (default 5m)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Sumo API base URL")
    parser.add_argument("--recent-minutes", type=int, default=60,
                        help="a finding is recent within this many minutes (default 60)")
    parser.add_argument("--job-timeout", type=float, default=1800.0,
                        help="seconds to wait for one search job (default 1800)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m learning.train.sumo_anomaly", description=__doc__
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch", help="pull metrics from Sumo")
    _add_common(fetch_parser)
    fetch_parser.add_argument(
        "--now", default="", help="ISO timestamp for the window end (default: now)"
    )

    train_parser = subparsers.add_parser("train", help="fit IsolationForest on the stored series")
    _add_common(train_parser)
    train_parser.add_argument("--contamination", default=0.02,
                              help="expected outlier fraction or 'auto' (default 0.02)")
    train_parser.add_argument("--estimators", type=int, default=200, help="trees (default 200)")

    check_parser = subparsers.add_parser("check", help="report current anomalies")
    _add_common(check_parser)
    check_parser.add_argument("--refresh", action="store_true", help="fetch before checking")
    check_parser.add_argument("--z-threshold", type=float, default=3.0,
                              help="robust z threshold for naming a deviating metric")
    check_parser.add_argument("--flag-threshold", type=float, default=4.0,
                              help="robust z required for the univariate flag (default 4.0)")
    check_parser.add_argument("--top", type=int, default=10, help="top anomalies to print")
    check_parser.add_argument("--now", default="", help="ISO timestamp for the recent-window reference")
    check_parser.add_argument("--fail-on-anomaly", action="store_true",
                              help="exit 3 when a recent anomaly exists")

    return parser


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[..., SearchClient] = build_client,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command is None:
        parser.print_help()
        return 2
    if command == "train" and isinstance(args.contamination, str) and args.contamination != "auto":
        try:
            args.contamination = float(args.contamination)
        except ValueError:
            print("--contamination must be a float or 'auto'", file=sys.stderr)
            return 2
    try:
        if command == "fetch":
            return cmd_fetch(args, client_factory=client_factory)
        if command == "train":
            return cmd_train(args)
        if command == "check":
            return cmd_check(args, client_factory=client_factory)
    except (SumoError, TrainingError, ValueError) as error:
        print(f"sumo_anomaly: {error}", file=sys.stderr)
        return 2 if isinstance(error, (SumoCredentialError, ValueError)) else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    # ``python -m`` executes this file as __main__, but joblib must record a
    # portable class path. Register the running module under its canonical
    # name so pickle's identity check succeeds and the artifact loads from any
    # other process.
    sys.modules["learning.train.sumo_anomaly"] = sys.modules[__name__]
    sys.exit(main())
