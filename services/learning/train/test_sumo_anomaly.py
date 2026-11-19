"""Tests for the Sumo search-metric anomaly pipeline.

Every test is offline: the fetcher is driven by :class:`FakeSearchClient` (or a
fake ``requests.Session`` for the client itself), and an autouse fixture makes
constructing a real ``requests.Session`` fail loudly. Fixture series contain
normal traffic plus injected latency / error / zero-result / traffic-drop
outliers, mirroring the shapes observed in the live ``rh_cpav-001`` index.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from learning.train import paths
from learning.train import sumo_anomaly as sa
from learning.train.sumo_anomaly import (
    MetricPoint,
    MetricSeries,
    SumoAnomalyDetector,
    SumoConfig,
    SumoCredentialError,
    SumoDataError,
    build_queries,
    fetch_metrics,
    load_credentials,
    merge_records,
)

UTC = timezone.utc
START = datetime(2026, 11, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Offline guard and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    class _Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("tests must not construct a real requests.Session")

    monkeypatch.setattr(sa.requests, "Session", _Boom)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _epoch_ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


def make_series(
    *,
    count: int = 300,
    start: datetime = START,
    with_outliers: bool = False,
    recent_outlier_at_end: bool = False,
) -> list[MetricPoint]:
    """Normal 5-minute series, optionally with four labelled injected outliers."""
    rng = np.random.default_rng(7)
    points: list[MetricPoint] = []
    for index in range(count):
        timestamp = start + timedelta(minutes=5 * index)
        # A slow diurnal cycle makes request_count non-constant and realistic.
        traffic = 1_000.0 + 250.0 * math.sin(2 * math.pi * index / 288.0)
        p95 = 400.0 + 35.0 * float(rng.standard_normal())
        average = 120.0 + 12.0 * float(rng.standard_normal())
        errors = float(rng.poisson(2.0))
        zero_rate = min(max(0.10 + 0.01 * float(rng.standard_normal()), 0.0), 1.0)
        points.append(
            MetricPoint(
                timestamp=_iso(timestamp),
                p95_ms=max(p95, 1.0),
                avg_ms=max(average, 1.0),
                request_count=max(traffic + 70.0 * float(rng.standard_normal()), 1.0),
                error_count=errors,
                zero_result_count=round(zero_rate * traffic),
                app_request_count=traffic,
                zero_result_rate=zero_rate,
            )
        )
    if not with_outliers:
        return points

    outliers = {
        count // 3: {"p95_ms": 5_500.0, "avg_ms": 1_100.0},
        count // 2: {"error_count": 400.0},
        (2 * count) // 3: {
            "zero_result_rate": 0.92,
            "zero_result_count": 920.0,
            "app_request_count": 1_000.0,
        },
    }
    if recent_outlier_at_end:
        outliers[count - 1] = {"p95_ms": 8_000.0, "avg_ms": 1_500.0}
    else:
        outliers[count - 3] = {"request_count": 5.0}
    for index, changes in outliers.items():
        points[index] = replace(points[index], **changes)
    return points


def sumo_records(points: list[MetricPoint]) -> dict[str, list[dict[str, str]]]:
    """Render points as the aggregate record maps the Sumo API returns."""
    latency = [
        {
            "_timeslice": _epoch_ms(_parse(point.timestamp)),
            "p95_ms": f"{point.p95_ms}",
            "avg_ms": f"{point.avg_ms}",
        }
        for point in points
    ]
    requests = [
        {"_timeslice": _epoch_ms(_parse(point.timestamp)), "request_count": str(int(point.request_count))}
        for point in points
    ]
    errors = [
        {"_timeslice": _epoch_ms(_parse(point.timestamp)), "error_count": str(int(point.error_count))}
        for point in points
        if point.error_count
    ]
    zero = [
        {
            "_timeslice": _epoch_ms(_parse(point.timestamp)),
            "zero_result_count": str(int(point.zero_result_count or 0)),
            "app_request_count": str(int(point.app_request_count or 0)),
        }
        for point in points
        if point.app_request_count
    ]
    return {"latency": latency, "requests": requests, "errors": errors, "zero_results": zero}


def _parse(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


class FakeSearchClient:
    """Query-kind router; records the calls so tests can prove no network I/O."""

    def __init__(self, records: dict[str, list[dict[str, str]]]) -> None:
        self.records = records
        self.calls: list[tuple[str, datetime, datetime]] = []

    def run_search(self, query: str, start: datetime, end: datetime):
        self.calls.append((query, start, end))
        # Most specific markers first: app_request_count contains request_count.
        if "zero_result_count" in query:
            return list(self.records.get("zero_results", []))
        if "error_count" in query:
            return list(self.records.get("errors", []))
        if "p95_ms" in query:
            return list(self.records.get("latency", []))
        if "request_count" in query:
            return list(self.records.get("requests", []))
        return []


class FakeResponse:
    def __init__(self, status: int, payload, *, headers=None) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Just enough of requests.Session for SumoClient."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []
        self.auth = None
        self.headers: dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra HTTP request")
        return self.responses.pop(0)

    def delete(self, url: str, **kwargs):
        self.calls.append(("DELETE", url, kwargs))
        return FakeResponse(200, {})


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_credentials_prefer_environment():
    access_id, access_key = load_credentials(
        environ={"SUMO_ACCESS_ID": "env-id", "SUMO_ACCESS_KEY": "env-key"},
        settings_path="/nonexistent/settings.json",
    )
    assert (access_id, access_key) == ("env-id", "env-key")


def test_credentials_fall_back_to_settings_env_block(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"env": {"SUMO_ACCESS_ID": "file-id", "SUMO_ACCESS_KEY": "file-key"}}),
        encoding="utf-8",
    )
    access_id, access_key = load_credentials(environ={}, settings_path=settings)
    assert (access_id, access_key) == ("file-id", "file-key")


def test_missing_credentials_raise_without_leaking(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"SUMO_ACCESS_ID": "only-id"}}), encoding="utf-8")
    with pytest.raises(SumoCredentialError) as error:
        load_credentials(environ={}, settings_path=settings)
    message = str(error.value)
    assert "only-id" not in message
    assert "SUMO_ACCESS_KEY" in message


def test_malformed_settings_do_not_echo_file_contents(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("SUMO_ACCESS_KEY=do-not-print-me", encoding="utf-8")
    with pytest.raises(SumoCredentialError) as error:
        load_credentials(environ={}, settings_path=settings)
    assert "do-not-print-me" not in str(error.value)


def test_config_holds_no_credentials():
    config = SumoConfig()
    assert "SUMO" not in repr(config)
    assert "access" not in repr(config).lower()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_queries_target_the_live_index_and_real_log_tokens():
    config = SumoConfig(index="rh_cpav-001", timeslice="5m")
    queries = build_queries(config)

    assert set(queries) == {"latency", "requests", "errors", "zero_results"}
    assert '_index=rh_cpav-001 "/rs/search"' in queries["latency"]
    assert 'processed-time-milliseconds=\\"*\\"' in queries["latency"]
    assert "pct(processed_time, 95) as p95_ms" in queries["latency"]
    assert "timeslice 5m" in queries["latency"]
    assert "count as request_count" in queries["requests"]
    assert "where status_code >= 400" in queries["errors"]
    assert "_sourceCategory=disseminator.log" in queries["zero_results"]
    assert "resultCount = 0" in queries["zero_results"]


@pytest.mark.parametrize("field,value", [("index", "../../etc"), ("index", ""), ("timeslice", "5 minutes")])
def test_config_rejects_unsafe_values(field, value):
    with pytest.raises(ValueError):
        SumoConfig(**{field: value})


def test_config_requires_https():
    with pytest.raises(ValueError):
        SumoConfig(base_url="http://api.sumologic.com/api/v1")


def test_encoder_is_pickled_under_its_canonical_module():
    # `python -m learning.train.sumo_anomaly` runs this module as __main__; the
    # artifact must still be loadable from any other process.
    assert sa.MetricFeatureEncoder.__module__ == "learning.train.sumo_anomaly"


# ---------------------------------------------------------------------------
# Merging and untrusted-input handling
# ---------------------------------------------------------------------------


def test_merge_joins_on_timeslice_and_fills_zero_counts():
    t0 = _epoch_ms(START)
    t1 = _epoch_ms(START + timedelta(minutes=5))
    points = merge_records(
        latency_records=[
            {"_timeslice": t0, "p95_ms": "500", "avg_ms": "100"},
            {"_timeslice": t1, "p95_ms": "420.5", "avg_ms": "90"},
        ],
        request_records=[
            {"_timeslice": t0, "request_count": "1000"},
            {"_timeslice": t1, "request_count": "800"},
        ],
        error_records=[{"_timeslice": t1, "error_count": "4"}],
        zero_records=[
            {"_timeslice": t0, "zero_result_count": "100", "app_request_count": "1000"},
        ],
    )
    assert [point.timestamp for point in points] == [_iso(START), _iso(START + timedelta(minutes=5))]
    assert points[0].error_count == 0.0
    assert points[0].zero_result_rate == pytest.approx(0.1)
    assert points[1].error_count == 4.0
    assert points[1].zero_result_rate is None


def test_merge_ignores_raw_and_malformed_fields():
    points = merge_records(
        latency_records=[
            {
                "_timeslice": _epoch_ms(START),
                "p95_ms": "not-a-number",
                "avg_ms": "NaN",
                "_raw": "SECRET LOG CONTENT",
            },
            # Untrusted non-dict-ish shapes and invalid timestamps are skipped.
            {"_timeslice": "not-a-time", "p95_ms": "100"},
        ],
        request_records=[{"_timeslice": _epoch_ms(START), "request_count": "10", "_raw": "SECRET"}],
        error_records=[],
        zero_records=[],
    )
    assert len(points) == 1
    assert points[0].p95_ms is None and points[0].avg_ms is None
    assert points[0].request_count == 10.0
    assert "SECRET" not in json.dumps([point.as_dict() for point in points])


def test_safe_float_rejects_non_finite_and_boolean():
    assert sa._safe_float("1.5e3") == 1500.0
    assert sa._safe_float("NaN") is None
    assert sa._safe_float("Infinity") is None
    assert sa._safe_float(True) is None
    assert sa._safe_float(["1"]) is None


def test_timeslice_iso_accepts_epoch_millis_and_iso():
    assert sa._timeslice_iso(_epoch_ms(START)) == _iso(START)
    assert sa._timeslice_iso(_iso(START)) == _iso(START)
    assert sa._timeslice_iso("garbage") is None


# ---------------------------------------------------------------------------
# Fetching through a fake client
# ---------------------------------------------------------------------------


def test_fetch_metrics_uses_four_queries_and_no_raw_content():
    points = make_series(count=40, with_outliers=False)
    records = sumo_records(points)
    # A hostile/extra field must never survive into the series.
    records["latency"][0]["_raw"] = "SECRET LOG CONTENT"
    client = FakeSearchClient(records)

    series = fetch_metrics(client, SumoConfig(index="rh_cpav-001"), now=START + timedelta(days=7))

    assert len(client.calls) == 4
    assert set(series.query_records) == {"latency", "requests", "errors", "zero_results"}
    assert len(series.points) == 40
    start, end = client.calls[0][1], client.calls[0][2]
    assert end - start == timedelta(days=7)
    payload = json.dumps(series.as_dict())
    assert "SECRET" not in payload
    assert all("_index=rh_cpav-001" in query for query, _, _ in client.calls)
    assert all('"/rs/search"' in query for query, _, _ in client.calls)


def test_window_is_aligned_and_partial_edge_slices_are_dropped():
    config = SumoConfig(timeslice="5m")
    now = datetime(2026, 11, 19, 22, 35, 51, tzinfo=UTC)
    start, end = config.window(now)
    assert end == datetime(2026, 11, 19, 22, 35, tzinfo=UTC)
    assert start == end - timedelta(days=7)

    points = make_series(count=5, start=start)
    records = sumo_records(points)
    # Add a bucket that starts before the window and one at the exclusive end.
    for kind in ("latency", "requests"):
        records[kind].insert(
            0,
            {
                "_timeslice": _epoch_ms(start - timedelta(minutes=5)),
                "p95_ms": "1",
                "avg_ms": "1",
                "request_count": "1",
            },
        )
        records[kind].append(
            {
                "_timeslice": _epoch_ms(end),
                "p95_ms": "1",
                "avg_ms": "1",
                "request_count": "1",
            }
        )
    client = FakeSearchClient(records)
    series = fetch_metrics(client, config, now=now)
    assert len(series.points) == 5
    assert series.dropped_partial == 2


# ---------------------------------------------------------------------------
# SumoClient protocol behavior (fake HTTP session, never real requests)
# ---------------------------------------------------------------------------


def test_sumo_client_posts_polls_and_reads_records():
    session = FakeSession(
        [
            FakeResponse(202, {"id": "ABCD1234"}),
            FakeResponse(200, {"state": "GATHERING RESULTS"}),
            FakeResponse(200, {"state": "DONE GATHERING RESULTS"}),
            FakeResponse(200, {"records": [{"map": {"_timeslice": "1", "p95_ms": "10"}}]}),
        ]
    )
    slept: list[float] = []
    client = sa.SumoClient(
        "id",
        "key",
        config=SumoConfig(poll_interval=0.01, page_size=10),
        session=session,
        sleep=slept.append,
    )
    records = client.run_search("q", START, START + timedelta(days=1))

    assert records == [{"_timeslice": "1", "p95_ms": "10"}]
    methods = [(method, url.rsplit("/api/v1", 1)[-1]) for method, url, _ in session.calls]
    assert methods[0] == ("POST", "/search/jobs")
    assert methods[1][0] == "GET" and "GATHERING" not in methods[1][1]
    assert methods[-1][1].endswith("/records")
    post_payload = session.calls[0][2]["json"]
    assert set(post_payload) == {"query", "from", "to", "timeZone"}
    assert "requireRawMessages" not in post_payload  # raw log content is never requested
    assert slept  # polled at least once


def test_sumo_client_retries_429_then_succeeds():
    session = FakeSession(
        [
            FakeResponse(429, {"code": "rate.limited"}, headers={"Retry-After": "0"}),
            FakeResponse(202, {"id": "ABCD1234"}),
            FakeResponse(200, {"state": "DONE GATHERING RESULTS"}),
            FakeResponse(200, {"records": []}),
        ]
    )
    slept: list[float] = []
    client = sa.SumoClient(
        "id", "key", config=SumoConfig(), session=session, sleep=slept.append
    )
    assert client.run_search("q", START, START + timedelta(hours=1)) == []
    assert len(slept) == 1


def test_sumo_client_failed_job_cancels_and_does_not_echo_body():
    session = FakeSession(
        [
            FakeResponse(202, {"id": "ABCD1234"}),
            FakeResponse(200, {"state": "FAILED"}),
        ]
    )
    client = sa.SumoClient(
        "id", "key", config=SumoConfig(), session=session, sleep=lambda _: None
    )
    with pytest.raises(sa.SumoAPIError):
        client.run_search("q", START, START + timedelta(hours=1))
    assert any(method == "DELETE" for method, _, _ in session.calls)


def test_api_error_body_is_not_echoed():
    secret_body = ValueError("not json: SECRET LOG CONTENT")
    response = FakeResponse(500, secret_body)
    message = sa._api_error_message(response)
    assert "SECRET" not in message
    assert "500" in message


def test_client_rejects_missing_credentials():
    with pytest.raises(SumoCredentialError):
        sa.SumoClient("", "", config=SumoConfig())


# ---------------------------------------------------------------------------
# Model: fit, detect injected outliers, explain, recency
# ---------------------------------------------------------------------------


def test_isolation_forest_flags_injected_outliers_and_names_metrics():
    normal = make_series(count=300, with_outliers=False)
    mixed = make_series(count=300, with_outliers=True)
    detector = SumoAnomalyDetector.fit(normal, contamination=0.02, n_estimators=150)

    # 3.5σ so the deliberately modest traffic-drop outlier (about -4σ against
    # realistic diurnal variance) is comfortably inside the univariate guard.
    findings = detector.score(
        mixed,
        now=_parse(_iso(START + timedelta(minutes=5 * 300))),
        flag_threshold=3.5,
    )
    by_timestamp = {finding.timestamp: finding for finding in findings}

    expected = {
        100: ("p95_ms", "above"),
        150: ("error_count", "above"),
        200: ("zero_result_rate", "above"),
        297: ("request_count", "below"),
    }
    for index, (metric, direction) in expected.items():
        finding = by_timestamp[_iso(START + timedelta(minutes=5 * index))]
        assert finding.is_anomaly, f"outlier at index {index} was not flagged"
        deviating = {deviation.metric: deviation for deviation in finding.deviations if deviation.is_deviating}
        assert metric in deviating, f"{metric} not named for index {index}: {deviating}"
        assert deviating[metric].direction == direction
        assert abs(deviating[metric].deviation) >= 3.0

    # The model was fit on normal data; it must not flag most normal points.
    normal_flagged = sum(
        finding.is_anomaly
        for finding in findings
        if finding.timestamp not in {
            _iso(START + timedelta(minutes=5 * index)) for index in (100, 150, 200, 297)
        }
    )
    assert normal_flagged <= 0.10 * len(findings)


def test_recent_flag_uses_the_reference_time():
    normal = make_series(count=120, with_outliers=False)
    mixed = make_series(count=120, with_outliers=True, recent_outlier_at_end=True)
    detector = SumoAnomalyDetector.fit(normal, contamination=0.05, n_estimators=100)
    reference = START + timedelta(minutes=5 * 120) + timedelta(minutes=10)

    findings = detector.score(mixed, now=reference)
    by_timestamp = {finding.timestamp: finding for finding in findings}
    recent = by_timestamp[_iso(START + timedelta(minutes=5 * 119))]
    old = by_timestamp[_iso(START + timedelta(minutes=5 * 80))]
    assert recent.is_anomaly and recent.recent is True
    assert old.is_anomaly and old.recent is False


def test_fit_requires_enough_points():
    with pytest.raises(sa.InsufficientSeriesError):
        SumoAnomalyDetector.fit(make_series(count=5), contamination=0.02)


def test_naming_threshold_is_lower_than_the_flag_threshold():
    deviation = sa._deviation(
        "p95_ms", 135.0, {"median": 100.0, "scale": 10.0}, threshold=3.0
    )
    assert deviation.deviation == pytest.approx(3.5)
    assert deviation.is_deviating is True  # named in the report
    assert sa._metric_flagged([deviation], flag_threshold=4.0) is False  # not flagged
    assert sa._metric_flagged([deviation], flag_threshold=3.0) is True


def test_constant_features_do_not_crash_the_encoder():
    points = [
        MetricPoint(
            timestamp=_iso(START + timedelta(minutes=5 * index)),
            p95_ms=100.0,
            avg_ms=50.0,
            request_count=10.0,
            error_count=0.0,
            zero_result_rate=0.0,
        )
        for index in range(20)
    ]
    detector = SumoAnomalyDetector.fit(points, contamination=0.1, n_estimators=20)
    findings = detector.score(points)
    assert len(findings) == 20
    assert all(math.isfinite(finding.anomaly_score) for finding in findings)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_series_and_model_round_trip_with_private_permissions(tmp_path):
    points = make_series(count=60, with_outliers=True)
    series = MetricSeries(
        points=points,
        index="rh_cpav-001",
        window_start=_iso(START),
        window_end=_iso(START + timedelta(minutes=5 * 60)),
        timeslice="5m",
        query_records={"latency": 60, "requests": 60, "errors": 10, "zero_results": 60},
    )
    series_file = series.save(tmp_path / "series.json")
    assert stat.S_IMODE(os.stat(series_file).st_mode) == 0o600
    loaded = MetricSeries.load(series_file)
    assert loaded.points == points
    assert loaded.data_hash() == series.data_hash()

    detector = SumoAnomalyDetector.fit(points, series=series, contamination=0.05)
    artifact = detector.save(tmp_path / "model.joblib")
    assert stat.S_IMODE(os.stat(artifact).st_mode) == 0o600

    restored = SumoAnomalyDetector.load(artifact)
    assert restored.trained.extras["feature_names"] == list(sa.FEATURE_NAMES)
    assert restored.trained.extras["recent_minutes"] == 60
    assert restored.trained.extras["baseline"]["p95_ms"]["median"] > 0
    before = detector.score(points, now=_parse(_iso(START + timedelta(minutes=300))))
    after = restored.score(points, now=_parse(_iso(START + timedelta(minutes=300))))
    assert [finding.anomaly_score for finding in before] == [
        finding.anomaly_score for finding in after
    ]


def test_series_load_rejects_wrong_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "something-else", "points": []}), encoding="utf-8")
    with pytest.raises(SumoDataError):
        MetricSeries.load(path)


def test_cli_artifact_loads_from_another_process(tmp_path):
    """`python -m` must pickle the encoder under a portable module path."""
    import subprocess
    import sys

    points = make_series(count=40, with_outliers=False)
    MetricSeries(
        points=points,
        index="rh_cpav-001",
        window_start=_iso(START),
        window_end=_iso(START + timedelta(minutes=5 * 40)),
        timeslice="5m",
    ).save(tmp_path / sa.DEFAULT_SERIES_NAME)

    services_dir = str(Path(__file__).resolve().parents[2])
    env = {
        **os.environ,
        "PYTHONPATH": services_dir,
        "POC_CLAUDE_ENSEMBLE_TRAIN_DIR": str(tmp_path),
    }
    trained = subprocess.run(
        [sys.executable, "-m", "learning.train.sumo_anomaly", "train", "--estimators", "20"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert trained.returncode == 0, trained.stderr

    code = (
        "import sys; from learning.train.sumo_anomaly import SumoAnomalyDetector; "
        "d = SumoAnomalyDetector.load(sys.argv[1]); "
        "print(d.trained.encoder.__class__.__module__)"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "sumo_anomaly.joblib")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "learning.train.sumo_anomaly"


# ---------------------------------------------------------------------------
# CLI (fake client, redirected artifact directory)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_TRAIN_DIR", str(tmp_path / "train"))
    return tmp_path


def _fake_factory(records):
    def factory(config, **kwargs):
        return FakeSearchClient(records)

    return factory


def _series_for_cli():
    # 80 points ending with a latency spike so "recent" is demonstrable.
    points = make_series(count=80, with_outliers=True, recent_outlier_at_end=True)
    return points, sumo_records(points)


WINDOW_NOW = _iso(START + timedelta(days=7))
REFERENCE = _iso(START + timedelta(minutes=5 * 80) + timedelta(minutes=10))


def test_cli_fetch_train_check_end_to_end(cli_env, capsys):
    points, records = _series_for_cli()
    factory = _fake_factory(records)

    assert sa.main(["fetch", "--now", WINDOW_NOW], client_factory=factory) == 0
    out = capsys.readouterr().out
    assert "Sumo anomaly fetch" in out and "merged:" in out
    assert (paths.train_dir() / sa.DEFAULT_SERIES_NAME).exists()

    assert sa.main(["train", "--estimators", "80"], client_factory=factory) == 0
    out = capsys.readouterr().out
    assert "Sumo anomaly train" in out and "baseline:" in out
    assert sa.model_path().exists()

    assert sa.main(["check", "--json", "--now", REFERENCE], client_factory=factory) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["points"] == 80
    assert report["anomalies"] >= 1
    assert report["recent_anomalies"] >= 1
    assert report["model"]["n_estimators"] == 80
    # Top anomalies must not include non-anomalous findings.
    assert all(item["is_anomaly"] for item in report["top_anomalies"])
    assert len(report["top_anomalies"]) <= report["anomalies"]


def test_cli_check_without_model_returns_usage_error(cli_env, capsys):
    _, records = _series_for_cli()
    factory = _fake_factory(records)
    assert sa.main(["fetch", "--now", WINDOW_NOW], client_factory=factory) == 0
    capsys.readouterr()
    assert sa.main(["check"], client_factory=factory) == 2
    assert "no trained artifact" in capsys.readouterr().err


def test_cli_refresh_keeps_json_output_machine_readable(cli_env, capsys):
    _, records = _series_for_cli()
    factory = _fake_factory(records)
    assert sa.main(["fetch", "--now", WINDOW_NOW], client_factory=factory) == 0
    assert sa.main(["train", "--estimators", "40"], client_factory=factory) == 0
    capsys.readouterr()

    assert sa.main(
        ["check", "--refresh", "--json", "--now", REFERENCE], client_factory=factory
    ) == 0
    report = json.loads(capsys.readouterr().out)  # must be JSON, not fetch prose + JSON
    assert report["points"] == 80


def test_cli_fail_on_anomaly_exits_three(cli_env, capsys):
    _, records = _series_for_cli()
    factory = _fake_factory(records)
    sa.main(["fetch", "--now", WINDOW_NOW], client_factory=factory)
    sa.main(["train", "--estimators", "40"], client_factory=factory)
    capsys.readouterr()
    assert sa.main(["check", "--fail-on-anomaly", "--now", REFERENCE], client_factory=factory) == 3


def test_cli_reports_credentials_error_without_secret(cli_env, capsys, monkeypatch):
    monkeypatch.delenv("SUMO_ACCESS_ID", raising=False)
    monkeypatch.delenv("SUMO_ACCESS_KEY", raising=False)
    monkeypatch.setattr(sa, "DEFAULT_SETTINGS_PATH", "/nonexistent/settings.json")
    assert sa.main(["fetch"]) == 2
    error = capsys.readouterr().err
    assert "credentials" in error.lower()
    assert "SUMO_ACCESS_KEY" in error
