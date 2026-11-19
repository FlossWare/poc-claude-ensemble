"""Bounded accessibility probe: mapping, fallback, cache and politeness."""
from __future__ import annotations

import pytest

from ingestion.datasets import probe
from ingestion.datasets.netguard import NetguardError

from ingestion.datasets.tests.helpers import FakeProbeTransport, make_config


def _config(tmp_path, **overrides):
    base = make_config(tmp_path).replace(
        probe_enabled=True, per_host_rps=1000.0, concurrency=2
    )
    return base.replace(**overrides) if overrides else base


@pytest.mark.parametrize(
    "status,access,reason",
    [
        (200, "public", None),
        (204, "public", None),
        (401, "gated", "gated_authentication"),
        (403, "blocked", "blocked"),
        (429, "blocked", "rate_limited"),
        (404, "inaccessible", "not_found"),
        (410, "inaccessible", "gone"),
        (500, "inaccessible", "server_error"),
        (418, "inaccessible", "http_error"),
    ],
)
def test_status_mapping(status, access, reason, tmp_path):
    transport = FakeProbeTransport(
        {("https://example.org/x", "HEAD"): {"status": status}}
    )
    result = probe.probe_one(
        "https://example.org/x", transport, now="2026-01-01T00:00:00Z"
    )
    assert result.status == status
    assert result.access == access
    assert result.reason_code == reason


def test_head_falls_back_to_range_get(tmp_path):
    transport = FakeProbeTransport(
        {
            ("https://example.org/x", "HEAD"): {"status": 405},
            ("https://example.org/x", "GET"): {
                "status": 206,
                "headers": {
                    "content-type": "text/csv; charset=utf-8",
                    "content-length": "1",
                    "content-range": "bytes 0-0/999",
                },
            },
        }
    )
    result = probe.probe_one(
        "https://example.org/x", transport, now="2026-01-01T00:00:00Z"
    )
    assert result.method == "GET"
    assert result.status == 206
    assert result.content_length == 999
    assert result.content_type == "text/csv"
    get_calls = [call for call in transport.calls if call["method"] == "GET"]
    assert get_calls and get_calls[0]["headers"]["Range"] == "bytes=0-0"
    # The probe never asks the transport to retain a body.
    assert all(call["store_body"] is False for call in transport.calls)
    assert all(call["max_bytes"] == 0 for call in transport.calls)


def test_forbidden_head_does_not_fall_back(tmp_path):
    transport = FakeProbeTransport(
        {("https://example.org/x", "HEAD"): {"status": 403}}
    )
    result = probe.probe_one(
        "https://example.org/x", transport, now="2026-01-01T00:00:00Z"
    )
    assert result.method == "HEAD"
    assert result.access == "blocked"
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "error_code,access,reason",
    [
        ("timeout", "inaccessible", "timeout"),
        ("dns_failure", "inaccessible", "dns_failure"),
        ("tls_failure", "inaccessible", "tls_failure"),
        ("ssrf_blocked", "inaccessible", "ssrf_blocked"),
        ("too_many_redirects", "inaccessible", "too_many_redirects"),
        ("unsafe_redirect", "inaccessible", "unsafe_redirect"),
    ],
)
def test_transport_errors_map_to_reason_codes(error_code, access, reason, tmp_path):
    transport = FakeProbeTransport(
        {("https://example.org/x", "HEAD"): NetguardError(error_code)}
    )
    result = probe.probe_one(
        "https://example.org/x", transport, now="2026-01-01T00:00:00Z"
    )
    assert result.access == access
    assert result.reason_code == reason
    assert result.error == error_code


def test_probe_cache_is_reused_without_network(tmp_path):
    config = _config(tmp_path)
    rows = [{"entry_id": "apd:Test/a.yml", "homepage": "https://example.org/a", "sources": []}]
    first_transport = FakeProbeTransport(
        {
            ("https://example.org/a", "HEAD"): {
                "status": 200,
                "headers": {"content-length": "10"},
            }
        }
    )
    first = probe.probe_entries(rows, config, transport=first_transport, now="2026-01-01T00:00:00Z")
    assert first["apd:Test/a.yml"].status == 200
    assert len(first_transport.calls) == 1

    second_transport = FakeProbeTransport(default=AssertionError("network used"))
    second = probe.probe_entries(
        rows, config, transport=second_transport, now="2026-01-01T00:00:00Z"
    )
    assert second["apd:Test/a.yml"].status == 200
    assert second["apd:Test/a.yml"].cached is True
    assert second_transport.calls == []


def test_probe_disabled_returns_nothing(tmp_path):
    config = make_config(tmp_path).replace(probe_enabled=False)
    rows = [{"entry_id": "a", "homepage": "https://example.org/", "sources": []}]
    assert probe.probe_entries(rows, config, transport=FakeProbeTransport()) == {}


def test_missing_url_is_reported_without_network(tmp_path):
    config = _config(tmp_path)
    rows = [{"entry_id": "a", "homepage": None, "sources": []}]
    transport = FakeProbeTransport(default=AssertionError("network used"))
    results = probe.probe_entries(rows, config, transport=transport, now="2026-01-01T00:00:00Z")
    assert results["a"].access == "unknown"
    assert results["a"].reason_code == "missing_url"
    assert transport.calls == []


def test_host_rate_limiter_spaces_requests_per_host():
    now = [0.0]
    sleeps: list[float] = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = probe.HostRateLimiter(1.0, clock=clock, sleep=sleep)
    with limiter.slot("example.org"):
        pass
    now[0] = 0.1
    with limiter.slot("example.org"):
        pass
    with limiter.slot("other.org"):
        pass
    assert sleeps == [pytest.approx(0.9)]


def test_choose_probe_url_prefers_homepage_then_source(tmp_path):
    assert probe.choose_probe_url(
        {"homepage": "https://example.org/", "sources": []}
    ) == "https://example.org/"
    assert probe.choose_probe_url(
        {
            "homepage": None,
            "sources": [{"name": "x", "access_url": "https://example.org/source.csv"}],
        }
    ) == "https://example.org/source.csv"
    assert probe.choose_probe_url({"homepage": None, "sources": []}) is None
