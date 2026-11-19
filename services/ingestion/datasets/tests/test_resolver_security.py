"""Hostile resolver inputs: SSRF redirects, downgrades, HTML and API abuse."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.datasets import resolver
from ingestion.datasets.config import Config
from ingestion.datasets.netguard import RawResponse

from ingestion.datasets.tests.helpers import FakeProbeTransport, StubTransport, stub_resolver

NOW = "2026-01-01T00:00:00Z"


def _config(tmp_path, **overrides):
    config = Config.load(data_root=tmp_path / "data", ledger_enabled=False)
    base = config.replace(
        per_host_rps=10_000.0,
        concurrency=1,
        resolve_cache_enabled=False,
        respect_robots=False,
    )
    return base.replace(**overrides) if overrides else base


def _row(homepage=None, sources=None, entry_id="apd:Test/entry.yml"):
    return {
        "entry_id": entry_id,
        "metadata_sha256": "meta-1",
        "status": "deferred",
        "homepage": homepage,
        "sources": sources or [],
        "disposition": "metadata_only_review",
        "title": entry_id,
        "category": "Test",
    }


def _context(tmp_path, transport, **overrides):
    return resolver.ResolverContext(
        config=_config(tmp_path, **overrides), transport=transport, now=NOW
    )


def test_landing_page_redirect_to_metadata_endpoint_is_blocked(tmp_path):
    config = _config(tmp_path)
    responses = {
        "https://public.example/page": RawResponse(
            status=302, headers={"location": "https://169.254.169.254/latest/meta-data/"}
        )
    }
    transport = StubTransport(
        config,
        stub_resolver(
            {
                "public.example": "93.184.216.34",
                "169.254.169.254": "169.254.169.254",
            }
        ),
        responses,
    )
    result = resolver.resolve_entry(
        _row(homepage="https://public.example/page"),
        _context(tmp_path, transport),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "ssrf_blocked"


def test_https_to_http_downgrade_is_refused(tmp_path):
    config = _config(tmp_path)
    responses = {
        "https://public.example/page": RawResponse(
            status=302, headers={"location": "http://other.example/page"}
        )
    }
    transport = StubTransport(
        config,
        stub_resolver(
            {
                "public.example": "93.184.216.34",
                "other.example": "93.184.216.35",
            }
        ),
        responses,
    )
    result = resolver.resolve_entry(
        _row(homepage="https://public.example/page"),
        _context(tmp_path, transport),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "unsafe_redirect"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/\x00data.csv",
        "https://example.org/a\nb.csv",
        "https://example.org/\x7f",
    ],
)
def test_control_characters_are_rejected_before_network(url, tmp_path):
    fake = FakeProbeTransport(default=None)
    result = resolver.resolve_entry(_row(homepage=url), _context(tmp_path, fake))
    assert result.reason_code == "invalid_url"
    assert fake.calls == []


def test_landing_link_parser_is_bounded(tmp_path):
    anchors = "".join(f'<a href="/d{i}.csv">x</a>' for i in range(500))
    html_text = f"<html><body>{anchors}</body></html>"
    fake = FakeProbeTransport(
        {
            ("https://example.org/page", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
            },
            ("https://example.org/page", "GET"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
                "body": html_text,
            },
        }
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"),
        _context(tmp_path, fake, landing_page_max_links=5, max_resolved_sources=20),
    )
    assert result.status == "resolved"
    assert len(result.sources) == 5
    assert result.total_sources == 5


def test_landing_link_with_control_characters_is_ignored(tmp_path):
    html_text = (
        '<html><body>'
        '<a href="/bad\x00.csv">bad</a>'
        '<a href="/good.csv">good</a>'
        "</body></html>"
    )
    fake = FakeProbeTransport(
        {
            ("https://example.org/page", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
            },
            ("https://example.org/page", "GET"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
                "body": html_text,
            },
        }
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert [source.uri for source in result.sources] == [
        "https://example.org/good.csv"
    ]


def test_landing_body_is_not_persisted_in_the_cache(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    marker = "SECRET-LANDING-BODY-MARKER"
    fake = FakeProbeTransport(
        {
            ("https://example.org/page", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
            },
            ("https://example.org/page", "GET"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
                "body": f'<html><body>{marker}<a href="/d.csv">d</a></body></html>',
            },
        }
    )
    resolver.resolve_entries(
        [_row(homepage="https://example.org/page")],
        {"apd:Test/entry.yml"},
        config,
        transport=fake,
        cache=resolver.ResolutionCache(config),
        now=NOW,
    )
    cache_root = tmp_path / "data" / "cache" / "resolutions"
    files = list(cache_root.glob("*.json"))
    assert files
    assert all(marker.encode() not in path.read_bytes() for path in files)


def test_api_response_is_not_persisted_verbatim(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    marker = "SECRET-API-BODY-MARKER"
    fake = FakeProbeTransport(
        {
            "https://huggingface.co/api/datasets/org/name": {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {
                        "id": "org/name",
                        "sha": "abc",
                        "gated": False,
                        "siblings": [{"rfilename": "data.csv"}],
                        "cardData": {"description": marker},
                    }
                ),
            }
        }
    )
    resolver.resolve_entries(
        [_row(sources=[{"access_url": "https://huggingface.co/datasets/org/name"}])],
        {"apd:Test/entry.yml"},
        config,
        transport=fake,
        cache=resolver.ResolutionCache(config),
        now=NOW,
    )
    cache_root = tmp_path / "data" / "cache" / "resolutions"
    files = list(cache_root.glob("*.json"))
    assert files
    assert all(marker.encode() not in path.read_bytes() for path in files)


def test_robots_file_is_bounded_and_optional(tmp_path):
    html_text = '<html><body><a href="/data.csv">data</a></body></html>'
    fake = FakeProbeTransport(
        {
            ("https://example.org/page", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
            },
            ("https://example.org/page", "GET"): {
                "status": 200,
                "headers": {"content-type": "text/html"},
                "body": html_text,
            },
            ("https://example.org/robots.txt", "GET"): {
                "status": 500,
                "headers": {"content-type": "text/plain"},
                "body": "not a robots file",
            },
        }
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"),
        _context(tmp_path, fake, respect_robots=True),
    )
    # A broken robots.txt is treated permissively, never as a hard failure.
    assert result.status == "resolved"


def test_relative_and_pathological_urls_are_never_followed(tmp_path):
    assert resolver._scheme_reason("//example.org/x") == "invalid_url"
    assert resolver._scheme_reason("example.org/x") == "invalid_url"
    assert resolver._scheme_reason("https:///nohost") == "invalid_url"
    assert resolver._scheme_reason("https://example.org:99999/") == "invalid_url"
    assert resolver._scheme_reason("https://example.org/x") is None
