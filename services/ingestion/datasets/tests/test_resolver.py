"""PR 2 resolvers: candidate order, host APIs, landing pages, cache, accounting."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import resolver
from ingestion.datasets.config import Config
from ingestion.datasets.netguard import NetguardError

from ingestion.datasets.tests.helpers import FakeProbeTransport

NOW = "2026-01-01T00:00:00Z"


def _config(tmp_path, **overrides):
    config = Config.load(data_root=tmp_path / "data", ledger_enabled=False)
    base = config.replace(
        per_host_rps=10_000.0,
        concurrency=2,
        resolve_cache_enabled=False,
        respect_robots=False,
    )
    return base.replace(**overrides) if overrides else base


def _context(tmp_path, fake, **overrides):
    return resolver.ResolverContext(
        config=_config(tmp_path, **overrides), transport=fake, now=NOW
    )


def _row(
    entry_id="apd:Test/entry.yml",
    *,
    homepage=None,
    sources=None,
    status="deferred",
    reason_code=None,
    metadata_sha256="meta-1",
    disposition="metadata_only_review",
):
    return {
        "entry_id": entry_id,
        "metadata_sha256": metadata_sha256,
        "status": status,
        "reason_code": reason_code,
        "homepage": homepage,
        "sources": sources or [],
        "disposition": disposition,
        "title": entry_id,
        "category": "Test",
    }


def _json_body(payload) -> dict[str, object]:
    return {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


# ---------------------------------------------------------------------------
# Candidates and URL policy
# ---------------------------------------------------------------------------


def test_extract_candidates_accepts_all_source_shapes():
    row = _row(
        homepage="https://home.example/page",
        sources=[
            {"access_url": ["https://a.example/one", "https://a.example/two"]},
            "https://b.example/one",
            {"access_url": "https://c.example/one"},
            {"access_url": None},
            "https://home.example/page",
        ],
    )
    candidates = resolver.extract_candidates(row)
    assert [candidate.url for candidate in candidates] == [
        "https://a.example/one",
        "https://a.example/two",
        "https://b.example/one",
        "https://c.example/one",
        "https://home.example/page",
    ]
    assert candidates[0].origin == "sources.access_url"
    # The homepage duplicates a declared source, so it keeps the source origin.
    assert candidates[-1].origin == "sources.access_url"


def test_extract_candidates_homepage_is_the_last_fallback():
    row = _row(homepage="https://home.example/page", sources=[])
    candidates = resolver.extract_candidates(row)
    assert [(candidate.url, candidate.origin) for candidate in candidates] == [
        ("https://home.example/page", "homepage")
    ]


@pytest.mark.parametrize(
    "url,status,reason",
    [
        ("hhttps://example.org/x", "unresolved", "invalid_url"),
        ("ftp://example.org/x", "unresolved", "unsupported_scheme"),
        ("s3://bucket/key", "unresolved", "unsupported_scheme"),
        ("file:///etc/passwd", "unresolved", "unsupported_scheme"),
        ("https://user:pass@example.org/x", "unresolved", "invalid_url"),
        ("https://example.org:8443/x", "unresolved", "invalid_url"),
        ("/relative/path", "unresolved", "invalid_url"),
    ],
)
def test_hostile_or_unsupported_urls_never_touch_the_network(url, status, reason, tmp_path):
    fake = FakeProbeTransport(default=None)
    result = resolver.resolve_entry(_row(homepage=url), _context(tmp_path, fake))
    assert result.status == status
    assert result.reason_code == reason
    assert fake.calls == []


def test_entry_with_no_urls_has_an_explicit_reason(tmp_path):
    fake = FakeProbeTransport(default=None)
    result = resolver.resolve_entry(_row(homepage=None, sources=[]), _context(tmp_path, fake))
    assert result.status == "unresolved"
    assert result.reason_code == "no_source_url"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# HEAD / Range probing
# ---------------------------------------------------------------------------


def test_head_probe_resolves_a_direct_file(tmp_path):
    fake = FakeProbeTransport(
        {
            ("https://example.org/data.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv; charset=utf-8", "content-length": "123"},
            }
        }
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/data.csv"), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    assert result.kind == "direct_file"
    assert result.method == "head_probe"
    assert result.sources[0].uri == "https://example.org/data.csv"
    assert result.sources[0].size_bytes == 123
    assert result.sources[0].media_type == "text/csv"
    assert all(call["store_body"] is False for call in fake.calls)


def test_head_falls_back_to_a_bounded_range_get(tmp_path):
    fake = FakeProbeTransport(
        {
            ("https://example.org/archive.zip", "HEAD"): {"status": 405},
            ("https://example.org/archive.zip", "GET"): {
                "status": 206,
                "headers": {
                    "content-type": "application/zip",
                    "content-length": "1",
                    "content-range": "bytes 0-0/999",
                },
            },
        }
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/archive.zip"), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    assert result.kind == "archive"
    # A HEAD-capable file yields 206 from the Range fallback; the designed
    # total comes from Content-Range, which the resolver leaves to acquisition.
    assert result.sources[0].media_type == "application/zip"
    range_calls = [call for call in fake.calls if call["method"] == "GET"]
    assert range_calls and range_calls[0]["headers"]["Range"] == "bytes=0-0"
    assert all(call["store_body"] is False for call in fake.calls)


@pytest.mark.parametrize(
    "http_status,status,reason,retryable",
    [
        (401, "unresolved", "gated_authentication", False),
        (403, "deferred", "blocked", True),
        (404, "deferred", "not_found", True),
        (410, "unresolved", "gone", False),
        (429, "deferred", "rate_limited", True),
        (500, "deferred", "server_error", True),
        (418, "unresolved", "http_error", False),
    ],
)
def test_http_status_mapping(http_status, status, reason, retryable, tmp_path):
    fake = FakeProbeTransport({("https://example.org/x", "HEAD"): {"status": http_status}})
    result = resolver.resolve_entry(_row(homepage="https://example.org/x"), _context(tmp_path, fake))
    assert result.status == status
    assert result.reason_code == reason
    assert result.retryable is retryable


@pytest.mark.parametrize(
    "code,status,reason",
    [
        ("timeout", "deferred", "timeout"),
        ("dns_failure", "deferred", "dns_failure"),
        ("tls_failure", "deferred", "tls_failure"),
        ("ssrf_blocked", "unresolved", "ssrf_blocked"),
        ("unsafe_redirect", "unresolved", "unsafe_redirect"),
        ("too_many_redirects", "unresolved", "too_many_redirects"),
    ],
)
def test_transport_error_mapping(code, status, reason, tmp_path):
    fake = FakeProbeTransport({("https://example.org/x", "HEAD"): NetguardError(code)})
    result = resolver.resolve_entry(_row(homepage="https://example.org/x"), _context(tmp_path, fake))
    assert result.status == status
    assert result.reason_code == reason
    assert result.method == "head_probe"


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def test_github_blob_is_transformed_to_raw(tmp_path):
    raw = "https://raw.githubusercontent.com/o/r/main/data/x.csv"
    fake = FakeProbeTransport(
        {
            (raw, "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv", "content-length": "7"},
            }
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r/blob/main/data/x.csv"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "github_blob_raw"
    assert result.sources[0].uri == raw
    assert result.sources[0].version == "main"
    assert result.sources[0].member == "x.csv"


def test_github_repository_resolves_as_repository(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://api.github.com/repos/o/r": _json_body(
                {"full_name": "o/r", "default_branch": "main", "size": 4}
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r"}]), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    assert result.kind == "repository"
    assert result.method == "github_repo_api"
    assert result.version == "main"
    assert result.sources[0].uri == "https://github.com/o/r"
    assert result.sources[0].size_bytes == 4096


def test_github_tree_uses_contents_api(tmp_path):
    api = "https://api.github.com/repos/o/r/contents/data?ref=main"
    fake = FakeProbeTransport(
        {
            api: _json_body(
                [
                    {
                        "type": "file",
                        "name": "x.csv",
                        "path": "data/x.csv",
                        "size": 10,
                        "download_url": "https://raw.githubusercontent.com/o/r/main/data/x.csv",
                    },
                    {
                        "type": "file",
                        "name": "README.md",
                        "path": "data/README.md",
                        "size": 2,
                        "download_url": "https://raw.githubusercontent.com/o/r/main/data/README.md",
                    },
                ]
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r/tree/main/data"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "github_contents_api"
    assert [source.member for source in result.sources] == ["x.csv"]
    assert result.sources[0].version == "main"


def test_github_tree_without_data_falls_back_to_repository(tmp_path):
    contents = "https://api.github.com/repos/o/r/contents/docs?ref=main"
    repo = "https://api.github.com/repos/o/r"
    fake = FakeProbeTransport(
        {
            contents: _json_body(
                [
                    {
                        "type": "file",
                        "name": "README.md",
                        "size": 2,
                        "download_url": "https://raw.githubusercontent.com/o/r/main/docs/README.md",
                    }
                ]
            ),
            repo: _json_body({"full_name": "o/r", "default_branch": "main", "size": 1}),
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r/tree/main/docs"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.kind == "repository"
    assert result.method == "github_repo_api"


def test_github_release_asset_falls_back_to_a_direct_probe(tmp_path):
    url = "https://github.com/o/r/releases/download/v1/data.csv"
    fake = FakeProbeTransport(
        {
            (url, "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv", "content-length": "4"},
            }
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": url}]), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    assert result.sources[0].uri == url
    assert result.kind == "direct_file"


def test_github_api_rate_limit_is_retryable(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://api.github.com/repos/o/r": {
                "status": 403,
                "headers": {"content-type": "application/json"},
                "body": json.dumps({"message": "API rate limit exceeded"}),
            }
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r"}]), _context(tmp_path, fake)
    )
    assert result.status == "deferred"
    assert result.reason_code == "rate_limited"


# ---------------------------------------------------------------------------
# Zenodo, DOI, Hugging Face, archive.org, Dataverse
# ---------------------------------------------------------------------------


def test_zenodo_record_files(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://zenodo.org/api/records/123": _json_body(
                {
                    "id": 123,
                    "metadata": {"version": "2.0"},
                    "files": [
                        {
                            "key": "data.zip",
                            "size": 10,
                            "links": {"self": "https://zenodo.org/api/records/123/files/data.zip/content"},
                        }
                    ],
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://zenodo.org/records/123"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "zenodo_api"
    assert result.kind == "archive"
    assert result.version == "2.0"
    assert result.sources[0].uri.endswith("/data.zip/content")


def test_host_listing_accepts_unknown_extensions(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://zenodo.org/api/records/9": _json_body(
                {
                    "id": 9,
                    "metadata": {},
                    "files": [{"key": "data.bin", "size": 4, "links": {}}],
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://zenodo.org/records/9"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.sources[0].kind == "direct_file"


def test_documentation_only_listing_does_not_resolve(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://zenodo.org/api/records/9": _json_body(
                {
                    "id": 9,
                    "metadata": {},
                    "files": [{"key": "README.md", "size": 4, "links": {}}],
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://zenodo.org/records/9"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "deferred"
    assert result.reason_code == "landing_page_only"


def test_zenodo_doi_routes_to_the_api(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://zenodo.org/api/records/123": _json_body(
                {
                    "id": 123,
                    "metadata": {},
                    "files": [{"key": "x.csv", "size": 3, "links": {}}],
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://doi.org/10.5281/zenodo.123"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "zenodo_api"
    assert result.sources[0].kind == "direct_file"


def test_doi_content_negotiation(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://doi.org/10.1234/abc": _json_body(
                {"URL": "https://example.org/data.csv"}
            ),
            ("https://example.org/data.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv", "content-length": "5"},
            },
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://doi.org/10.1234/abc"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "doi_content_negotiation"
    assert result.sources[0].uri == "https://example.org/data.csv"


def test_doi_without_a_target_is_retryable(tmp_path):
    fake = FakeProbeTransport(
        {"https://doi.org/10.1234/abc": _json_body({"title": "no target"})}
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://doi.org/10.1234/abc"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "deferred"
    assert result.reason_code == "doi_unresolved"


def test_huggingface_dataset_files(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://huggingface.co/api/datasets/org/name": _json_body(
                {
                    "id": "org/name",
                    "sha": "abc123",
                    "gated": False,
                    "siblings": [{"rfilename": "data.csv"}, {"rfilename": "README.md"}],
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://huggingface.co/datasets/org/name"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "huggingface_api"
    assert result.version == "abc123"
    assert result.sources[0].uri == (
        "https://huggingface.co/datasets/org/name/resolve/abc123/data.csv"
    )


def test_huggingface_gated_is_explicit(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://huggingface.co/api/datasets/org/name": _json_body(
                {"id": "org/name", "gated": True, "siblings": []}
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://huggingface.co/datasets/org/name"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "gated_authentication"


def test_archive_org_metadata(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://archive.org/metadata/item1": _json_body(
                {
                    "files": [
                        {"name": "data.csv", "size": "12"},
                        {"name": "item1_meta.xml", "size": "1"},
                    ]
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://archive.org/details/item1"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "archive_org_metadata_api"
    assert result.sources[0].uri == "https://archive.org/download/item1/data.csv"
    assert result.sources[0].size_bytes == 12


def test_dataverse_dataset(tmp_path):
    api = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi%3A10.7910%2FDVN%2FX"
    )
    fake = FakeProbeTransport(
        {
            api: _json_body(
                {
                    "data": {
                        "latestVersion": {
                            "versionNumber": 3,
                            "versionMinorNumber": 1,
                            "files": [
                                {
                                    "dataFile": {
                                        "id": 7,
                                        "filename": "data.tab",
                                        "filesize": 99,
                                        "contentType": "text/tab-separated-values",
                                    }
                                }
                            ],
                        }
                    }
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(
            sources=[
                {
                    "access_url": (
                        "https://dataverse.harvard.edu/dataset.xhtml"
                        "?persistentId=doi:10.7910/DVN/X"
                    )
                }
            ]
        ),
        _context(tmp_path, fake),
    )
    assert result.status == "resolved"
    assert result.method == "dataverse_api"
    assert result.version == "3.1"
    assert result.sources[0].uri == "https://dataverse.harvard.edu/api/access/datafile/7"


def test_dataverse_restricted_files_are_gated(tmp_path):
    api = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi%3A10.7910%2FDVN%2FX"
    )
    fake = FakeProbeTransport(
        {
            api: _json_body(
                {
                    "data": {
                        "latestVersion": {
                            "files": [
                                {
                                    "dataFile": {
                                        "id": 7,
                                        "filename": "data.csv",
                                        "restricted": True,
                                    }
                                }
                            ]
                        }
                    }
                }
            )
        }
    )
    result = resolver.resolve_entry(
        _row(
            sources=[
                {
                    "access_url": (
                        "https://dataverse.harvard.edu/dataset.xhtml"
                        "?persistentId=doi:10.7910/DVN/X"
                    )
                }
            ]
        ),
        _context(tmp_path, fake),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "gated_authentication"


# ---------------------------------------------------------------------------
# Landing pages
# ---------------------------------------------------------------------------


def _landing_fixture(url, html_text, *, robots=None, extra=None):
    responses = {
        (url, "HEAD"): {"status": 200, "headers": {"content-type": "text/html"}},
        (url, "GET"): {
            "status": 200,
            "headers": {"content-type": "text/html; charset=utf-8"},
            "body": html_text,
        },
    }
    if robots is not None:
        responses[("https://example.org/robots.txt", "GET")] = {
            "status": 200,
            "headers": {"content-type": "text/plain"},
            "body": robots,
        }
    if extra:
        responses.update(extra)
    return responses


def test_landing_page_finds_relative_data_links(tmp_path):
    html_text = '<html><body><a href="files/data.csv">download</a></body></html>'
    fake = FakeProbeTransport(_landing_fixture("https://example.org/page", html_text))
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    assert result.method == "landing_page"
    assert result.sources[0].uri == "https://example.org/files/data.csv"
    landing_gets = [
        call for call in fake.calls if call["method"] == "GET" and call["url"].endswith("/page")
    ]
    assert landing_gets and landing_gets[0]["store_body"] is True
    assert landing_gets[0]["max_bytes"] <= _config(tmp_path).landing_page_max_bytes


def test_landing_page_prefers_same_host_and_ignores_hostile_links(tmp_path):
    html_text = """
    <html><body>
      <a href="javascript:alert(1)">js</a>
      <a href="data:text/csv;base64,AAAA">data</a>
      <a href="file:///etc/passwd">file</a>
      <a href="mailto:x@example.org">mail</a>
      <a href="https://other.example/data.csv">remote</a>
      <a href="/local.csv">local</a>
    </body></html>
    """
    fake = FakeProbeTransport(_landing_fixture("https://example.org/page", html_text))
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert result.status == "resolved"
    uris = [source.uri for source in result.sources]
    assert uris[0] == "https://example.org/local.csv"
    assert all(not uri.startswith(("javascript:", "data:", "file:", "mailto:")) for uri in uris)


def test_landing_page_javascript_and_interaction(tmp_path):
    js = "<html><body>Please enable JavaScript to view this dataset.</body></html>"
    fake = FakeProbeTransport(_landing_fixture("https://example.org/page", js))
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert result.reason_code == "requires_javascript"

    form = '<html><body><form action="/login"><input type="password"></form></body></html>'
    fake = FakeProbeTransport(_landing_fixture("https://example.org/page", form))
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert result.reason_code == "requires_interaction"


def test_landing_page_without_links_is_retryable(tmp_path):
    fake = FakeProbeTransport(
        _landing_fixture("https://example.org/page", "<html><body>nothing here</body></html>")
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"), _context(tmp_path, fake)
    )
    assert result.status == "deferred"
    assert result.reason_code == "landing_page_only"
    assert result.method == "landing_page"


def test_robots_disallowed_prevents_the_landing_fetch(tmp_path):
    responses = _landing_fixture(
        "https://example.org/page",
        '<a href="/data.csv">data</a>',
        robots="User-agent: *\nDisallow: /",
    )
    fake = FakeProbeTransport(responses)
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"),
        _context(tmp_path, fake, respect_robots=True),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "robots_disallowed"
    assert result.method == "landing_page"
    assert not any(
        call["method"] == "GET" and call["url"] == "https://example.org/page"
        for call in fake.calls
    )


def test_landing_page_link_cap_is_numeric(tmp_path):
    anchors = "".join(f'<a href="/data{i}.csv">d</a>' for i in range(10))
    fake = FakeProbeTransport(
        _landing_fixture("https://example.org/page", f"<html><body>{anchors}</body></html>")
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"),
        _context(tmp_path, fake, max_resolved_sources=3),
    )
    assert result.status == "resolved"
    assert len(result.sources) == 3
    assert result.total_sources == 10


def test_oversized_landing_page_is_refused(tmp_path):
    fake = FakeProbeTransport(
        _landing_fixture("https://example.org/page", "x" * 500)
    )
    result = resolver.resolve_entry(
        _row(homepage="https://example.org/page"),
        _context(tmp_path, fake, landing_page_max_bytes=100),
    )
    assert result.status == "deferred"
    assert result.reason_code == "oversized_declared"


# ---------------------------------------------------------------------------
# API hardening and partial failure
# ---------------------------------------------------------------------------


def test_api_response_limit_and_shape(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://api.github.com/repos/o/r": {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": "{" + "x" * 100 + "}",
            }
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r"}]),
        _context(tmp_path, fake, api_max_bytes=20),
    )
    assert result.reason_code == "oversized_declared"

    fake = FakeProbeTransport(
        {
            "https://api.github.com/repos/o/r": _json_body(["not", "an", "object"])
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r"}]),
        _context(tmp_path, fake),
    )
    assert result.reason_code == "parse_error"


def test_invalid_json_is_a_parse_error(tmp_path):
    fake = FakeProbeTransport(
        {
            "https://api.github.com/repos/o/r": {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": "not json at all",
            }
        }
    )
    result = resolver.resolve_entry(
        _row(sources=[{"access_url": "https://github.com/o/r"}]),
        _context(tmp_path, fake),
    )
    assert result.status == "unresolved"
    assert result.reason_code == "parse_error"


def test_partial_failure_is_isolated_per_entry(tmp_path):
    fake = FakeProbeTransport(
        {
            ("https://example.org/ok.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv"},
            }
        }
    )
    rows = [
        _row("apd:Test/a.yml", homepage="https://example.org/ok.csv"),
        _row("apd:Test/b.yml", homepage="https://example.org/missing"),
    ]
    resolutions, unhandled, _ = resolver.resolve_entries(
        rows,
        {"apd:Test/a.yml", "apd:Test/b.yml"},
        _config(tmp_path),
        transport=fake,
        now=NOW,
    )
    by_entry = {resolution.entry_id: resolution for resolution in resolutions}
    assert by_entry["apd:Test/a.yml"].status == "resolved"
    # FakeProbeTransport raises AssertionError for the unconfigured URL, which
    # the candidate router turns into an explicit resolver_error.
    assert by_entry["apd:Test/b.yml"].status == "unresolved"
    assert by_entry["apd:Test/b.yml"].reason_code == "resolver_error"
    assert unhandled == 0


# ---------------------------------------------------------------------------
# Cache, idempotency and resume
# ---------------------------------------------------------------------------


def test_resolved_results_are_cached_without_network(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    rows = [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")]
    cache = resolver.ResolutionCache(config)

    first_transport = FakeProbeTransport(
        {
            ("https://example.org/a.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv", "content-length": "3"},
            }
        }
    )
    first, _, context = resolver.resolve_entries(
        rows, {"apd:Test/a.yml"}, config, transport=first_transport, cache=cache, now=NOW
    )
    assert first[0].status == "resolved"
    assert context.cache_misses == 1
    assert first_transport.calls

    second_transport = FakeProbeTransport(default=None)
    second, _, context = resolver.resolve_entries(
        rows, {"apd:Test/a.yml"}, config, transport=second_transport, cache=cache, now=NOW
    )
    assert second[0].status == "resolved"
    assert second[0].cached is True
    assert context.cache_hits == 1
    assert second_transport.calls == []


def test_retryable_failure_is_reused_until_next_retry_at(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    rows = [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")]
    cache = resolver.ResolutionCache(config)
    first_transport = FakeProbeTransport(
        {("https://example.org/a.csv", "HEAD"): {"status": 404}}
    )
    first, _, _ = resolver.resolve_entries(
        rows, {"apd:Test/a.yml"}, config, transport=first_transport, cache=cache, now=NOW
    )
    assert first[0].status == "deferred"
    assert first[0].next_retry_at is not None

    early_transport = FakeProbeTransport(default=None)
    early, _, _ = resolver.resolve_entries(
        rows,
        {"apd:Test/a.yml"},
        config,
        transport=early_transport,
        cache=cache,
        now="2026-01-01T00:30:00Z",
    )
    assert early[0].status == "deferred"
    assert early_transport.calls == []

    due_transport = FakeProbeTransport(
        {
            ("https://example.org/a.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv"},
            }
        }
    )
    due, _, _ = resolver.resolve_entries(
        rows,
        {"apd:Test/a.yml"},
        config,
        transport=due_transport,
        cache=cache,
        now="2026-01-02T00:00:00Z",
    )
    assert due[0].status == "resolved"
    assert due_transport.calls


def test_corrupt_cache_entry_is_ignored(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    cache = resolver.ResolutionCache(config)
    path = cache.path_for("apd:Test/a.yml", "meta-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "entry_id": "apd:Test/a.yml",
                "status": "bogus",
                "checked_at": NOW,
            }
        ),
        encoding="utf-8",
    )
    fake = FakeProbeTransport(
        {
            ("https://example.org/a.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv"},
            }
        }
    )
    results, _, _ = resolver.resolve_entries(
        [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")],
        {"apd:Test/a.yml"},
        config,
        transport=fake,
        cache=cache,
        now=NOW,
    )
    assert results[0].status == "resolved"
    assert fake.calls


def test_cache_is_invalidated_by_metadata_change_and_disabled_by_flag(tmp_path):
    config = _config(tmp_path, resolve_cache_enabled=True)
    cache = resolver.ResolutionCache(config)
    fake = FakeProbeTransport(
        {
            ("https://example.org/a.csv", "HEAD"): {
                "status": 200,
                "headers": {"content-type": "text/csv"},
            }
        }
    )
    resolver.resolve_entries(
        [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")],
        {"apd:Test/a.yml"},
        config,
        transport=fake,
        cache=cache,
        now=NOW,
    )
    calls_before = len(fake.calls)
    resolver.resolve_entries(
        [
            _row(
                "apd:Test/a.yml",
                homepage="https://example.org/a.csv",
                metadata_sha256="meta-2",
            )
        ],
        {"apd:Test/a.yml"},
        config,
        transport=fake,
        cache=cache,
        now=NOW,
    )
    assert len(fake.calls) > calls_before

    disabled = _config(tmp_path, resolve_cache_enabled=False)
    resolver.resolve_entries(
        [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")],
        {"apd:Test/a.yml"},
        disabled,
        transport=fake,
        cache=cache,
        now=NOW,
    )
    assert len(fake.calls) > calls_before * 2 - calls_before


# ---------------------------------------------------------------------------
# Selection, accounting, artifacts and outcomes
# ---------------------------------------------------------------------------


def test_stratified_sample_spans_dispositions_deterministically():
    rows = []
    for disposition, count in (
        ("metadata_only_review", 5),
        ("index_content", 4),
        ("blocked", 1),
    ):
        for index in range(count):
            rows.append(
                _row(
                    f"apd:Test/{disposition}-{index}.yml",
                    homepage="https://example.org/x",
                    disposition=disposition,
                )
            )
    selected, truncated = resolver.select_entries(rows, sample=6)
    assert truncated is True
    chosen = [row for row in rows if row["entry_id"] in selected]
    assert {row["disposition"] for row in chosen} == {
        "metadata_only_review",
        "index_content",
        "blocked",
    }
    # Round-robin: two from each disposition before any gets a third.
    counts = {}
    for row in chosen:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    assert counts == {"blocked": 1, "index_content": 3, "metadata_only_review": 2}

    again, _ = resolver.select_entries(list(reversed(rows)), sample=6)
    assert again == selected


def test_complete_resolutions_and_accounting_hold(tmp_path):
    rows = [
        _row(f"apd:Test/{index}.yml", homepage=f"https://example.org/{index}.csv")
        for index in range(5)
    ]
    resolutions = [
        resolver.Resolution(
            entry_id="apd:Test/0.yml",
            status="resolved",
            reason_code=None,
            checked_at=NOW,
        )
    ]
    completed = resolver.complete_resolutions(
        rows, resolutions, now=NOW, config=_config(tmp_path)
    )
    assert len(completed) == 5
    accounting = resolver.check_resolution_accounting(completed, 5)
    assert accounting.ok is True
    assert accounting.status_counts == {"deferred": 4, "resolved": 1}
    assert all(item.reason_code for item in completed if item.status != "resolved")

    bad = resolver.check_resolution_accounting(completed[:4], 5)
    assert bad.ok is False
    assert any(problem.startswith("row_count:") for problem in bad.problems)


def test_budget_rows_do_not_make_the_manifest_time_dependent(tmp_path):
    rows = [
        _row(f"apd:Test/{index}.yml", homepage=f"https://example.org/{index}.csv")
        for index in range(3)
    ]
    config = _config(tmp_path)
    first = resolver.complete_resolutions(rows, [], now=NOW, config=config)
    second = resolver.complete_resolutions(
        rows, [], now="2026-06-01T00:00:00Z", config=config
    )
    first_bytes = resolver.resolution_manifest_bytes(rows, first)
    second_bytes = resolver.resolution_manifest_bytes(rows, second)
    assert first_bytes == second_bytes
    artifact = json.loads(
        resolver.resolution_manifest_bytes(rows, first).decode("utf-8").splitlines()[0]
    )
    assert artifact["resolution"]["checked_at"] is None
    assert artifact["resolution"]["next_retry_at"] is None
    assert artifact["resolution"]["next_action"] == "resume"


def test_failed_and_duplicate_reasons_are_accounted(tmp_path):
    rows = [
        _row("apd:Test/failed.yml", status="failed", reason_code="catalog_parse_error"),
        _row("apd:Test/reason.yml", homepage="ftp://example.org/x"),
    ]
    fake = FakeProbeTransport(default=None)
    reason_resolution = resolver.resolve_entry(rows[1], _context(tmp_path, fake))
    assert fake.calls == []
    completed = resolver.complete_resolutions(
        rows, [reason_resolution], now=NOW, config=_config(tmp_path)
    )
    accounting = resolver.check_resolution_accounting(completed, 2)
    assert accounting.ok is True
    assert accounting.status_counts == {"failed": 1, "unresolved": 1}
    by_entry = {item.entry_id: item for item in completed}
    assert by_entry["apd:Test/failed.yml"].reason_code == "catalog_parse_error"
    assert by_entry["apd:Test/reason.yml"].reason_code == "unsupported_scheme"


def test_artifacts_and_outcomes_are_deterministic_and_append_once(tmp_path):
    rows = [_row("apd:Test/a.yml", homepage="https://example.org/a.csv")]
    resolution = resolver.Resolution(
        entry_id="apd:Test/a.yml",
        status="resolved",
        reason_code=None,
        checked_at=NOW,
        sources=(
            resolver.ResolvedSource(
                uri="https://example.org/a.csv",
                kind="direct_file",
                method="head_probe",
                media_type="text/csv",
                size_bytes=3,
            ),
        ),
        method="head_probe",
        kind="direct_file",
        total_sources=1,
    )
    first = resolver.resolution_manifest_bytes(rows, [resolution])
    second = resolver.resolution_manifest_bytes(rows, [resolution])
    assert first == second
    artifact = json.loads(first.decode("utf-8").splitlines()[0])
    assert artifact["resolution"]["next_action"] == "acquire"

    path = tmp_path / "outcomes.jsonl"
    added = resolver.append_resolution_outcomes(
        rows,
        [resolution],
        path,
        run_id="run",
        catalog_commit="a" * 40,
        retrieved_at=NOW,
    )
    assert added == 1
    again = resolver.append_resolution_outcomes(
        rows,
        [resolution],
        path,
        run_id="run",
        catalog_commit="a" * 40,
        retrieved_at=NOW,
    )
    assert again == 0
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_artifact_sanitizes_queries(tmp_path):
    rows = [
        _row(
            "apd:Test/dv.yml",
            sources=[
                {
                    "access_url": (
                        "https://dataverse.harvard.edu/dataset.xhtml"
                        "?persistentId=doi:SECRET"
                    )
                }
            ],
        )
    ]
    resolution = resolver.Resolution(
        entry_id="apd:Test/dv.yml",
        status="unresolved",
        reason_code="gated_authentication",
        checked_at=NOW,
        attempts=(
            resolver.Attempt(
                url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:SECRET",
                method="dataverse_api",
                status="failed",
                reason_code="gated_authentication",
            ),
        ),
    )
    encoded = resolver.resolution_manifest_bytes(rows, [resolution])
    assert b"SECRET" not in encoded
    assert b"?" not in encoded
