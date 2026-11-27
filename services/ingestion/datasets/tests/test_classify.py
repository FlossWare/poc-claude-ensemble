"""Classification of access, format, scheme, size and disposition."""
from __future__ import annotations

import pytest

from ingestion.datasets import classify, licenses, probe

from ingestion.datasets.tests.helpers import make_config


def _row(
    license_raw,
    *,
    homepage="https://example.org/data",
    sources=None,
    extra=None,
    entry_id="apd:Test/entry.yml",
):
    result = licenses.normalize_license(license_raw)
    policy, reason, review = licenses.policy_for(result, licenses.ALLOWED_FAMILIES)
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "homepage": homepage,
        "sources": sources or [],
        "extra": extra or {},
        "license": {
            "dataset": {
                "family": result.family,
                "normalized": result.normalized,
                "policy": policy,
                "reason_code": reason,
                "review_required": review,
            }
        },
    }


def test_formats_from_extensions_and_source_names(tmp_path):
    config = make_config(tmp_path)
    row = _row(
        "MIT",
        sources=[
            {"name": "CSV download", "access_url": "https://example.org/x.csv"},
            {"name": "JSON mirror", "access_url": "https://example.org/api/x"},
            {"name": "Parquet mirror", "access_url": "https://example.org/x.PARQUET"},
        ],
    )
    assert classify.formats_for(row) == ["csv", "json", "parquet"]


def test_unknown_format_is_explicit(tmp_path):
    row = _row("MIT", homepage="https://example.org/landing", sources=[])
    assert classify.formats_for(row) == ["unknown"]


def test_scheme_anomalies(tmp_path):
    config = make_config(tmp_path)
    row = _row(
        "MIT",
        homepage="hhttps://example.org/h",
        sources=[{"name": "x", "access_url": "ftp://ftp.example.org/x.csv"}],
    )
    schemes, unsupported = classify.schemes_for(row, config)
    assert schemes == ["ftp", "hhttps"]
    assert unsupported == ["ftp", "hhttps"]


def test_access_hints(tmp_path):
    config = make_config(tmp_path)
    assert classify.access_hint(_row("MIT", homepage="https://www.kaggle.com/x"), config) == (
        "gated",
        "gated_authentication",
    )
    assert classify.access_hint(_row("MIT", homepage="ftp://example.org/x"), config) == (
        "inaccessible",
        "unsupported_scheme",
    )
    assert classify.access_hint(_row("MIT", homepage="hhttps://example.org/x"), config) == (
        "inaccessible",
        "unsupported_scheme",
    )
    assert classify.access_hint(_row("MIT", homepage=None, sources=[]), config) == (
        "unknown",
        "missing_url",
    )
    assert classify.access_hint(_row("MIT", homepage="https://huggingface.co/x"), config) == (
        "unknown",
        None,
    )


@pytest.mark.parametrize(
    "size,expected",
    [
        (None, "unknown"),
        (0, "small"),
        (64 * 1024 * 1024, "small"),
        (64 * 1024 * 1024 + 1, "medium"),
        (1024 * 1024 * 1024, "medium"),
        (1024 * 1024 * 1024 + 1, "large"),
    ],
)
def test_size_classes(size, expected):
    assert classify.size_class(size) == expected


def _classified(row, config, probe_result=None):
    classify.classify_row(row, config, probe_result)
    return row


def test_status_for_allowed_license_is_deferred_for_resolution(tmp_path):
    config = make_config(tmp_path)
    row = _classified(_row("CC BY 4.0"), config)
    assert row["disposition"] == "index_content"
    assert row["status"] == "deferred"
    assert row["reason_code"] == "resolver_not_implemented"
    assert row["next_action"] == "resolve"
    assert row["classes"]["size"] == {"class": "unknown", "bytes": None, "source": "metadata"}
    assert row["classes"]["access"] == "unknown"


def test_status_for_unknown_license_is_metadata_only(tmp_path):
    config = make_config(tmp_path)
    row = _classified(_row(None), config)
    assert row["disposition"] == "metadata_only_review"
    assert row["status"] == "deferred"
    assert row["reason_code"] == "license_unknown_metadata_only"
    assert row["next_action"] == "review_license"


def test_status_for_restricted_license_is_metadata_only(tmp_path):
    config = make_config(tmp_path)
    row = _classified(_row("CC BY-NC-ND 4.0"), config)
    assert row["disposition"] == "metadata_only_review"
    assert row["reason_code"] == "license_restricted"


def test_status_for_proprietary_license_is_skipped(tmp_path):
    config = make_config(tmp_path)
    row = _classified(_row("Proprietary (paid)"), config)
    assert row["disposition"] == "blocked"
    assert row["status"] == "skipped"
    assert row["reason_code"] == "license_restricted"
    assert row["next_action"] == "none"


def test_allowed_but_ambiguous_license_needs_review(tmp_path):
    config = make_config(tmp_path)
    row = _classified(_row("CC BY 4.0 (data), MIT (code)"), config)
    assert row["disposition"] == "index_content"
    assert row["status"] == "deferred"
    assert row["reason_code"] == "license_review_required"


def test_probe_result_overrides_access_and_size(tmp_path):
    config = make_config(tmp_path)
    row = _row("CC BY 4.0")
    result = probe.ProbeResult(
        url="https://example.org/data",
        method="HEAD",
        status=200,
        final_url="https://example.org/data",
        content_type="text/csv",
        content_length=1234,
        redirect_chain=(),
        checked_at="2026-01-01T00:00:00Z",
        access="public",
        reason_code=None,
    )
    _classified(row, config, result)
    assert row["classes"]["access"] == "public"
    assert row["classes"]["size"] == {"class": "small", "bytes": 1234, "source": "probe"}
    assert row["probe"]["status"] == 200
    assert row["probe"]["checked_at"] == "2026-01-01T00:00:00Z"


def test_probe_failure_overrides_access_but_not_license_status(tmp_path):
    config = make_config(tmp_path)
    row = _row("CC BY 4.0")
    result = probe.ProbeResult(
        url="https://example.org/data",
        method=None,
        status=404,
        final_url="https://example.org/data",
        content_type=None,
        content_length=None,
        redirect_chain=(),
        checked_at="2026-01-01T00:00:00Z",
        access="inaccessible",
        reason_code="not_found",
    )
    _classified(row, config, result)
    assert row["classes"]["access"] == "inaccessible"
    assert row["classes"]["access_reason"] == "not_found"
    assert row["status"] == "deferred"
    # the probe's concrete verdict surfaces as reason_code; reporting
    # resolver_not_implemented when a probe measured not_found is the
    # defect the full-codebase review confirmed
    assert row["reason_code"] == "not_found"
