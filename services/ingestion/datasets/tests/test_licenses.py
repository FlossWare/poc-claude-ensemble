"""Table-driven license normalization and policy tests."""
from __future__ import annotations

import pytest

from ingestion.datasets import licenses


@pytest.mark.parametrize(
    "raw,family,normalized",
    [
        ("CC BY 4.0", "cc-by", "CC-BY-4.0"),
        ("CC-BY-4.0", "cc-by", "CC-BY-4.0"),
        ("https://creativecommons.org/licenses/by/4.0/", "cc-by", "CC-BY-4.0"),
        ("http://creativecommons.org/licenses/by/4.0", "cc-by", "CC-BY-4.0"),
        ("Creative Commons Attribution 4.0 International", "cc-by", "CC-BY-4.0"),
        ("Creative Commons Attribution 4.0 International (CC BY 4.0)", "cc-by", "CC-BY-4.0"),
        ("CC0 1.0", "cc0", "CC0-1.0"),
        ("CC0-1.0", "cc0", "CC0-1.0"),
        ("CC0", "cc0", "CC0-1.0"),
        ("CC BY-SA 4.0", "cc-by-sa", "CC-BY-SA-4.0"),
        ("CC BY-NC 4.0", "cc-by-nc", "CC-BY-NC-4.0"),
        ("CC-BY-NC-4.0", "cc-by-nc", "CC-BY-NC-4.0"),
        ("CC BY-ND 4.0", "cc-by-nd", "CC-BY-ND-4.0"),
        ("CC BY-NC-SA 4.0", "cc-by-nc-sa", "CC-BY-NC-SA-4.0"),
        ("CC BY-NC-ND", "cc-by-nc-nd", "CC-BY-NC-ND"),
        ("MIT", "mit", "MIT"),
        ("MIT License", "mit", "MIT"),
        ("Apache License", "apache", "Apache-2.0"),
        ("Apache 2.0", "apache", "Apache-2.0"),
        ("ODbL 1.0", "odbl", "ODbL-1.0"),
        ("ODbL-1.0", "odbl", "ODbL-1.0"),
        ("Open Government Licence v3.0", "ogl", "OGL-3.0"),
        ("etalab-2.0", "etalab", "etalab-2.0"),
        ("Public domain (U.S. government data)", "public-domain", "public-domain"),
        ("Public Domain (US Government work)", "public-domain", "public-domain"),
        ("other-pd", "public-domain", "public-domain"),
        ("No reuse license granted", "none", "none"),
        ("Proprietary (free tier + paid plans)", "proprietary", "proprietary"),
        ("PolyForm Noncommercial 1.0.0 (source-available)", "proprietary", "proprietary"),
        ("GNU General Public License v3.0", "gpl-3", "GPL-3.0-only"),
    ],
)
def test_observed_values(raw, family, normalized):
    result = licenses.normalize_license(raw)
    assert result.family == family
    assert result.normalized == normalized


def test_empty_and_missing_are_unknown():
    for raw in (None, "", "   ", []):
        result = licenses.normalize_license(raw)
        assert result.family == "unknown"
        assert result.normalized is None
        assert result.review_required is False


def test_unrecognised_value_is_custom_and_needs_review():
    result = licenses.normalize_license("Totally bespoke terms")
    assert result.family == "custom"
    assert result.normalized is None
    assert result.review_required is True


def test_compound_declaration_is_ambiguous_but_primary_is_first():
    result = licenses.normalize_license("CC BY 4.0 (data), MIT (code)")
    assert result.family == "cc-by"
    assert result.ambiguous is True
    assert set(result.matched_families) == {"cc-by", "mit"}


def test_list_of_dicts_license():
    result = licenses.normalize_license(
        [
            {"name": "CC BY 4.0 (VIIRS-derived data)", "url": "https://creativecommons.org/licenses/by/4.0/"},
            {"name": "MIT (code)", "url": "https://opensource.org/licenses/MIT"},
        ]
    )
    assert result.family == "cc-by"
    assert result.ambiguous is True
    assert "https://creativecommons.org/licenses/by/4.0/" in result.evidence_urls


def test_extra_rights_caveat_requires_review():
    result = licenses.normalize_license(
        "CC BY 4.0; quoted source excerpts retain their owners' rights"
    )
    assert result.family == "cc-by"
    assert result.review_required is True


def test_policy_projection():
    allow = licenses.ALLOWED_FAMILIES
    policy, reason, review = licenses.policy_for(
        licenses.normalize_license("CC BY 4.0"), allow
    )
    assert (policy, reason, review) == ("index_content", None, False)

    policy, reason, review = licenses.policy_for(
        licenses.normalize_license(None), allow
    )
    assert policy == "metadata_only_review"
    assert reason == "license_unknown_metadata_only"
    assert review is True

    policy, reason, review = licenses.policy_for(
        licenses.normalize_license("CC BY-NC-ND 4.0"), allow
    )
    assert policy == "metadata_only_review"
    assert reason == "license_restricted"
    assert review is True

    policy, reason, review = licenses.policy_for(
        licenses.normalize_license("Proprietary (paid)"), allow
    )
    assert policy == "blocked"
    assert reason == "license_restricted"
    assert review is True
