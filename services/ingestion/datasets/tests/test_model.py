"""Normalization, stable identifiers and YAML loading behaviour."""
from __future__ import annotations

import datetime

import pytest

from ingestion.datasets import model

from ingestion.datasets.tests.helpers import FIXTURES, make_config


def _load(name: str, config):
    return model.load_yaml_entry((FIXTURES / name).read_bytes(), config)


def _normalize(name: str, config, **overrides):
    document, warnings = _load(name, config)
    values = {
        "category": "Test",
        "filename": name,
        "relpath": f"Test/{name}",
        "meta_url": "https://example.org/meta",
        "readme_status": "OK",
        "warnings": warnings,
    }
    values.update(overrides)
    return model.normalize_entry(document, **values)


def test_every_fixture_normalizes(tmp_path):
    config = make_config(tmp_path)
    for name in (
        "full.yml",
        "sparse.yml",
        "license-list.yml",
        "typo-name.yml",
        "hhttps.yml",
        "ftp.yml",
        "San Diego, CA.yml",
        "OpenNEURO",
        "LocationData.wml",
        "duplicate-keys.yml",
        "restricted.yml",
        "proprietary.yml",
    ):
        result = _normalize(name, config)
        assert result["entry_id"].startswith("apd:Test/")
        assert len(result["metadata_sha256"]) == 64


def test_full_entry_fields_and_unknown_keys(tmp_path):
    config = make_config(tmp_path)
    result = _normalize("full.yml", config)
    fields = result["normalized"]
    assert fields["title"] == "Full Entry"
    assert fields["keywords"] == ["alpha", "beta"]
    assert fields["tags"] == ["one", "two"]
    assert fields["publisher"] == [{"name": "Example Publisher", "web": "https://example.org"}]
    assert fields["organization"] == [{"name": "Example Org", "web": "https://example.org"}]
    assert fields["issued_time"] == "2024-01-01"
    assert fields["sources"][0]["access_url"] == "https://example.org/full.csv"
    assert result["extra"]["unknown_key"] == {"nested": "kept"}


def test_publisher_and_organization_variants(tmp_path):
    config = make_config(tmp_path)
    document, _ = _load("license-list.yml", config)
    result = model.normalize_entry(
        document,
        category="Test",
        filename="license-list.yml",
        relpath="Test/license-list.yml",
        meta_url=None,
        readme_status="OK",
    )
    assert result["normalized"]["publisher"] == [{"name": "Example String", "web": None}]
    assert result["normalized"]["organization"] == [
        {"name": "Dict Org", "web": "https://example.org/org"}
    ]
    # list[str] sources: URLs become access_url values, prose becomes a name.
    sources = result["normalized"]["sources"]
    assert sources[0]["access_url"] == "https://example.org/list.json"
    assert sources[1]["name"] == "Some prose source"
    assert sources[1]["access_url"] is None
    # list[dict] license: first recognised family wins, ambiguity recorded.
    assert result["license"].family == "cc-by"
    assert result["license"].ambiguous is True
    assert result["normalized"]["issued_time"] == "2024.5"


def test_typo_name_key_is_accepted_and_flagged(tmp_path):
    config = make_config(tmp_path)
    result = _normalize("typo-name.yml", config)
    assert result["normalized"]["publisher"] == [
        {"name": "Typo Publisher", "web": "https://example.org/typo"}
    ]
    assert result["normalized"]["organization"] == [
        {"name": "Typo Organization", "web": None}
    ]
    assert "publisher_typo_key:-name" in result["warnings"]
    assert "organization_typo_key:-name" in result["warnings"]


def test_duplicate_keys_are_detected_not_silent(tmp_path):
    config = make_config(tmp_path)
    document, warnings = _load("duplicate-keys.yml", config)
    assert "duplicate_key:version" in warnings
    assert "duplicate_key:license" in warnings
    # SafeLoader's deterministic last-wins behaviour is retained.
    assert document["version"] == 2.0
    assert document["license"] == "CC BY 4.0"


def test_issued_time_variants(tmp_path):
    config = make_config(tmp_path)
    document = {
        "title": "Dates",
        "homepage": "https://example.org",
        "category": "Test",
        "issued_time": datetime.date(2020, 5, 4),
    }
    result = model.normalize_entry(
        document,
        category="Test",
        filename="dates.yml",
        relpath="Test/dates.yml",
        meta_url=None,
        readme_status="missing",
    )
    assert result["normalized"]["issued_time"] == "2020-05-04"
    for value, expected in ((2024, "2024"), (2024.5, "2024.5"), (" 2024-06 ", "2024-06")):
        document["issued_time"] = value
        result = model.normalize_entry(
            document,
            category="Test",
            filename="dates.yml",
            relpath="Test/dates.yml",
            meta_url=None,
            readme_status="missing",
        )
        assert result["normalized"]["issued_time"] == expected


def test_entry_ids_are_path_based_percent_encoded_and_unique(tmp_path):
    identifier = model.entry_id_for("Test", "San Diego, CA.yml")
    assert identifier == "apd:Test/San%20Diego%2C%20CA.yml"
    assert model.entry_id_for("Test", "source{d}-code~duplicates.yml") == (
        "apd:Test/source%7Bd%7D-code%7Eduplicates.yml"
    )
    assert model.entry_id_for("Test", "OpenNEURO") == "apd:Test/OpenNEURO"
    assert len(model.entry_id_for("Test", "x" * 200)) <= 512
    # Independent of title/homepage by construction; same path => same id.
    assert model.entry_id_for("Test", "full.yml") == model.entry_id_for("Test", "full.yml")


def test_metadata_hash_is_deterministic_and_path_independent(tmp_path):
    config = make_config(tmp_path)
    first = _normalize("full.yml", config, relpath="Test/full.yml", meta_url="https://a")
    second = _normalize("full.yml", config, relpath="Other/full.yml", meta_url="https://b")
    assert first["metadata_sha256"] == second["metadata_sha256"]

    document, warnings = _load("full.yml", config)
    document["title"] = "Changed title"
    changed = model.normalize_entry(
        document,
        category="Test",
        filename="full.yml",
        relpath="Test/full.yml",
        meta_url="https://a",
        readme_status="OK",
        warnings=warnings,
    )
    assert changed["metadata_sha256"] != first["metadata_sha256"]


def test_non_mapping_document_is_rejected(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(b"- one\n- two\n", config)
    assert error.value.code == "catalog_entry_invalid"
