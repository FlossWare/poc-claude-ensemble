"""PR 4 normalization: record envelope, projection, sharding and determinism."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from ingestion.datasets import adapters, normalize
from ingestion.datasets.config import Config

from ingestion.datasets.tests.helpers import make_catalog


def _row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "entry_id": "apd:Test/example.yml",
        "title": "Example Dataset",
        "category": "Test",
        "homepage": "https://example.org/data?token=SECRET#frag",
        "description": "A small example.",
        "keywords": ["alpha", "beta"],
        "tags": ["tag"],
        "publisher": [{"name": "Example Org", "web": "https://example.org"}],
        "sources": [{"name": "csv", "access_url": "https://data.example.org/table.csv"}],
        "metadata_sha256": "a" * 64,
        "license": {
            "dataset": {
                "policy": "index_content",
                "family": "cc0",
                "reason_code": None,
            }
        },
        "extra": {"custom": {"nested": True}},
        "catalog": {"commit": "c" * 40, "meta_url": "https://github.com/x/y"},
        "classes": {"format": ["csv"], "access": "open"},
    }
    row.update(overrides)
    return row


def _config(tmp_path: Path, **overrides) -> Config:
    root = make_catalog(tmp_path)
    base = Config.load(catalog_dir=root, data_root=tmp_path / "data", ledger_enabled=False)
    return base.replace(**overrides)


def _store(config: Config, name: str, data: bytes, *, media_type: str | None = None) -> dict[str, object]:
    digest = hashlib.sha256(data).hexdigest()
    blob = Path(config.data_root) / "cache" / "blobs" / digest[:2] / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    return {
        "status": "acquired",
        "files": [
            {
                "uri": f"https://data.example.org/{name}",
                "final_url": f"https://data.example.org/{name}?download=1",
                "member": None,
                "status": "acquired",
                "media_type": media_type,
                "sha256": digest,
                "bytes": len(data),
                "blob": str(blob.relative_to(config.data_root)),
            }
        ],
    }


def test_project_data_is_inert_and_deterministic():
    assert normalize.project_data("row", {"b": "2", "a": "1"}) == "b: 2\na: 1"
    assert normalize.project_data("text", {"line": 3, "text": "hello"}) == "hello"
    assert normalize.project_data("value", [1, 2]) == "- 1\n- 2"
    assert normalize.project_data("object", {"x": True}) == "x: true"


def test_metadata_text_includes_searchable_fields_and_sanitizes_urls():
    text = normalize.metadata_text(_row())
    assert "title: Example Dataset" in text
    assert "keywords" in text and "alpha" in text
    assert "license_family: cc0" in text
    assert "extra:" in text
    # Query strings never reach the indexed projection.
    assert "SECRET" not in text
    assert "example.org/data" in text


def test_metadata_text_is_stable_across_calls():
    first = normalize.metadata_text(_row())
    second = normalize.metadata_text(_row())
    assert first == second


def test_shard_texts_is_bounded_and_counts_records_per_shard():
    pieces = normalize.shard_texts(
        ["a" * 5, "b" * 5, "c" * 5], max_chars=12, max_bytes=10_000
    )
    assert [count for _, count in pieces] == [2, 1]
    assert all(len(content) <= 12 for content, _ in pieces)
    assert normalize.shard_texts(["x"], max_chars=10, max_bytes=10_000) == [("x", 1)]


def test_shard_texts_respects_the_byte_budget():
    # Each character is three UTF-8 bytes; four characters exceed ten bytes.
    pieces = normalize.shard_texts(["éééé"], max_chars=100, max_bytes=10)
    assert len(pieces) == 1
    assert len(pieces[0][0].encode("utf-8")) <= 10


def test_oversized_single_record_is_split_deterministically():
    pieces = normalize.shard_texts(["x" * 100], max_chars=30, max_bytes=10_000)
    assert "".join(content for content, _ in pieces) == "x" * 100
    assert all(len(content) <= 30 for content, _ in pieces)
    assert pieces[0][1] == 1


def test_content_source_id_is_stable_bounded_and_location_specific():
    first = normalize.content_source_id("apd:Test/a.csv", "https://x/y.csv", 0)
    assert first == normalize.content_source_id("apd:Test/a.csv", "https://x/y.csv", 0)
    assert first.endswith("/shard/0000")
    assert first != normalize.content_source_id("apd:Test/a.csv", "https://x/z.csv", 0)
    assert first != normalize.content_source_id("apd:Test/a.csv", "https://x/y.csv", 1)
    long_entry = "apd:Test/" + "a" * 600
    identifier = normalize.content_source_id(long_entry, "https://x/y.csv", 0)
    assert len(identifier) <= normalize.MAX_SOURCE_ID_CHARS
    assert identifier.startswith("apd:src/")


def test_revision_id_changes_with_every_input():
    base = dict(
        source_id="apd:Test/a.csv",
        content_sha256="1" * 64,
        normalized_sha256="2" * 64,
        adapter_version=2,
        license_policy="index_content",
        profiles={"chunking": "window-v1@1"},
        metadata_sha256="3" * 64,
    )
    first = normalize.revision_id_for(**base)
    assert first == normalize.revision_id_for(**base)
    assert first != normalize.revision_id_for(**{**base, "content_sha256": "4" * 64})
    assert first != normalize.revision_id_for(**{**base, "adapter_version": 3})
    assert first != normalize.revision_id_for(**{**base, "license_policy": "metadata_only_review"})
    assert first != normalize.revision_id_for(
        **{**base, "profiles": {"chunking": "window-v1@2"}}
    )


def test_metadata_record_has_the_canonical_envelope(tmp_path):
    config = _config(tmp_path)
    record, reason = normalize.metadata_record(
        _row(), commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert reason is None
    assert record is not None
    assert record.source_id == "apd:Test/example.yml"
    assert record.kind == "metadata"
    assert record.origin == "apd-core@" + "c" * 40
    assert record.locator == "https://example.org/data"
    assert record.content_sha256 == normalize.sha256_text(record.content)
    assert record.revision_id == record.revision_id
    assert len(record.revision_id) == 64
    artifact = record.to_artifact()
    assert "content" not in artifact
    assert artifact["metadata_sha256"] == "a" * 64


def test_metadata_record_is_capped_below_the_knowledge_limit(tmp_path):
    config = _config(tmp_path, max_shard_chars=200, max_shard_bytes=10_000)
    record, _ = normalize.metadata_record(
        _row(extra={"blob": "x" * 1000}),
        commit="c" * 40,
        profiles={"chunking": "window-v1@1"},
        config=config,
    )
    assert record is not None
    assert record.truncated is True
    assert len(record.content) <= 200


def test_normalize_entry_builds_one_shard_per_small_csv_file(tmp_path):
    config = _config(tmp_path)
    acquisition = _store(config, "table.csv", b"name,value\nalpha,1\nbeta,2\n", media_type="text/csv")
    result = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert result.metadata is not None
    assert len(result.records) == 1
    record = result.records[0]
    assert record.kind == "content"
    assert record.shard_index == 0 and record.shard_count == 1
    assert record.record_count == 2
    assert record.source_id.startswith("apd:Test/example.yml/src/")
    assert "name: alpha" in record.content and "value: 1" in record.content
    assert "name: beta" in record.content and "value: 2" in record.content
    assert result.records_total == 2
    assert result.files[0].status == "parsed"
    assert result.files[0].adapter == "csv"


def test_normalize_entry_shards_large_content_deterministically(tmp_path):
    config = _config(tmp_path, max_shard_chars=120, max_shard_bytes=10_000)
    rows = "\n".join(f"row{index},value{index}" for index in range(20)).encode()
    acquisition = _store(config, "table.csv", b"a,b\n" + rows + b"\n", media_type="text/csv")
    first = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    second = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert len(first.records) > 1
    assert [record.content_sha256 for record in first.records] == [
        record.content_sha256 for record in second.records
    ]
    assert {record.shard_count for record in first.records} == {len(first.records)}
    assert [record.shard_index for record in first.records] == list(range(len(first.records)))


def test_normalize_entry_indexes_archive_members(tmp_path):
    config = _config(tmp_path)
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("one.csv", "a,b\n1,2\n")
        archive.writestr("two.csv", "a,b\n3,4\n")
    acquisition = _store(config, "bundle.zip", buffer.getvalue(), media_type="application/zip")
    result = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert len(result.records) == 2
    assert {record.record_count for record in result.records} == {1}
    assert {item.member for item in result.files} == {"one.csv", "two.csv"}
    assert all(record.source_sha256 for record in result.records)


def test_normalize_entry_skips_content_for_metadata_only_licenses(tmp_path):
    config = _config(tmp_path)
    acquisition = _store(config, "table.csv", b"a\n1\n2\n")
    row = _row(
        license={
            "dataset": {
                "policy": "metadata_only_review",
                "family": "unknown",
                "reason_code": "license_unknown_metadata_only",
            }
        }
    )
    result = normalize.normalize_entry(
        row, acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert result.metadata is not None
    assert result.records == []
    assert result.content_reason == "license_unknown_metadata_only"


def test_normalize_entry_reports_acquisition_reason(tmp_path):
    config = _config(tmp_path)
    acquisition = {"status": "skipped", "reason_code": "license_restricted", "files": []}
    result = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert result.records == []
    assert result.content_reason == "license_restricted"


def test_normalize_entry_fails_closed_on_a_parse_error(tmp_path):
    config = _config(tmp_path)
    acquisition = _store(config, "broken.csv", b'a,b\n"unbalanced,2\n', media_type="text/csv")
    result = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert result.records == []
    assert result.files[0].status == "failed"
    assert result.files[0].reason_code == "parse_error"
    assert result.content_reason == "parse_error"


def test_normalize_entry_reports_pdf_as_adapter_unavailable(tmp_path):
    config = _config(tmp_path)
    acquisition = _store(config, "report.pdf", b"%PDF-1.4\n", media_type="application/pdf")
    result = normalize.normalize_entry(
        _row(), acquisition, commit="c" * 40, profiles={"chunking": "window-v1@1"}, config=config
    )
    assert result.records == []
    assert result.files[0].reason_code == "adapter_unavailable"
    assert result.content_reason == "adapter_unavailable"


def test_iter_records_streams_every_record_not_just_the_sample(tmp_path):
    config = _config(tmp_path, adapter_max_sample_records=2)
    path = tmp_path / "many.csv"
    path.write_text("a\n" + "\n".join(str(index) for index in range(50)) + "\n", encoding="utf-8")
    parsed = adapters.parse_file(path, name="many.csv", media_type=None, config=config)
    assert parsed.record_count == 50
    assert len(parsed.sample) == 2
    summary = adapters.StreamSummary()
    records = list(
        adapters.iter_records(path, name="many.csv", media_type=None, config=config, summary=summary)
    )
    assert len(records) == 50
    assert summary.count == 50
    assert [record.index for record in records] == list(range(50))
