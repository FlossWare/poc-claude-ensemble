"""Adapter parsing: shapes, malformed input and every configured limit."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import adapters

from ingestion.datasets.tests.helpers import make_config


def _write(tmp_path, name: str, data: bytes | str):
    path = tmp_path / name
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)
    return path


def _parse(tmp_path, name, data, config=None):
    config = config or make_config(tmp_path)
    path = _write(tmp_path, name, data)
    return adapters.parse_file(path, name=name, media_type=None, config=config)


def test_csv_rows_are_mapped_to_header_fields(tmp_path):
    result = _parse(tmp_path, "table.csv", "name,value\nalpha,1\nbeta,2\n")
    assert result.status == "parsed"
    assert result.adapter == "csv"
    assert result.record_count == 2
    assert result.fields == ("name", "value")
    assert result.sample[0].data == {"name": "alpha", "value": "1"}
    assert result.sample[0].kind == "row"


def test_csv_cells_stay_inert_text(tmp_path):
    result = _parse(tmp_path, "formula.csv", 'a,b\n=1+1,"@SUM(A1)"\n')
    assert result.sample[0].data == {"a": "=1+1", "b": "@SUM(A1)"}


def test_tsv_uses_a_tab_delimiter(tmp_path):
    result = _parse(tmp_path, "table.tsv", "a\tb\n1\t2\n")
    assert result.adapter == "tsv"
    assert result.sample[0].data == {"a": "1", "b": "2"}


def test_csv_without_data_rows_is_empty_content(tmp_path):
    result = _parse(tmp_path, "only-header.csv", "a,b\n")
    assert result.status == "skipped"
    assert result.reason_code == "empty_content"
    assert result.fields == ("a", "b")


def test_malformed_csv_is_a_parse_error(tmp_path):
    data = 'a,b\n"unbalanced,2\n'
    result = _parse(tmp_path, "broken.csv", data)
    assert result.status == "failed"
    assert result.reason_code == "parse_error"


def test_csv_duplicate_and_empty_headers_are_renamed(tmp_path):
    result = _parse(tmp_path, "dup.csv", "a,,a\n1,2,3\n")
    assert result.fields == ("a", "col_2", "a#2")
    assert result.sample[0].data == {"a": "1", "col_2": "2", "a#2": "3"}
    assert "duplicate_header:a" in result.warnings


def test_csv_short_and_long_rows_are_padded_and_reported(tmp_path):
    result = _parse(tmp_path, "ragged.csv", "a,b\n1\n1,2,3\n")
    assert result.record_count == 2
    assert result.sample[0].data == {"a": "1", "b": ""}
    assert result.sample[1].data == {"a": "1", "b": "2"}
    assert "missing_fields:1" in result.warnings
    assert "extra_fields:1" in result.warnings


def test_csv_field_and_cell_limits(tmp_path):
    config = make_config(tmp_path).replace(max_fields=2)
    result = _parse(tmp_path, "wide.csv", "a,b,c\n1,2,3\n", config=config)
    assert result.reason_code == "limits_exceeded"

    config = make_config(tmp_path).replace(max_cell_chars=3)
    result = _parse(tmp_path, "cells.csv", "a\nabcdef\n", config=config)
    assert result.reason_code == "limits_exceeded"


def test_csv_row_cap_truncates_instead_of_failing(tmp_path):
    config = make_config(tmp_path).replace(max_records_per_file=2)
    result = _parse(tmp_path, "long.csv", "a\n1\n2\n3\n4\n", config=config)
    assert result.status == "parsed"
    assert result.record_count == 2
    assert result.truncated is True


def test_json_object_is_one_record(tmp_path):
    result = _parse(tmp_path, "doc.json", json.dumps({"a": 1, "b": [1, 2]}))
    assert result.adapter == "json"
    assert result.record_count == 1
    assert result.sample[0].kind == "object"
    assert result.fields == ("a", "b")


def test_json_array_is_one_record_per_element(tmp_path):
    result = _parse(tmp_path, "rows.json", json.dumps([{"a": 1}, {"a": 2}]))
    assert result.record_count == 2
    assert result.sample[1].data == {"a": 2}


def test_json_scalar_is_one_record(tmp_path):
    result = _parse(tmp_path, "scalar.json", "42")
    assert result.record_count == 1
    assert result.sample[0].kind == "value"


def test_malformed_json_is_a_parse_error(tmp_path):
    result = _parse(tmp_path, "broken.json", '{"a": ')
    assert result.status == "failed"
    assert result.reason_code == "parse_error"


def test_json_depth_is_capped(tmp_path):
    config = make_config(tmp_path).replace(max_json_depth=4)
    payload = json.dumps({"a": {"b": {"c": {"d": {"e": 1}}}}})
    result = _parse(tmp_path, "deep.json", payload, config=config)
    assert result.reason_code == "limits_exceeded"


def test_json_object_field_cap(tmp_path):
    config = make_config(tmp_path).replace(max_fields=2)
    result = _parse(tmp_path, "wide.json", json.dumps({"a": 1, "b": 2, "c": 3}), config=config)
    assert result.reason_code == "limits_exceeded"


def test_jsonl_is_one_record_per_line(tmp_path):
    data = '{"a": 1}\n\n{"a": 2}\n'
    result = _parse(tmp_path, "rows.jsonl", data)
    assert result.adapter == "jsonl"
    assert result.record_count == 2
    assert result.fields == ("a",)


def test_jsonl_tolerates_a_few_malformed_lines(tmp_path):
    data = '{"a": 1}\nnot json\n{"a": 2}\n'
    result = _parse(tmp_path, "rows.ndjson", data)
    assert result.status == "parsed"
    assert result.record_count == 2
    assert "malformed_lines:1" in result.warnings


def test_jsonl_rejects_many_malformed_lines(tmp_path):
    data = "".join("junk\n" for _ in range(12))
    result = _parse(tmp_path, "rows.ndjson", data)
    assert result.status == "failed"
    assert result.reason_code == "parse_error"


def test_jsonl_line_length_limit(tmp_path):
    config = make_config(tmp_path).replace(max_line_chars=10)
    result = _parse(tmp_path, "rows.jsonl", '{"a": "aaaaaaaaaaaa"}\n', config=config)
    assert result.reason_code == "limits_exceeded"


def test_xml_records_are_child_elements(tmp_path):
    data = "<rows><row><a>1</a><b>2</b></row><row><a>3</a></row></rows>"
    result = _parse(tmp_path, "rows.xml", data)
    assert result.adapter == "xml"
    assert result.record_count == 2
    assert result.sample[0].data == {"tag": "row", "a": "1", "b": "2"}


def test_xml_doctype_and_entity_are_refused(tmp_path):
    billion = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
        "<lolz>&lol;</lolz>"
    )
    result = _parse(tmp_path, "bomb.xml", billion)
    assert result.status == "failed"
    assert result.reason_code == "xml_unsafe"


def test_xml_depth_is_capped(tmp_path):
    config = make_config(tmp_path).replace(max_xml_depth=3)
    data = "<a><b><c><d><e/></d></c></b></a>"
    result = _parse(tmp_path, "deep.xml", data, config=config)
    assert result.reason_code == "limits_exceeded"


def test_malformed_xml_is_a_parse_error(tmp_path):
    result = _parse(tmp_path, "broken.xml", "<a><b></a>")
    assert result.reason_code == "parse_error"


def test_text_lines_become_records(tmp_path):
    result = _parse(tmp_path, "notes.txt", "one\n\ntwo\n")
    assert result.adapter == "text"
    assert result.record_count == 2
    assert result.sample[0].data == {"line": 1, "text": "one"}


def test_markdown_uses_the_text_adapter(tmp_path):
    result = _parse(tmp_path, "README.md", "# Title\ntext\n")
    assert result.adapter == "markdown"
    assert result.record_count == 2


def test_unsupported_format_is_explicit(tmp_path):
    result = _parse(tmp_path, "page.html", "<html><body>x</body></html>")
    assert result.status == "skipped"
    assert result.reason_code == "unsupported_format"


def test_pdf_is_reported_as_a_disabled_adapter(tmp_path):
    result = _parse(tmp_path, "doc.pdf", b"%PDF-1.7\n%%EOF\n")
    assert result.status == "skipped"
    assert result.reason_code == "adapter_unavailable"
    assert result.media_kind == "pdf"


def test_parquet_is_unavailable_without_pyarrow(tmp_path):
    config = make_config(tmp_path).replace(parquet_enabled=False)
    result = _parse(tmp_path, "data.parquet", b"PAR1", config=config)
    assert result.status == "skipped"
    assert result.reason_code == "adapter_unavailable"


def test_parquet_round_trip_when_pyarrow_is_installed(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    config = make_config(tmp_path).replace(parquet_enabled=True)
    table = pyarrow.table({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "data.parquet"
    parquet.write_table(table, path)
    result = adapters.parse_file(path, name="data.parquet", media_type=None, config=config)
    assert result.status == "parsed"
    assert result.fields == ("a", "b")
    assert result.record_count == 2


def test_latin1_fallback_is_recorded(tmp_path):
    path = _write(tmp_path, "latin.csv", b"name\nJos\xe9\n")
    config = make_config(tmp_path)
    result = adapters.parse_file(path, name="latin.csv", media_type=None, config=config)
    assert result.encoding == "latin-1"
    assert "encoding_fallback:latin-1" in result.warnings
    assert result.sample[0].data == {"name": "José"}


def test_strict_encoding_failure_is_an_encoding_error(tmp_path):
    path = _write(tmp_path, "latin.csv", b"name\nJos\xe9\n")
    config = make_config(tmp_path).replace(adapter_encoding_fallback=False)
    result = adapters.parse_file(path, name="latin.csv", media_type=None, config=config)
    assert result.status == "failed"
    assert result.reason_code == "encoding_error"


def test_kind_detection_prefers_extension_then_media_type():
    assert adapters.normalize_kind("x.CSV", "text/plain", None) == "csv"
    assert adapters.normalize_kind("x", "application/json", None) == "json"
    assert adapters.normalize_kind("x", None, b'{"a": 1}') == "json"
    assert adapters.normalize_kind("x", None, b"<?xml version='1.0'?>") == "xml"
    assert adapters.normalize_kind("x", None, b"a,b\n") is None


def test_sample_is_bounded_by_configuration(tmp_path):
    config = make_config(tmp_path).replace(adapter_max_sample_records=1)
    result = _parse(tmp_path, "many.csv", "a\n1\n2\n3\n", config=config)
    assert result.record_count == 3
    assert len(result.sample) == 1


def test_artifact_is_json_serializable(tmp_path):
    result = _parse(tmp_path, "table.csv", "a,b\n1,2\n")
    payload = result.to_artifact()
    assert payload["adapter"] == "csv"
    assert payload["record_count"] == 1
    json.dumps(payload)
