#!/usr/bin/env python3
"""Tests for tools/chromadb_doctor.py using real fixture stores.

Every fixture is a genuine ChromaDB store built in a temporary directory, so
collection counts, metadata, and the query round-trip exercise the real client
rather than a mock.

Run with pytest, or standalone: python3 tools/test_chromadb_doctor.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import chromadb  # noqa: E402

import chromadb_doctor as doctor  # noqa: E402

DIM = 8
MODEL = "fixture-model"


def _vector(seed: float):
    return [seed + i / 100.0 for i in range(DIM)]


def _metadata(model=MODEL, dimensions=DIM):
    return {"hnsw:space": "cosine", "model": model, "dimensions": dimensions}


def make_store(path, collections=None):
    """Create a real store; ``collections`` maps name -> list of ids to upsert."""
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    for name, ids in (collections or {}).items():
        col = client.create_collection(name, metadata=_metadata())
        if ids:
            col.upsert(
                ids=ids,
                embeddings=[_vector(i) for i in range(len(ids))],
                documents=[f"document {i}" for i in range(len(ids))],
            )
    return path


def by_name(report):
    return {c["name"]: c for c in report["collections"]}


def test_healthy_store_reports_counts_and_passing_query(tmp_path):
    store = make_store(tmp_path / "1.5.9", {"knowledge_chunks": ["a", "b", "c"],
                                            "transcript_text": ["x", "y"]})

    report = doctor.inspect(str(store))

    assert report["status"] == "ok"
    assert report["version_compatible"] is True
    assert report["total_records"] == 5
    cols = by_name(report)
    assert cols["knowledge_chunks"]["count"] == 3
    assert cols["transcript_text"]["count"] == 2
    for col in report["collections"]:
        assert col["query_test"]["status"] == "ok"
        assert col["query_test"]["top_id"] == col["query_test"]["sample_id"]
        assert col["query_test"]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert report["problems"] == []


def test_empty_store_is_a_warning(tmp_path):
    store = make_store(tmp_path / "1.5.9", {})

    report = doctor.inspect(str(store))

    assert report["status"] == "warn"
    assert report["collections"] == []
    assert any("no collections" in p for p in report["problems"])


def test_empty_collection_skips_query_but_stays_healthy(tmp_path):
    store = make_store(tmp_path / "1.5.9", {"knowledge_chunks": []})

    report = doctor.inspect(str(store))

    query = by_name(report)["knowledge_chunks"]["query_test"]
    assert query["status"] == "skipped"
    assert report["status"] == "ok"


def test_missing_store_errors_without_creating_it(tmp_path):
    missing = tmp_path / "nope"

    report = doctor.inspect(str(missing))

    assert report["status"] == "error"
    assert report["checked_path"] is False
    assert not missing.exists(), "doctor must not create a store to inspect it"


def test_path_trap_detects_populated_version_child(tmp_path):
    parent = tmp_path / "chromadb"
    # The decoy: an empty store created by pointing a client at the parent.
    chromadb.PersistentClient(path=str(parent))
    populated = make_store(parent / "1.5.9", {"knowledge_chunks": ["a"]})

    report = doctor.inspect(str(parent))

    assert report["collections"] == []
    assert report["path_trap"] is not None
    assert report["path_trap"]["store"] == os.path.abspath(str(populated))
    assert report["path_trap"]["collections"] == ["knowledge_chunks"]
    assert report["status"] == "warn"
    assert any("version directory" in p for p in report["problems"])


def test_version_mismatch_is_flagged(tmp_path):
    store = make_store(tmp_path / "9.9.9", {"knowledge_chunks": ["a"]})

    report = doctor.inspect(str(store))

    assert report["storage_version"] == "9.9.9"
    assert report["version_compatible"] is False
    assert report["status"] == "warn"
    assert any("installed chromadb" in p for p in report["problems"])


def test_orphaned_segment_directory_is_reported(tmp_path):
    store = make_store(tmp_path / "1.5.9", {"knowledge_chunks": ["a"]})
    orphan = Path(store) / "deadbeef-0000-0000-0000-000000000000"
    orphan.mkdir()
    (orphan / "header.bin").write_bytes(b"orphan")
    (orphan / "data_level0.bin").write_bytes(b"x" * 32)

    report = doctor.inspect(str(store))

    ids = [o["id"] for o in report["orphaned_segments"]]
    assert ids == ["deadbeef-0000-0000-0000-000000000000"]
    assert report["status"] == "warn"
    assert any("orphaned segment" in p for p in report["problems"])


def test_expected_model_mismatch_is_flagged(tmp_path):
    store = make_store(tmp_path / "1.5.9", {"knowledge_chunks": ["a"]})

    report = doctor.inspect(str(store), expected_model="some-other-model",
                            expected_dimensions=DIM)

    assert report["status"] == "warn"
    assert any("some-other-model" in p for p in report["problems"])


def test_cli_json_output_and_exit_code(tmp_path, capsys):
    store = make_store(tmp_path / "1.5.9", {"knowledge_chunks": ["a"]})

    code = doctor.main(["--store", str(store), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["total_records"] == 1


def test_cli_error_exit_code_for_missing_store(tmp_path):
    code = doctor.main(["--store", str(tmp_path / "absent")])
    assert code == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
