#!/usr/bin/env python3
"""Diagnose a ChromaDB vector store.

The POC keeps the on-disk store under a storage-format-version directory:

    ~/.FlossWare/poc-claude-ensemble/data/chromadb/1.5.9/

Pointing ``chromadb.PersistentClient`` at the *parent* (``.../data/chromadb``)
silently creates an empty ``chroma.sqlite3`` there and reports zero
collections. That is the most common failure mode and the one this tool calls
out explicitly as a path trap.

Checks performed:

  * store present (never creates a directory just to inspect it)
  * installed ``chromadb`` version vs the versioned store directory
  * ``migrations`` schema version and client open (real compatibility test)
  * collection count, per-collection record count, model/dimension agreement
  * query round-trip: read a stored vector, query it back, require a self-hit
  * decoy / path-trap detection (empty parent with a populated version child)
  * orphaned HNSW segment directories no longer referenced by ``segments``

Usage::

    python3 tools/chromadb_doctor.py
    python3 tools/chromadb_doctor.py --store /path/to/1.5.9
    python3 tools/chromadb_doctor.py --json
    python3 tools/chromadb_doctor.py --expected-model nomic-ai/nomic-embed-text-v1.5 \\
        --expected-dimensions 768

Exit codes: 0 healthy, 1 warnings, 2 errors.

Read-only against the store: it opens the client to verify compatibility and
run a query, but never upserts, deletes, or imports documents.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Sequence

# Never emit telemetry from a diagnostic run.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402

DEFAULT_STORE = "~/.FlossWare/poc-claude-ensemble/data/chromadb/1.5.9"
STORE_ENV = "POC_CLAUDE_ENSEMBLE_VECTOR_STORE"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Severity ordering used to fold check results into one status.
_RANK = {"ok": 0, "warn": 1, "error": 2}


def default_store() -> str:
    return os.path.expanduser(os.environ.get(STORE_ENV, DEFAULT_STORE))


def _worst(statuses: Sequence[str]) -> str:
    worst = "ok"
    for status in statuses:
        if _RANK.get(status, 0) > _RANK[worst]:
            worst = status
    return worst


def _db_path(store: str) -> str:
    return os.path.join(store, "chroma.sqlite3")


def _open_ro(db_path: str) -> Optional[sqlite3.Connection]:
    if not os.path.isfile(db_path):
        return None
    try:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def schema_version(db_path: str) -> Optional[int]:
    """Highest applied ``sysdb`` migration, or None if unreadable."""
    conn = _open_ro(db_path)
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT max(version) FROM migrations").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def referenced_segments(db_path: str) -> Optional[set]:
    conn = _open_ro(db_path)
    if conn is None:
        return None
    try:
        return {row[0] for row in conn.execute("SELECT id FROM segments")}
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def orphaned_segments(store: str, db_path: str) -> List[Dict[str, Any]]:
    """HNSW segment directories on disk that no ``segments`` row references."""
    referenced = referenced_segments(db_path)
    if referenced is None:
        return []
    orphans: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(store))
    except OSError:
        return []
    for name in entries:
        child = os.path.join(store, name)
        if not os.path.isdir(child) or name in referenced:
            continue
        # A HNSW segment directory always carries a header.
        if not os.path.isfile(os.path.join(child, "header.bin")):
            continue
        orphans.append({"id": name, "bytes": _dir_size(child)})
    return orphans


def _version_dirs(path: str) -> List[str]:
    """Child directories whose name is a version and that hold a store."""
    found: List[str] = []
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return found
    for name in entries:
        child = os.path.join(path, name)
        if _VERSION_RE.match(name) and os.path.isfile(_db_path(child)):
            found.append(child)
    return found


def detect_path_trap(store: str, collections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Report a populated version child when the inspected store is empty."""
    if collections:
        return None
    for child in _version_dirs(store):
        try:
            client = chromadb.PersistentClient(path=child)
            names = sorted(c.name for c in client.list_collections())
        except Exception:  # noqa: BLE001 - diagnosis must not abort on one bad child
            continue
        if names:
            return {"store": child, "collections": names}
    return None


def _metadata_value(metadata: Optional[Dict[str, Any]], key: str) -> Any:
    return (metadata or {}).get(key)


def query_roundtrip(collection) -> Dict[str, Any]:
    """Read one stored vector and require the collection to return it first."""
    try:
        rec = collection.get(limit=1, include=["documents", "embeddings"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"get failed: {type(exc).__name__}: {exc}"}
    ids = rec.get("ids") or []
    if not ids:
        return {"status": "skipped", "detail": "collection is empty"}
    sample_id = ids[0]
    embeddings = rec.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return {"status": "error", "detail": "stored vector not returned for sample record"}
    vector = embeddings[0]
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    try:
        result = collection.query(query_embeddings=[vector], n_results=1)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"query failed: {type(exc).__name__}: {exc}"}
    top_ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    top_id = top_ids[0] if top_ids else None
    distance = float(distances[0]) if distances else None
    hit = top_id == sample_id
    detail = "self-query returned the source record" if hit else "self-query did not return the source record"
    return {
        "status": "ok" if hit else "error",
        "sample_id": sample_id,
        "top_id": top_id,
        "distance": distance,
        "detail": detail,
    }


def inspect(
    store: str,
    expected_model: Optional[str] = None,
    expected_dimensions: Optional[int] = None,
) -> Dict[str, Any]:
    """Run every check against ``store`` and return a report dict."""
    store = os.path.abspath(os.path.expanduser(store))
    db_path = _db_path(store)
    problems: List[str] = []
    report: Dict[str, Any] = {
        "store": store,
        "chromadb_version": chromadb.__version__,
        "storage_version": os.path.basename(os.path.normpath(store)),
        "schema_version": None,
        "version_compatible": None,
        "collections": [],
        "total_records": 0,
        "orphaned_segments": [],
        "path_trap": None,
        "checked_path": True,
        "problems": problems,
    }

    installed = chromadb.__version__
    storage_version = report["storage_version"]
    if _VERSION_RE.match(storage_version):
        report["version_compatible"] = storage_version == installed
        if not report["version_compatible"]:
            problems.append(
                f"store directory is {storage_version} but installed chromadb is {installed}"
            )
    else:
        # Unversioned path (often the decoy parent). Compatibility is judged by
        # whether the client can open and list.
        report["version_compatible"] = None

    if not os.path.isdir(store):
        problems.append(f"store directory does not exist: {store}")
        report["status"] = "error"
        report["checked_path"] = False
        return report
    if not os.path.isfile(db_path):
        problems.append(f"no chroma.sqlite3 in {store} (empty or wrong path)")
        report["status"] = "error"
        report["checked_path"] = False
        return report

    report["schema_version"] = schema_version(db_path)

    # Opening the client is the real compatibility test: an incompatible
    # on-disk schema fails here even when directory names happen to match.
    try:
        client = chromadb.PersistentClient(path=store)
        listed = list(client.list_collections())
    except Exception as exc:  # noqa: BLE001
        problems.append(f"client could not open store: {type(exc).__name__}: {exc}")
        report["version_compatible"] = False
        report["status"] = "error"
        return report

    if report["version_compatible"] is None:
        report["version_compatible"] = True

    models, dimensions = set(), set()
    for col in sorted(listed, key=lambda c: c.name):
        metadata = col.metadata or {}
        entry: Dict[str, Any] = {"name": col.name, "metadata": metadata}
        try:
            entry["count"] = col.count()
        except Exception as exc:  # noqa: BLE001
            entry["count"] = None
            entry["count_error"] = f"{type(exc).__name__}: {exc}"
            problems.append(f"collection {col.name}: count failed: {exc}")
        entry["query_test"] = query_roundtrip(col)
        if entry["query_test"]["status"] == "error":
            problems.append(f"collection {col.name}: {entry['query_test']['detail']}")
        model = _metadata_value(metadata, "model")
        dims = _metadata_value(metadata, "dimensions")
        if model is not None:
            models.add(model)
            if expected_model is not None and model != expected_model:
                problems.append(f"collection {col.name}: model {model!r} != expected {expected_model!r}")
        if dims is not None:
            dimensions.add(dims)
            if expected_dimensions is not None and dims != expected_dimensions:
                problems.append(
                    f"collection {col.name}: dimensions {dims!r} != expected {expected_dimensions!r}"
                )
        report["collections"].append(entry)
        if isinstance(entry["count"], int):
            report["total_records"] += entry["count"]

    report["models"] = sorted(models)
    report["dimensions"] = sorted(dimensions)
    if len(models) > 1:
        problems.append(f"collections disagree on embedding model: {sorted(models)}")
    if len(dimensions) > 1:
        problems.append(f"collections disagree on dimensions: {sorted(dimensions)}")

    report["orphaned_segments"] = orphaned_segments(store, db_path)
    if report["orphaned_segments"]:
        total = sum(o["bytes"] for o in report["orphaned_segments"])
        problems.append(
            f"{len(report['orphaned_segments'])} orphaned segment dir(s), "
            f"{total} bytes not referenced by any collection"
        )

    report["path_trap"] = detect_path_trap(store, report["collections"])
    if report["path_trap"]:
        problems.append(
            "pointed at an empty parent store; populated version directory is "
            f"{report['path_trap']['store']}"
        )
    elif not report["collections"]:
        problems.append("store is readable but contains no collections")

    statuses = [c["query_test"]["status"] for c in report["collections"]]
    statuses = ["ok" if s == "skipped" else s for s in statuses]
    report["status"] = _worst(statuses + (["warn"] if problems else []))
    if report["status"] == "ok" and problems:
        report["status"] = "warn"
    return report


def _human(report: Dict[str, Any]) -> str:
    lines = [
        f"store:            {report['store']}",
        f"chromadb:         {report['chromadb_version']}",
        f"storage version:  {report['storage_version']}",
        f"schema migration: {report['schema_version']}",
        f"compatible:       {report['version_compatible']}",
        f"status:           {report['status'].upper()}",
        f"collections:      {len(report['collections'])}",
        f"records:          {report['total_records']}",
    ]
    if report.get("models"):
        lines.append(f"models:           {', '.join(map(str, report['models']))}")
    if report.get("dimensions"):
        lines.append(f"dimensions:       {', '.join(map(str, report['dimensions']))}")
    for col in report["collections"]:
        qt = col["query_test"]
        dist = qt.get("distance")
        dist_text = f" distance={dist:.6f}" if isinstance(dist, float) else ""
        lines.append(
            f"  - {col['name']}: count={col['count']} query={qt['status']}{dist_text}"
        )
    for orphan in report["orphaned_segments"]:
        lines.append(f"  ! orphaned segment {orphan['id']} ({orphan['bytes']} bytes)")
    if report.get("path_trap"):
        lines.append(
            f"  ! path trap: use {report['path_trap']['store']} "
            f"(collections: {', '.join(report['path_trap']['collections'])})"
        )
    if report["problems"]:
        lines.append("problems:")
        lines.extend(f"  - {p}" for p in report["problems"])
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ChromaDB vector store health.")
    parser.add_argument("--store", default=default_store(),
                        help=f"vector store directory (default: ${STORE_ENV} or {DEFAULT_STORE})")
    parser.add_argument("--expected-model", default=None,
                        help="flag collections whose stored model differs")
    parser.add_argument("--expected-dimensions", type=int, default=None,
                        help="flag collections whose stored dimensions differ")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit the report as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = inspect(args.store, args.expected_model, args.expected_dimensions)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(_human(report))
    return {"ok": 0, "warn": 1, "error": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
