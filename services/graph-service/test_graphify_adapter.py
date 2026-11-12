#!/usr/bin/env python3
"""Tests for the Graphify adapter and atomic graph batch ingestion."""

import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from graph_service import GraphStore
from graphify_adapter import GraphifyInputError, normalize_extraction, extract_local, _safe_target


class GraphifyAdapterTests(unittest.TestCase):
    def test_normalize_preserves_provenance_and_confidence(self):
        raw = {
            "nodes": [
                {"id": "file_a", "label": "A", "type": "file", "source_file": "src/a.py"},
                {"id": "symbol_b", "label": "B", "type": "function", "source_file": "src/b.py",
                 "source_location": "L12"},
            ],
            "edges": [
                {"source": "file_a", "target": "symbol_b", "relation": "imports",
                 "confidence": "INFERRED", "confidence_score": 0.75,
                 "source_file": "src/a.py", "source_location": "L4"},
            ],
        }
        nodes, edges = normalize_extraction(raw)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["source"], "graphify:file_a")
        self.assertEqual(edges[0]["target"], "graphify:symbol_b")
        self.assertEqual(edges[0]["type"], "imports")
        self.assertEqual(edges[0]["properties"]["confidence"], "INFERRED")
        self.assertEqual(edges[0]["properties"]["source_location"], "L4")
        self.assertEqual(nodes[1]["properties"]["source_file"], "src/b.py")

    def test_normalize_ignores_malformed_and_dangling_records(self):
        nodes, edges = normalize_extraction({
            "nodes": [{"id": "valid", "label": "Valid"}, {"label": "missing id"}],
            "edges": [{"source": "valid", "target": "absent", "relation": "calls"}],
        })
        self.assertEqual(len(nodes), 1)
        self.assertEqual(edges, [])

    def test_target_must_stay_inside_configured_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            outside = Path(tmp) / "outside.py"
            outside.write_text("def outside(): pass\n", encoding="utf-8")
            inside = root / "inside.py"
            inside.write_text("def inside(): pass\n", encoding="utf-8")
            allowed, target = _safe_target(root, "inside.py")
            self.assertEqual(allowed, root.resolve())
            self.assertEqual(target, inside.resolve())
            with self.assertRaises(GraphifyInputError):
                _safe_target(root, outside)

    def test_extract_local_uses_graphify_structural_api_without_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.py"
            source.write_text("def sample(): pass\n", encoding="utf-8")
            calls = {}
            graphify_package = types.ModuleType("graphify")
            graphify_package.__path__ = []
            graphify_extract = types.ModuleType("graphify.extract")

            def collect_files(target, *, root):
                calls["target"] = target
                calls["root"] = root
                return [source]

            def extract(files, **kwargs):
                calls["extract_kwargs"] = kwargs
                return {"nodes": [{"id": "sample", "label": "sample"}],
                        "edges": [], "failed_sources": [], "skipped_sources": []}

            graphify_extract.collect_files = collect_files
            graphify_extract.extract = extract
            with patch.dict(sys.modules, {
                "graphify": graphify_package,
                "graphify.extract": graphify_extract,
            }):
                result = extract_local(root, "sample.py", cache_root=root / "cache")

            self.assertEqual(result["files"], 1)
            self.assertEqual(len(result["nodes"]), 1)
            self.assertFalse(calls["extract_kwargs"]["parallel"])
            self.assertEqual(calls["extract_kwargs"]["cache_root"], root / "cache")
            self.assertEqual(calls["extract_kwargs"]["root"], root.resolve())

    def test_batch_ingestion_is_idempotent_and_persists_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GraphStore(Path(tmp) / "graph.json")
            nodes = [
                {"id": "graphify:a", "type": "function", "properties": {"label": "a"}},
                {"id": "graphify:b", "type": "function", "properties": {"label": "b"}},
            ]
            edges = [{
                "source": "graphify:a", "target": "graphify:b", "type": "calls",
                "properties": {"confidence": "EXTRACTED"},
            }]
            first = store.add_batch(nodes, edges)
            self.assertEqual(first, {"added_nodes": 2, "added_edges": 1})
            second = store.add_batch(nodes, edges)
            self.assertEqual(second, {"added_nodes": 0, "added_edges": 0})
            reloaded = GraphStore(Path(tmp) / "graph.json")
            self.assertIsNotNone(reloaded.get_node("graphify:a"))
            self.assertEqual(len(reloaded.query_edges(relationship="calls")), 1)

    def test_batch_validation_failure_does_not_mutate_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GraphStore(Path(tmp) / "graph.json")
            with self.assertRaises(KeyError):
                store.add_batch(
                    [{"id": "graphify:a", "type": "function", "properties": {}}],
                    [{"source": "graphify:a", "target": "missing", "type": "calls", "properties": {}}],
                )
            self.assertIsNone(store.get_node("graphify:a"))
            self.assertFalse(store.path.exists())


if __name__ == "__main__":
    unittest.main()
