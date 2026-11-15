import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import awesome_datasets as datasets


class AwesomeDatasetCatalogTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is installed by the full POC requirements")
    def test_parses_sparse_metadata_and_source_access_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "core" / "Biology" / "penguins.yml"
            path.parent.mkdir(parents=True)
            path.write_text(
                "title: Penguins\ncategory: Biology\ndescription: Bird data\n"
                "keywords: [birds, ecology]\nlicense: CC0\n"
                "sources:\n  - name: download\n    access_url: https://data.example/penguins.csv\n",
                encoding="utf-8",
            )
            record = datasets.parse_metadata(path, root)
        self.assertEqual(record.title, "Penguins")
        self.assertEqual(record.category, "Biology")
        self.assertEqual(record.source_urls, ["https://data.example/penguins.csv"])
        self.assertEqual(record.keywords, ["birds", "ecology"])
        self.assertEqual(len(record.metadata_sha256), 64)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is installed by the full POC requirements")
    def test_discovery_accounts_for_malformed_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core" / "Data").mkdir(parents=True)
            (root / "core" / "Data" / "good.yml").write_text("title: Good\nlicense: CC0\n")
            (root / "core" / "Data" / "bad.yaml").write_text("title: [broken")
            records, errors = datasets.discover(root)
        self.assertEqual([record.title for record in records], ["Good"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["status"], "metadata_error")

    def test_license_allow_list_is_conservative(self):
        self.assertTrue(datasets.license_is_acquirable("CC0"))
        self.assertTrue(datasets.license_is_acquirable("CC-BY-4.0"))
        self.assertTrue(datasets.license_is_acquirable("ODbL"))
        self.assertFalse(datasets.license_is_acquirable(None))
        self.assertFalse(datasets.license_is_acquirable("Research use only"))
        self.assertFalse(datasets.license_is_acquirable("CC-BY-NC-4.0"))

    def test_homepage_is_not_assumed_to_be_a_direct_download(self):
        self.assertFalse(datasets.is_direct_asset("https://example.org/dataset/"))
        self.assertTrue(datasets.is_direct_asset("https://example.org/files/data.csv"))

    def test_landing_page_resolves_explicit_asset_links(self):
        class Headers(dict):
            def get_content_type(self): return self.get("Content-Type", "application/octet-stream")

        class Response:
            headers = Headers({"Content-Type": "text/html"})
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, _limit):
                return b'<a href="/files/table.csv">CSV</a><a href="/about">About</a>'

        record = datasets.DatasetRecord(
            dataset_id="id", title="Example", category="Data", description="",
            keywords=[], homepage="https://example.org", source_urls=["https://example.org"],
            specification=None, data_dictionary=None, license="CC0", access_level="public",
            metadata_path="core/Data/example.yml", metadata_sha256="0" * 64,
        )
        with patch.object(datasets, "_safe_open", return_value=Response()):
            urls, reason = datasets.resolve_assets(record)
        self.assertEqual(urls, ["https://example.org/files/table.csv"])
        self.assertIsNone(reason)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is installed by the full POC requirements")
    def test_manifest_records_unknown_license_as_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "catalog"
            meta = root / "core" / "Data" / "record.yml"
            meta.parent.mkdir(parents=True)
            meta.write_text(
                "title: Restricted\nlicense: Research only\n"
                "sources:\n  - access_url: https://data.example/file.csv\n"
            )
            manifest = base / "manifest.jsonl"
            counts = datasets.run_inventory(
                root, manifest, base / "data", max_file_bytes=1024,
                max_dataset_bytes=1024, max_total_bytes=2048, acquire=True,
            )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(counts["deferred"], 1)
        self.assertEqual(rows[0]["status"], "deferred")
        self.assertIn("license", rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()
