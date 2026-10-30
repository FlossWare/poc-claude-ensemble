"""Regression tests for the GA scheduler/extractor path contract."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ga_tuning import extract_and_apply_parameters as extractor_module


class GASchedulerContractTests(unittest.TestCase):
    def test_default_settings_target_is_user_claude_settings_not_repo_settings(self):
        args = extractor_module.parse_args([])
        expected = (Path.home() / ".claude" / "settings.json").resolve()
        self.assertEqual(args.settings_path.expanduser().resolve(), expected)
        self.assertNotEqual(args.settings_path.expanduser().resolve(),
                            (Path(__file__).resolve().parent.parent / "settings.json").resolve())

    def test_explicit_paths_are_parsed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = extractor_module.parse_args([
                "--results-dir", str(root / "results"),
                "--settings-path", str(root / "settings.json"),
                "--evolution-log", str(root / "evolution.md"),
            ])
        self.assertEqual(args.results_dir, root / "results")
        self.assertEqual(args.settings_path, root / "settings.json")
        self.assertEqual(args.evolution_log, root / "evolution.md")

    def test_main_fails_closed_when_settings_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            results.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Runtime settings file does not exist"):
                extractor_module.main([
                    "--results-dir", str(results),
                    "--settings-path", str(root / "missing.json"),
                    "--evolution-log", str(root / "evolution.md"),
                ])

    def test_main_rejects_invalid_settings_structure_before_learning_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            results.mkdir()
            settings = root / "settings.json"
            settings.write_text(json.dumps({"env": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object-valued 'env'"):
                extractor_module.main([
                    "--results-dir", str(results),
                    "--settings-path", str(settings),
                    "--evolution-log", str(root / "evolution.md"),
                ])
            self.assertEqual(json.loads(settings.read_text(encoding="utf-8")), {"env": []})

    def test_valid_explicit_paths_reach_extractor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            results.mkdir()
            settings = root / "settings.json"
            settings.write_text(json.dumps({"env": {}}), encoding="utf-8")
            log = root / "evolution.md"
            with patch.object(extractor_module.ParameterExtractor, "run") as run:
                extractor_module.main([
                    "--results-dir", str(results),
                    "--settings-path", str(settings),
                    "--evolution-log", str(log),
                ])
            run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
