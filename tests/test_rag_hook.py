#!/usr/bin/env python3
"""Run the prompt-time RAG hook's behavior tests under the repo's test runner.

The hook (tools/rag_hook/memory-search-on-prompt.js) is JavaScript, so its
tests live beside it in tools/rag_hook/test_rag_hook.mjs and use node's
built-in test runner. This wrapper makes them discoverable from
``python -m pytest tests`` and ``python -m unittest discover -s tests``.
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = REPO_ROOT / "tools" / "rag_hook" / "test_rag_hook.mjs"


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class RagHookTests(unittest.TestCase):
    def test_node_suite_passes(self) -> None:
        self.assertTrue(TEST_FILE.is_file(), f"missing {TEST_FILE}")
        completed = subprocess.run(
            ["node", "--test", str(TEST_FILE)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        self.assertEqual(0, completed.returncode, f"{completed.stdout}\n{completed.stderr}")
        # Node's reporter prefixes summary lines with "#" (TAP) or "ℹ" (spec).
        self.assertRegex(completed.stdout, r"(?m)^(?:#|ℹ) fail 0\s*$", completed.stdout)


if __name__ == "__main__":
    unittest.main()
