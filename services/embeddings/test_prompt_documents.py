import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import build_knowledge_chunks as bk


def row(event_id, prompt, timestamp="2026-11-04T07:45:36.513Z", cwd="/home/u/proj", kind="prompt"):
    return {"schema_version": 1, "producer": "claude-code", "kind": kind, "event_id": event_id,
            "session_id": "s", "timestamp": timestamp, "cwd": cwd, "prompt": prompt}


class PromptDocumentTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ledger = os.path.join(self.dir, "prompts.jsonl")
        patch = mock.patch.object(bk, "PROMPT_LEDGER", self.ledger)
        patch.start()
        self.addCleanup(patch.stop)
        self.logged = []
        log_patch = mock.patch.object(bk, "log", lambda message: self.logged.append(message))
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def write(self, rows):
        with open(self.ledger, "w", encoding="utf-8") as handle:
            for item in rows:
                handle.write((item if isinstance(item, str) else json.dumps(item)) + "\n")

    def test_each_prompt_becomes_one_document(self):
        self.write([row("e1", "why is the embedding service slow"), row("e2", "status")])
        docs = list(bk.prompt_documents())
        self.assertEqual([d["url"] for d in docs], ["prompt://e1", "prompt://e2"])
        self.assertEqual(docs[0]["content"], "why is the embedding service slow")
        self.assertEqual(docs[0]["source"], "prompts")
        self.assertEqual(docs[0]["title"], "Prompt 2026-11-04 (proj)")

    def test_credential_looking_prompts_are_skipped(self):
        self.write([row("keep", "how do I rotate the database"),
                    row("token", "use ATATT3xFfGF0abcdefghijklmnopqrstuvwxyz0123456789 for jira"),
                    row("pw", "the password: hunter2hunter2 is in the vault"),
                    row("key", "-----BEGIN RSA PRIVATE KEY-----")])
        docs = list(bk.prompt_documents())
        self.assertEqual([d["url"] for d in docs], ["prompt://keep"])
        self.assertEqual(sum("credential" in m for m in self.logged), 3)

    def test_other_kinds_empty_prompts_and_bad_lines_are_ignored(self):
        self.write([row("a", "real", kind="fetch"), row("b", "   "), "not json", row("c", "kept")])
        self.assertEqual([d["url"] for d in bk.prompt_documents()], ["prompt://c"])
        self.assertTrue(any("malformed" in m for m in self.logged))

    def test_bad_timestamp_is_skipped(self):
        self.write([row("bad", "no time", timestamp=""), row("good", "has time")])
        self.assertEqual([d["url"] for d in bk.prompt_documents()], ["prompt://good"])

    def test_missing_ledger_yields_nothing(self):
        self.assertEqual(list(bk.prompt_documents()), [])

    def test_documents_includes_prompts_only_when_selected(self):
        self.write([row("e1", "a prompt")])
        empty_scrape = os.path.join(self.dir, "fetches.jsonl")
        open(empty_scrape, "w").close()
        with mock.patch.object(bk, "LEDGER", empty_scrape):
            self.assertEqual([d["source"] for d in bk.documents(["prompts"])], ["prompts"])
            self.assertEqual([d["source"] for d in bk.documents(["rfc"])], [])


if __name__ == "__main__":
    unittest.main()
