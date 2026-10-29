"""Tests for the lexical tokenization migration: full run on the old layout, rerun, resume, and backup."""
import os
import sqlite3
import sys
import tempfile
import threading
import unittest

_SERVICES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SERVICES, "tokenization"))

import lexical_service as lx  # noqa: E402
import migrate_lexical_tokenization as mig  # noqa: E402
import tokenization_service as tk  # noqa: E402

# The layout the deployed database had before the migration.
OLD_SCHEMA = """
CREATE TABLE sessions(
    session_id TEXT, source TEXT, project TEXT, title TEXT,
    first_ts TEXT, last_ts TEXT, file TEXT PRIMARY KEY,
    is_subagent INTEGER, bytes INTEGER);
CREATE TABLE messages(
    id INTEGER PRIMARY KEY, session_id TEXT, uuid TEXT, role TEXT,
    kind TEXT, ts TEXT, text TEXT, file TEXT);
CREATE INDEX messages_session ON messages(session_id);
CREATE INDEX messages_file ON messages(file);
CREATE VIRTUAL TABLE messages_fts USING fts5(
    text, content='messages', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2');
"""

TEXTS = [
    "the memory-service restarted after the crash",
    "edited /src/app/tokenizer.py during the review",
    "plain sentence about deployment windows",
    "trace 3f2a9c1b-7d4e-4f10-8a2b-9c0d1e2f3a4b logged",
]


class MigrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server = tk.create_server("127.0.0.1", 0)
        cls.server = server
        cls.url = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "old.db")
        conn = sqlite3.connect(self.db)
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO sessions (session_id, source, project, file) VALUES ('s', 'x', 'p', '/p/s.jsonl')")
        for index, text in enumerate(TEXTS, start=1):
            conn.execute("INSERT INTO messages (id, session_id, uuid, role, kind, ts, text, file)"
                         " VALUES (?, 's', ?, 'user', 'text', '', ?, '/p/s.jsonl')", (index, f"u{index}", text))
        conn.execute("INSERT INTO messages_fts (rowid, text) SELECT id, text FROM messages")
        conn.commit()
        conn.close()
        self._saved = lx.TOKENIZATION_URL
        lx.TOKENIZATION_URL = self.url

    def tearDown(self):
        lx.TOKENIZATION_URL = self._saved

    def columns(self, table):
        conn = sqlite3.connect(self.db)
        try:
            return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        finally:
            conn.close()


class FullMigrationTests(MigrationTestCase):
    def test_migrates_old_layout_and_search_finds_identifiers(self):
        result = mig.migrate(self.db, batch=2, log=lambda *_: None)
        self.assertEqual(result, {"messages": len(TEXTS), "analyzed_this_run": len(TEXTS)})
        self.assertIn("analyzed", self.columns("messages"))
        self.assertEqual(self.columns("messages_fts"), ["analyzed"])

        lx.DB_PATH = self.db
        self.assertEqual(len(lx.search("memory-service", 5)), 1)
        self.assertEqual(len(lx.search("/src/app/tokenizer.py", 5)), 1)
        self.assertEqual(len(lx.search("3f2a9c1b-7d4e-4f10-8a2b-9c0d1e2f3a4b", 5)), 1)

    def test_rerun_after_completion_analyzes_nothing(self):
        mig.migrate(self.db, batch=2, log=lambda *_: None)
        again = mig.migrate(self.db, batch=2, log=lambda *_: None)
        self.assertEqual(again["analyzed_this_run"], 0)
        self.assertEqual(again["messages"], len(TEXTS))

    def test_interrupted_run_resumes_without_reanalyzing_finished_rows(self):
        real_analyze = lx.analyze
        calls = {"n": 0}

        def flaky(text):
            calls["n"] += 1
            if calls["n"] == 3:
                raise lx.AnalysisUnavailable("simulated outage mid-migration")
            return real_analyze(text)

        lx.analyze = flaky
        try:
            with self.assertRaises(lx.AnalysisUnavailable):
                mig.migrate(self.db, batch=1, log=lambda *_: None)
        finally:
            lx.analyze = real_analyze

        conn = sqlite3.connect(self.db)
        try:
            done = conn.execute("SELECT id, analyzed FROM messages WHERE analyzed IS NOT NULL ORDER BY id").fetchall()
        finally:
            conn.close()
        self.assertEqual([row[0] for row in done], [1, 2])
        self.assertEqual(self.columns("messages_fts"), ["text"])  # FTS is only rebuilt once every row is analyzed

        result = mig.migrate(self.db, batch=1, log=lambda *_: None)
        self.assertEqual(result["analyzed_this_run"], len(TEXTS) - 2)
        self.assertEqual(self.columns("messages_fts"), ["analyzed"])

    def test_tokenization_outage_at_start_leaves_index_unchanged(self):
        lx.TOKENIZATION_URL = "http://127.0.0.1:1"
        with self.assertRaises(lx.AnalysisUnavailable):
            mig.migrate(self.db, batch=2, log=lambda *_: None)
        self.assertEqual(self.columns("messages_fts"), ["text"])
        conn = sqlite3.connect(self.db)
        try:
            analyzed = conn.execute("SELECT COUNT(analyzed) FROM messages").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(analyzed, 0)

    def test_backup_is_a_readable_copy_of_the_old_layout(self):
        backup_dir = os.path.join(self.dir, "backups")
        path = mig.back_up(self.db, backup_dir)
        self.assertTrue(path.startswith(backup_dir))
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], len(TEXTS))
            self.assertNotIn("analyzed", [row[1] for row in conn.execute("PRAGMA table_info(messages)")])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
