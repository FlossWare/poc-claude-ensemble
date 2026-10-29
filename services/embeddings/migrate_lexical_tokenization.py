#!/usr/bin/env python3
"""One-time migration of the transcript keyword index to Tokenization-service analysis.

Run with the lexical service STOPPED, because the migration rewrites the FTS table. Steps:

  1. Back up the database with SQLite's online backup API into --backup-dir.
  2. Add messages.analyzed if it is missing.
  3. Fill messages.analyzed through the Tokenization service, in batches. Only rows where
     analyzed IS NULL are processed, so an interrupted run resumes where it stopped.
  4. Recreate messages_fts over the analyzed column and rebuild it, if it does not already index it.
  5. Verify that every message has an analyzed value and that the FTS index covers every row.

Rerunning after completion does nothing. The lexical service must be stopped; the script checks
the port and refuses to run if a service answers there.
"""

import argparse
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lexical_service as lx  # noqa: E402

LEXICAL_PORT = 8772


def _lexical_running():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LEXICAL_PORT}/health", timeout=2):
            return True
    except OSError:
        return False


def _columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def back_up(db_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    target = os.path.join(backup_dir, f"transcripts-before-tokenization-{time.strftime('%Y%m%d-%H%M%S')}.db")
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def migrate(db_path, batch=500, log=print):
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        if "analyzed" not in _columns(conn, "messages"):
            with conn:
                conn.execute("ALTER TABLE messages ADD COLUMN analyzed TEXT")
            log("added messages.analyzed")

        filled, started = 0, time.monotonic()
        while True:
            rows = conn.execute(
                "SELECT id, text FROM messages WHERE analyzed IS NULL ORDER BY id LIMIT ?", (batch,)).fetchall()
            if not rows:
                break
            updates = [(" ".join(lx.analyze(text)), row_id) for row_id, text in rows]
            with conn:
                conn.executemany("UPDATE messages SET analyzed = ? WHERE id = ?", updates)
            filled += len(rows)
            log(f"analyzed {filled} messages ({filled / max(time.monotonic() - started, 1e-9):.0f}/s)")

        fts_columns = _columns(conn, "messages_fts")
        if fts_columns != ["analyzed"]:
            with conn:
                conn.execute("DROP TABLE IF EXISTS messages_fts")
                # CREATE VIRTUAL TABLE does not accept bound parameters, so the tokenizer spec is a
                # string literal with its single quotes doubled.
                literal = lx.FTS_TOKENIZE.replace("'", "''")
                conn.execute(
                    "CREATE VIRTUAL TABLE messages_fts USING fts5(analyzed, content='messages',"
                    f" content_rowid='id', tokenize='{literal}')")
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            log("rebuilt messages_fts over analyzed")

        missing = conn.execute("SELECT COUNT(*) FROM messages WHERE analyzed IS NULL").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        if missing or indexed != total:
            raise RuntimeError(f"verification failed: {missing} unanalyzed, {indexed} indexed of {total}")
        log(f"verified: {total} messages analyzed and indexed")
        return {"messages": total, "analyzed_this_run": filled}
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=lx.DB_PATH, help="transcript database to migrate")
    parser.add_argument("--backup-dir", required=True, help="directory for the pre-migration backup")
    parser.add_argument("--batch", type=int, default=500, help="messages analyzed per transaction")
    parser.add_argument("--tokenization-url", default=lx.TOKENIZATION_URL)
    parser.add_argument("--force-running", action="store_true",
                        help="skip the check that the lexical service is stopped (tests only)")
    args = parser.parse_args(argv)
    if _lexical_running() and not args.force_running:
        parser.error("the lexical service is running on port 8772; stop it before migrating")
    lx.TOKENIZATION_URL = args.tokenization_url
    backup = back_up(args.db, args.backup_dir)
    print(f"backup written: {backup}")
    result = migrate(args.db, batch=args.batch)
    print(f"done: {result}")


if __name__ == "__main__":
    main()
