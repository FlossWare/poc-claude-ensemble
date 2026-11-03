"""Exact-match index over knowledge chunks, in its own SQLite FTS5 database.

Each knowledge chunk is analyzed with the same lexical-v1 analyzer as the transcript index, so
identifiers such as RFC numbers, patent numbers, and dotted names match exactly. The analyzer is
passed in by the caller, so this module does not depend on the lexical service's module.

Rows are keyed by chunk_id. Writes are idempotent: a chunk already present is skipped, so a
retried batch or a backfill never duplicates text.
"""

import hashlib
import os
import sqlite3
import threading

DB_PATH = os.path.expanduser(os.environ.get(
    "POC_CLAUDE_ENSEMBLE_KNOWLEDGE_LEXICAL_DB", "~/.FlossWare/poc-claude-ensemble/data/sqlite/knowledge-lexical.sqlite3"))
# Same tokenizer settings as the transcript index, so identifiers keep their dashes, dots, and slashes.
FTS_TOKENIZE = "porter unicode61 remove_diacritics 2 tokenchars '-._:/'"
MAX_CHUNKS_PER_REQUEST = 500
MAX_TEXT_CHARS = 200_000
_WRITE_LOCK = threading.Lock()


class ChunkValidationError(ValueError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    vector_id TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    analyzed TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(analyzed, tokenize='{tokenize}');
""".format(tokenize=FTS_TOKENIZE.replace("'", "''"))


def connect_write(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def connect_read(path=None):
    return sqlite3.connect(f"file:{path or DB_PATH}?mode=ro", uri=True, timeout=10, check_same_thread=False)


def validate_chunks(chunks):
    """Return the chunks as clean dicts, or raise ChunkValidationError. Nothing is written on failure."""
    if not isinstance(chunks, list) or not chunks:
        raise ChunkValidationError("chunks must be a non-empty list")
    if len(chunks) > MAX_CHUNKS_PER_REQUEST:
        raise ChunkValidationError(f"at most {MAX_CHUNKS_PER_REQUEST} chunks per request")
    clean = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ChunkValidationError(f"chunk {index} must be an object")
        for field in ("chunk_id", "vector_id", "url", "source", "title", "text"):
            value = chunk.get(field)
            if not isinstance(value, str):
                raise ChunkValidationError(f"chunk {index}: {field} must be a string")
        if not chunk["chunk_id"].strip() or not chunk["vector_id"].strip():
            raise ChunkValidationError(f"chunk {index}: chunk_id and vector_id must be non-empty")
        if not chunk["text"].strip():
            raise ChunkValidationError(f"chunk {index}: text must be non-empty")
        if len(chunk["text"]) > MAX_TEXT_CHARS:
            raise ChunkValidationError(f"chunk {index}: text exceeds {MAX_TEXT_CHARS} characters")
        clean.append({key: chunk[key] for key in ("chunk_id", "vector_id", "url", "source", "title", "text")})
    return clean


def ingest(chunks, analyze, path=None):
    """Index chunks. Analysis runs before the write lock. Returns inserted and already_present counts."""
    clean = validate_chunks(chunks)
    analyzed = [" ".join(analyze(chunk["text"])) for chunk in clean]
    conn = connect_write(path)
    inserted = already = 0
    try:
        with _WRITE_LOCK:
            with conn:
                for chunk, terms in zip(clean, analyzed):
                    present = conn.execute("SELECT 1 FROM knowledge_chunks WHERE chunk_id = ?",
                                           (chunk["chunk_id"],)).fetchone()
                    if present:
                        already += 1
                        continue
                    cursor = conn.execute(
                        "INSERT INTO knowledge_chunks (chunk_id, vector_id, url, source, title, text, analyzed) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (chunk["chunk_id"], chunk["vector_id"], chunk["url"], chunk["source"],
                         chunk["title"], chunk["text"], terms))
                    conn.execute("INSERT INTO knowledge_fts (rowid, analyzed) VALUES (?, ?)",
                                 (cursor.lastrowid, terms))
                    inserted += 1
    finally:
        conn.close()
    return {"inserted": inserted, "already_present": already}


def match_expression(terms):
    seen, quoted = set(), []
    for term in terms:
        if term not in seen:
            seen.add(term)
            quoted.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(quoted)


def search(query, top_k, analyze, path=None):
    """Exact-match search. Returns ranked chunks with their vector_id, so callers can fuse with vector hits."""
    terms = analyze(query)
    if not terms or not os.path.exists(path or DB_PATH):
        return []
    conn = connect_read(path)
    try:
        rows = conn.execute(
            """SELECT k.vector_id, k.chunk_id, k.url, k.source, k.title, k.text
               FROM knowledge_fts
               JOIN knowledge_chunks k ON k.id = knowledge_fts.rowid
               WHERE knowledge_fts MATCH ?
               ORDER BY bm25(knowledge_fts)
               LIMIT ?""",
            (match_expression(terms), top_k),
        ).fetchall()
    finally:
        conn.close()
    return [{"vector_id": vector_id, "chunk_id": chunk_id, "url": url, "source": source, "title": title,
             "text": text, "key": vector_id} for vector_id, chunk_id, url, source, title, text in rows]


def count(path=None):
    if not os.path.exists(path or DB_PATH):
        return 0
    conn = connect_read(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]
    finally:
        conn.close()
