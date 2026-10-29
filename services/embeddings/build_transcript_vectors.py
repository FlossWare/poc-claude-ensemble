#!/usr/bin/env python3
"""Embed user and assistant text blocks from the transcript index, through REST services.

This is a client: it reads texts from the transcript index (read-only SQLite), asks the
embedding service (8770) for document vectors, and writes them through the vector
service (8771). It does not load the model or open the vector store itself.

Resumable: record IDs are the SHA-256 of the text, so re-running skips anything already
stored and identical texts are embedded once. Commits after every batch.
"""

import hashlib
import json
import os
import sqlite3
import time
import urllib.request

HOME = os.path.expanduser("~")
TRANSCRIPTS_DB = os.path.join(HOME, ".FlossWare/claude-ensemble/shared/data/transcripts/transcripts.db")
EMBEDDING_URL = os.environ.get("ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
VECTOR_URL = os.environ.get("ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
COLLECTION = "transcript_text"
LOG = os.path.join(HOME, ".FlossWare/claude-ensemble-state/logs/embed-transcripts.log")
BATCH = 64
EMBED_CHARS = 4000   # the model reads at most 512 tokens (about 2-3k chars), so longer text adds nothing
CHUNK = 1024      # sort by length within chunks to limit padding
MIN_CHARS = 20
TIMEOUT = 300


RETRY_SECONDS = (2, 5, 10, 20, 30, 60)   # a service restart pauses the job rather than killing it


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    for attempt, wait in enumerate((*RETRY_SECONDS, None)):
        request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"},
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = json.loads(response.read())
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if wait is None:
                raise
            log(f"{url} unavailable ({exc}); retry {attempt + 1} in {wait}s")
            time.sleep(wait)
    if body.get("ok") is not True:
        raise RuntimeError(f"{url} returned {body}")
    return body


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log(message):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def existing_ids(ids):
    found = set()
    for start in range(0, len(ids), 1000):
        body = _post(f"{VECTOR_URL}/collections/{COLLECTION}/existing", {"ids": ids[start : start + 1000]})
        found.update(body["found"])
    return found


def main():
    source = sqlite3.connect(f"file:{TRANSCRIPTS_DB}?mode=ro", uri=True)
    rows = source.execute(
        "SELECT DISTINCT text FROM messages "
        "WHERE kind = 'text' AND role IN ('user', 'assistant') AND length(text) >= ? "
        "ORDER BY length(text)",
        (MIN_CHARS,),
    ).fetchall()
    texts = [r[0] for r in rows]
    ids = [text_id(t) for t in texts]
    done = existing_ids(ids)
    pending = [(i, t) for i, t in zip(ids, texts) if i not in done]
    log(f"start: {len(texts)} distinct texts, {len(done)} already stored, {len(pending)} pending")
    print(f"{len(texts)} distinct texts, {len(pending)} pending", flush=True)
    if not pending:
        return

    started = time.time()
    processed = 0
    for chunk_start in range(0, len(pending), CHUNK):
        chunk = sorted(pending[chunk_start : chunk_start + CHUNK], key=lambda p: len(p[1]))
        for start in range(0, len(chunk), BATCH):
            batch = chunk[start : start + BATCH]
            texts_out = [t for _, t in batch]
            embedded = _post(f"{EMBEDDING_URL}/embeddings",
                             {"texts": [t[:EMBED_CHARS] for t in texts_out], "mode": "document"})
            _post(f"{VECTOR_URL}/collections/{COLLECTION}/upsert", {
                "ids": [i for i, _ in batch],
                "embeddings": embedded["embeddings"],
                "documents": texts_out,
                "metadatas": [{"kind": "transcript_text"}] * len(batch),
            })
            processed += len(batch)
        rate = processed / max(time.time() - started, 1e-9)
        log(f"progress: {processed}/{len(pending)} at {rate:.2f} texts/s")
        print(f"  {processed}/{len(pending)} at {rate:.2f} texts/s", flush=True)
    log(f"done: {processed} embedded in {time.time() - started:.0f}s")
    print("done", flush=True)


if __name__ == "__main__":
    main()
