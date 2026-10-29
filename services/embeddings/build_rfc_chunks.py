#!/usr/bin/env python3
"""Embed every chunk of every RFC into the rfc_chunks collection, through REST services.

Client only: reads RFC texts from the scrape ledger's records, asks the embedding service
(full precision) for document vectors, and writes through the vector service. Chunks are
overlapping character windows. Each chunk's ID is a hash of its URL and index, so re-running
skips anything already stored. Commits per batch and resumes after interruption.
"""

import hashlib
import json
import os
import time
import urllib.request

HOME = os.path.expanduser("~")
RAW = "/exports/claude-orchestrator/scraped-data/raw"
LEDGER = os.path.join(HOME, ".FlossWare/claude-ensemble/shared/data/scrape/ledger.jsonl")
EMBEDDING_URL = os.environ.get("ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
VECTOR_URL = os.environ.get("ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
COLLECTION = "rfc_chunks"
SOURCE = "rfc"
LOG = os.path.join(HOME, ".FlossWare/claude-ensemble-state/logs/embed-rfc-chunks.log")
CHUNK_CHARS = 1800      # the model reads at most 512 tokens, about 2,000 characters
OVERLAP_CHARS = 200
EMBED_BATCH = 64        # the embedding service's per-request cap
UPSERT_BATCH = 128
TIMEOUT = 600


def log(message):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = json.loads(response.read())
    if body.get("ok") is not True:
        raise RuntimeError(f"{url} returned {body}")
    return body


def chunks_of(text):
    text = " ".join(text.split())
    if not text:
        return []
    pieces, start = [], 0
    step = CHUNK_CHARS - OVERLAP_CHARS
    while start < len(text):
        pieces.append(text[start:start + CHUNK_CHARS])
        if start + CHUNK_CHARS >= len(text):
            break
        start += step
    return pieces


def chunk_id(url, index):
    return hashlib.sha256(f"{url}|{index}".encode("utf-8")).hexdigest()


def rfc_documents():
    with open(LEDGER, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        if row.get("source") != SOURCE:
            continue
        path = os.path.join(RAW, row["record"])
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        content = record.get("content") or ""
        pieces = chunks_of(content)
        if pieces:
            yield row["url"], (record.get("title") or ""), pieces


def existing(ids):
    found = set()
    for start in range(0, len(ids), 1000):
        found.update(post(f"{VECTOR_URL}/collections/{COLLECTION}/existing",
                          {"ids": ids[start:start + 1000]})["found"])
    return found


def main():
    started = time.time()
    stored = documents = 0
    pending = []   # (id, text, metadata)

    def flush(batch):
        nonlocal stored
        texts = [t for _, t, _ in batch]
        embedded = post(f"{EMBEDDING_URL}/embeddings", {"texts": texts, "mode": "document"})
        post(f"{VECTOR_URL}/collections/{COLLECTION}/upsert", {
            "ids": [i for i, _, _ in batch], "embeddings": embedded["embeddings"],
            "documents": texts, "metadatas": [m for _, _, m in batch]})
        stored += len(batch)

    for url, title, pieces in rfc_documents():
        documents += 1
        ids = [chunk_id(url, i) for i in range(len(pieces))]
        present = existing(ids)
        for index, (cid, text) in enumerate(zip(ids, pieces)):
            if cid in present:
                continue
            pending.append((cid, text, {"kind": "rfc_chunk", "source": SOURCE, "url": url, "title": title[:300],
                                        "chunk": index, "chunks": len(pieces)}))
            if len(pending) >= EMBED_BATCH:
                flush(pending)
                pending = []
        if documents % 50 == 0:
            log(f"progress: {documents} RFCs read, {stored} chunks stored, {time.time() - started:.0f}s")
    if pending:
        flush(pending)
    log(f"done: {documents} RFCs, {stored} new chunks stored in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
