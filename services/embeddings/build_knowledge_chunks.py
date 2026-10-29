#!/usr/bin/env python3
"""Ingest scraped documents as Knowledge records, then chunk, embed, and store their chunks.

For each scrape-ledger row of the selected sources:

  1. Knowledge stores the document as a sourced record. Knowledge asks Index whether the revision
     may publish. A refused revision is logged and not chunked.
  2. Chunking splits the stored text with the window-v1 profile. Chunk IDs come from that service.
  3. Vectors are written for chunks not stored yet: the embedding service, then the vector service.

Client only. Every step runs in the owning service over REST. A dependency failure stops the run
loudly; rerunning resumes, because a chunk is keyed by its chunk ID and content revision and
stored chunks are skipped.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime

HOME = os.path.expanduser("~")
RAW = "/exports/claude-orchestrator/scraped-data/raw"
LEDGER = os.path.join(HOME, ".FlossWare/ledger/scrape/fetches.jsonl")
LOG = os.path.join(HOME, ".FlossWare/poc-claude-ensemble/state/logs/embed-knowledge-chunks.log")
EMBEDDING_URL = os.environ.get("ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
VECTOR_URL = os.environ.get("ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
CHUNKING_URL = os.environ.get("ENSEMBLE_CHUNKING_URL", "http://127.0.0.1:8775")
KNOWLEDGE_URL = os.environ.get("ENSEMBLE_KNOWLEDGE_URL", "http://127.0.0.1:8781")
COLLECTION = "knowledge_chunks"
CHUNK_PROFILE = "window-v1"
PROFILES = {"chunking": "window-v1@1", "embedding": "nomic-embed-text-v1.5@fp32", "vector": "knowledge_chunks@1"}
SOURCES = ("rfc", "patents", "research_paper", "research")
EMBED_BATCH = 64        # the embedding service's per-request cap
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


def revision_sequence(fetched_at):
    """A later fetch of the same URL gets a larger sequence, so Knowledge and Index accept it as newer."""
    try:
        return max(1, int(datetime.fromisoformat(fetched_at).timestamp()))
    except (TypeError, ValueError):
        return 1


def documents(sources):
    """Yield every ledger document of the selected sources whose raw record can be read."""
    wanted = set(sources)
    with open(LEDGER, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        if row.get("source") not in wanted:
            continue
        try:
            with open(os.path.join(RAW, row["record"]), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError, KeyError):
            log(f"skipped unreadable record for {row.get('url')}")
            continue
        content = record.get("content") or ""
        if not content.strip():
            continue
        yield {"url": row["url"], "source": row["source"], "fetched_at": row.get("fetched_at"),
               "title": (record.get("title") or row["url"])[:300], "content": content}


def store_knowledge(doc):
    """Store the document as a Knowledge record. Returns True when its revision may be chunked."""
    body = post(f"{KNOWLEDGE_URL}/records", {
        "source_id": doc["url"],
        "revision_sequence": revision_sequence(doc["fetched_at"]),
        "revision_id": hashlib.sha256(doc["content"].encode("utf-8")).hexdigest(),
        "title": doc["title"],
        "content": doc["content"],
        "provenance": {"origin": doc["source"], "locator": doc["url"]},
        "profiles": PROFILES,
    })
    return body["decision"] in ("accepted", "duplicate")


def chunk(doc):
    body = post(f"{CHUNKING_URL}/chunk", {"profile": CHUNK_PROFILE, "source_id": doc["url"],
                                          "media_type": "text/plain", "text": doc["content"]})
    return body["chunks"]


def vector_id(chunk_id, content_sha256):
    """Chunk identity plus the content revision, so changed text never reuses a stale vector."""
    return hashlib.sha256(f"{chunk_id}|{content_sha256}".encode("utf-8")).hexdigest()


def existing(ids):
    found = set()
    for start in range(0, len(ids), 1000):
        found.update(post(f"{VECTOR_URL}/collections/{COLLECTION}/existing",
                          {"ids": ids[start:start + 1000]})["found"])
    return found


def flush(batch):
    texts = [text for _, text, _ in batch]
    embedded = post(f"{EMBEDDING_URL}/embeddings", {"texts": texts, "mode": "document"})
    post(f"{VECTOR_URL}/collections/{COLLECTION}/upsert", {
        "ids": [vid for vid, _, _ in batch], "embeddings": embedded["embeddings"],
        "documents": texts, "metadatas": [meta for _, _, meta in batch]})
    return len(batch)


def ingest(doc, pending):
    """Chunk one stored document and queue chunks that are not yet vectors. Returns chunks queued."""
    content_sha = hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()
    chunks = chunk(doc)
    ids = [vector_id(c["chunk_id"], content_sha) for c in chunks]
    present = existing(ids) if ids else set()
    queued = 0
    for vid, record in zip(ids, chunks):
        if vid in present:
            continue
        pending.append((vid, record["text"], {
            "kind": "knowledge_chunk", "source": doc["source"], "url": doc["url"], "title": doc["title"],
            "chunk": record["index"], "chunks": len(chunks), "chunk_id": record["chunk_id"],
            "content_sha256": content_sha}))
        queued += 1
    return queued


def run(sources, limit=None):
    started = time.time()
    documents_seen = documents_accepted = chunks_stored = 0
    pending = []
    for doc in documents(sources):
        if limit is not None and documents_seen >= limit:
            break
        documents_seen += 1
        if not store_knowledge(doc):
            log(f"refused by Index: {doc['url']}")
            continue
        documents_accepted += 1
        ingest(doc, pending)
        while len(pending) >= EMBED_BATCH:
            batch, pending = pending[:EMBED_BATCH], pending[EMBED_BATCH:]
            chunks_stored += flush(batch)
        if documents_seen % 50 == 0:
            log(f"progress: {documents_seen} documents read, {chunks_stored} chunks stored, "
                f"{time.time() - started:.0f}s")
    if pending:
        chunks_stored += flush(pending)
    summary = {"documents": documents_seen, "accepted": documents_accepted, "chunks_stored": chunks_stored,
               "seconds": round(time.time() - started)}
    log(f"done: {json.dumps(summary)}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", choices=SOURCES,
                        help="ledger source to ingest; repeat for several (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many documents (trial runs)")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.source or list(SOURCES), args.limit)))


if __name__ == "__main__":
    main()
