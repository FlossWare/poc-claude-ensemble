#!/usr/bin/env python3
"""Fill the knowledge exact-match index from the knowledge vector collection.

Chunks built before the builder wrote exact-match rows have vectors but no exact-match rows. This
reads the stored records page by page (text and metadata, no embeddings) and posts them to the
lexical service's /knowledge/documents. That route is idempotent, so the backfill can be rerun, and
it can run while the builder is writing, without duplicating rows.

Usage: python3 backfill_knowledge_lexical.py [--limit N]
"""

import argparse
import json
import os
import time
import urllib.request

VECTOR_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
LEXICAL_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_LEXICAL_URL", "http://127.0.0.1:8772")
COLLECTION = "knowledge_chunks"
PAGE = 500
POST_BATCH = 100
LOG = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/state/logs/backfill-knowledge-lexical.log")


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=120) as response:
        return json.loads(response.read())


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def chunk_row(record):
    """One exact-match row from a stored record, or None when its metadata cannot identify the chunk."""
    meta = record.get("metadata") or {}
    if not isinstance(meta, dict) or not meta.get("chunk_id") or not meta.get("url"):
        return None
    return {"chunk_id": str(meta["chunk_id"]), "vector_id": str(record["id"]), "url": str(meta["url"]),
            "source": str(meta.get("source") or ""), "title": str(meta.get("title") or ""),
            "text": str(record.get("document") or "")}


def backfill(vector_url=VECTOR_URL, lexical_url=LEXICAL_URL, limit=None, log=None):
    """Copy stored knowledge records into the exact-match index. Returns the totals."""
    totals = {"read": 0, "skipped": 0, "inserted": 0, "already_present": 0}
    offset = 0
    while True:
        page_size = PAGE if limit is None else min(PAGE, limit - totals["read"])
        if page_size <= 0:
            break
        page = get(f"{vector_url}/collections/{COLLECTION}/records?offset={offset}&limit={page_size}")
        records = page["records"]
        if not records:
            break
        rows = []
        for record in records:
            totals["read"] += 1
            row = chunk_row(record)
            if row is None:
                totals["skipped"] += 1
            else:
                rows.append(row)
        for start in range(0, len(rows), POST_BATCH):
            result = post(f"{lexical_url}/knowledge/documents", {"chunks": rows[start:start + POST_BATCH]})
            totals["inserted"] += result["inserted"]
            totals["already_present"] += result["already_present"]
        offset += len(records)
        if log and offset % 5000 < PAGE:
            log(f"progress: {totals} of {page['total']}")
        if offset >= page["total"]:
            break
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many records (trial runs)")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    def log(message):
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")

    started = time.time()
    totals = backfill(limit=args.limit, log=log)
    summary = dict(totals, seconds=round(time.time() - started))
    log(f"done: {json.dumps(summary)}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
