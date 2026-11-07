#!/usr/bin/env python3
"""Fill the knowledge exact-match index from the knowledge vector collection.

Chunks built before the builder wrote exact-match rows have vectors but no exact-match rows. This
reads the stored records page by page (text and metadata, no embeddings) and posts them to the
lexical service's /knowledge/documents. That route is idempotent, so the backfill can be rerun, and
it can run while the builder is writing, without duplicating rows.

Pages are read in order. Batches are posted by a small worker pool, so several requests to the
lexical service run at once. Totals are updated only by the reading thread.

Usage: python3 backfill_knowledge_lexical.py [--limit N] [--start-offset N] [--workers N]
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request

VECTOR_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
LEXICAL_URL = os.environ.get("POC_CLAUDE_ENSEMBLE_LEXICAL_URL", "http://127.0.0.1:8772")
COLLECTION = "knowledge_chunks"
PAGE = 500
POST_BATCH = 100
DEFAULT_WORKERS = 4
LOG = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/state/logs/backfill-knowledge-lexical.log")
TOTAL_KEYS = ("inserted", "already_present", "repaired")


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
            "text": str(record.get("document") or ""),
            "chunk_index": meta.get("chunk"), "chunk_count": meta.get("chunks"),
            "content_sha256": meta.get("content_sha256")}


def backfill(vector_url=VECTOR_URL, lexical_url=LEXICAL_URL, limit=None, log=None,
             start_offset=0, workers=DEFAULT_WORKERS):
    """Copy stored knowledge records into the exact-match index. Returns the totals."""
    totals = {"read": 0, "skipped": 0, "inserted": 0, "already_present": 0, "repaired": 0}
    offset = start_offset
    max_in_flight = max(1, workers) * 2
    in_flight = set()

    def absorb(done):
        for future in done:
            result = future.result()  # re-raises a failed post, so the run stops rather than skipping rows
            for key in TOTAL_KEYS:
                totals[key] += result.get(key, 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
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
                in_flight.add(pool.submit(post, f"{lexical_url}/knowledge/documents",
                                          {"chunks": rows[start:start + POST_BATCH]}))
                while len(in_flight) >= max_in_flight:
                    done, in_flight = concurrent.futures.wait(
                        in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
                    absorb(done)
            offset += len(records)
            if log and offset % 5000 < PAGE:
                log(f"progress: offset {offset}, {totals} of {page['total']}")
            if offset >= page["total"]:
                break
        done, in_flight = concurrent.futures.wait(in_flight)
        absorb(done)
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many records (trial runs)")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="first record offset to read (resume a run that already covered earlier offsets)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="concurrent posts to the lexical service (default %(default)s)")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    def log(message):
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")

    started = time.time()
    log(f"start: start_offset={args.start_offset} workers={args.workers} limit={args.limit}")
    totals = backfill(limit=args.limit, log=log, start_offset=args.start_offset, workers=args.workers)
    summary = dict(totals, seconds=round(time.time() - started))
    log(f"done: {json.dumps(summary)}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
