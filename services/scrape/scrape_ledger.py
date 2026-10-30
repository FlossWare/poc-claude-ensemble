#!/usr/bin/env python3
"""Ledger of fetched scrape URLs, so rescrapes can skip what is already stored.

One row per URL: url, source directory, fetched_at, record path, content SHA-256.
Built from the scraped JSON records. Rebuilds are idempotent: an existing URL is not re-added.
"""
import hashlib
import json
import os
import time

RAW = "/exports/claude-orchestrator/scraped-data/raw"
LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_SCRAPE_LEDGER", "~/.FlossWare/ledger/scrape/fetches.jsonl"))
SOURCES = ["patents", "rfc", "research", "research_paper", "books"]


def load_ledger():
    seen = {}
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    seen[row["url"]] = row
    return seen


def already_fetched(url):
    return url in load_ledger()


def _add_book(out, seen, root, relpath):
    """Add one PDF from the books source (relpath is under the books directory, subfolders included).

    Its identity is its file path, since it has no web URL.
    """
    path = os.path.join(root, relpath)
    url = "file://" + path
    if url in seen:
        return 0
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    row = {"schema_version": 1, "producer": "scrape", "kind": "fetch", "url": url, "source": "books",
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(os.path.getmtime(path))),
           "record": os.path.join("books", relpath), "sha256": digest}
    out.write(json.dumps(row) + "\n")
    seen[url] = row
    return 1


def build():
    seen = load_ledger()
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    added = skipped = bad = 0
    with open(LEDGER, "a", encoding="utf-8") as out:
        for source in SOURCES:
            root = os.path.join(RAW, source)
            if not os.path.isdir(root):
                continue
            if source == "books":
                for dirpath, _dirs, files in os.walk(root):
                    for name in sorted(files):
                        if name.lower().endswith(".pdf"):
                            added += _add_book(out, seen, root, os.path.relpath(os.path.join(dirpath, name), root))
                continue
            for name in os.listdir(root):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                try:
                    raw = open(path, "rb").read()
                    record = json.loads(raw)
                except (OSError, ValueError):
                    bad += 1
                    continue
                url = record.get("url") or record.get("original_url")
                if not url:
                    bad += 1
                    continue
                if url in seen:
                    skipped += 1
                    continue
                row = {"schema_version": 1, "producer": "scrape", "kind": "fetch",
                       "url": url, "source": source,
                       "fetched_at": record.get("fetched_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(os.path.getmtime(path))),
                       "record": os.path.join(source, name),
                       "sha256": hashlib.sha256(raw).hexdigest()}
                out.write(json.dumps(row) + "\n")
                seen[url] = row
                added += 1
    return {"added": added, "already_present": skipped, "unreadable_or_no_url": bad, "total_urls": len(seen)}


if __name__ == "__main__":
    print(json.dumps(build()))
