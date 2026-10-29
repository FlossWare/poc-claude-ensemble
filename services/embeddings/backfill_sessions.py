#!/usr/bin/env python3
"""Backfill older finished sessions into the indexes, without recording Thompson outcomes.

Sends each top-level session transcript (oldest first) to the hybrid service's /sessions/ingest with
record_outcomes set to false, then waits until the stored offset reaches the file size before the next
one. Sessions already fully ingested are skipped without any work.

Usage: backfill_sessions.py [--root DIR] [--limit N] [--dry-run]
"""

import argparse
import glob
import json
import os
import time
import urllib.error
import urllib.request

HYBRID_URL = os.environ.get("ENSEMBLE_HYBRID_URL", "http://127.0.0.1:8773")
STATE = os.path.expanduser(os.environ.get("ENSEMBLE_INGEST_STATE",
                                          "~/.FlossWare/poc-claude-ensemble/state/ingest-state.json"))
DEFAULT_ROOT = os.path.expanduser("~/.claude/projects")
WAIT_SECONDS = 900


def load_state(path=STATE):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def pending(root, state):
    """Top-level transcripts whose stored offset does not yet cover the file, oldest first."""
    files = glob.glob(os.path.join(root, "*", "*.jsonl"))
    files.sort(key=os.path.getmtime)
    result = []
    for path in files:
        real = os.path.realpath(path)
        entry = state.get(real) if isinstance(state.get(real), dict) else {}
        offset = entry.get("offset") or 0
        if offset < os.path.getsize(real):
            result.append(real)
    return result


def submit(path):
    body = json.dumps({"transcript_path": path, "record_outcomes": False}).encode("utf-8")
    request = urllib.request.Request(f"{HYBRID_URL}/sessions/ingest", data=body, method="POST",
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def wait_until_done(path, timeout=WAIT_SECONDS):
    """Done when the stored offset covers the file. Returns False if the timeout passes first."""
    deadline = time.time() + timeout
    size = os.path.getsize(path)
    while time.time() < deadline:
        entry = load_state().get(path)
        if isinstance(entry, dict) and (entry.get("offset") or 0) >= size:
            return True
        time.sleep(2)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    todo = pending(args.root, load_state())
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} transcripts to backfill")
    if args.dry_run:
        return
    done = failed = 0
    for index, path in enumerate(todo, 1):
        try:
            submit(path)
        except (urllib.error.URLError, OSError) as exc:
            failed += 1
            print(f"[{index}] submit failed {os.path.basename(path)}: {exc}")
            continue
        if wait_until_done(path):
            done += 1
        else:
            failed += 1
            print(f"[{index}] timed out {os.path.basename(path)}")
        if index % 25 == 0:
            print(f"progress {index}/{len(todo)} done={done} failed={failed}")
    print(f"finished: done={done} failed={failed}")


if __name__ == "__main__":
    main()
