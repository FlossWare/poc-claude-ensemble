#!/usr/bin/env python3
"""Rebuild Thompson outcome statistics from every top-level session transcript, oldest first.

Each finished turn is recorded once, through the same ledger-first path as live ingest, so a rerun
never records a turn twice. Run it after Thompson's statistics have been reset and the outcome
ledger has been moved aside. Subagent transcripts are not included.

Usage: rebuild_outcomes.py [--root DIR] [--limit N]
"""

import argparse
import glob
import os

import turn_outcomes

DEFAULT_ROOT = os.path.expanduser("~/.claude/projects")


def session_files(root):
    files = glob.glob(os.path.join(root, "*", "*.jsonl"))
    files.sort(key=os.path.getmtime)
    return [os.path.realpath(path) for path in files]


def rebuild(root, limit=None):
    totals = {"files": 0, "accepted": 0, "pending": 0}
    for path in session_files(root)[:limit]:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines(True)
        accepted, pending = turn_outcomes.record_pending(turn_outcomes.outcomes_from_lines(lines))
        totals["files"] += 1
        totals["accepted"] += accepted
        totals["pending"] += pending
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    print(rebuild(args.root, args.limit))


if __name__ == "__main__":
    main()
