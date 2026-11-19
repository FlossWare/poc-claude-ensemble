#!/usr/bin/env python3
"""Mirror NEW distilled Claude Code memory entries into the CE memory corpus.

What this does
==============
Claude Code distills durable memories into per-project directories under
``~/.claude/projects/<project>/memory/``. This script mirrors those entry
files into the Claude Ensemble memory corpus
(``~/.FlossWare/poc-claude-ensemble/shared/claude-code/memory/``) so CE
sessions can draw on them.

Safety: the sanitize gate is never bypassed
===========================================
Every file's content is passed through the memory service's own
``sanitize_content`` gate (imported from ``services/memory/memory_client.py``
-- the same function ``MemoryClient.write`` applies to every write). Content
the gate rejects (PII, API keys, tokens, emails, usernames) is NEVER written
to the corpus; it is logged as blocked. There is no flag, code path, or
fallback in this script that writes unsanitized content. This is a direct
consequence of the ADP/API-key incident, which came from unsanitized direct
writes to the corpus.

Idempotency
===========
A target file is only written when its content hash (SHA-256) differs from
the source. Unchanged files are skipped, so running this at every session
end is cheap and repeat-free. The corpus itself is the state: no sidecar
index is kept. Changed source content is re-mirrored (overwritten).
Deletions are NOT propagated: this is a one-way add/update mirror of NEW
entries.

Naming and conflicts
====================
Entries are mirrored flat under the corpus using their source filename.
``MEMORY.md`` (the per-project index) is never mirrored. If two projects
hold the same filename with DIFFERENT content, the name is a conflict and
is not written (nothing is clobbered); identical content from several
projects is mirrored once.

Wiring into the deployed session-end hook (DOCUMENTATION ONLY)
==============================================================
This script is not wired anywhere by itself, and this repository must not
modify deployed files from a worktree. To wire it into the EXISTING
session-end hook in the deployed hooks dir, edit the deployed copy of

    ~/.claude/hooks/session-end-memory-capture.js
    (mirror copy:
     ~/.FlossWare/poc-claude-ensemble/shared/claude-code/hooks/session-end-memory-capture.js)

and spawn this script fire-and-forget alongside the existing capture, e.g.
near the top of ``capture(event)``:

    const { spawn } = await import('node:child_process');
    spawn(
      'python3',
      ['/home/sfloess/.FlossWare/poc-claude-ensemble/tools/memory_mirror_sync.py', '--quiet'],
      { detached: true, stdio: 'ignore' }
    ).unref();

The equivalent alternative, without touching the hook file, is a second
SessionEnd command in ~/.claude/settings.json:

    "SessionEnd": [{
      "hooks": [
        { "command": "/home/sfloess/.claude/hooks/session-end-memory-capture.js" },
        { "command": "python3 /home/sfloess/.FlossWare/poc-claude-ensemble/tools/memory_mirror_sync.py --quiet" }
      ]
    }]

Deploy the script first (it locates the sanitize gate relative to itself,
at ``<script>/../services/memory``, which resolves both in the repository
and under the deployed root ``~/.FlossWare/poc-claude-ensemble``).

Usage
=====
    python3 tools/memory_mirror_sync.py [--dry-run] [--quiet] [--json]
        [--source-root DIR] [--corpus DIR] [--memory-service-dir DIR]

Exit status is 0 even when entries are blocked or conflicting (those are
normal, logged outcomes); nonzero only on operational failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("memory_mirror_sync")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_SERVICE_DIR = REPO_ROOT / "services" / "memory"
DEFAULT_SOURCE_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_CORPUS = (
    Path.home() / ".FlossWare" / "poc-claude-ensemble" / "shared" / "claude-code" / "memory"
)

INDEX_FILENAME = "MEMORY.md"


def load_sanitize_gate(memory_service_dir: Path) -> Tuple[Callable[[str], Optional[str]], List[str]]:
    """Import the memory service's own sanitize_content gate and its patterns.

    The gate is imported from the service code, never reimplemented here, so
    this mirror cannot drift from what MemoryClient.write enforces.
    """
    memory_service_dir = Path(memory_service_dir).resolve()
    client_path = memory_service_dir / "memory_client.py"
    if not client_path.is_file():
        raise FileNotFoundError(f"memory service client not found: {client_path}")
    if str(memory_service_dir) not in sys.path:
        sys.path.insert(0, str(memory_service_dir))
    from memory_client import BLOCKED_PATTERNS, sanitize_content  # type: ignore

    return sanitize_content, list(BLOCKED_PATTERNS)


def _blocking_pattern(patterns: List[str], content: str) -> Optional[str]:
    """Return the first service pattern that rejects content, for logging."""
    import re

    for pattern in patterns:
        if re.search(pattern, content):
            return pattern
    return None


def collect_sources(source_root: Path) -> List[Tuple[str, Path]]:
    """Return (project, path) pairs for distilled memory entries.

    Sources are ``<source_root>/<project>/memory/**/*.md``. Index files
    (MEMORY.md) are excluded: they are per-project indexes, not entries,
    and the corpus maintains its own.
    """
    source_root = Path(source_root)
    sources: List[Tuple[str, Path]] = []
    if not source_root.is_dir():
        return sources
    for project_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            continue
        for path in sorted(memory_dir.rglob("*.md")):
            if not path.is_file() or path.name == INDEX_FILENAME:
                continue
            sources.append((project_dir.name, path))
    return sources


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_once(
    sanitize: Callable[[str], Optional[str]],
    patterns: List[str],
    source_root: Path,
    corpus: Path,
    dry_run: bool = False,
) -> Dict[str, List]:
    """Mirror new/changed distilled entries into the corpus, one pass.

    Returns a report with ``synced``, ``updated``, ``skipped`` (unchanged,
    by content hash), ``blocked`` (sanitize gate; never written),
    ``conflicts`` (same name, different content; never written) and
    ``errors`` lists. Every decision is logged.
    """
    source_root = Path(source_root)
    corpus = Path(corpus)
    report: Dict[str, List] = {
        "synced": [],
        "updated": [],
        "skipped": [],
        "blocked": [],
        "conflicts": [],
        "errors": [],
    }

    # Group sources by target filename so cross-project collisions are
    # detected before anything is written.
    by_name: Dict[str, List[Tuple[str, bytes]]] = {}
    for project, path in collect_sources(source_root):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            report["errors"].append(f"{path}: {exc}")
            LOG.error("error reading %s: %s", path, exc)
            continue
        by_name.setdefault(path.name, []).append((project, raw))

    for name, candidates in sorted(by_name.items()):
        digests = {_sha256(raw) for _, raw in candidates}
        if len(digests) > 1:
            report["conflicts"].append(name)
            LOG.warning(
                "conflict: %s -- %d projects hold different content; not written",
                name,
                len(candidates),
            )
            continue
        project, raw = candidates[0]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            report["blocked"].append(f"{name}: not valid UTF-8 ({exc})")
            LOG.warning("blocked: %s -- not valid UTF-8", name)
            continue
        if sanitize(text) is None:
            pattern = _blocking_pattern(patterns, text)
            report["blocked"].append(f"{name}: sanitize gate ({pattern})")
            LOG.warning(
                "blocked: %s -- content failed the memory service sanitize gate (%s); not written",
                name,
                pattern,
            )
            continue
        target = corpus / name
        existed = target.is_file()
        if existed and _sha256(target.read_bytes()) == _sha256(raw):
            report["skipped"].append(name)
            LOG.info("skipped unchanged: %s (from %s)", name, project)
            continue
        if dry_run:
            action = "updated" if existed else "synced"
            report[action].append(name)
            LOG.info("dry-run %s: %s (from %s)", action, name, project)
            continue
        try:
            corpus.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(text.encode("utf-8"))
            os.replace(tmp, target)
        except OSError as exc:
            report["errors"].append(f"{name}: {exc}")
            LOG.error("error writing %s: %s", target, exc)
            continue
        action = "updated" if existed else "synced"
        report[action].append(name)
        LOG.info("%s: %s (from %s)", action, name, project)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="root of Claude Code project dirs (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="CE memory corpus dir (default: ~/.FlossWare/poc-claude-ensemble/shared/claude-code/memory)",
    )
    parser.add_argument(
        "--memory-service-dir",
        type=Path,
        default=DEFAULT_MEMORY_SERVICE_DIR,
        help="dir containing the memory service's memory_client.py (the sanitize gate)",
    )
    parser.add_argument("--dry-run", action="store_true", help="log decisions without writing")
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        sanitize, patterns = load_sanitize_gate(args.memory_service_dir)
    except (FileNotFoundError, ImportError) as exc:
        LOG.error("cannot load the memory service sanitize gate: %s", exc)
        return 2

    report = sync_once(sanitize, patterns, args.source_root, args.corpus, dry_run=args.dry_run)
    summary = (
        "memory mirror: {synced} synced, {updated} updated, {skipped} unchanged, "
        "{blocked} blocked, {conflicts} conflicts, {errors} errors".format(
            **{key: len(value) for key, value in report.items()}
        )
    )
    if not args.quiet or any(report[key] for key in ("blocked", "conflicts", "errors")):
        print(summary, file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
