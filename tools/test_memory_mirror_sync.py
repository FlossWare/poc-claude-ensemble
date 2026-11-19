#!/usr/bin/env python3
"""Tests for tools/memory_mirror_sync.py.

Covers the two contract points that matter most:

1. Sanitize enforcement -- content the memory service's sanitize_content
   gate rejects is NEVER written to the corpus, and the gate used is the
   service's own function (not a copy).
2. Idempotency -- unchanged files (by content hash) are skipped, so a
   second run performs no writes.

Run with pytest, or standalone: python3 tools/test_memory_mirror_sync.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import memory_mirror_sync as mirror  # noqa: E402


@pytest.fixture(scope="module")
def gate():
    sanitize, patterns = mirror.load_sanitize_gate(mirror.DEFAULT_MEMORY_SERVICE_DIR)
    return sanitize, patterns


def make_source(source_root: Path, project: str, name: str, content: str) -> Path:
    memory_dir = source_root / project / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def test_gate_is_the_memory_service_function():
    """The mirror must use the service's own sanitize_content, not a copy."""
    sys.modules.pop("memory_client", None)
    loaded, _ = mirror.load_sanitize_gate(mirror.DEFAULT_MEMORY_SERVICE_DIR)
    import memory_client

    assert loaded is memory_client.sanitize_content
    assert Path(memory_client.__file__) == (
        mirror.DEFAULT_MEMORY_SERVICE_DIR / "memory_client.py"
    ).resolve()


def test_clean_entry_is_synced(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "process_release.md", "# Release\n\nMark version released.\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["synced"] == ["process_release.md"]
    assert (corpus / "process_release.md").read_text(encoding="utf-8") == "# Release\n\nMark version released.\n"
    assert not report["blocked"] and not report["errors"] and not report["conflicts"]


def test_unchanged_entry_is_skipped_by_content_hash(tmp_path, gate):
    """Idempotency: a second run writes nothing for unchanged content."""
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "process_release.md", "stable content\n")

    first = mirror.sync_once(sanitize, patterns, source_root, corpus)
    assert first["synced"] == ["process_release.md"]
    target = corpus / "process_release.md"
    first_mtime_ns = target.stat().st_mtime_ns

    time.sleep(0.01)  # ensure a rewrite would produce a different mtime
    second = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert second["synced"] == [] and second["updated"] == []
    assert second["skipped"] == ["process_release.md"]
    assert target.stat().st_mtime_ns == first_mtime_ns


def test_changed_entry_is_updated(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "process_release.md", "original\n")
    mirror.sync_once(sanitize, patterns, source_root, corpus)
    make_source(source_root, "proj-alpha", "process_release.md", "revised\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["updated"] == ["process_release.md"]
    assert (corpus / "process_release.md").read_text(encoding="utf-8") == "revised\n"


BLOCKED_SAMPLES = {
    "api_key": "the deployment uses api_key ABC123 in the header",
    "password": "password: hunter2",
    "bearer": "Authorization: Bearer eyJhbGciOi.example.sig",
    "email": "contact someone@redhat.com about this",
    "username": "owned by sfloess",
}


@pytest.mark.parametrize("label,sample", sorted(BLOCKED_SAMPLES.items()))
def test_blocked_content_is_never_written(tmp_path, gate, label, sample):
    """Sanitize enforcement: gate-rejected content never reaches the corpus."""
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", f"secret_{label}.md", f"# Note\n\n{sample}\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert len(report["blocked"]) == 1 and report["blocked"][0].startswith(f"secret_{label}.md")
    assert report["synced"] == [] and report["updated"] == []
    assert not (corpus / f"secret_{label}.md").exists()


def test_gate_patterns_are_enforced_not_local_ones(tmp_path, gate):
    """The real service patterns apply: even the word 'token' is blocked."""
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "benign_mention.md", "uses a token bucket\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["blocked"] and not report["synced"]
    assert not (corpus / "benign_mention.md").exists()


def test_memory_md_index_is_not_mirrored(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "MEMORY.md", "- [Entry](entry.md)\n")
    make_source(source_root, "proj-alpha", "entry.md", "entry body\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["synced"] == ["entry.md"]
    assert not (corpus / "MEMORY.md").exists()


def test_identical_content_across_projects_mirrors_once(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "shared_note.md", "same\n")
    make_source(source_root, "proj-beta", "shared_note.md", "same\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["synced"] == ["shared_note.md"]
    assert not report["conflicts"]
    assert (corpus / "shared_note.md").read_text(encoding="utf-8") == "same\n"


def test_conflicting_content_is_not_written(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "shared_note.md", "alpha version\n")
    make_source(source_root, "proj-beta", "shared_note.md", "beta version\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus)

    assert report["conflicts"] == ["shared_note.md"]
    assert report["synced"] == [] and report["updated"] == []
    assert not (corpus / "shared_note.md").exists()


def test_dry_run_writes_nothing(tmp_path, gate):
    sanitize, patterns = gate
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "process_release.md", "content\n")

    report = mirror.sync_once(sanitize, patterns, source_root, corpus, dry_run=True)

    assert report["synced"] == ["process_release.md"]
    assert not corpus.exists() or not list(corpus.glob("*.md"))


def test_main_end_to_end(tmp_path, gate, capsys):
    source_root = tmp_path / "projects"
    corpus = tmp_path / "corpus"
    make_source(source_root, "proj-alpha", "process_release.md", "body\n")
    make_source(source_root, "proj-beta", "leak.md", "api_key=xyz\n")

    exit_code = mirror.main(
        [
            "--source-root", str(source_root),
            "--corpus", str(corpus),
            "--memory-service-dir", str(mirror.DEFAULT_MEMORY_SERVICE_DIR),
            "--json",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    report = __import__("json").loads(out)
    assert report["synced"] == ["process_release.md"]
    assert len(report["blocked"]) == 1
    assert (corpus / "process_release.md").exists()
    assert not (corpus / "leak.md").exists()


if __name__ == "__main__":
    # Standalone entry point: run this file through pytest (repo convention,
    # see requirements-dev.txt) and mirror pytest's exit status.
    sys.exit(pytest.main([__file__, "-v"]))
