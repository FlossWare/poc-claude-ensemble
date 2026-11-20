"""PR 4 integration: revision gate, idempotency and explicit outcomes."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.datasets import indexing, normalize

from ingestion.datasets.tests.helpers import FakeIntegrationBackend

PROFILES = {"chunking": "window-v1@1", "embedding": "e@1", "vector": "v@1"}


def _record(*, text: str = "alpha beta gamma", entry: str = "apd:Test/example.yml") -> normalize.NormalizedRecord:
    digest = normalize.sha256_text(text)
    source_id = f"{entry}/src/0123456789abcdef/shard/0000"
    return normalize.NormalizedRecord(
        source_id=source_id,
        title="Example",
        content=text,
        content_sha256=digest,
        origin="apd-core@" + "c" * 40,
        locator="https://data.example.org/table.csv",
        revision_id=normalize.revision_id_for(
            source_id=source_id,
            content_sha256=digest,
            normalized_sha256=digest,
            adapter_version=2,
            license_policy="index_content",
            profiles=PROFILES,
        ),
        entry_id=entry,
        kind="content",
        normalized_sha256=digest,
        adapter_version=2,
        license_policy="index_content",
        shard_index=0,
        shard_count=1,
        record_count=1,
    )


def _endpoints() -> indexing.Endpoints:
    return indexing.Endpoints(
        knowledge="http://knowledge",
        chunking="http://chunking",
        embedding="http://embedding",
        lexical="http://lexical",
        vector="http://vector",
        index="http://index",
    )


def _journal(tmp_path: Path) -> indexing.CompletionJournal:
    return indexing.CompletionJournal(tmp_path / "journal")


def _run(monkeypatch, backend, record, journal):
    monkeypatch.setattr(indexing, "source_state", backend.state)
    monkeypatch.setattr(indexing, "post_json", backend.post)
    return indexing.integrate_record(
        record, endpoints=_endpoints(), journal=journal, profiles=PROFILES
    )


def test_new_record_runs_the_full_sequence(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    journal = _journal(tmp_path)
    record = _record()
    outcome = _run(monkeypatch, backend, record, journal)

    assert outcome.status == "indexed"
    assert outcome.reason_code is None
    assert outcome.revision_sequence == 1
    assert outcome.knowledge_decision == "accepted"
    assert outcome.chunks >= 1 and outcome.vectors == outcome.chunks
    paths = backend.paths()
    assert paths[0] == "/records"
    for expected in (
        "/chunk",
        "/embeddings",
        "/knowledge/documents",
        "/collections/knowledge_chunks/upsert",
    ):
        assert expected in paths
    # A brand-new source has no older revision to supersede, and avoiding the
    # call also avoids the vector service's full-collection reload.
    assert "/collections/knowledge_chunks/delete-superseded" not in paths
    # Knowledge got the canonical envelope and the deterministic sequence.
    knowledge = backend.knowledge[record.source_id]
    assert knowledge["revision_id"] == record.revision_id
    assert knowledge["revision_sequence"] == 1
    assert knowledge["provenance"] == {
        "origin": record.origin,
        "locator": record.locator,
    }
    assert journal.has(record.revision_id)


def test_unchanged_record_is_skipped_without_any_call(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    journal = _journal(tmp_path)
    record = _record()
    assert _run(monkeypatch, backend, record, journal).status == "indexed"
    backend.calls.clear()

    outcome = _run(monkeypatch, backend, record, journal)
    assert outcome.status == "unchanged"
    assert outcome.reason_code == "content_unchanged"
    assert backend.calls == []


def test_matching_revision_without_journal_recovers_downstream_only(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    journal = _journal(tmp_path)
    record = _record()
    _run(monkeypatch, backend, record, journal)
    # Simulate a lost local journal: Index still holds the revision.
    journal.path_for(record.revision_id).unlink()
    backend.calls.clear()

    outcome = _run(monkeypatch, backend, record, journal)
    assert outcome.status == "indexed"
    assert outcome.reason_code == "indexing_recovered"
    assert "/records" not in backend.paths()
    assert "/chunk" in backend.paths()
    assert journal.has(record.revision_id)


def test_conflict_is_reread_and_retried_once(tmp_path, monkeypatch):
    class ConflictOnce(FakeIntegrationBackend):
        conflicts = 1

        def _records(self, payload):
            if self.conflicts:
                self.conflicts -= 1
                source_id = payload["source_id"]
                self.index[source_id] = {
                    "high_sequence": int(payload["revision_sequence"]),
                    "revision_id": "concurrent-writer",
                    "tombstoned": False,
                }
                return {"ok": True, "decision": "conflict", "stored": False}
            return super()._records(payload)

    backend = ConflictOnce()
    record = _record()
    outcome = _run(monkeypatch, backend, record, _journal(tmp_path))
    assert outcome.status == "indexed"
    assert outcome.revision_sequence == 2
    assert backend.paths().count("/records") == 2


def test_persistent_conflict_is_explicit(tmp_path, monkeypatch):
    class AlwaysConflict(FakeIntegrationBackend):
        def _records(self, payload):
            return {"ok": True, "decision": "conflict", "stored": False}

    backend = AlwaysConflict()
    backend.index[_record().source_id] = {
        "high_sequence": 1,
        "revision_id": "other",
        "tombstoned": False,
    }
    outcome = _run(monkeypatch, backend, _record(), _journal(tmp_path))
    assert outcome.status == "failed"
    assert outcome.reason_code == "index_conflict"


def test_tombstoned_source_is_skipped_with_a_reason(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    record = _record()
    backend.index[record.source_id] = {
        "high_sequence": 3,
        "revision_id": "old",
        "tombstoned": True,
    }
    outcome = _run(monkeypatch, backend, record, _journal(tmp_path))
    assert outcome.status == "skipped"
    assert outcome.reason_code == "source_tombstoned"
    assert backend.calls == []


def test_downstream_failure_is_explicit_and_the_next_run_recovers(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    journal = _journal(tmp_path)
    record = _record()
    backend.fail_paths = {"/chunk"}

    failed = _run(monkeypatch, backend, record, journal)
    assert failed.status == "failed"
    assert failed.reason_code == "dependency_unavailable"
    assert not journal.has(record.revision_id)
    # Knowledge accepted the revision before the downstream failure.
    assert record.source_id in backend.index

    backend.fail_paths = set()
    backend.calls.clear()
    recovered = _run(monkeypatch, backend, record, journal)
    assert recovered.status == "indexed"
    assert recovered.reason_code == "indexing_recovered"
    assert "/records" not in backend.paths()
    assert journal.has(record.revision_id)


def test_rejected_knowledge_decision_never_reaches_chunking(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    record = _record()
    backend.index[record.source_id] = {
        "high_sequence": 9,
        "revision_id": "newer",
        "tombstoned": False,
    }
    # The client computes sequence 10 and the fake accepts it, so force a
    # rejection the way a tombstone race would.
    original = backend._records

    def reject(payload):
        return {"ok": True, "decision": "superseded", "stored": False}

    backend._records = reject  # type: ignore[method-assign]
    outcome = _run(monkeypatch, backend, record, _journal(tmp_path))
    backend._records = original  # type: ignore[method-assign]
    assert outcome.status == "skipped"
    assert outcome.reason_code == "knowledge_rejected"
    assert "/chunk" not in backend.paths()


def test_index_text_batches_embeddings_and_writes_each_batch(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    backend.WINDOW = 10
    backend.OVERLAP = 0
    monkeypatch.setattr(indexing, "source_state", backend.state)
    monkeypatch.setattr(indexing, "post_json", backend.post)
    result = indexing.index_text(
        endpoints=_endpoints(),
        source_id="apd:Test/example.yml/src/x/shard/0000",
        title="Example",
        origin="apd-core@" + "c" * 40,
        url="apd:Test/example.yml/src/x/shard/0000",
        text="x" * 700,
        profiles=PROFILES,
    )
    assert result["chunks"] == 70
    assert result["vectors"] == 70
    assert backend.paths().count("/embeddings") == 2
    assert backend.paths().count("/collections/knowledge_chunks/upsert") == 2


def test_source_state_404_is_absent_and_other_errors_raise(monkeypatch):
    calls = []

    def fake_get(base, path, timeout=10):
        calls.append(path)
        return 404, {"ok": False}

    monkeypatch.setattr(indexing, "get_json", fake_get)
    assert indexing.source_state("http://index", "apd:Test/a.csv") is None
    assert calls == ["/sources/apd%3ATest%2Fa.csv"]

    def failing_get(base, path, timeout=10):
        return 503, {"ok": False}

    monkeypatch.setattr(indexing, "get_json", failing_get)
    with pytest.raises(indexing.IndexingError) as caught:
        indexing.source_state("http://index", "apd:Test/a.csv")
    assert caught.value.code == "index_unavailable"


def test_conflict_that_resolves_to_duplicate_completes_downstream(tmp_path, monkeypatch):
    """A concurrent writer won the revision; we still finish the indexes."""
    backend = FakeIntegrationBackend()
    record = _record()
    backend._records = lambda payload: {"ok": True, "decision": "conflict", "stored": False}
    states = [
        {"source_id": record.source_id, "high_sequence": 1, "revision_id": "other", "tombstoned": False},
        {"source_id": record.source_id, "high_sequence": 1, "revision_id": record.revision_id, "tombstoned": False},
    ]

    def scripted_state(base, source_id, timeout=10):
        return states.pop(0) if states else None

    monkeypatch.setattr(indexing, "source_state", scripted_state)
    monkeypatch.setattr(indexing, "post_json", backend.post)
    outcome = indexing.integrate_record(
        record, endpoints=_endpoints(), journal=_journal(tmp_path), profiles=PROFILES
    )
    assert outcome.status == "indexed"
    assert outcome.reason_code == "indexing_recovered"
    assert outcome.knowledge_decision == "duplicate"
    assert backend.paths().count("/records") == 1
    assert "/chunk" in backend.paths()


def test_superseding_revision_removes_stale_vectors(tmp_path, monkeypatch):
    backend = FakeIntegrationBackend()
    journal = _journal(tmp_path)
    record = _record()
    backend.index[record.source_id] = {
        "high_sequence": 1,
        "revision_id": "older-revision",
        "tombstoned": False,
    }
    backend.vectors["stale"] = {
        "url": record.source_id,
        "content_sha256": "f" * 64,
    }
    outcome = _run(monkeypatch, backend, record, journal)
    assert outcome.status == "indexed"
    assert outcome.revision_sequence == 2
    assert "/collections/knowledge_chunks/delete-superseded" in backend.paths()
    assert "stale" not in backend.vectors
    assert backend.vectors, "the new revision must still be stored"
