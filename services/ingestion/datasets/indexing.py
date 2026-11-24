"""PR 4: shared Knowledge -> Chunk -> Embed -> Lexical/Vector orchestration.

This is the one implementation of the integration sequence.  The existing
``/ingest`` endpoint and the dataset connector both call it, which is why the
Knowledge revision sequence is a *parameter*: ``/ingest`` keeps its historical
``time.time_ns()`` value, byte-compatible with its earlier behaviour, while the
dataset path passes a deterministic ``high_sequence + 1`` so a re-run can be a
no-op instead of a new revision.

Nothing here hard-codes a host or port.  Endpoints come from the existing
``POC_CLAUDE_ENSEMBLE_*_URL`` environment variables through
:class:`ingestion.datasets.config.Config`.

Idempotency has two layers:

1. Index is the revision gate.  If the source's current ``revision_id`` already
   equals the one we would publish, nothing is written.
2. A local, rebuildable completion journal records which revisions finished the
   downstream stages.  If Knowledge accepted a revision but chunk/embed/index
   failed, the next run re-runs *only* those stages (Knowledge would call the
   identical revision a duplicate), so a partial failure heals instead of
   leaving an unsearchable record.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .normalize import NormalizedRecord, sha256_text

INDEXING_VERSION = 1
#: Knowledge and Chunking both reject a request body over 8 MiB.
MAX_DOWNSTREAM_BODY_BYTES = 8 * 1024 * 1024

PROFILES = {
    "chunking": "window-v1@1",
    "embedding": "nomic-embed-text-v1.5@fp32",
    "vector": "knowledge_chunks@1",
}
CHUNK_PROFILE = "window-v1"
EMBED_BATCH = 64


class IndexingError(ValueError):
    """A downstream capability is unavailable or rejected the request."""

    def __init__(self, message: str, status: int = 502, code: str = "downstream_failure") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Endpoints:
    knowledge: str
    chunking: str
    embedding: str
    lexical: str
    vector: str
    index: str

    @classmethod
    def from_config(cls, config: Config) -> "Endpoints":
        return cls(
            knowledge=config.knowledge_url,
            chunking=config.chunking_url,
            embedding=config.embedding_url,
            lexical=config.lexical_url,
            vector=config.vector_url,
            index=config.index_url,
        )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _encode(payload: dict[str, Any]) -> bytes:
    # ``ensure_ascii=False`` keeps multibyte content at its real size, so a
    # shard bounded in bytes cannot expand past a service's body limit.
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def request_json(
    url: str,
    *,
    method: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """One JSON request, preserving the HTTP status.

    Public because the PR 5 graph client needs to tell an immutable-store
    conflict (HTTP 400) apart from a transport failure.
    """
    data = _encode(payload) if payload is not None else None
    if data is not None and len(data) > MAX_DOWNSTREAM_BODY_BYTES:
        raise IndexingError(
            "request body exceeds the 8 MiB downstream limit", 413, "payload_too_large"
        )
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        with exc:
            status = exc.code
            body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = {}
        return status, parsed if isinstance(parsed, dict) else {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise IndexingError(
            "a required ingestion capability is unavailable", 503, "dependency_unavailable"
        ) from exc
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexingError(
            "a required ingestion capability returned invalid JSON", 502, "invalid_downstream_response"
        ) from exc
    if not isinstance(parsed, dict):
        raise IndexingError(
            "a required ingestion capability returned invalid JSON", 502, "invalid_downstream_response"
        )
    return status, parsed


def post_json(base: str, path: str, payload: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    """POST JSON and require the services' ``{"ok": true, ...}`` envelope."""
    status, result = request_json(
        base.rstrip("/") + path, method="POST", payload=payload, timeout=timeout
    )
    if status >= 400:
        raise IndexingError(f"downstream service returned HTTP {status}", 502, "downstream_failure")
    if result.get("ok") is not True:
        raise IndexingError(
            "a downstream ingestion capability rejected the request", 502, "downstream_failure"
        )
    return result


def get_json(base: str, path: str, timeout: float = 10) -> tuple[int, dict[str, Any]]:
    """GET JSON, preserving the status so a caller can treat 404 as "absent"."""
    return request_json(base.rstrip("/") + path, method="GET", payload=None, timeout=timeout)


def source_state(index_base: str, source_id: str, *, timeout: float = 10) -> dict[str, Any] | None:
    """Current Index state for one source, or ``None`` when it is unknown."""
    status, payload = get_json(
        index_base, "/sources/" + urllib.parse.quote(source_id, safe=""), timeout=timeout
    )
    if status == 404:
        return None
    if status >= 400 or payload.get("ok") is not True:
        raise IndexingError(
            "the index service could not report revision state", 502, "index_unavailable"
        )
    source = payload.get("source")
    if not isinstance(source, dict):
        raise IndexingError(
            "the index service returned an invalid source state", 502, "invalid_downstream_response"
        )
    return source


# ---------------------------------------------------------------------------
# Knowledge and the retrieval indexes
# ---------------------------------------------------------------------------


def submit_record(
    knowledge_base: str,
    *,
    source_id: str,
    revision_sequence: int,
    revision_id: str,
    title: str,
    content: str,
    origin: str,
    locator: str,
    profiles: dict[str, str] | None = None,
    timeout: float = 60,
    post=None,
) -> dict[str, Any]:
    """Store one revision in Knowledge (which consults Index itself)."""
    post = post or post_json
    return post(
        knowledge_base,
        "/records",
        {
            "source_id": source_id,
            "revision_sequence": revision_sequence,
            "revision_id": revision_id,
            "title": title,
            "content": content,
            "provenance": {"origin": origin, "locator": locator},
            "profiles": profiles or PROFILES,
        },
        timeout=timeout,
    )


def index_text(
    *,
    endpoints: Endpoints,
    source_id: str,
    title: str,
    origin: str,
    url: str,
    text: str,
    profiles: dict[str, str] | None = None,
    chunk_profile: str = CHUNK_PROFILE,
    media_type: str = "text/plain",
    timeout: float = 120,
    post=None,
) -> dict[str, Any]:
    """Chunk, embed, and write the lexical and vector indexes for one text.

    Returns ``{"chunks": n, "vectors": n, "content_sha256": ...}``.
    """
    post = post or post_json
    chunked = post(
        endpoints.chunking,
        "/chunk",
        {
            "profile": chunk_profile,
            "source_id": source_id,
            "media_type": media_type,
            "text": text,
        },
        timeout=timeout,
    )
    chunks = chunked.get("chunks")
    if not isinstance(chunks, list):
        raise IndexingError(
            "chunking service returned an invalid result", 502, "invalid_downstream_response"
        )
    content_sha = sha256_text(text)
    stored = 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        texts = [item["text"] for item in batch]
        embedded = post(
            endpoints.embedding, "/embeddings", {"texts": texts, "mode": "document"}, timeout=timeout
        )
        vectors = embedded.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise IndexingError(
                "embedding service returned an invalid vector count",
                502,
                "invalid_downstream_response",
            )
        rows = []
        ids = []
        metadatas = []
        for item in batch:
            chunk_id = item["chunk_id"]
            vector_id = hashlib.sha256(f"{chunk_id}|{content_sha}".encode("utf-8")).hexdigest()
            meta = {
                "kind": "knowledge_chunk",
                "source": origin,
                "url": url,
                "title": title,
                "chunk": item.get("index"),
                "chunks": len(chunks),
                "chunk_id": chunk_id,
                "content_sha256": content_sha,
            }
            ids.append(vector_id)
            metadatas.append(meta)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "vector_id": vector_id,
                    "url": url,
                    "source": origin,
                    "title": title,
                    "text": item["text"],
                    "chunk_index": item.get("index"),
                    "chunk_count": len(chunks),
                    "content_sha256": content_sha,
                }
            )
        post(endpoints.lexical, "/knowledge/documents", {"chunks": rows}, timeout=timeout)
        post(
            endpoints.vector,
            "/collections/knowledge_chunks/upsert",
            {
                "ids": ids,
                "embeddings": vectors,
                "documents": texts,
                "metadatas": metadatas,
            },
            timeout=timeout,
        )
        stored += len(batch)
    return {"chunks": len(chunks), "vectors": stored, "content_sha256": content_sha}


def delete_superseded(
    vector_base: str, *, url: str, keep_sha256: str, timeout: float = 60, post=None
) -> dict[str, Any]:
    """Remove other revisions of one indexed source, once the new one is stored."""
    post = post or post_json
    return post(
        vector_base,
        "/collections/knowledge_chunks/delete-superseded",
        {"url": url, "keep_sha256": keep_sha256},
        timeout=timeout,
    )


def submit_records_batch(
    knowledge_base: str,
    *,
    records: list[dict[str, Any]],
    timeout: float = 120,
    post=None,
) -> list[dict[str, Any]]:
    """Store many revisions in Knowledge in one request/transaction.

    Each record follows the :func:`submit_record` payload shape.  Returns
    the per-record decisions in input order.
    """
    post = post or post_json
    body = post(knowledge_base, "/records/batch", {"records": records}, timeout=timeout)
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(records):
        raise IndexingError(
            "knowledge batch returned an invalid result count", 502, "invalid_downstream_response"
        )
    return results


def index_texts_batch(
    *,
    endpoints: Endpoints,
    sources: list[dict[str, Any]],
    profiles: dict[str, str] | None = None,
    chunk_profile: str = CHUNK_PROFILE,
    media_type: str = "text/plain",
    timeout: float = 300,
    post=None,
) -> list[dict[str, Any]]:
    """Chunk, embed, and index MANY texts with cross-item embedding batches.

    Same per-item semantics as :func:`index_text` (identical chunk metadata,
    vector ids, lexical rows, and superseded-revision cleanup), but the
    embedding calls span items: all chunks from the whole batch are embedded
    in ``EMBED_BATCH``-sized requests instead of one tiny request per item.
    Small scraped items average ~2 chunks, so per-item embedding calls spend
    most of their time on request overhead while the GPU idles; batching
    across items is what makes bulk ingestion GPU-bound instead of
    HTTP-bound.

    Returns one ``{"chunks", "vectors", "content_sha256"}`` dict per source,
    aligned with the input order.  A failure indexing one source raises
    ``IndexingError`` naming it; sources before it are already stored (the
    pipeline is idempotent, so a retry re-ingests cleanly).
    """
    post = post or post_json

    # Phase 1: chunk every text (per-item REST; chunking is cheap and the
    # service is stateless per call).
    chunked_all: list[list[dict[str, Any]]] = []
    for source in sources:
        chunked = post(
            endpoints.chunking,
            "/chunk",
            {
                "profile": chunk_profile,
                "source_id": source["source_id"],
                "media_type": media_type,
                "text": source["text"],
            },
            timeout=timeout,
        )
        chunks = chunked.get("chunks")
        if not isinstance(chunks, list):
            raise IndexingError(
                f"chunking service returned an invalid result for {source['source_id']!r}",
                502,
                "invalid_downstream_response",
            )
        chunked_all.append(chunks)

    # Phase 2: embed every chunk of every source in cross-item batches.
    flat: list[tuple[int, dict[str, Any]]] = []
    for source_idx, chunks in enumerate(chunked_all):
        for chunk in chunks:
            flat.append((source_idx, chunk))
    vectors_flat: list[list[float] | None] = [None] * len(flat)
    for start in range(0, len(flat), EMBED_BATCH):
        batch = flat[start : start + EMBED_BATCH]
        embedded = post(
            endpoints.embedding,
            "/embeddings",
            {"texts": [chunk["text"] for _, chunk in batch], "mode": "document"},
            timeout=timeout,
        )
        vectors = embedded.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise IndexingError(
                "embedding service returned an invalid vector count",
                502,
                "invalid_downstream_response",
            )
        for offset, vector in enumerate(vectors):
            vectors_flat[start + offset] = vector

    # Phase 3: aggregate the lexical and vector writes across the WHOLE
    # batch (both APIs accept lists), then per-source superseded cleanup.
    results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_vectors: list[list[float]] = []
    cursor = 0
    for source_idx, source in enumerate(sources):
        chunks = chunked_all[source_idx]
        count = len(chunks)
        batch_vectors = vectors_flat[cursor : cursor + count]
        cursor += count
        content_sha = sha256_text(source["text"])
        for item, vector in zip(chunks, batch_vectors):
            chunk_id = item["chunk_id"]
            vector_id = hashlib.sha256(f"{chunk_id}|{content_sha}".encode("utf-8")).hexdigest()
            meta = {
                "kind": "knowledge_chunk",
                "source": source["origin"],
                "url": source["source_id"],
                "title": source["title"],
                "chunk": item.get("index"),
                "chunks": count,
                "chunk_id": chunk_id,
                "content_sha256": content_sha,
            }
            all_ids.append(vector_id)
            all_metadatas.append(meta)
            all_texts.append(item["text"])
            all_vectors.append(vector)
            all_rows.append(
                {
                    "chunk_id": chunk_id,
                    "vector_id": vector_id,
                    "url": source["source_id"],
                    "source": source["origin"],
                    "title": source["title"],
                    "text": item["text"],
                    "chunk_index": item.get("index"),
                    "chunk_count": count,
                    "content_sha256": content_sha,
                }
            )
        results.append({"chunks": count, "vectors": count, "content_sha256": content_sha})
    if all_rows:
        post(endpoints.lexical, "/knowledge/documents", {"chunks": all_rows}, timeout=timeout)
        post(
            endpoints.vector,
            "/collections/knowledge_chunks/upsert",
            {
                "ids": all_ids,
                "embeddings": all_vectors,
                "documents": all_texts,
                "metadatas": all_metadatas,
            },
            timeout=timeout,
        )
    for source_idx, source in enumerate(sources):
        post(
            endpoints.vector,
            "/collections/knowledge_chunks/delete-superseded",
            {"url": source["source_id"], "keep_sha256": results[source_idx]["content_sha256"]},
            timeout=timeout,
        )
    return results


# ---------------------------------------------------------------------------
# Completion journal
# ---------------------------------------------------------------------------


class CompletionJournal:
    """Rebuildable record of revisions whose downstream indexing completed.

    One small file per revision, named by the revision id.  A missing journal
    never loses data: re-running the downstream stages is idempotent.
    """

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def path_for(self, revision_id: str) -> Path:
        return self.root / f"{revision_id}.json"

    def has(self, revision_id: str) -> bool:
        if not self.enabled:
            return False
        try:
            info = self.path_for(revision_id).lstat()
        except OSError:
            return False
        return info.st_size > 0

    def count(self) -> int:
        try:
            return sum(
                1
                for path in self.root.glob("*.json")
                if not path.name.startswith(".")
            )
        except OSError:
            return 0

    def record(self, revision_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path_for(revision_id)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Per-record integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordOutcome:
    """What happened to one canonical record."""

    source_id: str
    entry_id: str
    kind: str
    content_sha256: str
    revision_id: str
    status: str
    locator: str | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    reason_code: str | None = None
    detail: str | None = None
    revision_sequence: int | None = None
    knowledge_decision: str | None = None
    chunks: int = 0
    vectors: int = 0

    @property
    def indexed(self) -> bool:
        return self.status in {"indexed", "unchanged"}

    def to_artifact(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "content_sha256": self.content_sha256,
            "revision_id": self.revision_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:300] if self.detail else None,
            "locator": self.locator,
            "revision_sequence": self.revision_sequence,
            "knowledge_decision": self.knowledge_decision,
            "chunks": self.chunks,
            "vectors": self.vectors,
        }


def _high_sequence(state: dict[str, Any] | None) -> int:
    if state is None:
        return 1
    value = state.get("high_sequence")
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndexingError(
            "the index service returned an invalid source state", 502, "invalid_downstream_response"
        )
    return value + 1


def _outcome(
    record: NormalizedRecord,
    status: str,
    *,
    reason_code: str | None = None,
    detail: str | None = None,
    revision_sequence: int | None = None,
    knowledge_decision: str | None = None,
    chunks: int = 0,
    vectors: int = 0,
) -> RecordOutcome:
    return RecordOutcome(
        source_id=record.source_id,
        entry_id=record.entry_id,
        kind=record.kind,
        content_sha256=record.content_sha256,
        revision_id=record.revision_id,
        status=status,
        locator=record.locator,
        shard_index=record.shard_index,
        shard_count=record.shard_count,
        reason_code=reason_code,
        detail=detail,
        revision_sequence=revision_sequence,
        knowledge_decision=knowledge_decision,
        chunks=chunks,
        vectors=vectors,
    )


def integrate_record(
    record: NormalizedRecord,
    *,
    endpoints: Endpoints,
    journal: CompletionJournal,
    profiles: dict[str, str] | None = None,
    timeout: float = 120,
    post=None,
) -> RecordOutcome:
    """Publish one record, skipping it when its content is already indexed."""
    post = post or post_json
    active_profiles = profiles or PROFILES

    def complete_existing(
        sequence: int | None, *, recovered: bool
    ) -> RecordOutcome:
        """Re-run only the downstream stages for an already-stored revision."""
        indexed = index_text(
            endpoints=endpoints,
            source_id=record.source_id,
            title=record.title,
            origin=record.origin,
            url=record.source_id,
            text=record.content,
            profiles=active_profiles,
            timeout=timeout,
            post=post,
        )
        journal.record(
            record.revision_id,
            {
                "source_id": record.source_id,
                "revision_id": record.revision_id,
                "revision_sequence": sequence,
                "chunks": indexed["chunks"],
                "vectors": indexed["vectors"],
                "recovered": recovered,
            },
        )
        return _outcome(
            record,
            "indexed",
            reason_code="indexing_recovered",
            revision_sequence=sequence,
            knowledge_decision="duplicate",
            chunks=indexed["chunks"],
            vectors=indexed["vectors"],
        )

    try:
        state = source_state(endpoints.index, record.source_id)
        if state is not None and state.get("tombstoned"):
            return _outcome(record, "skipped", reason_code="source_tombstoned")
        if state is not None and state.get("revision_id") == record.revision_id:
            if journal.has(record.revision_id):
                return _outcome(
                    record,
                    "unchanged",
                    reason_code="content_unchanged",
                    revision_sequence=state.get("high_sequence"),
                )
            # Knowledge already holds this revision but the downstream stages
            # did not finish: complete them without a new Knowledge revision.
            return complete_existing(state.get("high_sequence"), recovered=True)
        supersedes = state is not None and state.get("revision_id") != record.revision_id
        sequence = _high_sequence(state)
        knowledge = submit_record(
            endpoints.knowledge,
            source_id=record.source_id,
            revision_sequence=sequence,
            revision_id=record.revision_id,
            title=record.title,
            content=record.content,
            origin=record.origin,
            locator=record.locator,
            profiles=active_profiles,
            timeout=timeout,
            post=post,
        )
        decision = str(knowledge.get("decision") or "")
        if decision == "conflict":
            # One re-read and retry, as the design specifies.
            state = source_state(endpoints.index, record.source_id)
            if state is not None and state.get("revision_id") == record.revision_id:
                decision = "duplicate"
                sequence = state.get("high_sequence")
                supersedes = False
            else:
                supersedes = state is not None and (
                    state.get("revision_id") != record.revision_id
                )
                sequence = _high_sequence(state)
                knowledge = submit_record(
                    endpoints.knowledge,
                    source_id=record.source_id,
                    revision_sequence=sequence,
                    revision_id=record.revision_id,
                    title=record.title,
                    content=record.content,
                    origin=record.origin,
                    locator=record.locator,
                    profiles=active_profiles,
                    timeout=timeout,
                    post=post,
                )
                decision = str(knowledge.get("decision") or "")
            if decision == "conflict":
                return _outcome(
                    record,
                    "failed",
                    reason_code="index_conflict",
                    revision_sequence=sequence,
                    knowledge_decision=decision,
                )
        if decision == "duplicate":
            if journal.has(record.revision_id):
                return _outcome(
                    record,
                    "unchanged",
                    reason_code="content_unchanged",
                    revision_sequence=sequence,
                    knowledge_decision=decision,
                )
            return complete_existing(sequence, recovered=True)
        if decision != "accepted":
            return _outcome(
                record,
                "skipped",
                reason_code="source_tombstoned"
                if decision == "tombstoned"
                else "knowledge_rejected",
                revision_sequence=sequence,
                knowledge_decision=decision,
            )
        indexed = index_text(
            endpoints=endpoints,
            source_id=record.source_id,
            title=record.title,
            origin=record.origin,
            url=record.source_id,
            text=record.content,
            profiles=active_profiles,
            timeout=timeout,
            post=post,
        )
        if supersedes:
            # Only a record that replaces an older revision can leave stale
            # vectors behind.  Skipping this for new sources also avoids the
            # vector service's full-collection reload that this call triggers.
            delete_superseded(
                endpoints.vector,
                url=record.source_id,
                keep_sha256=indexed["content_sha256"],
                post=post,
            )
        journal.record(
            record.revision_id,
            {
                "source_id": record.source_id,
                "revision_id": record.revision_id,
                "revision_sequence": sequence,
                "chunks": indexed["chunks"],
                "vectors": indexed["vectors"],
            },
        )
        return _outcome(
            record,
            "indexed",
            revision_sequence=sequence,
            knowledge_decision=decision,
            chunks=indexed["chunks"],
            vectors=indexed["vectors"],
        )
    except IndexingError as exc:
        return _outcome(record, "failed", reason_code=exc.code, detail=str(exc))


def integrate_records(
    records: Iterable[NormalizedRecord],
    *,
    endpoints: Endpoints,
    journal: CompletionJournal,
    profiles: dict[str, str] | None = None,
    timeout: float = 120,
    post=None,
) -> list[RecordOutcome]:
    return [
        integrate_record(
            record,
            endpoints=endpoints,
            journal=journal,
            profiles=profiles,
            timeout=timeout,
            post=post,
        )
        for record in records
    ]


def tombstone_surplus_shards(
    *,
    entry_id: str,
    source_url: str,
    shard_count: int,
    endpoints: Endpoints,
    knowledge_sequence: int,
) -> list[str]:
    """Tombstone contiguous shards beyond a shrunken shard family.

    Shard ids are contiguous from zero, so the first missing index ends the old
    family.  Bounded: at most 64 surplus shards are considered.
    """
    from .normalize import content_source_id

    tombstoned: list[str] = []
    for shard_index in range(shard_count, shard_count + 64):
        source_id = content_source_id(entry_id, source_url, shard_index)
        try:
            state = source_state(endpoints.index, source_id)
        except IndexingError:
            break
        if state is None or state.get("tombstoned"):
            break
        sequence = _high_sequence(state)
        try:
            post_json(
                endpoints.knowledge,
                "/records/" + urllib.parse.quote(source_id, safe="") + "/tombstone",
                {"revision_sequence": max(sequence, knowledge_sequence)},
            )
        except IndexingError:
            break
        tombstoned.append(source_id)
    return tombstoned
