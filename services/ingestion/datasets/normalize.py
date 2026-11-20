"""PR 4: canonical record envelope, text projection and deterministic sharding.

A *record* is the unit the Knowledge service stores and the retrieval stack
indexes.  Every record carries the canonical fields the design requires::

    source_id, title, content, content_sha256, origin, locator, revision_id

Two record families exist:

``metadata``
    One per catalog entry, ``source_id == entry_id``.  Its content is a stable
    text projection of the normalized catalog metadata, so license-unknown
    entries are still discoverable by title, description and keywords.

``content``
    One per shard of one acquired file (or archive member) whose license policy
    is ``index_content``.  ``source_id`` is
    ``{entry_id}/src/{sha256(canonical_source_url)[:16]}/shard/{nnnn}`` so the
    same source location keeps its identity when its content changes, and the
    change becomes a new revision rather than a new source.

Everything here is deterministic: the same catalog row, acquisition artifact,
adapter version and configuration produce byte-identical records, which is what
makes the content-hash idempotency check in :mod:`ingestion.datasets.indexing`
safe.

Untrusted content never reaches a log or a manifest through this module: the
record's ``content`` stays in memory and only hashes and counts are reported.
"""
from __future__ import annotations

import hashlib
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from . import adapters, archive, model
from .config import Config

NORMALIZE_VERSION = 1
SHARD_WIDTH = 4
MAX_SOURCE_ID_CHARS = 512
#: Knowledge rejects content over 4,000,000 characters; stay below it.
MAX_KNOWLEDGE_CHARS = 3_900_000
#: Longest title the existing ingestion contract accepts.
MAX_TITLE_CHARS = 300


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scalar_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return model.canonical_json(value)


def project_data(kind: str, data: object) -> str:
    """Project one adapter record onto inert text, deterministically.

    Dict records use ``field: value`` lines in their stable key order; lists
    become ``- value`` items; scalars use canonical JSON; text lines use the
    line itself.  Nothing is interpreted and no value is coerced.
    """
    if kind == "text" and isinstance(data, dict):
        value = data.get("text")
        if isinstance(value, str):
            return value
    if isinstance(data, dict):
        return "\n".join(f"{key}: {_scalar_text(value)}" for key, value in data.items())
    if isinstance(data, list):
        return "\n".join(f"- {_scalar_text(item)}" for item in data)
    return _scalar_text(data)


def _add_line(lines: list[str], label: str, value: object) -> None:
    if value is None or value == "" or value == [] or value == {}:
        return
    # Defense in depth: the manifest already sanitizes URLs, but the
    # projection must never carry a query string into an indexed record.
    lines.append(f"{label}: {_scalar_text(model.sanitize_nested(value))}")


def metadata_text(row: dict[str, object]) -> str:
    """Stable text projection of one manifest row (no timestamps, no URLs' queries)."""
    lines: list[str] = []
    _add_line(lines, "entry_id", row.get("entry_id"))
    _add_line(lines, "title", row.get("title"))
    _add_line(lines, "category", row.get("category"))
    _add_line(lines, "homepage", row.get("homepage"))
    _add_line(lines, "description", row.get("description"))
    _add_line(lines, "keywords", row.get("keywords"))
    _add_line(lines, "tags", row.get("tags"))
    _add_line(lines, "version", row.get("version"))
    _add_line(lines, "issued_time", row.get("issued_time"))
    _add_line(lines, "temporal", row.get("temporal"))
    _add_line(lines, "spatial", row.get("spatial"))
    for person in row.get("publisher") or []:
        if isinstance(person, dict):
            label = person.get("name") or "unknown"
            web = person.get("web")
            _add_line(lines, "publisher", f"{label} ({web})" if web else label)
    for person in row.get("organization") or []:
        if isinstance(person, dict):
            _add_line(lines, "organization", person.get("name") or "unknown")
    for source in row.get("sources") or []:
        if not isinstance(source, dict):
            continue
        name = source.get("name")
        url = source.get("access_url")
        _add_line(lines, "source", f"{name} ({url})" if name and url else (url or name))
    license_block = row.get("license")
    if isinstance(license_block, dict):
        dataset = license_block.get("dataset")
        if isinstance(dataset, dict):
            _add_line(lines, "license_declared", dataset.get("declared_raw"))
            _add_line(lines, "license_family", dataset.get("family"))
            _add_line(lines, "license_policy", dataset.get("policy"))
            _add_line(lines, "license_review_required", dataset.get("review_required"))
    classes = row.get("classes")
    if isinstance(classes, dict):
        _add_line(lines, "format", classes.get("format"))
        _add_line(lines, "access", classes.get("access"))
    catalog = row.get("catalog")
    if isinstance(catalog, dict):
        _add_line(lines, "catalog_path", catalog.get("path"))
        _add_line(lines, "meta_url", catalog.get("meta_url"))
    extra = row.get("extra")
    if extra:
        _add_line(lines, "extra", extra)
    return "\n".join(lines).strip()


def _fit_bytes(value: str, max_bytes: int) -> str:
    """Longest prefix of ``value`` whose UTF-8 encoding fits ``max_bytes``."""
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(value[:middle].encode("utf-8")) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return value[:low]


def _truncate_chars(value: str, *, max_chars: int, max_bytes: int) -> tuple[str, bool]:
    """Cut a projection to both the character and the UTF-8 byte budget."""
    if len(value) <= max_chars and len(value.encode("utf-8")) <= max_bytes:
        return value, False
    return _fit_bytes(value[:max_chars], max_bytes), True


def canonical_source_url(uri: str, member: str | None = None) -> str:
    """The stable upstream address of one content source.

    Query strings, fragments and userinfo are stripped; an archive member is
    folded into the hash input, not into the persisted URL.
    """
    base = model.sanitize_url(uri) or uri
    return f"{base}#{member}" if member else base


def content_source_id(entry_id: str, source_url: str, shard_index: int) -> str:
    """Stable per-shard source identifier, bounded to Knowledge's 512 characters."""
    digest = sha256_text(source_url)
    shard = f"shard/{shard_index:0{SHARD_WIDTH}d}"
    identifier = f"{entry_id}/src/{digest[:16]}/{shard}"
    if len(identifier) > MAX_SOURCE_ID_CHARS:
        # The entry id is already close to the limit; keep the same stability
        # properties by hashing the entry id instead of embedding it.
        identifier = f"apd:src/{sha256_text(entry_id)[:16]}/src/{digest[:16]}/{shard}"
    return identifier


def _profile_key(profiles: dict[str, str]) -> dict[str, str]:
    return dict(sorted(profiles.items()))


def revision_id_for(
    *,
    source_id: str,
    content_sha256: str,
    normalized_sha256: str,
    adapter_version: int,
    license_policy: str | None,
    profiles: dict[str, str],
    metadata_sha256: str | None = None,
    source_sha256: str | None = None,
) -> str:
    """Deterministic revision identity: any input change makes a new revision."""
    return sha256_text(
        model.canonical_json(
            {
                "source_id": source_id,
                "content_sha256": content_sha256,
                "normalized_sha256": normalized_sha256,
                "metadata_sha256": metadata_sha256,
                "source_sha256": source_sha256,
                "adapter_version": adapter_version,
                "license_policy": license_policy,
                "profiles": _profile_key(profiles),
                "normalize_version": NORMALIZE_VERSION,
            }
        )
    )


@dataclass(frozen=True)
class NormalizedRecord:
    """One canonical record ready for Knowledge and the retrieval indexes."""

    source_id: str
    title: str
    content: str
    content_sha256: str
    origin: str
    locator: str
    revision_id: str
    entry_id: str
    kind: str
    normalized_sha256: str
    adapter_version: int
    license_policy: str | None
    metadata_sha256: str | None = None
    source_sha256: str | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    record_count: int = 0
    truncated: bool = False

    @property
    def characters(self) -> int:
        return len(self.content)

    def to_artifact(self) -> dict[str, object]:
        """Manifest-safe projection: hashes and counts only, never content."""
        return {
            "source_id": self.source_id,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "title": self.title,
            "content_sha256": self.content_sha256,
            "origin": self.origin,
            "locator": self.locator,
            "revision_id": self.revision_id,
            "normalized_sha256": self.normalized_sha256,
            "metadata_sha256": self.metadata_sha256,
            "source_sha256": self.source_sha256,
            "adapter_version": self.adapter_version,
            "license_policy": self.license_policy,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "record_count": self.record_count,
            "characters": self.characters,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class FileOutcome:
    """What happened to one acquired file (or archive member)."""

    uri: str
    member: str | None
    status: str
    adapter: str | None = None
    reason_code: str | None = None
    detail: str | None = None
    records: int = 0
    shards: int = 0
    truncated: bool = False

    def to_artifact(self) -> dict[str, object]:
        return {
            "uri": model.sanitize_url(self.uri) or self.uri,
            "member": self.member,
            "status": self.status,
            "adapter": self.adapter,
            "reason_code": self.reason_code,
            "detail": self.detail[:300] if self.detail else None,
            "records": self.records,
            "shards": self.shards,
            "truncated": self.truncated,
        }


@dataclass
class EntryNormalization:
    """The normalize-stage result for one catalog entry."""

    entry_id: str
    metadata: NormalizedRecord | None
    metadata_reason: str | None
    records: list[NormalizedRecord] = field(default_factory=list)
    files: list[FileOutcome] = field(default_factory=list)
    records_total: int = 0
    content_reason: str | None = None
    truncated: bool = False

    @property
    def shards(self) -> int:
        return len(self.records)


def license_policy(row: dict[str, object]) -> str | None:
    """The dataset license policy recorded by PR 1 (``index_content`` etc.)."""
    license_block = row.get("license")
    if not isinstance(license_block, dict):
        return None
    dataset = license_block.get("dataset")
    if not isinstance(dataset, dict):
        return None
    policy = dataset.get("policy")
    return policy if isinstance(policy, str) else None


def license_reason(row: dict[str, object]) -> str | None:
    """The reason code behind a non-indexable license policy."""
    license_block = row.get("license")
    if not isinstance(license_block, dict):
        return None
    dataset = license_block.get("dataset")
    if not isinstance(dataset, dict):
        return None
    reason = dataset.get("reason_code")
    return reason if isinstance(reason, str) else None


def metadata_record(
    row: dict[str, object],
    *,
    commit: str,
    profiles: dict[str, str],
    config: Config,
) -> tuple[NormalizedRecord | None, str | None]:
    """Build the one metadata record every entry receives.

    Returns ``(record, reason)``; ``reason`` is set only when no record could be
    built (an entry with no metadata at all).
    """
    entry_id = str(row.get("entry_id") or "")
    text = metadata_text(row)
    if not text.strip():
        return None, "empty_metadata"
    max_bytes = max(1024, config.max_shard_bytes)
    text, truncated = _truncate_chars(
        text, max_chars=min(config.max_shard_chars, MAX_KNOWLEDGE_CHARS), max_bytes=max_bytes
    )
    policy = license_policy(row)
    title = str(row.get("title") or entry_id)[:MAX_TITLE_CHARS]
    catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
    assert isinstance(catalog, dict)
    locator = (
        model.sanitize_url(row.get("homepage"))
        or model.sanitize_url(catalog.get("meta_url"))
        or entry_id
    )
    content_sha = sha256_text(text)
    metadata_sha = str(row.get("metadata_sha256") or "")
    record = NormalizedRecord(
        source_id=entry_id,
        title=title,
        content=text,
        content_sha256=content_sha,
        origin=f"apd-core@{commit}",
        locator=str(locator),
        revision_id=revision_id_for(
            source_id=entry_id,
            content_sha256=content_sha,
            normalized_sha256=metadata_sha,
            metadata_sha256=metadata_sha,
            adapter_version=adapters.ADAPTER_VERSION,
            license_policy=policy,
            profiles=profiles,
        ),
        entry_id=entry_id,
        kind="metadata",
        normalized_sha256=metadata_sha,
        adapter_version=adapters.ADAPTER_VERSION,
        license_policy=policy,
        metadata_sha256=metadata_sha,
        truncated=truncated,
    )
    return record, None


# ---------------------------------------------------------------------------
# Content shards
# ---------------------------------------------------------------------------


class _ShardAccumulator:
    """Accumulates projected record text into deterministic, bounded shards."""

    def __init__(self, *, max_chars: int, max_bytes: int) -> None:
        self.max_chars = max(1, max_chars)
        self.max_bytes = max(1, max_bytes)
        #: A single record may be split; each piece is bounded by four bytes per
        #: character, so this limit satisfies both budgets.
        self.piece_chars = min(self.max_chars, max(1, self.max_bytes // 4))
        self.current: list[str] = []
        self.chars = 0
        self.bytes = 0
        self.records = 0
        self.pending_records = 0
        self.pieces: list[tuple[str, int]] = []

    def add(self, text: str) -> None:
        self.records += 1
        if not text:
            self.pending_records += 1
            return
        offset = 0
        first = True
        while offset < len(text):
            piece = text[offset : offset + self.piece_chars]
            offset += self.piece_chars
            added_chars = len(piece) + (2 if self.current else 0)
            added_bytes = len(piece.encode("utf-8")) + (2 if self.current else 0)
            if self.current and (
                self.chars + added_chars > self.max_chars
                or self.bytes + added_bytes > self.max_bytes
            ):
                self.flush()
                added_chars = len(piece)
                added_bytes = len(piece.encode("utf-8"))
            if first:
                # A record belongs to the shard its first piece lands in.
                self.pending_records += 1
                first = False
            self.current.append(piece)
            self.chars += added_chars
            self.bytes += added_bytes

    def flush(self) -> None:
        if not self.current:
            return
        self.pieces.append(("\n\n".join(self.current), self.pending_records))
        self.current = []
        self.chars = 0
        self.bytes = 0
        self.pending_records = 0

    def finish(self) -> list[tuple[str, int]]:
        self.flush()
        return self.pieces


def shard_texts(texts: Iterable[str], *, max_chars: int, max_bytes: int) -> list[tuple[str, int]]:
    """Shard projected texts, returning ``(content, record_count)`` pairs."""
    accumulator = _ShardAccumulator(max_chars=max_chars, max_bytes=max_bytes)
    for text in texts:
        accumulator.add(text)
    return accumulator.finish()


def _hash_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _display_name(uri: str) -> str:
    try:
        return urllib.parse.unquote(urllib.parse.urlsplit(uri).path.rsplit("/", 1)[-1])
    except ValueError:
        return ""


@dataclass(frozen=True)
class _FilePlan:
    uri: str
    member: str | None
    path: Path
    media_type: str | None
    sha256: str | None
    name: str


def _file_plan(item: dict[str, object], config: Config) -> _FilePlan | None:
    blob = item.get("blob")
    if not isinstance(blob, str) or not blob:
        return None
    path = Path(blob)
    if not path.is_absolute():
        path = Path(config.data_root) / path
    uri = str(item.get("final_url") or item.get("uri") or "")
    member = item.get("member") if isinstance(item.get("member"), str) else None
    media_type = item.get("media_type") if isinstance(item.get("media_type"), str) else None
    sha256 = item.get("sha256") if isinstance(item.get("sha256"), str) else None
    name = member or _display_name(uri) or uri
    return _FilePlan(uri=uri, member=member, path=path, media_type=media_type,
                     sha256=sha256, name=name)


def _file_shards(
    plan: _FilePlan,
    *,
    entry_title: str,
    config: Config,
) -> tuple[list[tuple[str, int]], int, bool, str, str | None, str | None]:
    """Project and shard one file.

    Returns ``(shards, records, truncated, normalized_sha256, adapter, error)``.
    On a parse error the accumulated shards are discarded (fail closed) and the
    error is returned instead.
    """
    accumulator = _ShardAccumulator(
        max_chars=config.max_shard_chars, max_bytes=config.max_shard_bytes
    )
    digest = hashlib.sha256()
    header = f"[{entry_title}] {plan.name}\n\n"
    summary = adapters.StreamSummary()
    try:
        for record in adapters.iter_records(
            plan.path,
            name=plan.name,
            media_type=plan.media_type,
            config=config,
            summary=summary,
        ):
            text = project_data(record.kind, record.data)
            digest.update(text.encode("utf-8"))
            digest.update(b"\n\x1e\n")
            accumulator.add(text)
    except adapters.AdapterError as exc:
        return [], 0, False, "", summary.adapter, f"{exc.code}:{exc.detail}" if exc.detail else exc.code
    except (OSError, ValueError, UnicodeError) as exc:
        return [], 0, False, "", summary.adapter, f"parse_error:{type(exc).__name__}"
    pieces = accumulator.finish()
    if header.strip() and pieces:
        pieces = [(header + content, count) for content, count in pieces]
    return pieces, accumulator.records, summary.truncated, digest.hexdigest(), summary.adapter, None


def _content_record(
    *,
    row: dict[str, object],
    plan: _FilePlan,
    content: str,
    record_count: int,
    shard_index: int,
    shard_count: int,
    normalized_sha256: str,
    commit: str,
    profiles: dict[str, str],
    policy: str | None,
    truncated: bool,
) -> NormalizedRecord:
    entry_id = str(row.get("entry_id") or "")
    source_url = canonical_source_url(plan.uri, plan.member)
    title = f"{row.get('title') or entry_id} — {plan.name}"[:MAX_TITLE_CHARS]
    content_sha = sha256_text(content)
    source_id = content_source_id(entry_id, source_url, shard_index)
    return NormalizedRecord(
        source_id=source_id,
        title=title,
        content=content,
        content_sha256=content_sha,
        origin=f"apd-core@{commit}",
        locator=model.sanitize_url(plan.uri) or plan.uri or entry_id,
        revision_id=revision_id_for(
            source_id=source_id,
            content_sha256=content_sha,
            normalized_sha256=normalized_sha256,
            source_sha256=plan.sha256,
            adapter_version=adapters.ADAPTER_VERSION,
            license_policy=policy,
            profiles=profiles,
        ),
        entry_id=entry_id,
        kind="content",
        normalized_sha256=normalized_sha256,
        adapter_version=adapters.ADAPTER_VERSION,
        license_policy=policy,
        source_sha256=plan.sha256,
        shard_index=shard_index,
        shard_count=shard_count,
        record_count=record_count,
        truncated=truncated,
    )


def _acquired_files(acquisition: dict[str, object]) -> list[dict[str, object]]:
    files = acquisition.get("files")
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict) and item.get("status") == "acquired"]


def _archive_member_plans(
    plan: _FilePlan, config: Config
) -> tuple[list[_FilePlan], str | None, str | None]:
    """Inspect an archive without extracting it; the caller extracts to stream.

    Returns ``(members, archive_format, error)``.  An empty member list with no
    error means the file is not an archive at all.
    """
    prefix = b""
    try:
        with plan.path.open("rb") as handle:
            prefix = handle.read(config.acquire_sniff_prefix_bytes)
    except OSError:
        return [], None, "parse_error:archive could not be opened"
    archive_format = archive.archive_format_for(plan.name, plan.media_type, prefix)
    if not archive_format:
        return [], None, None
    try:
        inspection = archive.inspect(
            plan.path, name=plan.name, archive_format=archive_format, config=config
        )
    except archive.ArchiveError as exc:
        return [], archive_format, f"{exc.code}:{exc.detail}"
    budget = max(1, config.acquire_max_files_per_dataset)
    members = [
        _FilePlan(
            uri=plan.uri,
            member=member.name,
            path=plan.path,
            media_type=None,
            sha256=None,
            name=member.name,
        )
        for member in inspection.data_members[:budget]
    ]
    return members, archive_format, None


def normalize_entry(
    row: dict[str, object],
    acquisition: dict[str, object] | None,
    *,
    commit: str,
    profiles: dict[str, str],
    config: Config,
) -> EntryNormalization:
    """Normalize one entry: its metadata record plus every content shard."""
    entry_id = str(row.get("entry_id") or "")
    metadata, metadata_reason = metadata_record(
        row, commit=commit, profiles=profiles, config=config
    )
    result = EntryNormalization(
        entry_id=entry_id, metadata=metadata, metadata_reason=metadata_reason
    )
    policy = license_policy(row)
    if acquisition is None:
        result.content_reason = "not_acquired"
        return result
    status = str(acquisition.get("status") or "")
    if status not in {"acquired", "unchanged"}:
        result.content_reason = str(acquisition.get("reason_code") or "not_acquired")
        return result
    if policy != "index_content":
        result.content_reason = license_reason(row) or "license_not_indexable"
        return result
    files = _acquired_files(acquisition)
    if not files:
        result.content_reason = "no_acquired_files"
        return result

    entry_title = str(row.get("title") or entry_id)
    failures = 0
    first_failure: str | None = None
    for item in files:
        plan = _file_plan(item, config)
        if plan is None:
            failures += 1
            first_failure = first_failure or "missing_blob"
            result.files.append(
                FileOutcome(uri=str(item.get("uri") or ""),
                            member=item.get("member") if isinstance(item.get("member"), str) else None,
                            status="failed", reason_code="missing_blob")
            )
            continue
        members, archive_format, archive_error = _archive_member_plans(plan, config)
        if archive_error:
            failures += 1
            reason, _, detail = archive_error.partition(":")
            first_failure = first_failure or (reason or "parse_error")
            result.files.append(
                FileOutcome(uri=plan.uri, member=None, status="failed",
                            reason_code=reason or "parse_error", detail=detail or None)
            )
            continue
        if members:
            try:
                with archive.extracted_members(
                    plan.path,
                    name=plan.name,
                    archive_format=archive_format,
                    config=config,
                ) as extracted:
                    by_name = {member.name: member for member in extracted}
                    for member_plan in members:
                        extracted_member = by_name.get(member_plan.name or "")
                        if extracted_member is None:
                            continue
                        resolved = _FilePlan(
                            uri=member_plan.uri,
                            member=member_plan.member,
                            path=extracted_member.path,
                            media_type=None,
                            sha256=_hash_file(extracted_member.path),
                            name=member_plan.name,
                        )
                        failed = _normalize_file(
                            row, resolved, result, entry_title=entry_title,
                            commit=commit, profiles=profiles, policy=policy, config=config,
                        )
                        if failed:
                            failures += 1
                            first_failure = first_failure or result.files[-1].reason_code
            except archive.ArchiveError as exc:
                failures += 1
                first_failure = first_failure or exc.code
                result.files.append(
                    FileOutcome(uri=plan.uri, member=None, status="failed",
                                reason_code=exc.code, detail=exc.detail)
                )
            continue
        failed = _normalize_file(
            row, plan, result, entry_title=entry_title,
            commit=commit, profiles=profiles, policy=policy, config=config,
        )
        if failed:
            failures += 1
            first_failure = first_failure or result.files[-1].reason_code

    if not result.records and result.content_reason is None:
        result.content_reason = first_failure or "empty_content"
    return result


def _normalize_file(
    row: dict[str, object],
    plan: _FilePlan,
    result: EntryNormalization,
    *,
    entry_title: str,
    commit: str,
    profiles: dict[str, str],
    policy: str | None,
    config: Config,
) -> bool:
    """Append one file's shards to ``result``; return True when it failed."""
    pieces, records, truncated, normalized_sha, adapter, error = _file_shards(
        plan, entry_title=entry_title, config=config
    )
    result.records_total += records
    if error is not None:
        code, _, detail = error.partition(":")
        result.files.append(
            FileOutcome(uri=plan.uri, member=plan.member, status="failed",
                        adapter=adapter, reason_code=code or "parse_error",
                        detail=detail or None)
        )
        return True
    result.truncated = result.truncated or truncated
    result.files.append(
        FileOutcome(uri=plan.uri, member=plan.member, status="parsed",
                    adapter=adapter, records=records, shards=len(pieces),
                    truncated=truncated)
    )
    for shard_index, (content, record_count) in enumerate(pieces):
        result.records.append(
            _content_record(
                row=row,
                plan=plan,
                content=content,
                record_count=record_count,
                shard_index=shard_index,
                shard_count=len(pieces),
                normalized_sha256=normalized_sha,
                commit=commit,
                profiles=profiles,
                policy=policy,
                truncated=truncated,
            )
        )
    return False
