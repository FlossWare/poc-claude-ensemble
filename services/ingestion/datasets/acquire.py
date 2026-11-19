"""PR 3: acquire dataset content into a content-addressed store.

Everything here is bounded and fail-closed:

* downloads stream through :mod:`ingestion.datasets.netguard` (SSRF guard, pinned
  DNS, per-hop revalidation, HTTPS downgrade refusal) and never buffer a body in
  memory;
* ``Content-Length`` is checked before the first byte and the running total is
  checked again while streaming, against the per-file, per-dataset and per-run
  byte caps plus a free-disk guard;
* SHA-256 is computed while writing and, when the resolver captured an upstream
  checksum, that checksum is verified before the blob is adopted;
* a ``.part`` file plus a sidecar is resumed with ``Range``/``If-Range``; only a
  ``206`` whose ``Content-Range`` starts at the resume offset is accepted, and a
  server that answers ``200`` or ``416`` makes the download restart from zero;
* blobs are named by their SHA-256 and adopted with an atomic rename, so a
  filename supplied by a remote server is never used as a path;
* archives are inspected by :mod:`ingestion.datasets.archive` and extracted
  member-by-member, and files are parsed by
  :mod:`ingestion.datasets.adapters`.

Every entry that is not acquired carries an explicit reason code.  Knowledge,
Index, Graph and Lexical writes belong to PR 4/5 and deliberately do not happen
here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
import urllib.parse
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import adapters, archive
from .config import Config
from .model import canonical_json, sanitize_url, sha256_hex
from .netguard import NetguardError, Transport, status_error
from .resolver import HostRateLimiter, Resolution, ResolvedSource, build_transport

ACQUIRE_VERSION = 1

#: ``acquired`` means new bytes were stored this run; ``unchanged`` means the
#: resolved content was already stored (reused, not re-downloaded).
SUCCESS_ACQUISITION_STATUSES = frozenset({"acquired", "unchanged"})
VALID_ACQUISITION_STATUSES = frozenset(
    {"acquired", "unchanged", "skipped", "deferred", "failed"}
)

#: Reason codes a later run may retry (mirrors the design's acquisition table).
RETRYABLE_REASONS = frozenset(
    {
        "run_budget_exhausted",
        "oversized_declared",
        "oversized_stream",
        "checksum_mismatch",
        "partial_download",
        "disk_guard",
        "blocked",
        "not_found",
        "rate_limited",
        "timeout",
        "dns_failure",
        "tls_failure",
        "connection_failed",
        "server_error",
        "http_error",
        "unsupported_format",
        "adapter_unavailable",
        "encoding_error",
        "limits_exceeded",
        # resolution-stage reasons that the design marks retryable; they are
        # inherited by acquisition when the resolution did not find a source.
        "landing_page_only",
        "resolver_not_implemented",
        "doi_unresolved",
    }
)

#: Resolved kinds that name a whole repository/collection rather than a file.
#: A clone is out of scope for PR 3, so these are refused explicitly instead of
#: downloading the HTML page they point at.
REPOSITORY_KINDS = frozenset({"repository", "collection"})

_CHUNK = 64 * 1024
_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_HASHERS = {"sha256": hashlib.sha256, "sha1": hashlib.sha1, "md5": hashlib.md5}


class AcquireError(Exception):
    """A download or local write failed with a stable reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


class BudgetExceeded(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_checksum(value: object) -> tuple[str, str] | None:
    """Parse ``md5:...``/``sha256:...``/``sha1:...`` or a bare hex digest."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    if ":" in text:
        algorithm, _, digest = text.partition(":")
        algorithm = algorithm.strip()
        digest = digest.strip()
        if algorithm in _HASHERS and digest:
            return algorithm, digest
        return None
    expected = {32: "md5", 40: "sha1", 64: "sha256"}.get(len(text))
    if expected and re.fullmatch(r"[0-9a-f]+", text):
        return expected, text
    return None


# ---------------------------------------------------------------------------
# Sources and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRef:
    """The acquisition view of a resolved source."""

    uri: str
    kind: str
    media_type: str | None = None
    declared_bytes: int | None = None
    member: str | None = None
    checksum: str | None = None
    version: str | None = None

    @classmethod
    def from_resolved(cls, source: ResolvedSource) -> "SourceRef":
        return cls(
            uri=source.uri,
            kind=source.kind,
            media_type=source.media_type,
            declared_bytes=source.size_bytes,
            member=source.member,
            checksum=getattr(source, "checksum", None),
            version=source.version,
        )

    def name(self) -> str:
        if self.member:
            return self.member
        try:
            return urllib.parse.urlsplit(self.uri).path.rsplit("/", 1)[-1]
        except ValueError:
            return ""

    def to_artifact(self) -> dict[str, object]:
        return {
            "uri": sanitize_url(self.uri),
            "kind": self.kind,
            "media_type": self.media_type,
            "declared_bytes": self.declared_bytes,
            "member": self.member,
            "checksum": self.checksum,
            "version": self.version,
        }


@dataclass(frozen=True)
class AcquiredFile:
    uri: str
    member: str | None
    kind: str
    status: str
    reason_code: str | None = None
    detail: str | None = None
    media_type: str | None = None
    declared_bytes: int | None = None
    bytes: int = 0
    sha256: str | None = None
    blob: str | None = None
    checksum_declared: str | None = None
    checksum_verified: bool | None = None
    etag: str | None = None
    last_modified: str | None = None
    final_url: str | None = None
    retrieved_at: str | None = None
    resumed: bool = False
    cached: bool = False
    attempts: int = 1
    archive: dict[str, object] | None = None
    parse: dict[str, object] | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "uri": sanitize_url(self.uri),
            "member": self.member,
            "kind": self.kind,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:300] if self.detail else None,
            "media_type": self.media_type,
            "declared_bytes": self.declared_bytes,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "blob": self.blob,
            "checksum_declared": self.checksum_declared,
            "checksum_verified": self.checksum_verified,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "final_url": sanitize_url(self.final_url),
            "retrieved_at": self.retrieved_at,
            "resumed": self.resumed,
            "cached": self.cached,
            "attempts": self.attempts,
            "archive": self.archive,
            "parse": self.parse,
        }

    @classmethod
    def from_cache(cls, payload: dict[str, object]) -> "AcquiredFile":
        return cls(
            uri=str(payload.get("uri") or ""),
            member=payload.get("member") if isinstance(payload.get("member"), str) else None,
            kind=str(payload.get("kind") or "unknown"),
            status=str(payload.get("status") or "acquired"),
            reason_code=payload.get("reason_code")
            if isinstance(payload.get("reason_code"), str)
            else None,
            detail=payload.get("detail") if isinstance(payload.get("detail"), str) else None,
            media_type=payload.get("media_type")
            if isinstance(payload.get("media_type"), str)
            else None,
            declared_bytes=payload.get("declared_bytes")
            if isinstance(payload.get("declared_bytes"), int)
            else None,
            bytes=payload.get("bytes") if isinstance(payload.get("bytes"), int) else 0,
            sha256=payload.get("sha256") if isinstance(payload.get("sha256"), str) else None,
            blob=payload.get("blob") if isinstance(payload.get("blob"), str) else None,
            checksum_declared=payload.get("checksum_declared")
            if isinstance(payload.get("checksum_declared"), str)
            else None,
            checksum_verified=payload.get("checksum_verified")
            if isinstance(payload.get("checksum_verified"), bool)
            else None,
            etag=payload.get("etag") if isinstance(payload.get("etag"), str) else None,
            last_modified=payload.get("last_modified")
            if isinstance(payload.get("last_modified"), str)
            else None,
            final_url=payload.get("final_url")
            if isinstance(payload.get("final_url"), str)
            else None,
            retrieved_at=payload.get("retrieved_at")
            if isinstance(payload.get("retrieved_at"), str)
            else None,
            resumed=False,
            cached=True,
            attempts=payload.get("attempts")
            if isinstance(payload.get("attempts"), int)
            else 1,
            archive=payload.get("archive") if isinstance(payload.get("archive"), dict) else None,
            parse=payload.get("parse") if isinstance(payload.get("parse"), dict) else None,
        )


@dataclass(frozen=True)
class Acquisition:
    entry_id: str
    status: str
    reason_code: str | None = None
    detail: str | None = None
    files: tuple[AcquiredFile, ...] = ()
    bytes_total: int = 0
    records_total: int = 0
    parse_status: str | None = None
    retryable: bool = False
    next_retry_at: str | None = None
    attempts_count: int = 1
    checked_at: str | None = None
    resolution_method: str | None = None
    cached: bool = False

    @property
    def files_acquired(self) -> int:
        return sum(1 for item in self.files if item.status == "acquired")

    @property
    def files_failed(self) -> int:
        return sum(1 for item in self.files if item.status == "failed")

    @property
    def next_action(self) -> str:
        if self.status in SUCCESS_ACQUISITION_STATUSES:
            return "normalize"
        if self.reason_code == "run_budget_exhausted":
            return "resume"
        if self.reason_code == "not_resolved":
            return "resolve"
        if self.reason_code and self.reason_code.startswith("license_"):
            return "review_license"
        if self.status == "deferred":
            return "retry"
        if self.reason_code in {"invalid_url", "unsupported_scheme", "no_source_url"}:
            return "fix_metadata"
        return "review"

    def to_artifact(self) -> dict[str, object]:
        budget = self.reason_code == "run_budget_exhausted"
        return {
            "acquire_version": ACQUIRE_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:500] if self.detail else None,
            "files_total": len(self.files),
            "files_acquired": self.files_acquired,
            "files_failed": self.files_failed,
            "bytes_total": self.bytes_total,
            "records_total": self.records_total,
            "parse_status": self.parse_status,
            "retryable": self.retryable,
            "next_retry_at": None if budget else self.next_retry_at,
            "attempts_count": self.attempts_count,
            "checked_at": None if budget else self.checked_at,
            "resolution_method": self.resolution_method,
            "cached": self.cached,
            "next_action": self.next_action,
            "files": [item.to_artifact() for item in self.files],
        }

    def to_cache(self) -> dict[str, object]:
        return self.to_artifact() | {"entry_id": self.entry_id}

    @classmethod
    def from_cache(cls, payload: dict[str, object], *, entry_id: str) -> "Acquisition":
        files = tuple(
            AcquiredFile.from_cache(item)
            for item in payload.get("files") or []
            if isinstance(item, dict)
        )
        return cls(
            entry_id=entry_id,
            status=str(payload.get("status") or "acquired"),
            reason_code=payload.get("reason_code")
            if isinstance(payload.get("reason_code"), str)
            else None,
            detail=payload.get("detail") if isinstance(payload.get("detail"), str) else None,
            files=files,
            bytes_total=payload.get("bytes_total")
            if isinstance(payload.get("bytes_total"), int)
            else 0,
            records_total=payload.get("records_total")
            if isinstance(payload.get("records_total"), int)
            else 0,
            parse_status=payload.get("parse_status")
            if isinstance(payload.get("parse_status"), str)
            else None,
            retryable=bool(payload.get("retryable")),
            next_retry_at=payload.get("next_retry_at")
            if isinstance(payload.get("next_retry_at"), str)
            else None,
            attempts_count=payload.get("attempts_count")
            if isinstance(payload.get("attempts_count"), int)
            else 1,
            checked_at=payload.get("checked_at")
            if isinstance(payload.get("checked_at"), str)
            else None,
            resolution_method=payload.get("resolution_method")
            if isinstance(payload.get("resolution_method"), str)
            else None,
            cached=True,
        )


# ---------------------------------------------------------------------------
# Local stores
# ---------------------------------------------------------------------------


class ContentStore:
    """Content-addressed blobs plus resumable ``.part`` files.

    ``cache/blobs/<sha[:2]>/<sha>`` holds completed content; a remote filename
    is never part of a path.  ``cache/partial/<sha(url)[:2]>/<sha(url)>.part``
    holds an in-progress download with a JSON sidecar recording its URL and HTTP
    validators.
    """

    def __init__(self, config: Config, *, root: Path | None = None) -> None:
        self.config = config
        self.root = Path(root) if root is not None else Path(config.data_root) / "cache" / "blobs"
        self.partial_root = Path(config.data_root) / "cache" / "partial"

    def blob_path(self, sha256: str) -> Path:
        return self.root / sha256[:2] / sha256

    def has(self, sha256: str) -> bool:
        path = self.blob_path(sha256)
        try:
            info = path.lstat()
        except OSError:
            return False
        return stat.S_ISREG(info.st_mode)

    def blob_size(self, sha256: str) -> int:
        try:
            return self.blob_path(sha256).stat().st_size
        except OSError:
            return 0

    def adopt(self, part: Path, sha256: str) -> Path:
        target = self.blob_path(sha256)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(part, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return target

    def part_paths(self, uri: str) -> tuple[Path, Path]:
        key = sha256_hex(uri)
        directory = self.partial_root / key[:2]
        return directory / f"{key}.part", directory / f"{key}.json"

    def discard(self, uri: str) -> None:
        part, sidecar = self.part_paths(uri)
        for path in (part, sidecar):
            try:
                path.unlink()
            except OSError:
                pass

    def write_sidecar(self, uri: str, payload: dict[str, object]) -> None:
        part, sidecar = self.part_paths(uri)
        sidecar.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, sidecar)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

    def read_sidecar(self, uri: str) -> dict[str, object] | None:
        part, sidecar = self.part_paths(uri)
        try:
            info = sidecar.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024:
                return None
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("url") != uri:
            return None
        try:
            part_info = part.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(part_info.st_mode):
            return None
        if payload.get("bytes") != part_info.st_size:
            return None
        return payload


class AcquisitionCache:
    """Per-entry acquisition artifacts, keyed by metadata and resolved sources."""

    def __init__(self, config: Config, *, root: Path | None = None) -> None:
        self.config = config
        self.root = (
            Path(root) if root is not None else Path(config.data_root) / "cache" / "acquisitions"
        )

    def key_for(
        self, entry_id: str, metadata_sha256: str, sources: list[SourceRef]
    ) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "acquire_version": ACQUIRE_VERSION,
                    "entry_id": entry_id,
                    "metadata_sha256": metadata_sha256,
                    "sources": [source.uri for source in sources],
                }
            )
        )

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(
        self,
        entry_id: str,
        metadata_sha256: str,
        sources: list[SourceRef],
        *,
        now: str | None = None,
    ) -> Acquisition | None:
        path = self.path_for(self.key_for(entry_id, metadata_sha256, sources))
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("entry_id") != entry_id:
            return None
        checked = _parse_iso(payload.get("checked_at") if isinstance(payload.get("checked_at"), str) else None)
        reference = _parse_iso(now or utc_now())
        if checked is None or reference is None:
            return None
        if reference - checked > timedelta(days=max(0, self.config.recheck_after_days)):
            return None
        try:
            return Acquisition.from_cache(payload, entry_id=entry_id)
        except (TypeError, ValueError):
            return None

    def put(self, acquisition: Acquisition, key: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path_for(key)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        data = canonical_json(acquisition.to_cache()).encode("utf-8")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(data)
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass


class RunBudget:
    """Per-run byte, dataset, wall-clock and free-disk guards."""

    def __init__(self, config: Config, *, clock=None, disk_usage=None) -> None:
        self.config = config
        self.clock = clock or time.monotonic
        self.disk_usage = disk_usage or shutil.disk_usage
        self.started = self.clock()
        self.bytes_used = 0
        self.datasets_used = 0
        self._mutex = threading.Lock()

    def check_run(self) -> None:
        if self.datasets_used >= self.config.max_run_datasets:
            raise BudgetExceeded(
                "run_budget_exhausted",
                f"run dataset cap reached ({self.config.max_run_datasets})",
            )
        elapsed = self.clock() - self.started
        if elapsed >= self.config.max_run_seconds:
            raise BudgetExceeded(
                "run_budget_exhausted",
                f"run time cap reached ({self.config.max_run_seconds}s)",
            )
        if self.bytes_used >= self.config.max_run_bytes:
            raise BudgetExceeded(
                "run_budget_exhausted",
                f"run byte cap reached ({self.config.max_run_bytes})",
            )
        self.check_disk()

    def check_disk(self) -> None:
        try:
            usage = self.disk_usage(str(self.config.data_root))
            free = usage.free if hasattr(usage, "free") else usage[2]
        except OSError:
            return
        if free < self.config.min_free_disk_bytes:
            raise BudgetExceeded(
                "disk_guard",
                f"free disk {free} is below the {self.config.min_free_disk_bytes} byte floor",
            )

    def remaining_bytes(self) -> int:
        return max(0, self.config.max_run_bytes - self.bytes_used)

    def charge(self, count: int) -> None:
        with self._mutex:
            self.bytes_used += max(0, count)

    def started_dataset(self) -> None:
        with self._mutex:
            self.datasets_used += 1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _sniff_family(prefix: bytes) -> str | None:
    if not prefix:
        return None
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    head = prefix[:1024].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return "html"
    if prefix.startswith(b"PK\x03\x04") or prefix.startswith(b"PK\x05\x06"):
        return "zip"
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if prefix.startswith(b"BZh"):
        return "bzip2"
    if prefix.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return None


def _declared_kind(ref: SourceRef) -> str | None:
    name = ref.member or ref.name()
    return adapters.normalize_kind(name, ref.media_type, None)


def _check_content_type(ref: SourceRef, prefix: bytes) -> None:
    """Refuse HTML or binary payloads that claim to be a data file."""
    family = _sniff_family(prefix)
    declared = _declared_kind(ref)
    if family == "html":
        raise AcquireError(
            "content_type_mismatch",
            f"server returned an HTML page for {ref.kind or 'dataset'} content",
        )
    if family == "pdf" and declared in adapters.TEXT_KINDS:
        raise AcquireError(
            "content_type_mismatch", "server returned a PDF for text content"
        )
    if (
        declared in adapters.TEXT_KINDS
        and b"\x00" in prefix
        and family not in {"zip", "gzip", "bzip2", "xz"}
    ):
        raise AcquireError(
            "content_type_mismatch", "server returned binary content for a text file"
        )


def _hash_prefix(part: Path, hashers: dict[str, object], expected: int) -> None:
    read = 0
    with part.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            read += len(chunk)
            if read > expected:
                raise AcquireError("partial_download", "partial file is larger than its sidecar")
            for hasher in hashers.values():
                hasher.update(chunk)
    if read != expected:
        raise AcquireError("partial_download", "partial file is shorter than its sidecar")


def _open_part(part: Path, mode: str) -> int:
    part.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if mode == "ab":
        flags |= os.O_APPEND
    else:
        flags |= os.O_TRUNC
    return os.open(part, flags, 0o600)


def _download_source(
    ref: SourceRef,
    ctx: "AcquireContext",
    *,
    max_bytes: int,
) -> AcquiredFile:
    """Fetch one source into the content store; never exceeds ``max_bytes``."""
    config = ctx.config
    store = ctx.store
    expected = parse_checksum(ref.checksum)
    if expected is not None and expected[0] == "sha256" and store.has(expected[1]):
        return _cached_blob(ref, expected, store, config)

    remaining_run = ctx.budget.remaining_bytes()
    file_cap = config.max_file_bytes
    limit = min(file_cap, remaining_run, max_bytes)
    if limit <= 0:
        raise AcquireError("run_budget_exhausted", "run byte budget is exhausted")
    # Name the binding constraint so a retry decision is meaningful: a run
    # budget shortfall is retryable as a budget, a file/dataset cap is not.
    run_binds = remaining_run <= min(file_cap, max_bytes)
    stream_code = "run_budget_exhausted" if run_binds else "oversized_stream"
    declared_code = "run_budget_exhausted" if run_binds else "oversized_declared"
    if ref.declared_bytes is not None and ref.declared_bytes > limit:
        raise AcquireError(
            declared_code,
            f"declared {ref.declared_bytes} bytes exceeds the {limit} byte cap",
        )

    part, _sidecar = store.part_paths(ref.uri)
    existing = 0
    validator = None
    resumed = False
    if config.acquire_resume_enabled:
        sidecar = store.read_sidecar(ref.uri)
        if sidecar is not None:
            existing = int(sidecar.get("bytes") or 0)
            validator = sidecar.get("etag") or sidecar.get("last_modified")
            resumed = existing > 0

    attempts = 0
    while True:
        attempts += 1
        headers: dict[str, str] = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            if isinstance(validator, str) and validator:
                headers["If-Range"] = validator
        restart = False
        try:
            with ctx.transport.open_stream(ref.uri, method="GET", headers=headers) as response:
                status = response.status
                if status == 416 and existing > 0:
                    restart = True
                elif status == 200 or status == 206:
                    total_declared = _declared_total(response.headers, status)
                    if status == 206:
                        start = _content_range_start(response.headers)
                        if start != existing:
                            restart = True
                        elif total_declared is not None and total_declared > limit:
                            raise AcquireError(
                                "oversized_declared",
                                f"declared {total_declared} bytes exceeds the {limit} byte cap",
                            )
                    else:
                        # The server ignored (or never received) the range.
                        existing = 0
                        resumed = False
                        if (
                            total_declared is not None
                            and total_declared > limit
                        ):
                            raise AcquireError(
                                "oversized_declared",
                                f"declared {total_declared} bytes exceeds the {limit} byte cap",
                            )
                    if not restart:
                        return _consume(
                            ref,
                            ctx,
                            response,
                            part=part,
                            existing=existing,
                            limit=limit,
                            expected=expected,
                            resumed=resumed,
                            attempts=attempts,
                            stream_code=stream_code,
                            declared_code=declared_code,
                        )
                elif status == 304:
                    raise AcquireError(
                        "partial_download", "server returned 304 without a stored revision"
                    )
                else:
                    raise status_error(status, response.final_url, response.redirect_chain)
        except NetguardError as exc:
            raise AcquireError(exc.code, str(exc)) from exc
        if not restart:
            raise AcquireError("partial_download", "download did not complete")
        if attempts >= 3:
            raise AcquireError("partial_download", "range request could not be resumed")
        store.discard(ref.uri)
        existing = 0
        validator = None
        resumed = False


def _declared_total(headers: dict[str, str], status: int) -> int | None:
    if status == 206:
        match = _CONTENT_RANGE_RE.search(headers.get("content-range", ""))
        if match and match.group(3).isdigit():
            return int(match.group(3))
        return None
    value = headers.get("content-length")
    if value and value.strip().isdigit():
        return int(value.strip())
    return None


def _content_range_start(headers: dict[str, str]) -> int | None:
    match = _CONTENT_RANGE_RE.search(headers.get("content-range", ""))
    if not match:
        return None
    return int(match.group(1))


def _consume(
    ref: SourceRef,
    ctx: "AcquireContext",
    response,
    *,
    part: Path,
    existing: int,
    limit: int,
    expected: tuple[str, str] | None,
    resumed: bool,
    attempts: int,
    stream_code: str = "oversized_stream",
    declared_code: str = "oversized_declared",
) -> AcquiredFile:
    config = ctx.config
    store = ctx.store
    hashers = {"sha256": hashlib.sha256()}
    if expected is not None:
        hashers.setdefault(expected[0], _HASHERS[expected[0]]())
    if existing > 0:
        try:
            _hash_prefix(part, hashers, existing)
        except AcquireError:
            store.discard(ref.uri)
            raise AcquireError("partial_download", "partial file could not be re-read")
    descriptor = _open_part(part, "ab" if existing > 0 else "wb")
    total = existing
    prefix = b""
    sniffed = False
    declared_total = _declared_total(response.headers, response.status)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            if existing == 0:
                handle.truncate(0)
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise AcquireError(
                        stream_code,
                        f"stream exceeded the {limit} byte cap",
                    )
                if not sniffed:
                    prefix += chunk[: config.acquire_sniff_prefix_bytes]
                    if len(prefix) >= min(512, config.acquire_sniff_prefix_bytes):
                        _check_content_type(ref, prefix)
                        sniffed = True
                for hasher in hashers.values():
                    hasher.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except AcquireError as exc:
        if exc.code in {"oversized_stream", "content_type_mismatch"}:
            store.discard(ref.uri)
        else:
            _remember_partial(store, ref, part, total, response)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if not sniffed:
        try:
            _check_content_type(ref, prefix)
        except AcquireError:
            store.discard(ref.uri)
            raise

    if declared_total is not None and total != declared_total:
        _remember_partial(store, ref, part, total, response)
        raise AcquireError(
            "partial_download",
            f"received {total} of {declared_total} declared bytes",
        )
    if expected is not None:
        algorithm, digest = expected
        actual = hashers[algorithm].hexdigest()
        if actual != digest:
            store.discard(ref.uri)
            raise AcquireError(
                "checksum_mismatch",
                f"declared {algorithm} does not match the downloaded bytes",
            )
    sha256 = hashers["sha256"].hexdigest()
    blob = store.adopt(part, sha256)
    store.discard(ref.uri)
    ctx.budget.charge(total)
    _validate_disk(ctx)
    return AcquiredFile(
        uri=ref.uri,
        member=ref.member,
        kind=ref.kind,
        status="acquired",
        media_type=ref.media_type,
        declared_bytes=ref.declared_bytes if ref.declared_bytes is not None else declared_total,
        bytes=total,
        sha256=sha256,
        blob=_blob_reference(blob, config),
        checksum_declared=ref.checksum,
        checksum_verified=None if expected is None else True,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        final_url=response.final_url,
        retrieved_at=utc_now(),
        resumed=resumed,
        attempts=attempts,
    )


def _blob_reference(path: Path, config: Config) -> str:
    """Store the blob locator relative to the data root when possible."""
    try:
        return str(path.relative_to(Path(config.data_root)))
    except ValueError:
        return str(path)


def _remember_partial(store: ContentStore, ref: SourceRef, part: Path, total: int, response) -> None:
    store.write_sidecar(
        ref.uri,
        {
            "url": ref.uri,
            "bytes": total,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "expected_checksum": ref.checksum,
            "updated_at": utc_now(),
        },
    )


def _validate_disk(ctx: "AcquireContext") -> None:
    try:
        ctx.budget.check_disk()
    except BudgetExceeded as exc:
        raise AcquireError(exc.code, exc.detail) from exc


def _cached_blob(
    ref: SourceRef, expected: tuple[str, str], store: ContentStore, config: Config
) -> AcquiredFile:
    sha256 = expected[1]
    return AcquiredFile(
        uri=ref.uri,
        member=ref.member,
        kind=ref.kind,
        status="acquired",
        media_type=ref.media_type,
        declared_bytes=ref.declared_bytes,
        bytes=store.blob_size(sha256),
        sha256=sha256,
        blob=_blob_reference(store.blob_path(sha256), config),
        checksum_declared=ref.checksum,
        checksum_verified=True,
        retrieved_at=utc_now(),
        cached=True,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _blob_prefix(blob: Path, limit: int) -> bytes:
    try:
        with blob.open("rb") as handle:
            return handle.read(max(1, limit))
    except OSError:
        return b""


def _parse_blob(item: AcquiredFile, ctx: "AcquireContext") -> tuple[AcquiredFile, int]:
    """Parse one stored blob; returns the updated file and its record count."""
    if item.status != "acquired" or not item.blob:
        return item, 0
    blob = Path(ctx.config.data_root) / item.blob if not os.path.isabs(item.blob) else Path(item.blob)
    name = item.member or item.uri
    prefix = _blob_prefix(blob, ctx.config.acquire_sniff_prefix_bytes)
    archive_format = archive.archive_format_for(name, item.media_type, prefix)
    if archive_format:
        return _parse_archive(item, blob, archive_format, ctx)
    result = adapters.parse_file(
        blob, name=name, media_type=item.media_type, config=ctx.config
    )
    return replace(item, parse=result.to_artifact()), result.record_count


def _parse_archive(
    item: AcquiredFile, blob: Path, archive_format: str, ctx: "AcquireContext"
) -> tuple[AcquiredFile, int]:
    config = ctx.config
    name = item.member or item.uri
    try:
        inspection = archive.inspect(blob, name=name, archive_format=archive_format, config=config)
    except archive.ArchiveError as exc:
        return (
            replace(
                item,
                archive={
                    "archive_version": archive.ARCHIVE_VERSION,
                    "format": archive_format,
                    "status": "failed",
                    "reason_code": exc.code,
                    "detail": exc.detail[:300],
                },
            ),
            0,
        )
    members: list[dict[str, object]] = []
    records = 0
    truncated = False
    budget_files = max(1, config.acquire_max_files_per_dataset)
    try:
        with archive.extracted_members(
            blob,
            name=name,
            archive_format=archive_format,
            config=config,
            inspection=inspection,
        ) as extracted:
            truncated = len(extracted) > budget_files
            for member in extracted[:budget_files]:
                result = adapters.parse_file(
                    member.path, name=member.name, media_type=None, config=config
                )
                records += result.record_count
                members.append(
                    {"member": member.name, "size": member.size, "parse": result.to_artifact()}
                )
    except archive.ArchiveError as exc:
        artifact = inspection.to_artifact()
        artifact.update({"status": "failed", "reason_code": exc.code, "detail": exc.detail[:300]})
        return replace(item, archive=artifact), records
    artifact = inspection.to_artifact()
    artifact["status"] = "parsed" if members else "skipped"
    artifact["members"] = members
    if truncated:
        artifact["truncated"] = True
    return replace(item, archive=artifact), records


def _parse_is_stale(item: AcquiredFile) -> bool:
    if item.parse and item.parse.get("adapter_version") != adapters.ADAPTER_VERSION:
        return True
    if item.archive and item.archive.get("archive_version") != archive.ARCHIVE_VERSION:
        return True
    return False


def _records_in(item: AcquiredFile) -> int:
    if item.archive:
        total = 0
        for member in item.archive.get("members") or []:
            if not isinstance(member, dict):
                continue
            parse = member.get("parse")
            if isinstance(parse, dict) and isinstance(parse.get("record_count"), int):
                total += int(parse["record_count"])
        return total
    if item.parse and isinstance(item.parse.get("record_count"), int):
        return int(item.parse["record_count"])
    return 0


def _parse_status(files: tuple[AcquiredFile, ...]) -> str | None:
    if not files:
        return None
    outcomes = []
    for item in files:
        if item.archive and item.archive.get("status"):
            outcomes.append(str(item.archive["status"]))
        elif item.parse and item.parse.get("status"):
            outcomes.append(str(item.parse["status"]))
    if not outcomes:
        return None
    if "parsed" in outcomes:
        return "parsed"
    if "failed" in outcomes:
        return "failed"
    return "skipped"


# ---------------------------------------------------------------------------
# Per-entry acquisition
# ---------------------------------------------------------------------------


@dataclass
class AcquireContext:
    config: Config
    transport: Transport
    now: str
    store: ContentStore
    budget: RunBudget
    limiter: HostRateLimiter
    cache: AcquisitionCache
    _mutex: threading.Lock = field(default_factory=threading.Lock)
    cache_hits: int = 0
    cache_misses: int = 0

    def record_cache(self, *, hit: bool) -> None:
        with self._mutex:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1


def _license_policy(row: dict[str, object]) -> tuple[str, str | None]:
    license_block = row.get("license") if isinstance(row.get("license"), dict) else {}
    dataset = license_block.get("dataset") if isinstance(license_block, dict) else {}
    if not isinstance(dataset, dict):
        return "metadata_only_review", "license_unknown_metadata_only"
    policy = str(dataset.get("policy") or "metadata_only_review")
    reason = dataset.get("reason_code")
    return policy, reason if isinstance(reason, str) else None


def _skip_reason_for_row(row: dict[str, object]) -> str:
    policy, reason = _license_policy(row)
    if reason:
        return reason
    if policy == "blocked":
        return "license_restricted"
    if policy == "index_content":
        return "not_selected"
    return "license_unknown_metadata_only"


def acquire_entry(
    row: dict[str, object],
    resolution: Resolution | None,
    ctx: AcquireContext,
) -> Acquisition:
    entry_id = str(row.get("entry_id") or "")
    policy, policy_reason = _license_policy(row)
    if policy != "index_content":
        return Acquisition(
            entry_id=entry_id,
            status="skipped",
            reason_code=policy_reason or "license_restricted",
            detail=f"license policy is {policy}",
            checked_at=ctx.now,
            resolution_method=resolution.method if resolution else None,
        )
    if resolution is None:
        return Acquisition(
            entry_id=entry_id,
            status="skipped",
            reason_code="not_resolved",
            detail="no resolution evidence; run the resolve stage first",
            checked_at=ctx.now,
        )
    if resolution.status != "resolved":
        # Trust the resolution's own retryability rather than its stage status:
        # an "unresolved" result can still be retryable (landing_page_only) and
        # a terminal network reason must not become a deferral.
        retryable = bool(resolution.retryable) or (
            resolution.reason_code in RETRYABLE_REASONS
        )
        status = "deferred" if retryable else "failed"
        return Acquisition(
            entry_id=entry_id,
            status=status,
            reason_code=resolution.reason_code or "not_resolved",
            detail=resolution.detail,
            retryable=retryable,
            next_retry_at=resolution.next_retry_at,
            attempts_count=resolution.attempts_count,
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )

    all_refs = [SourceRef.from_resolved(source) for source in resolution.sources]
    sliced = all_refs[: max(1, ctx.config.acquire_max_files_per_dataset)]
    repo_refs = [ref for ref in sliced if ref.kind in REPOSITORY_KINDS]
    refs = [ref for ref in sliced if ref.kind not in REPOSITORY_KINDS]
    if not sliced:
        return Acquisition(
            entry_id=entry_id,
            status="skipped",
            reason_code="no_source_url",
            detail="resolution produced no file sources",
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )
    if not refs:
        return Acquisition(
            entry_id=entry_id,
            status="skipped",
            reason_code="repository_requires_clone",
            detail="resolution identified a repository, not a downloadable file",
            files=tuple(
                AcquiredFile(
                    uri=ref.uri,
                    member=ref.member,
                    kind=ref.kind,
                    status="skipped",
                    reason_code="repository_requires_clone",
                    media_type=ref.media_type,
                    declared_bytes=ref.declared_bytes,
                )
                for ref in repo_refs
            ),
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )

    metadata_sha256 = str(row.get("metadata_sha256") or "")
    # The key covers every resolved source, so raising --max-files does not
    # silently reuse an artifact that holds fewer files than this run wants.
    key = ctx.cache.key_for(entry_id, metadata_sha256, all_refs)
    prior = (
        ctx.cache.get(entry_id, metadata_sha256, all_refs, now=ctx.now)
        if ctx.config.acquire_reuse_enabled
        else None
    )
    # Reuse is per file, not per entry: a source that failed last time must be
    # retried without re-downloading the siblings that already succeeded.
    reusable = {
        item.uri: item
        for item in (prior.files if prior is not None else ())
        if item.status == "acquired" and item.sha256 and ctx.store.has(item.sha256)
    }
    ctx.record_cache(hit=bool(reusable))

    # A mixed resolution (files plus a repository) keeps the repository entry
    # visible as an explicit skip rather than dropping it.
    files: list[AcquiredFile] = [
        AcquiredFile(
            uri=ref.uri,
            member=ref.member,
            kind=ref.kind,
            status="skipped",
            reason_code="repository_requires_clone",
            media_type=ref.media_type,
            declared_bytes=ref.declared_bytes,
        )
        for ref in repo_refs
    ]
    records = 0
    bytes_total = 0
    dataset_limit = ctx.config.max_dataset_bytes
    first_failure: AcquireError | None = None
    budget_failure: BudgetExceeded | None = None
    dataset_started = False
    downloaded = 0

    def start_dataset() -> BudgetExceeded | None:
        nonlocal dataset_started
        if dataset_started:
            return None
        try:
            ctx.budget.check_run()
            ctx.budget.started_dataset()
        except BudgetExceeded as exc:
            return exc
        dataset_started = True
        return None

    for ref in refs:
        cached_item = reusable.get(ref.uri)
        if cached_item is not None:
            item = replace(cached_item, cached=True)
            if _parse_is_stale(item):
                # Bytes are immutable and content-addressed; a parser bump only
                # needs a re-parse of the stored blob, never a refetch.
                try:
                    item, count = _parse_blob(item, ctx)
                    records += count
                except Exception:
                    records += _records_in(item)
            else:
                records += _records_in(item)
            files.append(item)
            bytes_total += item.bytes
            continue

        budget_error = start_dataset()
        if budget_error is not None:
            budget_failure = budget_error
            first_failure = first_failure or AcquireError(
                budget_error.code, budget_error.detail
            )
            break
        remaining = min(dataset_limit - bytes_total, ctx.budget.remaining_bytes())
        if remaining <= 0:
            first_failure = first_failure or AcquireError(
                "oversized_declared",
                "dataset byte cap reached before all files were fetched",
            )
            break
        host = _host(ref.uri)
        try:
            with ctx.limiter.slot(host):
                item = _download_source(ref, ctx, max_bytes=remaining)
        except AcquireError as exc:
            files.append(
                AcquiredFile(
                    uri=ref.uri,
                    member=ref.member,
                    kind=ref.kind,
                    status="failed",
                    reason_code=exc.code,
                    detail=exc.detail,
                    media_type=ref.media_type,
                    declared_bytes=ref.declared_bytes,
                )
            )
            first_failure = first_failure or exc
            if exc.code in {"run_budget_exhausted", "disk_guard"}:
                budget_failure = BudgetExceeded(exc.code, exc.detail)
                break
            continue
        except NetguardError as exc:  # defensive: _download_source already maps these
            files.append(
                AcquiredFile(
                    uri=ref.uri,
                    member=ref.member,
                    kind=ref.kind,
                    status="failed",
                    reason_code=exc.code,
                    detail=str(exc),
                    media_type=ref.media_type,
                )
            )
            first_failure = first_failure or AcquireError(exc.code, str(exc))
            continue
        downloaded += 1
        bytes_total += item.bytes
        files.append(item)

    acquired = [item for item in files if item.status == "acquired"]
    failures = [item for item in files if item.status == "failed"]
    # Parse only what this run downloaded; reused files were parsed above (and
    # re-parsed only when the adapter version moved).
    parsed_files: list[AcquiredFile] = []
    for item in files:
        if item.status != "acquired" or item.cached or item.parse or item.archive:
            parsed_files.append(item)
            continue
        try:
            updated, count = _parse_blob(item, ctx)
        except Exception:  # parsing must never lose an acquired blob
            updated, count = (
                replace(
                    item,
                    parse={
                        "status": "failed",
                        "reason_code": "parse_error",
                        "detail": "adapter raised",
                    },
                ),
                0,
            )
        records += count
        parsed_files.append(updated)
    files = parsed_files

    files_tuple = tuple(files)
    if acquired and not failures and budget_failure is None:
        acquisition = Acquisition(
            entry_id=entry_id,
            status="acquired" if downloaded else "unchanged",
            reason_code=None,
            detail=f"{len(acquired)} file(s) stored ({downloaded} downloaded this run)",
            files=files_tuple,
            bytes_total=bytes_total,
            records_total=records,
            parse_status=_parse_status(files_tuple),
            cached=not downloaded,
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )
    elif acquired and failures:
        # A partial dataset is not a success: the entry keeps a retryable or
        # terminal reason while the stored files stay recorded and reusable.
        code = first_failure.code if first_failure else "partial_download"
        acquisition = Acquisition(
            entry_id=entry_id,
            status="deferred" if code in RETRYABLE_REASONS else "failed",
            reason_code=code,
            detail=(
                f"{len(acquired)} file(s) stored, {len(failures)} failed: "
                + (first_failure.detail if first_failure else code)
            ),
            files=files_tuple,
            bytes_total=bytes_total,
            records_total=records,
            parse_status=_parse_status(files_tuple),
            retryable=code in RETRYABLE_REASONS,
            next_retry_at=_retry_at(ctx.now, ctx.config)
            if code in RETRYABLE_REASONS
            else None,
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )
    elif budget_failure is not None:
        acquisition = Acquisition(
            entry_id=entry_id,
            status="deferred",
            reason_code=budget_failure.code,
            detail=budget_failure.detail,
            files=files_tuple,
            bytes_total=bytes_total,
            retryable=True,
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )
    else:
        code = first_failure.code if first_failure else "no_source_url"
        detail = first_failure.detail if first_failure else "no source could be acquired"
        acquisition = Acquisition(
            entry_id=entry_id,
            status="deferred" if code in RETRYABLE_REASONS else "failed",
            reason_code=code,
            detail=detail,
            files=files_tuple,
            bytes_total=bytes_total,
            retryable=code in RETRYABLE_REASONS,
            next_retry_at=_retry_at(ctx.now, ctx.config)
            if code in RETRYABLE_REASONS
            else None,
            checked_at=ctx.now,
            resolution_method=resolution.method,
        )

    # The artifact is cached whenever content was stored, even for a partial
    # (deferred) entry: the stored blobs are what makes the next run incremental.
    if ctx.config.acquire_reuse_enabled and acquired:
        try:
            ctx.cache.put(acquisition, key)
        except OSError:
            pass
    return acquisition


def _host(uri: str) -> str:
    try:
        return (urllib.parse.urlsplit(uri).hostname or "").lower()
    except ValueError:
        return ""


def _retry_at(now: str, config: Config) -> str:
    parsed = _parse_iso(now)
    if parsed is None:
        return now
    return (
        (parsed + timedelta(seconds=max(60, config.resolve_retry_backoff_seconds)))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Selection, completion and accounting
# ---------------------------------------------------------------------------


def select_entries(
    rows: list[dict[str, object]],
    *,
    sample: int | None = None,
    limit: int | None = None,
) -> tuple[set[str], bool]:
    """Select acquirable entries: ``index_content`` only, deterministic."""
    candidates = [
        row
        for row in rows
        if row.get("status") != "failed"
        and _license_policy(row)[0] == "index_content"
    ]
    candidates.sort(key=lambda row: str(row.get("entry_id")))
    if sample is not None:
        chosen = _stratified_sample(candidates, max(0, sample))
        return {str(row.get("entry_id")) for row in chosen}, len(chosen) < len(candidates)
    if limit is not None:
        chosen = candidates[: max(0, limit)]
        return {str(row.get("entry_id")) for row in chosen}, len(chosen) < len(candidates)
    return {str(row.get("entry_id")) for row in candidates}, False


def select_named(
    rows: list[dict[str, object]], entry_ids: list[str]
) -> tuple[set[str], list[str]]:
    """Select explicit entry ids; returns the accepted set and unknown ids."""
    candidates = {
        str(row.get("entry_id"))
        for row in rows
        if row.get("status") != "failed" and _license_policy(row)[0] == "index_content"
    }
    accepted: set[str] = set()
    unknown: list[str] = []
    for entry_id in entry_ids:
        if entry_id in candidates:
            accepted.add(entry_id)
        else:
            unknown.append(entry_id)
    return accepted, unknown


def _primary_format(row: dict[str, object]) -> str:
    classes = row.get("classes") if isinstance(row.get("classes"), dict) else {}
    formats = classes.get("format") if isinstance(classes, dict) else None
    if isinstance(formats, list) and formats:
        return str(formats[0])
    return "unknown"


def _stratified_sample(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Round-robin by primary format so one sample spans the format classes."""
    if count <= 0:
        return []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(_primary_format(row), []).append(row)
    order = sorted(groups)
    indexes = {key: 0 for key in order}
    chosen: list[dict[str, object]] = []
    while len(chosen) < count:
        progressed = False
        for key in order:
            if len(chosen) >= count:
                break
            index = indexes[key]
            if index < len(groups[key]):
                chosen.append(groups[key][index])
                indexes[key] = index + 1
                progressed = True
        if not progressed:
            break
    return chosen


def budget_acquisition(row: dict[str, object], now: str, config: Config) -> Acquisition:
    return Acquisition(
        entry_id=str(row.get("entry_id") or ""),
        status="deferred",
        reason_code="run_budget_exhausted",
        detail="entry was outside the bounded sample for this run",
        retryable=True,
        next_retry_at=_retry_at(now, config),
        checked_at=now,
        attempts_count=1,
    )


def complete_acquisitions(
    rows: list[dict[str, object]],
    acquisitions: list[Acquisition],
    *,
    now: str,
    config: Config,
) -> list[Acquisition]:
    """One acquisition per discovered row; every non-success carries a reason."""
    by_entry = {item.entry_id: item for item in acquisitions}
    completed: list[Acquisition] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        result = by_entry.get(entry_id)
        if result is not None:
            completed.append(result)
            continue
        if row.get("status") == "failed":
            reason = str(row.get("reason_code") or "catalog_entry_invalid")
            completed.append(
                Acquisition(
                    entry_id=entry_id,
                    status="failed",
                    reason_code=reason,
                    detail=str(row.get("error") or reason)[:500],
                    retryable=reason in RETRYABLE_REASONS,
                    checked_at=now,
                )
            )
            continue
        policy, reason = _license_policy(row)
        if policy != "index_content":
            completed.append(
                Acquisition(
                    entry_id=entry_id,
                    status="skipped",
                    reason_code=reason or "license_restricted",
                    detail=f"license policy is {policy}",
                    checked_at=now,
                )
            )
            continue
        completed.append(budget_acquisition(row, now, config))
    return completed


@dataclass
class AcquisitionAccounting:
    ok: bool
    discovered: int
    status_total: int
    status_counts: dict[str, int]
    missing_reason: list[str]
    problems: list[str]


def check_acquisition_accounting(
    acquisitions: list[Acquisition], discovered: int
) -> AcquisitionAccounting:
    status_counts: dict[str, int] = {}
    missing_reason: list[str] = []
    problems: list[str] = []
    for item in acquisitions:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
        if item.status not in VALID_ACQUISITION_STATUSES:
            problems.append(f"invalid_status:{item.status}")
        if item.status not in SUCCESS_ACQUISITION_STATUSES and not item.reason_code:
            missing_reason.append(item.entry_id)
    status_total = sum(status_counts.values())
    ok = True
    if len(acquisitions) != discovered:
        ok = False
        problems.append(f"row_count:{len(acquisitions)}!=discovered:{discovered}")
    if status_total != discovered:
        ok = False
        problems.append(f"status_total:{status_total}!=discovered:{discovered}")
    if missing_reason:
        ok = False
        problems.append(f"missing_reason:{len(missing_reason)}")
    if problems:
        ok = False
    return AcquisitionAccounting(
        ok=ok,
        discovered=discovered,
        status_total=status_total,
        status_counts=dict(sorted(status_counts.items())),
        missing_reason=sorted(missing_reason),
        problems=sorted(set(problems)),
    )


def acquisition_manifest_bytes(
    rows: list[dict[str, object]], acquisitions: list[Acquisition]
) -> bytes:
    by_entry = {item.entry_id: item for item in acquisitions}
    lines: list[str] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        result = by_entry.get(entry_id)
        if result is None:
            continue
        catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
        policy, _reason = _license_policy(row)
        artifact = {
            "schema_version": 1,
            "entry_id": entry_id,
            "metadata_sha256": row.get("metadata_sha256"),
            "catalog": {
                "repo": catalog.get("repo"),
                "commit": catalog.get("commit"),
                "path": catalog.get("path"),
                "meta_url": catalog.get("meta_url"),
            },
            "category": row.get("category"),
            "title": row.get("title"),
            "disposition": row.get("disposition"),
            "license_policy": policy,
            "formats": (row.get("classes") or {}).get("format"),
            "acquisition": result.to_artifact(),
        }
        lines.append(canonical_json(artifact))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_acquisition_manifest(
    rows: list[dict[str, object]], acquisitions: list[Acquisition], path: Path
) -> tuple[bool, Path]:
    from .manifest import atomic_write

    data = acquisition_manifest_bytes(rows, acquisitions)
    existing = None
    try:
        existing = path.read_bytes()
    except OSError:
        existing = None
    changed = existing != data
    if changed:
        atomic_write(path, data)
    return changed, path


def acquisition_outcome_row(
    row: dict[str, object],
    acquisition: Acquisition,
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> dict[str, object]:
    outcome_code = f"{acquisition.status}:{acquisition.reason_code or ''}"
    event_id = sha256_hex(
        canonical_json(
            {
                "entry_id": acquisition.entry_id,
                "stage": "acquire",
                "input_hash": row.get("metadata_sha256"),
                "outcome_code": outcome_code,
                "attempt": 1,
            }
        )
    )
    first = next(
        (item for item in acquisition.files if item.status == "acquired"), None
    )
    return {
        "schema_version": 1,
        "event_id": event_id,
        "run_id": run_id,
        "entry_id": acquisition.entry_id,
        "catalog_commit": catalog_commit,
        "stage": "acquire",
        "status": acquisition.status,
        "reason_code": acquisition.reason_code,
        "detail": acquisition.detail[:500] if acquisition.detail else None,
        "source_uri": sanitize_url(first.uri) if first else None,
        "source_sha256": first.sha256 if first else None,
        "bytes": acquisition.bytes_total,
        "files_acquired": acquisition.files_acquired,
        "records_total": acquisition.records_total,
        "etag": first.etag if first else None,
        "last_modified": first.last_modified if first else None,
        "retrieved_at": retrieved_at,
        "attempts": acquisition.attempts_count,
        "retryable": acquisition.retryable,
        "next_retry_at": acquisition.next_retry_at,
        "knowledge_source_id": None,
        "revision_id": None,
        "revision_sequence": None,
        "knowledge_decision": None,
    }


def append_acquisition_outcomes(
    rows: list[dict[str, object]],
    acquisitions: list[Acquisition],
    path: Path,
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> int:
    from .manifest import append_outcome_rows

    by_entry = {item.entry_id: item for item in acquisitions}
    outcomes: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        result = by_entry.get(str(row.get("entry_id") or ""))
        if result is None:
            continue
        outcomes.append(
            acquisition_outcome_row(
                row,
                result,
                run_id=run_id,
                catalog_commit=catalog_commit,
                retrieved_at=retrieved_at,
            )
        )
    return append_outcome_rows(outcomes, path)


def acquisition_run_id(
    config: Config,
    snapshot,
    *,
    selected_ids: set[str],
    sample: int | None,
    limit: int | None,
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "stage": "acquire",
                "acquire_version": ACQUIRE_VERSION,
                "adapter_version": adapters.ADAPTER_VERSION,
                "archive_version": archive.ARCHIVE_VERSION,
                "schema_version": config.schema_version,
                "catalog_repo": snapshot.repo,
                "commit": snapshot.commit,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "sample": sample,
                "limit": limit,
                "selection_hash": sha256_hex(canonical_json(sorted(selected_ids)))[:16],
                "max_file_bytes": config.max_file_bytes,
                "max_dataset_bytes": config.max_dataset_bytes,
                "max_run_bytes": config.max_run_bytes,
            }
        )
    )[:16]


def build_context(
    config: Config,
    *,
    transport: Transport | None = None,
    store: ContentStore | None = None,
    budget: RunBudget | None = None,
    limiter: HostRateLimiter | None = None,
    cache: AcquisitionCache | None = None,
    now: str | None = None,
) -> AcquireContext:
    return AcquireContext(
        config=config,
        transport=transport or build_transport(config),
        now=now or utc_now(),
        store=store or ContentStore(config),
        budget=budget or RunBudget(config),
        limiter=limiter or HostRateLimiter(config.per_host_rps),
        cache=cache or AcquisitionCache(config),
    )


def acquire_entries(
    rows: list[dict[str, object]],
    resolutions: dict[str, Resolution],
    selected_ids: set[str],
    config: Config,
    *,
    context: AcquireContext | None = None,
) -> tuple[list[Acquisition], int, AcquireContext]:
    """Acquire every selected row; returns one result per selected row."""
    ctx = context or build_context(config)
    results: list[Acquisition] = []
    unhandled = 0
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        if entry_id not in selected_ids:
            continue
        try:
            results.append(acquire_entry(row, resolutions.get(entry_id), ctx))
        except Exception as exc:  # pragma: no cover - defensive isolation
            unhandled += 1
            results.append(
                Acquisition(
                    entry_id=entry_id,
                    status="failed",
                    reason_code="acquire_error",
                    detail=type(exc).__name__,
                    checked_at=ctx.now,
                )
            )
    return results, unhandled, ctx


@dataclass
class PlanEntry:
    entry_id: str
    title: str | None
    formats: list[str]
    method: str | None
    sources: list[dict[str, object]]


def build_plan(
    rows: list[dict[str, object]],
    resolutions: dict[str, Resolution],
    selected_ids: set[str],
    config: Config,
) -> list[dict[str, object]]:
    """A side-effect-free description of what an ``acquire`` run would fetch."""
    plan: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        if entry_id not in selected_ids:
            continue
        resolution = resolutions.get(entry_id)
        policy, _reason = _license_policy(row)
        sources: list[dict[str, object]] = []
        status = "planned"
        reason = None
        if policy != "index_content":
            status, reason = "skipped", "license_policy"
        elif resolution is None:
            status, reason = "skipped", "not_resolved"
        elif resolution.status != "resolved":
            status, reason = "blocked", resolution.reason_code
        else:
            for source in list(resolution.sources)[: max(1, config.acquire_max_files_per_dataset)]:
                sources.append(
                    {
                        "uri": sanitize_url(source.uri),
                        "kind": source.kind,
                        "media_type": source.media_type,
                        "declared_bytes": source.size_bytes,
                        "checksum": getattr(source, "checksum", None),
                    }
                )
        classes = row.get("classes") if isinstance(row.get("classes"), dict) else {}
        formats = classes.get("format") if isinstance(classes, dict) else []
        plan.append(
            {
                "entry_id": entry_id,
                "title": row.get("title"),
                "disposition": row.get("disposition"),
                "license_policy": policy,
                "formats": formats if isinstance(formats, list) else [],
                "status": status,
                "reason_code": reason,
                "resolution_method": resolution.method if resolution else None,
                "sources": sources,
            }
        )
    return plan
