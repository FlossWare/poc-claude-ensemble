"""PR 2: resolve a catalog entry to its authoritative dataset source.

Resolution never downloads dataset bodies.  It tries, in order:

1. ``sources[].access_url`` (a ``str``, ``list[str]`` or ``list[dict]``);
2. the GitHub ``blob`` -> ``raw`` transform;
3. documented host APIs (GitHub, Zenodo, Hugging Face, DOI content
   negotiation, archive.org, Dataverse);
4. one bounded same-host landing-page fetch.

Every candidate attempt is recorded with a stable reason code.  The module is
idempotent and resumable: per-entry results are cached under the data root and
re-used until the entry's metadata hash changes, a resolved cache entry ages
past ``RECHECK_AFTER_DAYS``, or a retryable failure reaches ``next_retry_at``.

All network access goes through :mod:`ingestion.datasets.netguard` (SSRF guard,
pinned DNS, redirect revalidation, HTTPS downgrade refusal).  JSON and HTML are
untrusted input: response sizes, link counts and JSON shapes are bounded.
"""
from __future__ import annotations

import html.parser
import json
import os
import re
import stat
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .model import canonical_json, sanitize_url, sha256_hex
from .netguard import NetguardError, Transport

RESOLVER_VERSION = 3

ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tbz2",
    ".txz",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".zst",
)

DATA_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".tab",
        ".json",
        ".jsonl",
        ".ndjson",
        ".xml",
        ".parquet",
        ".arrow",
        ".feather",
        ".avro",
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
        ".geojson",
        ".shp",
        ".nc",
        ".nc4",
        ".h5",
        ".hdf5",
        ".sqlite",
        ".db",
        ".xlsx",
        ".xls",
        ".sav",
        ".dta",
        ".rds",
        ".yaml",
        ".yml",
        ".wml",
    }
)

DATA_MEDIA_TYPES = frozenset(
    {
        "text/csv",
        "text/tab-separated-values",
        "text/plain",
        "text/markdown",
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "text/xml",
        "application/octet-stream",
        "application/parquet",
        "application/vnd.apache.parquet",
        "application/geo+json",
        "application/pdf",
        "application/zip",
        "application/gzip",
        "application/x-tar",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-7z-compressed",
        "application/vnd.rar",
    }
)

HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

#: Reason codes that a later run may retry.  Everything else is terminal until
#: the entry metadata changes.
RETRYABLE_REASONS = frozenset(
    {
        "blocked",
        "not_found",
        "rate_limited",
        "timeout",
        "dns_failure",
        "tls_failure",
        "server_error",
        "connection_failed",
        "oversized_declared",
        "oversized_stream",
        "landing_page_only",
        "resolver_not_implemented",
        "doi_unresolved",
        "run_budget_exhausted",
        "dependency_unavailable",
    }
)

#: When several candidates fail, report the most informative reason.  Lower
#: index wins.  Generic "no direct file" outcomes sort last so a concrete
#: network or policy reason is preferred.
REASON_PRIORITY = (
    "invalid_url",
    "unsupported_scheme",
    "no_source_url",
    "ssrf_blocked",
    "unsafe_redirect",
    "too_many_redirects",
    "gated_authentication",
    "robots_disallowed",
    "requires_interaction",
    "requires_javascript",
    "gone",
    "not_found",
    "rate_limited",
    "blocked",
    "timeout",
    "dns_failure",
    "tls_failure",
    "connection_failed",
    "server_error",
    "oversized_declared",
    "oversized_stream",
    "doi_unresolved",
    "landing_page_only",
    "resolver_not_implemented",
    "http_error",
    "parse_error",
    "resolver_error",
)
_REASON_RANK = {reason: index for index, reason in enumerate(REASON_PRIORITY)}

VALID_RESOLUTION_STATUSES = frozenset(
    {"resolved", "unresolved", "deferred", "failed"}
)
SUCCESS_RESOLUTION_STATUSES = frozenset({"resolved"})

_NETGUARD_REASONS = {
    "invalid_url": "invalid_url",
    "unsupported_scheme": "unsupported_scheme",
    # The design's resolution taxonomy folds non-standard ports into
    # invalid_url rather than surfacing the transport's unsafe_port.
    "unsafe_port": "invalid_url",
    "ssrf_blocked": "ssrf_blocked",
    "unsafe_redirect": "unsafe_redirect",
    "too_many_redirects": "too_many_redirects",
    "gated_authentication": "gated_authentication",
    "blocked": "blocked",
    "not_found": "not_found",
    "gone": "gone",
    "rate_limited": "rate_limited",
    "timeout": "timeout",
    "dns_failure": "dns_failure",
    "tls_failure": "tls_failure",
    "server_error": "server_error",
    "connection_failed": "connection_failed",
    "oversized_declared": "oversized_declared",
    "oversized_stream": "oversized_stream",
}

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$", re.IGNORECASE)
_ZENODO_PATH_RE = re.compile(r"^/(?:records|record)/(\d+)(?:/|$)", re.IGNORECASE)
_GITHUB_BLOB_RE = re.compile(
    r"^/([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+)$", re.IGNORECASE
)
_GITHUB_TREE_RE = re.compile(r"^/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.*))?$", re.IGNORECASE)
_GITHUB_REPO_RE = re.compile(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$")
_HF_DATASET_RE = re.compile(
    r"^/datasets/([^/]+(?:/[^/]+)?)(?:/(?:blob|tree|resolve)/([^/]+)(?:/(.*))?)?$",
    re.IGNORECASE,
)
_ARCHIVE_DETAILS_RE = re.compile(r"^/(?:details|download)/([^/]+)(?:/|$)", re.IGNORECASE)
_JS_HINT_RE = re.compile(
    r"enable\s+javascript|javascript\s+is\s+required|requires\s+javascript|"
    r"please\s+enable\s+js|turn\s+on\s+javascript",
    re.IGNORECASE,
)
_INTERACTION_HINT_RE = re.compile(
    r"sign\s+in\s+to|log\s+in\s+to|request\s+access|accept\s+the\s+terms|"
    r"click\s+.{0,20}download|add\s+to\s+cart",
    re.IGNORECASE,
)
_PASSWORD_FORM_RE = re.compile(
    r"<form[^>]*(?:password|login|signin|sign-in)", re.IGNORECASE
)

_USER_AGENT_SKIP_SCHEMES = frozenset(
    {"javascript", "mailto", "tel", "data", "file", "ftp", "about", "blob"}
)


class Reject(Exception):
    """A candidate failed with a stable resolution reason code."""

    def __init__(
        self, reason_code: str, detail: str = "", *, http_status: int | None = None
    ) -> None:
        super().__init__(detail or reason_code)
        self.reason_code = reason_code
        self.detail = detail or reason_code
        self.http_status = http_status


@dataclass(frozen=True)
class Candidate:
    url: str
    origin: str


@dataclass(frozen=True)
class Attempt:
    url: str
    method: str
    status: str
    reason_code: str | None = None
    http_status: int | None = None
    detail: str | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "url": sanitize_url(self.url),
            "method": self.method,
            "status": self.status,
            "reason_code": self.reason_code,
            "http_status": self.http_status,
            "detail": self.detail[:300] if self.detail else None,
        }


@dataclass(frozen=True)
class ResolvedSource:
    uri: str
    kind: str
    method: str
    media_type: str | None = None
    size_bytes: int | None = None
    version: str | None = None
    member: str | None = None
    evidence_url: str | None = None
    #: Upstream-declared digest, ``<algorithm>:<hex>`` (for example
    #: ``md5:...`` from a Zenodo or archive.org file listing).  Acquisition
    #: verifies it before adopting the bytes.
    checksum: str | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "uri": sanitize_url(self.uri),
            "kind": self.kind,
            "method": self.method,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "version": self.version,
            "member": self.member,
            "evidence_url": sanitize_url(self.evidence_url),
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class Resolution:
    entry_id: str
    status: str
    reason_code: str | None
    checked_at: str
    sources: tuple[ResolvedSource, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    method: str | None = None
    kind: str | None = None
    version: str | None = None
    total_sources: int | None = None
    detail: str | None = None
    retryable: bool = False
    next_retry_at: str | None = None
    attempts_count: int = 1
    cached: bool = False

    @property
    def next_action(self) -> str:
        if self.status == "resolved":
            return "acquire"
        if self.reason_code == "run_budget_exhausted":
            return "resume"
        if self.reason_code and self.reason_code.startswith("catalog_"):
            return "fix_metadata"
        if self.reason_code in {"invalid_url", "unsupported_scheme", "no_source_url"}:
            return "fix_metadata"
        if self.status == "deferred":
            return "retry"
        return "review"

    def to_artifact(self) -> dict[str, object]:
        # Budget rows were never checked: a per-run timestamp here would make
        # the resolution manifest differ on every re-run.
        budget = self.reason_code == "run_budget_exhausted"
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "method": self.method,
            "kind": self.kind,
            "version": self.version,
            "source_count": len(self.sources),
            "total_sources": self.total_sources,
            "retryable": self.retryable,
            "next_retry_at": None if budget else self.next_retry_at,
            "checked_at": None if budget else self.checked_at,
            "attempts_count": self.attempts_count,
            "next_action": self.next_action,
            "detail": self.detail[:500] if self.detail else None,
            "sources": [source.to_artifact() for source in self.sources],
            "attempts": [attempt.to_artifact() for attempt in self.attempts],
        }

    def to_cache(self) -> dict[str, object]:
        return {
            "resolver_version": RESOLVER_VERSION,
            "entry_id": self.entry_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "checked_at": self.checked_at,
            "method": self.method,
            "kind": self.kind,
            "version": self.version,
            "total_sources": self.total_sources,
            "detail": self.detail,
            "retryable": self.retryable,
            "next_retry_at": self.next_retry_at,
            "attempts_count": self.attempts_count,
            "sources": [source.to_artifact() for source in self.sources],
            "attempts": [attempt.to_artifact() for attempt in self.attempts],
        }

    @classmethod
    def from_cache(cls, payload: dict[str, object]) -> "Resolution":
        entry_id = payload.get("entry_id")
        status = payload.get("status")
        checked_at = payload.get("checked_at")
        if not isinstance(entry_id, str) or not isinstance(status, str):
            raise ValueError("invalid cached resolution")
        if status not in VALID_RESOLUTION_STATUSES:
            raise ValueError("invalid cached resolution status")
        if not isinstance(checked_at, str):
            raise ValueError("invalid cached resolution timestamp")
        sources = tuple(
            _source_from_cache(item) for item in payload.get("sources") or [] if isinstance(item, dict)
        )
        attempts = tuple(
            _attempt_from_cache(item) for item in payload.get("attempts") or [] if isinstance(item, dict)
        )
        return cls(
            entry_id=entry_id,
            status=status,
            reason_code=_optional_str(payload.get("reason_code")),
            checked_at=checked_at,
            sources=sources,
            attempts=attempts,
            method=_optional_str(payload.get("method")),
            kind=_optional_str(payload.get("kind")),
            version=_optional_str(payload.get("version")),
            total_sources=_optional_int(payload.get("total_sources")),
            detail=_optional_str(payload.get("detail")),
            retryable=bool(payload.get("retryable")),
            next_retry_at=_optional_str(payload.get("next_retry_at")),
            attempts_count=_optional_int(payload.get("attempts_count")) or 1,
            cached=True,
        )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _source_from_cache(payload: dict[str, object]) -> ResolvedSource:
    return ResolvedSource(
        uri=str(payload.get("uri") or ""),
        kind=str(payload.get("kind") or "direct_file"),
        method=str(payload.get("method") or "cached"),
        media_type=_optional_str(payload.get("media_type")),
        size_bytes=_optional_int(payload.get("size_bytes")),
        version=_optional_str(payload.get("version")),
        member=_optional_str(payload.get("member")),
        evidence_url=_optional_str(payload.get("evidence_url")),
        checksum=_optional_str(payload.get("checksum")),
    )


def _attempt_from_cache(payload: dict[str, object]) -> Attempt:
    return Attempt(
        url=str(payload.get("url") or ""),
        method=str(payload.get("method") or "cached"),
        status=str(payload.get("status") or "rejected"),
        reason_code=_optional_str(payload.get("reason_code")),
        http_status=_optional_int(payload.get("http_status")),
        detail=_optional_str(payload.get("detail")),
    )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_after(value: str, seconds: int) -> str:
    parsed = _parse_iso(value) or datetime.now(timezone.utc)
    return (
        (parsed + timedelta(seconds=max(0, seconds)))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _host_for(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "_").lower()
    except ValueError:
        return "_"


def _scheme_reason(url: str) -> str | None:
    """Classify a URL before any network call, without resolving DNS."""
    text = url.strip()
    if not text:
        return "invalid_url"
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return "invalid_url"
    match = _SCHEME_RE.match(text)
    if not match:
        return "invalid_url"
    scheme = match.group(1).lower()
    if scheme in {"ftp", "s3", "s3a", "gs", "gcs", "file", "az", "ssh", "git"}:
        return "unsupported_scheme"
    if scheme not in {"http", "https"}:
        # Includes the observed `hhttps` typo and other invented schemes.
        return "invalid_url"
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return "invalid_url"
    if not parsed.hostname:
        return "invalid_url"
    if parsed.username is not None or parsed.password is not None:
        return "invalid_url"
    try:
        port = parsed.port
    except ValueError:
        return "invalid_url"
    expected = 443 if scheme == "https" else 80
    if port is not None and port != expected:
        return "invalid_url"
    if any(ord(char) < 32 or ord(char) == 127 for char in parsed.hostname):
        return "invalid_url"
    return None


def _precheck(url: str) -> urllib.parse.SplitResult:
    reason = _scheme_reason(url)
    if reason is not None:
        raise Reject(reason, "URL is not a supported HTTP(S) source")
    try:
        return urllib.parse.urlsplit(url.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise Reject("invalid_url", "URL could not be parsed") from exc


def _media_type(headers: dict[str, str]) -> str | None:
    value = headers.get("content-type")
    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _content_length(headers: dict[str, str]) -> int | None:
    value = headers.get("content-length")
    if value and value.strip().isdigit():
        return int(value.strip())
    return None


def _is_html(media_type: str | None, url: str) -> bool:
    if media_type in HTML_MEDIA_TYPES:
        return True
    if media_type is not None:
        return False
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith((".html", ".htm", "/"))


def _has_archive_suffix(name: str) -> bool:
    low = name.lower().split("?", 1)[0].split("#", 1)[0]
    return any(low.endswith(suffix) for suffix in ARCHIVE_SUFFIXES) or low.endswith(".zip")


def _extension(name: str) -> str:
    low = name.lower().split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(low)[1]


def _kind_for(name: str, media_type: str | None = None) -> str:
    if _has_archive_suffix(name):
        return "archive"
    extension = _extension(name)
    if extension in DATA_EXTENSIONS:
        return "direct_file"
    if media_type in DATA_MEDIA_TYPES:
        return "direct_file"
    return "direct_file"


def _is_data_like(name: str, media_type: str | None = None) -> bool:
    if _has_archive_suffix(name):
        return True
    if _extension(name) in DATA_EXTENSIONS:
        return True
    return media_type in DATA_MEDIA_TYPES


_DOCUMENTATION_PREFIXES = (
    "readme",
    "license",
    "licence",
    "citation",
    "copying",
    "changelog",
    ".gitattributes",
    ".gitignore",
)


def _is_documentation(name: str) -> bool:
    return name.lower().startswith(_DOCUMENTATION_PREFIXES)


def _prefer_data_names(
    names: list[tuple[str, str | None, int | None, str | None]],
    *,
    allow_unknown: bool = False,
) -> list[tuple[str, str | None, int | None, str | None]]:
    """Keep data-like entries, preferring non-documentation files.

    ``allow_unknown`` is used for host file listings (Zenodo, Hugging Face,
    archive.org, Dataverse): an extension we do not recognise is still an
    authoritative dataset member, so it is resolved rather than silently
    dropped.  GitHub directory listings stay strict so a code repository is
    reported as a repository instead of a list of source files.
    """
    preferred = [
        item
        for item in names
        if _is_data_like(item[0], item[1]) and not _is_documentation(item[0])
    ]
    if preferred:
        return preferred
    if allow_unknown:
        unknown = [item for item in names if not _is_documentation(item[0])]
        if unknown:
            return unknown
    return [
        item
        for item in names
        if _is_data_like(item[0], item[1]) and not _is_documentation(item[0])
    ]


def _split_urls(value: object) -> list[str]:
    """Accept ``str``, ``dict``, ``list[str]`` and ``list[dict]`` access URLs."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return _split_urls(value.get("access_url"))
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_split_urls(item))
        return result
    return []


def extract_candidates(row: dict[str, object]) -> list[Candidate]:
    """Ordered, de-duplicated candidate URLs for one manifest row."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for source in row.get("sources") or []:
        for url in _split_urls(source):
            if url in seen:
                continue
            seen.add(url)
            candidates.append(Candidate(url, "sources.access_url"))
    homepage = row.get("homepage")
    if isinstance(homepage, str) and homepage.strip() and homepage.strip() not in seen:
        candidates.append(Candidate(homepage.strip(), "homepage"))
    return candidates


# ---------------------------------------------------------------------------
# Context, pacing and cache
# ---------------------------------------------------------------------------


class HostRateLimiter:
    """One in-flight request per host, at most ``rps`` requests per second."""

    def __init__(self, rps: float, *, clock=None, sleep=None) -> None:
        self.rps = max(0.001, float(rps))
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self._mutex = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._next: dict[str, float] = {}

    def _lock_for(self, host: str) -> threading.Lock:
        with self._mutex:
            return self._host_locks.setdefault(host, threading.Lock())

    def slot(self, host: str):
        lock = self._lock_for(host)
        lock.acquire()
        with self._mutex:
            now = self.clock()
            next_at = self._next.get(host, now)
            wait = max(0.0, next_at - now)
            self._next[host] = max(now, next_at) + (1.0 / self.rps)
        if wait > 0:
            self.sleep(wait)
        return _Release(lock)


class _Release:
    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.lock.release()


def build_transport(config: Config) -> Transport:
    return Transport(config)


@dataclass
class ResolverContext:
    config: Config
    transport: object
    now: str
    cache: "ResolutionCache | None" = None
    limiter: HostRateLimiter | None = None
    robots: dict[str, bool] = field(default_factory=dict)
    robots_lock: threading.Lock = field(default_factory=threading.Lock)
    stats_lock: threading.Lock = field(default_factory=threading.Lock)
    cache_hits: int = 0
    cache_misses: int = 0

    def record_cache(self, *, hit: bool) -> None:
        with self.stats_lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def __post_init__(self) -> None:
        if self.limiter is None:
            self.limiter = HostRateLimiter(self.config.per_host_rps)

    def request(self, url: str, **kwargs):
        assert self.limiter is not None
        with self.limiter.slot(_host_for(url)):
            return self.transport.request(url, **kwargs)


class ResolutionCache:
    """Per-entry resolution evidence, keyed by metadata hash and version."""

    def __init__(self, config: Config, *, root: Path | None = None) -> None:
        self.config = config
        self.root = root or (config.data_root / "cache" / "resolutions")

    def key_for(self, entry_id: str, metadata_sha256: str) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "resolver_version": RESOLVER_VERSION,
                    "entry_id": entry_id,
                    "metadata_sha256": metadata_sha256,
                }
            )
        )

    def path_for(self, entry_id: str, metadata_sha256: str) -> Path:
        return self.root / f"{self.key_for(entry_id, metadata_sha256)}.json"

    def get(
        self, entry_id: str, metadata_sha256: str, now: str | None = None
    ) -> Resolution | None:
        path = self.path_for(entry_id, metadata_sha256)
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            result = Resolution.from_cache(payload)
        except (TypeError, ValueError):
            return None
        if result.entry_id != entry_id:
            return None
        if not _is_fresh(result, now or utc_now(), self.config):
            return None
        return replace(result, cached=True)

    def put(self, resolution: Resolution, metadata_sha256: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path_for(resolution.entry_id, metadata_sha256)
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        data = canonical_json(resolution.to_cache()).encode("utf-8")
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


def _is_fresh(result: Resolution, now: str, config: Config) -> bool:
    checked = _parse_iso(result.checked_at)
    reference = _parse_iso(now)
    if checked is None or reference is None:
        return False
    if checked > reference:
        return False
    if result.status == "resolved":
        age = (reference - checked).total_seconds()
        return age <= max(0, config.recheck_after_days) * 86_400
    if result.retryable:
        due = _parse_iso(result.next_retry_at)
        return due is not None and reference < due
    # Terminal failures remain cached until the entry metadata changes.
    return True


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _reject_from_netguard(exc: NetguardError) -> Reject:
    code = _NETGUARD_REASONS.get(exc.code, "resolver_error")
    return Reject(code, exc.code, http_status=exc.status)


def _reason_for_status(status: int, body: bytes | None = None) -> str | None:
    if 200 <= status < 300:
        return None
    if status == 401:
        return "gated_authentication"
    if status == 403:
        snippet = (body or b"")[:4096].decode("utf-8", errors="replace").lower()
        if "rate limit" in snippet:
            return "rate_limited"
        return "blocked"
    if status == 404:
        return "not_found"
    if status == 410:
        return "gone"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "server_error"
    return "http_error"


def _decode_text(body: bytes, headers: dict[str, str], cap: int) -> str:
    content_type = headers.get("content-type") or ""
    match = re.search(r"charset=([A-Za-z0-9._\-]+)", content_type)
    encoding = match.group(1) if match else "utf-8"
    try:
        return body[:cap].decode(encoding, errors="replace")
    except LookupError:
        return body[:cap].decode("utf-8", errors="replace")


def _api_get_json(
    ctx: ResolverContext,
    url: str,
    *,
    accept: str = "application/json",
    method: str,
    expect: str = "object",
):
    cap = max(1, ctx.config.api_max_bytes)
    try:
        response = ctx.request(
            url,
            method="GET",
            headers={"Accept": accept},
            max_bytes=cap,
            store_body=True,
        )
    except NetguardError as exc:
        raise _reject_from_netguard(exc) from exc
    reason = _reason_for_status(response.status, response.body)
    if reason is not None:
        raise Reject(reason, f"API returned HTTP {response.status}", http_status=response.status)
    body = response.body or b""
    if len(body) > cap:
        raise Reject("oversized_declared", "API response exceeds the byte limit")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Reject("parse_error", "API response is not valid JSON") from exc
    if expect == "list":
        if not isinstance(payload, list):
            raise Reject("parse_error", "API response is not a JSON array")
        return payload
    if not isinstance(payload, dict):
        raise Reject("parse_error", "API response is not a JSON object")
    return payload


def _head_or_range(ctx: ResolverContext, url: str):
    try:
        response = ctx.request(url, method="HEAD")
        if response.status in {405, 501}:
            response = ctx.request(
                url, method="GET", headers={"Range": "bytes=0-0"}
            )
    except NetguardError as exc:
        raise _reject_from_netguard(exc) from exc
    return response


# ---------------------------------------------------------------------------
# Host resolvers
# ---------------------------------------------------------------------------


def _resolved_from_names(
    names: list[tuple[str, str | None, int | None, str | None]],
    *,
    make_uri,
    method: str,
    version: str | None,
    evidence_url: str | None,
    config: Config,
    kind_method=None,
    checksums: dict[str, str] | None = None,
) -> tuple[tuple[ResolvedSource, ...], int]:
    preferred = _prefer_data_names(names, allow_unknown=True)
    total = len(preferred)
    limit = max(1, config.max_resolved_sources)
    sources: list[ResolvedSource] = []
    for name, media_type, size, member in preferred[:limit]:
        kind = kind_method(name) if kind_method else _kind_for(name, media_type)
        sources.append(
            ResolvedSource(
                uri=make_uri(name),
                kind=kind,
                method=method,
                media_type=media_type,
                size_bytes=size,
                version=version,
                member=member,
                evidence_url=evidence_url,
                checksum=(checksums or {}).get(name),
            )
        )
    if not sources:
        raise Reject("landing_page_only", "source exposes no data-like files")
    return tuple(sources), total


def _github_contents(
    ctx: ResolverContext, owner: str, repo: str, ref: str, path: str
) -> tuple[ResolvedSource, ...]:
    api_path = urllib.parse.quote(path.strip("/"), safe="/")
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{api_path}"
        f"?ref={urllib.parse.quote(ref, safe='')}"
    )
    entries = _api_get_json(
        ctx, api_url, method="github_contents_api", expect="list"
    )
    names: list[tuple[str, str | None, int | None, str | None]] = []
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/"
    for item in entries:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        name = item.get("name")
        download = item.get("download_url")
        if not isinstance(name, str) or not isinstance(download, str):
            continue
        size = item.get("size") if isinstance(item.get("size"), int) else None
        names.append((name, None, size, download))
    sources: list[ResolvedSource] = []
    preferred = [
        item
        for item in _prefer_data_names(names)
        if not item[0].lower().startswith(("readme", "license", "citation", ".gitattributes"))
    ]
    for name, media_type, size, _download in preferred[: max(1, ctx.config.max_resolved_sources)]:
        sources.append(
            ResolvedSource(
                uri=raw_base + urllib.parse.quote(name, safe="/"),
                kind=_kind_for(name, media_type),
                method="github_contents_api",
                media_type=media_type,
                size_bytes=size,
                version=ref,
                member=name,
                evidence_url=api_url,
            )
        )
    if not sources:
        raise Reject("repository_only", "GitHub path contains no data-like files")
    return tuple(sources)


def _resolve_github(ctx: ResolverContext, url: str) -> tuple[ResolvedSource, ...]:
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path
    blob = _GITHUB_BLOB_RE.match(path)
    if blob:
        owner, repo, ref, rest = blob.groups()
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{rest}"
        response = _head_or_range(ctx, raw_url)
        reason = _reason_for_status(response.status, response.body)
        if reason is not None:
            raise Reject(reason, f"raw file returned HTTP {response.status}", http_status=response.status)
        name = rest.rsplit("/", 1)[-1]
        return (
            ResolvedSource(
                uri=raw_url,
                kind=_kind_for(name, _media_type(response.headers)),
                method="github_blob_raw",
                media_type=_media_type(response.headers),
                size_bytes=_content_length(response.headers),
                version=ref,
                member=name,
                evidence_url=url,
            ),
        )
    tree = _GITHUB_TREE_RE.match(path)
    if tree:
        owner, repo, ref, subpath = tree.groups()
        try:
            return _github_contents(ctx, owner, repo, ref, subpath or "")
        except Reject as exc:
            if exc.reason_code != "repository_only":
                raise
            return _github_repository(ctx, owner, repo)
    repo_match = _GITHUB_REPO_RE.match(path)
    if not repo_match:
        raise Reject("resolver_not_implemented", "unrecognised GitHub URL")
    owner, repo = repo_match.group(1), repo_match.group(2)
    return _github_repository(ctx, owner, repo)


def _github_repository(
    ctx: ResolverContext, owner: str, repo: str
) -> tuple[ResolvedSource, ...]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    payload = _api_get_json(ctx, api_url, method="github_repo_api")
    full_name = payload.get("full_name") if isinstance(payload.get("full_name"), str) else f"{owner}/{repo}"
    default_branch = (
        payload.get("default_branch")
        if isinstance(payload.get("default_branch"), str)
        else "HEAD"
    )
    size_kib = payload.get("size") if isinstance(payload.get("size"), int) else None
    return (
        ResolvedSource(
            uri=f"https://github.com/{full_name}",
            kind="repository",
            method="github_repo_api",
            media_type=None,
            size_bytes=size_kib * 1024 if size_kib is not None else None,
            version=default_branch,
            member=None,
            evidence_url=api_url,
        ),
    )


def _zenodo_id(parsed: urllib.parse.SplitResult) -> str | None:
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host in {"zenodo.org", "www.zenodo.org", "sandbox.zenodo.org"}:
        match = _ZENODO_PATH_RE.match(path)
        if match:
            return match.group(1)
        # DOI style path on the Zenodo host.
        doi = urllib.parse.unquote(path.lstrip("/"))
        match = _ZENODO_DOI_RE.match(doi)
        if match:
            return match.group(1)
    if host in {"doi.org", "dx.doi.org"}:
        doi = urllib.parse.unquote(path.lstrip("/"))
        match = _ZENODO_DOI_RE.match(doi)
        if match:
            return match.group(1)
    return None


def _resolve_zenodo(
    ctx: ResolverContext, record_id: str, *, evidence_url: str
) -> tuple[tuple[ResolvedSource, ...], int]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    payload = _api_get_json(ctx, api_url, method="zenodo_api")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise Reject("landing_page_only", "Zenodo record has no files list")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    version = metadata.get("version") if isinstance(metadata.get("version"), str) else str(record_id)
    names: list[tuple[str, str | None, int | None, str | None]] = []
    checksums: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str):
            continue
        size = item.get("size") if isinstance(item.get("size"), int) else None
        checksum = item.get("checksum")
        if isinstance(checksum, str) and checksum.strip():
            # Zenodo declares "md5:<hex>".
            checksums[key] = checksum.strip()
        links = item.get("links") if isinstance(item.get("links"), dict) else {}
        download = links.get("self") or links.get("download") or links.get("content")
        if not isinstance(download, str):
            download = f"https://zenodo.org/records/{record_id}/files/{urllib.parse.quote(key)}"
        elif not _SCHEME_RE.match(download):
            download = urllib.parse.urljoin("https://zenodo.org/", download)
        names.append((key, None, size, download))
    sources, total = _resolved_from_names(
        names,
        make_uri=lambda name: _name_member_uri(names, name),
        method="zenodo_api",
        version=version,
        evidence_url=api_url,
        config=ctx.config,
        checksums=checksums,
    )
    return sources, total


def _name_member_uri(
    names: list[tuple[str, str | None, int | None, str | None]], name: str
) -> str:
    for item in names:
        if item[0] == name and item[3]:
            return item[3]
    return name


def _hf_repo(parsed: urllib.parse.SplitResult) -> tuple[str, str | None, str | None] | None:
    host = (parsed.hostname or "").lower()
    if host not in {"huggingface.co", "www.huggingface.co"}:
        return None
    match = _HF_DATASET_RE.match(parsed.path or "")
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _resolve_huggingface(
    ctx: ResolverContext, repo: str, revision: str | None, path: str | None
) -> tuple[tuple[ResolvedSource, ...], int]:
    api_url = f"https://huggingface.co/api/datasets/{repo}"
    payload = _api_get_json(ctx, api_url, method="huggingface_api")
    gated = payload.get("gated")
    if payload.get("private") is True or gated not in {None, False, "false"}:
        raise Reject("gated_authentication", "Hugging Face dataset is gated or private")
    sha = payload.get("sha") if isinstance(payload.get("sha"), str) else (revision or "main")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise Reject("landing_page_only", "Hugging Face dataset has no file listing")
    names: list[tuple[str, str | None, int | None, str | None]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str):
            continue
        if path and name != path:
            continue
        names.append((name, None, None, name))
    sources, total = _resolved_from_names(
        names,
        make_uri=lambda name: (
            f"https://huggingface.co/datasets/{repo}/resolve/{sha}/"
            f"{urllib.parse.quote(name, safe='/')}"
        ),
        method="huggingface_api",
        version=sha,
        evidence_url=api_url,
        config=ctx.config,
    )
    return sources, total


def _resolve_doi(
    ctx: ResolverContext, url: str, *, depth: int = 0
) -> tuple[tuple[ResolvedSource, ...], int]:
    if depth > 1:
        raise Reject("doi_unresolved", "DOI target did not resolve to a source")
    parsed = urllib.parse.urlsplit(url)
    doi = urllib.parse.unquote(parsed.path.lstrip("/"))
    if doi.lower().startswith("doi:"):
        doi = doi[4:]
    if not _DOI_RE.match(doi):
        raise Reject("doi_unresolved", "URL does not contain a DOI")
    zenodo = _ZENODO_DOI_RE.match(doi)
    if zenodo:
        return _resolve_zenodo(ctx, zenodo.group(1), evidence_url=url)
    api_url = f"https://doi.org/{doi}"
    try:
        payload = _api_get_json(
            ctx,
            api_url,
            accept="application/vnd.citationstyles.csl+json",
            method="doi_content_negotiation",
        )
    except Reject as exc:
        if exc.reason_code in {"parse_error", "http_error", "not_found", "gone"}:
            raise Reject("doi_unresolved", "DOI content negotiation failed") from exc
        raise
    target = _doi_target(payload)
    if not target:
        raise Reject("doi_unresolved", "DOI metadata did not include a target URL")
    if target.rstrip("/") == url.rstrip("/"):
        raise Reject("doi_unresolved", "DOI metadata points at itself")
    target_candidate = Candidate(target, "doi_content_negotiation")
    outcome = _route_url(target_candidate, ctx, depth=depth + 1)
    if outcome.status == "landing":
        outcome = _landing_outcome(target_candidate, ctx)
    if outcome.status != "resolved":
        raise Reject(outcome.reason_code or "doi_unresolved", "DOI target did not resolve")
    sources = [
        replace(source, method="doi_content_negotiation", evidence_url=api_url)
        for source in outcome.sources
    ]
    return tuple(sources), outcome.total_sources or len(sources)


def _doi_target(payload: dict[str, object]) -> str | None:
    for key in ("URL", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    resource = payload.get("resource")
    if isinstance(resource, dict):
        primary = resource.get("primary")
        if isinstance(primary, dict):
            value = primary.get("URL")
            if isinstance(value, str) and value.strip():
                return value.strip()
    links = payload.get("link")
    if isinstance(links, list):
        for item in links:
            if isinstance(item, dict):
                value = item.get("URL") or item.get("url")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _resolve_archive_org(
    ctx: ResolverContext, identifier: str, *, evidence_url: str
) -> tuple[tuple[ResolvedSource, ...], int]:
    api_url = f"https://archive.org/metadata/{urllib.parse.quote(identifier)}"
    payload = _api_get_json(ctx, api_url, method="archive_org_metadata_api")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise Reject("landing_page_only", "archive.org item has no files list")
    names: list[tuple[str, str | None, int | None, str | None]] = []
    checksums: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        low = name.lower()
        if low.startswith("_") or low.endswith(("_meta.xml", "_files.xml", "_reviews.xml", ".torrent")):
            continue
        size = item.get("size")
        size_value = int(size) if isinstance(size, (int, str)) and str(size).isdigit() else None
        md5 = item.get("md5")
        if isinstance(md5, str) and md5.strip():
            checksums[name] = f"md5:{md5.strip()}"
        names.append(
            (
                name,
                None,
                size_value,
                f"https://archive.org/download/{identifier}/{urllib.parse.quote(name, safe='/')}",
            )
        )
    sources, total = _resolved_from_names(
        names,
        make_uri=lambda name: _name_member_uri(names, name),
        method="archive_org_metadata_api",
        version=None,
        evidence_url=api_url,
        config=ctx.config,
        checksums=checksums,
    )
    return sources, total


def _is_dataverse(parsed: urllib.parse.SplitResult) -> bool:
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = urllib.parse.parse_qs(parsed.query)
    if "persistentid" in {key.lower() for key in query} and (
        path.endswith("dataset.xhtml") or path.endswith("file.xhtml")
    ):
        return True
    return "dataverse" in host and "persistentid" in {key.lower() for key in query}


def _resolve_dataverse(
    ctx: ResolverContext, url: str, parsed: urllib.parse.SplitResult
) -> tuple[tuple[ResolvedSource, ...], int]:
    query = {key.lower(): value for key, value in urllib.parse.parse_qs(parsed.query).items()}
    pid = (query.get("persistentid") or [None])[0]
    if not pid:
        raise Reject("resolver_not_implemented", "Dataverse URL has no persistentId")
    host = parsed.hostname or ""
    base = f"https://{host}"
    path_low = (parsed.path or "").lower()
    if path_low.endswith("file.xhtml"):
        api_url = (
            f"{base}/api/files/:persistentId/?persistentId="
            f"{urllib.parse.quote(pid, safe='')}"
        )
        payload = _api_get_json(ctx, api_url, method="dataverse_api")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        data_file = data.get("dataFile") if isinstance(data.get("dataFile"), dict) else data
        file_id = data_file.get("id") if isinstance(data_file, dict) else None
        if not isinstance(file_id, int):
            raise Reject("parse_error", "Dataverse file API returned no file id")
        name = data_file.get("filename") if isinstance(data_file.get("filename"), str) else f"file-{file_id}"
        return (
            ResolvedSource(
                uri=f"{base}/api/access/datafile/{file_id}",
                kind=_kind_for(name, None),
                method="dataverse_api",
                size_bytes=data_file.get("filesize") if isinstance(data_file.get("filesize"), int) else None,
                version=None,
                member=name,
                evidence_url=api_url,
            ),
        ), 1
    api_url = (
        f"{base}/api/datasets/:persistentId/?persistentId="
        f"{urllib.parse.quote(pid, safe='')}"
    )
    payload = _api_get_json(ctx, api_url, method="dataverse_api")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    latest = data.get("latestVersion") if isinstance(data.get("latestVersion"), dict) else {}
    version = None
    if isinstance(latest.get("versionNumber"), int) and isinstance(latest.get("versionMinorNumber"), int):
        version = f"{latest['versionNumber']}.{latest['versionMinorNumber']}"
    raw_files = latest.get("files")
    if not isinstance(raw_files, list):
        raise Reject("landing_page_only", "Dataverse dataset has no files list")
    entries: list[tuple[str, str | None, int | None, str | None]] = []
    restricted = 0
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        data_file = item.get("dataFile")
        if not isinstance(data_file, dict):
            continue
        file_id = data_file.get("id")
        name = data_file.get("filename")
        if not isinstance(file_id, int) or not isinstance(name, str):
            continue
        if data_file.get("restricted") is True:
            restricted += 1
            continue
        size = data_file.get("filesize") if isinstance(data_file.get("filesize"), int) else None
        content_type = data_file.get("contentType") if isinstance(data_file.get("contentType"), str) else None
        entries.append((name, content_type, size, f"{base}/api/access/datafile/{file_id}"))
    if restricted and not entries:
        raise Reject("gated_authentication", "Dataverse files are restricted")
    sources, total = _resolved_from_names(
        entries,
        make_uri=lambda name: _name_member_uri(entries, name),
        method="dataverse_api",
        version=version,
        evidence_url=api_url,
        config=ctx.config,
    )
    return sources, total


# ---------------------------------------------------------------------------
# Landing pages
# ---------------------------------------------------------------------------


class _LinkParser(html.parser.HTMLParser):
    """Extract ``href`` targets and a coarse text projection."""

    def __init__(self, max_links: int) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self.script_count = 0
        self.form_html: list[str] = []
        self.password_inputs = 0
        self.max_links = max_links

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        attributes = {str(key).lower(): value for key, value in attrs}
        if lowered == "a":
            href = attributes.get("href")
            if isinstance(href, str) and len(self.links) < self.max_links:
                self.links.append(href)
        elif lowered == "script":
            self.script_count += 1
        elif lowered == "input":
            if str(attributes.get("type") or "").lower() == "password":
                self.password_inputs += 1
        elif lowered == "form" and len(self.form_html) < 20:
            rendered = " ".join(
                f"{key}={value}" for key, value in attributes.items() if value is not None
            )
            self.form_html.append(f"<form {rendered}>")

    def handle_data(self, data: str) -> None:
        if len(self.text_parts) < 5_000:
            self.text_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)[:200_000]


def _acceptable_link(base_url: str, href: str) -> str | None:
    text = href.strip()
    if not text or text.startswith("#"):
        return None
    scheme_match = _SCHEME_RE.match(text)
    if scheme_match and scheme_match.group(1).lower() in _USER_AGENT_SKIP_SCHEMES:
        return None
    try:
        target = urllib.parse.urljoin(base_url, text)
    except ValueError:
        return None
    try:
        parsed = urllib.parse.urlsplit(target)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if _scheme_reason(target) is not None:
        return None
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, "")
    )


def _link_score(base_url: str, link: str) -> int:
    score = 0
    path = urllib.parse.urlsplit(link).path
    if _is_data_like(path):
        score += 2
    if _has_archive_suffix(path):
        score += 1
    lowered = link.lower()
    if any(token in lowered for token in ("/download", "/files/", "format=csv", "format=json", "?download")):
        score += 1
    if _host_for(link) == _host_for(base_url):
        score += 1
    return score


def _data_links(base_url: str, html_text: str, max_links: int) -> list[str]:
    parser = _LinkParser(max_links)
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:  # html.parser is lenient; never let a page abort a run
        pass
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, href in enumerate(parser.links):
        target = _acceptable_link(base_url, href)
        if target is None or target in seen:
            continue
        score = _link_score(base_url, target)
        if score < 2:
            continue
        seen.add(target)
        scored.append((-score, index, target))
    scored.sort()
    return [item[2] for item in scored[: max(1, max_links)]]


def _robots_allowed(ctx: ResolverContext, url: str) -> bool:
    if not ctx.config.respect_robots:
        return True
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return True
    cache_key = f"{parsed.scheme}://{parsed.netloc}"
    with ctx.robots_lock:
        cached = ctx.robots.get(cache_key)
    if cached is not None:
        return cached
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    allowed = True
    try:
        response = ctx.request(
            robots_url, method="GET", max_bytes=64 * 1024, store_body=True
        )
    except NetguardError:
        response = None
    except Exception:
        response = None
    if response is not None and response.status == 200 and response.body:
        allowed = _robots_parse(
            response.body.decode("utf-8", errors="replace"), parsed.path or "/"
        )
    with ctx.robots_lock:
        ctx.robots[cache_key] = allowed
    return allowed


def _robots_parse(text: str, path: str) -> bool:
    active = False
    matched_group = False
    disallow: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            if active and matched_group:
                break
            active = value == "*" or "flossware" in value.lower()
            matched_group = matched_group or active
        elif key == "disallow" and active and value:
            disallow.append(value)
    for prefix in disallow:
        if path.startswith(prefix):
            return False
    return True


def _looks_javascript(html_text: str, script_count: int) -> bool:
    if _JS_HINT_RE.search(html_text) is not None:
        return True
    # A page whose entire visible text is a short shell plus scripts is a
    # client-rendered application; there is nothing useful to parse.
    return script_count > 0 and len(html_text.strip()) < 200 and "<a " not in html_text.lower()


def _looks_interaction(html_text: str, forms: list[str], password_inputs: int = 0) -> bool:
    if password_inputs > 0:
        return True
    rendered = " ".join(forms)
    if _PASSWORD_FORM_RE.search(rendered):
        return True
    return _INTERACTION_RE.search(html_text) is not None


_INTERACTION_RE = _INTERACTION_HINT_RE


def _resolve_landing(
    ctx: ResolverContext, candidate: Candidate
) -> tuple[tuple[ResolvedSource, ...], int]:
    url = candidate.url
    if not _robots_allowed(ctx, url):
        raise Reject("robots_disallowed", "robots.txt disallows landing-page fetch")
    cap = max(1, ctx.config.landing_page_max_bytes)
    try:
        response = ctx.request(
            url,
            method="GET",
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"},
            max_bytes=cap,
            store_body=True,
        )
    except NetguardError as exc:
        raise _reject_from_netguard(exc) from exc
    reason = _reason_for_status(response.status, response.body)
    if reason is not None:
        raise Reject(reason, f"landing page returned HTTP {response.status}", http_status=response.status)
    body = response.body or b""
    if len(body) > cap:
        raise Reject("oversized_declared", "landing page exceeds the byte limit")
    media_type = _media_type(response.headers)
    if not _is_html(media_type, response.final_url) and _is_data_like(
        urllib.parse.urlsplit(response.final_url).path, media_type
    ):
        return (
            ResolvedSource(
                uri=response.final_url,
                kind=_kind_for(response.final_url, media_type),
                method="landing_page",
                media_type=media_type,
                size_bytes=_content_length(response.headers),
                evidence_url=url,
            ),
        ), 1
    text = _decode_text(body, response.headers, cap)
    links = _data_links(response.final_url, text, ctx.config.landing_page_max_links)
    if links:
        sources = tuple(
            ResolvedSource(
                uri=link,
                kind=_kind_for(urllib.parse.urlsplit(link).path),
                method="landing_page",
                evidence_url=url,
            )
            for link in links[: max(1, ctx.config.max_resolved_sources)]
        )
        return sources, len(links)
    parser = _LinkParser(ctx.config.landing_page_max_links)
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        pass
    if _looks_interaction(parser.text, parser.form_html, parser.password_inputs):
        raise Reject("requires_interaction", "landing page requires interaction")
    if _looks_javascript(parser.text, parser.script_count):
        raise Reject("requires_javascript", "landing page requires JavaScript")
    raise Reject("landing_page_only", "landing page exposes no data-like links")


# ---------------------------------------------------------------------------
# Candidate routing
# ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    status: str
    sources: tuple[ResolvedSource, ...] = ()
    method: str | None = None
    kind: str | None = None
    version: str | None = None
    total_sources: int | None = None
    reason_code: str | None = None
    detail: str | None = None
    http_status: int | None = None
    attempt_method: str = "unknown"
    landing: bool = False


def _route_url(candidate: Candidate, ctx: ResolverContext, *, depth: int = 0) -> _Outcome:
    """Route one candidate to the right resolver, without landing-page fallback."""
    url = candidate.url
    try:
        parsed = _precheck(url)
    except Reject as exc:
        return _Outcome(
            "failed",
            reason_code=exc.reason_code,
            detail=exc.detail,
            http_status=exc.http_status,
            attempt_method=candidate.origin,
        )
    host = (parsed.hostname or "").lower()
    try:
        if _zenodo_id(parsed):
            record_id = _zenodo_id(parsed)
            assert record_id is not None
            sources, total = _resolve_zenodo(ctx, record_id, evidence_url=url)
            return _Outcome(
                "resolved",
                sources=sources,
                method="zenodo_api",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=total,
                attempt_method="zenodo_api",
            )
        if host in {"doi.org", "dx.doi.org"}:
            sources, total = _resolve_doi(ctx, url, depth=depth)
            return _Outcome(
                "resolved",
                sources=sources,
                method="doi_content_negotiation",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=total,
                attempt_method="doi_content_negotiation",
            )
        if host in {"github.com", "www.github.com"}:
            try:
                sources = _resolve_github(ctx, url)
            except Reject as exc:
                if exc.reason_code != "resolver_not_implemented":
                    raise
                # Release assets and archive links are not repository views;
                # confirm them as ordinary direct files.
                return _head_probe(candidate, ctx)
            return _Outcome(
                "resolved",
                sources=sources,
                method=sources[0].method if sources else "github",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=len(sources),
                attempt_method=sources[0].method if sources else "github",
            )
        hf = _hf_repo(parsed)
        if hf is not None:
            repo, revision, path = hf
            sources, total = _resolve_huggingface(ctx, repo, revision, path)
            return _Outcome(
                "resolved",
                sources=sources,
                method="huggingface_api",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=total,
                attempt_method="huggingface_api",
            )
        archive_match = _ARCHIVE_DETAILS_RE.match(parsed.path or "")
        if host in {"archive.org", "www.archive.org"} and archive_match:
            sources, total = _resolve_archive_org(ctx, archive_match.group(1), evidence_url=url)
            return _Outcome(
                "resolved",
                sources=sources,
                method="archive_org_metadata_api",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=total,
                attempt_method="archive_org_metadata_api",
            )
        if _is_dataverse(parsed):
            sources, total = _resolve_dataverse(ctx, url, parsed)
            return _Outcome(
                "resolved",
                sources=sources,
                method="dataverse_api",
                kind=_top_kind(sources),
                version=sources[0].version if sources else None,
                total_sources=total,
                attempt_method="dataverse_api",
            )
    except Reject as exc:
        return _Outcome(
            "failed",
            reason_code=exc.reason_code,
            detail=exc.detail,
            http_status=exc.http_status,
            attempt_method=_route_method(candidate, parsed),
        )
    except Exception as exc:  # one bad resolver must not abort the entry
        return _Outcome(
            "failed",
            reason_code="resolver_error",
            detail=type(exc).__name__,
            attempt_method=_route_method(candidate, parsed),
        )
    try:
        return _head_probe(candidate, ctx)
    except Exception as exc:
        return _Outcome(
            "failed",
            reason_code="resolver_error",
            detail=type(exc).__name__,
            attempt_method=candidate.origin,
        )


def _route_method(candidate: Candidate, parsed: urllib.parse.SplitResult) -> str:
    host = (parsed.hostname or "").lower()
    if _zenodo_id(parsed):
        return "zenodo_api"
    if host in {"doi.org", "dx.doi.org"}:
        return "doi_content_negotiation"
    if host in {"github.com", "www.github.com"}:
        path = parsed.path or ""
        if _GITHUB_BLOB_RE.match(path):
            return "github_blob_raw"
        if _GITHUB_TREE_RE.match(path):
            return "github_contents_api"
        if _GITHUB_REPO_RE.match(path):
            return "github_repo_api"
        return "github"
    if _hf_repo(parsed) is not None:
        return "huggingface_api"
    if host in {"archive.org", "www.archive.org"}:
        return "archive_org_metadata_api"
    if _is_dataverse(parsed):
        return "dataverse_api"
    return candidate.origin


def _head_probe(candidate: Candidate, ctx: ResolverContext) -> _Outcome:
    method = "source_access_url" if candidate.origin != "homepage" else "head_probe"
    try:
        response = _head_or_range(ctx, candidate.url)
    except Reject as exc:
        return _Outcome(
            "failed",
            reason_code=exc.reason_code,
            detail=exc.detail,
            http_status=exc.http_status,
            attempt_method=method,
        )
    reason = _reason_for_status(response.status, response.body)
    if reason is not None:
        return _Outcome(
            "failed",
            reason_code=reason,
            detail=f"HTTP {response.status}",
            http_status=response.status,
            attempt_method=method,
        )
    media_type = _media_type(response.headers)
    if _is_html(media_type, response.final_url):
        return _Outcome(
            "landing",
            reason_code="landing_page_only",
            detail="candidate is an HTML landing page",
            http_status=response.status,
            attempt_method=method,
            landing=True,
        )
    return _Outcome(
        "resolved",
        sources=(
            ResolvedSource(
                uri=response.final_url,
                kind=_kind_for(response.final_url, media_type),
                method=method,
                media_type=media_type,
                size_bytes=_content_length(response.headers),
                evidence_url=candidate.url,
            ),
        ),
        method=method,
        kind=_kind_for(response.final_url, media_type),
        version=None,
        total_sources=1,
        attempt_method=method,
    )


def _top_kind(sources: tuple[ResolvedSource, ...]) -> str | None:
    if not sources:
        return None
    kinds = {source.kind for source in sources}
    if len(kinds) == 1:
        return next(iter(kinds))
    if "archive" in kinds:
        return "archive"
    return "collection"


def _better(current: _Outcome | None, candidate: _Outcome) -> _Outcome:
    if current is None:
        return candidate
    current_rank = _REASON_RANK.get(current.reason_code or "", len(REASON_PRIORITY))
    candidate_rank = _REASON_RANK.get(candidate.reason_code or "", len(REASON_PRIORITY))
    return candidate if candidate_rank < current_rank else current


def _finalize(
    entry_id: str,
    ctx: ResolverContext,
    *,
    sources: tuple[ResolvedSource, ...] = (),
    outcome: _Outcome | None = None,
    reason_code: str | None = None,
    detail: str | None = None,
    attempts: list[Attempt],
    total_sources: int | None = None,
) -> Resolution:
    if sources:
        status = "resolved"
        actual_reason = None
    elif reason_code is not None:
        actual_reason = reason_code
        status = "deferred" if actual_reason in RETRYABLE_REASONS else "unresolved"
    else:
        actual_reason = "resolver_not_implemented"
        status = "deferred"
    retryable = actual_reason in RETRYABLE_REASONS if actual_reason else False
    selected = outcome or _Outcome("failed")
    return Resolution(
        entry_id=entry_id,
        status=status,
        reason_code=actual_reason,
        checked_at=ctx.now,
        sources=sources,
        attempts=tuple(attempts),
        method=selected.method if sources else (selected.attempt_method if selected else None),
        kind=_top_kind(sources) if sources else None,
        version=sources[0].version if sources else None,
        total_sources=total_sources if total_sources is not None else (len(sources) or None),
        detail=(detail or selected.detail),
        retryable=retryable,
        next_retry_at=(
            _iso_after(ctx.now, ctx.config.resolve_retry_backoff_seconds)
            if retryable
            else None
        ),
        attempts_count=1,
    )


def resolve_entry(row: dict[str, object], ctx: ResolverContext) -> Resolution:
    """Resolve one manifest row.  Never downloads dataset bodies."""
    entry_id = str(row.get("entry_id") or "")
    metadata_sha256 = str(row.get("metadata_sha256") or "")

    if row.get("status") == "failed":
        reason = str(row.get("reason_code") or "catalog_entry_invalid")
        return Resolution(
            entry_id=entry_id,
            status="failed",
            reason_code=reason,
            checked_at=ctx.now,
            detail=str(row.get("error") or reason)[:500],
            retryable=reason in RETRYABLE_REASONS,
            attempts_count=1,
        )

    if ctx.cache is not None:
        cached = ctx.cache.get(entry_id, metadata_sha256, ctx.now)
        if cached is not None:
            ctx.record_cache(hit=True)
            return cached
    ctx.record_cache(hit=False)

    candidates = extract_candidates(row)
    attempts: list[Attempt] = []
    if not candidates:
        return _finalize(
            entry_id,
            ctx,
            reason_code="no_source_url",
            detail="entry declares no homepage or source access URL",
            attempts=attempts,
        )

    best: _Outcome | None = None
    landings: list[Candidate] = []
    for candidate in candidates:
        try:
            outcome = _route_url(candidate, ctx)
        except Exception as exc:  # defensive: never let one candidate abort the entry
            outcome = _Outcome(
                "failed",
                reason_code="resolver_error",
                detail=type(exc).__name__,
                attempt_method=candidate.origin,
            )
        attempts.append(
            Attempt(
                url=candidate.url,
                method=outcome.attempt_method,
                status="resolved" if outcome.status == "resolved" else outcome.status,
                reason_code=outcome.reason_code,
                http_status=outcome.http_status,
                detail=outcome.detail,
            )
        )
        if outcome.status == "resolved":
            return _finalize(
                entry_id,
                ctx,
                sources=outcome.sources,
                outcome=outcome,
                attempts=attempts,
                total_sources=outcome.total_sources,
            )
        if outcome.landing:
            landings.append(candidate)
        else:
            best = _better(best, outcome)

    if landings:
        landing = landings[0]
        outcome = _landing_outcome(landing, ctx)
        attempts.append(
            Attempt(
                url=landing.url,
                method="landing_page",
                status="resolved" if outcome.status == "resolved" else "rejected",
                reason_code=outcome.reason_code,
                http_status=outcome.http_status,
                detail=outcome.detail,
            )
        )
        if outcome.status == "resolved":
            return _finalize(
                entry_id,
                ctx,
                sources=outcome.sources,
                outcome=outcome,
                attempts=attempts,
                total_sources=outcome.total_sources,
            )
        best = _better(best, outcome)

    selected = best or _Outcome("failed", reason_code="landing_page_only", detail="no candidate resolved")
    return _finalize(
        entry_id,
        ctx,
        reason_code=selected.reason_code,
        detail=selected.detail,
        attempts=attempts,
        outcome=selected,
    )


def _landing_outcome(candidate: Candidate, ctx: ResolverContext) -> _Outcome:
    try:
        sources, total = _resolve_landing(ctx, candidate)
    except Reject as exc:
        return _Outcome(
            "failed",
            reason_code=exc.reason_code,
            detail=exc.detail,
            http_status=exc.http_status,
            attempt_method="landing_page",
        )
    except Exception as exc:
        return _Outcome(
            "failed",
            reason_code="resolver_error",
            detail=type(exc).__name__,
            attempt_method="landing_page",
        )
    return _Outcome(
        "resolved",
        sources=sources,
        method="landing_page",
        kind=_top_kind(sources),
        version=sources[0].version if sources else None,
        total_sources=total,
        attempt_method="landing_page",
    )


def resolve_entries(
    rows: list[dict[str, object]],
    selected_ids: set[str],
    config: Config,
    *,
    transport=None,
    cache: ResolutionCache | None = None,
    limiter: HostRateLimiter | None = None,
    now: str | None = None,
    cache_enabled: bool | None = None,
) -> tuple[list[Resolution], int, ResolverContext]:
    """Resolve selected rows concurrently; every row gets exactly one result."""
    reference = now or utc_now()
    enabled = config.resolve_cache_enabled if cache_enabled is None else cache_enabled
    context = ResolverContext(
        config=config,
        transport=transport or build_transport(config),
        now=reference,
        cache=(cache or ResolutionCache(config)) if enabled else None,
        limiter=limiter or HostRateLimiter(config.per_host_rps),
    )
    results: dict[str, Resolution] = {}
    unhandled = 0
    pending = [row for row in rows if str(row.get("entry_id")) in selected_ids]

    def perform(row: dict[str, object]) -> Resolution:
        resolution = resolve_entry(row, context)
        metadata_sha256 = str(row.get("metadata_sha256") or "")
        if enabled and context.cache is not None and resolution.status != "failed":
            try:
                context.cache.put(resolution, metadata_sha256)
            except OSError:
                pass
        return resolution

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
            futures = {executor.submit(perform, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                entry_id = str(row.get("entry_id") or "")
                try:
                    results[entry_id] = future.result()
                except Exception as exc:  # pragma: no cover - defensive isolation
                    unhandled += 1
                    results[entry_id] = Resolution(
                        entry_id=entry_id,
                        status="failed",
                        reason_code="resolver_error",
                        checked_at=reference,
                        detail=type(exc).__name__,
                        attempts_count=1,
                    )
    return [results[key] for key in sorted(results)], unhandled, context


# ---------------------------------------------------------------------------
# Selection, accounting, artifacts and outcomes
# ---------------------------------------------------------------------------


def select_entries(
    rows: list[dict[str, object]],
    *,
    sample: int | None = None,
    limit: int | None = None,
) -> tuple[set[str], bool]:
    """Choose which entries to process, deterministically.

    ``--sample N`` is a round-robin stratification by disposition so every
    disposition class is represented.  ``--limit N`` takes the first N
    processable rows by entry id.  Rows with a catalog-level failure are always
    represented (with their inherited reason) and never consume the budget.
    """
    processable = [row for row in rows if row.get("status") != "failed"]
    processable.sort(key=lambda row: str(row.get("entry_id")))
    if sample is not None:
        chosen = _stratified_sample(processable, max(0, sample))
        return {str(row.get("entry_id")) for row in chosen}, len(chosen) < len(processable)
    if limit is not None:
        chosen = processable[: max(0, limit)]
        return {str(row.get("entry_id")) for row in chosen}, len(chosen) < len(processable)
    return {str(row.get("entry_id")) for row in processable}, False


def _stratified_sample(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0:
        return []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        disposition = str(row.get("disposition") or "unknown")
        groups.setdefault(disposition, []).append(row)
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


@dataclass
class ResolutionAccounting:
    ok: bool
    discovered: int
    status_total: int
    status_counts: dict[str, int]
    missing_reason: list[str]
    problems: list[str]


def check_resolution_accounting(
    resolutions: list[Resolution], discovered: int
) -> ResolutionAccounting:
    status_counts: dict[str, int] = {}
    missing_reason: list[str] = []
    problems: list[str] = []
    for resolution in resolutions:
        status_counts[resolution.status] = status_counts.get(resolution.status, 0) + 1
        if resolution.status not in VALID_RESOLUTION_STATUSES:
            problems.append(f"invalid_status:{resolution.status}")
        if resolution.status not in SUCCESS_RESOLUTION_STATUSES and not resolution.reason_code:
            missing_reason.append(resolution.entry_id)
    status_total = sum(status_counts.values())
    ok = True
    if len(resolutions) != discovered:
        ok = False
        problems.append(f"row_count:{len(resolutions)}!=discovered:{discovered}")
    if status_total != discovered:
        ok = False
        problems.append(f"status_total:{status_total}!=discovered:{discovered}")
    if missing_reason:
        ok = False
        problems.append(f"missing_reason:{len(missing_reason)}")
    if problems:
        ok = False
    return ResolutionAccounting(
        ok=ok,
        discovered=discovered,
        status_total=status_total,
        status_counts=dict(sorted(status_counts.items())),
        missing_reason=sorted(missing_reason),
        problems=sorted(set(problems)),
    )


def budget_resolution(row: dict[str, object], now: str, config: Config) -> Resolution:
    return Resolution(
        entry_id=str(row.get("entry_id") or ""),
        status="deferred",
        reason_code="run_budget_exhausted",
        checked_at=now,
        detail="entry was outside the bounded sample for this run",
        retryable=True,
        next_retry_at=_iso_after(now, config.resolve_retry_backoff_seconds),
        attempts_count=1,
    )


def complete_resolutions(
    rows: list[dict[str, object]],
    resolutions: list[Resolution],
    *,
    now: str,
    config: Config,
) -> list[Resolution]:
    """Return one resolution per row, filling unselected rows with the budget reason."""
    by_entry = {resolution.entry_id: resolution for resolution in resolutions}
    completed: list[Resolution] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        resolution = by_entry.get(entry_id)
        if resolution is None:
            if row.get("status") == "failed":
                reason = str(row.get("reason_code") or "catalog_entry_invalid")
                resolution = Resolution(
                    entry_id=entry_id,
                    status="failed",
                    reason_code=reason,
                    checked_at=now,
                    detail=str(row.get("error") or reason)[:500],
                    retryable=reason in RETRYABLE_REASONS,
                    attempts_count=1,
                )
            else:
                resolution = budget_resolution(row, now, config)
        elif row.get("status") == "failed" and resolution.status != "failed":
            # Catalog failures always win over any network outcome.
            reason = str(row.get("reason_code") or "catalog_entry_invalid")
            resolution = Resolution(
                entry_id=entry_id,
                status="failed",
                reason_code=reason,
                checked_at=now,
                detail=str(row.get("error") or reason)[:500],
                retryable=reason in RETRYABLE_REASONS,
                attempts_count=1,
            )
        completed.append(resolution)
    return completed


def artifact_row(row: dict[str, object], resolution: Resolution) -> dict[str, object]:
    catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
    dataset_license = (row.get("license") or {}).get("dataset") if isinstance(row.get("license"), dict) else {}
    if not isinstance(dataset_license, dict):
        dataset_license = {}
    return {
        "schema_version": 1,
        "entry_id": resolution.entry_id,
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
        "license": {
            "family": dataset_license.get("family"),
            "policy": dataset_license.get("policy"),
        },
        "resolution": resolution.to_artifact(),
    }


def resolution_manifest_bytes(
    rows: list[dict[str, object]], resolutions: list[Resolution]
) -> bytes:
    by_entry = {resolution.entry_id: resolution for resolution in resolutions}
    lines: list[str] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        resolution = by_entry.get(entry_id)
        if resolution is None:
            continue
        lines.append(canonical_json(artifact_row(row, resolution)))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_resolution_manifest(
    rows: list[dict[str, object]], resolutions: list[Resolution], path: Path
) -> tuple[bool, Path]:
    data = resolution_manifest_bytes(rows, resolutions)
    from .manifest import atomic_write

    existing = None
    try:
        existing = path.read_bytes()
    except OSError:
        existing = None
    changed = existing != data
    if changed:
        atomic_write(path, data)
    return changed, path


def resolution_outcome_row(
    row: dict[str, object],
    resolution: Resolution,
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> dict[str, object]:
    outcome_code = f"{resolution.status}:{resolution.reason_code or ''}"
    event_id = sha256_hex(
        canonical_json(
            {
                "entry_id": resolution.entry_id,
                "stage": "resolve",
                "input_hash": row.get("metadata_sha256"),
                "outcome_code": outcome_code,
                "attempt": 1,
            }
        )
    )
    first = resolution.sources[0] if resolution.sources else None
    return {
        "schema_version": 1,
        "event_id": event_id,
        "run_id": run_id,
        "entry_id": resolution.entry_id,
        "catalog_commit": catalog_commit,
        "stage": "resolve",
        "status": resolution.status,
        "reason_code": resolution.reason_code,
        "detail": resolution.detail[:500] if resolution.detail else None,
        "source_uri": sanitize_url(first.uri) if first else None,
        "source_sha256": None,
        "bytes": first.size_bytes if first else None,
        "upstream_version": resolution.version,
        "source_count": len(resolution.sources),
        "total_sources": resolution.total_sources,
        "method": resolution.method,
        "kind": resolution.kind,
        "etag": None,
        "last_modified": None,
        "retrieved_at": retrieved_at,
        "attempts": resolution.attempts_count,
        "retryable": resolution.retryable,
        "next_retry_at": resolution.next_retry_at,
        "knowledge_source_id": None,
        "revision_id": None,
        "revision_sequence": None,
        "knowledge_decision": None,
    }


def append_resolution_outcomes(
    rows: list[dict[str, object]],
    resolutions: list[Resolution],
    path: Path,
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> int:
    from .manifest import append_outcome_rows

    by_entry = {resolution.entry_id: resolution for resolution in resolutions}
    outcomes: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        resolution = by_entry.get(str(row.get("entry_id") or ""))
        if resolution is None:
            continue
        outcomes.append(
            resolution_outcome_row(
                row,
                resolution,
                run_id=run_id,
                catalog_commit=catalog_commit,
                retrieved_at=retrieved_at,
            )
        )
    return append_outcome_rows(outcomes, path)


def selection_hash(selected_ids: set[str]) -> str:
    return sha256_hex(canonical_json(sorted(selected_ids)))[:16]


def resolution_run_id(
    config: Config,
    snapshot,
    *,
    selected_ids: set[str],
    sample: int | None,
    limit: int | None,
) -> str:
    selection_hash_value = selection_hash(selected_ids)
    return sha256_hex(
        canonical_json(
            {
                "stage": "resolve",
                "resolver_version": RESOLVER_VERSION,
                "schema_version": config.schema_version,
                "catalog_repo": snapshot.repo,
                "commit": snapshot.commit,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "sample": sample,
                "limit": limit,
                "selection_hash": selection_hash_value,
                "cache_enabled": bool(config.resolve_cache_enabled),
            }
        )
    )[:16]
