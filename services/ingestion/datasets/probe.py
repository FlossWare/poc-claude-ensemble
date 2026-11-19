"""Optional bounded accessibility probe.

``--probe`` sends HEAD and falls back to ``GET Range: bytes=0-0`` only when the
server rejects HEAD (405/501).  Responses are never retained.  Global
concurrency is bounded (default 4) and each host is limited to one in-flight
request at one request per second.  Results are cached without bodies so a
re-run on the same snapshot performs no network I/O.
"""
from __future__ import annotations

import json
import os
import stat
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .model import canonical_json, sanitize_url, sha256_hex
from .netguard import NetguardError, Transport


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


def _int_header(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    return None


def _content_range_total(value: str | None) -> int | None:
    if not value:
        return None
    # bytes 0-0/12345
    if "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return int(total) if total.isdigit() else None


def _status_access(status: int) -> tuple[str, str | None]:
    if 200 <= status < 300:
        return "public", None
    if status == 401:
        return "gated", "gated_authentication"
    if status == 403:
        return "blocked", "blocked"
    if status == 429:
        return "blocked", "rate_limited"
    if status == 404:
        return "inaccessible", "not_found"
    if status == 410:
        return "inaccessible", "gone"
    if status >= 500:
        return "inaccessible", "server_error"
    return "inaccessible", "http_error"


def _error_access(code: str | None) -> tuple[str, str]:
    mapping = {
        "ssrf_blocked": ("inaccessible", "ssrf_blocked"),
        "dns_failure": ("inaccessible", "dns_failure"),
        "tls_failure": ("inaccessible", "tls_failure"),
        "timeout": ("inaccessible", "timeout"),
        "too_many_redirects": ("inaccessible", "too_many_redirects"),
        "unsafe_redirect": ("inaccessible", "unsafe_redirect"),
        "invalid_url": ("inaccessible", "invalid_url"),
        "unsupported_scheme": ("inaccessible", "unsupported_scheme"),
        "unsafe_port": ("inaccessible", "unsafe_port"),
        "oversized_declared": ("inaccessible", "oversized_declared"),
        "oversized_stream": ("inaccessible", "oversized_stream"),
        "gated_authentication": ("gated", "gated_authentication"),
        "blocked": ("blocked", "blocked"),
        "not_found": ("inaccessible", "not_found"),
        "gone": ("inaccessible", "gone"),
        "rate_limited": ("blocked", "rate_limited"),
        "server_error": ("inaccessible", "server_error"),
    }
    return mapping.get(str(code), ("inaccessible", str(code) if code else "network_error"))


@dataclass(frozen=True)
class ProbeResult:
    url: str | None
    method: str | None
    status: int | None
    final_url: str | None
    content_type: str | None
    content_length: int | None
    redirect_chain: tuple[str, ...]
    checked_at: str
    access: str
    reason_code: str | None
    error: str | None = None
    cached: bool = False

    def to_manifest(self) -> dict[str, object]:
        return {
            "url": sanitize_url(self.url),
            "method": self.method,
            "status": self.status,
            "final_url": sanitize_url(self.final_url),
            "content_type": self.content_type,
            "content_length": self.content_length,
            "redirect_chain": [
                sanitize_url(item) or "" for item in self.redirect_chain
            ],
            "checked_at": self.checked_at,
            "reason_code": self.reason_code,
            "error": self.error,
        }

    def to_cache(self) -> dict[str, object]:
        return {**self.to_manifest(), "access": self.access}

    @classmethod
    def from_cache(cls, payload: dict[str, object]) -> "ProbeResult":
        return cls(
            url=_optional_str(payload.get("url")),
            method=_optional_str(payload.get("method")),
            status=_optional_int(payload.get("status")),
            final_url=_optional_str(payload.get("final_url")),
            content_type=_optional_str(payload.get("content_type")),
            content_length=_optional_int(payload.get("content_length")),
            redirect_chain=tuple(
                str(item) for item in payload.get("redirect_chain") or [] if isinstance(item, str)
            ),
            checked_at=str(payload.get("checked_at") or ""),
            access=str(payload.get("access") or "unknown"),
            reason_code=_optional_str(payload.get("reason_code")),
            error=_optional_str(payload.get("error")),
            cached=True,
        )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def choose_probe_url(row: dict[str, object]) -> str | None:
    homepage = row.get("homepage")
    if isinstance(homepage, str) and homepage.strip():
        return homepage.strip()
    for source in row.get("sources") or []:
        if isinstance(source, dict):
            access_url = source.get("access_url")
            if isinstance(access_url, str) and access_url.strip():
                return access_url.strip()
    return None


def probe_one(
    url: str,
    transport: Transport,
    *,
    now: str | None = None,
) -> ProbeResult:
    checked_at = now or utc_now()
    try:
        response = transport.request(url, method="HEAD")
        method = "HEAD"
        if response.status in {405, 501}:
            response = transport.request(
                url, method="GET", headers={"Range": "bytes=0-0"}
            )
            method = "GET"
    except NetguardError as exc:
        access, reason = _error_access(exc.code)
        return ProbeResult(
            url=url,
            method=None,
            status=exc.status,
            final_url=exc.final_url or url,
            content_type=None,
            content_length=None,
            redirect_chain=tuple(exc.redirect_chain),
            checked_at=checked_at,
            access=access,
            reason_code=reason,
            error=exc.code,
        )
    except Exception as exc:  # never let one probe abort the inventory
        return ProbeResult(
            url=url,
            method=None,
            status=None,
            final_url=url,
            content_type=None,
            content_length=None,
            redirect_chain=(),
            checked_at=checked_at,
            access="inaccessible",
            reason_code="probe_error",
            error=type(exc).__name__,
        )

    access, reason = _status_access(response.status)
    content_type = response.headers.get("content-type")
    if content_type:
        content_type = content_type.split(";", 1)[0].strip().lower() or None
    content_length = _int_header(response.headers.get("content-length"))
    if response.status == 206:
        total = _content_range_total(response.headers.get("content-range"))
        if total is not None:
            content_length = total
    return ProbeResult(
        url=url,
        method=method,
        status=response.status,
        final_url=response.final_url,
        content_type=content_type,
        content_length=content_length,
        redirect_chain=tuple(response.redirect_chain),
        checked_at=checked_at,
        access=access,
        reason_code=reason,
        error=None,
    )


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

    @contextmanager
    def slot(self, host: str):
        lock = self._lock_for(host)
        with lock:
            with self._mutex:
                now = self.clock()
                next_at = self._next.get(host, now)
                wait = max(0.0, next_at - now)
                self._next[host] = max(now, next_at) + (1.0 / self.rps)
            if wait > 0:
                self.sleep(wait)
            yield


class ProbeCache:
    """Per-URL probe evidence cache; no response bodies are stored."""

    def __init__(self, config: Config, *, root: Path | None = None, now: str | None = None) -> None:
        self.config = config
        self.root = root or (config.data_root / "cache" / "probes")
        self.now = now

    def path_for(self, url: str) -> Path:
        return self.root / f"{sha256_hex(url)}.json"

    def get(self, url: str, now: str | None = None) -> ProbeResult | None:
        path = self.path_for(url)
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            result = ProbeResult.from_cache(payload)
        except (TypeError, ValueError):
            return None
        checked = _parse_iso(result.checked_at)
        reference = _parse_iso(now or self.now) or datetime.now(timezone.utc)
        if checked is None:
            return None
        age = (reference - checked).total_seconds()
        if age < 0 or age > timedelta(days=self.config.recheck_after_days).total_seconds():
            return None
        return replace(result, cached=True)

    def put(self, result: ProbeResult) -> None:
        if not result.url:
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path_for(result.url)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        data = canonical_json(result.to_cache()).encode("utf-8")
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


def build_transport(config: Config) -> Transport:
    return Transport(config)


def _host_for(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "_").lower()
    except ValueError:
        return "_"


def probe_entries(
    rows: list[dict[str, object]],
    config: Config,
    *,
    transport: Transport | None = None,
    cache: ProbeCache | None = None,
    limiter: HostRateLimiter | None = None,
    now: str | None = None,
) -> dict[str, ProbeResult]:
    if not config.probe_enabled:
        return {}
    transport = transport or build_transport(config)
    cache = cache if cache is not None else ProbeCache(config, now=now)
    limiter = limiter or HostRateLimiter(config.per_host_rps)
    reference = now or utc_now()

    results: dict[str, ProbeResult] = {}
    pending: list[tuple[str, str]] = []
    for row in rows:
        entry_id = str(row.get("entry_id"))
        url = choose_probe_url(row)
        if not url:
            results[entry_id] = ProbeResult(
                url=None,
                method=None,
                status=None,
                final_url=None,
                content_type=None,
                content_length=None,
                redirect_chain=(),
                checked_at=reference,
                access="unknown",
                reason_code="missing_url",
                error=None,
            )
            continue
        cached = cache.get(url, reference)
        if cached is not None:
            results[entry_id] = cached
            continue
        pending.append((entry_id, url))

    def perform(item: tuple[str, str]) -> ProbeResult:
        _entry_id, url = item
        with limiter.slot(_host_for(url)):
            return probe_one(url, transport, now=reference)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
            futures = {executor.submit(perform, item): item for item in pending}
            for future in as_completed(futures):
                entry_id, url = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ProbeResult(
                        url=url,
                        method=None,
                        status=None,
                        final_url=url,
                        content_type=None,
                        content_length=None,
                        redirect_chain=(),
                        checked_at=reference,
                        access="inaccessible",
                        reason_code="probe_error",
                        error=type(exc).__name__,
                    )
                results[entry_id] = result
                if result.status is not None:
                    cache.put(result)
    return results
