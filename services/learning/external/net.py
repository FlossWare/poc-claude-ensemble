"""Network and cache plumbing shared by the external-source modules.

External calls use the same SSRF guard as dataset ingestion
(:mod:`ingestion.datasets.netguard`): http/https only, no credentials in the
URL, standard ports, every resolved address must be globally routable, the
connection is pinned to the validated IP, and every redirect hop is validated
again.  No credentials are read or sent by this module.

One host-specific detail: :mod:`netguard` chooses the first address returned by
the resolver.  On a host with global IPv6 addresses but no IPv6 route that
fails even though IPv4 works.  :func:`_ipv4_first` therefore presents IPv4
addresses first; netguard still validates whichever family it receives.  If a
host has only IPv6 the IPv6 answers are passed through unchanged.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from learning.external import paths

#: Hard cap for every external document this package reads.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

USER_AGENT = "FlossWare-POC-External-Priors/1"


class ExternalFetchError(RuntimeError):
    """A remote document could not be fetched or was unusable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ipv4_first(host: str, *args: Any, **kwargs: Any):
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    ipv4 = [info for info in infos if info[0] == socket.AF_INET]
    return ipv4 or infos


def _transport_factory(transport: Any = None):
    if transport is not None:
        return transport
    try:
        from ingestion.datasets.config import Config
        from ingestion.datasets.netguard import Transport
    except ImportError as exc:  # pragma: no cover - services/ is on sys.path in practice
        raise ExternalFetchError(
            "the ingestion netguard is unavailable; run from the repository with "
            "services/ on PYTHONPATH"
        ) from exc
    config = Config.load(ledger_enabled=False)
    return Transport(config, resolver=_ipv4_first)


def fetch_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: Any = None,
    headers: Optional[dict[str, str]] = None,
) -> tuple[bytes, dict[str, Any]]:
    """Fetch *url* through netguard and return ``(body, response_info)``.

    ``transport`` is injectable for tests; it only needs netguard's ``request``
    contract.  The response info deliberately omits signed query strings: it
    records the final host and status, never the full URL.
    """
    client = _transport_factory(transport)
    merged = {"Accept": "application/json, text/plain, */*", "User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    try:
        response = client.request(
            url,
            method="GET",
            headers=merged,
            max_bytes=max_bytes,
            store_body=True,
        )
    except Exception as exc:  # netguard raises NetguardError, tests may raise anything
        raise ExternalFetchError(f"{type(exc).__name__}: {exc}") from exc
    status = int(getattr(response, "status", 0) or 0)
    if status < 200 or status >= 300:
        raise ExternalFetchError(f"HTTP {status} from {_host_of(url)}")
    body = getattr(response, "body", None) or b""
    if len(body) > max_bytes:
        raise ExternalFetchError(f"response from {_host_of(url)} exceeded {max_bytes} bytes")
    return body, {
        "status": status,
        "host": _host_of(url),
        "final_host": _host_of(getattr(response, "final_url", "") or url),
        "bytes": len(body),
    }


def fetch_json(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: Any = None,
) -> Any:
    body, _ = fetch_bytes(url, max_bytes=max_bytes, transport=transport)
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalFetchError(f"invalid JSON from {_host_of(url)}: {exc}") from exc


def _host_of(url: str) -> str:
    try:
        from urllib.parse import urlsplit

        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Cache documents
# ---------------------------------------------------------------------------


def read_cache(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON cache document; missing/corrupt files are absent, not fatal."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def write_cache(path: Path, document: dict[str, Any]) -> Path:
    """Atomically write a 0600 JSON cache document."""
    target = Path(path).expanduser()
    paths.ensure_dir(target.parent)
    encoded = (json.dumps(document, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def cache_age_seconds(document: Optional[dict[str, Any]], *, now: Optional[datetime] = None) -> Optional[float]:
    """Age of a cache document in seconds, or ``None`` when unreadable."""
    if not isinstance(document, dict):
        return None
    stamp = document.get("fetched_at")
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed).total_seconds())


def is_fresh(document: Optional[dict[str, Any]], max_age_seconds: float) -> bool:
    age = cache_age_seconds(document)
    return age is not None and age <= max_age_seconds


__all__ = [
    "DEFAULT_MAX_BYTES",
    "ExternalFetchError",
    "cache_age_seconds",
    "fetch_bytes",
    "fetch_json",
    "is_fresh",
    "read_cache",
    "utc_now",
    "write_cache",
]
