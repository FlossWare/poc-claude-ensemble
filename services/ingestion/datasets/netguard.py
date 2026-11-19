"""SSRF-guarded HTTP transport.

This mirrors the guard already used by ``ingestion_service.py``:
http/https only, no userinfo, standard ports only, every resolved address must
be globally routable, the connection is pinned to the validated IP (so DNS is
never re-resolved), and every redirect hop is validated again.  HTTPS to HTTP
downgrades are refused.  Bodies are never retained by default.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field

from .config import Config

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
USER_AGENT = "FlossWare-POC-Dataset-Ingest/1"
STREAM_CHUNK_BYTES = 64 * 1024
#: Statuses whose body carries resource bytes.
BODY_STATUSES = frozenset({200, 206})


class NetguardError(Exception):
    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        status: int | None = None,
        retryable: bool = False,
        final_url: str | None = None,
        redirect_chain: list[str] | None = None,
    ) -> None:
        super().__init__(message or code)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.final_url = final_url
        self.redirect_chain = list(redirect_chain or [])


@dataclass(frozen=True)
class ValidatedURL:
    url: str
    scheme: str
    host: str
    port: int
    path_query: str
    address: str


@dataclass
class RawResponse:
    status: int
    headers: dict[str, str]
    body: bytes | None = None
    truncated: bool = False


@dataclass
class NetResponse:
    status: int
    headers: dict[str, str]
    final_url: str
    redirect_chain: list[str] = field(default_factory=list)
    body: bytes | None = None
    truncated: bool = False


class Transport:
    """A small, deliberately narrow HTTP client with an SSRF guard."""

    def __init__(
        self,
        config: Config,
        *,
        resolver=None,
        connector=None,
        ssl_context: ssl.SSLContext | None = None,
        clock=None,
    ) -> None:
        self.config = config
        self.resolver = resolver or socket.getaddrinfo
        self.connector = connector or socket.create_connection
        self.ssl_context = ssl_context or ssl.create_default_context()
        self.clock = clock or time.monotonic

    # -- validation -----------------------------------------------------
    def _resolve(self, host: str) -> str:
        try:
            results = self.resolver(host, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise NetguardError("dns_failure", "host could not be resolved", retryable=True) from exc
        addresses = sorted({item[4][0] for item in results})
        if not addresses:
            raise NetguardError("dns_failure", "host resolved to no addresses", retryable=True)
        parsed: list[object] = []
        for value in addresses:
            try:
                parsed.append(ipaddress.ip_address(value.split("%", 1)[0]))
            except ValueError as exc:
                raise NetguardError("dns_failure", "host resolved to an invalid address") from exc
        if any(not address.is_global for address in parsed):
            raise NetguardError("ssrf_blocked", "URI resolves to a non-public network address")
        return addresses[0]

    def validate_url(self, url: str) -> ValidatedURL:
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError as exc:
            raise NetguardError("invalid_url", "URI could not be parsed") from exc
        scheme = parsed.scheme.lower()
        allowed = set(self.config.schemes)
        if self.config.enable_ftp:
            allowed.add("ftp")
        if scheme not in allowed:
            raise NetguardError("unsupported_scheme", f"unsupported URI scheme: {scheme or '(missing)'}")
        if scheme not in {"http", "https"}:
            # FTP is opt-in for the existing service, but the dataset connector
            # does not implement an FTP transport in PR 1.
            raise NetguardError("unsupported_scheme", "FTP transport is not implemented")
        if not parsed.hostname:
            raise NetguardError("invalid_url", "URI must have a host")
        if parsed.username is not None or parsed.password is not None:
            raise NetguardError("invalid_url", "URI must not contain credentials")
        try:
            port = parsed.port
        except ValueError as exc:
            raise NetguardError("invalid_url", "URI has an invalid port") from exc
        default_port = 443 if scheme == "https" else 80
        if port is not None and port != default_port:
            raise NetguardError("unsafe_port", "only standard HTTP and HTTPS ports are enabled")
        port = default_port
        raw_host = parsed.hostname
        try:
            # IP literals (notably IPv6) must reach the address guard rather
            # than fail IDNA encoding first.
            ipaddress.ip_address(raw_host)
            host = raw_host
        except ValueError:
            try:
                host = raw_host.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise NetguardError("invalid_url", "URI host is not valid IDNA") from exc
        if any(ord(char) < 32 or ord(char) == 127 for char in host):
            raise NetguardError("invalid_url", "URI host contains control characters")
        path_query = urllib.parse.urlunsplit(
            ("", "", parsed.path or "/", parsed.query, "")
        )
        address = self._resolve(host)
        return ValidatedURL(
            url=urllib.parse.urlunsplit((scheme, parsed.netloc, parsed.path or "/", parsed.query, "")),
            scheme=scheme,
            host=host,
            port=port,
            path_query=path_query,
            address=address,
        )

    # -- transport ------------------------------------------------------
    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        max_bytes: int = 0,
        store_body: bool = False,
    ) -> NetResponse:
        return self._follow(
            url,
            lambda validated, deadline: self._request_once(
                validated,
                method=method,
                headers=headers,
                max_bytes=max_bytes,
                store_body=store_body,
                deadline=deadline,
            ),
        )

    @contextmanager
    def open_stream(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ):
        """Yield a live response whose body the caller reads in chunks.

        Nothing is buffered, so a multi-hundred-megabyte dataset cannot exhaust
        memory, and the caller can inspect the status and headers (needed for
        ``Range`` resume) *before* deciding what to do with the bytes.  Every
        redirect hop goes through the same SSRF and downgrade guard.
        """
        deadline = self.clock() + self.config.request_max_seconds
        current = url
        redirect_chain: list[str] = []
        previous_scheme: str | None = None
        for hop in range(self.config.max_redirects + 1):
            validated = self.validate_url(current)
            if previous_scheme == "https" and validated.scheme == "http":
                raise NetguardError(
                    "unsafe_redirect",
                    "HTTPS to HTTP redirect is not allowed",
                    final_url=current,
                    redirect_chain=redirect_chain,
                )
            connection, sock, response, response_headers = self._open(
                validated, method=method, headers=headers, deadline=deadline
            )
            if response.status in REDIRECT_STATUSES:
                location = response_headers.get("location")
                if not location:
                    raise NetguardError(
                        "invalid_url",
                        "redirect response has no Location header",
                        status=response.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                if hop >= self.config.max_redirects:
                    raise NetguardError(
                        "too_many_redirects",
                        "redirect limit exceeded",
                        status=response.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                target = urllib.parse.urljoin(current, location)
                target_scheme = urllib.parse.urlsplit(target).scheme.lower()
                if validated.scheme == "https" and target_scheme == "http":
                    raise NetguardError(
                        "unsafe_redirect",
                        "HTTPS to HTTP redirect is not allowed",
                        status=response.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                previous_scheme = validated.scheme
                current = target
                redirect_chain.append(target)
                continue
            stream = StreamResponse(
                status=response.status,
                headers=response_headers,
                final_url=current,
                redirect_chain=redirect_chain,
                response=response,
                deadline=deadline,
                clock=self.clock,
            )
            try:
                yield stream
            finally:
                connection.close()
                try:
                    sock.close()
                except OSError:
                    pass
            return
        raise NetguardError(
            "too_many_redirects",
            "redirect limit exceeded",
            final_url=current,
            redirect_chain=redirect_chain,
        )

    def _follow(self, url: str, perform) -> NetResponse:
        deadline = self.clock() + self.config.request_max_seconds
        current = url
        redirect_chain: list[str] = []
        previous_scheme: str | None = None
        for hop in range(self.config.max_redirects + 1):
            validated = self.validate_url(current)
            if previous_scheme == "https" and validated.scheme == "http":
                raise NetguardError(
                    "unsafe_redirect",
                    "HTTPS to HTTP redirect is not allowed",
                    final_url=current,
                    redirect_chain=redirect_chain,
                )
            raw = perform(validated, deadline)
            if raw.status in REDIRECT_STATUSES:
                location = raw.headers.get("location")
                if not location:
                    raise NetguardError(
                        "invalid_url",
                        "redirect response has no Location header",
                        status=raw.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                if hop >= self.config.max_redirects:
                    raise NetguardError(
                        "too_many_redirects",
                        "redirect limit exceeded",
                        status=raw.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                target = urllib.parse.urljoin(current, location)
                target_scheme = urllib.parse.urlsplit(target).scheme.lower()
                if validated.scheme == "https" and target_scheme == "http":
                    raise NetguardError(
                        "unsafe_redirect",
                        "HTTPS to HTTP redirect is not allowed",
                        status=raw.status,
                        final_url=current,
                        redirect_chain=redirect_chain,
                    )
                previous_scheme = validated.scheme
                current = target
                redirect_chain.append(target)
                continue
            return NetResponse(
                status=raw.status,
                headers=raw.headers,
                final_url=current,
                redirect_chain=redirect_chain,
                body=raw.body,
                truncated=raw.truncated,
            )
        raise NetguardError(
            "too_many_redirects",
            "redirect limit exceeded",
            final_url=current,
            redirect_chain=redirect_chain,
        )

    def _open(
        self,
        validated: ValidatedURL,
        *,
        method: str,
        headers: dict[str, str] | None,
        deadline: float,
    ) -> tuple[http.client.HTTPConnection, object, object, dict[str, str]]:
        """Connect to the validated IP and issue the request.

        Returns ``(connection, sock, response, response_headers)``.  The caller
        owns both the connection and the socket and must close them.
        """
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise NetguardError("timeout", "request deadline exceeded", retryable=True)
        connect_timeout = min(self.config.connect_timeout, remaining)
        try:
            sock = self.connector((validated.address, validated.port), timeout=connect_timeout)
        except (TimeoutError, socket.timeout) as exc:
            raise NetguardError("timeout", "connection timed out", retryable=True) from exc
        except OSError as exc:
            raise NetguardError("connection_failed", "connection failed", retryable=True) from exc
        try:
            if validated.scheme == "https":
                try:
                    sock = self.ssl_context.wrap_socket(sock, server_hostname=validated.host)
                except ssl.SSLError as exc:
                    raise NetguardError("tls_failure", "TLS handshake failed", retryable=True) from exc
                except OSError as exc:
                    raise NetguardError("tls_failure", "TLS handshake failed", retryable=True) from exc
            sock.settimeout(min(self.config.read_timeout, max(0.1, deadline - self.clock())))
            connection = http.client.HTTPConnection(
                validated.host,
                validated.port,
                timeout=min(self.config.read_timeout, max(0.1, deadline - self.clock())),
            )
            connection.sock = sock
            host_header = (
                f"[{validated.host}]"
                if ":" in validated.host and not validated.host.startswith("[")
                else validated.host
            )
            request_headers = {
                "Host": host_header,
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Accept-Encoding": "identity",
                "Connection": "close",
            }
            if headers:
                request_headers.update(headers)
            if (
                self.config.github_token
                and validated.host == "api.github.com"
                and validated.scheme == "https"
            ):
                request_headers["Authorization"] = f"Bearer {self.config.github_token}"
            try:
                connection.request(method, validated.path_query, headers=request_headers)
                response = connection.getresponse()
            except (TimeoutError, socket.timeout) as exc:
                raise NetguardError("timeout", "response timed out", retryable=True) from exc
            except ssl.SSLError as exc:
                raise NetguardError("tls_failure", "TLS failure", retryable=True) from exc
            except (http.client.HTTPException, OSError) as exc:
                raise NetguardError("connection_failed", "HTTP request failed", retryable=True) from exc
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
            return connection, sock, response, response_headers
        except BaseException:
            try:
                sock.close()
            except OSError:
                pass
            raise

    def _request_once(
        self,
        validated: ValidatedURL,
        *,
        method: str,
        headers: dict[str, str] | None,
        max_bytes: int,
        store_body: bool,
        deadline: float,
    ) -> RawResponse:
        connection, sock, response, response_headers = self._open(
            validated, method=method, headers=headers, deadline=deadline
        )
        try:
            if response.status in REDIRECT_STATUSES:
                response.read(64 * 1024)
                return RawResponse(status=response.status, headers=response_headers, body=None)
            if not store_body:
                # Discard a small amount; probes must never retain a body.
                response.read(min(64 * 1024, max(1, max_bytes)))
                return RawResponse(status=response.status, headers=response_headers, body=None)
            declared = response_headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise NetguardError("oversized_declared", "resource exceeds the byte limit", status=response.status)
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise NetguardError("oversized_stream", "resource exceeds the byte limit", status=response.status)
            return RawResponse(
                status=response.status,
                headers=response_headers,
                body=payload,
                truncated=False,
            )
        except NetguardError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise NetguardError("timeout", "response body timed out", retryable=True) from exc
        except ssl.SSLError as exc:
            raise NetguardError("tls_failure", "TLS failure while reading the response", retryable=True) from exc
        except OSError as exc:
            raise NetguardError("connection_failed", "connection failed while reading the response", retryable=True) from exc
        finally:
            connection.close()
            try:
                sock.close()
            except OSError:
                pass

class StreamResponse:
    """A live, unbuffered response body handed out by :meth:`Transport.open_stream`."""

    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str],
        final_url: str,
        redirect_chain: list[str],
        response,
        deadline: float,
        clock,
    ) -> None:
        self.status = status
        self.headers = headers
        self.final_url = final_url
        self.redirect_chain = list(redirect_chain)
        self._response = response
        self._deadline = deadline
        self._clock = clock

    @property
    def has_body(self) -> bool:
        return self.status in BODY_STATUSES

    def read(self, size: int = STREAM_CHUNK_BYTES) -> bytes:
        if self._clock() > self._deadline:
            raise NetguardError("timeout", "response body timed out", retryable=True)
        try:
            return self._response.read(size)
        except (TimeoutError, socket.timeout) as exc:
            raise NetguardError("timeout", "response body timed out", retryable=True) from exc
        except ssl.SSLError as exc:
            raise NetguardError("tls_failure", "TLS failure while reading the response", retryable=True) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise NetguardError("connection_failed", "connection failed while reading the response", retryable=True) from exc

    # -- convenience ----------------------------------------------------
    def head(self, url: str) -> NetResponse:
        return self.request(url, method="HEAD")

    def get_range_probe(self, url: str) -> NetResponse:
        return self.request(url, method="GET", headers={"Range": "bytes=0-0"})

    def fetch_bytes(self, url: str, *, max_bytes: int) -> tuple[bytes, NetResponse]:
        response = self.request(url, method="GET", max_bytes=max_bytes, store_body=True)
        if response.status < 200 or response.status >= 300:
            raise status_error(response.status, response.final_url, response.redirect_chain)
        return response.body or b"", response


def status_error(status: int, final_url: str, redirect_chain: list[str]) -> NetguardError:
    if status == 401:
        return NetguardError("gated_authentication", "authentication required", status=status,
                             final_url=final_url, redirect_chain=redirect_chain)
    if status == 403:
        return NetguardError("blocked", "server refused the request", status=status, retryable=True,
                             final_url=final_url, redirect_chain=redirect_chain)
    if status == 404:
        return NetguardError("not_found", "resource not found", status=status, retryable=True,
                             final_url=final_url, redirect_chain=redirect_chain)
    if status == 410:
        return NetguardError("gone", "resource is gone", status=status,
                             final_url=final_url, redirect_chain=redirect_chain)
    if status == 429:
        return NetguardError("rate_limited", "rate limited", status=status, retryable=True,
                             final_url=final_url, redirect_chain=redirect_chain)
    if status >= 500:
        return NetguardError("server_error", "server error", status=status, retryable=True,
                             final_url=final_url, redirect_chain=redirect_chain)
    return NetguardError("http_error", f"HTTP {status}", status=status,
                         final_url=final_url, redirect_chain=redirect_chain)
