#!/usr/bin/env python3
"""Awesome Public Datasets discovery and bounded acquisition for the POC.

The catalog is a discovery index, not a license grant. This module inventories
all apd-core YAML records, resolves obvious downloadable assets, and records
every acquisition outcome in a resumable JSONL manifest. It never executes
downloaded content.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mimetypes
import os
import re
import socket
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

CATALOG_REPOSITORY = "https://github.com/awesomedata/apd-core"
DEFAULT_CATALOG_ROOT = Path("var/awesome-public-datasets/apd-core")
DEFAULT_DATA_ROOT = Path("var/awesome-public-datasets/data")
DEFAULT_MANIFEST = Path("var/awesome-public-datasets/manifest.jsonl")
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DATASET_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 20 * 1024 * 1024 * 1024
ASSET_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml", ".txt", ".md",
    ".pdf", ".parquet", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".7z", ".arrow", ".feather", ".avro", ".orc",
}
TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml", ".txt", ".md", ".pdf"}


@dataclass
class DatasetRecord:
    dataset_id: str
    title: str
    category: str
    description: str
    keywords: list[str]
    homepage: str | None
    source_urls: list[str]
    specification: str | None
    data_dictionary: str | None
    license: str | None
    access_level: str | None
    metadata_path: str
    metadata_sha256: str


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_strings(item))
        return result
    if isinstance(value, dict):
        for key in ("access_url", "url", "web", "name", "reference"):
            if key in value:
                return _strings(value[key])
    return []


def _valid_http_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return value.strip()


def parse_metadata(path: Path, root: Path) -> DatasetRecord:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: install requirements.txt") from exc
    raw = path.read_bytes()
    item = yaml.safe_load(raw) or {}
    if not isinstance(item, dict):
        raise ValueError("metadata root must be a YAML mapping")
    rel = path.relative_to(root).as_posix()
    category = path.parent.name
    title = str(item.get("title") or path.stem).strip()[:500]
    source_urls: list[str] = []
    for candidate in _strings(item.get("sources")):
        url = _valid_http_url(candidate)
        if url:
            source_urls.append(url)
    homepage = _valid_http_url(item.get("homepage"))
    # Preserve explicit source URLs first; a homepage is only a discovery fallback.
    if not source_urls and homepage:
        source_urls.append(homepage)
    keywords = _strings(item.get("keywords"))
    return DatasetRecord(
        dataset_id=hashlib.sha256(rel.encode()).hexdigest(),
        title=title,
        category=str(item.get("category") or category).strip(),
        description=str(item.get("description") or "").strip()[:20000],
        keywords=keywords,
        homepage=homepage,
        source_urls=list(dict.fromkeys(source_urls)),
        specification=_valid_http_url(item.get("specification")),
        data_dictionary=_valid_http_url(item.get("data_dictionary")),
        license=str(item.get("license") or item.get("copyrights") or "").strip() or None,
        access_level=str(item.get("access_level") or "").strip() or None,
        metadata_path=rel,
        metadata_sha256=hashlib.sha256(raw).hexdigest(),
    )


def discover(catalog_root: Path) -> tuple[list[DatasetRecord], list[dict[str, str]]]:
    root = catalog_root.resolve()
    records: list[DatasetRecord] = []
    errors: list[dict[str, str]] = []
    for path in sorted((root / "core").rglob("*.y*ml")):
        if path.suffix.lower() not in {".yml", ".yaml"}:
            continue
        try:
            records.append(parse_metadata(path, root))
        except Exception as exc:
            errors.append({"metadata_path": path.relative_to(root).as_posix(),
                           "status": "metadata_error", "reason": type(exc).__name__ + ": " + str(exc)})
    return records, errors


def _public_ip(host: str) -> str:
    addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
    if not addresses:
        raise ValueError("hostname resolved to no addresses")
    parsed = [ipaddress.ip_address(addr.split("%", 1)[0]) for addr in addresses]
    if any(not address.is_global for address in parsed):
        raise ValueError("host resolves to a non-public address")
    return addresses[0]


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError("redirect to unsupported URI")
        if target.port not in {None, 80, 443}:
            raise ValueError("redirect to non-standard port")
        _public_ip(target.hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_open(url: str, timeout: int = 30):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only HTTP(S) URLs are supported")
    if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
        raise ValueError("credentials and non-standard ports are not allowed")
    _public_ip(parsed.hostname)
    opener = urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request = urllib.request.Request(url, headers={"User-Agent": "FlossWare-POC-Dataset-Ingest/1",
                                                    "Accept": "*/*"})
    return opener.open(request, timeout=timeout)


def is_direct_asset(url: str) -> bool:
    path = urllib.parse.urlsplit(url).path.lower()
    return PurePosixPath(path).suffix in ASSET_SUFFIXES or "/raw/" in path or "download=" in urllib.parse.urlsplit(url).query.lower()


class _AssetLinks(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            absolute = urllib.parse.urljoin(self.base_url, href)
            if _valid_http_url(absolute) and is_direct_asset(absolute):
                self.links.append(absolute)


def resolve_assets(record: DatasetRecord) -> tuple[list[str], str | None]:
    direct = [url for url in record.source_urls if is_direct_asset(url)]
    if direct:
        return list(dict.fromkeys(direct)), None
    for landing in record.source_urls[:2]:
        try:
            with _safe_open(landing) as response:
                media = response.headers.get_content_type()
                if media not in {"text/html", "application/xhtml+xml"}:
                    continue
                payload = response.read(2 * 1024 * 1024 + 1)
                if len(payload) > 2 * 1024 * 1024:
                    return [], "landing page exceeds 2 MiB discovery limit"
                parser = _AssetLinks(landing)
                parser.feed(payload.decode("utf-8", errors="replace"))
                if parser.links:
                    return list(dict.fromkeys(parser.links)), None
        except Exception as exc:
            return [], f"source discovery failed: {type(exc).__name__}: {exc}"
    return [], "no unambiguous direct asset link found; source may require an API, registration, or manual resolver"


def _safe_name(url: str) -> str:
    name = PurePosixPath(urllib.parse.urlsplit(url).path).name or "dataset"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:160]
    return name or "dataset"


def download(url: str, destination: Path, max_bytes: int) -> tuple[int, str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with _safe_open(url) as response, temporary.open("wb") as output:
            declared = response.headers.get("Content-Length")
            if declared and (not declared.isdigit() or int(declared) > max_bytes):
                raise ValueError(f"declared content length exceeds {max_bytes} bytes")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"download exceeded {max_bytes} bytes")
                digest.update(block)
                output.write(block)
        os.replace(temporary, destination)
        return size, digest.hexdigest(), "acquired"
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _manifest_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def license_is_acquirable(value: str | None) -> bool:
    """Conservative allow-list; public availability alone never grants reuse."""
    if not value:
        return False
    normalized = " ".join(value.strip().lower().split())
    allowed = {
        "cc0", "cc0 1.0", "cc0-1.0", "public domain",
        "cc-by", "cc-by 4.0", "cc-by-4.0", "cc-by 3.0", "cc-by-3.0",
        "cc-by-sa", "cc-by-sa 4.0", "cc-by-sa-4.0", "cc-by-sa 3.0", "cc-by-sa-3.0",
        "odc-by", "odc-by-1.0", "odbl", "odbl-1.0",
        "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    }
    return normalized in allowed


def run_inventory(catalog_root: Path, manifest: Path, data_root: Path, *,
                  max_file_bytes: int, max_dataset_bytes: int, max_total_bytes: int,
                  acquire: bool) -> dict[str, int]:
    records, errors = discover(catalog_root)
    counts = {"discovered": len(records), "metadata_errors": len(errors), "acquired": 0,
              "deferred": 0, "failed": 0, "bytes": 0}
    for error in errors:
        _manifest_append(manifest, {"timestamp": int(time.time()), **error})
    for record in records:
        base = {"timestamp": int(time.time()), **asdict(record)}
        # Catalog and dataset licenses are distinct. Unknown license is not permission.
        if license_is_acquirable(record.license):
            base["license_status"] = "recognized_permissive_or_open"
        elif record.license:
            base["license_status"] = "unrecognized_or_restrictive_requires_review"
        else:
            base["license_status"] = "unknown_requires_review"
        assets, reason = resolve_assets(record)
        if not assets:
            _manifest_append(manifest, {**base, "status": "deferred", "reason": reason})
            counts["deferred"] += 1
            continue
        if not acquire:
            _manifest_append(manifest, {**base, "status": "candidate", "asset_urls": assets})
            continue
        dataset_bytes = 0
        for index, url in enumerate(assets, start=1):
            if base["license_status"] != "recognized_permissive_or_open":
                _manifest_append(manifest, {**base, "status": "deferred", "asset_url": url,
                                            "reason": "license is missing, unrecognized, or restrictive; acquisition requires review"})
                counts["deferred"] += 1
                continue
            if counts["bytes"] >= max_total_bytes or dataset_bytes >= max_dataset_bytes:
                _manifest_append(manifest, {**base, "status": "deferred", "asset_url": url,
                                            "reason": "configured total or per-dataset byte budget reached"})
                counts["deferred"] += 1
                continue
            digest_key = hashlib.sha256((record.dataset_id + "|" + url).encode()).hexdigest()[:20]
            destination = data_root / record.category / digest_key / _safe_name(url)
            if destination.exists():
                existing_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
                row = {**base, "status": "acquired", "asset_url": url, "path": str(destination),
                       "sha256": existing_hash, "bytes": destination.stat().st_size, "reused": True}
                _manifest_append(manifest, row)
                counts["acquired"] += 1
                continue
            limit = min(max_file_bytes, max_dataset_bytes - dataset_bytes, max_total_bytes - counts["bytes"])
            try:
                size, digest, status = download(url, destination, limit)
                dataset_bytes += size
                counts["bytes"] += size
                counts["acquired"] += 1
                _manifest_append(manifest, {**base, "status": status, "asset_url": url,
                                            "path": str(destination), "sha256": digest,
                                            "bytes": size, "reused": False})
            except Exception as exc:
                counts["failed"] += 1
                _manifest_append(manifest, {**base, "status": "failed", "asset_url": url,
                                            "reason": type(exc).__name__ + ": " + str(exc)})
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--acquire", action="store_true", help="download resolvable assets; otherwise inventory only")
    parser.add_argument("--max-file-mib", type=int, default=64)
    parser.add_argument("--max-dataset-mib", type=int, default=256)
    parser.add_argument("--max-total-gib", type=int, default=20)
    args = parser.parse_args(argv)
    if not (args.catalog_root / "core").is_dir():
        parser.error(f"expected apd-core checkout with core/ at {args.catalog_root}")
    counts = run_inventory(
        args.catalog_root, args.manifest, args.data_root,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
        max_dataset_bytes=args.max_dataset_mib * 1024 * 1024,
        max_total_bytes=args.max_total_gib * 1024 * 1024 * 1024,
        acquire=args.acquire,
    )
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] or counts["metadata_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
