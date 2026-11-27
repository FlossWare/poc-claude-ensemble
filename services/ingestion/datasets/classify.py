"""License / access / format / scheme classification and disposition."""
from __future__ import annotations

import os
import re
import urllib.parse

from .config import Config

MIB = 1024 * 1024
GIB = 1024 * MIB

FORMAT_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".avro": "avro",
    ".zip": "zip",
    ".tar": "tar",
    ".tgz": "gzip",
    ".gz": "gzip",
    ".bz2": "bzip2",
    ".xz": "xz",
    ".7z": "7z",
    ".rar": "rar",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".geojson": "geojson",
    ".shp": "shapefile",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".sqlite": "sqlite",
    ".db": "sqlite",
    ".xlsx": "excel",
    ".xls": "excel",
    ".sav": "spss",
    ".dta": "stata",
    ".rds": "rds",
    ".yaml": "yaml",
    ".yml": "yaml",
}

FORMAT_HINTS = (
    (re.compile(r"\bcsv\b"), "csv"),
    (re.compile(r"\btsv\b"), "tsv"),
    (re.compile(r"\bjsonl\b|\bndjson\b"), "jsonl"),
    (re.compile(r"\bjson\b"), "json"),
    (re.compile(r"\bparquet\b"), "parquet"),
    (re.compile(r"\bxml\b"), "xml"),
    (re.compile(r"\bnetcdf\b"), "netcdf"),
    (re.compile(r"\bhdf5?\b"), "hdf5"),
    (re.compile(r"\bshapefile\b"), "shapefile"),
    (re.compile(r"\bgeojson\b"), "geojson"),
    (re.compile(r"\bsqlite\b"), "sqlite"),
    (re.compile(r"\bapi\b"), "api"),
    (re.compile(r"\bzip\b"), "zip"),
    (re.compile(r"\btarball\b|\btar\b"), "tar"),
    (re.compile(r"\bpdf\b"), "pdf"),
    (re.compile(r"\bmarkdown\b"), "markdown"),
    (re.compile(r"\btext\b|\btxt\b"), "text"),
)

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")


def _urls(row: dict[str, object]) -> list[str]:
    urls: list[str] = []
    homepage = row.get("homepage")
    if isinstance(homepage, str) and homepage.strip():
        urls.append(homepage.strip())
    for source in row.get("sources") or []:
        if isinstance(source, dict):
            access_url = source.get("access_url")
            if isinstance(access_url, str) and access_url.strip():
                urls.append(access_url.strip())
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _allowed_schemes(config: Config) -> set[str]:
    allowed = set(config.schemes)
    if config.enable_ftp:
        allowed.add("ftp")
    return allowed


def formats_for(row: dict[str, object]) -> list[str]:
    formats: set[str] = set()
    for url in _urls(row):
        try:
            path = urllib.parse.urlsplit(url).path
        except ValueError:
            continue
        extension = os.path.splitext(path)[1].lower()
        mapped = FORMAT_EXTENSIONS.get(extension)
        if mapped:
            formats.add(mapped)
    for source in row.get("sources") or []:
        if not isinstance(source, dict):
            continue
        name = source.get("name")
        if isinstance(name, str):
            low = name.lower()
            for pattern, fmt in FORMAT_HINTS:
                if pattern.search(low):
                    formats.add(fmt)
    extra = row.get("extra") or {}
    if isinstance(extra, dict):
        declared = extra.get("format")
        declared_items = declared if isinstance(declared, list) else [declared]
        for item in declared_items:
            if not isinstance(item, str):
                continue
            low = item.lower().strip()
            if low.startswith(".") and low in FORMAT_EXTENSIONS:
                formats.add(FORMAT_EXTENSIONS[low])
            for pattern, fmt in FORMAT_HINTS:
                if pattern.search(low):
                    formats.add(fmt)
    if not formats:
        formats.add("unknown")
    return sorted(formats)[:16]


def schemes_for(row: dict[str, object], config: Config) -> tuple[list[str], list[str]]:
    schemes: set[str] = set()
    unsupported: set[str] = set()
    allowed = _allowed_schemes(config)
    for url in _urls(row):
        match = _SCHEME_RE.match(url)
        if not match:
            continue
        scheme = match.group(1).lower()
        schemes.add(scheme)
        if scheme not in allowed:
            unsupported.add(scheme)
    return sorted(schemes), sorted(unsupported)


def access_hint(row: dict[str, object], config: Config) -> tuple[str, str | None]:
    """Metadata-only access hint; never claims a successful measurement."""
    urls = _urls(row)
    if not urls:
        return "unknown", "missing_url"
    homepage = row.get("homepage")
    primary = homepage.strip() if isinstance(homepage, str) and homepage.strip() else urls[0]
    match = _SCHEME_RE.match(primary)
    if not match:
        return "unknown", "invalid_url"
    scheme = match.group(1).lower()
    if scheme not in _allowed_schemes(config):
        # known-but-disabled schemes are unsupported_scheme; anything
        # else (including the observed "hhttps" typo in the wild) is
        # invalid_url — but http/https disabled by config must be
        # labeled unsupported_scheme, not invalid_url
        known = {"ftp", "http", "https", "magnet", "svn", "svn+ssh", "git", "s3",
                 "hhttps"}  # observed typo in wild rows
        code = "unsupported_scheme" if scheme in known else "invalid_url"
        return "inaccessible", code
    try:
        host = (urllib.parse.urlsplit(primary).hostname or "").lower()
    except ValueError:
        return "unknown", "invalid_url"
    if host == "kaggle.com" or host.endswith(".kaggle.com"):
        return "gated", "gated_authentication"
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        # Hugging Face has public and gated datasets; only a probe can tell.
        return "unknown", None
    return "unknown", None


def size_class(byte_count: int | None) -> str:
    if byte_count is None:
        return "unknown"
    if byte_count <= 64 * MIB:
        return "small"
    if byte_count <= GIB:
        return "medium"
    return "large"


def _next_action(status: str, reason_code: str | None) -> str:
    if status == "skipped":
        return "none"
    if status == "failed":
        return "fix_metadata"
    if reason_code == "resolver_not_implemented":
        return "resolve"
    if reason_code == "run_budget_exhausted":
        return "resume"
    if reason_code and reason_code.startswith("license_"):
        return "review_license"
    if reason_code in {
        "catalog_parse_error",
        "catalog_entry_invalid",
        "catalog_duplicate_id",
        "limits_exceeded",
    }:
        return "fix_metadata"
    return "review"


def classify_row(row: dict[str, object], config: Config, probe_result=None) -> None:
    """Fill ``classes``, ``disposition``, ``status`` and ``reason_code`` in place."""
    schemes, unsupported = schemes_for(row, config)
    formats = formats_for(row)

    license_field = row.get("license")
    dataset_license = license_field.get("dataset") if isinstance(license_field, dict) else None
    if not isinstance(dataset_license, dict):
        # malformed rows classify as invalid, not KeyError
        row["classes"] = {"access": "unknown", "access_reason": None,
                          "size": {"class": "unknown", "bytes": None, "source": "metadata"},
                          "format": formats, "scheme": schemes,
                          "scheme_unsupported": unsupported}
        row["disposition"] = "invalid"
        row["status"] = "failed"
        row["reason_code"] = "catalog_entry_invalid"
        row["next_action"] = "fix_metadata"
        return
    policy = str(dataset_license.get("policy") or "metadata_only_review")
    policy_reason = dataset_license.get("reason_code")
    review_required = bool(dataset_license.get("review_required"))

    if probe_result is not None and getattr(probe_result, "url", None):
        access = probe_result.access
        access_reason = probe_result.reason_code
        if probe_result.content_length is not None:
            size = {
                "class": size_class(probe_result.content_length),
                "bytes": probe_result.content_length,
                "source": "probe",
            }
        else:
            size = {"class": "unknown", "bytes": None, "source": "probe"}
        row["probe"] = probe_result.to_manifest()
    else:
        access, access_reason = access_hint(row, config)
        size = {"class": "unknown", "bytes": None, "source": "metadata"}
        row["probe"] = None

    row["classes"] = {
        "access": access,
        "access_reason": access_reason,
        "size": size,
        "format": formats,
        "scheme": schemes,
        "scheme_unsupported": unsupported,
    }
    row["disposition"] = policy

    if policy == "blocked":
        status, reason_code = "skipped", "license_restricted"
    elif policy == "metadata_only_review":
        status, reason_code = "deferred", policy_reason or "license_unknown_metadata_only"
    elif review_required:
        status, reason_code = "deferred", "license_review_required"
    else:
        # License policy sets the status (the test suite encodes that a
        # probe verdict overrides access but NOT license status). But the
        # ACCESS reason must still be surfaced — the old code dropped the
        # probe/hint reason_code entirely and always reported
        # resolver_not_implemented even when a concrete reason existed.
        status = "deferred"
        reason_code = access_reason or policy_reason or "resolver_not_implemented"
    row["status"] = status
    row["reason_code"] = reason_code
    row["next_action"] = _next_action(status, reason_code)
