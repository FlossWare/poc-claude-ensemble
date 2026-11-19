"""Configuration for the dataset ingestion library.

Every knob is read from ``POC_CLAUDE_ENSEMBLE_DATASET_<KNOB>`` and may be
overridden by a CLI flag.  Defaults are deliberately small for the POC.
Values are resolved when :meth:`Config.load` is called (not at import time)
so tests can redirect the data root safely.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "POC_CLAUDE_ENSEMBLE_DATASET_"

DEFAULT_ALLOW_FAMILIES = (
    "public-domain",
    "cc0",
    "cc-by",
    "cc-by-sa",
    "mit",
    "apache",
    "odbl",
    "ogl",
)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    return default


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(os.path.expanduser(raw.strip()))


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _env_families(name: str, default: frozenset[str]) -> frozenset[str]:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def _pyarrow_available() -> bool:
    try:
        return importlib.util.find_spec("pyarrow") is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class Config:
    """Resolved configuration for one invocation."""

    data_root: Path
    catalog_repo: str = "awesomedata/apd-core"
    catalog_ref: str = "master"
    catalog_dir: Path | None = None
    readme_repo: str = "awesomedata/awesome-public-datasets"
    readme_ref: str = "master"
    readme_path: Path | None = None

    catalog_tarball_max_bytes: int = 64 * 1024 * 1024
    catalog_max_files: int = 5_000
    catalog_max_file_bytes: int = 256 * 1024
    catalog_readme_max_bytes: int = 2 * 1024 * 1024
    catalog_license_max_bytes: int = 64 * 1024

    yaml_max_depth: int = 32
    yaml_max_nodes: int = 10_000
    yaml_max_aliases: int = 0

    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    request_max_seconds: float = 120.0
    max_redirects: int = 3

    concurrency: int = 4
    per_host_concurrency: int = 1
    per_host_rps: float = 1.0

    schemes: tuple[str, ...] = ("https", "http")
    enable_ftp: bool = False
    respect_robots: bool = True

    max_file_bytes: int = 64 * 1024 * 1024
    max_dataset_bytes: int = 1024 * 1024 * 1024
    max_run_bytes: int = 2 * 1024 * 1024 * 1024
    max_run_datasets: int = 50
    max_run_seconds: int = 3_600
    min_free_disk_bytes: int = 5 * 1024 * 1024 * 1024

    archive_max_entries: int = 10_000
    archive_max_uncompressed: int = 1024 * 1024 * 1024
    archive_max_ratio: int = 100
    archive_max_depth: int = 0

    max_records_per_file: int = 100_000
    max_fields: int = 512
    max_cell_chars: int = 100_000
    max_line_chars: int = 1_000_000
    max_json_depth: int = 64
    max_xml_depth: int = 64
    max_shard_chars: int = 3_500_000

    embed_batch: int = 64
    lexical_batch: int = 128
    vector_batch: int = 128

    license_allow_families: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_ALLOW_FAMILIES)
    )
    parquet_enabled: bool = field(default_factory=_pyarrow_available)
    pdf_enabled: bool = False

    github_token: str | None = None
    recheck_after_days: int = 30

    # PR 2: resolution.  These knobs bound every resolver network call.  File
    # bodies are never downloaded at this stage; only bounded API JSON and one
    # bounded landing page per entry are read.
    landing_page_max_bytes: int = 512 * 1024
    landing_page_max_links: int = 200
    api_max_bytes: int = 2 * 1024 * 1024
    max_resolved_sources: int = 20
    resolve_retry_backoff_seconds: int = 3_600
    resolve_cache_enabled: bool = True

    probe_enabled: bool = False
    limit: int | None = None
    sample: int | None = None
    ledger_enabled: bool = True
    schema_version: int = 1

    def replace(self, **changes: object) -> "Config":
        return dataclasses.replace(self, **changes)

    @classmethod
    def load(
        cls,
        *,
        catalog_dir: str | os.PathLike[str] | None = None,
        catalog_ref: str | None = None,
        readme_path: str | os.PathLike[str] | None = None,
        probe_enabled: bool | None = None,
        limit: int | None = None,
        sample: int | None = None,
        ledger_enabled: bool | None = None,
        resolve_cache_enabled: bool | None = None,
        data_root: str | os.PathLike[str] | None = None,
    ) -> "Config":
        default_root = Path(
            os.path.expanduser("~/.FlossWare/poc-claude-ensemble/data/datasets")
        )
        env_root = _env_path("DATA_ROOT", default_root)
        env_catalog_dir = _env_path("CATALOG_DIR", None)
        env_readme = _env_path("README_PATH", None)

        resolved_root = (
            Path(os.path.expanduser(str(data_root))) if data_root is not None else env_root
        )
        assert resolved_root is not None
        resolved_catalog_dir = (
            Path(os.path.expanduser(str(catalog_dir)))
            if catalog_dir is not None
            else env_catalog_dir
        )
        resolved_readme = (
            Path(os.path.expanduser(str(readme_path)))
            if readme_path is not None
            else env_readme
        )

        enabled = _env_bool("LEDGER_ENABLED", True) if ledger_enabled is None else ledger_enabled
        probe = _env_bool("PROBE_ENABLED", False) if probe_enabled is None else probe_enabled
        env_limit = _env_int("LIMIT", -1)
        if limit is None:
            limit = None if env_limit < 0 else env_limit
        env_sample = _env_int("SAMPLE", -1)
        if sample is None:
            sample = None if env_sample < 0 else env_sample
        cache_enabled = (
            _env_bool("RESOLVE_CACHE_ENABLED", True)
            if resolve_cache_enabled is None
            else resolve_cache_enabled
        )

        return cls(
            data_root=resolved_root,
            catalog_repo=_env("CATALOG_REPO", "awesomedata/apd-core") or "awesomedata/apd-core",
            catalog_ref=catalog_ref or _env("CATALOG_REF", "master") or "master",
            catalog_dir=resolved_catalog_dir,
            readme_repo=_env("README_REPO", "awesomedata/awesome-public-datasets")
            or "awesomedata/awesome-public-datasets",
            readme_ref=_env("README_REF", "master") or "master",
            readme_path=resolved_readme,
            catalog_tarball_max_bytes=_env_int("CATALOG_TARBALL_MAX_BYTES", 64 * 1024 * 1024),
            catalog_max_files=_env_int("CATALOG_MAX_FILES", 5_000),
            catalog_max_file_bytes=_env_int("CATALOG_MAX_FILE_BYTES", 256 * 1024),
            catalog_readme_max_bytes=_env_int("CATALOG_README_MAX_BYTES", 2 * 1024 * 1024),
            catalog_license_max_bytes=_env_int("CATALOG_LICENSE_MAX_BYTES", 64 * 1024),
            yaml_max_depth=_env_int("YAML_MAX_DEPTH", 32),
            yaml_max_nodes=_env_int("YAML_MAX_NODES", 10_000),
            yaml_max_aliases=_env_int("YAML_MAX_ALIASES", 0),
            connect_timeout=_env_float("CONNECT_TIMEOUT", 10.0),
            read_timeout=_env_float("READ_TIMEOUT", 30.0),
            request_max_seconds=_env_float("REQUEST_MAX_SECONDS", 120.0),
            max_redirects=_env_int("MAX_REDIRECTS", 3),
            concurrency=max(1, _env_int("CONCURRENCY", 4)),
            per_host_concurrency=max(1, _env_int("PER_HOST_CONCURRENCY", 1)),
            per_host_rps=max(0.001, _env_float("PER_HOST_RPS", 1.0)),
            schemes=_env_tuple("SCHEMES", ("https", "http")),
            enable_ftp=_env_bool("ENABLE_FTP", False),
            respect_robots=_env_bool("RESPECT_ROBOTS", True),
            max_file_bytes=_env_int("MAX_FILE_BYTES", 64 * 1024 * 1024),
            max_dataset_bytes=_env_int("MAX_DATASET_BYTES", 1024 * 1024 * 1024),
            max_run_bytes=_env_int("MAX_RUN_BYTES", 2 * 1024 * 1024 * 1024),
            max_run_datasets=_env_int("MAX_RUN_DATASETS", 50),
            max_run_seconds=_env_int("MAX_RUN_SECONDS", 3_600),
            min_free_disk_bytes=_env_int("MIN_FREE_DISK_BYTES", 5 * 1024 * 1024 * 1024),
            archive_max_entries=_env_int("ARCHIVE_MAX_ENTRIES", 10_000),
            archive_max_uncompressed=_env_int("ARCHIVE_MAX_UNCOMPRESSED", 1024 * 1024 * 1024),
            archive_max_ratio=_env_int("ARCHIVE_MAX_RATIO", 100),
            archive_max_depth=_env_int("ARCHIVE_MAX_DEPTH", 0),
            max_records_per_file=_env_int("MAX_RECORDS_PER_FILE", 100_000),
            max_fields=_env_int("MAX_FIELDS", 512),
            max_cell_chars=_env_int("MAX_CELL_CHARS", 100_000),
            max_line_chars=_env_int("MAX_LINE_CHARS", 1_000_000),
            max_json_depth=_env_int("MAX_JSON_DEPTH", 64),
            max_xml_depth=_env_int("MAX_XML_DEPTH", 64),
            max_shard_chars=_env_int("MAX_SHARD_CHARS", 3_500_000),
            embed_batch=_env_int("EMBED_BATCH", 64),
            lexical_batch=_env_int("LEXICAL_BATCH", 128),
            vector_batch=_env_int("VECTOR_BATCH", 128),
            license_allow_families=_env_families(
                "LICENSE_ALLOW_FAMILIES", frozenset(DEFAULT_ALLOW_FAMILIES)
            ),
            parquet_enabled=_env_bool("PARQUET_ENABLED", _pyarrow_available()),
            pdf_enabled=_env_bool("PDF_ENABLED", False),
            github_token=_env("GITHUB_TOKEN") or None,
            recheck_after_days=_env_int("RECHECK_AFTER_DAYS", 30),
            landing_page_max_bytes=_env_int("LANDING_PAGE_MAX_BYTES", 512 * 1024),
            landing_page_max_links=_env_int("LANDING_PAGE_MAX_LINKS", 200),
            api_max_bytes=_env_int("API_MAX_BYTES", 2 * 1024 * 1024),
            max_resolved_sources=_env_int("MAX_RESOLVED_SOURCES", 20),
            resolve_retry_backoff_seconds=_env_int("RESOLVE_RETRY_BACKOFF_SECONDS", 3_600),
            resolve_cache_enabled=cache_enabled,
            probe_enabled=probe,
            limit=limit,
            sample=sample,
            ledger_enabled=enabled,
        )
