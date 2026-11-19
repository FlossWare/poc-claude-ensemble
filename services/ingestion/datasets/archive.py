"""PR 3: bounded archive inspection and extraction.

An archive that arrived over the network is untrusted.  This module never calls
``extractall`` and never trusts a member name, a declared size or a compression
ratio:

* names are validated component by component (no absolute paths, ``..``,
  backslashes, drive letters, NULs or control characters);
* links and devices are refused rather than materialised;
* encrypted members and nested archives are reported with their own outcomes;
* entry count, per-member and total uncompressed bytes and the compression
  ratio are checked from the central directory or headers **and** enforced
  again while the bytes stream out;
* extraction opens every destination with ``O_NOFOLLOW | O_CREAT | O_EXCL``
  under a directory this module created with mode ``0700``.

The member-name rules live here rather than in :mod:`ingestion.datasets.catalog`
so the catalog tarball and dataset archives cannot drift apart.
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import Config

ARCHIVE_VERSION = 1

ZIP_SUFFIXES = (".zip",)
TAR_SUFFIXES = (".tar", ".tgz", ".tbz2", ".txz")
COMPRESSED_TAR_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz")
SINGLE_COMPRESSED_SUFFIXES = (".gz", ".bz2", ".xz")
#: Formats this module can open.  ``.7z``/``.rar``/``.zst`` are recognised as
#: archives by the resolver but are not implemented here.
SUPPORTED_SUFFIXES = (
    COMPRESSED_TAR_SUFFIXES
    + TAR_SUFFIXES
    + ZIP_SUFFIXES
    + SINGLE_COMPRESSED_SUFFIXES
)

ARCHIVE_MEDIA_TYPES = {
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
    "application/x-tar": "tar",
    "application/gzip": "tar.gz",
    "application/x-gzip": "tar.gz",
    "application/x-bzip2": "tar.bz2",
    "application/x-xz": "tar.xz",
}

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_CHUNK = 64 * 1024
_JUNK_PREFIXES = ("__MACOSX/",)
_JUNK_BASENAMES = ("._", ".DS_Store")


class ArchiveError(Exception):
    """An archive failed a safety or integrity check."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


@dataclass(frozen=True)
class Member:
    name: str
    size: int
    compressed_size: int | None
    kind: str = "file"

    def to_artifact(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size": self.size,
            "compressed_size": self.compressed_size,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Inspection:
    archive_format: str
    members: tuple[Member, ...]
    data_members: tuple[Member, ...]
    nested: tuple[str, ...]
    junk: tuple[str, ...]
    total_uncompressed: int
    compressed_bytes: int
    ratio: float | None

    def to_artifact(self) -> dict[str, object]:
        return {
            "archive_version": ARCHIVE_VERSION,
            "format": self.archive_format,
            "member_count": len(self.members),
            "file_count": sum(1 for member in self.members if member.kind == "file"),
            "data_member_count": len(self.data_members),
            "total_uncompressed": self.total_uncompressed,
            "compressed_bytes": self.compressed_bytes,
            "ratio": self.ratio,
            "nested": list(self.nested[:20]),
            "junk": list(self.junk[:20]),
            "members": [member.to_artifact() for member in self.members[:50]],
        }


# ---------------------------------------------------------------------------
# Name validation (shared with the catalog tarball)
# ---------------------------------------------------------------------------


def unsafe_member_reason(name: str) -> str | None:
    """Return a short reason why ``name`` is unsafe, or ``None``."""
    if not name or "\x00" in name:
        return "member has an empty or NUL name"
    if "\\" in name:
        return "member contains a backslash"
    if name.startswith("/"):
        return "member uses an absolute path"
    if _DRIVE_RE.match(name):
        return "member uses a drive letter"
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        return "member contains control characters"
    parts = [part for part in name.split("/") if part not in {"", "."}]
    if not parts:
        return "member has no path components"
    if any(part == ".." for part in parts):
        return "member traverses parent directories"
    if any(_DRIVE_RE.match(part) for part in parts):
        return "member uses a drive letter"
    return None


def validate_member_name(name: str) -> list[str]:
    reason = unsafe_member_reason(name)
    if reason is not None:
        raise ArchiveError("archive_unsafe_path", reason)
    return [part for part in name.split("/") if part not in {"", "."}]


def is_junk_member(name: str) -> bool:
    lowered = name.lower()
    if any(lowered.startswith(prefix.lower()) for prefix in _JUNK_PREFIXES):
        return True
    basename = lowered.rsplit("/", 1)[-1]
    return any(basename.startswith(marker.lower()) for marker in _JUNK_BASENAMES)


def has_archive_suffix(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def archive_format_for(
    name: str | None, media_type: str | None = None, prefix: bytes | None = None
) -> str | None:
    """Best-effort archive format from the name, the declared type, or magic."""
    if name:
        lowered = name.lower()
        for suffix, value in (
            (".tar.gz", "tar.gz"),
            (".tar.bz2", "tar.bz2"),
            (".tar.xz", "tar.xz"),
            (".tgz", "tar.gz"),
            (".tbz2", "tar.bz2"),
            (".txz", "tar.xz"),
            (".zip", "zip"),
            (".tar", "tar"),
            (".gz", "gz"),
            (".bz2", "bz2"),
            (".xz", "xz"),
        ):
            if lowered.endswith(suffix):
                return value
    if media_type:
        value = ARCHIVE_MEDIA_TYPES.get(media_type.split(";", 1)[0].strip().lower())
        if value:
            return value
    if prefix:
        if prefix.startswith(b"PK\x03\x04") or prefix.startswith(b"PK\x05\x06"):
            return "zip"
        if prefix.startswith(b"\x1f\x8b"):
            return "tar.gz"
        if prefix.startswith(b"BZh"):
            return "tar.bz2"
        if prefix.startswith(b"\xfd7zXZ\x00"):
            return "tar.xz"
        if len(prefix) > 262 and prefix[257:262] == b"ustar":
            return "tar"
    return None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _magic_prefix(path: Path, limit: int = 512) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except OSError:
        return b""


def _looks_like_tar(path: Path, archive_format: str) -> bool:
    opener = {
        "tar.gz": gzip.open,
        "tar.bz2": bz2.open,
        "tar.xz": lzma.open,
        "gz": gzip.open,
        "bz2": bz2.open,
        "xz": lzma.open,
    }[archive_format]
    try:
        with opener(path, "rb") as handle:
            head = handle.read(512)
    except (OSError, EOFError, ValueError, lzma.LZMAError):
        return False
    return len(head) >= 512 and head[257:262] == b"ustar"


def _zip_inspection(path: Path, config: Config) -> Inspection:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > config.archive_max_entries:
                raise ArchiveError(
                    "archive_too_many_entries",
                    f"archive has {len(infos)} members (cap {config.archive_max_entries})",
                )
            members: list[Member] = []
            seen: set[str] = set()
            total = 0
            nested: list[str] = []
            junk: list[str] = []
            for info in infos:
                validate_member_name(info.filename)
                if info.filename in seen:
                    raise ArchiveError(
                        "archive_unsafe_path", f"duplicate member: {info.filename}"
                    )
                seen.add(info.filename)
                if info.flag_bits & 0x1:
                    raise ArchiveError(
                        "archive_encrypted", f"member is encrypted: {info.filename}"
                    )
                if info.is_dir():
                    members.append(
                        Member(info.filename, 0, info.compress_size, kind="dir")
                    )
                    continue
                # External attributes: unix mode lives in the high 16 bits.
                mode = (info.external_attr >> 16) & 0o170000
                if mode and mode not in (0o100000, 0o040000):
                    raise ArchiveError(
                        "archive_link_or_device",
                        f"member is a link or device: {info.filename}",
                    )
                if info.file_size > config.max_file_bytes:
                    raise ArchiveError(
                        "archive_bomb",
                        f"member exceeds the per-file cap: {info.filename}",
                    )
                if info.compress_size > 0 and (
                    info.file_size / info.compress_size > config.archive_max_ratio
                ):
                    raise ArchiveError(
                        "archive_bomb",
                        f"member exceeds the compression ratio cap: {info.filename}",
                    )
                total += info.file_size
                if total > config.archive_max_uncompressed:
                    raise ArchiveError(
                        "archive_bomb", "archive exceeds the uncompressed byte cap"
                    )
                if is_junk_member(info.filename):
                    junk.append(info.filename)
                elif has_archive_suffix(info.filename) and config.archive_max_depth <= 0:
                    nested.append(info.filename)
                members.append(Member(info.filename, info.file_size, info.compress_size))
    except ArchiveError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise ArchiveError("parse_error", f"zip could not be read: {type(exc).__name__}") from exc
    return _finish("zip", members, nested, junk, total, path)


def _tar_inspection(path: Path, config: Config, archive_format: str) -> Inspection:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            infos = archive.getmembers()
            if len(infos) > config.archive_max_entries:
                raise ArchiveError(
                    "archive_too_many_entries",
                    f"archive has {len(infos)} members (cap {config.archive_max_entries})",
                )
            members: list[Member] = []
            seen: set[str] = set()
            total = 0
            nested: list[str] = []
            junk: list[str] = []
            for info in infos:
                validate_member_name(info.name)
                if info.name in seen:
                    raise ArchiveError(
                        "archive_unsafe_path", f"duplicate member: {info.name}"
                    )
                seen.add(info.name)
                if info.isdir():
                    members.append(Member(info.name, 0, None, kind="dir"))
                    continue
                if not info.isfile():
                    raise ArchiveError(
                        "archive_link_or_device",
                        f"member is a link or device: {info.name}",
                    )
                if info.size > config.max_file_bytes:
                    raise ArchiveError(
                        "archive_bomb",
                        f"member exceeds the per-file cap: {info.name}",
                    )
                total += info.size
                if total > config.archive_max_uncompressed:
                    raise ArchiveError(
                        "archive_bomb", "archive exceeds the uncompressed byte cap"
                    )
                if is_junk_member(info.name):
                    junk.append(info.name)
                elif has_archive_suffix(info.name) and config.archive_max_depth <= 0:
                    nested.append(info.name)
                members.append(Member(info.name, info.size, None))
    except ArchiveError:
        raise
    except (tarfile.TarError, OSError, ValueError, EOFError) as exc:
        raise ArchiveError(
            "parse_error", f"tar could not be read: {type(exc).__name__}"
        ) from exc
    return _finish(archive_format, members, nested, junk, total, path)


def _single_inspection(
    path: Path, config: Config, archive_format: str, *, name: str
) -> Inspection:
    """A single compressed file (``.gz``/``.bz2``/``.xz`` that is not a tar)."""
    inner = re.sub(r"\.(gz|bz2|xz)$", "", name.rsplit("/", 1)[-1], flags=re.IGNORECASE)
    inner = inner or "content"
    opener = {"gz": gzip.open, "bz2": bz2.open, "xz": lzma.open}[archive_format]
    total = 0
    try:
        with opener(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.archive_max_uncompressed:
                    raise ArchiveError(
                        "archive_bomb", "archive exceeds the uncompressed byte cap"
                    )
                if total > config.max_file_bytes:
                    raise ArchiveError(
                        "archive_bomb", "member exceeds the per-file cap"
                    )
    except ArchiveError:
        raise
    except (OSError, EOFError, ValueError, lzma.LZMAError) as exc:
        raise ArchiveError(
            "parse_error", f"compressed file could not be read: {type(exc).__name__}"
        ) from exc
    member = Member(inner, total, path.stat().st_size if path.exists() else None)
    return _finish(
        archive_format,
        [member],
        [],
        [inner] if is_junk_member(inner) else [],
        total,
        path,
    )


def _finish(
    archive_format: str,
    members: list[Member],
    nested: list[str],
    junk: list[str],
    total: int,
    path: Path,
) -> Inspection:
    try:
        compressed = path.stat().st_size
    except OSError:
        compressed = 0
    ratio = (total / compressed) if compressed > 0 else None
    excluded = set(nested) | set(junk)
    data_members = tuple(
        member
        for member in members
        if member.kind == "file" and member.name not in excluded
    )
    return Inspection(
        archive_format=archive_format,
        members=tuple(members),
        data_members=data_members,
        nested=tuple(nested),
        junk=tuple(junk),
        total_uncompressed=total,
        compressed_bytes=compressed,
        ratio=ratio,
    )


def inspect(
    path: Path,
    *,
    name: str,
    archive_format: str | None,
    config: Config,
) -> Inspection:
    """Inspect an archive without extracting it."""
    path = Path(path)
    if archive_format is None:
        archive_format = archive_format_for(name, None, _magic_prefix(path))
    if archive_format is None:
        raise ArchiveError("unsupported_format", "archive format could not be determined")
    if archive_format in {"gz", "bz2", "xz"} and _looks_like_tar(path, archive_format):
        # A compressed file whose payload is a tar is a tarball, whatever the
        # name says (``bundle.gz`` containing a ustar stream).
        archive_format = f"tar.{archive_format}"
    if archive_format == "zip":
        return _zip_inspection(path, config)
    if archive_format in {"tar.gz", "tar.bz2", "tar.xz"}:
        # Compressed bytes without a ``.tar.*`` name (or a sniffed gzip) may be
        # a single compressed file rather than a tar.  The ustar magic in the
        # first decompressed header block decides, so a ``.csv.gz`` served at a
        # URL with no suffix is still handled.
        if _looks_like_tar(path, archive_format):
            return _tar_inspection(path, config, archive_format)
        return _single_inspection(
            path, config, archive_format.split(".", 1)[1], name=name
        )
    if archive_format == "tar":
        return _tar_inspection(path, config, archive_format)
    if archive_format in {"gz", "bz2", "xz"}:
        return _single_inspection(path, config, archive_format, name=name)
    raise ArchiveError(
        "adapter_unavailable", f"archive format is not implemented: {archive_format}"
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedMember:
    path: Path
    name: str
    size: int

    def to_artifact(self) -> dict[str, object]:
        return {"name": self.name, "size": self.size}


@contextmanager
def extracted_members(
    path: Path,
    *,
    name: str,
    archive_format: str | None,
    config: Config,
    inspection: Inspection | None = None,
):
    """Yield the safe, data-like members of an archive, then clean up.

    The caller must not keep the yielded paths past the ``with`` block: the
    temporary directory is removed on exit.
    """
    report = inspection or inspect(
        path, name=name, archive_format=archive_format, config=config
    )
    with temporary_directory(config) as root:
        yield _extract_into(path, root, report, config)


@contextmanager
def temporary_directory(config: Config):
    """A fresh ``0700`` directory under the data root, removed on exit."""
    base = Path(config.data_root) / "tmp"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = Path(tempfile.mkdtemp(prefix="archive-", dir=str(base)))
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _safe_destination(root: Path, parts: list[str]) -> Path:
    target = root.joinpath(*parts)
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    if target_real != root_real and not target_real.startswith(root_real + os.sep):
        raise ArchiveError("archive_unsafe_path", "member escapes the extraction root")
    return target


def _open_destination(target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )


def _copy_bounded(source, target: Path, *, declared: int, config: Config) -> int:
    descriptor = _open_destination(target)
    written = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > declared:
                    raise ArchiveError(
                        "archive_bomb",
                        "member produced more bytes than its header declared",
                    )
                if written > config.max_file_bytes:
                    raise ArchiveError("archive_bomb", "member exceeds the per-file cap")
                output.write(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if written != declared:
        raise ArchiveError("parse_error", "member is truncated")
    return written


def _extract_into(
    path: Path, root: Path, inspection: Inspection, config: Config
) -> list[ExtractedMember]:
    selected = {member.name for member in inspection.data_members}
    extracted: list[ExtractedMember] = []
    written_total = 0
    try:
        if inspection.archive_format == "zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.filename not in selected:
                        continue
                    parts = validate_member_name(info.filename)
                    target = _safe_destination(root, parts)
                    with archive.open(info) as source:
                        size = _copy_bounded(
                            source, target, declared=info.file_size, config=config
                        )
                    written_total += size
                    if written_total > config.archive_max_uncompressed:
                        raise ArchiveError(
                            "archive_bomb", "archive exceeds the uncompressed byte cap"
                        )
                    extracted.append(ExtractedMember(target, info.filename, size))
        elif inspection.archive_format in {"tar", "tar.gz", "tar.bz2", "tar.xz"}:
            with tarfile.open(path, mode="r:*") as archive:
                for info in archive.getmembers():
                    if info.name not in selected:
                        continue
                    if not info.isfile():
                        raise ArchiveError(
                            "archive_link_or_device",
                            f"member is a link or device: {info.name}",
                        )
                    parts = validate_member_name(info.name)
                    target = _safe_destination(root, parts)
                    source = archive.extractfile(info)
                    if source is None:
                        raise ArchiveError("parse_error", f"member has no data: {info.name}")
                    with source:
                        size = _copy_bounded(
                            source, target, declared=info.size, config=config
                        )
                    written_total += size
                    if written_total > config.archive_max_uncompressed:
                        raise ArchiveError(
                            "archive_bomb", "archive exceeds the uncompressed byte cap"
                        )
                    extracted.append(ExtractedMember(target, info.name, size))
        else:
            member = inspection.data_members[0]
            opener = {
                "gz": gzip.open,
                "bz2": bz2.open,
                "xz": lzma.open,
            }[inspection.archive_format]
            target = _safe_destination(root, validate_member_name(member.name))
            with opener(path, "rb") as source:
                size = _copy_bounded(
                    source, target, declared=member.size, config=config
                )
            extracted.append(ExtractedMember(target, member.name, size))
    except ArchiveError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, OSError, ValueError, EOFError) as exc:
        raise ArchiveError(
            "parse_error", f"archive could not be extracted: {type(exc).__name__}"
        ) from exc
    if not extracted:
        if inspection.nested:
            raise ArchiveError(
                "archive_nested", "archive contains only nested archives"
            )
        raise ArchiveError("empty_content", "archive contains no data members")
    return extracted
