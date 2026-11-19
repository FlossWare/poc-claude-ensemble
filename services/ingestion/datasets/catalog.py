"""Catalog snapshot acquisition, validation and README reconciliation.

The snapshot is untrusted.  Local directories are walked with symlinks
refused; remote snapshots are downloaded through :mod:`netguard` and expanded
member-by-member with no ``extractall``.  Every metadata file under ``core/``
is inventoried, regardless of suffix, so extensionless and ``.wml`` entries
are never silently dropped.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .config import Config

_META_RE = re.compile(r"\[`Meta <([^>]+)>`_\]")
_HEX_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class CatalogError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


@dataclass(frozen=True)
class CatalogFile:
    relpath: str
    category: str
    filename: str
    sha256: str
    size: int
    unsafe_reason: str | None = None


@dataclass
class CatalogSnapshot:
    repo: str
    commit: str
    snapshot_sha256: str
    root: Path
    core_dir: Path
    mode: str
    files: list[CatalogFile]
    readme_path: Path | None = None
    readme_entries: dict[str, dict[str, str]] = field(default_factory=dict)
    readme_meta_total: int = 0
    readme_warnings: list[str] = field(default_factory=list)
    catalog_license: str = "MIT"
    metadata_repo_license: str = "GPL-3.0"
    warnings: list[str] = field(default_factory=list)


def _split_relpath(relpath: str) -> tuple[str, str]:
    parts = PurePosixPath(relpath).parts
    if not parts:
        return "root", relpath
    if len(parts) == 1:
        return "root", parts[0]
    return parts[0], "/".join(parts[1:])


def _read_git_commit(root: Path) -> str | None:
    """Read HEAD without subprocesses so the connector never shells out."""
    git = root / ".git"
    try:
        if git.is_file():
            text = git.read_text(encoding="utf-8", errors="replace")
            match = re.match(r"gitdir:\s*(.+)", text.strip())
            if not match:
                return None
            gitdir = (root / match.group(1).strip()).resolve()
        elif git.is_dir():
            gitdir = git
        else:
            return None
        head = (gitdir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head[4:].strip()
            ref_file = gitdir / ref
            if ref_file.is_file():
                value = ref_file.read_text(encoding="utf-8", errors="replace").strip()
            else:
                common = gitdir
                commondir = gitdir / "commondir"
                if commondir.is_file():
                    common = (gitdir / commondir.read_text(encoding="utf-8").strip()).resolve()
                value = None
                packed = common / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                        if not line or line.startswith("#") or line.startswith("^"):
                            continue
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            value = parts[0]
                            break
                if value is None:
                    return None
        else:
            value = head
        return value.lower() if _HEX_RE.fullmatch(value) else None
    except OSError:
        return None


def _sha256_regular_file(path: Path, max_bytes: int) -> tuple[str | None, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None, 0
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return None, info.st_size
        if info.st_size > max_bytes:
            return None, info.st_size
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return None, total
            digest.update(chunk)
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _read_text_limited(path: Path, max_bytes: int) -> str | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return None
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _detect_license(text: str | None) -> str | None:
    if not text:
        return None
    upper = text.upper()
    if "GNU GENERAL PUBLIC LICENSE" in upper and "VERSION 3" in upper:
        return "GPL-3.0"
    if "APACHE LICENSE" in upper:
        return "Apache-2.0"
    if "MIT LICENSE" in upper or "PERMISSION IS HEREBY GRANTED, FREE OF CHARGE" in upper:
        return "MIT"
    if "CREATIVE COMMONS ATTRIBUTION 4.0" in upper:
        return "CC-BY-4.0"
    return None


def _license_files(root: Path, config: Config) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    metadata_license = "GPL-3.0"
    catalog_license = "MIT"
    metadata_text = _read_text_limited(root / "LICENSE", config.catalog_license_max_bytes)
    detected = _detect_license(metadata_text)
    if detected:
        metadata_license = detected
    elif metadata_text is None:
        warnings.append("metadata_repo_license_file_missing")

    sibling = root.parent / "awesome-public-datasets" / "LICENSE"
    catalog_text = _read_text_limited(sibling, config.catalog_license_max_bytes)
    detected = _detect_license(catalog_text)
    if detected:
        catalog_license = detected
    return catalog_license, metadata_license, warnings


def discover_readme(
    root: Path, config: Config, override: Path | None = None
) -> tuple[Path | None, str | None, list[str]]:
    """Pick the generated README (the one with Meta links) when available."""
    warnings: list[str] = []
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    if config.readme_path is not None:
        candidates.append(config.readme_path)
    candidates.extend(
        [
            root / "README.rst",
            root / "README.md",
            root.parent / "awesome-public-datasets" / "README.rst",
            root.parent / "awesome-public-datasets" / "README.md",
        ]
    )
    readable: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            info = candidate.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            warnings.append(f"readme_not_regular:{candidate.name}")
            continue
        if info.st_size > config.catalog_readme_max_bytes:
            warnings.append(f"readme_too_large:{candidate.name}")
            continue
        text = _read_text_limited(candidate, config.catalog_readme_max_bytes)
        if text is None:
            warnings.append(f"readme_unreadable:{candidate.name}")
            continue
        explicit = (
            (override is not None and resolved == override.resolve())
            or (config.readme_path is not None and resolved == config.readme_path.resolve())
        )
        if _META_RE.search(text):
            return candidate, text, warnings
        if explicit:
            warnings.append("readme_without_meta_links")
            return candidate, text, warnings
        readable.append((candidate, text))
    if readable:
        warnings.append("readme_without_meta_links")
        return readable[0][0], readable[0][1], warnings
    warnings.append("readme_unavailable")
    return None, None, warnings


def parse_readme(text: str) -> tuple[dict[str, dict[str, str]], int, list[str]]:
    """Map ``core/<category>/<file>`` paths to README OK/FIXME health flags."""
    entries: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    total = 0
    for line in text.splitlines():
        match = _META_RE.search(line)
        if not match:
            continue
        total += 1
        url = match.group(1).strip()
        if "OK_ICON" in line:
            status = "OK"
        elif "FIXME_ICON" in line:
            status = "FIXME"
        else:
            status = "missing"
        try:
            path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
        except ValueError:
            warnings.append("readme_unparseable_meta_url")
            continue
        marker = "/core/"
        if marker not in path:
            warnings.append("readme_meta_outside_core")
            continue
        relpath = re.sub(r"/+", "/", path.split(marker, 1)[1]).strip("/")
        parts = relpath.split("/")
        if not relpath or any(part in {"", ".", ".."} for part in parts):
            warnings.append("readme_meta_unsafe_path")
            continue
        if relpath in entries:
            warnings.append(f"readme_duplicate_meta:{relpath}")
            continue
        entries[relpath] = {"status": status, "url": url}
    return entries, total, warnings


def snapshot_from_directory(
    path: str | os.PathLike[str],
    config: Config,
    *,
    commit_override: str | None = None,
    mode: str = "dir",
    readme_override: Path | None = None,
) -> CatalogSnapshot:
    root = Path(os.path.expanduser(str(path))).resolve()
    core = root / "core"
    if not core.is_dir():
        raise CatalogError("catalog_snapshot_failed", f"catalog has no core/ directory: {root}")
    core_real = os.path.realpath(core)
    files: list[CatalogFile] = []

    for dirpath, dirnames, filenames in os.walk(core, topdown=True, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        # Refuse symlinked directories without following them.
        for name in list(dirnames):
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                relpath = candidate.relative_to(core).as_posix()
                category, filename = _split_relpath(relpath)
                files.append(
                    CatalogFile(relpath, category, filename, "", 0, unsafe_reason="symlink")
                )
                dirnames.remove(name)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            relpath = full.relative_to(core).as_posix()
            category, filename = _split_relpath(relpath)
            unsafe: str | None = None
            size = 0
            try:
                info = full.lstat()
            except OSError:
                unsafe = "unreadable"
                info = None
            if info is not None:
                size = info.st_size
                if stat.S_ISLNK(info.st_mode):
                    unsafe = "symlink"
                elif not stat.S_ISREG(info.st_mode):
                    unsafe = "not_regular"
                elif size > config.catalog_max_file_bytes:
                    unsafe = "oversized"
            if unsafe is None:
                real = os.path.realpath(full)
                if real != core_real and not real.startswith(core_real + os.sep):
                    unsafe = "path_escape"
            digest = ""
            if unsafe is None:
                digest, size = _sha256_regular_file(full, config.catalog_max_file_bytes)
                if digest is None:
                    unsafe = (
                        "oversized"
                        if size > config.catalog_max_file_bytes
                        else "unreadable"
                    )
            files.append(CatalogFile(relpath, category, filename, digest or "", size, unsafe))

    if len(files) > config.catalog_max_files:
        raise CatalogError(
            "catalog_snapshot_failed",
            f"catalog contains more than {config.catalog_max_files} files",
        )
    files.sort(key=lambda item: item.relpath)
    snapshot_digest = hashlib.sha256()
    for item in files:
        snapshot_digest.update(item.relpath.encode("utf-8"))
        snapshot_digest.update(b"\x00")
        snapshot_digest.update((item.sha256 or f"unsafe:{item.unsafe_reason}").encode("ascii"))
        snapshot_digest.update(b"\n")
    snapshot_sha = snapshot_digest.hexdigest()
    commit = commit_override or _read_git_commit(root) or snapshot_sha

    readme_path, readme_text, readme_warnings = discover_readme(
        root, config, readme_override
    )
    readme_entries: dict[str, dict[str, str]] = {}
    readme_total = 0
    if readme_path is not None and readme_text is not None:
        readme_entries, readme_total, parse_warnings = parse_readme(readme_text)
        readme_warnings.extend(parse_warnings)
        if readme_total == 0:
            readme_warnings.append("readme_without_meta_links")

    catalog_license, metadata_repo_license, license_warnings = _license_files(root, config)
    warnings: list[str] = list(license_warnings)
    unsafe_count = sum(1 for item in files if item.unsafe_reason)
    if unsafe_count:
        warnings.append(f"unsafe_catalog_files:{unsafe_count}")
    missing_from_readme = [item.relpath for item in files if item.relpath not in readme_entries]
    extra_in_readme = [relpath for relpath in readme_entries if relpath not in {i.relpath for i in files}]
    if readme_path is not None and missing_from_readme:
        readme_warnings.append(f"readme_missing_entries:{len(missing_from_readme)}")
    if extra_in_readme:
        readme_warnings.append(f"readme_extra_entries:{len(extra_in_readme)}")

    return CatalogSnapshot(
        repo=config.catalog_repo,
        commit=commit,
        snapshot_sha256=snapshot_sha,
        root=root,
        core_dir=core,
        mode=mode,
        files=files,
        readme_path=readme_path,
        readme_entries=readme_entries,
        readme_meta_total=readme_total,
        readme_warnings=sorted(set(readme_warnings)),
        catalog_license=catalog_license,
        metadata_repo_license=metadata_repo_license,
        warnings=sorted(set(warnings)),
    )


def _validate_member_name(name: str) -> list[str]:
    # The rules are shared with dataset archives (``archive.py``) so the two
    # extractors cannot drift apart; only the error code differs here.
    from .archive import unsafe_member_reason

    reason = unsafe_member_reason(name)
    if reason is not None:
        raise CatalogError("catalog_snapshot_failed", reason)
    return [part for part in name.split("/") if part not in {"", "."}]


def _safe_join(root: Path, parts: list[str]) -> Path:
    target = root.joinpath(*parts)
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    if target_real != root_real and not target_real.startswith(root_real + os.sep):
        raise CatalogError("catalog_snapshot_failed", "tar member escapes the snapshot root")
    return target


def extract_tarball(
    data: bytes,
    destination: Path,
    *,
    expected_roots: set[str] | None,
    config: Config,
) -> Path:
    """Expand a catalog tarball member-by-member; never call ``extractall``."""
    destination = Path(destination)
    if destination.exists():
        raise CatalogError("catalog_snapshot_failed", "snapshot destination already exists")
    destination.mkdir(parents=True, mode=0o700)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > config.catalog_max_files:
                raise CatalogError(
                    "catalog_snapshot_failed",
                    f"tarball contains more than {config.catalog_max_files} members",
                )
            validated: list[tuple[tarfile.TarInfo, list[str]]] = []
            roots: set[str] = set()
            seen: set[str] = set()
            total_uncompressed = 0
            for member in members:
                parts = _validate_member_name(member.name)
                roots.add(parts[0])
                validated.append((member, parts))
            if len(roots) != 1:
                raise CatalogError("catalog_snapshot_failed", "tarball must have exactly one root directory")
            root_name = next(iter(roots))
            if expected_roots is not None and root_name not in expected_roots:
                raise CatalogError(
                    "catalog_snapshot_failed",
                    f"tarball root directory does not match the pinned commit: {root_name}",
                )
            for member, parts in validated:
                rel_parts = parts[1:]
                if not rel_parts:
                    if member.isdir():
                        continue
                    raise CatalogError("catalog_snapshot_failed", "tarball root is not a directory")
                relpath = "/".join(rel_parts)
                if relpath in seen:
                    raise CatalogError("catalog_snapshot_failed", f"duplicate tar member: {relpath}")
                seen.add(relpath)
                if member.isdir():
                    directory = _safe_join(destination, rel_parts)
                    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                    try:
                        directory.chmod(0o700)
                    except OSError:
                        pass
                    continue
                if not member.isfile():
                    raise CatalogError(
                        "catalog_snapshot_failed",
                        f"tar member is a link or device: {member.name}",
                    )
                if member.size > config.catalog_max_file_bytes:
                    raise CatalogError("catalog_snapshot_failed", f"tar member exceeds the size cap: {relpath}")
                total_uncompressed += member.size
                if total_uncompressed > config.catalog_tarball_max_bytes:
                    raise CatalogError("catalog_snapshot_failed", "tarball expands beyond the configured cap")
                target = _safe_join(destination, rel_parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    source = archive.extractfile(member)
                    if source is None:
                        raise CatalogError("catalog_snapshot_failed", f"tar member has no data: {relpath}")
                    remaining = member.size
                    with os.fdopen(descriptor, "wb") as output:
                        descriptor = -1
                        while remaining > 0:
                            chunk = source.read(min(64 * 1024, remaining))
                            if not chunk:
                                raise CatalogError(
                                    "catalog_snapshot_failed",
                                    f"tar member is truncated: {relpath}",
                                )
                            output.write(chunk)
                            remaining -= len(chunk)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
        return destination
    except CatalogError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (tarfile.TarError, OSError, ValueError, EOFError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise CatalogError("catalog_snapshot_failed", f"tarball could not be expanded: {type(exc).__name__}") from exc


def _resolve_remote_commit(config: Config) -> tuple[str | None, list[str]]:
    from .netguard import NetguardError, Transport

    url = f"https://api.github.com/repos/{config.catalog_repo}/commits/{config.catalog_ref}"
    try:
        transport = Transport(config)
        body, _response = transport.fetch_bytes(url, max_bytes=1024 * 1024)
        payload = json.loads(body.decode("utf-8"))
        sha = payload.get("sha") if isinstance(payload, dict) else None
        if isinstance(sha, str) and _HEX_RE.fullmatch(sha):
            return sha.lower(), []
        return None, ["commit_resolution_failed:invalid_response"]
    except (NetguardError, ValueError, UnicodeDecodeError) as exc:
        return None, [f"commit_resolution_failed:{type(exc).__name__}"]


def _fetch_remote_readme(
    config: Config, destination: Path
) -> tuple[Path | None, list[str]]:
    """Fetch the generated README used for OK/FIXME reconciliation."""
    from .netguard import NetguardError, Transport

    path = destination.parent / f"{destination.name}.README.rst"
    try:
        info = path.lstat()
    except OSError:
        info = None
    if info is not None and stat.S_ISREG(info.st_mode) and info.st_size <= config.catalog_readme_max_bytes:
        return path, []
    url = (
        f"https://raw.githubusercontent.com/{config.readme_repo}/"
        f"{config.readme_ref}/README.rst"
    )
    try:
        body, _response = Transport(config).fetch_bytes(
            url, max_bytes=config.catalog_readme_max_bytes
        )
    except NetguardError as exc:
        return None, [f"readme_fetch_failed:{exc.code}"]
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(body)
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
    return path, []


def snapshot_from_remote(config: Config) -> CatalogSnapshot:
    from .netguard import NetguardError, Transport

    resolved, warnings = _resolve_remote_commit(config)
    commit = resolved or config.catalog_ref
    repo_name = config.catalog_repo.rsplit("/", 1)[-1]
    tarball_url = f"https://codeload.github.com/{config.catalog_repo}/tar.gz/{commit}"
    try:
        transport = Transport(config)
        body, _response = transport.fetch_bytes(
            tarball_url, max_bytes=config.catalog_tarball_max_bytes
        )
    except NetguardError as exc:
        raise CatalogError(
            "catalog_snapshot_failed", f"catalog tarball could not be fetched: {exc.code}"
        ) from exc

    destination = config.data_root / "catalog" / f"{config.catalog_repo.replace('/', '_')}@{commit}"
    expected_roots = {f"{repo_name}-{commit}", f"{repo_name}-{config.catalog_ref}"}
    if not (destination / "core").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        extract_tarball(body, temporary, expected_roots=expected_roots, config=config)
        if destination.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        else:
            os.replace(temporary, destination)

    readme_override, readme_warnings = _fetch_remote_readme(config, destination)
    snapshot = snapshot_from_directory(
        destination,
        config,
        commit_override=commit,
        mode="tarball",
        readme_override=readme_override,
    )
    snapshot.warnings = sorted(set(snapshot.warnings) | set(warnings))
    snapshot.readme_warnings = sorted(
        set(snapshot.readme_warnings) | set(readme_warnings)
    )
    return snapshot


def load_catalog(config: Config) -> CatalogSnapshot:
    if config.catalog_dir is not None:
        return snapshot_from_directory(config.catalog_dir, config, mode="dir")
    return snapshot_from_remote(config)
