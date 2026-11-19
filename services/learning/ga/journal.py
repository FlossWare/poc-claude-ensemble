"""Flocked, append-only change journal -- the source of truth for the GA.

The journal is the only durable record of what the tuner changed.  It also
carries the exact previous policy-file bytes (base64, capped) so ``rollback``
can restore the prior file byte for byte, even if a human wrote it by hand.

Every entry is restricted to an explicit field allowlist.  In particular no
ledger ``text``, ``subject``, or ``query`` value can reach the journal, and
error entries store only the exception class and message -- never a traceback
containing file contents.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared import runtime_config

SCHEMA_VERSION = 1
MAX_ENTRY_BYTES = 96 * 1024
MAX_PREVIOUS_FILE_BYTES = 64 * 1024
KEEP_RUN_DIRS = 30

JOURNAL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "change_id",
        "target",
        "run_id",
        "seed",
        "corpus",
        "code_hash",
        "estimator",
        "evidence",
        "baseline_fitness",
        "candidate_fitness",
        "delta",
        "ci95",
        "old_params",
        "new_params",
        "previous_file_sha256",
        "new_file_sha256",
        "previous_file_existed",
        "previous_file_b64",
        "applied_by",
        "approval",
        "reason",
        "recorded_at",
        "error_type",
        "error_message",
    }
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,96}$")


class JournalError(Exception):
    """A journal entry or path is not acceptable."""


def ga_root() -> Path:
    return runtime_config.ga_dir()


def journal_path() -> Path:
    return ga_root() / "journal.jsonl"


def locks_dir() -> Path:
    return ga_root() / "locks"


def runs_dir() -> Path:
    return ga_root() / "runs"


def run_dir(run_id: str) -> Path:
    if not _RUN_ID_RE.match(run_id):
        raise JournalError(f"invalid run id {run_id!r}")
    return runs_dir() / run_id


def ensure_private_dir(path: Path, mode: int = 0o700) -> Path:
    """Create a directory with 0700 perms and refuse symlinked/non-owned paths."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise JournalError(f"{path} is not a real directory")
    if info.st_uid != os.getuid():
        raise JournalError(f"{path} is not owned by the current user")
    os.chmod(path, mode)
    return path


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write ``data`` atomically: tmp file, fsync, rename, fsync directory."""
    path = Path(path)
    ensure_private_dir(path.parent)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(tmp, flags, mode)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def change_id(run_id: str, candidate: str, target: str) -> str:
    payload = f"{run_id}\x1f{candidate}\x1f{target}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def project_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the field allowlist and encode the entry as one JSONL line."""
    unknown = sorted(set(entry) - JOURNAL_FIELDS)
    if unknown:
        raise JournalError(f"journal entry has non-allowlisted field(s): {', '.join(unknown)}")
    if "kind" not in entry:
        raise JournalError("journal entry is missing 'kind'")
    document = dict(entry)
    document.setdefault("schema_version", SCHEMA_VERSION)
    document.setdefault("recorded_at", _now())
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_ENTRY_BYTES:
        raise JournalError(f"journal entry is larger than {MAX_ENTRY_BYTES} bytes")
    return document


def append(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append one entry under an exclusive ``flock``.  Returns the stored entry."""
    document = project_entry(entry)
    line = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )
    ensure_private_dir(ga_root())
    ensure_private_dir(locks_dir())
    lock_path = locks_dir() / "journal.lock"
    lock_fd = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        target = journal_path()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        handle = os.open(target, flags, 0o600)
        with os.fdopen(handle, "wb") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return document


def read_entries() -> List[Dict[str, Any]]:
    path = journal_path()
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def last_apply() -> Optional[Dict[str, Any]]:
    for entry in reversed(read_entries()):
        if entry.get("kind") == "apply":
            return entry
    return None


def find_change(change: str) -> Optional[Dict[str, Any]]:
    for entry in read_entries():
        if entry.get("change_id") == change:
            return entry
    return None


def already_applied(change: str) -> bool:
    for entry in read_entries():
        if entry.get("kind") == "apply" and entry.get("change_id") == change:
            return True
    return False


def rolled_back(change: str) -> bool:
    for entry in read_entries():
        if entry.get("kind") == "rollback" and entry.get("change_id") == change:
            return True
    return False


def cooldown_status(
    *,
    total_outcomes: int,
    now: Optional[float] = None,
    cooldown_days: int = 7,
    min_new_outcomes: int = 100,
) -> Tuple[bool, str]:
    """Promotion is blocked until both the time and new-evidence floors are met."""
    entry = last_apply()
    if entry is None:
        return True, "no previous promotion"
    recorded = entry.get("recorded_at")
    try:
        parsed = datetime.strptime(str(recorded), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        elapsed_days = ((now if now is not None else time.time()) - parsed.timestamp()) / 86400.0
    except (TypeError, ValueError):
        return False, "previous promotion has an unreadable timestamp"
    previous = 0
    try:
        previous = int(entry.get("corpus", {}).get("turn", {}).get("accepted", 0))
    except (TypeError, ValueError):
        previous = 0
    new_outcomes = max(0, int(total_outcomes) - previous)
    ok = elapsed_days >= cooldown_days and new_outcomes >= min_new_outcomes
    detail = (
        f"{elapsed_days:.2f} days and {new_outcomes} new outcomes since "
        f"{entry.get('change_id')} (need >= {cooldown_days} days and >= {min_new_outcomes})"
    )
    return ok, detail


def read_previous_file(entry: Dict[str, Any]) -> Optional[bytes]:
    encoded = entry.get("previous_file_b64")
    if encoded is None:
        return None
    try:
        return base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise JournalError(f"previous policy content is corrupt: {exc}") from exc


def encode_previous_file(data: Optional[bytes]) -> Dict[str, Any]:
    if data is None:
        return {"previous_file_existed": False, "previous_file_b64": None}
    if len(data) > MAX_PREVIOUS_FILE_BYTES:
        raise JournalError("previous policy file is too large to journal for byte-for-byte rollback")
    return {
        "previous_file_existed": True,
        "previous_file_b64": base64.b64encode(data).decode("ascii"),
    }


def prune_runs(keep: int = KEEP_RUN_DIRS) -> List[str]:
    """Delete all but the newest ``keep`` run directories.  Returns removed names."""
    root = runs_dir()
    if not root.is_dir():
        return []
    names = sorted((entry.name for entry in root.iterdir() if entry.is_dir()), reverse=True)
    removed: List[str] = []
    for name in names[keep:]:
        target = root / name
        for child in sorted(target.rglob("*"), reverse=True):
            try:
                child.unlink() if child.is_file() or child.is_symlink() else child.rmdir()
            except OSError:
                pass
        try:
            target.rmdir()
            removed.append(name)
        except OSError:
            pass
    return removed


def error_entry(stage: str, exc: BaseException, **extra: Any) -> Dict[str, Any]:
    """Journal row for a failure: exception class + message only, no traceback."""
    entry = {
        "kind": "error",
        "reason": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000],
    }
    entry.update(extra)
    return entry


__all__ = [
    "JOURNAL_FIELDS",
    "JournalError",
    "already_applied",
    "append",
    "atomic_write",
    "change_id",
    "cooldown_status",
    "encode_previous_file",
    "ensure_private_dir",
    "error_entry",
    "find_change",
    "ga_root",
    "journal_path",
    "last_apply",
    "locks_dir",
    "prune_runs",
    "project_entry",
    "read_entries",
    "read_previous_file",
    "rolled_back",
    "run_dir",
    "runs_dir",
]
