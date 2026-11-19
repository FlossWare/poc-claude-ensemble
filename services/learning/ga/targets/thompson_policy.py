"""Target A adapter: the Thompson selection policy.

This module is intentionally free of every network import.  Promotion is a
file write (atomic replace) and rollback is the journal's stored prior bytes;
there is no endpoint URL and nothing to POST.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from learning.ga import corpus as corpus_mod
from learning.ga import journal as journal_mod
from learning.ga.engine import EvalResult, SearchConfig, replay_review, replay_turn
from learning.ga.schema import gene_specs, validate_params
from shared import thompson_policy as policy_mod
from shared import runtime_config

TARGET = "thompson_policy"
PROMOTABLE = True
TASK_DEFAULT = "default"
TASK_CODE_REVIEW = "code-review"
ESTIMATORS = {
    "turn": "logged_bandit_direct_method",
    "review": "full_information_replay",
}


@dataclass
class AppliedChange:
    change_id: str
    path: str
    previous_sha256: str
    new_sha256: str
    previous_file: Optional[bytes]

    def journal_fields(self) -> Dict[str, Any]:
        fields = journal_mod.encode_previous_file(self.previous_file)
        fields.update(
            {
                "previous_file_sha256": self.previous_sha256,
                "new_file_sha256": self.new_sha256,
            }
        )
        return fields


def param_specs() -> Dict[str, Dict[str, float]]:
    return gene_specs()


def load_corpora() -> Dict[str, corpus_mod.Corpus]:
    return corpus_mod.build_corpora()


def baseline_policies() -> Dict[str, Dict[str, float]]:
    """Policies currently live in the policy file (defaults when absent/invalid)."""
    live = policy_mod.load_policy_file()
    return {
        TASK_DEFAULT: policy_mod.policy_for_task(live, None),
        TASK_CODE_REVIEW: policy_mod.policy_for_task(live, TASK_CODE_REVIEW),
    }


def evaluate_policies(
    policies: Mapping[str, Mapping[str, Any]],
    corpora: Mapping[str, corpus_mod.Corpus],
    window: str,
    seed: int,
    cfg: SearchConfig | None = None,
    trace: bool = False,
) -> Dict[str, EvalResult]:
    """Replay both corpora for the given policy set."""
    cfg = cfg or SearchConfig()
    default_params = policy_mod.policy_for_task(policies, None)
    code_review_params = policy_mod.policy_for_task(policies, TASK_CODE_REVIEW)
    return {
        "turn": replay_turn(default_params, corpora["turn"], window, seed, cfg, trace=trace),
        "review": replay_review(code_review_params, corpora["review"], window, seed, cfg, trace=trace),
    }


def search_params(policies: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """The policy the search evolves (the ``default`` entry)."""
    return policy_mod.policy_for_task(policies, None)


# ---------------------------------------------------------------------------
# Code identity / deployment skew
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3].parent


def repo_service_path() -> Path:
    return Path(__file__).resolve().parents[3] / "thompson-service" / "thompson_service.py"


def deployed_service_path() -> Path:
    configured = os.environ.get("POC_CLAUDE_ENSEMBLE_DEPLOYED_SERVICES_DIR")
    base = Path(configured).expanduser() if configured else Path.home() / ".FlossWare" / "poc-claude-ensemble" / "services"
    return base / "thompson-service" / "thompson_service.py"


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def code_hash() -> str:
    """Hash of the evaluated module set: policy, adapter, and service selector."""
    digest = hashlib.sha256()
    for path in (
        Path(policy_mod.__file__),
        Path(__file__),
        repo_service_path(),
    ):
        digest.update(path.name.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def deployed_status() -> Dict[str, Any]:
    """Compare the repo Thompson service with the deployed copy, if one exists."""
    repo_sha = _sha256_file(repo_service_path())
    deployed_path = deployed_service_path()
    deployed_sha = _sha256_file(deployed_path)
    matches: Optional[bool]
    if deployed_sha is None:
        matches = None
    else:
        matches = deployed_sha == repo_sha
    return {
        "repo_sha256": repo_sha,
        "deployed_path": str(deployed_path),
        "deployed_sha256": deployed_sha,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Apply / rollback
# ---------------------------------------------------------------------------


def _allowed_file_target(path: Path) -> None:
    parent = journal_mod.ensure_private_dir(path.parent)
    resolved = parent.resolve(strict=True)
    roots = corpus_mod.allowed_roots()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise journal_mod.JournalError(f"policy file directory {resolved} is outside the allowlist")


def _read_current(path: Path) -> Optional[bytes]:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise journal_mod.JournalError(f"{path} is not a regular file")
    if info.st_uid != os.getuid():
        raise journal_mod.JournalError(f"{path} is not owned by the current user")
    if info.st_size > policy_mod.MAX_POLICY_BYTES:
        raise journal_mod.JournalError(f"{path} is larger than {policy_mod.MAX_POLICY_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(path, flags)
    with os.fdopen(handle, "rb") as stream:
        data = stream.read(policy_mod.MAX_POLICY_BYTES + 1)
    if len(data) > policy_mod.MAX_POLICY_BYTES:
        raise journal_mod.JournalError(f"{path} is larger than {policy_mod.MAX_POLICY_BYTES} bytes")
    return data


def prepare_apply(policies: Mapping[str, Mapping[str, Any]], provenance: Mapping[str, Any]) -> Tuple[Path, bytes, Optional[bytes]]:
    """Validate and render the new policy file without writing it."""
    resolved_policies = {
        name: validate_params(values, require_all=True, source=f"policy {name!r}")
        for name, values in policies.items()
    }
    document = policy_mod.policy_document(resolved_policies, provenance)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = runtime_config.thompson_policy_file()
    _allowed_file_target(path)
    previous = _read_current(path)
    return path, encoded, previous


def apply_policies(
    policies: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> AppliedChange:
    """Atomically write the policy file and verify it reads back correctly."""
    path, encoded, previous = prepare_apply(policies, provenance)
    journal_mod.atomic_write(path, encoded, 0o600)
    policy_mod.clear_cache()

    reloaded = policy_mod.load_policy_file(path)
    for name, values in policies.items():
        expected = validate_params(values, require_all=True, source=f"policy {name!r}")
        if reloaded.get(name) != expected:
            raise journal_mod.JournalError(
                f"policy file did not read back as written for {name!r}"
            )

    return AppliedChange(
        change_id=str(provenance.get("change_id", "")),
        path=str(path),
        previous_sha256=hashlib.sha256(previous).hexdigest() if previous is not None else hashlib.sha256(b"").hexdigest(),
        new_sha256=hashlib.sha256(encoded).hexdigest(),
        previous_file=previous,
    )


def rollback_change(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Restore the policy file recorded in an ``apply`` journal entry."""
    path = runtime_config.thompson_policy_file()
    _allowed_file_target(path)
    previous = journal_mod.read_previous_file(dict(entry))

    if previous is None:
        if entry.get("previous_file_existed"):
            raise journal_mod.JournalError("journal entry claims a previous file but has no content")
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            info = None
        if info is not None:
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise journal_mod.JournalError(f"{path} is not a regular file")
            os.unlink(path)
        policy_mod.clear_cache()
        return {"path": str(path), "removed": True, "sha256": hashlib.sha256(b"").hexdigest()}

    expected = entry.get("previous_file_sha256")
    actual = hashlib.sha256(previous).hexdigest()
    if expected and expected != actual:
        raise journal_mod.JournalError("stored previous policy content does not match its hash")
    journal_mod.atomic_write(path, previous, 0o600)
    policy_mod.clear_cache()
    restored = _read_current(path)
    if restored != previous:
        raise journal_mod.JournalError("rollback did not restore the previous policy file byte for byte")
    return {"path": str(path), "removed": False, "sha256": actual}


__all__ = [
    "AppliedChange",
    "ESTIMATORS",
    "PROMOTABLE",
    "TARGET",
    "TASK_CODE_REVIEW",
    "TASK_DEFAULT",
    "apply_policies",
    "baseline_policies",
    "code_hash",
    "deployed_service_path",
    "deployed_status",
    "evaluate_policies",
    "load_corpora",
    "param_specs",
    "prepare_apply",
    "repo_service_path",
    "rollback_change",
    "search_params",
]
