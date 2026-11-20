"""Registry of external priors and the code that applies them to Thompson.

The registry is the durable record of *why* a brand-new arm starts with a
non-neutral prior.  It is deliberately separate from the live Thompson state:

* ``priors.json`` holds the external evidence (Arena ELO, benchmark proxy,
  source snapshot, match quality) keyed by arm;
* applying a prior initialises the arm's Beta pseudo-counts in Thompson so that
  ``successes + prior_strength == alpha`` and ``failures + prior_strength ==
  beta``.  ``calls`` stays 0, so the arm is still "untested" for cost and
  capability filters;
* real outcomes are added on top, and a later registration never overwrites an
  arm that already has observed calls.

Two application paths exist:

* ``seed_prior`` on the live Thompson service (authoritative; the daemon owns
  its state);
* a direct, locked, atomic edit of the state file (fallback when the service is
  unreachable or predates the ``seed_prior`` action).  The running daemon caches
  state in memory, so the fallback is applied on its next start -- the caller
  reports which path was used rather than pretending it took effect live.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from learning.external import net, paths

SCHEMA_VERSION = 1
DEFAULT_TASK_TYPE = "default"


class RegistryError(RuntimeError):
    """The prior registry or Thompson state could not be read/written."""


@dataclass
class PriorRecord:
    arm: str
    model_id: str
    alpha: float
    beta: float
    source: str
    match_type: str = "none"
    confidence: float = 1.0
    snapshot: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""
    applied_at: Optional[str] = None
    applied_via: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm.strip():
            raise ValueError("arm must be a non-empty string")
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        if not (self.alpha > 0.0 and self.beta > 0.0):
            raise ValueError("alpha and beta must be positive")
        if not self.registered_at:
            self.registered_at = net.utc_now()

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "model_id": self.model_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "source": self.source,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "snapshot": self.snapshot,
            "detail": self.detail,
            "registered_at": self.registered_at,
            "applied_at": self.applied_at,
            "applied_via": self.applied_via,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriorRecord":
        return cls(
            arm=str(payload["arm"]),
            model_id=str(payload.get("model_id") or payload["arm"]),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            source=str(payload.get("source") or "unknown"),
            match_type=str(payload.get("match_type") or "none"),
            confidence=float(payload.get("confidence", 1.0)),
            snapshot=str(payload.get("snapshot") or ""),
            detail=dict(payload.get("detail") or {}),
            registered_at=str(payload.get("registered_at") or ""),
            applied_at=payload.get("applied_at"),
            applied_via=payload.get("applied_via"),
        )


# ---------------------------------------------------------------------------
# Registry file
# ---------------------------------------------------------------------------


def registry_path(path: Optional[Path] = None) -> Path:
    return Path(path).expanduser() if path is not None else paths.priors_path()


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    paths.ensure_dir(lock_path.parent)
    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    target = registry_path(path)
    document = net.read_cache(target)
    if not document:
        return {"schema_version": SCHEMA_VERSION, "updated_at": net.utc_now(), "priors": {}}
    if not isinstance(document.get("priors"), dict):
        document["priors"] = {}
    document.setdefault("schema_version", SCHEMA_VERSION)
    return document


def _save_registry(document: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    target = registry_path(path)
    payload = dict(document)
    payload["updated_at"] = net.utc_now()
    return net.write_cache(target, payload)


def list_priors(path: Optional[Path] = None) -> List[PriorRecord]:
    document = load_registry(path)
    records = []
    for payload in (document.get("priors") or {}).values():
        try:
            records.append(PriorRecord.from_dict(payload))
        except (KeyError, TypeError, ValueError):
            continue
    records.sort(key=lambda record: record.registered_at, reverse=True)
    return records


def register_prior(record: PriorRecord, path: Optional[Path] = None) -> PriorRecord:
    """Upsert *record* under an exclusive lock and return the stored copy."""
    target = registry_path(path)
    with _file_lock(target):
        document = load_registry(target)
        priors = document.setdefault("priors", {})
        existing = priors.get(record.arm)
        if isinstance(existing, Mapping) and existing.get("applied_at"):
            # Keep the original application timestamp/source, but refresh the
            # evidence so a later refresh is visible.
            record.applied_at = existing.get("applied_at")
            record.applied_via = existing.get("applied_via")
        priors[record.arm] = record.as_dict()
        _save_registry(document, target)
    return record


def mark_applied(
    arm: str,
    *,
    via: str,
    path: Optional[Path] = None,
) -> Optional[PriorRecord]:
    target = registry_path(path)
    with _file_lock(target):
        document = load_registry(target)
        priors = document.setdefault("priors", {})
        payload = priors.get(arm)
        if not isinstance(payload, Mapping):
            return None
        payload = dict(payload)
        payload["applied_at"] = net.utc_now()
        payload["applied_via"] = via
        priors[arm] = payload
        _save_registry(document, target)
        return PriorRecord.from_dict(payload)


def get_prior(arm: str, path: Optional[Path] = None) -> Optional[PriorRecord]:
    payload = (load_registry(path).get("priors") or {}).get(arm)
    if not isinstance(payload, Mapping):
        return None
    try:
        return PriorRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Thompson helpers
# ---------------------------------------------------------------------------


def thompson_state_path() -> Path:
    try:
        from shared.runtime_config import thompson_state_file

        return thompson_state_file()
    except Exception:  # pragma: no cover - services/shared unavailable
        configured = os.environ.get("POC_CLAUDE_ENSEMBLE_THOMPSON_STATE_FILE")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / ".FlossWare" / "claude-ensemble-state" / "data" / "learning" / "thompson-sampling-state.json"


def policy_prior_strength(task_type: str = DEFAULT_TASK_TYPE) -> float:
    """The prior strength the live service will add to these pseudo-counts."""
    try:
        from shared.thompson_policy import load_policy_file, policy_for_task

        return float(policy_for_task(load_policy_file(), task_type)["prior_strength"])
    except Exception:
        return 1.0


def _pseudo_counts(alpha: float, beta: float, prior_strength: float) -> Tuple[float, float]:
    return max(0.0, alpha - prior_strength), max(0.0, beta - prior_strength)


def apply_priors_to_state(
    records: Optional[Iterable[PriorRecord]] = None,
    *,
    state_path: Optional[Path] = None,
    prior_strength: Optional[float] = None,
    task_type: str = DEFAULT_TASK_TYPE,
) -> Dict[str, Any]:
    """Initialise missing/untested Thompson arms from registered priors.

    Returns ``{"applied": [...], "skipped": [...], "state_path": ...}``.  An
    arm with observed calls is never modified: real evidence wins.
    """
    target = Path(state_path).expanduser() if state_path else thompson_state_path()
    selected = list(records) if records is not None else list_priors()
    strength = float(prior_strength) if prior_strength is not None else policy_prior_strength(task_type)
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    with _file_lock(target):
        state = net.read_cache(target) or {"last_updated": net.utc_now(), "models": {}}
        models = state.setdefault("models", {})
        if not isinstance(models, dict):
            raise RegistryError(f"{target} has a non-object models field")
        for record in selected:
            current = models.get(record.arm)
            if isinstance(current, Mapping) and int(current.get("calls") or 0) > 0:
                skipped.append({"arm": record.arm, "reason": "observed_calls",
                                "calls": int(current.get("calls") or 0)})
                continue
            successes, failures = _pseudo_counts(record.alpha, record.beta, strength)
            models[record.arm] = {
                "model_name": record.arm,
                "successes": successes,
                "failures": failures,
                "total_cost": 0.0,
                "total_tokens": 0,
                "calls": 0,
                "last_updated": net.utc_now(),
                "prior": {
                    "alpha": record.alpha,
                    "beta": record.beta,
                    "prior_strength": strength,
                    "source": record.source,
                    "match_type": record.match_type,
                    "snapshot": record.snapshot,
                    "model_id": record.model_id,
                    "registered_at": record.registered_at,
                },
            }
            applied.append({"arm": record.arm, "alpha": record.alpha,
                            "beta": record.beta, "successes": successes,
                            "failures": failures})
        state["last_updated"] = net.utc_now()
        net.write_cache(target, state)

    for entry in applied:
        mark_applied(entry["arm"], via="state_file")
    return {"applied": applied, "skipped": skipped, "state_path": str(target),
            "prior_strength": strength}


def apply_priors_via_client(
    client: Any,
    records: Optional[Iterable[PriorRecord]] = None,
    *,
    task_type: str = DEFAULT_TASK_TYPE,
) -> Dict[str, Any]:
    """Ask the live service to seed each prior; never raise on an old daemon."""
    selected = list(records) if records is not None else list_priors()
    strength = policy_prior_strength(task_type)
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for record in selected:
        seed = getattr(client, "seed_prior", None)
        if seed is None:
            errors.append({"arm": record.arm, "error": "client has no seed_prior"})
            continue
        try:
            result = seed(
                record.arm,
                record.alpha,
                record.beta,
                prior_strength=strength,
                source=record.source,
                detail=record.detail,
            )
        except Exception as exc:
            errors.append({"arm": record.arm, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result:
            mark_applied(record.arm, via="service")
            applied.append({"arm": record.arm, "alpha": record.alpha, "beta": record.beta})
        else:
            skipped.append({"arm": record.arm, "reason": "service declined (existing calls or unsupported)"})
    return {"applied": applied, "skipped": skipped, "errors": errors,
            "prior_strength": strength}


def unapplied(records: Optional[Iterable[PriorRecord]] = None,
              state: Optional[Mapping[str, Any]] = None) -> List[PriorRecord]:
    """Registered priors whose arm is absent or untested in a state snapshot."""
    selected = list(records) if records is not None else list_priors()
    models = (state or {}).get("models") if isinstance(state, Mapping) else None
    if not isinstance(models, Mapping):
        return [record for record in selected if not record.applied_at]
    pending = []
    for record in selected:
        current = models.get(record.arm)
        if isinstance(current, Mapping) and int(current.get("calls") or 0) > 0:
            continue
        if isinstance(current, Mapping) and isinstance(current.get("prior"), Mapping):
            continue
        pending.append(record)
    return pending


__all__ = [
    "DEFAULT_TASK_TYPE",
    "PriorRecord",
    "RegistryError",
    "apply_priors_to_state",
    "apply_priors_via_client",
    "get_prior",
    "list_priors",
    "load_registry",
    "mark_applied",
    "policy_prior_strength",
    "register_prior",
    "registry_path",
    "thompson_state_path",
    "unapplied",
]
