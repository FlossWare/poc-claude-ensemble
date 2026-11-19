"""Generalized Thompson meta-learner: one posterior per decision domain.

The live Thompson service and the GA already tune *model* selection.  Every
other discrete decision in the ensemble -- crew size, review depth, ingest
batch size, RAG scope, and the instrumentation domains that will feed future
targets -- is the same kind of bandit problem.  :class:`MetaLearner` gives each
one a Thompson posterior without another service:

* :meth:`MetaLearner.select` draws an arm for a domain, conditioned on task
  context;
* :meth:`MetaLearner.record` feeds an outcome back (graded rewards allowed);
* :meth:`MetaLearner.posterior` exposes the current Beta parameters.

Sampling goes through :meth:`shared.thompson_client.ThompsonClient.select_arm`,
which runs the *same* pure function the live Thompson service runs
(:func:`shared.thompson_policy.select`) with the same GA-tuned policy file.  A
policy promoted by GA Target A therefore changes every domain, not just model
routing.

State
-----
One JSON file per domain under ``~/.FlossWare/ledger/meta/<domain>.json``
(override with ``POC_CLAUDE_ENSEMBLE_META_LEDGER_DIR``), mode 0600, written by
atomic replace under an flock.  The file holds:

* ``arms`` -- aggregate Beta sufficient statistics per arm
* ``outcomes`` -- the append-only outcome log (bounded detail retention; the
  aggregates always cover every outcome ever recorded)
* ``outcome_ids`` -- every id ever seen, so re-running a backfill is idempotent

Recording is idempotent: an outcome carrying an ``outcome_id`` (or ``event_id``
in its metadata) that has already been seen is skipped, not double-counted.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from learning.meta.domains import domain_names, get_domain
from learning.meta.domains.base import Domain, DomainError

try:  # services/ on sys.path (services, CLI, tests)
    from shared.thompson_client import ThompsonClient
    from shared.thompson_policy import (
        Arm,
        default_params,
        load_policy_file,
        policy_for_task,
    )
except ImportError:  # tools/ insert services/shared on sys.path
    from thompson_client import ThompsonClient  # type: ignore[no-redef]
    from thompson_policy import (  # type: ignore[no-redef]
        Arm,
        default_params,
        load_policy_file,
        policy_for_task,
    )

LOG = logging.getLogger("learning.meta")

SCHEMA_VERSION = 1
META_DIR_ENV = "POC_CLAUDE_ENSEMBLE_META_LEDGER_DIR"
DEFAULT_META_DIR = "~/.FlossWare/ledger/meta"

#: A context bucket needs this many observations before it replaces the global
#: posterior for a draw.  Below it, context is recorded but the global stats are
#: used, so a new task type does not start from a one-sample posterior.
MIN_CONTEXT_OBSERVATIONS = 3

#: Raw outcome rows retained per domain.  Aggregates and ids are unaffected.
MAX_OUTCOME_LOG = 5000

#: Metadata larger than this is refused (it is caller data, not telemetry).
MAX_METADATA_BYTES = 16 * 1024


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _meta_dir(configured: Optional[os.PathLike | str] = None) -> Path:
    if configured is not None:
        return Path(configured).expanduser()
    return Path(os.environ.get(META_DIR_ENV, DEFAULT_META_DIR)).expanduser()


def _coerce_reward(outcome: Any) -> float:
    """A bool is a binary outcome; a number in [0, 1] is a graded reward."""
    if isinstance(outcome, bool):
        return 1.0 if outcome else 0.0
    if isinstance(outcome, (int, float)):
        reward = float(outcome)
        if math.isfinite(reward) and 0.0 <= reward <= 1.0:
            return reward
    raise ValueError(f"outcome must be a bool or a number in [0, 1], got {outcome!r}")


def _coerce_cost(cost: Any) -> float:
    if isinstance(cost, bool):
        raise ValueError(f"cost must be a non-negative number, got {cost!r}")
    if not isinstance(cost, (int, float)):
        raise ValueError(f"cost must be a non-negative number, got {cost!r}")
    value = float(cost)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"cost must be a finite non-negative number, got {cost!r}")
    return value


def _scalar_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep only JSON-safe scalar context keys; the rest stays in metadata."""
    if not context:
        return {}
    if not isinstance(context, Mapping):
        raise ValueError(f"context must be a mapping, got {type(context).__name__}")
    clean: Dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[key] = value
    return clean


def _empty_arm_stats() -> Dict[str, Any]:
    return {"successes": 0.0, "failures": 0.0, "calls": 0, "total_cost": 0.0, "total_reward": 0.0}


class MetaLearner:
    """Thompson posteriors for arbitrary discrete decisions."""

    def __init__(
        self,
        meta_dir: Optional[os.PathLike | str] = None,
        *,
        rng: Any = None,
        client: Any = None,
        min_context_observations: int = MIN_CONTEXT_OBSERVATIONS,
    ) -> None:
        self.meta_dir = _meta_dir(meta_dir)
        self.rng = rng if rng is not None else random.Random()
        # Local-only sampling: an unreachable Thompson service must not stop a
        # domain draw, so the circuit breaker is not used for arm sampling.
        self.client = client if client is not None else ThompsonClient(enable_circuit_breaker=False)
        self.min_context_observations = max(0, int(min_context_observations))

    # -- domain lookup ----------------------------------------------------

    @staticmethod
    def domain(name: str) -> type[Domain]:
        return get_domain(name)

    # -- paths ------------------------------------------------------------

    def state_path(self, domain: str) -> Path:
        return self.meta_dir / f"{self.domain(domain).name}.json"

    # -- state I/O --------------------------------------------------------

    def _load(self, spec: type[Domain]) -> Dict[str, Any]:
        """Load a domain state.  Corruption is quarantined, never fatal."""
        path = self.state_path(spec.name)
        state = {
            "schema_version": SCHEMA_VERSION,
            "domain": spec.name,
            "arms": {},
            "outcome_ids": [],
            "outcomes": [],
            "updated_at": None,
        }
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return state
        except OSError as exc:
            LOG.warning("meta state %s unreadable (%s); starting empty", path, exc)
            return state
        try:
            loaded = json.loads(raw)
        except ValueError as exc:
            self._quarantine(path, exc)
            return state
        if not isinstance(loaded, dict) or loaded.get("schema_version") != SCHEMA_VERSION:
            self._quarantine(path, ValueError("unsupported schema or shape"))
            return state
        for key in ("arms", "outcome_ids", "outcomes"):
            if isinstance(loaded.get(key), (dict, list)):
                state[key] = loaded[key]
        state["updated_at"] = loaded.get("updated_at")
        for arm in spec.arms:
            stats = state["arms"].get(arm)
            if not isinstance(stats, dict):
                state["arms"][arm] = _empty_arm_stats()
            else:
                clean = _empty_arm_stats()
                for field in clean:
                    value = stats.get(field)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        clean[field] = value
                state["arms"][arm] = clean
        return state

    @staticmethod
    def _quarantine(path: Path, exc: Exception) -> None:
        stamped = path.with_name(f"{path.name}.corrupt-{int(datetime.now().timestamp())}")
        try:
            os.replace(path, stamped)
            LOG.warning("meta state %s was invalid (%s); moved to %s", path, exc, stamped)
        except OSError:
            LOG.warning("meta state %s was invalid (%s); ignoring it", path, exc)

    def _save(self, spec: type[Domain], state: Dict[str, Any]) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.meta_dir, 0o700)
        except OSError:
            pass
        state["updated_at"] = _utcnow()
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=f".{spec.name}.", suffix=".tmp", dir=str(self.meta_dir))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.state_path(spec.name))
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    # -- posterior --------------------------------------------------------

    def _prior_strength(self, task_type: Optional[str]) -> float:
        try:
            return float(policy_for_task(load_policy_file(), task_type)["prior_strength"])
        except Exception:  # noqa: BLE001 - inspection must not fail on policy input
            return float(default_params()["prior_strength"])

    @staticmethod
    def _aggregate(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, Any]] = {}
        for row in outcomes:
            if not isinstance(row, Mapping):
                continue
            arm = row.get("arm")
            if not isinstance(arm, str):
                continue
            stats = totals.setdefault(arm, _empty_arm_stats())
            success = row.get("success_delta")
            failure = row.get("failure_delta")
            cost = row.get("cost")
            stats["successes"] += float(success) if isinstance(success, (int, float)) else 0.0
            stats["failures"] += float(failure) if isinstance(failure, (int, float)) else 0.0
            stats["calls"] += 1
            stats["total_cost"] += float(cost) if isinstance(cost, (int, float)) else 0.0
            reward = row.get("outcome")
            stats["total_reward"] += float(reward) if isinstance(reward, (int, float)) else 0.0
        return totals

    def _scoped_stats(
        self,
        spec: type[Domain],
        state: Mapping[str, Any],
        context: Optional[Mapping[str, Any]],
    ) -> Tuple[str, Dict[str, Dict[str, Any]], int]:
        """Choose context-bucket or global sufficient statistics."""
        global_stats: Dict[str, Dict[str, Any]] = {}
        for arm in spec.arms:
            stats = state["arms"].get(arm) or _empty_arm_stats()
            global_stats[arm] = {
                "successes": float(stats.get("successes", 0.0)),
                "failures": float(stats.get("failures", 0.0)),
                "calls": int(stats.get("calls", 0)),
                "total_cost": float(stats.get("total_cost", 0.0)),
                "total_reward": float(stats.get("total_reward", 0.0)),
            }
        clean = _scalar_context(context)
        if not clean:
            return "global", global_stats, sum(s["calls"] for s in global_stats.values())
        matched = [
            row for row in state.get("outcomes", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("context"), Mapping)
            and all(row["context"].get(key) == value for key, value in clean.items())
        ]
        bucket_stats = self._aggregate(matched)
        scoped = {arm: bucket_stats.get(arm, _empty_arm_stats()) for arm in spec.arms}
        observations = sum(stats["calls"] for stats in scoped.values())
        if observations < self.min_context_observations:
            return "global", global_stats, sum(s["calls"] for s in global_stats.values())
        return "context", scoped, observations

    def posterior(
        self,
        domain: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inspectable posterior for a domain (optionally context-filtered)."""
        spec = self.domain(domain)
        state = self._load(spec)
        task_type = (_scalar_context(context) or {}).get("task_type")
        prior = self._prior_strength(task_type if isinstance(task_type, str) else None)
        scope, stats, observations = self._scoped_stats(spec, state, context)

        arms: Dict[str, Any] = {}
        for arm in spec.arms:
            stats_for_arm = stats.get(arm, _empty_arm_stats())
            successes = float(stats_for_arm.get("successes", 0.0))
            failures = float(stats_for_arm.get("failures", 0.0))
            calls = int(stats_for_arm.get("calls", 0))
            alpha = successes + prior
            beta = failures + prior
            arms[arm] = {
                "successes": round(successes, 6),
                "failures": round(failures, 6),
                "calls": calls,
                "avg_cost": round(float(stats_for_arm.get("total_cost", 0.0)) / calls, 8) if calls else 0.0,
                "avg_reward": round(float(stats_for_arm.get("total_reward", 0.0)) / calls, 8) if calls else 0.0,
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "posterior_mean": round(alpha / (alpha + beta), 6) if alpha + beta else None,
                "value": spec.value(arm),
                "describe": spec.describe_arm(arm),
            }

        global_calls = sum(
            int((state["arms"].get(arm) or {}).get("calls", 0)) for arm in spec.arms
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "domain": spec.name,
            "description": spec.description,
            "instrumentation": spec.instrumentation,
            "outcome_source": spec.outcome_source,
            "arms": arms,
            "observations": observations,
            "total_observations": global_calls,
            "context": _scalar_context(context),
            "context_scope": scope,
            "prior_strength": round(prior, 6),
            "updated_at": state.get("updated_at"),
            "state_path": str(self.state_path(spec.name)),
            "outcome_log_rows": len(state.get("outcomes", [])),
        }

    # -- selection --------------------------------------------------------

    def _eligible(self, spec: type[Domain], exclude: Iterable[Any]) -> Tuple[list, list]:
        excluded = []
        for value in exclude or ():
            try:
                excluded.append(spec.normalize(value))
            except DomainError:
                LOG.warning("meta %s: ignoring unknown exclusion %r", spec.name, value)
        eligible = [arm for arm in spec.arms if arm not in excluded]
        return eligible, excluded

    def decide(
        self,
        domain: str,
        context: Optional[Mapping[str, Any]] = None,
        *,
        exclude: Iterable[Any] = (),
        required_capability: float = 0.0,
        max_cost: float = math.inf,
    ) -> Dict[str, Any]:
        """Draw an arm and return the draw plus why it was chosen."""
        spec = self.domain(domain)
        context_clean = _scalar_context(context)
        posterior = self.posterior(spec.name, context_clean)
        eligible, excluded = self._eligible(spec, exclude)
        detail: Dict[str, Any] = {
            "domain": spec.name,
            "context": context_clean,
            "excluded": excluded,
            "source": "thompson",
            "posterior": posterior,
        }
        if not eligible:
            arm = spec.default_arm
            detail.update({"arm": arm, "choice": spec.value(arm), "source": "default_fallback",
                           "reason": "all arms excluded"})
            LOG.warning("meta %s: every arm is excluded; using default %s", spec.name, arm)
            return detail

        # Fractional Beta counts are intentional: a graded reward of 0.75 is
        # 0.75 of a success, so the posterior mean equals the observed reward.
        arms = [
            Arm(
                arm,
                float(posterior["arms"][arm]["successes"]),
                float(posterior["arms"][arm]["failures"]),
                float(posterior["arms"][arm]["avg_cost"]),
                int(posterior["arms"][arm]["calls"]),
            )
            for arm in eligible
        ]
        task_type = context_clean.get("task_type")
        try:
            chosen = self.client.select_arm(
                spec.name,
                arms,
                task_type=task_type if isinstance(task_type, str) else None,
                required_capability=required_capability,
                max_cost=max_cost,
                rng=self.rng,
            )
            if chosen not in eligible:  # defensive: policy returned an unexpected arm
                raise ValueError(f"policy returned unknown arm {chosen!r}")
        except Exception as exc:  # noqa: BLE001 - a draw must never abort the caller
            LOG.warning("meta %s: Thompson draw failed (%s); using highest posterior mean",
                        spec.name, exc)
            chosen = max(eligible, key=lambda a: (
                posterior["arms"][a]["calls"] > 0,
                posterior["arms"][a]["posterior_mean"] or 0.0,
                posterior["arms"][a]["calls"],
                a == spec.default_arm,
            ))
            detail.update({"source": "posterior_fallback", "reason": f"{type(exc).__name__}: {exc}"})
        detail.update({"arm": chosen, "choice": spec.value(chosen)})
        return detail

    def select(
        self,
        domain: str,
        context: Optional[Mapping[str, Any]] = None,
        *,
        exclude: Iterable[Any] = (),
        required_capability: float = 0.0,
        max_cost: float = math.inf,
    ) -> Any:
        """The typed choice for a domain (int size, RAG mapping, or label)."""
        return self.decide(
            domain,
            context,
            exclude=exclude,
            required_capability=required_capability,
            max_cost=max_cost,
        )["choice"]

    # -- feedback ---------------------------------------------------------

    def record(
        self,
        domain: str,
        choice: Any,
        outcome: Any,
        cost: Any = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        outcome_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Feed one outcome back.  Returns the outcome row (or the duplicate).

        ``outcome`` is a bool or a reward in [0, 1].  Graded rewards are stored
        as fractional Beta counts, so a 0.75 becomes 0.75 success and 0.25
        failure -- the posterior mean still equals the observed reward.
        """
        spec = self.domain(domain)
        arm = spec.normalize(choice)
        reward = _coerce_reward(outcome)
        cost_value = _coerce_cost(cost)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be a mapping, got {type(metadata).__name__}")
        metadata_clean = dict(metadata or {})
        context_clean = _scalar_context(context if context is not None else metadata_clean.get("context"))
        encoded = json.dumps(metadata_clean, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} bytes")
        event_id = outcome_id or metadata_clean.get("outcome_id") or metadata_clean.get("event_id")
        event_id = str(event_id) if event_id else f"auto-{uuid.uuid4().hex}"
        if len(event_id) > 512:  # caller data cannot bloat the id set
            event_id = "sha256:" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()
        row = {
            "outcome_id": event_id,
            "recorded_at": _utcnow(),
            "arm": arm,
            "outcome": reward,
            "success_delta": round(reward, 12),
            "failure_delta": round(1.0 - reward, 12),
            "cost": cost_value,
            "context": context_clean,
            "metadata": metadata_clean,
        }

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.meta_dir, 0o700)
        except OSError:
            pass
        lock_path = self.meta_dir / ".meta.lock"
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            state = self._load(spec)
            if event_id in state["outcome_ids"]:
                return {"recorded": False, "duplicate": True, "outcome_id": event_id,
                        "domain": spec.name, "arm": arm}
            stats = state["arms"].setdefault(arm, _empty_arm_stats())
            stats["successes"] += reward
            stats["failures"] += 1.0 - reward
            stats["calls"] += 1
            stats["total_cost"] += cost_value
            stats["total_reward"] += reward
            state["outcome_ids"].append(event_id)
            state["outcomes"].append(row)
            if len(state["outcomes"]) > MAX_OUTCOME_LOG:
                state["outcomes"] = state["outcomes"][-MAX_OUTCOME_LOG:]
            self._save(spec, state)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        result = dict(row)
        result.update({"recorded": True, "duplicate": False, "domain": spec.name})
        return result

    # -- status -----------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """One-row summary per registered domain."""
        domains = []
        for name in domain_names():
            spec = self.domain(name)
            posterior = self.posterior(name)
            observed = [item for item in posterior["arms"].items() if item[1]["calls"] > 0]
            ranked = sorted(
                observed,
                key=lambda item: ((item[1]["posterior_mean"] or 0.0), item[1]["calls"],
                                  item[0] == spec.default_arm),
                reverse=True,
            )
            top_arm, top = ranked[0] if ranked else ("", {})
            if not ranked:
                # No evidence yet: show the documented default, not an
                # unobserved arm that merely ties at the prior mean.
                top_arm, top = spec.default_arm, posterior["arms"].get(spec.default_arm, {})
            domains.append({
                "domain": name,
                "description": spec.description,
                "arms": list(spec.arms),
                "default_arm": spec.default_arm,
                "instrumentation": spec.instrumentation,
                "outcome_source": spec.outcome_source,
                "observations": posterior["total_observations"],
                "top_arm": top_arm,
                "top_posterior_mean": (top.get("posterior_mean")
                                       if posterior["total_observations"] else None),
                "updated_at": posterior["updated_at"],
                "state_path": posterior["state_path"],
            })
        return {
            "schema_version": SCHEMA_VERSION,
            "meta_dir": str(self.meta_dir),
            "domains": domains,
        }
