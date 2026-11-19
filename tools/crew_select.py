#!/usr/bin/env python3
"""Thompson-driven crew selection: measured performance decides who works.

For a task type, ask the live Thompson router (through
``services/shared/thompson_client.py`` ``select_model``) which arms should staff
the crew.  The result is a lineup of *distinct* arms, with any named arbiter
excluded -- the arbiter stays separately configured and never reviews its own
crew's slot.

How a multi-worker crew is drawn
--------------------------------
``select_model`` samples one arm from the posterior.  It has no parameter for
"draw from these candidates" or "exclude these arms", so this module draws
repeatedly and rejects anything already drawn for this crew (plus the named
arbiter).  Rejection sampling until a fresh arm appears is a draw *without
replacement*: each accepted arm is a sample from the Thompson posterior
conditioned on the arms still available, so later slots compete only among
remaining arms.  This is what makes Thompson explore: a dominant arm takes the
first slot, and the next slot is drawn from the posterior with that arm removed.

If an arm keeps winning past ``MAX_DRAWS_PER_SLOT`` attempts, the slot is filled
from the highest posterior mean among the undrawn, non-excluded arms from the
``get_state`` snapshot (``source="posterior_fill"``), and a warning is logged.
A crew never contains the same arm twice.  An arm with no entry in
``ARM_MODEL_IDS`` passes through with the arm name as its model ID and is
flagged under ``unmapped_arms`` in the ledger; if the gateway cannot call it,
that worker failure becomes an outcome Thompson learns from.

Fallback
--------
If Thompson is unreachable (ping fails, ``get_state`` fails, or a draw fails
mid-selection), the documented default lineup -- today's free trio of Muse
Spark, GLM, and DeepSeek -- is used.  The fallback is logged at WARNING and the
ledger row records ``status="fallback"`` with the reason.  It is never silent.

Ledger (the exposure)
---------------------
Every selection is appended to ``~/.FlossWare/ledger/crews/<ts>.jsonl``
(override with ``POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR``).  The row holds the
task type, the drawn arms and their gateway model IDs, why each slot got its
arm (Thompson draw vs posterior fill vs fallback), how many draws and
collisions it took, and a posterior snapshot (successes/failures, Beta
alpha/beta, posterior mean, average cost) of every arm at selection time.

That row is the *exposure*.  Outcomes already flow back to Thompson through
``tools/ledger_to_thompson.py``: the review ledger records the model ID that
actually ran, and ``ledger_to_thompson`` maps it to the same arm name used
here (see ``ARM_MODEL_IDS`` / ``MODEL_ALIASES``), so selection and outcome
share one arm namespace and the loop closes without extra plumbing.

Crew size
---------
``--learn-size`` asks the meta-learning ``crew_size`` domain how many workers
to draw (arms 1-5).  The draw is recorded in the ledger row under
``learned_size``; ``--workers`` is ignored in that mode.  Outcomes are derived
from the review ledger by ``python -m learning.meta backfill crew_size`` (or
automatically by ``multi_model_review --learn-size``), so a crew size is only
credited once a later review of the same subject cannot overturn it.

Usage
-----
    crew_select.py [DESCRIPTION] [--task-type TYPE] [--workers N] [--learn-size]
                   [--arbiter MODEL] [--exclude ARM] [--json] [--no-record]

    crew_select.py "review the ingest retry path" --task-type code-review
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(REPO, "services", "shared")
if os.path.isdir(SHARED) and SHARED not in sys.path:
    sys.path.insert(0, SHARED)

from thompson_client import ThompsonClient  # noqa: E402

LOG = logging.getLogger("crew_select")

CREWS_DIR_ENV = "POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR"
DEFAULT_CREWS_DIR = "~/.FlossWare/ledger/crews"

DEFAULT_TASK_TYPE = "code-review"
DEFAULT_CREW_SIZE = 3
DEFAULT_ARBITER = "claude-opus-5-5"

# Bound on draws for one slot before falling back to a posterior-ranked fill.
# Loopback HTTP draws are ~1 ms, so 25 attempts is cheap; it only trips when one
# arm's posterior dominates almost every sample.
MAX_DRAWS_PER_SLOT = 25

# Matches the default Thompson policy gene; read from the policy file when it is
# available so the snapshot's Beta parameters match the service's draws.
DEFAULT_PRIOR_STRENGTH = 1.0

# Thompson arm name -> callable gateway model ID.  The arm names are the keys in
# the live thompson-sampling-state.json; the IDs are what the gateway accepts.
# Keep in sync with MODEL_ALIASES in tools/ledger_to_thompson.py, which maps the
# worker ledger's model IDs back to these same arm names.
ARM_MODEL_IDS: Dict[str, str] = {
    "deepseek-v4.1-flash": "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "glm-5-3": "rits/zai-org/glm-5-3",
    "muse-spark-open": "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
}

# Extra ledger spellings seen in the review ledger for the same arms.
_MODEL_TO_ARM: Dict[str, str] = {model: arm for arm, model in ARM_MODEL_IDS.items()}
_MODEL_TO_ARM.update({
    "rits/deepseek-ai/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash",
    "rits/meta-models/muse-spark-open-816b-a42b-nvfp4": "muse-spark-open",
    "claude-sonnet-5": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
})
_MODEL_TO_ARM_LOWER: Dict[str, str] = {model.lower(): arm for model, arm in _MODEL_TO_ARM.items()}

# Documented fallback: the free EnMaaS trio the review tool used before crews
# were Thompson-routed.  Same order and IDs as multi_model_review.WORKERS.
DEFAULT_LINEUP: List[str] = [
    ARM_MODEL_IDS["muse-spark-open"],
    ARM_MODEL_IDS["glm-5-3"],
    ARM_MODEL_IDS["deepseek-v4.1-flash"],
]

# Coarse keyword lookup for turning a free-text description into a task type.
# Explicit --task-type always wins; this is not a model call.
TASK_KEYWORDS: Sequence[tuple[str, tuple[str, ...]]] = (
    ("refactoring", ("refactor", "restructure", "clean up", "cleanup", "simplify")),
    ("testing", ("test", "pytest", "coverage", "fixture")),
    ("documentation", ("doc", "readme", "document", "changelog")),
    ("bug_analysis", ("root cause", "crash", "regression", "bug", "failure")),
    ("architecture", ("architecture", "design", "schema", "interface")),
    ("research", ("research", "investigate", "compare", "evaluate")),
    ("code-review", ("review", "defect", "audit", "security", "vulnerability")),
)


def arm_of(value: str) -> str:
    """Normalise a Thompson arm name or gateway model ID to an arm name."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid model or arm: {value!r}")
    if value in ARM_MODEL_IDS:
        return value
    return _MODEL_TO_ARM.get(value) or _MODEL_TO_ARM_LOWER.get(value.lower()) or value


def model_of(arm: str) -> str:
    """Gateway model ID for an arm; unknown arms pass through unchanged."""
    return ARM_MODEL_IDS.get(arm, arm)


# learning.train task types -> this module's Thompson task types. ``other`` and
# ``implementation`` have no exact equivalent here, so they fall through to the
# keyword rules below rather than being forced into a misleading bucket.
_CLASSIFIER_TASK_MAP: Dict[str, str] = {
    "code-review": "code-review",
    "qa": "testing",
    "documentation": "documentation",
    "research": "research",
}
# Below this confidence the trained classifier is not trusted for routing; the
# deterministic keyword lookup is more predictable.
CLASSIFIER_MIN_CONFIDENCE = 0.35
_TASK_CLASSIFIER = "unset"


def _load_task_classifier():
    """Optional trained task classifier; ``None`` when unavailable.

    Never fatal: crew selection must work before the model is trained.
    """
    global _TASK_CLASSIFIER
    if _TASK_CLASSIFIER != "unset":
        return _TASK_CLASSIFIER
    root = os.environ.get("POC_CLAUDE_ENSEMBLE_TRAIN_DIR")
    root = os.path.expanduser(root) if root else os.path.expanduser("~/.FlossWare/learning/train")
    path = os.path.join(root, "task_classifier.joblib")
    try:
        if not os.path.exists(path):
            _TASK_CLASSIFIER = None
            return None
        services = os.path.join(REPO, "services")
        if services not in sys.path:
            sys.path.insert(0, services)
        from learning.train.task_classifier import TaskClassifier

        _TASK_CLASSIFIER = TaskClassifier.load(path)
    except Exception:
        _TASK_CLASSIFIER = None
    return _TASK_CLASSIFIER


def infer_task_type(description: Optional[str]) -> str:
    """Task-type inference: trained classifier when confident, else keywords."""
    text = (description or "").lower()
    model = _load_task_classifier()
    if model is not None and text:
        try:
            label, confidence = model.classify(description or "")
            mapped = _CLASSIFIER_TASK_MAP.get(label)
            if mapped and confidence >= CLASSIFIER_MIN_CONFIDENCE:
                return mapped
        except Exception:
            pass
    for task_type, keywords in TASK_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return task_type
    return DEFAULT_TASK_TYPE


def crews_dir() -> Path:
    """Directory where crew-selection exposure rows are written."""
    return Path(os.path.expanduser(os.environ.get(CREWS_DIR_ENV, DEFAULT_CREWS_DIR)))


def meta_learner():
    """The meta-learning learner, or None when the services tree is absent."""
    try:
        services = os.path.join(REPO, "services")
        if services not in sys.path:
            sys.path.insert(0, services)
        from learning.meta import MetaLearner

        return MetaLearner()
    except Exception as exc:  # noqa: BLE001 - size learning is optional
        LOG.warning("crew-size meta-learning unavailable (%s); using requested size", exc)
        return None


def _prior_strength(task_type: str) -> float:
    """Prior strength the service uses for this task type, best effort.

    The live service reads the same policy module from services/shared; if the
    policy file is missing or unusable both sides fall back to 1.0.
    """
    try:
        from thompson_policy import load_policy_file, policy_for_task

        return float(policy_for_task(load_policy_file(), task_type)["prior_strength"])
    except Exception:  # noqa: BLE001 - snapshot must not fail on policy input
        return DEFAULT_PRIOR_STRENGTH


def posterior_snapshot(state: Optional[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
    """Beta parameters and posterior means for every arm in a get_state snapshot."""
    if not isinstance(state, dict):
        return None
    prior = _prior_strength(task_type)
    arms: Dict[str, Any] = {}
    for name, stats in sorted((state.get("models") or {}).items()):
        if not isinstance(stats, dict):
            continue
        successes = int(stats.get("successes") or 0)
        failures = int(stats.get("failures") or 0)
        calls = int(stats.get("calls") or 0)
        alpha = successes + prior
        beta = failures + prior
        arms[name] = {
            "successes": successes,
            "failures": failures,
            "calls": calls,
            "avg_cost": round(float(stats.get("total_cost") or 0.0) / calls, 8) if calls else 0.0,
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "posterior_mean": round(alpha / (alpha + beta), 6) if alpha + beta else None,
        }
    return {"prior_strength": prior,
            "state_last_updated": state.get("last_updated"),
            "arms": arms}


def _client_degraded(client: ThompsonClient) -> bool:
    """True when the last client request failed (the client falls back silently).

    ThompsonClient.select_model returns 'haiku' both when the service chooses
    haiku and when the request fails.  A failed request leaves the circuit
    breaker's consecutive-failure count above zero, which lets crew selection
    tell the two apart and abort to the documented fallback instead of
    recording a fabricated arm.
    """
    breaker = getattr(client, "circuit_breaker", None)
    if breaker is None:
        return False
    return bool(getattr(breaker, "is_open", lambda: False)()) or bool(
        getattr(breaker, "failure_count", 0))


def _posterior_fill(state: Dict[str, Any], task_type: str,
                    drawn: Sequence[str], excluded: Sequence[str]) -> Optional[str]:
    """Highest-posterior-mean arm not already drawn or excluded."""
    snapshot = posterior_snapshot(state, task_type) or {}
    remaining = [(name, values) for name, values in (snapshot.get("arms") or {}).items()
                 if name not in drawn and name not in excluded]
    if not remaining:
        return None
    remaining.sort(key=lambda item: (item[1]["posterior_mean"] or 0.0, item[1]["calls"], item[0]),
                   reverse=True)
    return remaining[0][0]


def _selection_id(task_type: str, timestamp: str) -> str:
    return hashlib.sha256(f"{task_type}|{timestamp}|{time.time_ns()}".encode("utf-8")).hexdigest()[:24]


def _write_record(record: Dict[str, Any], ledger_dir: Optional[os.PathLike | str]) -> Dict[str, Any]:
    """Append the selection row to its own <ts>.jsonl file. Never raises."""
    root = Path(ledger_dir).expanduser() if ledger_dir else crews_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = root / f"{base}.jsonl"
        attempt = 0
        while True:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                attempt += 1
                if attempt > 10:
                    raise
                path = root / f"{base}-{attempt}.jsonl"
        record["ledger_path"] = str(path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        LOG.error("crew selection ledger write failed: %s", exc)
        record["ledger_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _worker_row(slot: int, arm: str, source: str, draws: int) -> Dict[str, Any]:
    model = model_of(arm)
    return {
        "slot": slot,
        "arm": arm,
        "model": model,
        "mapped": model != arm,
        "source": source,
        "draws": draws,
    }


def _fallback(record: Dict[str, Any], reason: str, detail: str, *,
              count: int, excluded: Sequence[str], task_type: str,
              state: Optional[Dict[str, Any]], ledger_dir: Optional[os.PathLike | str],
              record_selection: bool) -> Dict[str, Any]:
    """Use the documented default lineup, loudly, and record it."""
    lineup = [model for model in DEFAULT_LINEUP if arm_of(model) not in excluded][:count]
    workers = [_worker_row(slot, arm_of(model), "fallback_default", 0)
               for slot, model in enumerate(lineup)]
    record.update({
        "status": "fallback",
        "reason": reason,
        "detail": detail,
        "workers": workers,
        "models": [worker["model"] for worker in workers],
        "posterior": posterior_snapshot(state, task_type),
    })
    LOG.warning("Thompson %s (%s); using documented default crew %s",
                reason, detail, record["models"])
    if len(workers) < count:
        LOG.warning("fallback crew has %d workers (requested %d) after arbiters/exclusions",
                    len(workers), count)
    return _write_record(record, ledger_dir) if record_selection else record


def select_crew(task_type: str = DEFAULT_TASK_TYPE, *, description: Optional[str] = None,
                count: int = DEFAULT_CREW_SIZE, arbiter: Optional[str] = DEFAULT_ARBITER,
                exclude: Iterable[str] = (), required_capability: float = 0.5,
                max_cost: float = math.inf, client: Optional[ThompsonClient] = None,
                ledger_dir: Optional[os.PathLike | str] = None,
                max_draws_per_slot: int = MAX_DRAWS_PER_SLOT,
                record_selection: bool = True,
                correlation_id: Optional[str] = None,
                learn_size: bool = False,
                learner: Optional[Any] = None) -> Dict[str, Any]:
    """Draw a distinct-arm worker crew from Thompson, excluding the arbiter.

    Args:
        task_type: Thompson task type (e.g. ``code-review``).
        description: Free-text task description, recorded for context.
        count: Desired number of workers (ignored when ``learn_size`` is set).
        arbiter: Model ID or arm that must never appear in the crew.  The
            arbiter is configured separately (``multi_model_review.ARBITER``).
        exclude: Additional models/arms to exclude.
        required_capability / max_cost: Passed through to Thompson.
        client: ThompsonClient (tests inject one pointed at a fake service).
        ledger_dir: Override for the crews ledger directory.
        max_draws_per_slot: Rejection-sampling budget per slot.
        record_selection: Write the exposure row (tests may disable).
        correlation_id: Optional review/task id recorded for later joins.
        learn_size: Draw the crew size from the meta-learning ``crew_size``
            domain instead of using ``count``.
        learner: MetaLearner override (tests); default constructs one.

    Returns:
        The ledger row (also written to ``<crews dir>/<ts>.jsonl``).
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if max_draws_per_slot < 1:
        raise ValueError("max_draws_per_slot must be at least 1")

    learned_size: Optional[Dict[str, Any]] = None
    if learn_size:
        active = learner if learner is not None else meta_learner()
        if active is None:
            LOG.warning("--learn-size requested but meta-learning is unavailable; "
                        "using %d workers", count)
        else:
            try:
                detail = active.decide("crew_size", {"task_type": task_type})
                count = int(detail["choice"])
                learned_size = {
                    "arm": detail["arm"],
                    "choice": count,
                    "source": detail["source"],
                    "context_scope": detail["posterior"]["context_scope"],
                    "posterior_mean": detail["posterior"]["arms"][detail["arm"]]["posterior_mean"],
                    "state_path": detail["posterior"]["state_path"],
                }
                LOG.info("meta-learning drew crew size %d (%s, %s)",
                         count, detail["arm"], detail["source"])
            except Exception as exc:  # noqa: BLE001 - fall back to the requested size
                LOG.warning("crew-size meta-learning draw failed (%s); using %d workers",
                            exc, count)

    excluded: List[str] = []
    for value in ([arbiter] if arbiter else []) + list(exclude):
        arm = arm_of(value)
        if arm not in excluded:
            excluded.append(arm)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    record: Dict[str, Any] = {
        "schema_version": 1,
        "producer": "crew-select",
        "kind": "crew_selection",
        "selection_id": _selection_id(task_type, timestamp),
        "recorded_at": timestamp,
        "correlation_id": correlation_id,
        "task_type": task_type,
        "description": description,
        "requested_workers": count,
        "learned_size": learned_size,
        "size_source": "meta_learner" if learned_size else "requested",
        "method": "thompson-draw-without-replacement",
        "arbiter": {"configured": arbiter, "arm": arm_of(arbiter) if arbiter else None,
                    "excluded": arm_of(arbiter) in excluded if arbiter else False},
        "excluded": excluded,
        "required_capability": required_capability,
        "max_cost": None if math.isinf(max_cost) else max_cost,
        "max_draws_per_slot": max_draws_per_slot,
        "status": None,
        "reason": None,
        "detail": None,
        "workers": [],
        "models": [],
        "draws": {"total": 0, "collisions": 0, "per_slot": []},
        "fill_used": False,
        "posterior": None,
    }

    client = client if client is not None else ThompsonClient()

    try:
        reachable = bool(client.ping())
    except Exception as exc:  # noqa: BLE001 - fallback must survive any client error
        reachable = False
        record["detail_ping"] = f"{type(exc).__name__}: {exc}"
    if not reachable:
        return _fallback(record, "thompson_unavailable", "ping failed",
                         count=count, excluded=excluded, task_type=task_type,
                         state=None, ledger_dir=ledger_dir,
                         record_selection=record_selection)

    try:
        state = client.get_state()
    except Exception as exc:  # noqa: BLE001
        state = None
        record["detail_state"] = f"{type(exc).__name__}: {exc}"
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        return _fallback(record, "thompson_state_unavailable", "get_state failed",
                         count=count, excluded=excluded, task_type=task_type,
                         state=None, ledger_dir=ledger_dir,
                         record_selection=record_selection)

    record["posterior"] = posterior_snapshot(state, task_type)
    eligible = [arm for arm in sorted(state["models"]) if arm not in excluded]
    record["eligible"] = eligible
    if len(eligible) < count:
        LOG.warning("only %d eligible arms for a %d-worker crew; crew will be short",
                    len(eligible), count)
        record["truncated"] = True

    drawn: List[str] = []
    per_slot: List[int] = []
    collisions = 0
    total_draws = 0

    for slot in range(count):
        slot_draws = 0
        chosen: Optional[str] = None
        source = "thompson_draw"
        while slot_draws < max_draws_per_slot:
            try:
                candidate = client.select_model(task_type, required_capability, max_cost)
            except Exception as exc:  # noqa: BLE001
                LOG.error("Thompson draw failed: %s", exc)
                candidate = None
            slot_draws += 1
            total_draws += 1
            if _client_degraded(client):
                record["draws"] = {"total": total_draws, "collisions": collisions,
                                   "per_slot": per_slot + [slot_draws]}
                return _fallback(record, "thompson_error", "draw failed mid-selection",
                                 count=count, excluded=excluded, task_type=task_type,
                                 state=state, ledger_dir=ledger_dir,
                                 record_selection=record_selection)
            if not isinstance(candidate, str) or not candidate:
                collisions += 1
                continue
            arm = arm_of(candidate)
            if arm in drawn or arm in excluded:
                collisions += 1
                LOG.debug("slot %d: rejected %s (already drawn or excluded)", slot, arm)
                continue
            chosen = arm
            break
        if chosen is None:
            chosen = _posterior_fill(state, task_type, drawn, excluded)
            source = "posterior_fill"
            record["fill_used"] = True
            if chosen is None:
                LOG.warning("no eligible arm left for slot %d; crew stops at %d",
                            slot, len(drawn))
                break
            LOG.warning("slot %d: no fresh Thompson arm after %d draws; "
                        "filled %s from posterior ranking", slot, slot_draws, chosen)
        per_slot.append(slot_draws)
        drawn.append(chosen)
        record["workers"].append(_worker_row(len(drawn) - 1, chosen, source, slot_draws))

    record["models"] = [worker["model"] for worker in record["workers"]]
    record["draws"] = {"total": total_draws, "collisions": collisions, "per_slot": per_slot}
    if len(record["workers"]) < count:
        record["status"] = "thompson"
        record["reason"] = "insufficient_eligible_arms"
        record["detail"] = f"drew {len(record['workers'])} of {count} requested workers"
        LOG.warning("crew has %d workers of %d requested: %s",
                    len(record["workers"]), count, record["detail"])
    else:
        record["status"] = "thompson"

    unmapped = [worker["arm"] for worker in record["workers"] if not worker["mapped"]]
    if unmapped:
        record["unmapped_arms"] = unmapped
        LOG.warning("Thompson returned arm(s) with no known gateway model ID: %s", unmapped)

    if record_selection:
        return _write_record(record, ledger_dir)
    return record


def _format_record(record: Dict[str, Any]) -> str:
    lines = [f"task_type: {record['task_type']}"
             + (f" (description: {record['description']})" if record.get("description") else ""),
             f"status: {record['status']}"
             + (f" ({record['reason']}: {record['detail']})" if record.get("reason") else "")]
    learned = record.get("learned_size")
    if learned:
        lines.append(f"size: {record['requested_workers']} workers "
                     f"(meta-learning arm {learned['arm']}, {learned['source']}, "
                     f"posterior mean {learned.get('posterior_mean')})")
    if record.get("excluded"):
        lines.append("excluded: " + ", ".join(record["excluded"]))
    lines.append("crew:")
    posterior = (record.get("posterior") or {}).get("arms") or {}
    for worker in record["workers"]:
        stats = posterior.get(worker["arm"]) or {}
        if stats:
            detail = (f"{stats['successes']}/{stats['successes'] + stats['failures']} successes, "
                      f"posterior mean {stats['posterior_mean']:.3f}, "
                      f"avg cost ${stats['avg_cost']:.4f}")
        else:
            detail = "no posterior snapshot"
        lines.append(f"  {worker['slot'] + 1}. {worker['arm']} -> {worker['model']} "
                     f"[{worker['source']}, {worker['draws']} draw(s)] {detail}")
    draws = record["draws"]
    lines.append(f"draws: total={draws['total']} collisions={draws['collisions']} "
                 f"per_slot={draws['per_slot']}")
    if record.get("ledger_path"):
        lines.append(f"recorded: {record['ledger_path']}")
    if record.get("ledger_error"):
        lines.append(f"ledger write FAILED: {record['ledger_error']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a review crew from live Thompson posteriors.")
    parser.add_argument("description", nargs="?",
                        help="task description (used for the ledger and task-type inference)")
    parser.add_argument("--task-type", default=None,
                        help=f"Thompson task type (default: infer from description, else {DEFAULT_TASK_TYPE})")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"number of workers (default {DEFAULT_CREW_SIZE}, or the "
                             f"meta-learning draw with --learn-size)")
    parser.add_argument("--learn-size", action="store_true",
                        help="let the meta-learning crew_size domain decide the worker count")
    parser.add_argument("--arbiter", default=DEFAULT_ARBITER,
                        help=f"arbiter model/arm never placed in the crew (default {DEFAULT_ARBITER})")
    parser.add_argument("--exclude", action="append", default=[], metavar="MODEL",
                        help="additional model/arm to exclude (repeatable)")
    parser.add_argument("--capability", type=float, default=0.5,
                        help="required capability passed to Thompson (default 0.5)")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="max average cost passed to Thompson (default: unlimited)")
    parser.add_argument("--max-draws-per-slot", type=int, default=MAX_DRAWS_PER_SLOT,
                        help=f"rejection-sampling budget per slot (default {MAX_DRAWS_PER_SLOT})")
    parser.add_argument("--json", action="store_true", help="print the full ledger row as JSON")
    parser.add_argument("--no-record", action="store_true",
                        help="do not write the exposure row to the crews ledger")
    parser.add_argument("--quiet", action="store_true", help="only errors on stderr")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0.0 <= args.capability <= 1.0:
        parser.error("--capability must be between 0 and 1")
    if args.max_cost is not None and args.max_cost < 0:
        parser.error("--max-cost must be at least 0")

    task_type = args.task_type or infer_task_type(args.description)
    if args.learn_size and args.workers is not None:
        LOG.warning("--workers is ignored when --learn-size draws the crew size")
    count = args.workers if args.workers is not None else DEFAULT_CREW_SIZE
    record = select_crew(
        task_type,
        description=args.description,
        count=count,
        arbiter=args.arbiter,
        exclude=args.exclude,
        required_capability=args.capability,
        max_cost=math.inf if args.max_cost is None else args.max_cost,
        max_draws_per_slot=args.max_draws_per_slot,
        record_selection=not args.no_record,
        learn_size=args.learn_size,
    )
    print(json.dumps(record, indent=2) if args.json else _format_record(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
