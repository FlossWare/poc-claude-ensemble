"""Pure Thompson selection policy shared by the live service and GA replay.

The live Thompson service and the offline GA replay must evaluate the *same*
function.  This module is that function: :func:`select` is pure (no globals
other than a read-only mtime cache for the policy file) and takes the RNG as an
argument, so replay can drive it with a seeded generator.

Four genes, all defaulting to today's behaviour:

======================  ======  ======  =======
gene                    low     high    default
======================  ======  ======  =======
``prior_strength``      0.1     5.0     1.0
``cost_weight``         0.0     1.0     0.0
``exploration_epsilon`` 0.0     0.2     0.0
``capability_margin``   -1.0    1.0     -1.0
======================  ======  ======  =======

With the default parameters :func:`select` performs exactly one Beta draw per
cost-eligible arm -- no additional draws -- so it reproduces the pre-GA
algorithm byte for byte when seeded the same way.

The policy file is JSON::

    {"schema_version": 1,
     "policies": {"default": {"prior_strength": 1.0, ...},
                  "code-review": {...}},
     "provenance": {...}}

A missing, unreadable, oversized, symlinked, or invalid file always means
"defaults".  :func:`load_policy_file` never raises, because it is called from
the live request path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_POLICY_BYTES = 64 * 1024
_MAX_POLICIES = 32


@dataclass(frozen=True)
class ParamSpec:
    """Bounds and default for one evolvable policy gene."""

    name: str
    low: float
    high: float
    default: float


PARAM_BOUNDS: Dict[str, ParamSpec] = {
    "prior_strength": ParamSpec("prior_strength", 0.1, 5.0, 1.0),
    "cost_weight": ParamSpec("cost_weight", 0.0, 1.0, 0.0),
    "exploration_epsilon": ParamSpec("exploration_epsilon", 0.0, 0.2, 0.0),
    "capability_margin": ParamSpec("capability_margin", -1.0, 1.0, -1.0),
}
GENES = tuple(PARAM_BOUNDS)

_TOP_LEVEL_KEYS = frozenset({"schema_version", "policies", "provenance"})


class ParamValidationError(ValueError):
    """A policy parameter set is not acceptable."""


class PolicyFileError(Exception):
    """The policy file exists but cannot be trusted."""


def default_params() -> Dict[str, float]:
    """Fresh copy of the default gene values."""
    return {name: spec.default for name, spec in PARAM_BOUNDS.items()}


DEFAULT_PARAMS: Dict[str, float] = default_params()


def validate_params(
    params: Mapping[str, Any],
    *,
    require_all: bool = True,
    source: str = "params",
) -> Dict[str, float]:
    """Validate a gene mapping and return a normalised float copy.

    Rejects: non-mappings, unknown keys, missing keys when ``require_all`` is
    true, booleans (``True`` is an ``int`` in Python), non-numbers, NaN, inf,
    and out-of-range values.  A partial mapping is filled from the defaults
    when ``require_all`` is false.
    """
    if not isinstance(params, Mapping):
        raise ParamValidationError(f"{source}: expected a mapping, got {type(params).__name__}")

    unknown = sorted(set(params) - set(PARAM_BOUNDS))
    if unknown:
        raise ParamValidationError(f"{source}: unknown parameter(s): {', '.join(unknown)}")

    normalised: Dict[str, float] = {}
    for name, spec in PARAM_BOUNDS.items():
        if name not in params:
            if require_all:
                raise ParamValidationError(f"{source}: missing parameter {name!r}")
            normalised[name] = spec.default
            continue
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamValidationError(
                f"{source}: {name} must be a number, got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ParamValidationError(f"{source}: {name} must be finite, got {number!r}")
        if number < spec.low or number > spec.high:
            raise ParamValidationError(
                f"{source}: {name}={number!r} outside [{spec.low}, {spec.high}]"
            )
        normalised[name] = number
    return normalised


class Arm(NamedTuple):
    """Minimal arm view consumed by :func:`select`."""

    name: str
    successes: int
    failures: int
    avg_cost: float
    calls: int

    @property
    def success_rate(self) -> float:
        if self.calls == 0:
            return 0.5  # neutral prior, matching ModelStats.success_rate
        return self.successes / self.calls


def _beta(rng: Any, alpha: float, beta: float) -> float:
    draw = getattr(rng, "beta", None)
    if draw is not None:
        return float(draw(alpha, beta))
    return float(rng.betavariate(alpha, beta))


def _unit(rng: Any) -> float:
    draw = getattr(rng, "random", None)
    if draw is None:
        draw = getattr(rng, "rand")
    return float(draw())


def _index(rng: Any, n: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(n))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(n))
    return int(rng.randint(0, n))


def select(
    params: Mapping[str, Any],
    arms: Iterable[Arm],
    task_type: Optional[str],
    required_capability: float,
    max_cost: float,
    rng: Any,
) -> str:
    """Pick an arm.  Pure: all randomness comes from ``rng``.

    Order of operations matches the pre-GA service exactly when the parameters
    are the defaults:

    1. drop arms whose average cost exceeds ``max_cost`` (untested arms stay),
       falling back to the single cheapest arm when everything is filtered;
    2. with ``capability_margin >= 0``, drop arms whose observed success rate is
       below ``max(required_capability, capability_margin)``, falling back to
       the unfiltered set when that empties it;
    3. one Beta(successes + prior, failures + prior) draw per remaining arm,
       optionally multiplied by a scale-free cost penalty;
    4. with ``exploration_epsilon > 0``, uniformly pick another candidate with
       that probability.
    """
    normalised = validate_params(params, require_all=True, source="select")
    candidates = list(arms)
    if not candidates:
        raise ValueError("select: no arms")

    eligible = [a for a in candidates if a.avg_cost <= max_cost or a.calls == 0]
    if not eligible:
        eligible = [min(candidates, key=lambda a: a.avg_cost)]

    margin = normalised["capability_margin"]
    if margin >= 0.0:
        floor = max(float(required_capability), margin)
        filtered = [a for a in eligible if a.success_rate >= floor]
        if filtered:
            eligible = filtered

    prior = normalised["prior_strength"]
    cost_weight = normalised["cost_weight"]
    max_avg_cost = max(a.avg_cost for a in eligible)

    best_name: Optional[str] = None
    best_score = -math.inf
    for arm in eligible:
        score = _beta(rng, arm.successes + prior, arm.failures + prior)
        if cost_weight > 0.0:
            normalised_cost = (arm.avg_cost / max_avg_cost) if max_avg_cost > 0.0 else 0.0
            score *= 1.0 - cost_weight * normalised_cost
        if best_name is None or score > best_score:
            best_name = arm.name
            best_score = score

    epsilon = normalised["exploration_epsilon"]
    if epsilon > 0.0 and _unit(rng) < epsilon:
        best_name = eligible[_index(rng, len(eligible))].name

    return best_name  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Policy file loading
# ---------------------------------------------------------------------------

_CACHE: Dict[str, tuple] = {}
_LOGGED: set = set()
_CACHE_LOCK = threading.Lock()


def policy_file_path() -> Path:
    """Configured policy file path (``$POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE``)."""
    try:
        from shared.runtime_config import thompson_policy_file

        return thompson_policy_file()
    except Exception:  # pragma: no cover - defensive when imported standalone
        configured = os.environ.get("POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / ".FlossWare" / "claude-ensemble-state" / "data" / "learning" / "thompson-policy.json"


def clear_cache() -> None:
    """Drop the mtime cache (used after the GA writes a new policy file)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _log_once(key: tuple, level: int, message: str) -> None:
    with _CACHE_LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    LOGGER.log(level, message)


def _read_policy_bytes(path: Path) -> bytes:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise PolicyFileError("policy file is not a regular file")
    if info.st_uid != os.getuid():
        raise PolicyFileError("policy file is not owned by the current user")
    if info.st_size > MAX_POLICY_BYTES:
        raise PolicyFileError(f"policy file is larger than {MAX_POLICY_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(path, flags)
    with os.fdopen(handle, "rb") as stream:
        raw = stream.read(MAX_POLICY_BYTES + 1)
    if len(raw) > MAX_POLICY_BYTES:
        raise PolicyFileError(f"policy file is larger than {MAX_POLICY_BYTES} bytes")
    return raw


def _parse_policy_file(path: Path) -> Dict[str, Dict[str, float]]:
    raw = _read_policy_bytes(path)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolicyFileError(f"policy file is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise PolicyFileError("policy file must contain a JSON object")
    unknown = sorted(set(document) - _TOP_LEVEL_KEYS)
    if unknown:
        raise PolicyFileError(f"unknown top-level key(s): {', '.join(unknown)}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PolicyFileError(f"unsupported schema_version {document.get('schema_version')!r}")
    if "provenance" in document and not isinstance(document["provenance"], dict):
        raise PolicyFileError("provenance must be an object")
    policies = document.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise PolicyFileError("policies must be a non-empty object")
    if len(policies) > _MAX_POLICIES:
        raise PolicyFileError(f"too many policies (>{_MAX_POLICIES})")
    parsed: Dict[str, Dict[str, float]] = {}
    for name, params in policies.items():
        if not isinstance(name, str) or not name or len(name) > 64:
            raise PolicyFileError(f"invalid policy name {name!r}")
        if not all(ch.isalnum() or ch in "._/-" for ch in name):
            raise PolicyFileError(f"invalid policy name {name!r}")
        if not isinstance(params, dict):
            raise PolicyFileError(f"policy {name!r} must be an object")
        try:
            parsed[name] = validate_params(params, require_all=False, source=f"policy {name!r}")
        except ParamValidationError as exc:
            raise PolicyFileError(str(exc)) from exc
    parsed.setdefault("default", default_params())
    return parsed


def load_policy_file(path: Optional[os.PathLike | str] = None) -> Dict[str, Dict[str, float]]:
    """Return ``{task_type_or_default: params}``.  Never raises.

    The file is re-stated on every call; only the JSON parse is cached, keyed
    by path + mtime + size + inode, so edits are picked up without a restart.
    Any failure (missing file, symlink, bad JSON, bad bounds) yields defaults.
    """
    target = Path(path).expanduser() if path is not None else policy_file_path()
    try:
        info = os.lstat(target)
        key = (str(target), info.st_mtime_ns, info.st_size, info.st_ino)
        with _CACHE_LOCK:
            cached = _CACHE.get(str(target))
        if cached is not None and cached[0] == key:
            return {name: dict(values) for name, values in cached[1].items()}
        policies = _parse_policy_file(target)
        with _CACHE_LOCK:
            _CACHE[str(target)] = (key, policies)
        return {name: dict(values) for name, values in policies.items()}
    except FileNotFoundError:
        _log_once((str(target), "missing"), logging.INFO, f"no Thompson policy file at {target}; using defaults")
    except Exception as exc:  # noqa: BLE001 - never raise into select_model
        _log_once(
            (str(target), type(exc).__name__, str(exc)),
            logging.WARNING,
            f"ignoring Thompson policy file {target}: {type(exc).__name__}: {exc}",
        )
    return {"default": default_params()}


def policy_for_task(policies: Mapping[str, Mapping[str, Any]], task_type: Optional[str]) -> Dict[str, float]:
    """The policy for a task type, falling back to ``default``."""
    if task_type is not None and isinstance(task_type, str) and task_type in policies:
        return dict(policies[task_type])
    return dict(policies.get("default", DEFAULT_PARAMS))


def policy_document(policies: Mapping[str, Mapping[str, Any]], provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Canonical policy-file document for the given policies."""
    document: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policies": {
            name: validate_params(values, require_all=True, source=f"policy {name!r}")
            for name, values in sorted(policies.items())
        },
    }
    if provenance is not None:
        document["provenance"] = dict(provenance)
    return document
