"""Linear Thompson Sampling over task features.

Per arm a, maintain the standard LinTS sufficient statistics (A_a, b_a):

    theta_a ~ N(A_a^-1 b_a, v^2 A_a^-1)      (drawn per decision)
    score_a = theta_a . x                     (x = task feature vector)
    pick argmax_a score_a

After the chosen arm observes reward r in [0, 1]:

    A_a += x x^T ;  b_a += r x

With a bias feature and a Normal(0, 1)-ish prior scale (A init = lambda * I,
b init = 0), a fresh arm scores near zero everywhere and is explored through
posterior variance rather than a uniform prior -- new arms still get tried
because their theta draw is noisy.

Feature extraction is deliberately minimal and stable: a bias term plus a
one-hot over the known task types, plus two cheap numeric signals the ledger
already carries (log-scaled prompt size bucket indicator).  Feature indices
are derived from a frozen vocabulary so stored state stays valid across
restarts; unseen task types hash to a dedicated "other" bucket instead of
changing the dimension.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

#: Frozen task-type vocabulary.  Order defines feature indices 1..n; index 0
#: is the bias term and the last index is the "unseen task" bucket.
TASK_TYPES: Tuple[str, ...] = (
    "code-review",
    "documentation",
    "summarization",
    "translation",
    "analysis",
    "claude_code_interaction",
)
_OTHER = "__other__"

FEATURE_NAMES: Tuple[str, ...] = ("bias",) + TASK_TYPES + (_OTHER,) + ("long_input",)

FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
DIM = len(FEATURE_NAMES)

DEFAULT_STATE_PATH = "~/.FlossWare/learning/contextual/linear-ts-state.json"
STATE_PATH_ENV = "POC_CLAUDE_ENSEMBLE_CONTEXTUAL_STATE"

#: Prior precision and exploration noise.  lambda=1.0 with rewards in [0,1]
#: gives a mildly informative prior; v=0.5 keeps exploration proportionate.
PRIOR_LAMBDA = 1.0
EXPLORATION_V = 0.5


def features(task_type: Optional[str], prompt_chars: int = 0) -> np.ndarray:
    """Task feature vector; deterministic, fixed dimension."""
    x = np.zeros(DIM, dtype=np.float64)
    x[FEATURE_INDEX["bias"]] = 1.0
    key = task_type if task_type in TASK_TYPES else _OTHER
    x[FEATURE_INDEX[key]] = 1.0
    # Long-input indicator: the ledger's review subjects are commonly
    # truncated at 4000 chars; "long" means beyond that.
    if prompt_chars > 4000:
        x[FEATURE_INDEX["long_input"]] = 1.0
    return x


class ArmState:
    """LinTS sufficient statistics for one arm (JSON-serialisable)."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.a: np.ndarray = np.eye(DIM) * PRIOR_LAMBDA
        self.b: np.ndarray = np.zeros(DIM)
        self.updates = 0

    def draw(self, rng: np.random.RandomState, v: float = EXPLORATION_V) -> np.ndarray:
        """One theta sample from the posterior.

        A singular or otherwise unusable A degrades to a prior draw
        (zero mean, small noise) instead of raising: selection must never
        crash on pathological state.
        """
        try:
            mean = np.linalg.solve(self.a, self.b)
            cov = np.linalg.inv(self.a)
            chol = np.linalg.cholesky(cov * v * v + 1e-12 * np.eye(DIM))
        except np.linalg.LinAlgError:
            return rng.standard_normal(DIM) * 1e-3
        if not np.isfinite(mean).all():
            return rng.standard_normal(DIM) * 1e-3
        return mean + chol @ rng.standard_normal(DIM)

    def update(self, x: np.ndarray, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1]; got {reward}")
        self.a += np.outer(x, x)
        self.b += float(reward) * x
        self.updates += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmState":
        state = cls(str(data["arm"]))
        a = np.asarray(data.get("a"), dtype=np.float64)
        b = np.asarray(data.get("b"), dtype=np.float64)
        if a.shape != (DIM, DIM) or b.shape != (DIM,):
            raise ValueError(
                f"arm {data['arm']!r} has wrong feature dimension "
                f"(state file from a different FEATURE_NAMES?)"
            )
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"arm {data['arm']!r} contains non-finite values")
        state.a = a
        state.b = b
        state.updates = int(data.get("updates", 0))
        return state


class LinearTS:
    """Contextual Thompson policy with file persistence."""

    def __init__(self, state_path: Optional[os.PathLike | str] = None) -> None:
        raw = state_path or os.environ.get(STATE_PATH_ENV, DEFAULT_STATE_PATH)
        self.state_path = Path(os.path.expanduser(raw))
        self.arms: Dict[str, ArmState] = {}
        self.load_error: Optional[str] = None
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, dict):
                raise ValueError(f"state file is not a JSON object ({type(document).__name__})")
            if document.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {document.get('schema_version')!r}"
                )
            if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError("feature vocabulary changed; state is stale")
            arms = document.get("arms")
            if not isinstance(arms, dict):
                raise ValueError(f"state 'arms' is not an object ({type(arms).__name__})")
            self.arms = {
                arm: ArmState.from_dict(payload)
                for arm, payload in arms.items()
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # Corrupt or stale state must not take routing down: start fresh
            # and let the ledger refill the posteriors.  The bad file is
            # moved aside (never overwritten) so it stays available for
            # diagnosis or recovery -- the next save must not destroy it.
            self.arms = {}
            self.load_error = str(exc)
            aside = self.state_path.with_name(
                f"{self.state_path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self.state_path, aside)
            except OSError:
                pass  # unreadable/already moved; the fresh save will replace it

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "arms": {arm: state.to_dict() for arm, state in self.arms.items()},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.state_path.parent, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(document, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, self.state_path)

    # -- policy -----------------------------------------------------------

    def arm(self, name: str) -> ArmState:
        if name not in self.arms:
            self.arms[name] = ArmState(name)
        return self.arms[name]

    def select(
        self,
        arms: Sequence[str],
        task_type: Optional[str],
        prompt_chars: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> str:
        """Thompson draw over the arms for this context."""
        if not arms:
            raise ValueError("no arms to select from")
        rng = rng or np.random.RandomState()
        x = features(task_type, prompt_chars)
        best_arm: Optional[str] = None
        best_score = -math.inf
        for name in arms:
            theta = self.arm(name).draw(rng)
            score = float(theta @ x)
            # NaN scores (degenerate posterior) must never win a comparison
            # and must never leave best_arm unset under `python -O`.
            if math.isnan(score):
                continue
            if score > best_score:
                best_arm, best_score = name, score
        if best_arm is None:
            # Every draw was NaN (pathological state): fall back to a
            # deterministic choice rather than returning None.
            best_arm = arms[0]
        return best_arm

    def update(
        self, arm: str, task_type: Optional[str], reward: float, prompt_chars: int = 0
    ) -> None:
        """Fold one observed reward into the arm's posterior."""
        self.arm(arm).update(features(task_type, prompt_chars), reward)

    def mean_scores(
        self, task_type: Optional[str], prompt_chars: int = 0
    ) -> Dict[str, float]:
        """Posterior-mean score per arm (no exploration noise) for dashboards."""
        x = features(task_type, prompt_chars)
        scores: Dict[str, float] = {}
        for name, state in self.arms.items():
            try:
                mean = np.linalg.solve(state.a, state.b)
                value = float(mean @ x)
            except np.linalg.LinAlgError:
                # Singular A (pathological state): report the prior mean.
                value = 0.0
            scores[name] = value if math.isfinite(value) else 0.0
        return scores


def rewards_from_outcomes(success: bool, quality: Optional[float] = None) -> float:
    """Map ledger outcome fields to the [0, 1] reward LinTS consumes.

    Mirrors the Beta-Thompson quality bonus convention: a plain success is
    1.0; when the graded quality score exists it scales the reward so a
    selected-but-mediocre output earns less than a selected-and-good one.
    """
    if not success:
        return 0.0
    if quality is None:
        return 1.0
    return min(1.0, max(0.0, float(quality)))
