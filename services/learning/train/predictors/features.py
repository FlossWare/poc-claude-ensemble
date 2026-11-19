"""Feature extraction for pre-call routing predictors.

The only real source of labelled worker outcomes in this repository is the
review ledger::

    ~/.FlossWare/ledger/reviews/reviews.jsonl
    (override: POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER)

Each ``kind == "worker_result"`` row is one worker invocation inside one review
episode.  A review episode normally has one row per worker, so rows are *not*
independent observations -- several rows share the same task and subject.

Leakage policy
--------------
The ledger records both pre-call inputs and post-call outcomes in the same row.
Cost, latency, failure, stop reason, output length and the derived throughput
values are only known *after* the call.  Feeding them to a model that is meant
to choose an arm *before* the call produces spectacular but meaningless
cross-validation scores.

``extract_features`` therefore returns every requested field, while
``FeatureEncoder`` (used for training) consumes only fields that are genuinely
available before dispatch:

* ``model_id`` (one-hot)
* ``task_type`` (one-hot, derived from the review subject's file suffix)
* ``input_tokens`` (prompt size -- the pre-call length signal)

``text_length`` is extracted, but in this ledger it is the length of the
worker's *output* text, not the task input.  It is therefore treated as an
outcome field and is not used as a model input.  Callers may still pass
``text_length`` in a prediction request; it is reported back as ignored rather
than silently accepted.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1
WORKER_KIND = "worker_result"

DEFAULT_REVIEW_LEDGER = Path.home() / ".FlossWare" / "ledger" / "reviews" / "reviews.jsonl"
REVIEW_LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"

#: Numeric inputs available *before* the model call.  ``input_tokens`` is the
#: prompt size.  Everything else in ``ReviewFeatures`` is a post-call outcome.
PRECALL_NUMERIC_FEATURES: Tuple[str, ...] = ("input_tokens",)

#: Fields extracted from the ledger but deliberately excluded from model input.
OUTCOME_FIELDS: Tuple[str, ...] = (
    "output_tokens",
    "text_length",
    "stop_reason",
    "latency_ms",
    "cost_usd",
    "tokens_per_second",
    "cost_per_token",
)

TASK_TYPE_BY_SUFFIX: Dict[str, str] = {
    ".py": "code",
    ".pyi": "code",
    ".js": "code",
    ".cjs": "code",
    ".mjs": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".h": "code",
    ".cc": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".rb": "code",
    ".php": "code",
    ".cs": "code",
    ".kt": "code",
    ".swift": "code",
    ".scala": "code",
    ".yml": "config",
    ".yaml": "config",
    ".toml": "config",
    ".ini": "config",
    ".cfg": "config",
    ".conf": "config",
    ".json": "config",
    ".xml": "config",
    ".properties": "config",
    ".j2": "template",
    ".jinja": "template",
    ".jinja2": "template",
    ".tmpl": "template",
    ".mustache": "template",
    ".hbs": "template",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "shell",
    ".md": "docs",
    ".markdown": "docs",
    ".rst": "docs",
    ".txt": "docs",
    ".adoc": "docs",
}

UNKNOWN_CATEGORY = "__unseen__"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def ledger_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Resolve the review ledger path (argument > environment > default)."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get(REVIEW_LEDGER_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_REVIEW_LEDGER


def iter_ledger_rows(path: Optional[os.PathLike | str] = None) -> Iterator[Dict[str, Any]]:
    """Yield well-formed JSON objects from a JSONL ledger.

    Blank and malformed lines are skipped: a single corrupt line must not make
    historical training data unusable.  A missing file yields nothing.
    """
    ledger = ledger_path(path)
    try:
        handle = open(ledger, "r", encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(row, dict):
                yield row


def worker_rows(path: Optional[os.PathLike | str] = None) -> List[Dict[str, Any]]:
    """Return only ``worker_result`` rows (arbiter rows are not routable arms)."""
    return [row for row in iter_ledger_rows(path) if row.get("kind") == WORKER_KIND]


def derive_task_type(subject: Optional[str] = None, explicit: Optional[str] = None) -> str:
    """Normalise an explicit task type, else derive one from a file suffix.

    Unknown suffixes become ``"other"``.  The return value is always a short
    lower-case token so it can be one-hot encoded deterministically.
    """
    if explicit is not None:
        token = str(explicit).strip().lower().replace(" ", "_").replace("-", "_")
        return token or "other"
    if not subject:
        return "other"
    suffix = Path(str(subject)).suffix.lower()
    return TASK_TYPE_BY_SUFFIX.get(suffix, "other")


def _number(value: Any, default: float = 0.0) -> float:
    """Coerce a ledger value to float; booleans and junk become ``default``."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value, float(default))
    try:
        return int(number)
    except (OverflowError, ValueError):
        return default


def _output_text_length(row: Mapping[str, Any]) -> int:
    """Length of the worker output, preferring the pre-truncation length."""
    text = row.get("text")
    length = len(text) if isinstance(text, str) else 0
    if row.get("truncated"):
        original = _integer(row.get("original_length"), 0)
        if original > length:
            length = original
    return length


def is_success(row: Mapping[str, Any]) -> bool:
    """Usable worker output: normal stop, non-empty text, no error.

    ``end_turn`` matches the review ledger's canonical successful stop reason.
    A truncated (``max_tokens``) or empty response is not usable output for
    downstream review/consensus even though the HTTP call itself succeeded.
    An ``error`` key with a truthy value also disqualifies the row; an empty
    string or ``None`` does not.
    """
    text = row.get("text")
    has_text = isinstance(text, str) and bool(text.strip())
    return bool(row.get("stop_reason") == "end_turn" and has_text and not row.get("error"))


@dataclass(frozen=True)
class ReviewFeatures:
    """Per-worker-per-review features extracted from one ledger row."""

    review_id: str
    model_id: str
    task_type: str
    input_tokens: int
    output_tokens: int
    text_length: int
    stop_reason: str
    latency_ms: float
    cost_usd: float
    tokens_per_second: float
    cost_per_token: float
    success: bool
    recorded_at: str
    subject: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_row(row: Mapping[str, Any]) -> ReviewFeatures:
    """Build :class:`ReviewFeatures` from one review-ledger row."""
    model_id = str(row.get("model") or "unknown")
    subject = str(row.get("subject") or "")
    task_type = derive_task_type(subject, row.get("task_type"))
    input_tokens = max(0, _integer(row.get("input_tokens")))
    output_tokens = max(0, _integer(row.get("output_tokens")))
    latency_ms = max(0.0, _number(row.get("latency_ms")))
    cost_usd = max(0.0, _number(row.get("cost_usd")))
    text_length = _output_text_length(row)

    if latency_ms > 0:
        tokens_per_second = output_tokens / (latency_ms / 1000.0)
    else:
        tokens_per_second = 0.0
    total_tokens = input_tokens + output_tokens
    cost_per_token = cost_usd / total_tokens if total_tokens > 0 else 0.0

    return ReviewFeatures(
        review_id=str(row.get("review_id") or ""),
        model_id=model_id,
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        text_length=text_length,
        stop_reason=str(row.get("stop_reason") or "unknown"),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        tokens_per_second=tokens_per_second,
        cost_per_token=cost_per_token,
        success=is_success(row),
        recorded_at=str(row.get("recorded_at") or row.get("observed_at") or ""),
        subject=subject,
    )


def extract_features(
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    path: Optional[os.PathLike | str] = None,
) -> List[ReviewFeatures]:
    """Extract features from supplied rows, or from the review ledger.

    Supplied rows are filtered to worker results as well, so a caller can pass
    the whole ledger without accidentally training on arbiter records.
    """
    source = worker_rows(path) if rows is None else [
        row for row in rows if row.get("kind") == WORKER_KIND
    ]
    return [extract_row(row) for row in source]


def summarize(rows: Sequence[ReviewFeatures]) -> Dict[str, Any]:
    """Small provenance summary used by the CLI and stored in artifacts."""
    review_ids = sorted({row.review_id for row in rows if row.review_id})
    timestamps = sorted(row.recorded_at for row in rows if row.recorded_at)
    failures = sum(1 for row in rows if not row.success)
    return {
        "samples": len(rows),
        "review_episodes": len(review_ids),
        "models": sorted({row.model_id for row in rows}),
        "task_types": sorted({row.task_type for row in rows}),
        "failures": failures,
        "successes": len(rows) - failures,
        "positive_rate": failures / len(rows) if rows else 0.0,
        "first_recorded_at": timestamps[0] if timestamps else None,
        "last_recorded_at": timestamps[-1] if timestamps else None,
        "mean_input_tokens": float(np.mean([row.input_tokens for row in rows])) if rows else 0.0,
        "mean_latency_ms": float(np.mean([row.latency_ms for row in rows])) if rows else 0.0,
        "mean_cost_usd": float(np.mean([row.cost_usd for row in rows])) if rows else 0.0,
    }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class FeatureEncoder:
    """One-hot model/task categories plus numeric pre-call features.

    Unseen categories at prediction time are mapped to an explicit
    ``__unseen__`` indicator rather than silently collapsing to an all-zero
    vector, which would be indistinguishable from a known category with zero
    numeric inputs.
    """

    def __init__(
        self,
        model_categories: Sequence[str],
        task_categories: Sequence[str],
        numeric_features: Sequence[str] = PRECALL_NUMERIC_FEATURES,
    ) -> None:
        self.model_categories: Tuple[str, ...] = tuple(sorted(set(model_categories)))
        self.task_categories: Tuple[str, ...] = tuple(sorted(set(task_categories)))
        self.numeric_features: Tuple[str, ...] = tuple(numeric_features)
        if not self.numeric_features:
            raise ValueError("at least one numeric feature is required")

    @classmethod
    def fit(
        cls,
        rows: Sequence[ReviewFeatures],
        numeric_features: Sequence[str] = PRECALL_NUMERIC_FEATURES,
    ) -> "FeatureEncoder":
        return cls(
            model_categories=[row.model_id for row in rows],
            task_categories=[row.task_type for row in rows],
            numeric_features=numeric_features,
        )

    @property
    def feature_names(self) -> List[str]:
        names = [f"model={category}" for category in self.model_categories]
        names.append(f"model={UNKNOWN_CATEGORY}")
        names.extend(f"task_type={category}" for category in self.task_categories)
        names.append(f"task_type={UNKNOWN_CATEGORY}")
        names.extend(self.numeric_features)
        return names

    def _encode_categories(
        self, model_id: str, task_type: str
    ) -> Tuple[List[float], List[float]]:
        model_vector = [0.0] * (len(self.model_categories) + 1)
        if model_id in self.model_categories:
            model_vector[self.model_categories.index(model_id)] = 1.0
        else:
            model_vector[-1] = 1.0

        task_vector = [0.0] * (len(self.task_categories) + 1)
        if task_type in self.task_categories:
            task_vector[self.task_categories.index(task_type)] = 1.0
        else:
            task_vector[-1] = 1.0
        return model_vector, task_vector

    def encode(self, model_id: str, task_features: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        """Encode a prediction request.

        ``task_features`` must be a mapping.  Numeric values that are present
        but non-numeric raise ``ValueError`` -- silently treating
        ``input_tokens="lots"`` as zero would hide caller bugs.
        """
        features: Mapping[str, Any] = task_features or {}
        if not isinstance(features, Mapping):
            raise ValueError("task_features must be a JSON object / mapping")
        if not model_id:
            raise ValueError("model_id is required")

        task_type = features.get("task_type")
        if task_type is None:
            task_type = derive_task_type(features.get("subject"))
        else:
            task_type = derive_task_type(explicit=task_type)

        model_vector, task_vector = self._encode_categories(str(model_id), str(task_type))
        numeric: List[float] = []
        for name in self.numeric_features:
            value = features.get(name, 0.0)
            if value is None:
                value = 0.0
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"numeric feature {name!r} must be a number, got {value!r}")
            numeric.append(float(value))

        return np.asarray(model_vector + task_vector + numeric, dtype=np.float64)

    def encode_row(self, row: ReviewFeatures) -> np.ndarray:
        model_vector, task_vector = self._encode_categories(row.model_id, row.task_type)
        numeric = [float(getattr(row, name)) for name in self.numeric_features]
        return np.asarray(model_vector + task_vector + numeric, dtype=np.float64)

    def encode_rows(self, rows: Sequence[ReviewFeatures]) -> np.ndarray:
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack([self.encode_row(row) for row in rows])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_categories": list(self.model_categories),
            "task_categories": list(self.task_categories),
            "numeric_features": list(self.numeric_features),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "FeatureEncoder":
        encoder = cls(
            model_categories=document["model_categories"],
            task_categories=document["task_categories"],
            numeric_features=document["numeric_features"],
        )
        if list(document.get("feature_names", [])) != encoder.feature_names:
            raise ValueError("feature encoder artifact has inconsistent feature names")
        return encoder


def design_matrix(
    rows: Sequence[ReviewFeatures],
    encoder: FeatureEncoder,
) -> np.ndarray:
    """Convenience wrapper: encoded pre-call design matrix for training rows."""
    return encoder.encode_rows(rows)
