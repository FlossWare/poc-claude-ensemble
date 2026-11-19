"""Shared training infrastructure.

Everything here is deliberately small and dependency-light so the three model
modules stay readable:

* :class:`TrainingData` -- JSONL-backed features + labels with a **chronological**
  train/test split (no lookahead), class-balance reporting, and balanced sample
  weights for imbalanced ledgers.
* :class:`TrainedModel` -- a fitted estimator plus its feature encoder, metrics,
  feature importances, version stamp, and a 0600 joblib artifact.
* :class:`RetrainTrigger` -- "only retrain after N new rows" plus idempotence
  (a retrain over an unchanged data hash is skipped).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import joblib
import numpy as np

DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_MIN_NEW_ROWS = 10
#: Anything worse than this majority:minority ratio is called imbalanced.
IMBALANCE_RATIO = 1.5


class TrainingError(Exception):
    """Base error for training failures (insufficient or unusable data)."""


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | os.PathLike[str] | None) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from *path*, skipping blanks and malformed lines.

    A missing path yields nothing rather than raising: every caller is a
    cold-start path that must keep working before the ledger exists.
    """
    if not path:
        return
    try:
        handle = open(path, encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """Features, labels, and timestamps in ledger order.

    ``features`` holds raw feature dicts; each model module owns the encoder
    that turns them into a matrix. Keeping raw dicts here makes the artifact
    self-describing and lets the CLI print counts without knowing the model.
    """

    features: list[dict[str, Any]]
    labels: list[Any]
    timestamps: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    label_name: str = "label"
    label_source: str = ""
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        size = len(self.features)
        if len(self.labels) != size:
            raise ValueError(f"features ({size}) and labels ({len(self.labels)}) differ in length")
        if self.timestamps and len(self.timestamps) != size:
            raise ValueError(
                f"timestamps ({len(self.timestamps)}) and features ({size}) differ in length"
            )
        if not self.timestamps:
            self.timestamps = [""] * size
        if not self.metadata:
            self.metadata = [{} for _ in range(size)]
        elif len(self.metadata) != size:
            raise ValueError(f"metadata ({len(self.metadata)}) and features ({size}) differ in length")

    def __len__(self) -> int:
        return len(self.features)

    @property
    def is_empty(self) -> bool:
        return not self.features

    # -- balance ---------------------------------------------------------

    def class_balance(self) -> dict[str, int]:
        """Label counts, preserving first-seen order for stable reporting."""
        counts: dict[str, int] = {}
        for label in self.labels:
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def class_balance_report(self) -> dict[str, Any]:
        counts = self.class_balance()
        total = len(self.labels)
        shares = {key: (value / total if total else 0.0) for key, value in counts.items()}
        values = list(counts.values())
        ratio = (max(values) / min(values)) if values and min(values) > 0 else None
        return {
            "counts": counts,
            "shares": shares,
            "total": total,
            "imbalance_ratio": ratio,
            "is_imbalanced": ratio is not None and ratio >= IMBALANCE_RATIO,
        }

    def imbalance_ratio(self) -> float | None:
        return self.class_balance_report()["imbalance_ratio"]

    def is_imbalanced(self, threshold: float = IMBALANCE_RATIO) -> bool:
        ratio = self.imbalance_ratio()
        return ratio is not None and ratio >= threshold

    def sample_weights(self) -> np.ndarray:
        """Inverse-frequency weights, mean 1.0 (the standard balanced weighting)."""
        if not self.labels:
            return np.zeros(0, dtype=float)
        counts = self.class_balance()
        total = len(self.labels)
        classes = len(counts)
        weights = np.array(
            [total / (classes * counts[str(label)]) for label in self.labels], dtype=float
        )
        mean = weights.mean()
        return weights / mean if mean else weights

    # -- chronological split --------------------------------------------

    def _order(self) -> list[int]:
        """Indices sorted chronologically, stable within equal timestamps."""
        return sorted(range(len(self)), key=lambda index: (self.timestamps[index] or "", index))

    def split(self, train_fraction: float = DEFAULT_TRAIN_FRACTION) -> tuple["TrainingData", "TrainingData"]:
        """Split chronologically: the first ``train_fraction`` rows train, the rest test.

        Rows are ordered by timestamp first, so no test-period row can leak into
        training. With one row the train side keeps it and the test side is empty.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        order = self._order()
        n_train = int(len(order) * train_fraction)
        if len(order) and n_train == 0:
            n_train = 1
        if n_train > len(order):
            n_train = len(order)
        return self._take(order[:n_train]), self._take(order[n_train:])

    def _take(self, indices: Sequence[int]) -> "TrainingData":
        return TrainingData(
            features=[self.features[index] for index in indices],
            labels=[self.labels[index] for index in indices],
            timestamps=[self.timestamps[index] for index in indices],
            metadata=[self.metadata[index] for index in indices],
            label_name=self.label_name,
            label_source=self.label_source,
            feature_names=list(self.feature_names),
        )

    def sorted(self) -> "TrainingData":
        return self._take(self._order())

    # -- identity --------------------------------------------------------

    def data_hash(self) -> str:
        """Stable hash of features, labels, and their order (not metadata)."""
        payload = {
            "label_name": self.label_name,
            "label_source": self.label_source,
            "rows": [
                {"features": self.features[index], "label": self.labels[index]}
                for index in self._order()
            ],
        }
        return sha256_text(canonical_json(payload))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_score: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Accuracy / precision / recall / F1 / AUC for binary or multiclass labels.

    Missing scores or a single class in ``y_true`` leave AUC as ``None`` instead
    of raising: cold-start ledgers are the normal case here.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"n": 0, "accuracy": None, "precision": None, "recall": None,
                "f1": None, "auc": None}

    # Keep the caller's label values (ints compare against ints); only fall back
    # to first-seen order when the estimator did not report classes.
    labels = list(classes) if classes is not None else list(dict.fromkeys(y_true))
    binary = len(labels) <= 2
    average = "binary" if binary else "weighted"
    kwargs: dict[str, Any] = {"zero_division": 0}
    if binary:
        kwargs["pos_label"] = labels[-1] if len(labels) == 2 else labels[0]

    try:
        precision = precision_score(y_true, y_pred, average=average, **kwargs)
        recall = recall_score(y_true, y_pred, average=average, **kwargs)
        f1 = f1_score(y_true, y_pred, average=average, **kwargs)
    except ValueError:
        # e.g. labels passed as strings while y_true holds ints; fall back to
        # the values actually present rather than failing a cold-start run.
        precision = recall = f1 = None

    auc = None
    if y_score is not None:
        score = np.asarray(y_score)
        try:
            if binary and len(set(y_true)) == 2:
                auc = roc_auc_score(y_true, score[:, -1] if score.ndim > 1 else score)
            elif not binary and len(set(y_true)) > 2 and score.ndim > 1:
                auc = roc_auc_score(y_true, score, multi_class="ovr", average="macro")
        except (ValueError, TypeError):
            auc = None

    return {
        "n": len(y_true),
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "f1": _safe_float(f1),
        "auc": _safe_float(auc),
    }


# ---------------------------------------------------------------------------
# Trained model artifact
# ---------------------------------------------------------------------------


class FeatureEncoder:
    """Minimal protocol: subclasses turn raw dicts into a numeric matrix.

    Subclasses must implement ``fit``, ``transform``, and expose
    ``feature_names``. The object is persisted in the artifact so prediction
    after ``load`` uses exactly the training-time encoding.
    """

    feature_names: list[str]

    def fit(self, features: Sequence[Mapping[str, Any]]) -> "FeatureEncoder":  # pragma: no cover
        raise NotImplementedError

    def transform(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class TrainedModel:
    """A fitted estimator, its encoder, metrics, and version stamp."""

    name: str
    estimator: Any
    encoder: FeatureEncoder
    label_name: str = "label"
    label_source: str = ""
    trained_at: str = ""
    data_hash: str = ""
    n_train: int = 0
    n_test: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # -- inference -------------------------------------------------------

    def _matrix(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.encoder.transform(features)

    def predict(self, features: Sequence[Mapping[str, Any]]) -> list[Any]:
        return list(self.estimator.predict(self._matrix(features)))

    def predict_proba(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self._matrix(features)))

    @property
    def classes(self) -> list[Any]:
        return list(getattr(self.estimator, "classes_", []))

    def positive_score(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Probability of the last class (the positive class for binary models)."""
        proba = self.predict_proba(features)
        return proba[:, -1] if proba.ndim > 1 else proba

    # -- evaluation ------------------------------------------------------

    def evaluate(self, data: TrainingData) -> dict[str, Any]:
        if data.is_empty:
            return {"n": 0, "accuracy": None, "precision": None, "recall": None,
                    "f1": None, "auc": None}
        matrix = self._matrix(data.features)
        predictions = self.estimator.predict(matrix)
        scores = None
        try:
            scores = np.asarray(self.estimator.predict_proba(matrix))
        except (AttributeError, NotImplementedError):
            scores = None
        return evaluate_predictions(data.labels, predictions, scores, self.classes)

    # -- interpretability ------------------------------------------------

    def feature_importances(self) -> list[tuple[str, float]]:
        """Tree importances or linear coefficients, paired with feature names."""
        names = list(self.encoder.feature_names)
        estimator = self.estimator
        values: np.ndarray | None = None
        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None:
            values = np.asarray(importances, dtype=float)
        else:
            coef = getattr(estimator, "coef_", None)
            if coef is not None:
                coef = np.asarray(coef, dtype=float)
                values = coef[0] if coef.ndim == 1 else np.abs(coef).mean(axis=0)
        if values is None:
            return []
        if len(names) != len(values):
            names = [f"feature_{index}" for index in range(len(values))]
        return sorted(zip(names, (float(v) for v in values)), key=lambda item: item[1], reverse=True)

    def coefficient_table(self) -> dict[str, dict[str, float]]:
        """Per-class linear coefficients (``LogisticRegression`` inspection)."""
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            return {}
        coef = np.asarray(coef, dtype=float)
        classes = [str(value) for value in self.classes]
        names = list(self.encoder.feature_names)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.shape[0] == 1 and len(classes) != 1:
            # Binary linear models store one coefficient row, for classes_[1];
            # the other class is its negation.
            if not classes:
                classes = ["negative", "positive"]
            elif len(classes) == 1:
                classes = [classes[0], "other"]
            coef = np.vstack([-coef[0], coef[0]])
        elif not classes:
            classes = ["positive"]
        table: dict[str, dict[str, float]] = {}
        for row_index, class_name in enumerate(classes):
            if row_index >= coef.shape[0]:
                break
            table[class_name] = {
                name: float(coef[row_index, index])
                for index, name in enumerate(names[: coef.shape[1]])
            }
        return table

    # -- persistence -----------------------------------------------------

    def version_stamp(self) -> dict[str, Any]:
        import sklearn

        return {
            "model": self.name,
            "sklearn_version": sklearn.__version__,
            "trained_at": self.trained_at or utc_now(),
            "training_data_hash": self.data_hash,
            "label_name": self.label_name,
            "label_source": self.label_source,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metrics": self.metrics,
            "extras": self.extras,
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Persist with mode 0600 (the artifact may embed ledger-derived data)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        bundle = {
            "format": "learning.train.v1",
            "stamp": self.version_stamp(),
            "estimator": self.estimator,
            "encoder": self.encoder,
        }
        joblib.dump(bundle, destination)
        os.chmod(destination, stat.S_IRUSR | stat.S_IWUSR)
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str], name: str | None = None) -> "TrainedModel":
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or bundle.get("format") != "learning.train.v1":
            raise ValueError(f"not a learning.train artifact: {path}")
        stamp = bundle.get("stamp") or {}
        return cls(
            name=name or stamp.get("model") or Path(path).stem,
            estimator=bundle["estimator"],
            encoder=bundle["encoder"],
            label_name=stamp.get("label_name", "label"),
            label_source=stamp.get("label_source", ""),
            trained_at=stamp.get("trained_at", ""),
            data_hash=stamp.get("training_data_hash", ""),
            n_train=int(stamp.get("n_train") or 0),
            n_test=int(stamp.get("n_test") or 0),
            metrics=stamp.get("metrics") or {},
            extras=stamp.get("extras") or {},
        )

    def status(self) -> dict[str, Any]:
        """Compact status row for the CLI (no estimator internals)."""
        return {
            "name": self.name,
            "trained_at": self.trained_at,
            "data_hash": self.data_hash,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metrics": self.metrics,
            "label_source": self.label_source,
        }


# ---------------------------------------------------------------------------
# Retrain trigger
# ---------------------------------------------------------------------------


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    new_rows: int
    total_rows: int
    data_hash: str = ""


class RetrainTrigger:
    """Gate retraining on new data volume and data identity.

    State lives next to the artifact as JSON (0600). A decision is a pure read;
    only :meth:`mark_trained` advances the state, so a failed fit cannot look
    like a completed retrain.
    """

    def __init__(
        self,
        name: str,
        *,
        min_new_rows: int = DEFAULT_MIN_NEW_ROWS,
        state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.name = name
        self.min_new_rows = max(0, int(min_new_rows))
        self.state_dir = Path(state_dir) if state_dir is not None else None

    @property
    def state_path(self) -> Path:
        if self.state_dir is not None:
            return self.state_dir / f"retrain-{self.name}.json"
        from learning.train.paths import train_dir

        return train_dir() / f"retrain-{self.name}.json"

    def _load_state(self) -> dict[str, Any]:
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                value = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return value if isinstance(value, dict) else {}

    def decide(self, data: TrainingData, *, force: bool = False) -> RetrainDecision:
        state = self._load_state()
        data_hash = data.data_hash()
        total_rows = len(data)
        last_rows = int(state.get("rows") or 0)
        if force:
            return RetrainDecision(True, "forced", total_rows - last_rows, total_rows, data_hash)
        if not total_rows:
            return RetrainDecision(False, "no training data", 0, 0, data_hash)
        if state.get("data_hash") == data_hash:
            return RetrainDecision(False, "data hash unchanged", 0, total_rows, data_hash)
        new_rows = max(0, total_rows - last_rows)
        if new_rows < self.min_new_rows:
            needed = self.min_new_rows - new_rows
            return RetrainDecision(
                False,
                f"need {needed} more new row(s) (min {self.min_new_rows})",
                new_rows,
                total_rows,
                data_hash,
            )
        return RetrainDecision(True, f"{new_rows} new row(s)", new_rows, total_rows, data_hash)

    def mark_trained(self, data: TrainingData) -> None:
        self.mark(data_hash=data.data_hash(), rows=len(data))

    def mark(self, *, data_hash: str, rows: int) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {"model": self.name, "data_hash": data_hash, "rows": int(rows), "trained_at": utc_now()}
        fd = os.open(self.state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state(self) -> dict[str, Any]:
        return self._load_state()


def make_timestamp(value: Any) -> str:
    """Best-effort ISO-ish timestamp string used only for chronological ordering."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return ""
    return str(value)
