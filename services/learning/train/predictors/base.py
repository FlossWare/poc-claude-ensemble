"""Shared training, evaluation, persistence and status plumbing.

Every predictor is a small scikit-learn estimator plus a
:class:`~learning.train.predictors.features.FeatureEncoder`.  The base class
keeps three promises that matter for this work:

* metrics are cross-validated, never in-sample unless the data is too small to
  split (and that exception is recorded in ``evaluation`` and ``warnings``);
* artifacts carry their provenance (ledger path, sample counts, feature names,
  library versions, warnings) next to the model;
* a sidecar JSON metadata file lets the status command report on a model
  without unpickling it.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from learning.train.predictors.features import (
    OUTCOME_FIELDS,
    FeatureEncoder,
    ReviewFeatures,
    summarize,
)

ARTIFACT_MAGIC = "claude-ensemble-routing-predictor"
ARTIFACT_SCHEMA_VERSION = 1
RANDOM_STATE = 42
MIN_TRAINING_ROWS = 2

PREDICTOR_DIR_ENV = "POC_CLAUDE_ENSEMBLE_PREDICTOR_DIR"
LEARNING_DIR_ENV = "POC_CLAUDE_ENSEMBLE_LEARNING_DIR"


class PredictorError(RuntimeError):
    """Base class for predictor failures that callers can report cleanly."""


class InsufficientDataError(PredictorError):
    """Raised when there are too few labelled rows to fit a model."""


class NotTrainedError(PredictorError):
    """Raised when prediction is attempted before fitting or loading."""


def predictor_dir() -> Path:
    """Directory holding persisted predictor artifacts."""
    configured = os.environ.get(PREDICTOR_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    learning = os.environ.get(LEARNING_DIR_ENV)
    base = (
        Path(learning).expanduser()
        if learning
        else Path.home() / ".FlossWare" / "claude-ensemble-state" / "data" / "learning"
    )
    return base / "predictors"


def default_model_path(kind: str) -> Path:
    return predictor_dir() / f"{kind}.pkl"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    sidecar = path.with_suffix(".json")
    try:
        with open(sidecar, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


class BasePredictor:
    """Common fit/predict/evaluate/save/load implementation."""

    KIND: str = ""
    DISPLAY: str = ""
    TARGET: str = ""
    TASK: str = "regression"  # "regression" | "classification"
    UNIT: str = ""
    LABEL_DEFINITION: str = ""

    def __init__(
        self,
        estimator: Any = None,
        encoder: Optional[FeatureEncoder] = None,
        constant: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.estimator = estimator
        self.encoder = encoder
        self._constant = constant
        self.metadata: Dict[str, Any] = dict(metadata or {})

    # -- subclass hooks ----------------------------------------------------
    def _make_estimator(self) -> Any:
        raise NotImplementedError

    def _row_label(self, row: ReviewFeatures) -> float:
        raise NotImplementedError

    def _postprocess(self, value: float) -> float:
        return value

    # -- properties --------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.estimator is not None or self._constant is not None

    @property
    def constant(self) -> Optional[float]:
        return self._constant

    @property
    def feature_names(self) -> List[str]:
        return self.encoder.feature_names if self.encoder is not None else []

    # -- training ----------------------------------------------------------
    def fit(
        self,
        rows: Sequence[ReviewFeatures],
        *,
        evaluate: bool = True,
        ledger: Optional[os.PathLike | str] = None,
    ) -> Dict[str, Any]:
        """Fit on extracted review rows and return the training metadata.

        Metrics are computed by cross-validation on the same rows.  The model
        persisted by :meth:`save` is then fitted on all rows.  Cross-validated
        metrics are the honest estimate; the final model simply uses every
        labelled example.
        """
        rows = list(rows)
        if len(rows) < MIN_TRAINING_ROWS:
            raise InsufficientDataError(
                f"{self.KIND}: need at least {MIN_TRAINING_ROWS} worker rows, got {len(rows)}"
            )

        encoder = FeatureEncoder.fit(rows)
        matrix = encoder.encode_rows(rows)
        labels = np.asarray([self._row_label(row) for row in rows], dtype=np.float64)

        estimator: Any = None
        constant: Optional[float] = None
        if self.TASK == "classification":
            classes = np.unique(labels)
            if len(classes) == 1:
                # A single observed class is not a scikit-learn failure; it is
                # a real state of young ledgers.  Persist the base rate instead
                # of refusing to answer routing questions.
                constant = float(classes[0])
            else:
                estimator = self._make_estimator()
                estimator.fit(matrix, labels)
        else:
            estimator = self._make_estimator()
            estimator.fit(matrix, labels)

        self.estimator = estimator
        self._constant = constant
        self.encoder = encoder

        metrics: Dict[str, Any] = {}
        if evaluate:
            metrics = self.evaluate(rows)

        self.metadata = self._build_metadata(rows, ledger=ledger, metrics=metrics)
        return self.metadata

    def _build_metadata(
        self,
        rows: Sequence[ReviewFeatures],
        *,
        ledger: Optional[os.PathLike | str],
        metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary = summarize(rows)
        warnings = self._warnings(summary, metrics)
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": self.KIND,
            "display": self.DISPLAY,
            "target": self.TARGET,
            "task": self.TASK,
            "unit": self.UNIT,
            "label_definition": self.LABEL_DEFINITION,
            "feature_set": "precall",
            "numeric_features": list(self.encoder.numeric_features) if self.encoder else [],
            "feature_names": self.feature_names,
            "models": summary["models"],
            "task_types": summary["task_types"],
            "samples": summary["samples"],
            "review_episodes": summary["review_episodes"],
            "successes": summary["successes"],
            "failures": summary["failures"],
            "first_recorded_at": summary["first_recorded_at"],
            "last_recorded_at": summary["last_recorded_at"],
            "ledger": str(ledger) if ledger is not None else None,
            "trained_at": utc_now(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "metrics": dict(metrics),
            "warnings": warnings,
            "constant_prediction": self._constant,
        }

    def _warnings(self, summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> List[str]:
        warnings: List[str] = []
        samples = int(summary["samples"])
        episodes = int(summary["review_episodes"])
        if samples < 100:
            warnings.append(
                f"small sample: {samples} worker results from {episodes} review episodes; "
                "metrics are unstable and must not be treated as production accuracy"
            )
        if episodes and episodes < 30:
            warnings.append(
                f"only {episodes} review episodes: multiple worker rows share the same task, "
                "so rows are not independent observations"
            )
        if self.TASK == "classification":
            failures = int(summary["failures"])
            if 0 < failures < 10:
                warnings.append(
                    f"rare failure class ({failures} positives of {samples}); "
                    "AUC/accuracy are unreliable and the classifier is prone to predicting the majority class"
                )
            auc = metrics.get("roc_auc")
            if auc is not None and auc <= 0.5:
                warnings.append(
                    f"failure classifier ROC AUC is {auc:.3f}; no better than chance on "
                    "pre-call features in this sample"
                )
            accuracy = metrics.get("accuracy")
            baseline = metrics.get("baseline_accuracy")
            if accuracy is not None and baseline is not None and accuracy < baseline:
                warnings.append(
                    f"failure classifier accuracy is {accuracy:.3f}, below the "
                    f"majority-class baseline of {baseline:.3f}"
                )
        else:
            r2 = metrics.get("r2")
            if r2 is not None and r2 < 0:
                warnings.append(
                    f"cross-validated R2 is {r2:.3f}; the model is worse than predicting the mean"
                )
        if metrics.get("evaluation", "").startswith("in-sample"):
            warnings.append(
                "metrics are in-sample because the data is too small for a meaningful split"
            )
        warnings.append(
            "model inputs are pre-call only; outcome fields ("
            + ", ".join(OUTCOME_FIELDS)
            + ") were extracted but excluded to prevent target leakage"
        )
        return warnings

    # -- evaluation --------------------------------------------------------
    def evaluate(self, rows: Sequence[ReviewFeatures]) -> Dict[str, Any]:
        rows = list(rows)
        if len(rows) < MIN_TRAINING_ROWS:
            raise InsufficientDataError(
                f"{self.KIND}: need at least {MIN_TRAINING_ROWS} rows to evaluate"
            )
        encoder = self.encoder or FeatureEncoder.fit(rows)
        matrix = encoder.encode_rows(rows)
        labels = np.asarray([self._row_label(row) for row in rows], dtype=np.float64)
        if self.TASK == "classification":
            return self._evaluate_classification(matrix, labels)
        return self._evaluate_regression(matrix, labels)

    def _evaluate_regression(self, matrix: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        n = len(labels)
        folds = min(5, n)
        if folds >= 2:
            predictions = np.empty(n, dtype=np.float64)
            splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
            for train_index, test_index in splitter.split(matrix):
                estimator = clone(self._make_estimator())
                estimator.fit(matrix[train_index], labels[train_index])
                predictions[test_index] = estimator.predict(matrix[test_index])
            evaluation = f"{folds}-fold cross-validation"
        else:  # pragma: no cover - guarded by MIN_TRAINING_ROWS
            predictions = labels.copy()
            evaluation = "in-sample (insufficient rows for cross-validation)"

        constant_target = bool(np.allclose(labels, labels[0]))
        metrics: Dict[str, Any] = {
            "evaluation": evaluation,
            "n": n,
            "folds": folds if folds >= 2 else 0,
            "r2": None if constant_target else float(r2_score(labels, predictions)),
            "mae": float(mean_absolute_error(labels, predictions)),
            "rmse": _rmse(labels, predictions),
            "baseline_mae": float(mean_absolute_error(labels, np.full(n, labels.mean()))),
            "baseline_r2": 0.0,
            "target_mean": float(labels.mean()),
            "target_std": float(labels.std()),
        }
        return metrics

    def _positive_probabilities(self, estimator: Any, matrix: np.ndarray) -> np.ndarray:
        if not hasattr(estimator, "predict_proba"):
            return estimator.predict(matrix).astype(np.float64)
        probabilities = estimator.predict_proba(matrix)
        classes = list(getattr(estimator, "classes_", []))
        if 1.0 in classes:
            return probabilities[:, classes.index(1.0)]
        return np.zeros(matrix.shape[0], dtype=np.float64)

    def _evaluate_classification(self, matrix: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        n = len(labels)
        classes, counts = np.unique(labels, return_counts=True)
        positives = int(np.sum(labels == 1.0))
        negatives = int(n - positives)
        majority = max(positives, negatives) / n if n else 0.0

        max_folds = min(5, n, int(counts.min()) if len(counts) else 0)
        if len(classes) < 2:
            predictions = np.full(n, float(classes[0]), dtype=np.float64)
            evaluation = "in-sample (single observed class)"
            folds = 0
        elif max_folds >= 2:
            predictions = np.zeros(n, dtype=np.float64)
            splitter = StratifiedKFold(
                n_splits=max_folds, shuffle=True, random_state=RANDOM_STATE
            )
            for train_index, test_index in splitter.split(matrix, labels):
                train_labels = labels[train_index]
                if len(np.unique(train_labels)) < 2:
                    predictions[test_index] = float(train_labels.mean())
                else:
                    estimator = clone(self._make_estimator())
                    estimator.fit(matrix[train_index], train_labels)
                    predictions[test_index] = self._positive_probabilities(
                        estimator, matrix[test_index]
                    )
            evaluation = f"{max_folds}-fold stratified cross-validation"
            folds = max_folds
        else:
            predictions = np.full(n, float(labels.mean()), dtype=np.float64)
            evaluation = "in-sample (insufficient class diversity for cross-validation)"
            folds = 0

        binary = (predictions >= 0.5).astype(np.int64)
        truth = labels.astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(truth, binary, labels=[0, 1]).ravel()
        auc: Optional[float] = None
        if len(classes) >= 2 and len(np.unique(truth)) == 2:
            auc = float(roc_auc_score(truth, predictions))
        return {
            "evaluation": evaluation,
            "n": n,
            "folds": folds,
            "accuracy": float(accuracy_score(truth, binary)),
            "roc_auc": auc,
            "precision": float(precision_score(truth, binary, zero_division=0)),
            "recall": float(recall_score(truth, binary, zero_division=0)),
            "f1": float(f1_score(truth, binary, zero_division=0)),
            "positive_rate": positives / n if n else 0.0,
            "baseline_accuracy": float(majority),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "target_mean": float(labels.mean()),
            "target_std": float(labels.std()),
        }

    # -- prediction --------------------------------------------------------
    def predict_one(self, model_id: str, task_features: Optional[Mapping[str, Any]] = None) -> float:
        """Predict for one model + pre-call task feature mapping."""
        if not self.is_fitted or self.encoder is None:
            raise NotTrainedError(f"{self.KIND} predictor is not trained")
        matrix = self.encoder.encode(model_id, task_features).reshape(1, -1)
        if self._constant is not None:
            value = float(self._constant)
        elif self.TASK == "classification":
            value = float(self._positive_probabilities(self.estimator, matrix)[0])
        else:
            value = float(self.estimator.predict(matrix)[0])
        return self._postprocess(value)

    def predict(self, model_id: str, task_features: Optional[Mapping[str, Any]] = None) -> float:
        """Alias for :meth:`predict_one`."""
        return self.predict_one(model_id, task_features)

    def feature_usage(self, task_features: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Report which request fields were used and which were ignored."""
        features = dict(task_features or {})
        numeric = set(self.encoder.numeric_features) if self.encoder else set()
        used: Dict[str, Any] = {}
        for name in numeric:
            if name in features:
                used[name] = features[name]
        if "task_type" in features:
            used["task_type"] = features["task_type"]
        elif "subject" in features:
            used["subject"] = features["subject"]
        ignored = sorted(
            key
            for key in features
            if key not in numeric and key not in {"task_type", "subject", "model", "model_id"}
        )
        return {"used": used, "ignored": ignored}

    # -- persistence -------------------------------------------------------
    def save(self, path: Optional[os.PathLike | str] = None) -> Path:
        """Persist the estimator plus a plain-JSON metadata sidecar.

        Pickle artifacts must only be loaded from trusted locations: loading a
        pickle executes code.  The sidecar exists so status checks never need
        to unpickle anything.
        """
        if not self.is_fitted:
            raise NotTrainedError(f"{self.KIND} predictor is not trained")
        destination = Path(path).expanduser() if path is not None else default_model_path(self.KIND)
        payload = {
            "magic": ARTIFACT_MAGIC,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": self.KIND,
            "estimator": self.estimator,
            "constant": self._constant,
            "encoder": self.encoder.to_dict() if self.encoder else None,
            "metadata": self.metadata,
        }
        atomic_write(destination, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        sidecar = destination.with_suffix(".json")
        atomic_write(
            sidecar,
            (json.dumps(self.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        return destination

    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "BasePredictor":
        source = Path(path).expanduser() if path is not None else default_model_path(cls.KIND)
        try:
            with open(source, "rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError as exc:
            raise NotTrainedError(f"{cls.KIND} artifact not found: {source}") from exc
        if not isinstance(payload, dict) or payload.get("magic") != ARTIFACT_MAGIC:
            raise PredictorError(f"{source} is not a routing-predictor artifact")
        if payload.get("kind") != cls.KIND:
            raise PredictorError(
                f"{source} holds a {payload.get('kind')!r} artifact, expected {cls.KIND!r}"
            )
        encoder_document = payload.get("encoder")
        encoder = FeatureEncoder.from_dict(encoder_document) if encoder_document else None
        return cls(
            estimator=payload.get("estimator"),
            encoder=encoder,
            constant=payload.get("constant"),
            metadata=payload.get("metadata") or {},
        )

    @classmethod
    def status(cls, path: Optional[os.PathLike | str] = None) -> Dict[str, Any]:
        """Report artifact availability and metrics without unpickling."""
        source = Path(path).expanduser() if path is not None else default_model_path(cls.KIND)
        status: Dict[str, Any] = {
            "kind": cls.KIND,
            "display": cls.DISPLAY,
            "available": False,
            "path": str(source),
        }
        if not source.exists():
            status["reason"] = "artifact missing"
            return status
        document = read_sidecar(source)
        if document is None:
            status["reason"] = "artifact present but metadata sidecar is missing or unreadable"
            return status
        status.update(
            {
                "available": True,
                "trained_at": document.get("trained_at"),
                "samples": document.get("samples"),
                "review_episodes": document.get("review_episodes"),
                "target": document.get("target"),
                "feature_set": document.get("feature_set"),
                "numeric_features": document.get("numeric_features"),
                "metrics": document.get("metrics") or {},
                "warnings": document.get("warnings") or [],
            }
        )
        return status
