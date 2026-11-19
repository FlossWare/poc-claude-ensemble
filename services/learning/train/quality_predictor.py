"""Quality predictor: will the arbiter SELECT this worker's output?

Real gradient-boosted trees (``GradientBoostingClassifier``) over per-worker
features from the review ledger. The authoritative label is the arbiter's
``SELECTED:`` line matched against the worker's model.

**Real-ledger caveat.** The live ledger's arbiter rows were written before the
arbiter prompt asked for a ``SELECTED:`` line, so on 2026-11-19 the live ledger
contains zero ``SELECTED:`` lines. When authoritative labels are absent or
degenerate this module falls back to a documented *usable-output proxy*
(non-empty text, no error) and stamps ``label_source`` accordingly. The proxy
is what the Thompson router already treats as success; it is **not** arbiter
selection, and the artifact says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from learning.train.base import (
    FeatureEncoder,
    TrainingData,
    TrainedModel,
    TrainingError,
    make_timestamp,
    read_jsonl,
    utc_now,
)

ARTIFACT_NAME = "quality_predictor"

#: Model IDs -> human labels the arbiter may use in a SELECTED: line.
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "GLM 5.3": ("glm-5-3", "rits/zai-org/glm-5-3", "zai-org"),
    "DeepSeek V4.1 Flash": (
        "deepseek-v4.1-flash",
        "rits/deepseek-ai/deepseek-v4.1-flash-1017",
        "deepseek",
    ),
    "Muse Spark Open": (
        "muse-spark-open",
        "rits/meta-models/muse-spark-open-816b-a42b-nvfp4",
        "muse",
    ),
    "Claude Opus 5.5": ("claude-opus-5-5", "opus"),
    "Claude Sonnet 5.5": ("claude-sonnet-5-5", "sonnet"),
    "Claude Haiku 5.5": ("claude-haiku-5-5", "haiku"),
}

NUMERIC_FEATURES = ("input_tokens", "output_tokens", "latency_ms", "cost_usd", "text_length")
STOP_REASONS = ("end_turn", "max_tokens", "tool_use", "stop_sequence", "error", "unknown")
#: Below this many authoritative (SELECTED-derived) rows, train on the proxy.
MIN_AUTHORITATIVE_ROWS = 12


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def parse_selected(text: str | None) -> str | None:
    """The model label from the last ``SELECTED:`` line, or ``None``."""
    for line in reversed(_text(text).strip().splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("SELECTED:"):
            value = stripped.split(":", 1)[1].strip()
            return value or None
    return None


def label_matches(selected: str | None, model: str | None) -> bool:
    """Whether an arbiter SELECTED label refers to *model* (case-insensitive)."""
    if not selected or not model:
        return False
    selected_norm = selected.strip().lower()
    model_norm = model.strip().lower()
    if selected_norm == model_norm:
        return True
    for label, keys in MODEL_ALIASES.items():
        if selected_norm == label.lower():
            return any(key in model_norm for key in keys)
        if any(key in model_norm for key in keys) and selected_norm in label.lower():
            return True
    # Last resort: a bare model basename or a distinctive family token.
    basename = model_norm.rsplit("/", 1)[-1]
    return selected_norm == basename or selected_norm in basename


class InsufficientTrainingData(TrainingError):
    """Raised when a model cannot be fit (no rows or a single class)."""


class QualityFeatureEncoder(FeatureEncoder):
    """One-hot ``model_id`` and ``stop_reason`` plus numeric worker metrics."""

    def __init__(
        self,
        model_ids: Sequence[str] | None = None,
        stop_reasons: Sequence[str] | None = None,
    ) -> None:
        self.model_ids: list[str] = list(model_ids) if model_ids else []
        self.stop_reasons: list[str] = list(stop_reasons) if stop_reasons else list(STOP_REASONS)
        self.feature_names: list[str] = []
        self._rebuild_names()

    def _rebuild_names(self) -> None:
        self.feature_names = (
            list(NUMERIC_FEATURES)
            + [f"model_id={value}" for value in self.model_ids]
            + [f"stop_reason={value}" for value in self.stop_reasons]
        )

    def fit(self, features: Sequence[Mapping[str, Any]]) -> "QualityFeatureEncoder":
        self.model_ids = sorted({_text(feature.get("model_id")) or "unknown" for feature in features})
        observed = {_text(feature.get("stop_reason")) or "unknown" for feature in features}
        merged = list(STOP_REASONS) + sorted(observed - set(STOP_REASONS))
        self.stop_reasons = merged
        self._rebuild_names()
        return self

    def _numeric(self, feature: Mapping[str, Any], name: str) -> float:
        value = feature.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if np.isfinite(number) else 0.0

    def transform(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[list[float]] = []
        for feature in features:
            values = [self._numeric(feature, name) for name in NUMERIC_FEATURES]
            model_id = _text(feature.get("model_id")) or "unknown"
            values.extend(1.0 if model_id == candidate else 0.0 for candidate in self.model_ids)
            stop_reason = _text(feature.get("stop_reason")) or "unknown"
            values.extend(1.0 if stop_reason == candidate else 0.0 for candidate in self.stop_reasons)
            rows.append(values)
        return np.asarray(rows, dtype=float)

    def feature_vector(self, feature: Mapping[str, Any]) -> np.ndarray:
        return self.transform([feature])[0]


def worker_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """The seven documented features for one ``worker_result`` ledger row."""
    text = _text(row.get("text"))
    return {
        "model_id": _text(row.get("model")) or "unknown",
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "latency_ms": row.get("latency_ms"),
        "cost_usd": row.get("cost_usd"),
        "stop_reason": _text(row.get("stop_reason")) or "unknown",
        "text_length": len(text),
    }


@dataclass
class ExtractionInfo:
    reviews: int = 0
    worker_rows: int = 0
    reviews_with_selection: int = 0
    authoritative_labeled: int = 0
    authoritative_positive: int = 0
    label_source: str = ""
    used_proxy: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "reviews": self.reviews,
            "worker_rows": self.worker_rows,
            "reviews_with_selection": self.reviews_with_selection,
            "authoritative_labeled": self.authoritative_labeled,
            "authoritative_positive": self.authoritative_positive,
            "label_source": self.label_source,
            "used_proxy": self.used_proxy,
            "note": self.note,
        }


def _review_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    order: list[str] = []
    for row in rows:
        review_id = _text(row.get("review_id"))
        if review_id and review_id not in order:
            order.append(review_id)
    return order


def selected_for_review(review_rows: Sequence[Mapping[str, Any]]) -> str | None:
    """The arbiter's selection for one review: field first, then SELECTED: line."""
    for row in reversed(review_rows):
        if row.get("kind") != "arbiter_decision":
            continue
        if isinstance(row.get("selected"), str) and row["selected"].strip():
            return row["selected"].strip()
        selected = parse_selected(row.get("text"))
        if selected:
            return selected
    return None


def extract_training_data(
    rows: Iterable[Mapping[str, Any]],
    *,
    allow_proxy: bool = True,
) -> tuple[TrainingData, ExtractionInfo]:
    """Build a :class:`TrainingData` from review-ledger rows.

    Authoritative labels require an explicit arbiter selection. If fewer than
    :data:`MIN_AUTHORITATIVE_ROWS` exist (or one class is missing), and
    ``allow_proxy`` is true, every worker row is labelled with the documented
    usable-output proxy instead.
    """
    rows = list(rows)
    by_review: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        review_id = _text(row.get("review_id"))
        if review_id:
            by_review.setdefault(review_id, []).append(row)

    info = ExtractionInfo(reviews=len(by_review))
    authoritative: list[tuple[dict[str, Any], int, str, dict[str, Any]]] = []
    proxy: list[tuple[dict[str, Any], int, str, dict[str, Any]]] = []
    for review_id in _review_order(rows):
        review_rows = by_review[review_id]
        selected = selected_for_review(review_rows)
        if selected:
            info.reviews_with_selection += 1
        for row in review_rows:
            if row.get("kind") != "worker_result":
                continue
            info.worker_rows += 1
            features = worker_features(row)
            timestamp = make_timestamp(row.get("recorded_at") or row.get("observed_at"))
            metadata = {"review_id": review_id, "model": features["model_id"]}
            if selected:
                label = 1 if label_matches(selected, features["model_id"]) else 0
                authoritative.append((features, label, timestamp, metadata))
            usable = bool(_text(row.get("text")).strip()) and not row.get("error")
            proxy.append((features, 1 if usable else 0, timestamp, metadata))

    positives = sum(label for _, label, _, _ in authoritative)
    info.authoritative_labeled = len(authoritative)
    info.authoritative_positive = positives
    use_proxy = allow_proxy and (
        len(authoritative) < MIN_AUTHORITATIVE_ROWS
        or positives == 0
        or positives == len(authoritative)
    )
    examples = proxy if use_proxy else authoritative
    if use_proxy:
        info.used_proxy = True
        info.label_source = "usable_output_proxy"
        if not authoritative:
            info.note = (
                "no arbiter SELECTED line/field in the ledger; trained on the "
                "usable-output proxy (non-empty text, no error)"
            )
        else:
            info.note = (
                f"only {len(authoritative)} authoritative SELECTED row(s) "
                f"({positives} positive); trained on the usable-output proxy"
            )
    else:
        info.label_source = "arbiter_selected"
        info.note = "labels from arbiter SELECTED lines/fields"

    data = TrainingData(
        features=[features for features, _, _, _ in examples],
        labels=[label for _, label, _, _ in examples],
        timestamps=[timestamp for _, _, timestamp, _ in examples],
        metadata=[metadata for _, _, _, metadata in examples],
        label_name="arbiter_selected",
        label_source=info.label_source,
        feature_names=list(NUMERIC_FEATURES),
    )
    return data, info


class QualityPredictor:
    """Fitted quality model plus encoding and reporting helpers."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model

    # -- training --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        data: TrainingData,
        *,
        train_fraction: float = 0.7,
        random_state: int = 0,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.1,
    ) -> "QualityPredictor":
        from sklearn.ensemble import GradientBoostingClassifier

        if len(data) < 2:
            raise InsufficientTrainingData(
                f"quality predictor needs at least 2 rows, got {len(data)}"
            )
        classes = set(data.labels)
        if len(classes) < 2:
            raise InsufficientTrainingData(
                "quality predictor needs both classes; the ledger produced only "
                f"{classes or 'no labels'}"
            )
        train, test = data.split(train_fraction)
        if len(set(train.labels)) < 2:
            raise InsufficientTrainingData(
                "chronological training split contains a single class; "
                "collect more reviews before retraining"
            )
        encoder = QualityFeatureEncoder().fit(train.features)
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        estimator.fit(
            encoder.transform(train.features),
            np.asarray(train.labels),
            sample_weight=train.sample_weights(),
        )
        model = TrainedModel(
            name=ARTIFACT_NAME,
            estimator=estimator,
            encoder=encoder,
            label_name=data.label_name,
            label_source=data.label_source,
            trained_at=utc_now(),
            data_hash=data.data_hash(),
            n_train=len(train),
            n_test=len(test),
            extras={
                "train_fraction": train_fraction,
                "class_balance": data.class_balance_report(),
                "random_state": random_state,
            },
        )
        predictor = cls(model)
        model.metrics = {
            "train": predictor.evaluate(train),
            "test": predictor.evaluate(test),
        }
        return predictor

    # -- inference -------------------------------------------------------

    def predict_quality(self, features: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
        single = isinstance(features, Mapping)
        rows = [features] if single else list(features)
        if not rows:
            return [] if not single else 0.0
        scores = self.model.positive_score(rows)
        return float(scores[0]) if single else [float(score) for score in scores]

    def cold_start_prior(self, features: Mapping[str, Any], *, floor: float = 0.05,
                         ceil: float = 0.95) -> float:
        """A clipped, feature-based prior for a model with no Thompson history."""
        score = float(self.predict_quality(features))
        return max(floor, min(ceil, score))

    def evaluate(self, data: TrainingData) -> dict[str, Any]:
        return self.model.evaluate(data)

    def feature_importances(self, limit: int | None = None) -> list[tuple[str, float]]:
        values = self.model.feature_importances()
        return values[:limit] if limit else values

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "QualityPredictor":
        return cls(TrainedModel.load(path, name=ARTIFACT_NAME))


def default_worker_features(
    model_id: str,
    *,
    input_tokens: float = 1000,
    output_tokens: float = 500,
    latency_ms: float = 2000,
    cost_usd: float = 0.01,
    stop_reason: str = "end_turn",
    text_length: float = 4000,
) -> dict[str, Any]:
    """Representative features for a model with no review history.

    Cold start: feeding these to a trained predictor gives a feature-based
    estimate that can seed Thompson's prior for a new arm.
    """
    return {
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "stop_reason": stop_reason,
        "text_length": text_length,
    }


def cold_start_prior(model: Any, model_id: str, **overrides: Any) -> float:
    """A clipped ``[0.05, 0.95]`` prior for an arm with no Thompson history."""
    if isinstance(model, QualityPredictor):
        predictor = model
    elif isinstance(model, TrainedModel):
        predictor = QualityPredictor(model)
    else:
        predictor = QualityPredictor.load(model)
    return predictor.cold_start_prior(default_worker_features(model_id, **overrides))


def predict_quality(model: Any, features: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
    """Public entry point used by ``tools/ledger_to_thompson.py``.

    ``model`` may be a :class:`QualityPredictor`, a :class:`TrainedModel`, or a
    path to a saved artifact.
    """
    if isinstance(model, QualityPredictor):
        predictor = model
    elif isinstance(model, TrainedModel):
        predictor = QualityPredictor(model)
    elif isinstance(model, (str, bytes)):
        predictor = QualityPredictor.load(model)
    else:
        raise TypeError("model must be a QualityPredictor, TrainedModel, or artifact path")
    return predictor.predict_quality(features)


def load_rows(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
