"""Retrieval relevance: did the injected RAG context help the response?

Interpretable ``LogisticRegression`` over the exposure journal written by the
RAG hook (PR #69). Features come from the journal row:

============================  ============================================
feature                       source
============================  ============================================
``memory_count``              ``sections.memory.results``
``knowledge_count``           ``sections.knowledge.results``
``graph_count``               ``sections.graph.results`` (or 0)
``memory_chars``              ``sections.memory.chars``
``knowledge_chars``           ``sections.knowledge.chars``
``graph_chars``               ``sections.graph.chars`` (or 0)
``task_type``                 ``sections.intent.label`` (one-hot)
============================  ============================================

The hook journals **features only**; it deliberately does not link an exposure
to an outcome (see ``tools/rag_hook/README.md``). The label field
``"helped"`` therefore exists in the schema but is nullable, and today it is
null in every live row. When no row carries a label this module reports
``insufficient labeled data`` instead of fitting a model on invented labels.

Documented follow-up: derive ``helped`` from the next-turn analysis (for
example, a user correction or re-ask shortly after the injection suggests the
context did not help). Until that linkage lands, this model stays cold.
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

ARTIFACT_NAME = "retrieval_relevance"
LABEL_FIELD = "helped"

NUMERIC_FEATURES = (
    "memory_count",
    "knowledge_count",
    "graph_count",
    "memory_chars",
    "knowledge_chars",
    "graph_chars",
)


class InsufficientLabeledData(TrainingError):
    """No usable (non-null) labels exist, so no model is fit."""


class RetrievalFeatureEncoder(FeatureEncoder):
    """Numeric journal counts + one-hot ``task_type``, standardized for LR."""

    def __init__(self, task_types: Sequence[str] | None = None) -> None:
        self.task_types: list[str] = list(task_types) if task_types else []
        self.feature_names: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._rebuild_names()

    def _rebuild_names(self) -> None:
        self.feature_names = list(NUMERIC_FEATURES) + [
            f"task_type={value}" for value in self.task_types
        ]

    def fit(self, features: Sequence[Mapping[str, Any]]) -> "RetrievalFeatureEncoder":
        self.task_types = sorted(
            {_task_type(feature) for feature in features}
        )
        self._rebuild_names()
        raw = self._raw(features)
        self._mean = raw.mean(axis=0) if len(raw) else np.zeros(len(self.feature_names))
        scale = raw.std(axis=0) if len(raw) else np.ones(len(self.feature_names))
        # A constant column carries no information; keep it unscaled.
        scale[scale == 0] = 1.0
        self._scale = scale
        return self

    def _raw(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[list[float]] = []
        for feature in features:
            values = [_number(feature.get(name)) for name in NUMERIC_FEATURES]
            task_type = _task_type(feature)
            values.extend(1.0 if task_type == candidate else 0.0 for candidate in self.task_types)
            rows.append(values)
        return np.asarray(rows, dtype=float)

    def transform(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        raw = self._raw(features)
        if self._mean is None or self._scale is None:
            return raw
        return (raw - self._mean) / self._scale


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _int(value: Any) -> int:
    return int(_number(value))


def _section(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sections = row.get("sections")
    if isinstance(sections, Mapping):
        value = sections.get(name)
        if isinstance(value, Mapping):
            return value
    value = row.get(name)
    return value if isinstance(value, Mapping) else {}


def _task_type(feature: Mapping[str, Any]) -> str:
    for key in ("task_type", "intent", "intent_label"):
        value = feature.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def injection_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one ``rag_injection`` journal row to the documented feature dict."""
    memory = _section(row, "memory")
    knowledge = _section(row, "knowledge")
    graph = _section(row, "graph")
    intent = _section(row, "intent")

    def pick(top_level: str, section: Mapping[str, Any], nested: str, default: Any = 0) -> Any:
        if row.get(top_level) is not None:
            return row[top_level]
        if section.get(nested) is not None:
            return section[nested]
        return default

    return {
        "memory_count": pick("memory_count", memory, "results"),
        "knowledge_count": pick("knowledge_count", knowledge, "results"),
        "graph_count": pick("graph_count", graph, "results"),
        "memory_chars": pick("memory_chars", memory, "chars"),
        "knowledge_chars": pick("knowledge_chars", knowledge, "chars"),
        "graph_chars": pick("graph_chars", graph, "chars"),
        "task_type": _task_type(row)
        if _task_type(row) != "unknown"
        else (intent.get("label") or "unknown"),
    }


def _label(row: Mapping[str, Any]) -> int | None:
    """A nullable label: ``True``/``False`` become 1/0, anything else is null."""
    value = row.get(LABEL_FIELD)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    return None


@dataclass
class ExtractionInfo:
    total_rows: int = 0
    labeled_rows: int = 0
    null_label_rows: int = 0
    positive: int = 0
    label_field: str = LABEL_FIELD
    message: str = ""

    @property
    def sufficient(self) -> bool:
        return self.labeled_rows > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "labeled_rows": self.labeled_rows,
            "null_label_rows": self.null_label_rows,
            "positive": self.positive,
            "label_field": self.label_field,
            "sufficient": self.sufficient,
            "message": self.message,
        }


def insufficient_message(info: ExtractionInfo) -> str:
    if info.total_rows == 0:
        return "insufficient labeled data: no injection journal rows"
    return (
        f"insufficient labeled data: {info.labeled_rows} labeled of "
        f"{info.total_rows} journal rows (`{LABEL_FIELD}` is null; outcome linkage "
        "from next-turn analysis is a documented follow-up)"
    )


def extract_training_data(rows: Iterable[Mapping[str, Any]]) -> tuple[TrainingData, ExtractionInfo]:
    rows = list(rows)
    info = ExtractionInfo(total_rows=len(rows))
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    timestamps: list[str] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        label = _label(row)
        if label is None:
            info.null_label_rows += 1
            continue
        info.labeled_rows += 1
        info.positive += label
        features.append(injection_features(row))
        labels.append(label)
        timestamps.append(make_timestamp(row.get("timestamp") or row.get("recorded_at")))
        metadata.append({"event_id": row.get("event_id"), "session_id": row.get("session_id")})
    if not info.labeled_rows:
        info.message = insufficient_message(info)
    data = TrainingData(
        features=features,
        labels=labels,
        timestamps=timestamps,
        metadata=metadata,
        label_name=LABEL_FIELD,
        label_source="rag_injection_outcome" if info.labeled_rows else "",
        feature_names=list(NUMERIC_FEATURES),
    )
    return data, info


class RetrievalRelevanceModel:
    """``LogisticRegression`` over journal features; coefficients are exposed."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model

    @classmethod
    def fit(
        cls,
        data: TrainingData,
        *,
        train_fraction: float = 0.7,
        C: float = 1.0,
        random_state: int = 0,
    ) -> "RetrievalRelevanceModel":
        from sklearn.linear_model import LogisticRegression

        if len(data) < 2:
            raise InsufficientLabeledData(
                "insufficient labeled data: need at least 2 labeled rows, got "
                f"{len(data)}"
            )
        if len(set(data.labels)) < 2:
            raise InsufficientLabeledData(
                "insufficient labeled data: labels contain a single class "
                f"({sorted(set(data.labels))})"
            )
        train, test = data.split(train_fraction)
        if len(set(train.labels)) < 2:
            raise InsufficientLabeledData(
                "insufficient labeled data: chronological training split has one class"
            )
        encoder = RetrievalFeatureEncoder().fit(train.features)
        estimator = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        estimator.fit(encoder.transform(train.features), np.asarray(train.labels))
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
                "C": C,
            },
        )
        instance = cls(model)
        model.metrics = {
            "train": instance.evaluate(train),
            "test": instance.evaluate(test),
        }
        return instance

    def predict_relevance(self, features: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
        single = isinstance(features, Mapping)
        rows = [features] if single else list(features)
        if not rows:
            return [] if not single else 0.0
        scores = self.model.positive_score(rows)
        return float(scores[0]) if single else [float(score) for score in scores]

    def evaluate(self, data: TrainingData) -> dict[str, Any]:
        return self.model.evaluate(data)

    def coefficients(self) -> dict[str, dict[str, float]]:
        """Per-class standardized coefficients (interpretable feature effects)."""
        return self.model.coefficient_table()

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "RetrievalRelevanceModel":
        return cls(TrainedModel.load(path, name=ARTIFACT_NAME))


def predict_relevance(model: Any, features: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
    if isinstance(model, RetrievalRelevanceModel):
        instance = model
    elif isinstance(model, TrainedModel):
        instance = RetrievalRelevanceModel(model)
    elif isinstance(model, (str, bytes)):
        instance = RetrievalRelevanceModel.load(model)
    else:
        raise TypeError("model must be a RetrievalRelevanceModel, TrainedModel, or artifact path")
    return instance.predict_relevance(features)


def load_rows(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
