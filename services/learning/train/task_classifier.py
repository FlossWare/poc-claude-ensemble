"""Task classifier: route a prompt/description to a task type.

Self-contained TF-IDF (no embedding service) plus three cheap structural
features (token count, ``has_code``, ``has_url``), fit with ``LinearSVC``.
Class weights are balanced because the captured stream is dominated by
conversational prompts (``other``).

Confidence is the softmax of the SVC decision margins. It is a usable ranking
signal, not a calibrated probability (no ``CalibratedClassifierCV`` because
that hides the linear coefficients this module exposes for inspection).

**Label caveat.** The captured streams contain text, not human task-type
labels. Labels here are *weak supervision* from the documented keyword rules in
:func:`heuristic_label`; the artifact stamps ``label_source =
"heuristic_keyword_rules"``. This is enough to learn a routing prior and to
give `crew_select` a context signal, but it is not ground truth. Replacing
these rules with accepted/rejected routing outcomes is a documented follow-up.

Integration: :func:`classify_task` returns ``(task_type, confidence)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from learning.train.base import (
    FeatureEncoder,
    TrainingData,
    TrainedModel,
    TrainingError,
    make_timestamp,
    read_jsonl,
    utc_now,
)

ARTIFACT_NAME = "task_classifier"
TASK_TYPES = ("code-review", "implementation", "documentation", "research", "qa", "other")
LABEL_SOURCE = "heuristic_keyword_rules"
MIN_DESCRIPTION_CHARS = 4
#: Numerical feature scale: log1p(token_count) / this keeps counts near 0-1.
TOKEN_SCALE = 20.0

_CODE_RE = re.compile(r"```|^\s*(?:def |class |function |import |#include|package )", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"\S+")

# Ordered rules: first match wins. Put narrower intents before broad verbs.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "code-review",
        (
            "code review",
            "review this",
            "review the",
            "review my",
            "audit",
            "pull request",
            "merge request",
            "diff",
            "defect",
            "vulnerab",
            "critique",
            "find bugs",
            "look for bugs",
        ),
    ),
    (
        "qa",
        (
            "test",
            "pytest",
            "verify",
            "reproduce",
            "failing",
            "regression",
            "coverage",
            "assert",
        ),
    ),
    (
        "documentation",
        (
            "document",
            "readme",
            "docs",
            "docstring",
            "explain",
            "how does",
            "how do ",
            "what does",
            "tutorial",
            "write up",
        ),
    ),
    (
        "implementation",
        (
            "implement",
            "add ",
            "build",
            "create",
            "refactor",
            "fix ",
            "fix:",
            "patch",
            "write ",
            "migrate",
            "update ",
            "support ",
            "wire ",
            "make it",
        ),
    ),
    (
        "research",
        (
            "research",
            "investigate",
            "explore",
            "compare",
            "find out",
            "look into",
            "analysis",
            "analyze",
            "why ",
            "what if",
        ),
    ),
)


class InsufficientTrainingData(TrainingError):
    """Not enough distinct labelled examples to fit a classifier."""


def _rule_matches(text: str, needle: str) -> bool:
    """Word-boundary prefix match, so ``test`` does not fire inside ``latest``."""
    return re.search(r"\b" + re.escape(needle), text) is not None


def heuristic_label(description: str) -> str:
    """Weak-supervision label from keyword rules (documented, not ground truth)."""
    text = (description or "").lower()
    for task_type, needles in _RULES:
        if any(_rule_matches(text, needle) for needle in needles):
            return task_type
    return "other"


def task_features(description: str) -> dict[str, Any]:
    text = description or ""
    return {
        "text": text,
        "token_count": len(_TOKEN_RE.findall(text)),
        "has_code": 1.0 if _CODE_RE.search(text) else 0.0,
        "has_url": 1.0 if _URL_RE.search(text) else 0.0,
    }


class TaskFeatureEncoder(FeatureEncoder):
    """Word + character TF-IDF blocks hstacked with structural numeric features.

    Character n-grams catch task idioms (``pull request``, ``readme``) that word
    bigrams miss when the captured stream uses different phrasing.
    """

    NUMERIC = ("token_count", "has_code", "has_url")

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        self.feature_names: list[str] = []

    def fit(self, features: Sequence[Mapping[str, Any]]) -> "TaskFeatureEncoder":
        texts = [str(feature.get("text") or "") for feature in features]
        self.vectorizer.fit(texts)
        try:
            self.char_vectorizer.fit(texts)
        except ValueError:
            # Empty/degenerate vocabulary (tiny fixtures): skip the char block.
            self.char_vectorizer = None
        word_names = [f"word:{name}" for name in self.vectorizer.get_feature_names_out()]
        char_names = (
            [f"char:{name}" for name in self.char_vectorizer.get_feature_names_out()]
            if self.char_vectorizer is not None
            else []
        )
        self.feature_names = word_names + char_names + list(self.NUMERIC)
        return self

    def _numeric(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows = []
        for feature in features:
            rows.append(
                [
                    float(np.log1p(max(0.0, float(feature.get("token_count") or 0)))) / TOKEN_SCALE,
                    float(feature.get("has_code") or 0.0),
                    float(feature.get("has_url") or 0.0),
                ]
            )
        return np.asarray(rows, dtype=float)

    def transform(self, features: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
        texts = [str(feature.get("text") or "") for feature in features]
        blocks = [self.vectorizer.transform(texts)]
        if getattr(self, "char_vectorizer", None) is not None:
            blocks.append(self.char_vectorizer.transform(texts))
        blocks.append(sparse.csr_matrix(self._numeric(features)))
        return sparse.hstack(blocks, format="csr")


@dataclass
class ExtractionInfo:
    prompt_rows: int = 0
    review_rows: int = 0
    descriptions: int = 0
    duplicates_skipped: int = 0
    label_counts: dict[str, int] | None = None
    label_source: str = LABEL_SOURCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_rows": self.prompt_rows,
            "review_rows": self.review_rows,
            "descriptions": self.descriptions,
            "duplicates_skipped": self.duplicates_skipped,
            "label_counts": self.label_counts or {},
            "label_source": self.label_source,
        }


def _prompt_text(row: Mapping[str, Any]) -> str:
    for key in ("prompt", "first_request", "description", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_training_data(
    prompt_rows: Iterable[Mapping[str, Any]],
    review_rows: Iterable[Mapping[str, Any]] = (),
) -> tuple[TrainingData, ExtractionInfo]:
    """Collect unique descriptions from prompts and review episodes.

    Task type labels come from :func:`heuristic_label` (weak supervision).
    """
    prompt_rows = list(prompt_rows)
    review_rows = list(review_rows)
    info = ExtractionInfo(prompt_rows=len(prompt_rows), review_rows=len(review_rows))

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for row in prompt_rows:
        text = _prompt_text(row)
        if text:
            candidates.append((text, make_timestamp(row.get("timestamp")), {"source": "prompts"}))
    for row in review_rows:
        if row.get("kind") != "review_episode":
            continue
        text = _prompt_text(row)
        if text:
            candidates.append(
                (
                    text,
                    make_timestamp(row.get("observed_at") or row.get("first_seen")),
                    {"source": "review_episode", "project": row.get("project")},
                )
            )

    features: list[dict[str, Any]] = []
    labels: list[str] = []
    timestamps: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text, timestamp, meta in candidates:
        normalised = " ".join(text.split())
        if len(normalised) < MIN_DESCRIPTION_CHARS:
            continue
        if normalised in seen:
            info.duplicates_skipped += 1
            continue
        seen.add(normalised)
        features.append(task_features(text))
        labels.append(heuristic_label(text))
        timestamps.append(timestamp)
        metadata.append(meta)

    info.descriptions = len(features)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    info.label_counts = counts
    data = TrainingData(
        features=features,
        labels=labels,
        timestamps=timestamps,
        metadata=metadata,
        label_name="task_type",
        label_source=LABEL_SOURCE,
        feature_names=list(TaskFeatureEncoder.NUMERIC),
    )
    return data, info


class TaskClassifier:
    """``LinearSVC`` (balanced) over TF-IDF + structural features."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model

    @classmethod
    def fit(
        cls,
        data: TrainingData,
        *,
        train_fraction: float = 0.7,
        C: float = 1.0,
    ) -> "TaskClassifier":
        from sklearn.svm import LinearSVC

        if len(data) < 2:
            raise InsufficientTrainingData(
                f"task classifier needs at least 2 descriptions, got {len(data)}"
            )
        if len(set(data.labels)) < 2:
            raise InsufficientTrainingData(
                "task classifier needs at least 2 task types; labels are "
                f"{sorted(set(data.labels))}"
            )
        train, test = data.split(train_fraction)
        if len(set(train.labels)) < 2:
            raise InsufficientTrainingData(
                "chronological training split contains a single task type; "
                "collect more prompts before retraining"
            )
        encoder = TaskFeatureEncoder().fit(train.features)
        estimator = LinearSVC(
            C=C,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=0,
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
                "confidence": "softmax(decision_margin)",
            },
        )
        instance = cls(model)
        model.metrics = {
            "train": instance.evaluate(train),
            "test": instance.evaluate(test),
        }
        return instance

    def _probabilities(self, features: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Softmax over decision margins; ``predict_proba`` is not available."""
        from scipy.special import softmax

        matrix = self.model.encoder.transform(features)
        decision = np.asarray(self.model.estimator.decision_function(matrix))
        if decision.ndim == 1:  # binary SVC: one margin per row
            positive = 1.0 / (1.0 + np.exp(-decision))
            return np.column_stack([1.0 - positive, positive])
        return softmax(decision, axis=1)

    def classify(self, description: str) -> tuple[str, float]:
        return self.classify_batch([description])[0]

    def classify_batch(self, descriptions: Sequence[str]) -> list[tuple[str, float]]:
        if not descriptions:
            return []
        features = [task_features(description) for description in descriptions]
        probabilities = self._probabilities(features)
        classes = self.model.classes
        results = []
        for row in probabilities:
            index = int(np.argmax(row))
            results.append((str(classes[index]), float(row[index])))
        return results

    def evaluate(self, data: TrainingData) -> dict[str, Any]:
        return self.model.evaluate(data)

    def top_features(self, limit: int = 10) -> list[tuple[str, float]]:
        return self.model.feature_importances()[:limit]

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "TaskClassifier":
        return cls(TrainedModel.load(path, name=ARTIFACT_NAME))


def classify_task(model: Any, description: str) -> tuple[str, float]:
    """``(task_type, confidence)`` used by ``crew_select``/meta-learning."""
    if isinstance(model, TaskClassifier):
        instance = model
    elif isinstance(model, TrainedModel):
        instance = TaskClassifier(model)
    elif isinstance(model, (str, bytes)):
        instance = TaskClassifier.load(model)
    else:
        raise TypeError("model must be a TaskClassifier, TrainedModel, or artifact path")
    return instance.classify(description)


def load_rows(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
