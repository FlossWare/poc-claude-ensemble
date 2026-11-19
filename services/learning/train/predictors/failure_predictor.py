"""Pre-call failure-risk predictor.

Label: the inverse of :func:`~learning.train.predictors.features.is_success`,
i.e. ``1`` when a worker produced no usable output (truncated, empty, errored).
Model: gradient-boosted classification trees.  Routing use: ``crew_select`` can
avoid arms with a high predicted failure risk.

The review ledger's five failures are all ``max_tokens`` truncations.  Stop
reason is not a pre-call feature, so the model cannot see the direct cause; it
can only learn which models/task types correlate with truncation.  With five
positives this is a weak signal and the metadata says so.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import GradientBoostingClassifier

from learning.train.predictors.base import BasePredictor, RANDOM_STATE
from learning.train.predictors.features import ReviewFeatures


class FailurePredictor(BasePredictor):
    KIND = "failure"
    DISPLAY = "failure predictor"
    TARGET = "failure"
    TASK = "classification"
    UNIT = "probability"
    LABEL_DEFINITION = (
        "failure = NOT (stop_reason == 'end_turn' AND text is non-empty AND no error field); "
        "positive class is unusable output"
    )

    def _make_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )

    def _row_label(self, row: ReviewFeatures) -> float:
        return 0.0 if row.success else 1.0

    def _postprocess(self, value: float) -> float:
        return min(1.0, max(0.0, value))

    def predict_failure_risk(self, model: str, task_features: Optional[Any] = None) -> float:
        """Probability (0-1) that this arm produces unusable output."""
        return self.predict_one(model, task_features)
