"""Pre-call latency predictor.

Label: ``latency_ms`` from the review ledger.  Model: gradient-boosted
regression trees.  Routing use: time-sensitive tasks can prefer the arm with
the lowest expected wall-clock latency.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import GradientBoostingRegressor

from learning.train.predictors.base import BasePredictor, RANDOM_STATE
from learning.train.predictors.features import ReviewFeatures


class LatencyPredictor(BasePredictor):
    KIND = "latency"
    DISPLAY = "latency predictor"
    TARGET = "latency_ms"
    TASK = "regression"
    UNIT = "ms"
    LABEL_DEFINITION = "wall-clock latency_ms reported for one worker result"

    def _make_estimator(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )

    def _row_label(self, row: ReviewFeatures) -> float:
        return float(row.latency_ms)

    def _postprocess(self, value: float) -> float:
        return max(0.0, value)

    def predict_latency(self, model: str, task_features: Optional[Any] = None) -> float:
        """Predicted latency in milliseconds for one arm and task."""
        return self.predict_one(model, task_features)
