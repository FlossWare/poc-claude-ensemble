"""Pre-call cost predictor.

Label: ``cost_usd`` from the review ledger.  Model: gradient-boosted regression
trees.  Routing use: ``crew_select`` can drop arms whose predicted cost exceeds
a per-task budget before spending money on them.

Caveat: the ledger records actual output tokens only after the call.  The model
therefore predicts cost from model identity, task type and prompt size, which
explains far less variance than a post-hoc cost calculation would.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import GradientBoostingRegressor

from learning.train.predictors.base import BasePredictor, RANDOM_STATE
from learning.train.predictors.features import ReviewFeatures


class CostPredictor(BasePredictor):
    KIND = "cost"
    DISPLAY = "cost predictor"
    TARGET = "cost_usd"
    TASK = "regression"
    UNIT = "usd"
    LABEL_DEFINITION = "cost_usd reported by the provider for one worker result"

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
        return float(row.cost_usd)

    def _postprocess(self, value: float) -> float:
        return max(0.0, value)

    def predict_cost(self, model: str, task_features: Optional[Any] = None) -> float:
        """Predicted cost in USD for one arm and pre-call task features."""
        return self.predict_one(model, task_features)
