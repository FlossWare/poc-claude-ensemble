"""Real (gradient-based) model training for Claude Ensemble's learning signals.

This package is deliberately separate from ``learning.ga``: the GA searches
hyper-parameters, while these modules fit actual parameters by gradient descent
on a train/test split and report generalization error.

Three supervised models:

* :mod:`learning.train.quality_predictor` -- will the arbiter SELECT a worker?
* :mod:`learning.train.retrieval_relevance` -- did injected RAG context help?
* :mod:`learning.train.task_classifier` -- what kind of task is this prompt?

One unsupervised detector:

* :mod:`learning.train.sumo_anomaly` -- abnormal timeslices in Disseminator
  search metrics fetched from Sumo Logic (IsolationForest).

Run ``PYTHONPATH=services python3 -m learning.train`` for status.
"""

from __future__ import annotations

__all__ = [
    "base",
    "quality_predictor",
    "retrieval_relevance",
    "sumo_anomaly",
    "task_classifier",
]
