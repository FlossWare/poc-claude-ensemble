"""Generalized meta-learning framework: one Thompson posterior per decision.

Usage::

    from learning.meta import MetaLearner
    learner = MetaLearner()
    size = learner.select("crew_size", {"task_type": "code-review"})   # int
    learner.record("crew_size", size, outcome=0.8, cost=0.12,
                   metadata={"outcome_id": "review-123"})
    learner.posterior("crew_size")

CLI::

    PYTHONPATH=services python3 -m learning.meta                 # all domains
    PYTHONPATH=services python3 -m learning.meta select crew_size --task-type code-review
    PYTHONPATH=services python3 -m learning.meta backfill review_depth

Sampling reuses :class:`shared.thompson_client.ThompsonClient`, so the policy
GA Target A tunes applies to every domain.
"""

from learning.meta.domains import (
    DOMAIN_CLASSES,
    Domain,
    DomainError,
    domain_names,
    functional_domains,
    get_domain,
    instrumentation_domains,
)
from learning.meta.learner import (
    DEFAULT_META_DIR,
    MAX_OUTCOME_LOG,
    META_DIR_ENV,
    MIN_CONTEXT_OBSERVATIONS,
    MetaLearner,
)

__all__ = [
    "DEFAULT_META_DIR",
    "DOMAIN_CLASSES",
    "Domain",
    "DomainError",
    "MAX_OUTCOME_LOG",
    "META_DIR_ENV",
    "MIN_CONTEXT_OBSERVATIONS",
    "MetaLearner",
    "domain_names",
    "functional_domains",
    "get_domain",
    "instrumentation_domains",
]
