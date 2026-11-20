"""External quasi-learning signals for Claude Ensemble.

Three public sources feed the learning stack without credentials:

* :mod:`learning.external.elo_priors` -- LMSYS Arena ELO -> Beta prior
* :mod:`learning.external.benchmark_priors` -- Open LLM Leaderboard -> quality
* :mod:`learning.external.pricing` -- OpenRouter -> current token prices

:mod:`learning.external.registry` records which external prior was applied to
which Thompson arm, and :mod:`learning.external.net` is the netguard-backed
fetch/cache layer shared by all three.
"""

from __future__ import annotations

__all__ = ["benchmark_priors", "elo_priors", "net", "paths", "pricing", "registry"]
