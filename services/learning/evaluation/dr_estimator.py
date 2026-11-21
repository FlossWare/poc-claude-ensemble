"""Doubly-robust off-policy value estimator over logged bandit data.

Given logged tuples ``(context, action, reward, propensity)`` produced by
the BEHAVIOUR policy (the router that actually ran), estimate the value
(mean reward) of a TARGET policy:

    V_DR = (1/n) * sum_i [
        1[pi(x_i) == a_i] / p_i * (r_i - q(x_i, a_i))     # IPS residual
        + q(x_i, pi(x_i))                                  # model direct
    ]

* ``pi(context) -> action``: the target policy (e.g. a GA candidate).
* ``q(context, action) -> reward estimate``: a reward model (e.g. the
  quality/routing predictors).  If the model is wrong, the IPS term keeps
  the estimate consistent; if the propensities are wrong, the model term
  does.  Either one being right is enough.
* Propensities <= 0 on rows the target policy matches make the row's IPS
  term undefined; those rows are counted and reported (never silently
  dropped) and the estimator degrades to the model term for them.

All functions are pure; no I/O, no state, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

# A logged row: (context, action, reward, propensity).
LogRow = Tuple[Mapping, object, float, float]
# context -> action for the policy under evaluation.
Policy = Callable[[Mapping], object]
# (context, action) -> predicted reward.
RewardModel = Callable[[Mapping, object], float]


@dataclass
class DREstimate:
    value: float
    rows: int
    matched_rows: int          # rows where the target took the logged action
    zero_propensity_rows: int  # matched rows unusable for the IPS term
    model_only_value: float    # the pure direct-method estimate
    ips_value: float           # the pure inverse-propensity estimate


def doubly_robust_value(
    rows: Sequence[LogRow],
    policy: Policy,
    reward_model: RewardModel,
) -> DREstimate:
    """Estimate the target policy's mean reward from logged data."""
    if not rows:
        raise ValueError("no logged rows")
    total = 0.0
    model_total = 0.0
    ips_total = 0.0
    matched = 0
    zero_prop = 0
    for context, action, reward, propensity in rows:
        target_action = policy(context)
        q_target = reward_model(context, target_action)
        model_total += q_target
        contribution = q_target  # direct term always applies
        if target_action == action:
            matched += 1
            if propensity > 0.0:
                residual = reward - reward_model(context, action)
                ips_term = residual / propensity
                contribution += ips_term
                ips_total += ips_term
            else:
                zero_prop += 1
        total += contribution
    n = len(rows)
    return DREstimate(
        value=total / n,
        rows=n,
        matched_rows=matched,
        zero_propensity_rows=zero_prop,
        model_only_value=model_total / n,
        ips_value=ips_total / n,
    )


def clipped_doubly_robust_value(
    rows: Sequence[LogRow],
    policy: Policy,
    reward_model: RewardModel,
    *,
    clip: float = 10.0,
) -> DREstimate:
    """DR with importance weights clipped at ``clip`` (variance guard).

    Clipping introduces bias when the behaviour policy rarely took the
    target's action; that trade is standard (the alternative is a single
    lucky row dominating the estimate).  The clip is applied to the
    weight 1/p only, never to the reward.
    """
    if not rows:
        raise ValueError("no logged rows")
    if clip < 1.0:
        raise ValueError("clip must be >= 1")
    total = 0.0
    model_total = 0.0
    ips_total = 0.0
    matched = 0
    zero_prop = 0
    for context, action, reward, propensity in rows:
        target_action = policy(context)
        q_target = reward_model(context, target_action)
        model_total += q_target
        contribution = q_target
        if target_action == action:
            matched += 1
            if propensity > 0.0:
                weight = min(1.0 / propensity, clip)
                ips_term = weight * (reward - reward_model(context, action))
                contribution += ips_term
                ips_total += ips_term
            else:
                zero_prop += 1
        total += contribution
    n = len(rows)
    return DREstimate(
        value=total / n,
        rows=n,
        matched_rows=matched,
        zero_propensity_rows=zero_prop,
        model_only_value=model_total / n,
        ips_value=ips_total / n,
    )
