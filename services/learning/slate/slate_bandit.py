"""Thompson-scored slate selection with a diversity bonus.

``select_slate(arms, families, k)`` builds a crew of ``k`` distinct arms by
greedy submodular maximisation of

    score(S) = sum_{a in S} draw(a)            # marginal Thompson quality
             + lambda * distinct_families(S)   # diversity bonus

where ``draw(a)`` is one Beta(alpha, beta) sample per arm (exactly the
existing Thompson semantics) and ``families`` maps each arm to a family
label (provider, model lineage -- the caller's choice).  Greedy selection
of a coverage-style diversity term is (1 - 1/e)-optimal for the submodular
part; with the linear quality term the combined objective is still
monotone submodular, so greedy is near-optimal with the same guarantee.

The result records each arm's draw, its family, and whether the diversity
bonus or the raw draw admitted it -- evidence for the meta-learning
crew_size domain.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ArmStats = Tuple[float, float]  # (alpha, beta) of the Beta posterior


@dataclass
class SlateSelection:
    slate: List[str]
    draws: Dict[str, float]
    families: Dict[str, str]
    admitted_by: Dict[str, str]  # arm -> "draw" | "diversity"
    score: float


def _beta_draw(rng: random.Random, alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Beta posterior requires alpha, beta > 0; got ({alpha}, {beta})")
    return rng.betavariate(alpha, beta)


def select_slate(
    arms: Sequence[str],
    stats: Mapping[str, ArmStats],
    families: Mapping[str, str],
    k: int,
    *,
    lambda_diversity: float = 0.5,
    rng: Optional[random.Random] = None,
) -> SlateSelection:
    """Draw one crew of ``k`` arms, scored jointly with diversity.

    ``stats`` maps arm -> (alpha, beta); arms missing from ``stats`` get
    the uninformative Beta(1, 1) prior so new models are still explored.
    """
    if k < 1:
        raise ValueError("slate size must be >= 1")
    if len(set(arms)) != len(arms):
        raise ValueError("arms must be unique")
    if k > len(arms):
        raise ValueError(f"cannot draw {k} arms from {len(arms)}")
    if lambda_diversity < 0:
        raise ValueError("lambda_diversity must be >= 0")
    rng = rng or random.Random()

    remaining = sorted(arms)
    draws = {
        arm: _beta_draw(rng, *stats.get(arm, (1.0, 1.0))) for arm in arms
    }
    slate: List[str] = []
    admitted_by: Dict[str, str] = {}
    chosen_families = set()
    score = 0.0

    while len(slate) < k:
        best_arm, best_gain, best_reason = None, -math.inf, "draw"
        for arm in remaining:
            gain = draws[arm]
            reason = "draw"
            if families.get(arm) not in chosen_families and chosen_families:
                # First arm admits on draw alone; later arms can earn the
                # diversity bonus for bringing a NEW family.
                gain += lambda_diversity
                reason = "diversity"
            if gain > best_gain:
                best_arm, best_gain, best_reason = arm, gain, reason
        assert best_arm is not None
        slate.append(best_arm)
        remaining.remove(best_arm)
        admitted_by[best_arm] = best_reason
        family = families.get(best_arm)
        if family is not None:
            chosen_families.add(family)
        score += draws[best_arm] + (
            lambda_diversity if best_reason == "diversity" else 0.0)

    return SlateSelection(
        slate=slate,
        draws=draws,
        families={arm: families.get(arm, "?") for arm in slate},
        admitted_by=admitted_by,
        score=score,
    )
