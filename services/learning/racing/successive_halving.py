"""Successive halving over a bounded evaluation budget.

``race(arms, evaluate, budget, eta=3)`` runs one bracket of successive
halving:

* Round r gives every surviving arm ``n_r`` probes, where the per-round
  allocation grows geometrically by ``eta`` and the total is bounded by
  ``budget`` probe evaluations.
* After each round only the top ``1/eta`` of arms (by mean observed reward)
  survive; ties break deterministically on arm name so runs are reproducible.
* The race ends when one arm remains or the budget is exhausted; the result
  ranks the surviving arms by mean reward with their probe counts.

The ``evaluate(arm, round_index, probe_index) -> float`` callback returns a
reward in [0, 1] for one probe.  It is the caller's business how a probe is
realised -- for cold-start that is "run the model on one replayed task and
let the arbiter grade it" (see tools/replay_new_model.py).  The algorithm
itself is pure: no I/O, no state, fully determined by the callback's
behaviour, which is what makes it testable with synthetic evaluators.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

Evaluate = Callable[[str, int, int], float]


class RaceResult(Tuple[List[Tuple[str, float, int]], int]):
    """Ranked survivors (arm, mean_reward, probes) and total probes spent."""


def _plan(num_arms: int, budget: int, eta: int) -> List[int]:
    """Per-round probe counts so the whole bracket fits the budget.

    Classic successive halving with total resource R allocates
    n_r = ceil(eta^r * log_eta(K) / ... ) probes; here the closed form is
    replaced with a simple geometric plan normalised to the budget, which
    has the same shape (early rounds cheap, late rounds expensive) and
    respects the caller's hard limit.
    """
    if num_arms <= 1 or budget < num_arms:
        # Degenerate: one probe each, no elimination.
        return [max(1, budget // max(1, num_arms))]
    rounds = max(1, int(math.ceil(math.log(num_arms, eta))))
    # Geometric weights 1, eta, eta^2, ... normalised to the budget.
    weights = [eta ** r for r in range(rounds)]
    scale = budget / sum(w * max(1, num_arms // (eta ** r)) for r, w in enumerate(weights))
    plan: List[int] = []
    surviving = num_arms
    for r in range(rounds):
        if surviving <= 1:
            break
        per_arm = max(1, int(round(scale * eta ** r)))
        plan.append(per_arm)
        surviving = max(1, surviving // eta)
    return plan or [1]


def race(
    arms: Sequence[str],
    evaluate: Evaluate,
    budget: int,
    eta: int = 3,
) -> RaceResult:
    """Run one successive-halving bracket.  Never exceeds ``budget`` probes.

    Raises ``ValueError`` for an empty field, a budget below one probe per
    arm (nothing can be learned), or eta < 2.
    """
    if not arms:
        raise ValueError("no arms to race")
    if len(set(arms)) != len(arms):
        raise ValueError("arms must be unique")
    if eta < 2:
        raise ValueError("eta must be >= 2")
    if budget < len(arms):
        raise ValueError(
            f"budget {budget} cannot even probe each of {len(arms)} arms once"
        )

    survivors = sorted(arms)  # deterministic order for tie-breaking
    totals = {arm: [0.0, 0] for arm in arms}  # reward sum, probe count
    spent = 0
    plan = _plan(len(arms), budget, eta)

    for round_index, per_arm in enumerate(plan):
        if len(survivors) <= 1:
            break
        # Never blow the budget on the last round.
        per_arm = min(per_arm, max(0, (budget - spent) // len(survivors)))
        if per_arm <= 0:
            break
        round_means: List[Tuple[float, str]] = []
        for arm in survivors:
            for probe in range(per_arm):
                reward = evaluate(arm, round_index, probe)
                if not 0.0 <= reward <= 1.0:
                    raise ValueError(
                        f"evaluate({arm!r}) returned {reward!r}; rewards are [0, 1]"
                    )
                totals[arm][0] += reward
                totals[arm][1] += 1
                spent += 1
            mean = totals[arm][0] / totals[arm][1]
            round_means.append((mean, arm))
        # Top 1/eta advance; ties resolved by the deterministic arm order.
        round_means.sort(key=lambda item: (-item[0], item[1]))
        keep = max(1, len(survivors) // eta)
        survivors = [arm for _, arm in round_means[:keep]]

    ranked = sorted(
        ((arm, totals[arm][0] / totals[arm][1] if totals[arm][1] else 0.0,
          totals[arm][1]) for arm in arms),
        key=lambda item: (-item[1], item[0]),
    )
    return RaceResult((ranked, spent))
