"""Replay fitness and the ``random`` search strategy for Target A.

Fitness is:

    F = succ_hat - lambda * cost_hat / cost_ref - mu * unsupported_fraction

with ``lambda = 0.25`` and ``mu = 0.10``.  These are run configuration and are
deliberately **not** evolvable genes; making the objective tunable would let
the search optimise its own scoring rule.

Two estimators are used, because the two corpora carry different information:

``full_information_replay``
    Review rounds run every worker on the same subject, so whichever arm the
    candidate picks has a realized outcome in that round.  The candidate sees
    only statistics from prior rounds; scoring uses the realized row.

``logged_bandit_direct_method``
    Turn outcomes record only the arm that actually ran.  For each holdout
    step the candidate selects from pre-step statistics; the reward is the
    holdout-window mean of the chosen arm when that arm has at least
    ``min_support`` holdout rows, otherwise the train mean plus a support
    penalty.  The policy never sees those means.  Realized-agreement coverage
    is reported separately.  No propensities are logged, so IPS is unavailable.

The first step of a replay has no prior observations and therefore no defined
selection; such steps are skipped and counted, never invented.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from learning.ga import corpus as corpus_mod
from learning.ga.schema import (
    PARAM_BOUNDS,
    candidate_hash,
    clamp,
    default_candidate,
    make_candidate,
)
from shared.thompson_policy import Arm, select as policy_select

COST_LAMBDA = 0.25
UNSUPPORTED_MU = 0.10
MIN_SUPPORT = 30
MIN_REVIEW_SUPPORT = 1
INF = float("inf")


class BudgetExceeded(Exception):
    """The wall-clock budget for a run expired."""


@dataclass(frozen=True)
class SearchConfig:
    population: int = 24
    generations: int = 6
    max_evals: int = 144
    patience: int = 3
    min_delta: float = 1e-3
    sigma_fraction: float = 0.15
    elites: int = 2
    holdout_fraction: float = 0.30
    cost_lambda: float = COST_LAMBDA
    unsupported_mu: float = UNSUPPORTED_MU
    min_support: int = MIN_SUPPORT
    min_review_support: int = MIN_REVIEW_SUPPORT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population": self.population,
            "generations": self.generations,
            "max_evals": self.max_evals,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "sigma_fraction": self.sigma_fraction,
            "elites": self.elites,
            "holdout_fraction": self.holdout_fraction,
            "cost_lambda": self.cost_lambda,
            "unsupported_mu": self.unsupported_mu,
            "min_support": self.min_support,
            "min_review_support": self.min_review_support,
        }


@dataclass
class EvalResult:
    window: str
    estimator: str
    decisions: int = 0
    skipped_steps: int = 0
    succ_hat: float = 0.0
    cost_hat: float = 0.0
    cost_ref: float = 1.0
    unsupported_fraction: float = 0.0
    coverage: float = 0.0
    fitness: float = 0.0
    per_arm: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contributions: List[float] = field(default_factory=list)
    block_keys: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "estimator": self.estimator,
            "decisions": self.decisions,
            "skipped_steps": self.skipped_steps,
            "succ_hat": self.succ_hat,
            "cost_hat": self.cost_hat,
            "cost_ref": self.cost_ref,
            "unsupported_fraction": self.unsupported_fraction,
            "coverage": self.coverage,
            "fitness": self.fitness,
            "per_arm": {name: dict(stats) for name, stats in sorted(self.per_arm.items())},
        }


@dataclass
class RewardModel:
    success: Dict[str, float]
    cost: Dict[str, float]
    support: Dict[str, int]
    fallback_success: float
    fallback_cost: float

    def estimate(self, arm: str, min_support: int) -> Tuple[float, float, bool]:
        if self.support.get(arm, 0) >= min_support:
            return self.success[arm], self.cost[arm], True
        return self.fallback_success, self.fallback_cost, False


def build_reward_model(rows: Sequence[corpus_mod.Row], min_support: int) -> RewardModel:
    support: Dict[str, int] = {}
    successes: Dict[str, int] = {}
    costs: Dict[str, float] = {}
    for row in rows:
        support[row.arm] = support.get(row.arm, 0) + 1
        successes[row.arm] = successes.get(row.arm, 0) + (1 if row.success else 0)
        costs[row.arm] = costs.get(row.arm, 0.0) + row.cost
    total = len(rows)
    fallback_success = (sum(successes.values()) / total) if total else 0.0
    fallback_cost = (sum(costs.values()) / total) if total else 0.0
    return RewardModel(
        success={arm: successes[arm] / support[arm] for arm in support},
        cost={arm: costs[arm] / support[arm] for arm in support},
        support=dict(support),
        fallback_success=fallback_success,
        fallback_cost=fallback_cost,
    )


def _arms_from_state(state: Mapping[str, List[float]]) -> List[Arm]:
    arms = []
    for name, (successes, failures, total_cost, calls) in state.items():
        avg_cost = (total_cost / calls) if calls else 0.0
        arms.append(Arm(name, int(successes), int(failures), avg_cost, int(calls)))
    return arms


def _update_state(state: Dict[str, List[float]], row: corpus_mod.Row) -> None:
    entry = state.get(row.arm)
    if entry is None:
        entry = state[row.arm] = [0.0, 0.0, 0.0, 0.0]
    if row.success:
        entry[0] += 1
    else:
        entry[1] += 1
    entry[2] += row.cost
    entry[3] += 1


def _per_arm_stats(decisions: Dict[str, int], model: RewardModel) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for arm in sorted(set(model.support) | set(decisions)):
        stats[arm] = {
            "decisions": int(decisions.get(arm, 0)),
            "logged_rows": int(model.support.get(arm, 0)),
            "logged_success": model.success.get(arm),
            "logged_cost": model.cost.get(arm),
        }
    return stats


def _finalise(
    result: EvalResult,
    cfg: SearchConfig,
    decisions: int,
    unsupported: int,
    realized: int,
    success_total: float,
    cost_total: float,
) -> EvalResult:
    result.decisions = decisions
    if decisions:
        result.succ_hat = success_total / decisions
        result.cost_hat = cost_total / decisions
        result.fitness = (
            result.succ_hat
            - cfg.cost_lambda * (result.cost_hat / result.cost_ref)
            - cfg.unsupported_mu * (unsupported / decisions)
        )
    result.unsupported_fraction = (unsupported / decisions) if decisions else 0.0
    result.coverage = (realized / decisions) if decisions else 0.0
    return result


def replay_turn(
    params: Mapping[str, Any],
    corpus: corpus_mod.Corpus,
    window: str,
    seed: int,
    cfg: SearchConfig | None = None,
    trace: bool = False,
) -> EvalResult:
    """Logged-bandit replay of the turn corpus for one window."""
    cfg = cfg or SearchConfig()
    rows = corpus.rows
    train, holdout = corpus_mod.split_rows(rows, cfg.holdout_fraction)
    scored = holdout if window == "holdout" else train
    scored_keys = {row.row_key for row in scored}
    model = build_reward_model(scored, cfg.min_support)
    cost_ref = model.fallback_cost if model.fallback_cost > 0.0 else 1.0

    result = EvalResult(window=window, estimator="logged_bandit_direct_method", cost_ref=cost_ref)
    state: Dict[str, List[float]] = {}
    decisions = unsupported = realized = 0
    success_total = cost_total = 0.0
    decision_index = 0
    per_arm_decisions: Dict[str, int] = {}

    for row in rows:
        if row.row_key in scored_keys:
            prior_observations = sum(entry[3] for entry in state.values())
            if prior_observations == 0:
                result.skipped_steps += 1
            else:
                rng = random.Random(f"{seed}|turn|{window}|{decision_index}")
                chosen = policy_select(params, _arms_from_state(state), None, 0.0, INF, rng)
                decision_index += 1
                est_success, est_cost, supported = model.estimate(chosen, cfg.min_support)
                contribution = (
                    est_success
                    - cfg.cost_lambda * (est_cost / cost_ref)
                    - (0.0 if supported else cfg.unsupported_mu)
                )
                result.contributions.append(contribution)
                result.block_keys.append(row.day)
                success_total += est_success
                cost_total += est_cost
                if not supported:
                    unsupported += 1
                if model.support.get(chosen, 0) > 0:
                    realized += 1
                per_arm_decisions[chosen] = per_arm_decisions.get(chosen, 0) + 1
                decisions += 1
                if trace:
                    result.trace.append(
                        {
                            "row_key": row.row_key,
                            "step": decision_index - 1,
                            "arm": chosen,
                            "prior_calls": int(state.get(chosen, [0, 0, 0, 0])[3]),
                            "prior_total": int(prior_observations),
                        }
                    )
        _update_state(state, row)

    result.per_arm = _per_arm_stats(per_arm_decisions, model)
    return _finalise(result, cfg, decisions, unsupported, realized, success_total, cost_total)


def replay_review(
    params: Mapping[str, Any],
    corpus: corpus_mod.Corpus,
    window: str,
    seed: int,
    cfg: SearchConfig | None = None,
    trace: bool = False,
) -> EvalResult:
    """Full-information replay of the review corpus for one window.

    The candidate picks one arm per round from prior-round statistics and is
    scored on that arm's realized success and cost in the round.  Statistics
    are then updated with every row in the round, mirroring
    ``tools/ledger_to_thompson.py``.
    """
    cfg = cfg or SearchConfig()
    rounds = corpus.rounds()
    train, holdout = corpus_mod.split_rounds(rounds, cfg.holdout_fraction)
    scored = holdout if window == "holdout" else train
    scored_ids = {review_id for review_id, _ in scored}

    scored_rows = [row for _rid, rows in scored for row in rows]
    total = len(scored_rows)
    cost_ref = (sum(row.cost for row in scored_rows) / total) if total else 1.0
    if cost_ref <= 0.0:
        cost_ref = 1.0
    logged = build_reward_model(scored_rows, cfg.min_review_support)

    result = EvalResult(window=window, estimator="full_information_replay", cost_ref=cost_ref)
    state: Dict[str, List[float]] = {}
    decisions = unsupported = realized = 0
    success_total = cost_total = 0.0
    decision_index = 0
    per_arm_decisions: Dict[str, int] = {}

    for review_id, rows in rounds:
        if review_id in scored_ids:
            prior_observations = sum(entry[3] for entry in state.values())
            if prior_observations == 0:
                result.skipped_steps += 1
            else:
                rng = random.Random(f"{seed}|review|{window}|{decision_index}")
                chosen = policy_select(params, _arms_from_state(state), None, 0.0, INF, rng)
                decision_index += 1
                prior_calls = int(state.get(chosen, [0, 0, 0, 0])[3])
                realized_row = next((row for row in rows if row.arm == chosen), None)
                if realized_row is None:
                    result.skipped_steps += 1
                else:
                    contribution = (
                        (1.0 if realized_row.success else 0.0)
                        - cfg.cost_lambda * (realized_row.cost / cost_ref)
                        - (0.0 if prior_calls >= cfg.min_review_support else cfg.unsupported_mu)
                    )
                    result.contributions.append(contribution)
                    result.block_keys.append(realized_row.day)
                    success_total += 1.0 if realized_row.success else 0.0
                    cost_total += realized_row.cost
                    if prior_calls < cfg.min_review_support:
                        unsupported += 1
                    realized += 1
                    per_arm_decisions[chosen] = per_arm_decisions.get(chosen, 0) + 1
                    decisions += 1
                    if trace:
                        result.trace.append(
                            {
                                "review_id": review_id,
                                "step": decision_index - 1,
                                "arm": chosen,
                                "prior_calls": prior_calls,
                                "prior_total": int(prior_observations),
                            }
                        )
        for row in rows:
            _update_state(state, row)

    result.per_arm = _per_arm_stats(per_arm_decisions, logged)
    return _finalise(result, cfg, decisions, unsupported, realized, success_total, cost_total)


# ---------------------------------------------------------------------------
# Search: Latin hypercube + top-k + Gaussian refine + elitism
# ---------------------------------------------------------------------------


@dataclass
class SearchOutcome:
    best_params: Dict[str, float]
    best_fitness: float
    evaluations: int
    generations: List[Dict[str, Any]]
    stopped: str
    cache: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "generations": self.generations,
            "stopped": self.stopped,
        }


def latin_hypercube(rng: random.Random, count: int) -> List[Dict[str, float]]:
    """``count`` stratified samples over the gene box (one stratum per gene)."""
    if count <= 0:
        return []
    strata: Dict[str, List[float]] = {}
    for name in PARAM_BOUNDS:
        points = [(index + rng.random()) / count for index in range(count)]
        rng.shuffle(points)
        strata[name] = points
    samples = []
    for index in range(count):
        params = {}
        for name, spec in PARAM_BOUNDS.items():
            params[name] = clamp(name, spec.low + (spec.high - spec.low) * strata[name][index])
        samples.append(make_candidate(params))
    return samples


def mutate(base: Mapping[str, Any], rng: random.Random, cfg: SearchConfig) -> Dict[str, float]:
    """Gaussian refine: perturb each gene with probability 0.5, clipped to bounds."""
    child = dict(base)
    changed = False
    for name, spec in PARAM_BOUNDS.items():
        if rng.random() < 0.5:
            sigma = cfg.sigma_fraction * (spec.high - spec.low)
            child[name] = clamp(name, child[name] + rng.gauss(0.0, sigma))
            changed = True
    if not changed:
        name = rng.choice(list(PARAM_BOUNDS))
        spec = PARAM_BOUNDS[name]
        sigma = cfg.sigma_fraction * (spec.high - spec.low)
        child[name] = clamp(name, child[name] + rng.gauss(0.0, sigma))
    return make_candidate(child)


def search_turn(
    baseline: Mapping[str, Any],
    corpus: corpus_mod.Corpus,
    cfg: SearchConfig,
    seed: int,
    deadline: Optional[float] = None,
    clock: Callable[[], float] | None = None,
) -> SearchOutcome:
    """Search the train window of the turn corpus.

    Generation 0 is the default candidate plus a Latin-hypercube sample;
    later generations keep the top ``elites`` plus the default and fill the
    rest with Gaussian perturbations of an elite.  Evaluation is capped by
    ``max_evals`` and by the wall-clock ``deadline``.
    """
    import time

    clock = clock or time.monotonic
    rng = random.Random(f"{seed}|search|turn")
    cache: Dict[str, float] = {}
    evaluations = 0

    def evaluate(params: Mapping[str, Any]) -> Tuple[str, float]:
        nonlocal evaluations
        candidate = make_candidate(params)
        key = candidate_hash(candidate)
        if key not in cache:
            if deadline is not None and clock() > deadline:
                raise BudgetExceeded("wall-clock budget exhausted during search")
            cache[key] = replay_turn(candidate, corpus, "train", seed, cfg).fitness
            evaluations += 1
        return key, cache[key]

    default = default_candidate()
    population: List[Dict[str, float]] = [default]
    population.extend(latin_hypercube(rng, max(0, cfg.population - 1)))
    population = _dedupe(population)[: cfg.max_evals]

    generations: List[Dict[str, Any]] = []
    best_params = default
    best_fitness = -INF
    stale = 0
    stopped = "max_generations"

    for generation in range(cfg.generations):
        scored: List[Tuple[float, str, Dict[str, float]]] = []
        for candidate in population:
            key, fitness = evaluate(candidate)
            scored.append((fitness, key, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        generation_best = scored[0][0]
        generations.append(
            {
                "generation": generation,
                "evaluated": len(scored),
                "best_fitness": generation_best,
                "best_candidate": scored[0][1],
            }
        )

        if generation_best > best_fitness + cfg.min_delta:
            best_fitness = generation_best
            best_params = dict(scored[0][2])
            stale = 0
        else:
            if generation_best > best_fitness:
                best_fitness = generation_best
                best_params = dict(scored[0][2])
            stale += 1

        if evaluations >= cfg.max_evals:
            stopped = "max_evals"
            break
        if stale >= cfg.patience:
            stopped = "patience"
            break
        if generation == cfg.generations - 1:
            stopped = "max_generations"
            break
        if deadline is not None and clock() > deadline:
            stopped = "deadline"
            break

        elites = [candidate for _fitness, _key, candidate in scored[: cfg.elites]]
        parents = _dedupe(elites + [default])
        children: List[Dict[str, float]] = []
        seen = {candidate_hash(candidate) for candidate in parents}
        attempts = 0
        while len(parents) + len(children) < cfg.population and attempts < cfg.population * 20:
            attempts += 1
            base = rng.choice(elites)
            child = mutate(base, rng, cfg)
            key = candidate_hash(child)
            if key in seen:
                continue
            seen.add(key)
            children.append(child)
        population = parents + children

    return SearchOutcome(
        best_params=make_candidate(best_params),
        best_fitness=best_fitness if best_fitness > -INF else 0.0,
        evaluations=evaluations,
        generations=generations,
        stopped=stopped,
        cache=cache,
    )


def _dedupe(candidates: Iterable[Mapping[str, Any]]) -> List[Dict[str, float]]:
    seen: set = set()
    unique: List[Dict[str, float]] = []
    for candidate in candidates:
        normalised = make_candidate(candidate)
        key = candidate_hash(normalised)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalised)
    return unique


__all__ = [
    "BudgetExceeded",
    "COST_LAMBDA",
    "EvalResult",
    "MIN_REVIEW_SUPPORT",
    "MIN_SUPPORT",
    "RewardModel",
    "SearchConfig",
    "SearchOutcome",
    "UNSUPPORTED_MU",
    "build_reward_model",
    "latin_hypercube",
    "mutate",
    "replay_review",
    "replay_turn",
    "search_turn",
]
