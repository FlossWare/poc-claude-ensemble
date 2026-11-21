"""Compact CMA-ES for bounded, small-dimensional continuous search.

Implements the standard (mu/mu_w, lambda)-CMA-ES with rank-one and rank-mu
covariance updates (Hansen & Ostermeier).  Dimensions of interest here are
tiny (the GA's four policy genes), so the full O(n^2) machinery is cheap and
the implementation favours clarity over every last speedup.

Interface: ``CMAES(bounds).optimize(objective)`` where ``objective(vector)``
returns a scalar to MINIMISE (wrap maximisation with a minus sign).  Bounds
are respected by clipping every drawn candidate into the box before
evaluation (boundary points therefore carry some clipping bias; the mean
stays inside by construction).  Deterministic for a seeded
``numpy.random.RandomState``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Objective = Callable[[np.ndarray], float]


@dataclass
class CMAESResult:
    best_vector: np.ndarray
    best_value: float
    evaluations: int
    generations: int
    history: List[float] = field(default_factory=list)


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES over a box-bounded search space."""

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        *,
        sigma0: Optional[float] = None,
        popsize: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not bounds:
            raise ValueError("bounds must define at least one dimension")
        self.lower = np.array([b[0] for b in bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in bounds], dtype=np.float64)
        if np.any(self.upper <= self.lower):
            raise ValueError("every bound must satisfy lower < upper")
        self.dim = len(bounds)
        self.rng = np.random.RandomState(seed)
        # Population: the standard 4 + 3 ln(n) heuristic.
        self.popsize = popsize or int(4 + 3 * math.log(self.dim))
        if self.popsize < 2:
            self.popsize = 2
        self.mu = self.popsize // 2
        # Positive recombination weights (log-scale scheme).
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / float((self.weights ** 2).sum())
        # Step size.
        self.sigma = float(sigma0) if sigma0 else 0.3 * float(
            np.mean(self.upper - self.lower))
        # Covariance state.
        self.c = np.eye(self.dim)
        self.mean = (self.lower + self.upper) / 2.0
        # Adaptation constants (standard).
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                       / ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chi_n = math.sqrt(self.dim) * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self._t = 0

    def _sample_population(self) -> np.ndarray:
        """Draw popsize candidates, clipped into the box for evaluation."""
        try:
            chol = np.linalg.cholesky(self.c)
        except np.linalg.LinAlgError:
            self.c = np.eye(self.dim)  # reset pathological covariance
            chol = np.eye(self.dim)
        draws = self.mean + self.sigma * (
            self.rng.standard_normal((self.popsize, self.dim)) @ chol.T)
        return np.clip(draws, self.lower, self.upper)

    def _update(self, population: np.ndarray, values: np.ndarray) -> None:
        """Rank-based mean, step-size and covariance adaptation."""
        order = np.argsort(values)  # minimise
        elite = population[order[: self.mu]]
        old_mean = self.mean.copy()
        new_mean = self.weights @ elite
        y = (new_mean - old_mean) / max(self.sigma, 1e-12)

        # Conjugate evolution path (step size).  With C = L L^T the
        # whitened step is L^-1 y (NOT L^-T y).
        chol = np.linalg.cholesky(self.c)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mu_eff) * np.linalg.solve(chol, y)
        self.sigma *= math.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = min(max(self.sigma, 1e-9), float(
            np.mean(self.upper - self.lower)))

        # Covariance evolution path.  Hansen's stalled-path test divides
        # the normalised path length by chi_n before the threshold.
        path_norm = np.linalg.norm(self.ps) / math.sqrt(
            1 - (1 - self.cs) ** (2 * (self._t + 1)))
        h = 1.0 if path_norm / self.chi_n < (1.4 + 2 / (self.dim + 1)) else 0.0
        self.pc = (1 - self.cc) * self.pc + h * math.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff) * y

        # Rank-one + rank-mu updates.  When the path stalls (h=0) the
        # variance-loss compensation ADDS cc(2-cc) C (Hansen eq. 47).
        pc_pc = np.outer(self.pc, self.pc)
        rank_one = self.c1 * (
            pc_pc + (1 - h) * self.cc * (2 - self.cc) * self.c)
        yi = (elite - old_mean) / max(self.sigma, 1e-12)
        rank_mu = self.cmu * (
            sum(self.weights[i] * np.outer(yi[i], yi[i])
                for i in range(self.mu))
        )
        self.c = (1 - self.c1 - self.cmu) * self.c + rank_one + rank_mu
        # Symmetrise and condition-guard the covariance.
        self.c = (self.c + self.c.T) / 2
        eigvals = np.linalg.eigvalsh(self.c)
        if eigvals.min() <= 0:
            self.c = np.eye(self.dim) * max(eigvals.max(), 1.0)

        self.mean = np.clip(new_mean, self.lower, self.upper)
        self._t += 1

    def optimize(
        self,
        objective: Objective,
        *,
        max_generations: int = 200,
        tolerance: float = 1e-10,
    ) -> CMAESResult:
        """Run until the best value stops improving or the budget runs out."""
        best_vector = self.mean.copy()
        best_value = float("inf")
        history: List[float] = []
        evaluations = 0
        for _ in range(max_generations):
            population = self._sample_population()
            values = np.array([objective(x) for x in population])
            evaluations += len(population)
            generation_best = float(values.min())
            if generation_best < best_value:
                if best_value - generation_best < tolerance * max(1.0, abs(best_value)):
                    best_value = generation_best
                    best_vector = population[int(values.argmin())].copy()
                    history.append(best_value)
                    break
                best_value = generation_best
                best_vector = population[int(values.argmin())].copy()
            history.append(best_value)
            self._update(population, values)
        return CMAESResult(
            best_vector=best_vector,
            best_value=best_value,
            evaluations=evaluations,
            generations=len(history),
            history=history,
        )
