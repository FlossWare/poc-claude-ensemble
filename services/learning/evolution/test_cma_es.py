"""CMA-ES tests: convergence, bounds, determinism, validation."""
import unittest

import numpy as np

from learning.evolution.cma_es import CMAES


class ConvergenceTests(unittest.TestCase):
    def test_sphere_converges_to_minimum(self):
        cma = CMAES([(-5, 5)] * 3, seed=42)
        result = cma.optimize(lambda x: float(np.sum(x ** 2)), max_generations=150)
        self.assertLess(result.best_value, 1e-3)
        np.testing.assert_allclose(result.best_vector, np.zeros(3), atol=0.05)

    def test_elliptical_objective_converges(self):
        # Strongly scaled axes: CMA-ES must adapt the covariance shape.
        cma = CMAES([(-10, 10)] * 2, seed=7)
        result = cma.optimize(
            lambda x: float(x[0] ** 2 + 100 * x[1] ** 2), max_generations=300)
        self.assertLess(result.best_value, 1.0)

    def test_bounds_always_respected(self):
        cma = CMAES([(0.0, 1.0)] * 4, seed=1)
        seen = []

        def objective(x):
            seen.append(x.copy())
            return float(np.sum((x - 0.5) ** 2))

        cma.optimize(objective, max_generations=50)
        for x in seen:
            self.assertTrue(np.all(x >= 0.0) and np.all(x <= 1.0))

    def test_deterministic_under_seed(self):
        def run(seed):
            cma = CMAES([(-2, 2)] * 2, seed=seed)
            return cma.optimize(lambda x: float(np.sum(x ** 2)), max_generations=30)
        r1 = run(9)
        r2 = run(9)
        np.testing.assert_array_equal(r1.best_vector, r2.best_vector)
        self.assertEqual(r1.evaluations, r2.evaluations)


class ValidationTests(unittest.TestCase):
    def test_rejects_empty_bounds(self):
        with self.assertRaises(ValueError):
            CMAES([])

    def test_rejects_empty_interval(self):
        with self.assertRaises(ValueError):
            CMAES([(1.0, 1.0)])

    def test_rejects_inverted_interval(self):
        with self.assertRaises(ValueError):
            CMAES([(2.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
