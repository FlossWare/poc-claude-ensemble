"""Doubly-robust estimator tests: correctness, robustness, variance."""
import random
import unittest

from learning.evaluation.dr_estimator import (
    clipped_doubly_robust_value, doubly_robust_value,
)


def _logged_rows(n=2000, seed=3):
    """Synthetic logging: behaviour picks arm 0 with p=0.7 else arm 1.

    True expected reward: arm0=0.3, arm1=0.8 (arm1 is better but rarer).
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        context = {"x": rng.random()}
        action = 0 if rng.random() < 0.7 else 1
        propensity = 0.7 if action == 0 else 0.3
        mean_reward = 0.3 if action == 0 else 0.8
        reward = mean_reward + rng.uniform(-0.05, 0.05)
        rows.append((context, action, reward, propensity))
    return rows


class DRTests(unittest.TestCase):
    def test_unbiased_when_model_is_wrong(self):
        """DR stays correct even with a badly wrong reward model, because
        the propensities are exact."""
        rows = _logged_rows()
        target = lambda ctx: 1  # always take the better arm
        bad_model = lambda ctx, a: 0.0  # totally wrong
        estimate = doubly_robust_value(rows, target, bad_model)
        # True value of always-arm-1 = 0.8.
        self.assertAlmostEqual(estimate.value, 0.8, delta=0.03)

    def test_unbiased_when_propensities_unknown(self):
        """With a perfect model, the estimate is the model's direct value."""
        rows = _logged_rows()
        target = lambda ctx: 1
        good_model = lambda ctx, a: 0.8 if a == 1 else 0.3
        estimate = doubly_robust_value(rows, target, good_model)
        self.assertAlmostEqual(estimate.value, 0.8, delta=0.05)

    def test_direct_and_ips_components_reported(self):
        rows = _logged_rows()
        target = lambda ctx: 1
        model = lambda ctx, a: 0.8 if a == 1 else 0.3
        estimate = doubly_robust_value(rows, target, model)
        self.assertAlmostEqual(estimate.model_only_value, 0.8, delta=1e-9)
        self.assertGreater(estimate.matched_rows, 0)
        self.assertEqual(estimate.zero_propensity_rows, 0)

    def test_zero_propensity_rows_counted_not_crashed(self):
        rows = [({"x": 1}, 0, 1.0, 0.0)]  # matched row with p=0
        estimate = doubly_robust_value(rows, lambda ctx: 0, lambda c, a: 0.5)
        self.assertEqual(estimate.zero_propensity_rows, 1)
        # Degrades to the model term for that row.
        self.assertAlmostEqual(estimate.value, 0.5)

    def test_clipping_reduces_ips_variance(self):
        """A rare-action row (tiny propensity) must not dominate."""
        rows = _logged_rows()
        # Add one extreme row: behaviour almost never takes arm 1.
        rows.append(({"x": 0}, 1, 1.0, 1e-4))
        target = lambda ctx: 1
        model = lambda ctx, a: 0.5
        unclipped = doubly_robust_value(rows, target, model)
        clipped = clipped_doubly_robust_value(rows, target, model, clip=10.0)
        self.assertLess(abs(clipped.value - 0.8), abs(unclipped.value - 0.8))

    def test_validation(self):
        with self.assertRaises(ValueError):
            doubly_robust_value([], lambda c: 0, lambda c, a: 0.0)
        with self.assertRaises(ValueError):
            clipped_doubly_robust_value(
                [({"x": 1}, 0, 1.0, 0.5)], lambda c: 0, lambda c, a: 0.0, clip=0.5)


if __name__ == "__main__":
    unittest.main()
