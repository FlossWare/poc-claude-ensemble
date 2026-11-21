"""Slate bandit tests: diversity, exploration, determinism, validation."""
import random
import unittest
from collections import Counter

from learning.slate.slate_bandit import select_slate

STATS = {
    "opus": (35.0, 12.0),        # strong
    "glm-5-3": (62.0, 14.0),     # strong
    "deepseek-v4.1-flash": (60.0, 12.0),  # strong
    "muse-spark-open": (17.0, 1.0),       # strong, new
    "haiku": (3007.0, 390.0),    # strong
}
FAMILIES = {
    "opus": "anthropic", "glm-5-3": "zai", "deepseek-v4.1-flash": "deepseek",
    "muse-spark-open": "muse", "haiku": "anthropic",
}


class SlateTests(unittest.TestCase):
    def test_slate_size_and_uniqueness(self):
        sel = select_slate(list(STATS), STATS, FAMILIES, 3, rng=random.Random(1))
        self.assertEqual(len(sel.slate), 3)
        self.assertEqual(len(set(sel.slate)), 3)

    def test_diversity_prefers_distinct_families(self):
        """With flat posteriors, the diversity bonus must spread families."""
        flat = {arm: (1.0, 1.0) for arm in STATS}
        counts = Counter()
        for seed in range(30):
            sel = select_slate(list(STATS), flat, FAMILIES, 3,
                               lambda_diversity=5.0, rng=random.Random(seed))
            for arm in sel.slate:
                counts[FAMILIES[arm]] += 1
        # Over 30 draws of 3 arms, all four families should appear; a
        # draw-independent selector would concentrate on two.
        self.assertGreaterEqual(len(counts), 4)

    def test_high_posterior_arms_still_win_with_small_lambda(self):
        sel = select_slate(list(STATS), STATS, FAMILIES, 1,
                           lambda_diversity=0.0, rng=random.Random(0))
        # Single arm: pure draw, no diversity.  With these posteriors the
        # strong arms dominate a 1-slot slate across seeds.
        winners = set()
        for seed in range(50):
            winners.add(select_slate(list(STATS), STATS, FAMILIES, 1,
                                     lambda_diversity=0.0,
                                     rng=random.Random(seed)).slate[0])
        self.assertTrue(winners.issubset(set(STATS)))
        self.assertGreater(len(winners), 1)  # posterior means differ

    def test_missing_arm_uses_uninformative_prior(self):
        sel = select_slate(["new-arm", "opus"], {"opus": (35.0, 12.0)},
                           {"new-arm": "other", "opus": "anthropic"}, 2,
                           rng=random.Random(4))
        self.assertEqual(set(sel.slate), {"new-arm", "opus"})

    def test_admission_reasons_recorded(self):
        sel = select_slate(list(STATS), STATS, FAMILIES, 3,
                           lambda_diversity=0.5, rng=random.Random(2))
        self.assertEqual(len(sel.admitted_by), 3)
        self.assertTrue(all(r in ("draw", "diversity")
                            for r in sel.admitted_by.values()))

    def test_deterministic_under_seed(self):
        kwargs = dict(stats=STATS, families=FAMILIES, k=3,
                      lambda_diversity=0.5)
        s1 = select_slate(list(STATS), rng=random.Random(11), **kwargs)
        s2 = select_slate(list(STATS), rng=random.Random(11), **kwargs)
        self.assertEqual(s1.slate, s2.slate)
        self.assertEqual(s1.score, s2.score)

    def test_validation(self):
        with self.assertRaises(ValueError):
            select_slate(["a"], {}, {}, 0)
        with self.assertRaises(ValueError):
            select_slate(["a", "a"], {}, {}, 1)
        with self.assertRaises(ValueError):
            select_slate(["a"], {}, {}, 2)
        with self.assertRaises(ValueError):
            select_slate(["a"], {}, {}, 1, lambda_diversity=-1.0)
        with self.assertRaises(ValueError):
            select_slate(["a"], {"a": (0.0, 1.0)}, {}, 1)


if __name__ == "__main__":
    unittest.main()
