"""Successive halving tests: correctness, budget discipline, determinism."""
import statistics
import unittest

from learning.racing.successive_halving import race, _plan


class PlanTests(unittest.TestCase):
    def test_plan_is_geometric_with_exact_values(self):
        # 9 arms, budget 90, eta 3: two rounds (ceil(log_3(9))), weights
        # 1:3 normalised to the budget.  Per-arm probes [5, 15] -- exactly
        # the eta ratio -- and the third round never runs (one survivor
        # after round 2).
        plan = _plan(9, 90, 3)
        self.assertEqual(plan, [5, 15])
        self.assertEqual(plan[1] / plan[0], 3)  # geometric growth at eta

    def test_degenerate_fields(self):
        self.assertEqual(_plan(1, 10, 3), [10])
        # budget below arms: one probe each.  race() rejects this case with
        # ValueError; _plan alone has no budget to enforce against.
        self.assertEqual(_plan(4, 5, 3), [1])


class RaceTests(unittest.TestCase):
    def test_finds_best_arm_with_fraction_of_full_budget(self):
        """9 arms, best arm has mean 0.9; others 0.1-0.5.  Full evaluation
        of every arm on every probe would need many times more calls."""
        true_means = {
            "best": 0.9, "a": 0.5, "b": 0.45, "c": 0.4, "d": 0.35,
            "e": 0.3, "f": 0.25, "g": 0.2, "h": 0.1,
        }
        rng_state = {"value": 0}

        def evaluate(arm, round_index, probe_index):
            # Deterministic pseudo-noise in [-0.05, 0.05].
            rng_state["value"] = (rng_state["value"] * 1103515245 + 12345) % 2**31
            noise = (rng_state["value"] / 2**31 - 0.5) * 0.1
            return min(1.0, max(0.0, true_means[arm] + noise))

        ranked, spent = race(sorted(true_means), evaluate, budget=27, eta=3)
        self.assertLessEqual(spent, 27)
        self.assertEqual(ranked[0][0], "best")
        self.assertTrue(ranked[0][3])  # winner is a survivor
        # Full evaluation at the same probe depth would be 9 arms x many
        # probes; the race spent at most the budget.
        self.assertLess(spent, 9 * 6)

    def test_budget_never_exceeded(self):
        calls = []

        def evaluate(arm, round_index, probe_index):
            calls.append(arm)
            return 0.5 if arm == "winner" else 0.1

        ranked, spent = race(["a", "b", "c", "d", "winner"], evaluate, budget=7, eta=2)
        self.assertEqual(spent, len(calls))
        self.assertLessEqual(spent, 7)

    def test_reward_validation(self):
        def evaluate(arm, round_index, probe_index):
            return 1.5

        with self.assertRaises(ValueError):
            race(["a", "b"], evaluate, budget=4)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            race([], lambda *a: 0.5, budget=10)
        with self.assertRaises(ValueError):
            race(["a", "a"], lambda *a: 0.5, budget=10)
        with self.assertRaises(ValueError):
            race(["a", "b"], lambda *a: 0.5, budget=1)
        with self.assertRaises(ValueError):
            race(["a", "b"], lambda *a: 0.5, budget=10, eta=1)

    def test_deterministic_tie_break(self):
        def evaluate(arm, round_index, probe_index):
            return 0.5  # all arms tie

        ranked1, _ = race(["b", "a", "c"], evaluate, budget=12)
        ranked2, _ = race(["c", "b", "a"], evaluate, budget=12)
        self.assertEqual([arm for arm, _, _, _ in ranked1], [arm for arm, _, _, _ in ranked2])

    def test_ranked_output_shape_and_survivor_flag(self):
        def evaluate(arm, round_index, probe_index):
            return {"x": 0.8, "y": 0.4}[arm]

        ranked, spent = race(["x", "y"], evaluate, budget=10)
        self.assertEqual(ranked[0][0], "x")
        self.assertEqual(ranked[1][0], "y")
        self.assertGreater(ranked[0][2], 0)  # probe counts recorded
        self.assertTrue(ranked[0][3])  # x survived
        self.assertFalse(ranked[1][3])  # y was eliminated

    def test_single_arm_reports_unprobed_none(self):
        """One arm = no race to run; mean is None, not a fabricated 0.0."""
        def evaluate(arm, round_index, probe_index):  # pragma: no cover
            raise AssertionError("single-arm race must not probe")

        ranked, spent = race(["solo"], evaluate, budget=10)
        self.assertEqual(spent, 0)
        self.assertEqual(ranked[0], ("solo", None, 0, True))


if __name__ == "__main__":
    unittest.main()
