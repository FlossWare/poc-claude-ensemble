"""Linear Thompson Sampling tests: math, persistence, and context separation."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from learning.contextual.linear_ts import (
    DIM, FEATURE_NAMES, LinearTS, features, rewards_from_outcomes,
)


class FeatureTests(unittest.TestCase):
    def test_fixed_dimension_and_determinism(self):
        x1 = features("code-review", 100)
        x2 = features("code-review", 100)
        self.assertEqual(x1.shape, (DIM,))
        self.assertTrue(np.array_equal(x1, x2))

    def test_bias_and_one_hot(self):
        x = features("documentation")
        self.assertEqual(x[0], 1.0)
        self.assertEqual(int(x.sum()), 2)  # bias + one task type
        x2 = features("no-such-task")
        self.assertEqual(x2[FEATURE_NAMES.index("__other__")], 1.0)

    def test_long_input_indicator(self):
        self.assertEqual(features("analysis", 100)[FEATURE_NAMES.index("long_input")], 0.0)
        self.assertEqual(features("analysis", 5000)[FEATURE_NAMES.index("long_input")], 1.0)


class PolicyTests(unittest.TestCase):
    def test_select_returns_a_requested_arm(self):
        lts = LinearTS("/nonexistent/state.json")
        arms = ["glm-5-3", "deepseek-v4.1-flash", "haiku"]
        rng = np.random.RandomState(42)
        for _ in range(20):
            self.assertIn(lts.select(arms, "code-review", rng=rng), arms)

    def test_update_moves_posterior_mean_toward_reward(self):
        lts = LinearTS("/nonexistent/state.json")
        lts.arm("haiku")  # create the arm so mean_scores covers it
        before = lts.mean_scores("code-review")["haiku"]
        self.assertEqual(before, 0.0)  # prior mean
        for _ in range(30):
            lts.update("haiku", "code-review", 1.0)
        after = lts.mean_scores("code-review")["haiku"]
        self.assertGreater(after, before)

    def test_context_separation(self):
        """An arm that wins only for code-review must not dominate elsewhere."""
        lts = LinearTS("/nonexistent/state.json")
        # haiku is perfect for code-review, terrible for documentation.
        # glm is mediocre (0.5) everywhere.
        for _ in range(60):
            lts.update("haiku", "code-review", 1.0)
            lts.update("haiku", "documentation", 0.0)
            lts.update("glm-5-3", "code-review", 0.5)
            lts.update("glm-5-3", "documentation", 0.5)
        code_means = lts.mean_scores("code-review")
        doc_means = lts.mean_scores("documentation")
        self.assertGreater(code_means["haiku"], code_means["glm-5-3"])
        self.assertGreater(doc_means["glm-5-3"], doc_means["haiku"])

    def test_draws_explore(self):
        """Identical arms get different theta draws (posterior variance)."""
        lts = LinearTS("/nonexistent/state.json")
        arm = lts.arm("fresh")
        rng = np.random.RandomState(0)
        draws = [arm.draw(rng) for _ in range(5)]
        self.assertFalse(all(np.array_equal(draws[0], d) for d in draws[1:]))

    def test_reward_validation(self):
        lts = LinearTS("/nonexistent/state.json")
        with self.assertRaises(ValueError):
            lts.update("haiku", "code-review", 1.5)
        with self.assertRaises(ValueError):
            lts.update("haiku", "code-review", -0.1)

    def test_rewards_from_outcomes_mapping(self):
        self.assertEqual(rewards_from_outcomes(False), 0.0)
        self.assertEqual(rewards_from_outcomes(True), 1.0)
        self.assertEqual(rewards_from_outcomes(True, 0.8), 0.8)
        self.assertEqual(rewards_from_outcomes(True, 2.0), 1.0)  # clamped


class PersistenceTests(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lts.json"
            lts = LinearTS(path)
            lts.update("haiku", "code-review", 1.0)
            lts.update("glm-5-3", "documentation", 0.3)
            lts.save()
            reloaded = LinearTS(path)
            self.assertEqual(reloaded.arms["haiku"].updates, 1)
            self.assertEqual(reloaded.arms["glm-5-3"].updates, 1)
            for arm in ("haiku", "glm-5-3"):
                np.testing.assert_allclose(reloaded.arms[arm].a, lts.arms[arm].a)
                np.testing.assert_allclose(reloaded.arms[arm].b, lts.arms[arm].b)
            self.assertEqual(
                reloaded.mean_scores("code-review"), lts.mean_scores("code-review"))
            self.assertEqual(
                reloaded.mean_scores("documentation"), lts.mean_scores("documentation"))

    def test_corrupt_state_starts_fresh(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lts.json"
            path.write_text("{ not json")
            lts = LinearTS(path)
            self.assertEqual(lts.arms, {})
            self.assertIsNotNone(lts.load_error)

    def test_wrong_shaped_json_state_starts_fresh(self):
        """Valid JSON with the wrong shape must not crash the constructor."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lts.json"
            for bad in ('[1, 2, 3]', 'null', '42', '{"schema_version": 1, "arms": []}',
                        '{"schema_version": 1, "arms": null}'):
                with self.subTest(body=bad):
                    path.write_text(bad)
                    lts = LinearTS(path)
                    self.assertEqual(lts.arms, {})
                    self.assertIsNotNone(lts.load_error)

    def test_non_finite_state_values_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lts.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "feature_names": list(FEATURE_NAMES),
                "arms": {"haiku": {
                    "arm": "haiku",
                    "a": [[float("nan")] * DIM],
                    "b": [0.0] * DIM,
                }},
            }))
            lts = LinearTS(path)
            self.assertEqual(lts.arms, {})
            self.assertIsNotNone(lts.load_error)

    def test_select_survives_all_nan_scores(self):
        """Every draw NaN must still return a requested arm (no None)."""
        lts = LinearTS("/nonexistent/state.json")
        arm = lts.arm("haiku")
        arm.a = np.zeros((DIM, DIM))  # singular -> NaN draws
        arm.b = np.zeros(DIM)
        picked = lts.select(["haiku", "glm-5-3"], "code-review")
        self.assertIn(picked, ["haiku", "glm-5-3"])

    def test_stale_feature_vocabulary_starts_fresh(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lts.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "feature_names": ["bias", "something-else"],
                "arms": {"haiku": {"arm": "haiku", "a": [[1.0]], "b": [0.0]}},
            }))
            lts = LinearTS(path)
            self.assertEqual(lts.arms, {})
            self.assertIsNotNone(lts.load_error)


if __name__ == "__main__":
    unittest.main()
