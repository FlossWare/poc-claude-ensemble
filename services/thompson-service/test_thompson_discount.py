"""Discounted-Thompson tests: decay semantics, gamma=1 equivalence, persistence."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import thompson_service as ts


def _record_many(state, model, successes, failures, gamma_env=None):
    import os
    old = os.environ.get(ts.ThompsonState._DISCOUNT_ENV)
    if gamma_env is not None:
        os.environ[ts.ThompsonState._DISCOUNT_ENV] = gamma_env
    try:
        for _ in range(successes):
            state.record_outcome(model, "code-review", True, 0.0, 10)
        for _ in range(failures):
            state.record_outcome(model, "code-review", False, 0.0, 10)
    finally:
        if gamma_env is not None:
            if old is None:
                os.environ.pop(ts.ThompsonState._DISCOUNT_ENV, None)
            else:
                os.environ[ts.ThompsonState._DISCOUNT_ENV] = old


class DiscountTests(unittest.TestCase):
    def test_gamma_one_is_classic_thompson(self):
        with TemporaryDirectory() as tmp:
            state = ts.ThompsonState(Path(tmp) / "state.json")
            _record_many(state, "haiku", 5, 3, gamma_env="1.0")
            stats = state.models["haiku"]
            self.assertEqual(stats.successes, 5)
            self.assertEqual(stats.failures, 3)
            self.assertEqual(stats.eff_successes, 5.0)
            self.assertEqual(stats.eff_failures, 3.0)

    def test_default_is_classic_thompson(self):
        with TemporaryDirectory() as tmp:
            state = ts.ThompsonState(Path(tmp) / "state.json")
            _record_many(state, "haiku", 4, 2)  # no env set
            stats = state.models["haiku"]
            self.assertEqual(stats.eff_successes, 4.0)
            self.assertEqual(stats.eff_failures, 2.0)

    def test_discount_decays_old_evidence(self):
        with TemporaryDirectory() as tmp:
            state = ts.ThompsonState(Path(tmp) / "state.json")
            # 10 successes, then with gamma=0.5 two failures.
            _record_many(state, "sonnet", 10, 0, gamma_env="1.0")
            _record_many(state, "sonnet", 0, 2, gamma_env="0.5")
            stats = state.models["sonnet"]
            # eff_successes: 10 -> 5 -> 2.5 ; eff_failures: 0 -> 1 -> 1.5
            self.assertAlmostEqual(stats.eff_successes, 2.5)
            self.assertAlmostEqual(stats.eff_failures, 1.5)
            # Raw history untouched.
            self.assertEqual(stats.successes, 10)
            self.assertEqual(stats.failures, 2)

    def test_discounted_posterior_drops_regressed_model(self):
        """A model that was great then broke loses to a steady mediocre one.

        Verified against an inline reference implementation of the discount
        recurrence, so the assertion holds for the MATH, not for a lucky
        outcome ordering.
        """
        gamma = 0.5
        # regressed: 50 successes, then 6 failures.
        regressed_seq = [True] * 50 + [False] * 6
        # steady: alternating failure/success (recency-neutral order, true
        # rate 0.5, ends on a success).
        steady_seq = [i % 2 == 1 for i in range(40)]

        def reference(sequence):
            s = f = 0.0
            for outcome in sequence:
                s = gamma * s + (1.0 if outcome else 0.0)
                f = gamma * f + (0.0 if outcome else 1.0)
            return s, f

        with TemporaryDirectory() as tmp:
            state = ts.ThompsonState(Path(tmp) / "state.json")
            import os
            os.environ[ts.ThompsonState._DISCOUNT_ENV] = str(gamma)
            try:
                for outcome in regressed_seq:
                    state.record_outcome("regressed", "t", outcome, 0.0, 0)
                for outcome in steady_seq:
                    state.record_outcome("steady", "t", outcome, 0.0, 0)
            finally:
                os.environ.pop(ts.ThompsonState._DISCOUNT_ENV, None)
            r, s = state.models["regressed"], state.models["steady"]
            # Exact match with the closed recurrence.
            ref_r, ref_s = reference(regressed_seq), reference(steady_seq)
            self.assertAlmostEqual(r.eff_successes, ref_r[0])
            self.assertAlmostEqual(r.eff_failures, ref_r[1])
            self.assertAlmostEqual(s.eff_successes, ref_s[0])
            self.assertAlmostEqual(s.eff_failures, ref_s[1])
            # Classic view: regressed (50/6) looks far better than steady (20/20).
            self.assertGreater(r.success_rate, s.success_rate)
            # Discounted view: regressed's effective rate collapses below
            # steady's -- the drift is detected.
            eff_r = r.eff_successes / (r.eff_successes + r.eff_failures)
            eff_s = s.eff_successes / (s.eff_successes + s.eff_failures)
            self.assertGreater(eff_s, eff_r)

    def test_invalid_gamma_fails_closed_to_one(self):
        with TemporaryDirectory() as tmp:
            state = ts.ThompsonState(Path(tmp) / "state.json")
            _record_many(state, "haiku", 2, 1, gamma_env="not-a-number")
            stats = state.models["haiku"]
            self.assertEqual(stats.eff_successes, 2.0)
            _record_many(state, "haiku", 0, 1, gamma_env="1.5")  # out of range
            self.assertEqual(stats.eff_failures, 2.0)

    def test_state_round_trip_preserves_effective_counts(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = ts.ThompsonState(path)
            _record_many(state, "haiku", 10, 0, gamma_env="1.0")
            _record_many(state, "haiku", 0, 3, gamma_env="0.5")
            before = state.models["haiku"].eff_successes
            reloaded = ts.ThompsonState(path)
            self.assertAlmostEqual(reloaded.models["haiku"].eff_successes, before)

    def test_old_state_file_without_effective_counts_seeds_from_raw(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({
                "last_updated": "2026-01-01T00:00:00Z",
                "models": {
                    "haiku": {
                        "model_name": "haiku", "successes": 7, "failures": 2,
                        "total_cost": 0.1, "total_tokens": 99, "calls": 9,
                    }
                },
            }))
            state = ts.ThompsonState(path)
            self.assertAlmostEqual(state.models["haiku"].eff_successes, 7.0)
            self.assertAlmostEqual(state.models["haiku"].eff_failures, 2.0)


if __name__ == "__main__":
    unittest.main()
