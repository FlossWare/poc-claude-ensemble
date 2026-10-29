"""Pricing tests: base rates, cache multipliers, gateway models, and unknown models."""
import unittest

import pricing


class CostTests(unittest.TestCase):
    def test_base_rates_unchanged(self):
        self.assertAlmostEqual(pricing.calculate_cost("haiku", 1_000_000, 1_000_000), 0.80 + 2.40)

    def test_cache_write_costs_one_and_a_quarter_input(self):
        self.assertAlmostEqual(pricing.calculate_cost("haiku", 0, 0, cache_write_tokens=1_000_000), 0.80 * 1.25)

    def test_cache_read_costs_a_tenth_of_input(self):
        self.assertAlmostEqual(pricing.calculate_cost("opus", 0, 0, cache_read_tokens=1_000_000), 15.0 * 0.1)

    def test_gateway_models_use_the_opus_assumption(self):
        self.assertAlmostEqual(pricing.calculate_cost("rits/zai-org/glm-5-3", 1_000_000, 0), 15.0)
        self.assertAlmostEqual(pricing.calculate_cost("rits/deepseek-ai/DeepSeek-V4.1-Flash-1017", 0, 1_000_000), 45.0)

    def test_unknown_model_is_still_rejected(self):
        with self.assertRaises(ValueError):
            pricing.calculate_cost("unknown-model", 1, 1)


if __name__ == "__main__":
    unittest.main()
