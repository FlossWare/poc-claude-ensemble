"""Tests for the shared GA parameter schema."""
import unittest

from ga_tuning.parameter_schema import bounds_for_system, validate_parameter


class GAParameterSchemaTests(unittest.TestCase):
    def test_optimizer_bounds_are_derived_from_schema(self):
        self.assertEqual(bounds_for_system("caching")["ttl_seconds"], (60.0, 600.0))
        self.assertEqual(bounds_for_system("compression")["target_reduction"], (0.2, 0.7))

    def test_values_inside_bounds_are_normalized_to_float(self):
        self.assertEqual(validate_parameter("caching", "ttl_seconds", 120), 120.0)
        self.assertEqual(validate_parameter("compression", "compression_level", 3), 3.0)

    def test_out_of_range_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_parameter("caching", "ttl_seconds", 21_600)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_parameter("compression", "target_reduction", 0.99)

    def test_non_finite_and_boolean_values_are_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_parameter("caching", "ttl_seconds", value)

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_parameter("thompson", "invented_parameter", 0.5)


if __name__ == "__main__":
    unittest.main()
