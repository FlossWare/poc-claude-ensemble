"""
Test suite for CostValidator module.

Tests validator against sample logs with intentional errors:
- Duplicate entries
- Incorrect cost calculations
- Missing required fields
- Invalid field types
- Anomalies (spikes, high values)
"""

import json
import tempfile
import os
from pathlib import Path
from validator import CostValidator, Severity


def create_test_log_file(data: list, filename: str = None) -> str:
    """Helper to create temporary test log files."""
    if filename is None:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
    else:
        path = filename

    with open(path, "w") as f:
        json.dump(data, f)

    return path


def test_1_valid_log():
    """Test: Valid log with no errors."""
    print("\n" + "=" * 80)
    print("TEST 1: Valid Log - All entries correct")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
            "session_id": "sess_001",
            "task_description": "Code analysis",
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "sonnet",
            "input_tokens": 20000,
            "output_tokens": 5000,
            "cost": 0.135,
            "session_id": "sess_002",
            "task_description": "Architecture review",
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "opus",
            "input_tokens": 15000,
            "output_tokens": 8000,
            "cost": 0.585,
            "session_id": "sess_003",
            "task_description": "Complex analysis",
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    os.unlink(filepath)
    assert is_valid, "Expected valid log to pass"


def test_2_duplicate_entries():
    """Test: Duplicate entry detection."""
    print("\n" + "=" * 80)
    print("TEST 2: Duplicate Entries - Same timestamp + model + tokens")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },  # DUPLICATE
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify duplicate was detected
    errors = [r for r in results if r.severity == Severity.ERROR]
    duplicate_errors = [r for r in errors if "Duplicate" in r.message]
    assert len(duplicate_errors) > 0, "Expected duplicate detection error"

    os.unlink(filepath)


def test_3_wrong_cost_calculation():
    """Test: Incorrect cost calculation detection."""
    print("\n" + "=" * 80)
    print("TEST 3: Wrong Cost Calculation")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0099,  # WRONG: Should be (5000/1M)*0.80 + (1000/1M)*2.40 = 0.0064
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "sonnet",
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.045,  # WRONG: Should be (10000/1M)*3.00 + (5000/1M)*15.00 = 0.105
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify cost errors were detected
    errors = [r for r in results if r.severity == Severity.ERROR]
    cost_errors = [r for r in errors if "cost calculation" in r.message]
    assert len(cost_errors) == 2, "Expected 2 cost calculation errors"

    os.unlink(filepath)


def test_4_missing_fields():
    """Test: Missing required fields detection."""
    print("\n" + "=" * 80)
    print("TEST 4: Missing Required Fields")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            # MISSING: output_tokens and cost
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            # MISSING: model
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.105,
        },
        {
            # MISSING: timestamp
            "model": "opus",
            "input_tokens": 15000,
            "output_tokens": 8000,
            "cost": 0.585,
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify missing field errors were detected
    errors = [r for r in results if r.severity == Severity.ERROR]
    missing_field_errors = [r for r in errors if "missing required" in r.message]
    assert len(missing_field_errors) == 3, "Expected 3 missing field errors"

    os.unlink(filepath)


def test_5_invalid_types():
    """Test: Invalid field types detection."""
    print("\n" + "=" * 80)
    print("TEST 5: Invalid Field Types")
    print("=" * 80)

    logs = [
        {
            "timestamp": 12345,  # WRONG: Should be string
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": ["sonnet"],  # WRONG: Should be string
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.105,
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "opus",
            "input_tokens": "15000",  # WRONG: Should be int
            "output_tokens": 8000,
            "cost": 0.585,
        },
        {
            "timestamp": "2026-09-25T10:15:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000.5,  # WRONG: Should be int
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:20:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": "0.0064",  # WRONG: Should be numeric
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify type errors were detected
    errors = [r for r in results if r.severity == Severity.ERROR]
    type_errors = [r for r in errors if "type error" in r.message]
    assert len(type_errors) == 5, "Expected 5 type errors"

    os.unlink(filepath)


def test_6_anomalies_high_values():
    """Test: Anomaly detection - high token counts and costs."""
    print("\n" + "=" * 80)
    print("TEST 6: Anomaly Detection - High Values")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "haiku",
            "input_tokens": 150000,  # HIGH: Over 100k
            "output_tokens": 1000,
            "cost": 0.132,
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "opus",
            "input_tokens": 5000,
            "output_tokens": 150000,  # HIGH: Over 100k
            "cost": 6.825,
        },
        {
            "timestamp": "2026-09-25T10:15:00Z",
            "model": "sonnet",
            "input_tokens": 10000,
            "output_tokens": 50000,
            "cost": 780.0,  # HIGH: Over $50
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify anomalies were detected
    warnings = [r for r in results if r.severity == Severity.WARNING]
    anomaly_warnings = [
        r for r in warnings if "unusually high" in r.message or "high cost" in r.message
    ]
    assert len(anomaly_warnings) >= 3, "Expected at least 3 anomaly warnings"

    os.unlink(filepath)


def test_7_anomalies_spikes():
    """Test: Anomaly detection - cost and token spikes."""
    print("\n" + "=" * 80)
    print("TEST 7: Anomaly Detection - Spikes (3x-5x median)")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cost": 0.0018,
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "haiku",
            "input_tokens": 1200,
            "output_tokens": 600,
            "cost": 0.0022,
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "haiku",
            "input_tokens": 1100,
            "output_tokens": 550,
            "cost": 0.0020,
        },
        {
            "timestamp": "2026-09-25T10:15:00Z",
            "model": "haiku",
            "input_tokens": 5000,  # 4-5x median input
            "output_tokens": 2500,  # 4-5x median output
            "cost": 0.0088,  # 4-5x median cost (spike)
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify spike detection
    warnings = [r for r in results if r.severity == Severity.WARNING]
    spike_warnings = [r for r in warnings if "spike" in r.message]
    assert len(spike_warnings) >= 1, "Expected spike detection warnings"

    os.unlink(filepath)


def test_8_negative_values():
    """Test: Negative values detection."""
    print("\n" + "=" * 80)
    print("TEST 8: Negative Values - Invalid")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": -5000,  # WRONG: Negative
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": -1000,  # WRONG: Negative
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": -0.0064,  # WRONG: Negative
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify negative value errors
    errors = [r for r in results if r.severity == Severity.ERROR]
    negative_errors = [r for r in errors if "negative" in r.message]
    assert len(negative_errors) == 3, "Expected 3 negative value errors"

    os.unlink(filepath)


def test_9_unrecognized_model():
    """Test: Unrecognized model detection."""
    print("\n" + "=" * 80)
    print("TEST 9: Unrecognized Model")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "gpt-4",  # WRONG: Not recognized
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.15,
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "UNKNOWN_MODEL",  # WRONG: Not recognized
            "input_tokens": 15000,
            "output_tokens": 8000,
            "cost": 0.25,
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Verify unrecognized model warnings
    warnings = [r for r in results if r.severity == Severity.WARNING]
    model_warnings = [r for r in warnings if "unrecognized model" in r.message]
    assert len(model_warnings) == 2, "Expected 2 unrecognized model warnings"

    os.unlink(filepath)


def test_10_empty_log():
    """Test: Empty log file handling."""
    print("\n" + "=" * 80)
    print("TEST 10: Empty Log File")
    print("=" * 80)

    logs = []
    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    # Should pass with warning about empty
    warnings = [r for r in results if r.severity == Severity.WARNING]
    empty_warnings = [r for r in warnings if "empty" in r.message]
    assert len(empty_warnings) > 0, "Expected empty file warning"

    os.unlink(filepath)


def test_11_complex_mixed_errors():
    """Test: Complex scenario with multiple error types."""
    print("\n" + "=" * 80)
    print("TEST 11: Complex Scenario - Multiple Error Types")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "haiku",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cost": 0.0064,
        },  # DUPLICATE
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "sonnet",
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.050,  # WRONG COST
        },
        {
            "timestamp": "2026-09-25T10:10:00Z",
            "model": "opus",
            # MISSING: model field
            "input_tokens": "15000",  # WRONG TYPE
            "output_tokens": 8000,
            "cost": 0.585,
        },
        {
            "timestamp": "2026-09-25T10:15:00Z",
            "model": "haiku",
            "input_tokens": -5000,  # NEGATIVE
            "output_tokens": 1000,
            "cost": 0.0064,
        },
        {
            "timestamp": "2026-09-25T10:20:00Z",
            "model": "gpt-4",  # UNKNOWN MODEL
            "input_tokens": 10000,
            "output_tokens": 5000,
            "cost": 0.15,
        },
        {
            "timestamp": "2026-09-25T10:25:00Z",
            "model": "haiku",
            "input_tokens": 200000,  # HIGH VALUE
            "output_tokens": 1000,
            "cost": 0.172,
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    assert not is_valid, "Expected complex scenario to fail validation"

    os.unlink(filepath)


def test_12_gemini_pricing():
    """Test: Gemini model pricing calculation."""
    print("\n" + "=" * 80)
    print("TEST 12: Gemini Model - Lower Pricing")
    print("=" * 80)

    logs = [
        {
            "timestamp": "2026-09-25T10:00:00Z",
            "model": "gemini",
            "input_tokens": 100000,
            "output_tokens": 50000,
            "cost": 0.022500,  # (100k/1M)*0.075 + (50k/1M)*0.30
        },
        {
            "timestamp": "2026-09-25T10:05:00Z",
            "model": "Gemini",  # Test case insensitivity
            "input_tokens": 1000000,
            "output_tokens": 500000,
            "cost": 0.225000,  # (1M/1M)*0.075 + (500k/1M)*0.30
        },
    ]

    filepath = create_test_log_file(logs)

    validator = CostValidator()
    results, is_valid = validator.validate_log_file(filepath)

    print(validator.get_report())
    print(f"\nValid: {is_valid}")

    assert is_valid, "Expected Gemini pricing to validate correctly"

    os.unlink(filepath)


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# COST VALIDATOR TEST SUITE")
    print("#" * 80)

    tests = [
        ("Valid Log", test_1_valid_log),
        ("Duplicate Entries", test_2_duplicate_entries),
        ("Wrong Cost Calculation", test_3_wrong_cost_calculation),
        ("Missing Fields", test_4_missing_fields),
        ("Invalid Types", test_5_invalid_types),
        ("High Values", test_6_anomalies_high_values),
        ("Spikes", test_7_anomalies_spikes),
        ("Negative Values", test_8_negative_values),
        ("Unrecognized Model", test_9_unrecognized_model),
        ("Empty Log", test_10_empty_log),
        ("Complex Mixed Errors", test_11_complex_mixed_errors),
        ("Gemini Pricing", test_12_gemini_pricing),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {test_name}: PASSED")
        except AssertionError as e:
            failed += 1
            print(f"\n✗ {test_name}: FAILED - {e}")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: ERROR - {e}")

    print("\n" + "#" * 80)
    print(f"# TEST RESULTS: {passed} passed, {failed} failed")
    print("#" * 80)
