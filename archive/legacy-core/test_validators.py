#!/usr/bin/env python3
"""
Test field validation for daemon services
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared.validators import Validators

def test_thompson_validation():
    """Test Thompson service validation"""
    print("Testing Thompson service validation...")

    # Valid select_model request
    valid_req = {'action': 'select_model', 'task_type': 'search', 'max_cost': 0.5}
    is_valid, error = Validators.validate_thompson_request(valid_req, 'select_model')
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid select_model request")

    # Invalid: max_cost is string (should be float)
    invalid_req = {'action': 'select_model', 'max_cost': 'not_a_float'}
    is_valid, error = Validators.validate_thompson_request(invalid_req, 'select_model')
    assert not is_valid, "Should reject string for max_cost"
    assert 'max_cost' in error, f"Error should mention max_cost: {error}"
    print("✓ Rejected invalid max_cost type")

    # Valid record_outcome request
    valid_req = {
        'action': 'record_outcome',
        'model': 'haiku',
        'task_type': 'search',
        'success': True,
        'cost': 0.001,
        'tokens': 1000
    }
    is_valid, error = Validators.validate_thompson_request(valid_req, 'record_outcome')
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid record_outcome request")

    # Invalid: missing model field
    invalid_req = {'success': True, 'cost': 0.001}
    is_valid, error = Validators.validate_thompson_request(invalid_req, 'record_outcome')
    assert not is_valid, "Should reject missing model"
    assert 'model' in error, f"Error should mention model: {error}"
    print("✓ Rejected missing model field")

    # Invalid: cost is negative
    invalid_req = {'model': 'haiku', 'cost': -0.001}
    is_valid, error = Validators.validate_thompson_request(invalid_req, 'record_outcome')
    assert not is_valid, "Should reject negative cost"
    assert 'cost' in error, f"Error should mention cost: {error}"
    print("✓ Rejected negative cost")

def test_learning_validation():
    """Test Learning service validation"""
    print("\nTesting Learning service validation...")

    # Valid process_outcome request
    valid_req = {
        'op': 'process_outcome',
        'task_id': 'task-123',
        'task_type': 'code_review',
        'model': 'sonnet',
        'rating': 4,
        'tokens': 2000,
        'cost': 0.005
    }
    is_valid, error = Validators.validate_learning_request(valid_req, 'process_outcome')
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid process_outcome request")

    # Invalid: missing task_id
    invalid_req = {
        'op': 'process_outcome',
        'task_type': 'code_review',
        'model': 'sonnet',
        'rating': 4,
        'tokens': 2000,
        'cost': 0.005
    }
    is_valid, error = Validators.validate_learning_request(invalid_req, 'process_outcome')
    assert not is_valid, "Should reject missing task_id"
    assert 'task_id' in error, f"Error should mention task_id: {error}"
    print("✓ Rejected missing task_id field")

    # Invalid: rating out of range (0, should be 1-5)
    invalid_req = {
        'op': 'process_outcome',
        'task_id': 'task-123',
        'task_type': 'code_review',
        'model': 'sonnet',
        'rating': 0,
        'tokens': 2000,
        'cost': 0.005
    }
    is_valid, error = Validators.validate_learning_request(invalid_req, 'process_outcome')
    assert not is_valid, "Should reject rating outside 1-5"
    assert 'rating' in error, f"Error should mention rating: {error}"
    print("✓ Rejected invalid rating (0)")

    # Invalid: rating out of range (6, should be 1-5)
    invalid_req['rating'] = 6
    is_valid, error = Validators.validate_learning_request(invalid_req, 'process_outcome')
    assert not is_valid, "Should reject rating outside 1-5"
    assert 'rating' in error, f"Error should mention rating: {error}"
    print("✓ Rejected invalid rating (6)")

    # Invalid: tokens is string (should be int)
    invalid_req = {
        'op': 'process_outcome',
        'task_id': 'task-123',
        'task_type': 'code_review',
        'model': 'sonnet',
        'rating': 4,
        'tokens': '2000',
        'cost': 0.005
    }
    is_valid, error = Validators.validate_learning_request(invalid_req, 'process_outcome')
    assert not is_valid, "Should reject string for tokens"
    assert 'tokens' in error, f"Error should mention tokens: {error}"
    print("✓ Rejected invalid tokens type")

def test_alert_validation():
    """Test Alert service validation"""
    print("\nTesting Alert service validation...")

    # Valid acknowledge request
    valid_req = {'op': 'acknowledge', 'alert_id': 'alert-123'}
    is_valid, error = Validators.validate_alert_request(valid_req, 'acknowledge')
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid acknowledge request")

    # Invalid: missing alert_id
    invalid_req = {'op': 'acknowledge'}
    is_valid, error = Validators.validate_alert_request(invalid_req, 'acknowledge')
    assert not is_valid, "Should reject missing alert_id"
    assert 'alert_id' in error, f"Error should mention alert_id: {error}"
    print("✓ Rejected missing alert_id field")

    # Valid alert dict
    valid_alert = {
        'alert_type': 'cost_spike',
        'severity': 'warning',
        'message': 'Cost spike detected',
        'timestamp': '2026-09-26T12:00:00'
    }
    is_valid, error = Validators.validate_alert_dict(valid_alert)
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid alert dict (cost_spike)")

    # Valid alert dict with quality_drop
    valid_alert['alert_type'] = 'quality_drop'
    is_valid, error = Validators.validate_alert_dict(valid_alert)
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid alert dict (quality_drop)")

    # Valid alert dict with model_error
    valid_alert['alert_type'] = 'model_error'
    is_valid, error = Validators.validate_alert_dict(valid_alert)
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid alert dict (model_error)")

    # Invalid: bad alert_type
    invalid_alert = {
        'alert_type': 'invalid_type',
        'severity': 'warning',
        'message': 'Test'
    }
    is_valid, error = Validators.validate_alert_dict(invalid_alert)
    assert not is_valid, "Should reject invalid alert_type"
    assert 'alert_type' in error, f"Error should mention alert_type: {error}"
    print("✓ Rejected invalid alert_type")

    # Invalid: missing severity
    invalid_alert = {
        'alert_type': 'cost_spike',
        'message': 'Test'
    }
    is_valid, error = Validators.validate_alert_dict(invalid_alert)
    assert not is_valid, "Should reject missing severity"
    assert 'severity' in error, f"Error should mention severity: {error}"
    print("✓ Rejected missing severity field")

    # Invalid: missing message
    invalid_alert = {
        'alert_type': 'cost_spike',
        'severity': 'warning'
    }
    is_valid, error = Validators.validate_alert_dict(invalid_alert)
    assert not is_valid, "Should reject missing message"
    assert 'message' in error, f"Error should mention message: {error}"
    print("✓ Rejected missing message field")

if __name__ == '__main__':
    test_thompson_validation()
    test_learning_validation()
    test_alert_validation()
    print("\n✅ All validation tests passed!")
