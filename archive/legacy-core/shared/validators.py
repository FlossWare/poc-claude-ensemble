#!/usr/bin/env python3
"""
Field validators for all RH daemon services.

Provides strict type checking and validation for incoming requests
to Thompson, Learning, and Alert services to prevent silent failures.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class Validators:
    """Collection of validation functions for daemon requests"""

    @staticmethod
    def validate_thompson_request(req_data: Dict[str, Any], action: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Thompson service request.

        Args:
            req_data: Request dictionary
            action: Action being performed

        Returns:
            (is_valid, error_message)
        """
        if action == 'select_model':
            # Task type is optional but must be string if provided
            if 'task_type' in req_data and not isinstance(req_data['task_type'], str):
                return False, "field 'task_type': expected string, got {}".format(
                    type(req_data['task_type']).__name__)

            # Optional fields with type checks
            if 'required_capability' in req_data:
                if not isinstance(req_data['required_capability'], (int, float)):
                    return False, "field 'required_capability': expected float, got {}".format(
                        type(req_data['required_capability']).__name__)
                if not (0 <= req_data['required_capability'] <= 1):
                    return False, "field 'required_capability': must be between 0 and 1"

            if 'max_cost' in req_data:
                if not isinstance(req_data['max_cost'], (int, float)):
                    return False, "field 'max_cost': expected float, got {}".format(
                        type(req_data['max_cost']).__name__)
                if req_data['max_cost'] < 0:
                    return False, "field 'max_cost': must be >= 0"

        elif action == 'record_outcome':
            # Required fields
            if 'model' not in req_data:
                return False, "missing field 'model': expected string"
            if not isinstance(req_data['model'], str):
                return False, "field 'model': expected string, got {}".format(
                    type(req_data['model']).__name__)

            # Task type optional but must be string
            if 'task_type' in req_data and not isinstance(req_data['task_type'], str):
                return False, "field 'task_type': expected string, got {}".format(
                    type(req_data['task_type']).__name__)

            # Success must be boolean
            if 'success' in req_data and not isinstance(req_data['success'], bool):
                return False, "field 'success': expected boolean, got {}".format(
                    type(req_data['success']).__name__)

            # Cost must be non-negative float
            if 'cost' in req_data:
                if not isinstance(req_data['cost'], (int, float)):
                    return False, "field 'cost': expected float, got {}".format(
                        type(req_data['cost']).__name__)
                if req_data['cost'] < 0:
                    return False, "field 'cost': must be >= 0, got {}".format(req_data['cost'])

            # Tokens must be non-negative integer
            if 'tokens' in req_data:
                if not isinstance(req_data['tokens'], int):
                    return False, "field 'tokens': expected integer, got {}".format(
                        type(req_data['tokens']).__name__)
                if req_data['tokens'] < 0:
                    return False, "field 'tokens': must be >= 0"

        elif action == 'reset':
            if 'model' not in req_data:
                return False, "missing field 'model': expected string"
            if not isinstance(req_data['model'], str):
                return False, "field 'model': expected string, got {}".format(
                    type(req_data['model']).__name__)

        return True, None

    @staticmethod
    def validate_learning_request(req_data: Dict[str, Any], operation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Learning service request.

        Args:
            req_data: Request dictionary
            operation: Operation being performed

        Returns:
            (is_valid, error_message)
        """
        if operation == 'record_artifact':
            artifact = req_data.get('artifact')
            if not isinstance(artifact, dict):
                return False, "field 'artifact': expected object"
            if artifact.get('schema') != 'learning-artifact' or artifact.get('version') != '0.1':
                return False, "field 'artifact': unsupported learning artifact schema/version"
            if not isinstance(artifact.get('artifact_type'), str) or not artifact['artifact_type'].strip():
                return False, "field 'artifact.artifact_type': expected non-empty string"
            if not isinstance(artifact.get('source'), str) or not artifact['source'].strip():
                return False, "field 'artifact.source': expected non-empty string"
            payload = artifact.get('payload')
            if not isinstance(payload, dict):
                return False, "field 'artifact.payload': expected object"
            run_id = payload.get('run_id')
            if not isinstance(run_id, str) or not run_id.strip():
                return False, "field 'artifact.payload.run_id': expected non-empty string"

        elif operation == 'process_outcome':
            # Required fields
            if 'task_id' not in req_data:
                return False, "missing field 'task_id': expected string"
            if not isinstance(req_data['task_id'], str):
                return False, "field 'task_id': expected string, got {}".format(
                    type(req_data['task_id']).__name__)

            if 'task_type' not in req_data:
                return False, "missing field 'task_type': expected string"
            if not isinstance(req_data['task_type'], str):
                return False, "field 'task_type': expected string, got {}".format(
                    type(req_data['task_type']).__name__)

            if 'model' not in req_data:
                return False, "missing field 'model': expected string"
            if not isinstance(req_data['model'], str):
                return False, "field 'model': expected string, got {}".format(
                    type(req_data['model']).__name__)

            if 'rating' not in req_data:
                return False, "missing field 'rating': expected integer (1-5)"
            if not isinstance(req_data['rating'], int):
                return False, "field 'rating': expected integer, got {}".format(
                    type(req_data['rating']).__name__)
            if not (1 <= req_data['rating'] <= 5):
                return False, "field 'rating': must be between 1 and 5, got {}".format(req_data['rating'])

            if 'tokens' not in req_data:
                return False, "missing field 'tokens': expected integer"
            if not isinstance(req_data['tokens'], int):
                return False, "field 'tokens': expected integer, got {}".format(
                    type(req_data['tokens']).__name__)
            if req_data['tokens'] < 0:
                return False, "field 'tokens': must be >= 0"

            if 'cost' not in req_data:
                return False, "missing field 'cost': expected float"
            if not isinstance(req_data['cost'], (int, float)):
                return False, "field 'cost': expected float, got {}".format(
                    type(req_data['cost']).__name__)
            if req_data['cost'] < 0:
                return False, "field 'cost': must be >= 0"

        elif operation == 'get_recent_outcomes':
            # Days is optional but must be integer if provided
            if 'days' in req_data:
                if not isinstance(req_data['days'], int):
                    return False, "field 'days': expected integer, got {}".format(
                        type(req_data['days']).__name__)
                if req_data['days'] < 1:
                    return False, "field 'days': must be >= 1"

        return True, None

    @staticmethod
    def validate_alert_request(req_data: Dict[str, Any], operation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Alert service request.

        Args:
            req_data: Request dictionary
            operation: Operation being performed

        Returns:
            (is_valid, error_message)
        """
        if operation == 'acknowledge':
            if 'alert_id' not in req_data:
                return False, "missing field 'alert_id': expected string"
            if not isinstance(req_data['alert_id'], str):
                return False, "field 'alert_id': expected string, got {}".format(
                    type(req_data['alert_id']).__name__)

        elif operation == 'get_recent_alerts':
            # Days is optional but must be integer if provided
            if 'days' in req_data:
                if not isinstance(req_data['days'], int):
                    return False, "field 'days': expected integer, got {}".format(
                        type(req_data['days']).__name__)
                if req_data['days'] < 1:
                    return False, "field 'days': must be >= 1"

        return True, None

    @staticmethod
    def validate_alert_dict(alert: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an alert dictionary structure.

        Args:
            alert: Alert dictionary to validate

        Returns:
            (is_valid, error_message)
        """
        # alert_type must be one of the known types
        if 'alert_type' not in alert:
            return False, "missing field 'alert_type': expected string in ['cost_spike', 'quality_drop', 'model_error']"

        if not isinstance(alert['alert_type'], str):
            return False, "field 'alert_type': expected string, got {}".format(
                type(alert['alert_type']).__name__)

        valid_types = ['cost_spike', 'quality_drop', 'model_error']
        if alert['alert_type'] not in valid_types:
            return False, "field 'alert_type': must be one of {}, got '{}'".format(
                valid_types, alert['alert_type'])

        # severity must be string
        if 'severity' not in alert:
            return False, "missing field 'severity': expected string"
        if not isinstance(alert['severity'], str):
            return False, "field 'severity': expected string, got {}".format(
                type(alert['severity']).__name__)

        # message must be string
        if 'message' not in alert:
            return False, "missing field 'message': expected string"
        if not isinstance(alert['message'], str):
            return False, "field 'message': expected string, got {}".format(
                type(alert['message']).__name__)

        return True, None
