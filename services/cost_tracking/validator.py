"""
Cost Validator Module

Validates Claude API usage logs against official pricing, checks for duplicates,
validates schema, and detects cost anomalies.

Pricing Reference (per 1M tokens):
- Haiku: $0.80 input / $2.40 output
- Sonnet: $3.00 input / $15.00 output
- Opus: $15.00 input / $45.00 output
- Gemini: $0.075 input / $0.30 output
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from enum import Enum
import statistics


class Severity(Enum):
    """Severity levels for validation findings."""
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    severity: Severity
    message: str
    details: Optional[Dict] = None


class CostValidator:
    """Validates cost logs for accuracy, duplicates, schema compliance, and anomalies."""

    # Official Claude pricing (per 1M tokens)
    PRICING = {
        "haiku": {"input": 0.80, "output": 2.40},
        "sonnet": {"input": 3.00, "output": 15.00},
        "opus": {"input": 15.00, "output": 45.00},
        "gemini": {"input": 0.075, "output": 0.30},
        # Gateway models with no published rates, assumed at the Opus rate (see pricing.py)
        "rits/zai-org/glm-5-3": {"input": 15.00, "output": 45.00},
        "rits/deepseek-ai/deepseek-v4.1-flash-1017": {"input": 15.00, "output": 45.00},
    }

    # Required fields in log entries
    REQUIRED_FIELDS = {"timestamp", "model", "input_tokens", "output_tokens", "cost"}

    # Optional but expected fields
    OPTIONAL_FIELDS = {"session_id", "task_description", "project", "notes"}

    # Thresholds for anomaly detection
    ANOMALY_THRESHOLDS = {
        "high_input_tokens": 100000,      # Flag if > 100k input tokens
        "high_output_tokens": 100000,     # Flag if > 100k output tokens
        "high_cost": 50.0,                 # Flag if > $50 per call
        "cost_spike_multiplier": 5.0,     # Flag if 5x the median cost
        "token_spike_multiplier": 3.0,    # Flag if 3x the median tokens
    }

    def __init__(self):
        """Initialize the cost validator."""
        self.results: List[ValidationResult] = []

    def validate_log_file(self, filepath: str) -> Tuple[List[ValidationResult], bool]:
        """
        Validate an entire cost log file.

        Args:
            filepath: Path to the cost log JSON file

        Returns:
            Tuple of (results list, is_valid boolean)
        """
        self.results = []

        try:
            with open(filepath, "r") as f:
                logs = json.load(f)
        except FileNotFoundError:
            self.results.append(
                ValidationResult(Severity.ERROR, f"Log file not found: {filepath}")
            )
            return self.results, False
        except json.JSONDecodeError as e:
            self.results.append(
                ValidationResult(
                    Severity.ERROR,
                    f"Invalid JSON in log file: {e}",
                )
            )
            return self.results, False

        if not isinstance(logs, list):
            self.results.append(
                ValidationResult(
                    Severity.ERROR,
                    "Log file must contain a JSON array of log entries",
                )
            )
            return self.results, False

        if not logs:
            self.results.append(
                ValidationResult(Severity.WARNING, "Log file is empty")
            )
            return self.results, True

        # Run all validation checks
        self.validate_schema(logs)
        self.check_duplicates(logs)
        self.validate_cost_calculation(logs)
        self.detect_anomalies(logs)

        is_valid = all(r.severity != Severity.ERROR for r in self.results)
        return self.results, is_valid

    def validate_schema(self, logs: List[Dict]) -> List[ValidationResult]:
        """
        Validate that all log entries have required fields with correct types.

        Args:
            logs: List of log entry dictionaries

        Returns:
            List of validation results
        """
        schema_results = []

        for idx, entry in enumerate(logs):
            if not isinstance(entry, dict):
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} is not a dictionary",
                )
                schema_results.append(result)
                self.results.append(result)
                continue

            # Check required fields
            missing_fields = self.REQUIRED_FIELDS - set(entry.keys())
            if missing_fields:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} missing required fields: {missing_fields}",
                    {"entry_index": idx, "missing_fields": list(missing_fields)},
                )
                schema_results.append(result)
                self.results.append(result)
                continue

            # Validate field types
            type_errors = []

            if not isinstance(entry.get("timestamp"), str):
                type_errors.append(
                    f"timestamp: expected str, got {type(entry['timestamp']).__name__}"
                )

            if not isinstance(entry.get("model"), str):
                type_errors.append(
                    f"model: expected str, got {type(entry['model']).__name__}"
                )

            if not isinstance(entry.get("input_tokens"), int):
                type_errors.append(
                    f"input_tokens: expected int, got {type(entry['input_tokens']).__name__}"
                )

            if not isinstance(entry.get("output_tokens"), int):
                type_errors.append(
                    f"output_tokens: expected int, got {type(entry['output_tokens']).__name__}"
                )

            if not isinstance(entry.get("cost"), (int, float)):
                type_errors.append(
                    f"cost: expected int or float, got {type(entry['cost']).__name__}"
                )

            if type_errors:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} has type errors: {'; '.join(type_errors)}",
                    {"entry_index": idx, "type_errors": type_errors},
                )
                schema_results.append(result)
                self.results.append(result)

            # Validate model is recognized (only if it's a string after type check)
            if isinstance(entry.get("model"), str):
                model_lower = entry.get("model", "").lower()
                if model_lower not in self.PRICING:
                    result = ValidationResult(
                        Severity.WARNING,
                        f"Entry {idx} has unrecognized model: {entry.get('model')} "
                        f"(known models: {list(self.PRICING.keys())})",
                        {"entry_index": idx, "model": entry.get("model")},
                    )
                    schema_results.append(result)
                    self.results.append(result)

            # Validate tokens are non-negative (only if correct type)
            input_tokens_val = entry.get("input_tokens")
            if isinstance(input_tokens_val, int) and input_tokens_val < 0:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} has negative input_tokens: {input_tokens_val}",
                    {"entry_index": idx, "input_tokens": input_tokens_val},
                )
                schema_results.append(result)
                self.results.append(result)

            output_tokens_val = entry.get("output_tokens")
            if isinstance(output_tokens_val, int) and output_tokens_val < 0:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} has negative output_tokens: {output_tokens_val}",
                    {"entry_index": idx, "output_tokens": output_tokens_val},
                )
                schema_results.append(result)
                self.results.append(result)

            cost_val = entry.get("cost")
            if isinstance(cost_val, (int, float)) and cost_val < 0:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} has negative cost: {cost_val}",
                    {"entry_index": idx, "cost": cost_val},
                )
                schema_results.append(result)
                self.results.append(result)

        if not schema_results:
            result = ValidationResult(
                Severity.PASS, f"Schema validation passed for all {len(logs)} entries"
            )
            schema_results.append(result)
            self.results.append(result)

        return schema_results

    def check_duplicates(self, logs: List[Dict]) -> List[ValidationResult]:
        """
        Check for duplicate entries (same timestamp + model + input_tokens + output_tokens).

        Args:
            logs: List of log entry dictionaries

        Returns:
            List of validation results
        """
        duplicate_results = []
        seen = {}

        for idx, entry in enumerate(logs):
            # Create a key from timestamp, model, and token counts
            model = entry.get("model", "")
            model_lower = model.lower() if isinstance(model, str) else str(model).lower()

            key = (
                entry.get("timestamp"),
                model_lower,
                entry.get("input_tokens"),
                entry.get("output_tokens"),
            )

            if key in seen:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Duplicate entry detected at index {idx} "
                    f"(previously seen at index {seen[key]}): "
                    f"timestamp={entry.get('timestamp')}, "
                    f"model={entry.get('model')}, "
                    f"input_tokens={entry.get('input_tokens')}, "
                    f"output_tokens={entry.get('output_tokens')}",
                    {
                        "current_index": idx,
                        "first_occurrence_index": seen[key],
                        "duplicate_key": str(key),
                    },
                )
                duplicate_results.append(result)
                self.results.append(result)
            else:
                seen[key] = idx

        if not duplicate_results:
            result = ValidationResult(
                Severity.PASS, f"No duplicates found in {len(logs)} entries"
            )
            duplicate_results.append(result)
            self.results.append(result)

        return duplicate_results

    def validate_cost_calculation(self, logs: List[Dict]) -> List[ValidationResult]:
        """
        Verify that cost calculations match official pricing.

        Args:
            logs: List of log entry dictionaries

        Returns:
            List of validation results
        """
        calc_results = []
        tolerance = 0.0001  # Allow small rounding differences

        for idx, entry in enumerate(logs):
            model_value = entry.get("model", "")

            # Skip if model is not a string
            if not isinstance(model_value, str):
                continue

            model = model_value.lower()
            input_tokens = entry.get("input_tokens", 0)
            output_tokens = entry.get("output_tokens", 0)
            recorded_cost = entry.get("cost", 0)

            # Skip if types are invalid for cost calculation
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int) or not isinstance(recorded_cost, (int, float)):
                continue

            if model not in self.PRICING:
                continue  # Skip if model not recognized

            pricing = self.PRICING[model]
            expected_cost = (
                (input_tokens / 1_000_000) * pricing["input"]
                + (output_tokens / 1_000_000) * pricing["output"]
            )

            # Round to reasonable precision (6 decimal places for USD)
            expected_cost = round(expected_cost, 6)
            recorded_cost_rounded = round(recorded_cost, 6)

            if abs(expected_cost - recorded_cost_rounded) > tolerance:
                result = ValidationResult(
                    Severity.ERROR,
                    f"Entry {idx} cost calculation mismatch: "
                    f"expected ${expected_cost:.6f}, got ${recorded_cost:.6f}",
                    {
                        "entry_index": idx,
                        "model": entry.get("model"),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "expected_cost": expected_cost,
                        "recorded_cost": recorded_cost,
                        "difference": abs(expected_cost - recorded_cost_rounded),
                    },
                )
                calc_results.append(result)
                self.results.append(result)

        if not calc_results:
            result = ValidationResult(
                Severity.PASS, f"Cost calculations verified for all {len(logs)} entries"
            )
            calc_results.append(result)
            self.results.append(result)

        return calc_results

    def detect_anomalies(self, logs: List[Dict]) -> List[ValidationResult]:
        """
        Detect anomalies like unusual token counts, cost spikes, etc.

        Args:
            logs: List of log entry dictionaries

        Returns:
            List of validation results
        """
        anomaly_results = []

        if not logs:
            return anomaly_results

        # Extract valid entries for statistical analysis
        valid_entries = []
        for e in logs:
            if all(k in e for k in ["input_tokens", "output_tokens", "cost"]):
                # Only include if types are correct
                if (
                    isinstance(e.get("input_tokens"), int)
                    and isinstance(e.get("output_tokens"), int)
                    and isinstance(e.get("cost"), (int, float))
                ):
                    valid_entries.append(e)

        if not valid_entries:
            return anomaly_results

        # Calculate statistics
        costs = [e["cost"] for e in valid_entries if e["cost"] > 0]
        input_tokens = [
            e["input_tokens"]
            for e in valid_entries
            if e["input_tokens"] > 0
        ]
        output_tokens = [
            e["output_tokens"]
            for e in valid_entries
            if e["output_tokens"] > 0
        ]

        # Check individual entries for high values
        for idx, entry in enumerate(logs):
            # Skip entries with invalid types for anomaly checking
            entry_input_tokens = entry.get("input_tokens", 0)
            entry_output_tokens = entry.get("output_tokens", 0)
            entry_cost = entry.get("cost", 0)

            if not isinstance(entry_input_tokens, int) or not isinstance(entry_output_tokens, int) or not isinstance(entry_cost, (int, float)):
                continue

            # Check high input tokens
            if entry_input_tokens > self.ANOMALY_THRESHOLDS[
                "high_input_tokens"
            ]:
                result = ValidationResult(
                    Severity.WARNING,
                    f"Entry {idx} has unusually high input tokens: "
                    f"{entry_input_tokens:,}",
                    {"entry_index": idx, "input_tokens": entry_input_tokens},
                )
                anomaly_results.append(result)
                self.results.append(result)

            # Check high output tokens
            if entry_output_tokens > self.ANOMALY_THRESHOLDS[
                "high_output_tokens"
            ]:
                result = ValidationResult(
                    Severity.WARNING,
                    f"Entry {idx} has unusually high output tokens: "
                    f"{entry_output_tokens:,}",
                    {"entry_index": idx, "output_tokens": entry_output_tokens},
                )
                anomaly_results.append(result)
                self.results.append(result)

            # Check high cost
            if entry_cost > self.ANOMALY_THRESHOLDS["high_cost"]:
                result = ValidationResult(
                    Severity.WARNING,
                    f"Entry {idx} has high cost: ${entry_cost:.6f}",
                    {"entry_index": idx, "cost": entry_cost},
                )
                anomaly_results.append(result)
                self.results.append(result)

        # Detect spikes using median
        if len(costs) >= 3:
            median_cost = statistics.median(costs)
            threshold_cost = median_cost * self.ANOMALY_THRESHOLDS["cost_spike_multiplier"]
            if median_cost > 0:
                for idx, entry in enumerate(logs):
                    entry_cost = entry.get("cost", 0)
                    if isinstance(entry_cost, (int, float)) and entry_cost > threshold_cost:
                        result = ValidationResult(
                            Severity.WARNING,
                            f"Entry {idx} cost spike detected: "
                            f"${entry_cost:.6f} ({self.ANOMALY_THRESHOLDS['cost_spike_multiplier']}x median)",
                            {
                                "entry_index": idx,
                                "cost": entry_cost,
                                "median_cost": median_cost,
                                "spike_threshold": threshold_cost,
                            },
                        )
                        anomaly_results.append(result)
                        self.results.append(result)

        if input_tokens and len(input_tokens) >= 3:
            median_input = statistics.median(input_tokens)
            threshold_input = (
                median_input * self.ANOMALY_THRESHOLDS["token_spike_multiplier"]
            )
            if median_input > 0:
                for idx, entry in enumerate(logs):
                    entry_input = entry.get("input_tokens", 0)
                    if isinstance(entry_input, int) and entry_input > threshold_input:
                        result = ValidationResult(
                            Severity.WARNING,
                            f"Entry {idx} input token spike detected: "
                            f"{entry_input:,} "
                            f"({self.ANOMALY_THRESHOLDS['token_spike_multiplier']}x median)",
                            {
                                "entry_index": idx,
                                "input_tokens": entry_input,
                                "median_input_tokens": median_input,
                                "spike_threshold": threshold_input,
                            },
                        )
                        anomaly_results.append(result)
                        self.results.append(result)

        if output_tokens and len(output_tokens) >= 3:
            median_output = statistics.median(output_tokens)
            threshold_output = (
                median_output * self.ANOMALY_THRESHOLDS["token_spike_multiplier"]
            )
            if median_output > 0:
                for idx, entry in enumerate(logs):
                    entry_output = entry.get("output_tokens", 0)
                    if isinstance(entry_output, int) and entry_output > threshold_output:
                        result = ValidationResult(
                            Severity.WARNING,
                            f"Entry {idx} output token spike detected: "
                            f"{entry_output:,} "
                            f"({self.ANOMALY_THRESHOLDS['token_spike_multiplier']}x median)",
                            {
                                "entry_index": idx,
                                "output_tokens": entry_output,
                                "median_output_tokens": median_output,
                                "spike_threshold": threshold_output,
                            },
                        )
                        anomaly_results.append(result)
                        self.results.append(result)

        if not anomaly_results:
            result = ValidationResult(
                Severity.PASS, f"No anomalies detected in {len(logs)} entries"
            )
            anomaly_results.append(result)
            self.results.append(result)

        return anomaly_results

    def get_report(self) -> str:
        """
        Generate a formatted validation report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available."

        report_lines = ["COST VALIDATION REPORT", "=" * 80]

        # Group results by severity
        by_severity = {}
        for result in self.results:
            if result.severity not in by_severity:
                by_severity[result.severity] = []
            by_severity[result.severity].append(result)

        # Report order
        for severity in [Severity.ERROR, Severity.WARNING, Severity.PASS]:
            if severity in by_severity:
                report_lines.append(f"\n{severity.value}S ({len(by_severity[severity])}):")
                report_lines.append("-" * 80)
                for result in by_severity[severity]:
                    report_lines.append(f"  {result.message}")
                    if result.details:
                        for key, value in result.details.items():
                            report_lines.append(f"    - {key}: {value}")

        # Summary
        error_count = len(by_severity.get(Severity.ERROR, []))
        warning_count = len(by_severity.get(Severity.WARNING, []))
        pass_count = len(by_severity.get(Severity.PASS, []))

        report_lines.append("\n" + "=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append(f"  Errors:   {error_count}")
        report_lines.append(f"  Warnings: {warning_count}")
        report_lines.append(f"  Passed:   {pass_count}")
        report_lines.append(f"  Total:    {len(self.results)}")

        if error_count == 0:
            report_lines.append("\n✓ VALIDATION PASSED")
        else:
            report_lines.append("\n✗ VALIDATION FAILED")

        return "\n".join(report_lines)
