# Cost Validator Usage Guide

## Overview

The `CostValidator` module validates Claude API usage logs against official pricing, detects duplicates, validates schema, and identifies cost anomalies.

## Quick Start

```python
from validator import CostValidator

# Create a validator instance
validator = CostValidator()

# Validate an entire log file
results, is_valid = validator.validate_log_file("cost_log.json")

# Print formatted report
print(validator.get_report())
```

## Log File Format

Cost logs must be valid JSON arrays with entries containing these fields:

### Required Fields
- `timestamp` (string): ISO 8601 format (e.g., "2026-09-25T10:00:00Z")
- `model` (string): Model name (haiku, sonnet, opus, gemini)
- `input_tokens` (integer): Number of input tokens (non-negative)
- `output_tokens` (integer): Number of output tokens (non-negative)
- `cost` (number): Total cost in USD (non-negative)

### Optional Fields
- `session_id` (string): Session identifier
- `task_description` (string): What the model was asked to do
- `project` (string): Project name or ID
- `notes` (string): Additional notes

### Example Log Entry

```json
{
  "timestamp": "2026-09-25T10:00:00Z",
  "model": "haiku",
  "input_tokens": 5000,
  "output_tokens": 1000,
  "cost": 0.0064,
  "session_id": "sess_001",
  "task_description": "Code analysis"
}
```

## Validation Checks

### 1. Schema Validation
- Verifies all required fields are present
- Checks field types are correct
- Validates model names are recognized
- Detects negative token counts or costs

**Severity**: ERROR if required fields missing or types incorrect

### 2. Duplicate Detection
- Identifies entries with same timestamp + model + input/output tokens
- Uses exact value matching (case-insensitive model names)
- Reports first and current indices for easy tracking

**Severity**: ERROR for exact duplicates

### 3. Cost Calculation Verification
- Recalculates expected cost from tokens and official pricing
- Uses official Anthropic pricing:
  - Haiku: $0.80 input / $2.40 output per 1M tokens
  - Sonnet: $3.00 input / $15.00 output per 1M tokens
  - Opus: $15.00 input / $45.00 output per 1M tokens
  - Gemini: $0.075 input / $0.30 output per 1M tokens
- Allows 0.0001 USD tolerance for rounding

**Severity**: ERROR if calculation mismatch exceeds tolerance

### 4. Anomaly Detection
Flags unusual patterns that may indicate data quality issues:

#### High Individual Values
- Input tokens > 100,000 → WARNING
- Output tokens > 100,000 → WARNING
- Cost > $50 → WARNING

#### Statistical Spikes (based on median)
- Cost spike: > 5x median cost → WARNING
- Input spike: > 3x median input tokens → WARNING
- Output spike: > 3x median output tokens → WARNING

**Severity**: WARNING for all anomalies

## Interpreting Results

The validator returns a list of `ValidationResult` objects, each with:
- `severity`: PASS, WARNING, or ERROR
- `message`: Human-readable description
- `details`: Dict with specific values for investigation

## Example Usage

```python
from validator import CostValidator, Severity

validator = CostValidator()
results, is_valid = validator.validate_log_file("logs/costs.json")

# Access results programmatically
for result in results:
    if result.severity == Severity.ERROR:
        print(f"ERROR: {result.message}")
        if result.details:
            print(f"  Details: {result.details}")

# Get count by severity
errors = [r for r in results if r.severity == Severity.ERROR]
warnings = [r for r in results if r.severity == Severity.WARNING]

print(f"Found {len(errors)} errors and {len(warnings)} warnings")
```

## Handling Invalid Data

The validator gracefully handles edge cases:
- Missing optional fields → Skipped (not flagged as error)
- Invalid field types → Type error reported, validation continues
- String instead of number → Type error, value skipped for calculations
- Unrecognized models → Warning, but cost calculation skipped
- Empty log file → Warning, validation passes overall

## Pricing Verification

To verify official pricing is correct, check:
- Haiku: Input $0.80 / 1M, Output $2.40 / 1M
- Sonnet: Input $3.00 / 1M, Output $15.00 / 1M  
- Opus: Input $15.00 / 1M, Output $45.00 / 1M
- Gemini: Input $0.075 / 1M, Output $0.30 / 1M

These values are hardcoded in the `PRICING` dict and should be updated if Anthropic adjusts official rates.

## Common Issues

### "Cost calculation mismatch"
The recorded cost differs from expected cost. Causes:
- Decimal rounding error (check tolerance = 0.0001)
- Wrong model specified (typo in model name)
- Incorrect token counts
- Manual cost entry error

### "Unrecognized model"
Model name not in official pricing list. Causes:
- Typo in model name (e.g., "Haiku" vs "haiku")
- Using non-Anthropic model (e.g., "gpt-4")
- Custom model alias not registered

### "Duplicate entry"
Two entries have identical timestamp, model, and token counts. Causes:
- Accidental duplicate logging
- Same query run twice in same second
- Data loading/import error

### "Token spike detected"
Token count is 3x+ the median. This is not necessarily an error - causes include:
- Legitimate large batch processing
- Integration test with verbose output
- Complex analysis task with high context

## Performance

For typical cost logs (1000s of entries):
- Schema validation: ~5ms
- Duplicate detection: ~2ms  
- Cost calculation: ~10ms
- Anomaly detection: ~15ms
- Total: ~30ms for complete validation

Memory usage: ~1KB per log entry

## Testing

Run the included test suite:

```bash
python test_validator.py
```

Tests cover:
- Valid logs with no errors
- Duplicate entry detection
- Wrong cost calculations
- Missing required fields
- Invalid field types
- Anomaly detection (high values)
- Anomaly detection (spikes)
- Negative values
- Unrecognized models
- Empty log files
- Complex scenarios with multiple errors
- Gemini pricing

All 12 tests should pass with expected behavior.
