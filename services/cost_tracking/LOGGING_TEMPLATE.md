# Cost Logging Template

Simple examples for adding real API usage to `api_costs.jsonl`

---

## Quick Add: Manual JSONL Entry

Add directly to `api_costs.jsonl`:

```json
{"timestamp": "2026-09-27T14:30:00.000000+00:00", "model": "haiku", "input_tokens": 2000, "output_tokens": 800, "total_tokens": 2800, "cost_usd": 0.004, "task_name": "code_review", "source": "api", "metadata": {"pr_id": "5678", "task": "my_work"}}
```

Replace these fields:
- `timestamp` — When API call happened (ISO 8601)
- `model` — Which model (haiku, sonnet, opus, cursor, gemini)
- `input_tokens` — From API response
- `output_tokens` — From API response
- `cost_usd` — Calculated from RH rates (see RH_PRICING.md)
- `task_name` — What you were doing (code_review, security_audit, testing, etc.)
- `metadata` — Any extra context

---

## How to Get Token Counts

### From Anthropic API Response
```python
response = client.messages.create(...)
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens
```

### From Claude Code (current session)
```
This session shows token usage in brackets:
[Used X input tokens, Y output tokens]
```

### From GCP Billing Report
```
Column: "Usage" shows total tokens
Column: "SKU" shows if input or output
```

---

## How to Calculate Cost

Using RH rates from `RH_PRICING.md`:

### Haiku
```
cost = (input_tokens × $0.00000125) + (output_tokens × $0.000005)

Example: 2000 input + 800 output
= (2000 × 0.00000125) + (800 × 0.000005)
= 0.0025 + 0.004
= $0.0065
```

### Sonnet
```
cost = (input_tokens × ?) + (output_tokens × $0.000015)

Note: Input rate not yet in RH_PRICING.md
```

### Opus
```
cost = (input_tokens × ?) + (output_tokens × $0.000025)

Note: Input rate not yet in RH_PRICING.md
```

### Cursor
```
cost = TBD (awaiting JetBrains rates)
```

### Gemini
```
cost = TBD (awaiting Google rates)
```

---

## Real Examples

### Code Review (Haiku, $0.004)
```json
{"timestamp": "2026-09-27T10:15:00.000000+00:00", "model": "haiku", "input_tokens": 2500, "output_tokens": 1200, "total_tokens": 3700, "cost_usd": 0.00662, "task_name": "code_review", "source": "api", "metadata": {"pr_id": "1234", "repo": "cpsearch", "lines_reviewed": 150}}
```

### Security Audit (Opus, $0.3)
```json
{"timestamp": "2026-09-27T11:30:00.000000+00:00", "model": "opus", "input_tokens": 8000, "output_tokens": 4000, "total_tokens": 12000, "cost_usd": 0.1, "task_name": "security_audit", "source": "api", "metadata": {"files": 12, "vulns_found": 3, "severity": "high"}}
```

### Architecture Design (Sonnet + Opus consensus, $0.15)
```json
{"timestamp": "2026-09-27T13:00:00.000000+00:00", "model": "sonnet", "input_tokens": 5000, "output_tokens": 3000, "total_tokens": 8000, "cost_usd": 0.045, "task_name": "architecture_design", "source": "api", "metadata": {"phase": 1, "consensus": "sonnet_initial"}}
{"timestamp": "2026-09-27T13:15:00.000000+00:00", "model": "opus", "input_tokens": 6000, "output_tokens": 4500, "total_tokens": 10500, "cost_usd": 0.1125, "task_name": "architecture_design", "source": "api", "metadata": {"phase": 2, "consensus": "opus_challenge"}}
```

### Cache Hit (No additional cost)
```json
{"timestamp": "2026-09-27T14:00:00.000000+00:00", "model": "haiku", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "task_name": "code_review", "source": "cached", "metadata": {"cache_key": "abc123", "cache_age_minutes": 45}}
```

---

## Batch Import

If you have multiple entries, create a batch file:

**File: `daily_costs_2026-09-27.jsonl`**
```json
{"timestamp": "2026-09-27T10:00:00.000000+00:00", "model": "haiku", "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500, "cost_usd": 0.0035, "task_name": "testing", "source": "api"}
{"timestamp": "2026-09-27T11:00:00.000000+00:00", "model": "sonnet", "input_tokens": 3000, "output_tokens": 2000, "total_tokens": 5000, "cost_usd": 0.03, "task_name": "code_review", "source": "api"}
{"timestamp": "2026-09-27T12:00:00.000000+00:00", "model": "opus", "input_tokens": 5000, "output_tokens": 3500, "total_tokens": 8500, "cost_usd": 0.0875, "task_name": "security_review", "source": "api"}
```

Then append:
```bash
cat daily_costs_2026-09-27.jsonl >> cost_tracking/api_costs.jsonl
```

---

## Verify Your Entry

After adding, verify it calculates correctly:

```bash
python3 << 'EOF'
import json

# Your entry
entry = {
    "timestamp": "2026-09-27T10:15:00.000000+00:00",
    "model": "haiku",
    "input_tokens": 2000,
    "output_tokens": 800,
    "cost_usd": 0.0065  # What you calculated
}

# RH rates
RATES = {
    "haiku": {"input": 0.00000125, "output": 0.000005},
    "sonnet": {"input": 0.000003, "output": 0.000015},
    "opus": {"input": 0.000015, "output": 0.000025},
}

model = entry["model"]
calculated = (entry["input_tokens"] * RATES[model]["input"] +
              entry["output_tokens"] * RATES[model]["output"])

print(f"Your cost:        ${entry['cost_usd']:.6f}")
print(f"Calculated cost:  ${calculated:.6f}")
print(f"Match: {'✓' if abs(entry['cost_usd'] - calculated) < 0.00001 else '✗'}")
EOF
```

---

## Notes

- **JSONL format:** One JSON object per line (no pretty-printing)
- **No duplicates check:** System trusts you won't add same entry twice
- **Atomic appends:** Always append, never edit existing lines
- **Timestamps:** Use ISO 8601 with +00:00 UTC offset
- **Cost calculation:** Use rates from RH_PRICING.md

