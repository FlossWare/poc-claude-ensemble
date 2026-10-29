# Integration Guide: Cost Logger with Caching/Compression Modules

This guide shows how to integrate the CostLogger with existing caching and compression modules.

## Integration Pattern

The CostLogger should wrap API calls to capture both tokens and cost:

```python
# With your existing compression module
from cost_tracking import CostLogger
from compression import compress_prompt

logger = CostLogger()
compressor = Compressor()

# Log the compression benefit
prompt = "... original prompt ..."
compressed = compressor.compress(prompt)

savings_tokens = len(prompt) - len(compressed)
logger.log_call(
    model="sonnet",
    input_tokens=5000,
    output_tokens=3500,
    task_name="code_review",
    source="api",
    metadata={
        "compression_ratio": len(compressed) / len(prompt),
        "token_savings": savings_tokens
    }
)
```

## Caching Integration

When using with a caching layer:

```python
from cost_tracking import CostLogger
from cache import PromptCache

logger = CostLogger()
cache = PromptCache()

# Check cache first
cache_key = hash(prompt)
cached_result = cache.get(cache_key)

if cached_result:
    # Log zero tokens for cache hit
    logger.log_call(
        model="opus",
        input_tokens=0,
        output_tokens=0,
        task_name="code_review",
        source="cached",
        metadata={"cache_key": cache_key}
    )
else:
    # Make API call and log actual tokens
    response = call_api(prompt)
    logger.log_call(
        model="opus",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        task_name="code_review",
        source="api",
    )
    cache.set(cache_key, response)
```

## Workflow Context

The `metadata` field is flexible and should capture workflow-specific context:

### For Code Review
```python
metadata={
    "pr_id": "1234",
    "file_count": 5,
    "repo": "cpsearch",
    "severity": "blocker"  # if finding issues
}
```

### For Architecture Design
```python
metadata={
    "design_doc_id": "ARCH-001",
    "reviewed_components": 3,
    "async_recommended": True
}
```

### For Security Review
```python
metadata={
    "security_level": "critical",
    "findings_count": 2,
    "vulnerability_types": ["sql_injection", "xss"],
    "requires_follow_up": True
}
```

### For Consensus Reviews
```python
metadata={
    "phase": 1,  # Step in multi-model review
    "reviewer": "sonnet",  # Which model
    "consensus": "required",
    "blocker_count": 0,
    "agreement_level": "strong"
}
```

## Budget Alerting

Use get_stats() to trigger budget alerts:

```python
logger = CostLogger()
stats = logger.get_stats()

MONTHLY_BUDGET = 1000.00
spent = stats['total_cost_usd']
remaining = MONTHLY_BUDGET - spent

if remaining < 100:
    # Alert if under $100 remaining
    log.warning(f"Approaching budget limit: ${remaining:.2f} remaining")

# Check per-model spending
if stats['by_model']['opus']['cost'] > 500:
    log.warning("Opus spending exceeds $500 this month")
```

## Log Analysis

Query the logs for specific insights:

```python
logger = CostLogger()
logs = logger.read_logs()

# Find expensive tasks
expensive = sorted(
    [(l['task_name'], l['cost_usd']) for l in logs],
    key=lambda x: x[1],
    reverse=True
)
print("Most expensive tasks:", expensive[:5])

# Model usage ratio
model_calls = {}
for log in logs:
    model = log['model']
    model_calls[model] = model_calls.get(model, 0) + 1

# Cache hit analysis (if tracking)
cache_hits = sum(1 for l in logs if l['source'] == 'cached')
api_calls = sum(1 for l in logs if l['source'] == 'api')
cache_hit_rate = cache_hits / (cache_hits + api_calls) if (cache_hits + api_calls) > 0 else 0
print(f"Cache hit rate: {cache_hit_rate*100:.1f}%")
```

## Best Practices

1. **Always log immediately after API calls** — Don't wait or batch, log synchronously
2. **Use consistent task names** — Enables meaningful aggregation and trend analysis
3. **Include workflow context** — Metadata helps understand what drove cost
4. **Track cache hits** — Log with source="cached" and 0 tokens to quantify savings
5. **Monitor by model** — Helps identify when models are over/under-utilized
6. **Regular budget reviews** — Check stats weekly to catch overspending early

## File Structure

```
cost_tracking/
├── __init__.py              # Package exports
├── logger.py                # Main CostLogger class
├── test_logger.py           # Unit tests (run: python test_logger.py)
├── example_usage.py         # Integration examples
├── INTEGRATION.md           # This file
├── README.md                # Module documentation
└── api_costs.jsonl          # Log file (append-only JSON Lines)
```

## Performance Notes

- Logger is non-blocking (simple file append)
- get_stats() reads entire log file into memory — acceptable for typical usage
- For high-volume logging (>10k entries), consider archiving old logs
- JSON Lines format allows streaming analysis with standard tools

Example: Analyze logs with jq
```bash
cat api_costs.jsonl | jq '.cost_usd | add'  # Total cost
cat api_costs.jsonl | jq 'select(.model == "opus") | .cost_usd | add'  # Opus cost
```
