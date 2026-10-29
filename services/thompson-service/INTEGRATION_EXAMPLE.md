# Thompson Router Integration Examples

## Quick Start

### 1. Install the Thompson Router Daemon

```bash
./thompson-service/install.sh
```

This installs and starts the daemon as a systemd user service.

### 2. Use in Session Code

Import the client and use it to select models and record outcomes:

```python
from shared.thompson_client import ThompsonClient

client = ThompsonClient()

# Select the best model for a code-review task
model = client.select_model(
    task_type="code-review",
    required_capability=0.7,
    max_cost=0.05
)

# Use the selected model...
# (In real code, you'd invoke the model here)

# Record the outcome when done
success = True  # task succeeded
cost = 0.0234   # actual API cost
tokens = 1250   # tokens used

client.record_outcome(
    model=model,
    task_type="code-review",
    success=success,
    cost=cost,
    tokens=tokens
)
```

## Example: Hook Integration

In an `rh-tools` hook that runs code review:

```python
#!/usr/bin/env python3
"""
code-review hook - Select model dynamically
"""

import sys
import time
from pathlib import Path
from shared.thompson_client import ThompsonClient
from shared.anthropic_client import AnthropicClient

def code_review(pr_diff):
    """Perform code review with Thompson-selected model"""
    
    thompson = ThompsonClient()
    claude = AnthropicClient()
    
    # Select model: need high capability for code review, keep cost < $0.10
    model = thompson.select_model(
        task_type="code-review",
        required_capability=0.8,
        max_cost=0.10
    )
    
    print(f"Selected {model} for code review")
    
    # Make the call
    start_time = time.time()
    response = claude.call_model(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Review this code:\n{pr_diff}"
            }
        ]
    )
    elapsed = time.time() - start_time
    
    # Calculate cost (example - actual cost depends on model)
    cost = claude.calculate_cost(model, response.usage)
    success = response.content and len(response.content) > 0
    
    # Record outcome for future model selection
    thompson.record_outcome(
        model=model,
        task_type="code-review",
        success=success,
        cost=cost,
        tokens=response.usage.output_tokens
    )
    
    return response.content


if __name__ == '__main__':
    pr_diff = sys.stdin.read()
    review = code_review(pr_diff)
    print(review)
```

## Example: Multi-Task with Different Constraints

Different tasks may have different capability/cost requirements:

```python
from shared.thompson_client import ThompsonClient

thompson = ThompsonClient()

# Code review: high capability, moderate cost
review_model = thompson.select_model(
    task_type="code-review",
    required_capability=0.8,
    max_cost=0.10
)

# Simple documentation: low capability, very low cost
doc_model = thompson.select_model(
    task_type="documentation",
    required_capability=0.3,
    max_cost=0.02
)

# Architecture analysis: highest capability, higher cost OK
arch_model = thompson.select_model(
    task_type="architecture-analysis",
    required_capability=0.95,
    max_cost=0.50
)
```

## Example: Monitoring and Inspection

View current model statistics:

```python
from shared.thompson_client import ThompsonClient
import json

thompson = ThompsonClient()
state = thompson.get_state()

if state:
    print("Model Statistics:")
    for model_name, stats in state['models'].items():
        success_rate = stats['successes'] / stats['calls'] if stats['calls'] > 0 else 0
        avg_cost = stats['total_cost'] / stats['calls'] if stats['calls'] > 0 else 0
        
        print(f"\n{model_name}:")
        print(f"  Calls: {stats['calls']}")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Avg cost: ${avg_cost:.4f}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print(f"  Total tokens: {stats['total_tokens']}")
else:
    print("Thompson service unavailable")
```

## Example: Reset Model History

If a model's performance changes (update or recalibration):

```python
from shared.thompson_client import ThompsonClient

thompson = ThompsonClient()

# Reset haiku to neutral priors
success = thompson.reset('haiku')

if success:
    print("Reset haiku - starting fresh exploration")
else:
    print("Failed to reset (service unavailable)")
```

## Example: Cost-Constrained Selection

Select based on budget constraints:

```python
from shared.thompson_client import ThompsonClient

thompson = ThompsonClient()

# Cheap task (e.g., simple formatting)
model = thompson.select_model(
    task_type="formatting",
    max_cost=0.01  # Less than 1 cent
)

# Medium task
model = thompson.select_model(
    task_type="code-review",
    max_cost=0.05  # Less than 5 cents
)

# Premium task (complex analysis)
model = thompson.select_model(
    task_type="security-audit",
    max_cost=0.50  # Up to 50 cents
)
```

## Graceful Degradation

If the Thompson service is down, the client falls back to 'haiku':

```python
from shared.thompson_client import ThompsonClient

thompson = ThompsonClient()

# If daemon is running, uses Thompson Sampling
# If daemon is down, returns 'haiku' with a warning log
model = thompson.select_model("my-task")
```

This ensures code continues working even if the daemon crashes or hasn't been installed.

## Service Status Checks

```bash
# Check if service is running
systemctl --user status rh-thompson.service

# View recent logs
journalctl --user -u rh-thompson.service -n 20

# View service logs in real-time
journalctl --user -u rh-thompson.service -f

# Restart service
systemctl --user restart rh-thompson.service
```

## Debugging

Enable debug logging by setting `PYTHONPATH`:

```python
import logging

# In your hook/session code:
logging.basicConfig(level=logging.DEBUG)

from shared.thompson_client import ThompsonClient

# Now you'll see detailed logs
thompson = ThompsonClient()
```

Check the log file:

```bash
tail -f ~/.claude/rh-thompson-service.log
```

## Performance Notes

**Single-threaded design:**
- No mutex overhead
- Sequential request processing
- Typical response time: <5ms
- Socket timeout: 2 seconds (configurable)

**State persistence:**
- Atomic writes to JSON file
- Full state loaded on startup (~1-5ms)
- State saved after each outcome (~5-10ms)

**Model selection:**
- Beta distribution sampling: ~0.1ms
- Cost filtering: O(n) where n = number of models
- Typical selection time: <1ms

**Example timing:**
```
select_model() call:     ~1-2ms
record_outcome() call:   ~5-10ms (includes file write)
get_state() call:        ~1ms
```

The daemon runs with limited resources:
- Memory limit: 256MB
- CPU quota: 25% (1 core)
