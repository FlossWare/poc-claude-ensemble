# RH Learning Service

**Purpose:** Task outcome recording and autonomous learning

Records completions, calculates confidence, gets consensus ratings, updates Thompson priors.

## Quick Start

```bash
systemctl --user start rh-learning.service
systemctl --user status rh-learning.service
journalctl --user-unit rh-learning.service -f
```

## What It Does

1. Records outcomes (task completions with ratings)
2. Calculates system confidence (how well do we know this model?)
3. Gets consensus ratings (asks other models to rate independently)
4. Updates Thompson (feeds results back to model selector)
5. Checks for anomalies (triggers alerts on issues)

## Commands

```bash
# Status
systemctl --user status rh-learning.service

# Logs
journalctl --user-unit rh-learning.service -f

# Recent outcomes
ls -lt learning/post_task_outcomes/ | head -10

# Learning report
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from learning.learning_client import LearningClient
report = LearningClient().get_report()
print(f"Total: {report.get('total_outcomes')}")
print(f"By model: {report.get('by_model')}")
EOF
```

## Configuration

- Socket: `/tmp/rh-learning.sock`
- Storage: `learning/post_task_outcomes/`
- Memory: 512M max
- Confidence thresholds:
  - <3 calls: 20% confidence (ask user)
  - 3-10 calls: 50% confidence (ask sometimes)
  - 10-30: 75% confidence (ask rarely)
  - 30+: 90% confidence (auto-rate)

## Troubleshooting

**Won't start:**
```bash
journalctl --user-unit rh-learning.service -n 20
```

**Connection refused to Thompson:**
- Check Thompson is running
- Service has circuit breaker fallback

**Outcomes not recording:**
```bash
ls -la learning/post_task_outcomes/
```

## Architecture

- Single-threaded with atomic file writes
- Depends on Thompson service for updates
- Consensus model uses timeout of 30 seconds
- Falls back to heuristic if API fails

## Integration

- Used by: `rh-api-wrapper.py`, `post_task_analyzer.py`
- Outputs to: `/tmp/rh-learning.sock`, `learning/post_task_outcomes/`


## Portable learning artifacts

The Learning service accepts versioned provider-neutral artifacts through its `record_artifact` operation. The current GA integration emits `ga.tuning.result` artifacts with a stable run ID, optimizer provenance, top candidates per evaluator, selected parameters, current fallback settings, and an explicit synthetic-evidence marker. The service persists artifacts in its configured learning directory and writes an operational event through `OperationalMemoryWriter` before acknowledging success.

A failed Memory write is not reported as success, and GA settings are not changed unless the Learning service confirms Memory persistence. GA fitness is synthetic evaluator output, not proof of real-world improvement. Knowledge promotion remains ineligible until separate operational evidence and an explicit promotion rule exist.
