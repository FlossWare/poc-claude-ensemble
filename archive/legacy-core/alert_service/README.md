# RH Alert Service

**Purpose:** Monitor costs, quality, errors; send email alerts

Watches for anomalies and sends notifications to your-email@example.com via Postfix or Gmail.

## Quick Start

```bash
systemctl --user start rh-alert.service
systemctl --user status rh-alert.service
journalctl --user-unit rh-alert.service -f
```

## What It Does

1. Checks cost spikes (daily > 2x baseline)
2. Checks quality drops (avg rating < 3.0 over 7 days)
3. Checks model errors (fallback usage)
4. Sends email alerts via Postfix or Gmail

## Commands

```bash
# Status
systemctl --user status rh-alert.service

# Logs
journalctl --user-unit rh-alert.service -f

# Recent alerts
ls -lt alerts/ | head -10

# View alert file
cat alerts/cost_spike_*.json | jq '.'

# Delivery log
tail -20 alerts/delivery_log.jsonl
```

## Configuration

- Socket: `/tmp/rh-alert.sock`
- Recipient: your-email@example.com
- Storage: `alerts/`
- Methods: Postfix (primary) or Gmail (fallback)
- Thresholds:
  - Cost spike: 2x baseline
  - Quality drop: avg rating < 3.0 (7 day window)

## Troubleshooting

**Won't start:**
```bash
journalctl --user-unit rh-alert.service -n 20
```

**Alerts not sending:**
- Check Postfix (via SSH tunnel) is accessible
- Check Gmail credentials at ~/.google/service-account-key.json
- Check delivery log: `tail -20 alerts/delivery_log.jsonl`

**Alerts not triggering:**
```bash
# Check alert directory has recent files
ls -l alerts/

# Check learning has outcomes
ls -l learning/post_task_outcomes/

# Manually trigger check
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from tools.alert_manager import AlertManager
manager = AlertManager()
alerts = manager.check_all()
print(f"Triggered: {len(alerts)} alerts")
EOF
```

## Email Setup

### Postfix (via SSH tunnel, preferred)

```bash
# SSH port forward Postfix from mail server
ssh -L 2525:localhost:25 your-mail-server &

# Postfix will use localhost:2525
```

### Gmail (fallback)

```bash
# Place service account key at:
~/.google/service-account-key.json

# Or set environment:
export GOOGLE_APPLICATION_CREDENTIALS=~/.google/service-account-key.json
```

## Alert Types

**cost_spike**
- Triggered when: daily cost > 2x baseline
- Action: Review model selection or task complexity

**quality_drop**
- Triggered when: avg rating < 3.0 over 7 days
- Action: Consider returning to previous settings

**model_error**
- Triggered when: frequent fallback/errors detected
- Action: Check API connectivity and error logs

## Architecture

- Single-threaded event loop
- Async queue for email sending (non-blocking)
- Retry logic: up to 3 attempts with exponential backoff
- Delivery logging in JSONL format
- Graceful degradation (continues if email fails)

## Integration

- Checks: Learning outcomes, cost logs
- Outputs to: `alerts/`, delivery log, email
- Frequency: Checks run every time alert service processes

## Performance

**Response time:** <1 second per check

**Storage:**
- Alert files: ~200 bytes each
- Delivery log: ~100 bytes per attempt
- ~1-5 alerts per week = ~1-5 KB/week

**Email:**
- Immediate queue (async)
- Retry after 1s, 2s delays if fails
- Timeout: 10 seconds per attempt
