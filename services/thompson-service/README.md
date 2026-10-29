# RH Thompson Router Service

**Purpose:** Model selection authority using Thompson Sampling (Bayesian multi-armed bandit)

Tracks performance of all available models (Haiku, Sonnet, Opus, Cursor, Gemini) and selects the best one for each task type using probabilistic sampling.

---

## Quick Start

**Start service (no sudo needed):**
```bash
systemctl --user start rh-thompson.service
```

**Check status:**
```bash
systemctl --user status rh-thompson.service
```

**Watch logs:**
```bash
journalctl --user-unit rh-thompson.service -f
```

**Stop:**
```bash
systemctl --user stop rh-thompson.service
```

---

## What It Does

1. **Tracks model performance** — Maintains success/failure counts per model
2. **Learns task types** — Different tasks may need different models
3. **Bayesian sampling** — Uses Beta distributions to balance exploration vs exploitation
4. **Selects models** — Returns the best-performing model for a given task

**Example flow:**
```
Task: "code_review"
  → Thompson samples from Beta distribution for each model
  → Returns model with highest expected reward (e.g., "sonnet")
  
Later: Learning service records outcome (rating 4/5)
  → Thompson updates success count for sonnet on code_review
  → Next time, sonnet will have higher probability
```

---

## Command Reference

### Start/Stop/Status

```bash
# Start (user service, no sudo)
systemctl --user start rh-thompson.service

# Stop
systemctl --user stop rh-thompson.service

# Restart
systemctl --user restart rh-thompson.service

# Status
systemctl --user status rh-thompson.service

# Enable auto-start on login
systemctl --user enable rh-thompson.service
```

### Logs

```bash
# Last 50 lines
journalctl --user-unit rh-thompson.service -n 50

# Follow live
journalctl --user-unit rh-thompson.service -f

# Last hour
journalctl --user-unit rh-thompson.service --since "1 hour ago"
```

### Check State

```bash
# All models
cat learning/thompson-sampling-state.json | jq '.models'

# Just Haiku
cat learning/thompson-sampling-state.json | jq '.models.haiku'

# Pretty print
cat learning/thompson-sampling-state.json | jq '.'
```

---

## Troubleshooting

**Service won't start:**
```bash
journalctl --user-unit rh-thompson.service -n 20
```

**"Address already in use" (socket file stale):**
```bash
rm /tmp/rh-thompson.sock
systemctl --user restart rh-thompson.service
```

**"Connection refused":**
```bash
# Service not running
systemctl --user start rh-thompson.service
```

**Always selecting same model:**
- Check state: `cat learning/thompson-sampling-state.json | jq '.models'`
- Need ~10+ outcomes per model for Thompson to learn

---

## Architecture

- **Listen:** Unix socket `/tmp/rh-thompson.sock`
- **Protocol:** JSON-RPC
- **State:** `learning/thompson-sampling-state.json`
- **Thread model:** Single-threaded with atomic file locking
- **Memory:** 256M max, 25% CPU quota

---

## Integration

**Used by:**
- `rh-api-wrapper.py` — Model selection
- `learning-service` — Outcome recording
- Client library: `shared/thompson_client.py`

**Outputs:**
- `/tmp/rh-thompson.sock` — JSON-RPC socket
- `learning/thompson-sampling-state.json` — Persistent state
- `journalctl` — Structured logs
