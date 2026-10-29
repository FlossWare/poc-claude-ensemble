# Learning Service Quick Start

Get the Learning Service running and recording task outcomes in 5 minutes.

## 1. Install the Service

```bash
cd learning-service
./install.sh
```

The script will:
- Install the systemd unit
- Enable auto-start on login
- Start the service immediately

Verify it's running:
```bash
systemctl --user status rh-learning.service
```

You should see:
```
Active: active (running)
```

## 2. Test with Client

Quick test from Python:

```python
from learning.learning_client import LearningClient

client = LearningClient()

# Record a task
client.process_outcome(
    task_id='demo_001',
    task_type='code-review',
    model='haiku',
    rating=5,
    tokens=1500,
    cost=0.008
)

# Get report
report = client.get_report()
print(report)
```

Expected output:
```
{
  'ok': True,
  'total_outcomes': 1,
  'by_model': {
    'haiku': {
      'count': 1,
      'avg_rating': 5.0,
      'total_cost': 0.008
    }
  },
  'by_task_type': {
    'code-review': {
      'count': 1,
      'avg_rating': 5.0
    }
  }
}
```

## 3. Use in Your Sessions

At the end of a task, record the outcome:

```python
from learning.learning_client import LearningClient

# After completing a task...
client = LearningClient()
success = client.process_outcome(
    task_id=task_id,
    task_type='code-review',
    model='haiku',
    rating=user_rating,  # Ask user: 1-5 rating
    tokens=token_count,
    cost=api_cost
)

if success:
    print("✓ Outcome recorded to learning system")
else:
    print("Note: Learning service unavailable (local mode)")
```

## 4. Check Learning Progress

View aggregated statistics:

```python
from learning.learning_client import LearningClient

client = LearningClient()
report = client.get_report()

print(f"Total outcomes recorded: {report['total_outcomes']}")
print(f"\nBy Model:")
for model, stats in report['by_model'].items():
    print(f"  {model}: {stats['count']} tasks, avg rating {stats['avg_rating']:.1f}")

print(f"\nBy Task Type:")
for task_type, stats in report['by_task_type'].items():
    print(f"  {task_type}: {stats['count']} tasks, avg rating {stats['avg_rating']:.1f}")
```

## 5. Monitor Service Health

Check service logs:
```bash
journalctl --user -u rh-learning.service -f
```

Check learning outcomes recorded:
```bash
ls -la ~/.claude/projects/-home-sfloess/learning/autonomous_outcomes/
```

Check latest outcome:
```bash
ls -lt ~/.claude/projects/-home-sfloess/learning/autonomous_outcomes/ | head -2 | tail -1 | awk '{print $NF}' | xargs cat | jq .
```

## Rating Scale

Use a 0-5 scale:
- **5** = Excellent result, exactly what was needed
- **4** = Good result, minor issues
- **3** = Acceptable result, some rework needed
- **2** = Poor result, significant issues
- **1** = Very poor result, unusable
- **0** = Complete failure

The learning system treats rating >= 3 as "success" for Thompson sampling.

## Task Types

Common task types to use:
- `code-review` - Reviewing and analyzing code
- `documentation` - Writing documentation
- `testing` - Writing or running tests
- `refactoring` - Refactoring code
- `debugging` - Finding and fixing bugs
- `design` - Architecture/design decisions
- `research` - Information lookup and research

## Troubleshooting

### Service won't start
```bash
journalctl --user -u rh-learning.service -n 50
```

Check if port 5000 is in use (shouldn't be, we use Unix socket):
```bash
ls -la /tmp/rh-learning.sock
```

### Client can't connect
Check daemon is running:
```bash
systemctl --user status rh-learning.service
```

Check socket exists:
```bash
ls -la /tmp/rh-learning.sock
```

If socket doesn't exist but service shows active, restart:
```bash
systemctl --user restart rh-learning.service
sleep 1
ls -la /tmp/rh-learning.sock
```

### No outcomes recorded
Check file permissions:
```bash
ls -la ~/.claude/projects/-home-sfloess/learning/autonomous_outcomes/
```

Check latest logs:
```bash
journalctl --user -u rh-learning.service -n 20
```

## Next Steps

1. **Integrate with sessions**: Add outcome recording to your task completion hooks
2. **Monitor with Thompson**: Learning outcomes auto-feed to Thompson router
3. **Review trends**: Check monthly reports to see which models perform best for each task type
4. **Adjust routing**: Thompson automatically adjusts model selection based on outcomes

## Files

- `~/.config/systemd/user/rh-learning.service` - Installed unit
- `~/.claude/rh-learning-service.log` - Service log file
- `~/.claude/projects/-home-sfloess/learning/autonomous_outcomes/` - Recorded outcomes
- `~/.claude/projects/-home-sfloess/learning/autonomous_priors/` - Thompson priors (if updated)

## Commands Reference

Start service:
```bash
systemctl --user start rh-learning.service
```

Stop service:
```bash
systemctl --user stop rh-learning.service
```

Restart service:
```bash
systemctl --user restart rh-learning.service
```

View status:
```bash
systemctl --user status rh-learning.service
```

View logs (live):
```bash
journalctl --user -u rh-learning.service -f
```

View logs (last 50 lines):
```bash
journalctl --user -u rh-learning.service -n 50
```

Disable auto-start:
```bash
systemctl --user disable rh-learning.service
```

Re-enable auto-start:
```bash
systemctl --user enable rh-learning.service
```

## Tips

- Record outcomes soon after task completion while you remember quality
- Be honest with ratings (don't inflate scores)
- Use consistent task types for better aggregation
- Run learning reports monthly to spot trends
- Check Thompson router logs to see model selection changing over time
