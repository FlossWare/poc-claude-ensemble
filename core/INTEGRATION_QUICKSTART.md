# Integration & Learning Quickstart

Everything is now **integrated and learning-ready**. Here's how to use it.

---

## 1. Task-Based Work (Multi-Phase Arbitration)

For critical decisions that need consensus:

```bash
rh-api-wrapper.py --task code-review --input src/api --phases 2
```

What happens:
1. **Model selected** via Thompson router
2. **Phase 1:** Worker models analyze, Arbiter synthesizes
3. **Phase 2:** New workers receive arbiter's output, re-analyze, Arbiter 2 synthesizes
4. **Cost logged** to `cost_tracking/api_costs.jsonl`
5. **You rate outcome** (1-5) — will prompt after completion
6. **Consensus evaluation** — Different model evaluates your rating
7. **Thompson updated** — If you rated 5 and consensus agrees → boosts this model for similar tasks
8. **Alerts checked** — Email sent if cost spike or quality drop detected

---

## 2. Prompt-Based Work (Direct Execution)

For routine tasks, single model:

```bash
rh-api-wrapper.py --prompt "refactor this function" --input myfile.py --model sonnet
```

Or let Thompson pick:

```bash
rh-api-wrapper.py --prompt "analyze security" --input src/ --model auto
```

Same learning flow as above.

---

## 3. System Confidence & Rating Prompts

After your first task:

```
================================================================================
TASK COMPLETED
================================================================================
Task:        code-review
Model:       Haiku
Tokens:      2000 input, 1200 output
Cost:        $0.0065
Time:        3.2s

Rate this outcome (1-5, or press Enter to skip):
  1 = Poor (wrong/slow/expensive)
  2 = Below average
  3 = Acceptable (did the job)
  4 = Good
  5 = Excellent (fast/cheap/quality)

> _
```

**What happens based on your rating:**

| You Rate | Consensus Says | Outcome |
|----------|---|---|
| 5 | 5 | Perfect! Thompson boosted Haiku for code-review |
| 5 | 2 | Close call. Haiku rated 3.5 overall (avg) |
| 3 | 4 | You: acceptable. Consensus: good. Haiku → rated 3.5 |
| 2 | 2 | Agreement: Haiku bad for this task, priors updated |

**Skipping ratings:**
- Just press Enter
- System uses consensus model only
- Same learning happens (you just don't see it)

---

## 4. When Do I Stop Getting Asked?

Your task type starts like this:

```
Days 1-3:  (tasks 1-20)
  System confidence: 0-20%
  → Always ask you to rate
  
Week 2: (tasks 20-40)
  System confidence: 30-70%
  → Ask ~30% of the time
  
Week 3+: (tasks 40+)
  System confidence: 70-90%
  → Ask ~5% of the time
  
Month+:
  System confidence: 90%+
  → Skip asking, auto-rate via consensus
  → You can override anytime
```

**Thompson learns task types independently:**
- If you do 100 code reviews but only 5 security audits
- You'll keep rating security audits longer (need more signal)
- But code-review will be auto-rated much sooner

---

## 5. Emails You'll Receive

**Cost spike alert** (if daily > 2x baseline):
```
Subject: [Claude Ensemble AI Toolkit] COST_SPIKE: WARNING

MESSAGE:
Daily cost spike detected: $12.50 (baseline: $5.00)

METRICS:
- today_cost: 12.5
- baseline: 5.0
- threshold: 10.0
- multiplier: 2.0

RECOMMENDED ACTION:
Review model selection or task complexity
```

**Quality drop alert** (if avg rating < 3.0 over 7 days):
```
Subject: [Claude Ensemble AI Toolkit] QUALITY_DROP: CRITICAL

MESSAGE:
Quality degradation: avg rating 2.3 < 3.0

METRICS:
- avg_rating: 2.3
- threshold: 3.0
- sample_size: 15
- window_days: 7

RECOMMENDED ACTION:
Review model selection; consider returning to previous settings
```

---

## 6. Monitor Your Learning

Check recent outcomes:

```bash
# View recent task outcomes
ls -lt learning/post_task_outcomes/ | head -10

# View a specific outcome
cat learning/post_task_outcomes/task_001_code_review.json | jq .

# View Thompson state (models + success rates)
ls -la learning/thompson-sampling-state.json

# Check alerts sent
ls -lt alerts/ | head -10
```

---

## 7. Dashboard: See Your Progress

After 5+ tasks:

```bash
# Cost trends
cost-dashboard.py

# Thompson routing performance
thompson-dashboard.py

# Learning outcomes
autonomous-learning-dashboard.py

# Quality over time
ga-tuning-dashboard.py
```

---

## 8. Handling Disagreements

If consensus says 5 but you know it was actually 2:

**Option 1: Just rate it honestly**
- You rate: 2
- Consensus: 5
- System splits difference: ~3.5
- Thompson learns the truth over time

**Option 2: Override later**
```bash
# Manual override (future feature)
rh-api.py mark-outcome --task-id abc123 --rating 1 --note "This was wrong"
```

---

## 9. Real Example: First Day

```bash
# Morning: First code review
rh-api-wrapper.py --task code-review --input src/ --phases 2

# System confidence: 0% (no data yet)
# → Asks you to rate
# You: 4 (good work!)
# Consensus: 4 (agrees)
# Thompson: Haiku gets +1 success, total: 1 success / 1 call

# Afternoon: Second code review
rh-api-wrapper.py --task code-review --input src/api --phases 2

# System confidence: 40% (2 data points)
# → Might ask, might skip (randomized)
# You: 5
# Consensus: 4
# Thompson: Haiku gets +1 success, total: 2/2, quality_rate: 100%

# Week 2: 20+ code reviews done
# System confidence: 80%
# → Only asks on sketchy outcomes
# Most tasks auto-rated via consensus
# You don't see rating prompt unless system unsure
```

---

## 10. Fallback & Resilience

If Haiku fails or times out:

```bash
# Automatic fallback to next model
rh-api-wrapper.py --task code-review --input /path

# What happens:
# 1. Haiku selected (Thompson says it's best)
# 2. Haiku fails
# 3. System falls back to Sonnet
# 4. Sonnet succeeds, cost logged
# 5. Error recorded (used fallback)
# 6. Thompson gets signal: Haiku failed on this type
# 7. Priors adjust (Haiku less likely next time for this task)
```

---

## Quick Commands

```bash
# Run a task with automatic learning
rh-api-wrapper.py --task code-review --input src/

# Skip asking, auto-rate (trust consensus)
echo "3" | rh-api-wrapper.py --prompt "analyze" --input file.py

# Check for alerts
python3 tools/alert_manager.py

# View outcomes
ls learning/post_task_outcomes/ | wc -l

# Check Thompson state
grep -o '"calls": [0-9]*' learning/thompson-sampling-state.json
```

---

## Costs

**Per task (approximate):**
- Haiku: $0.005-$0.01
- Sonnet: $0.03-$0.1
- Opus: $0.1-$0.5
- Consensus eval: +50% cost (evaluates your outcome)

**Example:**
- 10 code reviews @ Haiku with consensus = ~$0.1 total
- System learns routing after 10 tasks
- Month+ of usage: $5-50 depending on task complexity

---

## Next: Real Work

You're ready. Start with:

```bash
rh-api-wrapper.py --task code-review --input /path/to/real/code --phases 2
```

The system will learn as you work. No setup needed — just rate tasks (1-5) as they complete.

Questions? Check `learning/post_task_analyzer.py` or `tools/alert_manager.py` for details.

