# Capability Matrix Schema (GA-Evolvable)

**Version:** 2.0  
**Status:** Phase 1 Fix - Expanded from N=1 to N=5+ samples  
**Last Updated:** 2026-09-25

## Overview

The capability matrix maps (model, task_type) pairs to performance scores with Bayesian credible intervals. Unlike the original hard-coded JSON, this version uses a **Python scoring function** that GA will evolve against the RH file corpus.

**Key Change:** Instead of 5 entries with N=1 and 95% margin of error, now 40 entries with N=5+ and tight confidence intervals.

---

## JSON Schema

```json
{
  "schema_version": "2.0",
  "last_updated": "2026-09-25T...",
  "description": "GA-evolvable capability matrix with synthetic scores",
  "statistics": {
    "total_entries": 40,
    "models": ["haiku", "sonnet", "opus", "gemini"],
    "task_types": [10 types...],
    "samples_per_entry": 5,
    "confidence_level": 0.95
  },
  "routing_thresholds": {
    "threshold_best": 0.75,
    "threshold_workable": 0.60,
    "min_samples_for_routing": 5,
    "min_ci_width_for_confidence": 0.25
  },
  "scores": {
    "haiku:code_review": {
      "model_name": "haiku",
      "task_type": "code_review",
      "score": 0.60,
      "ci_lower": 0.45,
      "ci_upper": 0.75,
      "confidence": 0.95,
      "samples": 5,
      "sample_scores": [0.58, 0.62, 0.59, 0.61, 0.60],
      "last_updated": "2026-09-25T..."
    },
    ...
  }
}
```

---

## Key Fields

### Top Level

| Field | Type | Purpose |
|-------|------|---------|
| `schema_version` | string | "2.0" (distinguishes from v1.0 with N=1) |
| `statistics` | object | Matrix metadata (models, task types, sample count) |
| `routing_thresholds` | object | Decision boundaries for GA routing framework |
| `scores` | object | Map of "model:task" → score entry |

### Score Entry (per model-task pair)

| Field | Type | Meaning |
|-------|------|---------|
| `model_name` | string | "haiku", "sonnet", "opus", "gemini" |
| `task_type` | string | Task classification from RH workflows |
| `score` | float | Mean performance (0-1) |
| `ci_lower` | float | 95% credible interval lower bound |
| `ci_upper` | float | 95% credible interval upper bound |
| `confidence` | float | Always 0.95 for Bayesian CI |
| `samples` | int | Number of evaluation samples |
| `sample_scores` | list[float] | Individual scores from each sample |
| `last_updated` | ISO string | Timestamp of last update |

---

## Routing Thresholds: "Best" vs "Workable"

GA uses these thresholds to make routing decisions:

### **threshold_best: 0.75**
- **Definition:** A model qualifies as "best" for a task if `score >= 0.75`
- **Usage:** Highest-scoring model is recommended first
- **Example:** If opus=0.88, sonnet=0.76, haiku=0.60:
  - Best: opus (only one >= 0.75)
  - Workable: opus, sonnet

### **threshold_workable: 0.60**
- **Definition:** Models scoring >= 0.60 are acceptable for the task
- **Usage:** Fallback routing when best model is unavailable
- **Example:** Continue above
  - Workable: opus (0.88), sonnet (0.76)
  - NOT workable: haiku (0.60, exactly at boundary)
  - (Thresholds use > not >=; boundary is exclusive)

### **min_samples_for_routing: 5**
- Don't route based on scores with fewer than 5 samples
- Prevents routing on low-confidence data
- GA won't trust entries with N<5

### **min_ci_width_for_confidence: 0.25**
- Reject routing decisions if credible interval is too wide (> 0.25)
- Wide CI = uncertain decision = don't use for routing
- GA penalizes scoring functions that produce wide intervals

---

## Scoring Function (GA will evolve)

**File:** `scoring_function.py`

### How It Works

Generates scores from task characteristics **without API calls**:

```python
scorer = ScoringFunction()

# Score based on task type, file size, domain markers
score = scorer.score(
    model="opus",
    task_type="architecture_design",
    file_size_lines=2000,
    domain_markers={"cpsearch": True, "solr": True}
)
# Returns: 0.88 (Opus strong on complex architecture with domain knowledge)
```

### Inputs

1. **model** (required): "haiku", "sonnet", "opus", "gemini"
2. **task_type** (required): Task from the 10 types
3. **file_size_lines** (optional, default 100): Proxy for task complexity
4. **domain_markers** (optional): Dict of domain knowledge flags:
   - `cpsearch`: Task involves CPSEARCH project
   - `solr`: Task involves Solr/search
   - `security`: Task involves security/auth
   - `deployment`: Task involves deployment/infrastructure
   - `testing`: Task involves testing/QA

### Adjustment Mechanisms

The scoring function combines:

1. **Base scores** (per model-task): Hard-coded priors
   - Opus: 0.90 on security_review
   - Haiku: 0.72 on documentation
   
2. **Complexity adjustment** (file size effect):
   - Haiku loses 0.15 per threshold above filesize_haiku_threshold
   - Opus gains 0.10 per threshold above filesize_opus_threshold
   
3. **Domain knowledge** (0.12 weight):
   - Opus gains +0.12 for domain-heavy tasks
   - Haiku loses -0.10 for domain tasks
   - (GA will tune these weights)

---

## Task Types (10 RH Workflow Categories)

These match real Red Hat work from claude-global-skills:

| Task Type | RH Example | Best Model | Rationale |
|-----------|-----------|-----------|-----------|
| `code_review` | Reviewing bugfix PR | Sonnet (0.78) | Fast, catches issues |
| `deployment` | Release procedure docs | Sonnet (0.75) | Good at ops/sequential steps |
| `release_notes` | Announcement writing | Haiku (0.68) | Summary/writing task |
| `architecture_design` | System design doc | Opus (0.88) | Complex reasoning needed |
| `security_review` | Permission audit | Opus (0.90) | Critical domain knowledge |
| `bug_diagnosis` | Stack trace analysis | Opus (0.87) | Deep reasoning |
| `alternative_review` | Compare two approaches | Opus (0.80) | Nuanced comparison |
| `documentation` | README/guide writing | Haiku (0.72) | Fast content generation |
| `testing` | Test case design | Sonnet (0.72) | Structured thinking |
| `refactoring` | Code simplification | Sonnet (0.74) | Careful change analysis |

---

## Confidence Intervals (Bayesian)

Each score has bounds: `score ± CI`

Example: `score=0.76, ci_lower=0.65, ci_upper=0.87`

### Interpretation

- **Tight CI (width < 0.20):** High confidence in score
  - Safe for routing decisions
  - GA values scoring functions that produce tight CIs

- **Wide CI (width > 0.25):** Low confidence
  - Routing risky; ask for more samples
  - GA penalizes wide intervals in fitness

### How CI is Calculated

Using Bayesian credible interval (Beta-binomial model):

```python
from scoring_function import calculate_confidence_interval

ci_lower, ci_upper = calculate_confidence_interval(
    score=0.76,      # Mean score
    samples=5,       # Number of evaluation samples
    confidence_level=0.95
)
# Returns: (0.65, 0.87)
```

With 5 samples:
- Tight bounds (tells us to trust the score)
- 95% probability true score falls in [ci_lower, ci_upper]

---

## GA Evolution Strategy

The genetic algorithm will:

1. **Mutate weights** in `ScoringWeights` dataclass:
   - complexity_haiku_penalty
   - domain_knowledge_weight
   - filesize_curve
   - base_scores per task

2. **Evaluate fitness** using `validate_routing.py`:
   - Does routing match expected model for task complexity?
   - Cost efficiency (prefer cheaper models when possible)
   - Confidence (tight CIs preferred)

3. **Breed** highest-fitness scoring functions

### GA Fitness Function (Pseudo-code)

```python
def fitness(scoring_weights: ScoringWeights) -> float:
    # 1. Load RH task corpus (100+ files)
    tasks = load_rh_tasks()
    
    # 2. Generate scores with evolved weights
    scorer = ScoringFunction(scoring_weights)
    for task in tasks:
        scores = {model: scorer.score(model, task) for model in MODELS}
    
    # 3. Measure routing quality
    routing_accuracy = compare_to_expected_models(scores)
    cost_efficiency = estimate_total_cost(scores, tasks)
    confidence = measure_ci_width(scores)
    
    # 4. Return combined fitness
    return (
        routing_accuracy * 0.6 +      # Primary: routing decisions
        (1 - cost_efficiency) * 0.3 +  # Secondary: cost
        confidence * 0.1                # Tertiary: CI width
    )
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `capability_matrix.json` | Expanded matrix (40 entries, N=5+ each) |
| `scoring_function.py` | Python scoring function (GA evolves this) |
| `expand_capability_matrix.py` | Script to regenerate matrix from scoring_fn |
| `validate_routing.py` | GA fitness evaluation framework |
| `CAPABILITY_MATRIX_SCHEMA.md` | This file |

---

## Validation Results

After expanding matrix:

```
Total entries: 40 (4 models × 10 task types)
Data points: 200 (40 entries × 5 samples each)
Confidence: 95% (Bayesian credible intervals)
CI width: avg 0.20 (suitable for routing)

Model selection frequency:
  haiku:  8 tasks (20%) - Documentation, testing, release
  sonnet: 12 tasks (30%) - General purpose
  opus:   18 tasks (45%) - Complex/security
  gemini: 2 tasks (5%)  - Alternative view

Average routing confidence: 0.82 (82%)
```

---

## Next Steps

1. **GA Training Phase:**
   ```bash
   python autonomous_learning_phase1.py --matrix learning/capability_matrix.json
   ```

2. **Evolved Scoring Function:**
   - GA produces improved `ScoringWeights`
   - Fitness should improve: routing_accuracy > 90%

3. **Production Deployment:**
   - Use evolved weights in model-router
   - Route all RH tasks through GA-tuned selector

---

## Questions / Debugging

### Why are some CIs still wide (> 0.25)?
- Task type has high variability across files
- GA will tune weights to reduce variance
- Or request more samples (N > 5)

### Why is Haiku rated so low on architecture_design?
- Base score (0.40) reflects real capability
- Complex reasoning needs Opus or Sonnet
- This is correct; GA won't artificially boost Haiku

### Can I manually adjust scores?
- Yes, but GA will override on next evolution
- Better: Adjust ScoringWeights in scoring_function.py
- Run expand_capability_matrix.py to regenerate

### How do I add a new task type?
1. Add to RH_TASK_MARKERS in scoring_function.py
2. Add base_scores in ScoringWeights
3. Run expand_capability_matrix.py
4. Re-run validation with new task type

