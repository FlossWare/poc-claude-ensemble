# GA Tuning Integration Guide

This guide shows how to apply the GA-optimized parameters to the 5 RH fixed tools.

## Quick Start

1. **Run GA tuning**:
   ```bash
   cd ga_tuning
   python ga_tuner.py
   ```

2. **Load best parameters**:
   ```python
   import json
   
   with open('results/ga_summary_<timestamp>.json') as f:
       summary = json.load(f)
   
   best_params = {
       'compression': summary['best_by_system'].get('compression', [{}])[0].get('parameters', {}),
       'thompson': summary['best_by_system'].get('thompson', [{}])[0].get('parameters', {}),
       'caching': summary['best_by_system'].get('caching', [{}])[0].get('parameters', {}),
       'matrix': summary['best_by_system'].get('matrix', [{}])[0].get('parameters', {}),
       'dashboard': summary['best_by_system'].get('dashboard', [{}])[0].get('parameters', {}),
   }
   ```

3. **Apply to each tool** (see sections below)

## 1. Compression GA

**File**: `../shared/compression.py` or equivalent

**Parameters**:
- `compression_level` (0-5): How aggressively to compress text
- `target_reduction` (0.2-0.7): Target percentage of token reduction

**Integration Example**:

```python
from compression_evaluator import CompressionEvaluator

# Load optimized parameters
compression_params = {
    'compression_level': 2.3,      # From GA
    'target_reduction': 0.35,      # From GA
}

class CompressedMemory:
    def __init__(self, params):
        self.compression_level = params['compression_level']
        self.target_reduction = params['target_reduction']
    
    def compress(self, memory_content):
        """Compress memory while preserving semantics"""
        # Use compression_level to control aggressiveness
        # Aim for target_reduction in token savings
        sentences = memory_content.split('. ')
        n_keep = int(len(sentences) * (1.0 - self.compression_level / 10.0))
        compressed = '. '.join(sentences[:n_keep])
        return compressed

# Apply
compressor = CompressedMemory(compression_params)
compressed_memory = compressor.compress(rh_memory)
```

**Expected Fitness**: 0.60-0.75 (token savings × semantic preservation)

## 2. Thompson Router GA

**File**: `../shared/thompson_router.py`

**Parameters**:
- `alpha_prior` (0.5-3.0): Beta distribution alpha (success belief)
- `beta_prior` (0.5-3.0): Beta distribution beta (failure belief)  
- `cost_weight` (0.1-0.5): Weight given to cost vs. quality in decision

**Integration Example**:

```python
from thompson_router import ThompsonRouter
from scipy.stats import beta as scipy_beta

# Load optimized parameters
thompson_params = {
    'alpha_prior': 1.2,     # From GA
    'beta_prior': 1.8,      # From GA
    'cost_weight': 0.28,    # From GA
}

class OptimizedThompsonRouter(ThompsonRouter):
    def __init__(self, params):
        super().__init__()
        self.alpha_prior = params['alpha_prior']
        self.beta_prior = params['beta_prior']
        self.cost_weight = params['cost_weight']
    
    def select_model(self, task_type):
        """Select model using Thompson sampling with optimized priors"""
        scores = {}
        
        for model, state in self.models.items():
            # Use GA-optimized priors
            alpha = state.get('alpha', self.alpha_prior)
            beta = state.get('beta', self.beta_prior)
            
            # Sample quality belief
            quality_belief = scipy_beta.rvs(alpha, beta)
            
            # Factor in cost with GA-optimized weight
            cost_penalty = self.cost_weight * model.cost
            
            scores[model] = quality_belief * (1.0 - cost_penalty)
        
        return max(scores, key=scores.get)

# Apply
router = OptimizedThompsonRouter(thompson_params)
selected_model = router.select_model('code_review')
```

**Expected Fitness**: 0.60-0.80 (cost savings × quality maintained)

## 3. Caching GA

**File**: `../cache_control.py`

**Parameters**:
- `ttl_seconds` (60-600): Time-to-live for cache entries
- `cache_threshold` (0.1-0.9): Minimum cacheable token ratio

**Integration Example**:

```python
from cache_control import PromptCacheControl

# Load optimized parameters
caching_params = {
    'ttl_seconds': 351.0,         # From GA
    'cache_threshold': 0.145,     # From GA
}

class OptimizedCacheControl(PromptCacheControl):
    def __init__(self, params):
        super().__init__()
        self.ttl_seconds = params['ttl_seconds']
        self.cache_threshold = params['cache_threshold']
    
    def should_cache(self, content, cacheable_tokens):
        """Decide if content should be cached"""
        cacheable_ratio = cacheable_tokens / len(content.split())
        return cacheable_ratio >= self.cache_threshold
    
    def get_cache_control_block(self, content, cacheable_tokens):
        """Build cache_control block with optimized TTL"""
        if self.should_cache(content, cacheable_tokens):
            return {
                "type": "ephemeral",
                "cache_control": {
                    "type": "ephemeral",
                    "max_tokens": cacheable_tokens,
                }
            }
        return None

# Apply
cache = OptimizedCacheControl(caching_params)

# Structure: cacheable content first, user query last
messages = [
    {"role": "user", "content": rh_memory},  # Large, cacheable
    {"role": "user", "content": rh_docs},    # Large, cacheable
    {"role": "user", "content": user_query}, # Small, dynamic
]

# Mark first messages as cacheable
result = cache.call_with_cache(
    messages=messages,
    cache_eligible_indices=[0, 1],  # First two are cacheable
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    cache_control_ttl=int(caching_params['ttl_seconds']),
)
```

**Expected Fitness**: 0.55-0.75 (cache hit rate × token savings)

## 4. Capability Matrix GA

**File**: `../shared/thompson_router.py` (capability_matrix section)

**Parameters**:
- `domain_weight` (0.1-0.5): Weight for domain expertise
- `complexity_weight` (0.2-0.6): Weight for task complexity
- `task_weight` (0.1-0.5): Weight for task type matching

**Integration Example**:

```python
# Load optimized parameters
matrix_params = {
    'domain_weight': 0.3,        # From GA
    'complexity_weight': 0.45,   # From GA
    'task_weight': 0.25,         # From GA
}

class OptimizedCapabilityMatrix:
    # Model capability matrix (from evaluator)
    MATRIX = {
        'haiku': {
            'code_reading': 0.85,
            'documentation': 0.90,
            'code_review': 0.65,
            'architecture': 0.50,
        },
        'sonnet': {
            'code_reading': 0.92,
            'code_review': 0.92,
            'architecture': 0.85,
        },
        'opus': {
            'code_review': 0.97,
            'architecture': 0.95,
        },
    }
    
    def score_model(self, model, task_type, complexity):
        """Score model using GA-optimized weights"""
        # Get base capability for task
        domain_score = self.MATRIX.get(model, {}).get(task_type, 0.5)
        
        # Complexity alignment
        if model == 'haiku':
            complexity_score = 1.0 - complexity  # Good for simple
        elif model == 'sonnet':
            complexity_score = 1.0 - abs(complexity - 0.5)  # Good for medium
        else:
            complexity_score = complexity  # Good for complex
        
        # Weighted combination
        score = (
            matrix_params['domain_weight'] * domain_score +
            matrix_params['complexity_weight'] * complexity_score +
            matrix_params['task_weight'] * (1.0 if model_optimal_for_task else 0.5)
        )
        
        return score / (matrix_params['domain_weight'] + matrix_params['complexity_weight'] + matrix_params['task_weight'])

# Apply
matrix = OptimizedCapabilityMatrix()
model_score = matrix.score_model('opus', 'code_review', complexity=0.8)
```

**Expected Fitness**: 0.75-0.95 (routing accuracy)

## 5. Dashboard GA

**File**: `../autonomous_learning_phase1.py` or dashboard module

**Parameters**:
- `learning_rate` (0.01-0.2): Speed of parameter updates
- `exploration_decay` (0.85-0.99): How fast exploration decays
- `alert_threshold` (0.3-0.9): When to alert on anomalies

**Integration Example**:

```python
import numpy as np
from scipy.stats import beta as scipy_beta

# Load optimized parameters
dashboard_params = {
    'learning_rate': 0.08,      # From GA
    'exploration_decay': 0.92,  # From GA
    'alert_threshold': 0.6,     # From GA
}

class OptimizedDashboard:
    def __init__(self, params):
        self.learning_rate = params['learning_rate']
        self.exploration_decay = params['exploration_decay']
        self.alert_threshold = params['alert_threshold']
        self.model_state = {}  # {model: {'alpha': ..., 'beta': ...}}
        self.task_count = 0
    
    def update_beliefs(self, model, quality_achieved):
        """Update Thompson beliefs with GA-optimized learning rate"""
        if model not in self.model_state:
            self.model_state[model] = {'alpha': 1.0, 'beta': 1.0}
        
        # Update with GA-optimized learning rate
        if quality_achieved > 0.85:  # Success
            self.model_state[model]['alpha'] += self.learning_rate
        else:  # Failure
            self.model_state[model]['beta'] += self.learning_rate
        
        self.task_count += 1
    
    def get_exploration_rate(self):
        """Decay exploration with GA-optimized schedule"""
        # Exploration decays from 1.0 to near-0 over time
        exploration_factor = self.exploration_decay ** (self.task_count / 1000)
        return exploration_factor
    
    def check_anomaly(self, quality_metric):
        """Alert if quality drops below GA-optimized threshold"""
        if quality_metric < self.alert_threshold:
            return True  # Anomaly detected
        return False

# Apply
dashboard = OptimizedDashboard(dashboard_params)

for task in tasks:
    model = router.select_model(task)
    quality = execute_task(model, task)
    
    # Update beliefs
    dashboard.update_beliefs(model, quality)
    
    # Check for anomalies
    if dashboard.check_anomaly(quality):
        alert(f"Quality dropped below {dashboard.alert_threshold}")
    
    # Monitor exploration
    exploration = dashboard.get_exploration_rate()
```

**Expected Fitness**: 0.65-0.85 (learning speed × stability)

## Validation Checklist

Before applying GA parameters to production:

- [ ] Run `test_evaluators.py` and all tests pass
- [ ] Run `ga_tuner.py` and review `results/ga_summary_*.json`
- [ ] Best fitness scores reasonable for each system:
  - Compression: > 0.50
  - Thompson: > 0.50
  - Caching: > 0.50
  - Matrix: > 0.75 (routing accuracy)
  - Dashboard: > 0.60
- [ ] A/B test parameters on staging before production
- [ ] Monitor for 2-4 weeks to verify improvements

## Results Interpretation

### Fitness Scores

| System | Min | Good | Excellent |
|--------|-----|------|-----------|
| Compression | 0.3 | 0.5+ | 0.65+ |
| Thompson | 0.4 | 0.6+ | 0.75+ |
| Caching | 0.4 | 0.65+ | 0.80+ |
| Matrix | 0.5 | 0.80+ | 0.92+ |
| Dashboard | 0.4 | 0.70+ | 0.85+ |

### Parameter Ranges

All optimized parameters should fall within their bounds:

```yaml
compression:
  compression_level: 0.0-5.0
  target_reduction: 0.2-0.7

thompson:
  alpha_prior: 0.5-3.0
  beta_prior: 0.5-3.0
  cost_weight: 0.1-0.5

caching:
  ttl_seconds: 60-600
  cache_threshold: 0.1-0.9

matrix:
  domain_weight: 0.1-0.5
  complexity_weight: 0.2-0.6
  task_weight: 0.1-0.5

dashboard:
  learning_rate: 0.01-0.2
  exploration_decay: 0.85-0.99
  alert_threshold: 0.3-0.9
```

## Troubleshooting

**Q: Fitness scores very low (< 0.3)**
- A: Check that test data is loading correctly
- Run `test_evaluators.py` to diagnose
- Adjust parameter ranges in `ga_config.yaml`

**Q: All systems converge to same parameters**
- A: May indicate insufficient diversity in evaluation
- Increase `population_size` in config
- Increase `generations` for better convergence
- Check that evaluators are using independent random streams

**Q: One system dominates (only that system in results)**
- A: Fitness scores may not be comparable across systems
- Normalize fitness scores to [0, 1] before comparison
- Consider separate GA runs per system

## Next Steps

1. Deploy optimized parameters to staging
2. Run A/B tests for 2-4 weeks
3. Measure: cost, quality, routing accuracy
4. If improvements verified, deploy to production
5. Re-run GA tuning monthly to adapt to workload changes
6. Track parameter drift over time (record all parameter sets)

## References

- `README.md` - System overview and architecture
- `ga_tuner.py` - Main GA implementation
- `evaluators/` - Individual evaluator modules
- `test_evaluators.py` - Validation tests
