# Compression Module - RH Cost Optimization

Token-efficient compression for Red Hat workflows targeting 20-30% reduction on paid LLM models.

## Quick Start

### Install
No installation needed - pure Python module.

```bash
python3 compression/summarizer.py  # Run tests
```

### Usage

```python
from compression.compression_api import compress_prompt

# Single prompt
result = compress_prompt(
    text="Long RH context...",
    target_reduction=0.35  # Compress to 65% of original
)

print(f"Tokens: {result.original_tokens} → {result.compressed_tokens}")
print(f"Reduction: {result.reduction_percent}%")
print(f"Semantic loss: {result.semantic_loss}")
print(f"Compressed:\n{result.text}")
```

### Batch Processing

```python
from compression.compression_api import batch_compress, measure_effectiveness

results = batch_compress(
    prompts=[prompt1, prompt2, prompt3],
    target_reduction=0.35
)

metrics = measure_effectiveness(results)
print(f"Average reduction: {metrics['avg_reduction_percent']}%")
print(f"Total tokens saved: {metrics['total_tokens_saved']}")
```

## How It Works

### 4-Level Recursive Compression

1. **Extract Key Facts** - Preserve numbers, proper nouns, code, domain terms
2. **Remove Redundancy** - Eliminate duplicate/near-duplicate sentences
3. **Consolidate Narrative** - Group and condense related statements
4. **Reduce Examples** - Keep 1-2 best examples instead of all

### Adaptive Strategy

**Short text (≤5 sentences):**
- Use all 4 compression levels
- Better for focused content

**Long text (>5 sentences):**
- Use hierarchical sentence selection
- Score by information value (metrics, entities, code)
- Keep 70% for semantic safety

## Performance

| Scenario | Reduction | Semantic Loss | Time |
|----------|-----------|---------------|------|
| Short context | 40-55% | 0.10-0.15 | <1ms |
| Medium context | 35-45% | 0.15-0.25 | 1-2ms |
| Long context | 30-40% | 0.20-0.31 | 2-5ms |

**Zero token overhead** - compression runs client-side, no API calls.

## RH Workflow Optimization

### CPSEARCH Issues
Preserves issue references, Solr configuration, test assertions

```python
# Example: CPSEARCH-10981
original = "CPSEARCH-10981 involves implementing keyset pagination for Solr..."
# → Compressed: "CPSEARCH-10981 keyset pagination Solr... (41% reduction)"
```

### Disseminator Deployments
Keeps deployment modes, team timezones, infrastructure details

### Multi-AI Consensus
Preserves model names, panel configurations, performance metrics

### API/Orchestrator Context
Maintains endpoint references, configuration flags, numeric specs

## Configuration

```python
from compression.compression_api import compress_prompt

result = compress_prompt(
    text="Long prompt...",
    target_reduction=0.35,          # 0.30-0.50 recommended
    preserve_code_refs=True,        # Always keep code
    min_semantic_threshold=0.3      # Max acceptable loss
)
```

## Metrics

All results include:
- **reduction_percent** - Tokens saved
- **semantic_loss** - Fact preservation score (0=perfect, 1=total loss)
- **key_facts_preserved** - Count of critical facts kept
- **compression_method** - Which strategy was used

## Integration Examples

### Pre-compress before API call
```python
result = compress_prompt(long_context, target_reduction=0.35)
response = anthropic.messages.create(
    model="claude-opus-5",
    messages=[{"role": "user", "content": result.text}]
)
```

### Compress conversation history
```python
history_compressed = [compress_prompt(msg['content']) for msg in conversation]
# Reduces context overhead in multi-turn workflows
```

### Batch reduce multiple queries
```python
queries = [q1, q2, q3, q4, q5]
compressed = batch_compress(queries)
# Use compressed versions in fleet dispatch
```

## Testing

```bash
# Run full test suite with 5 real RH prompts
python3 compression/summarizer.py

# Expected output:
# Tests run: 5
# Average token reduction: 41.2%
# Average semantic loss: 0.21
# Target achieved: ✓ (30-50%)
```

## Phase 1 Progress

**Status:** COMPLETE (Worker 1: Summarizer)

- ✓ Summarizer: 41.2% reduction
- ⏳ Deduplicator (Sonnet): Multi-turn conversation dedup
- ⏳ Context Windowing (Opus): Sliding-window relevance
- ⏳ Query Optimizer (Gemini): Efficient rephrasing

**Combined target:** 20-30% total reduction (when all workers complete)

## Files

- **summarizer.py** - Core recursive summarizer (360+ lines)
- **compression_api.py** - Production API and batch interface
- **worker1_summarizer_metrics.json** - Test metrics
- **WORKER1_REPORT.md** - Detailed technical report
- **PHASE1_STATUS.md** - Phase 1 progress tracking
- **README.md** - This file

## References

- [Phase 1 Status](PHASE1_STATUS.md) - Full project status
- [Worker 1 Report](WORKER1_REPORT.md) - Technical details
- [API Details](compression_api.py) - Code documentation

---

**Last Updated:** 2026-09-25
**Status:** Production-ready for Phase 2 testing
