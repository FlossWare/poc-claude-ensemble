# Prompt Caching Integration Guide

This guide shows how to integrate the prompt caching modules into RH workflows.

## Quick Start

### 1. Import and Initialize

```python
from caching.memory_cache_integration import MemoryCacheIntegrator
from caching.cache_metrics import CacheMetricsTracker

# Initialize for session
integrator = MemoryCacheIntegrator()
tracker = CacheMetricsTracker(workflow_name="my_workflow")
```

### 2. Session Start: Auto-detect Memory

```python
# At beginning of workflow session
memory_files = integrator.detect_session_memory_files()
print(f"Detected {len(memory_files)} memory files for caching")
# Output: Detected 4 memory files for caching
```

### 3. Build Cached Prompts

```python
# Before calling Anthropic API
structured_prompt = integrator.structure_prompt_with_caching(
    user_message="Please review this code",
    system_context="You are an expert code reviewer",
    memory_files=memory_files,
    cache_type="ephemeral"
)

# structured_prompt now has cache_control markers on memory blocks
```

### 4. Track Token Usage

```python
# Log request WITHOUT cache (baseline)
tracker.log_baseline_request(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
    model="claude-3-5-sonnet"
)

# Log request WITH cache
cache_info = integrator.extract_cache_usage_from_response(response)
tracker.log_cached_request(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
    cache_creation_tokens=cache_info.get("cache_creation_tokens", 0),
    cache_read_tokens=cache_info.get("cache_read_tokens", 0),
    cache_hit=cache_info.get("cache_hit", False)
)
```

### 5. Generate Reports

```python
# At end of workflow session
report = tracker.generate_report()

# Save to files
json_path = tracker.save_report_json(report)
md_path = tracker.save_report_markdown(report)
csv_path = tracker.export_to_csv()

# Print summary
tracker.print_summary()
```

## Workflow Integration Patterns

### Pattern 1: Multi-turn Conversation

Use `ephemeral` cache type for iterative workflows:

```python
integrator = MemoryCacheIntegrator(cache_control_version="ephemeral")

# Turn 1: Ask question
response1 = api.messages.create(**integrator.structure_prompt_with_caching(
    user_message="Explain the CPSEARCH-10981 fix",
    cache_type="ephemeral"
))

# Turn 2: Follow-up (reuses cached memory from Turn 1)
response2 = api.messages.create(**integrator.structure_prompt_with_caching(
    user_message="Now list the test cases affected",
    cache_type="ephemeral"
))
# Cache HIT: memory blocks cached in Turn 1 are reused
```

### Pattern 2: Batch Processing

Use `last_message` cache type for one-off batch operations:

```python
for item in batch_items:
    # Each call creates its own cache
    response = api.messages.create(**integrator.structure_prompt_with_caching(
        user_message=f"Process: {item}",
        cache_type="last_message"
    ))
    
    cache_info = integrator.extract_cache_usage_from_response(response)
    tracker.log_cached_request(...)
```

### Pattern 3: Team Workflows with Shared Memory

Track memory file changes to invalidate caches:

```python
# Store initial state
initial_files = integrator.detect_session_memory_files()

# ... do work ...

# Check if memory changed
current_files = integrator.detect_session_memory_files()

if len(current_files) != len(initial_files):
    # Memory directory changed, clear cache key cache
    integrator.clear_cache_key_cache()
    
    # Subsequent prompts will recompute cache keys
    memory_files = integrator.detect_session_memory_files()
```

## Hook Integration

For Claude Code workflows, register integration hooks:

```python
from caching.memory_cache_integration import (
    create_session_start_hook,
    create_prompt_processing_hook,
    create_response_handler_hook
)

hooks = [
    create_session_start_hook(),
    create_prompt_processing_hook(),
    create_response_handler_hook()
]

# Register with workflow harness (implementation-specific)
for hook in hooks:
    workflow.register_hook(hook)
```

## Common Use Cases

### Code Review with Consensus

```python
tracker = CacheMetricsTracker("code_review_consensus")
integrator = MemoryCacheIntegrator()

# Load all RH memory (feedback, patterns, decisions)
memory_files = integrator.detect_session_memory_files()

# First model review
review1 = api.messages.create(**integrator.structure_prompt_with_caching(
    user_message="Review for correctness",
    memory_files=memory_files,
    cache_type="ephemeral"
))
tracker.log_cached_request(...)

# Second model review (reuses cache)
review2 = api.messages.create(**integrator.structure_prompt_with_caching(
    user_message="Review for security",
    memory_files=memory_files,
    cache_type="ephemeral"
))
tracker.log_cached_request(...)

# Reports show cache hit on second review
report = tracker.generate_report()
report.print_summary()
# Cache Hit Rate: 100.0%
```

### Release Note Generation

```python
tracker = CacheMetricsTracker("release_notes")
integrator = MemoryCacheIntegrator()

# Load context memory once
memory_files = integrator.detect_session_memory_files()

# Process multiple releases
for release in releases:
    response = api.messages.create(**integrator.structure_prompt_with_caching(
        user_message=f"Generate notes for {release.version}",
        memory_files=memory_files,
        cache_type="last_message"
    ))
    tracker.log_cached_request(...)

# Summary shows token savings
report = tracker.generate_report()
total_savings = report.stats.total_baseline_cost - report.stats.total_cached_cost
print(f"Cost savings: ${total_savings:.2f}")
```

## Debugging

### Log Memory File Detection

```python
import logging
logging.basicConfig(level=logging.DEBUG)

integrator = MemoryCacheIntegrator()
files = integrator.detect_session_memory_files()

# Debug output shows each detected file:
# DEBUG - Detected memory file: /path/to/feedback_always_review.md
# DEBUG - Detected memory file: /path/to/reference_memory_file.md
```

### Inspect Cache Keys

```python
integrator = MemoryCacheIntegrator()

# Get cache key for a specific file
file_path = "/path/to/feedback_always_review.md"
cache_key = integrator._compute_cache_key(file_path)

print(f"File: {cache_key.file_path}")
print(f"  Size: {cache_key.file_size} bytes")
print(f"  Modified: {cache_key.modification_time}")
print(f"  Content hash: {cache_key.content_hash}")
print(f"  Cache key hash: {cache_key.to_hash()}")
```

### View Detailed Metrics

```python
# After collecting requests
tracker = CacheMetricsTracker("my_workflow")
# ... log requests ...

# Get current stats without finalizing
stats = tracker.get_current_stats()
print(f"Requests: {stats['total_requests']}")
print(f"Cache hits: {stats['cache_hit_requests']}")
print(f"Hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Tokens saved: {stats['total_tokens_saved']:,}")
print(f"Cost saved: ${stats['total_baseline_cost'] - stats['total_cached_cost']:.4f}")
```

## Performance Considerations

### Memory Files > 100KB

Files larger than 100KB are skipped by default:

```python
# Increase limit if needed
integrator = MemoryCacheIntegrator(max_cached_file_size=512000)  # 500KB

# Or check file sizes
for file_path in integrator.detect_session_memory_files():
    size_kb = os.path.getsize(file_path) / 1024
    print(f"{os.path.basename(file_path)}: {size_kb:.1f}KB")
```

### Cache Key Computation

Cache keys are computed once and cached in memory:

```python
# First call computes hash
key1 = integrator._compute_cache_key(file_path)  # ~1-5ms

# Second call is instant (cached)
key2 = integrator._compute_cache_key(file_path)  # <1ms

# Clear cache if memory files change
integrator.clear_cache_key_cache()
```

### Aggregate Metrics

When collecting metrics from many workflows:

```python
from caching.cache_metrics import AggregateMetricsCollector

collector = AggregateMetricsCollector()

# Add reports from multiple workflows
for workflow_name, tracker in workflow_trackers.items():
    report = tracker.generate_report()
    collector.add_report(report)

# Generate team summary
summary = collector.generate_aggregate_summary()
collector.save_aggregate_summary()

# Shows total savings across all workflows
print(f"Team cache hit rate: {summary['cache_hit_rate']:.1f}%")
print(f"Total cost savings: ${summary['total_savings']:.2f}")
```

## Error Handling

### Missing Memory Directory

```python
# Gracefully handles missing directory
integrator = MemoryCacheIntegrator(memory_dir="/nonexistent")
files = integrator.detect_session_memory_files()
# Returns empty list, logs debug message
```

### File Read Errors

```python
# Continue on permission errors
integrator = MemoryCacheIntegrator()
files = integrator.detect_session_memory_files()
# Skips unreadable files, logs warning
```

### Invalid Cache Types

```python
# Falls back to ephemeral on invalid type
structured = integrator.structure_prompt_with_caching(
    user_message="...",
    cache_type="invalid_type"
)
# Defaults to "ephemeral", logs warning
```

## Next Steps

Phase 2 will include:
- Comprehensive unit and integration tests
- Example workflows for common RH use cases
- Performance benchmarks
- Cache policy recommendations per workflow type

See [README.md](README.md) for complete API documentation.
