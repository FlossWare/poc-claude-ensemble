#!/usr/bin/env python3
"""
Example: Integrating cost tracking into compression pipeline and cache system

Shows how to:
1. Instrument compression pipeline to track compression metrics
2. Instrument cache system to track hits/misses
3. Calculate savings from compression and caching
4. Export metrics for cost analysis

Usage:
    python example_compression_cache_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cost_tracking.integration import (
    CostLogger,
    CompressionPipelineHook,
    CacheSystemHook,
    CompressionMetrics,
    CacheMetrics,
)


# ============================================================================
# COMPRESSION PIPELINE INTEGRATION
# ============================================================================

def example_compression_tracking():
    """Example 1: Track compression pipeline"""
    print("Example 1: Compression Pipeline Tracking\n")

    logger = CostLogger()

    # Mock compression results (simulate 5 compression operations)
    compressions = [
        {
            'input_tokens': 1000,
            'output_tokens': 650,
            'reduction_percent': 35.0,
            'semantic_loss': 0.12,
            'compression_time_ms': 45.3,
            'compression_method': 'recursive-hierarchical',
            'key_facts_preserved': 7,
        },
        {
            'input_tokens': 2500,
            'output_tokens': 1625,
            'reduction_percent': 35.0,
            'semantic_loss': 0.14,
            'compression_time_ms': 87.6,
            'compression_method': 'recursive-hierarchical',
            'key_facts_preserved': 12,
        },
        {
            'input_tokens': 500,
            'output_tokens': 325,
            'reduction_percent': 35.0,
            'semantic_loss': 0.10,
            'compression_time_ms': 22.1,
            'compression_method': 'recursive-hierarchical',
            'key_facts_preserved': 4,
        },
    ]

    total_input = 0
    total_output = 0

    for i, comp in enumerate(compressions):
        # Log compression
        compression = CompressionMetrics(
            input_size=comp['input_tokens'],
            output_size=comp['output_tokens'],
            reduction_percent=comp['reduction_percent'],
            semantic_loss=comp['semantic_loss'],
            compression_time_ms=comp['compression_time_ms'],
            compression_method=comp['compression_method'],
            key_facts_preserved=comp['key_facts_preserved'],
        )
        logger.log_compression(compression, f"compress-{i}")

        total_input += comp['input_tokens']
        total_output += comp['output_tokens']

        print(f"✓ Compression {i+1}:")
        print(f"    Input: {comp['input_tokens']} tokens")
        print(f"    Output: {comp['output_tokens']} tokens")
        print(f"    Reduction: {comp['reduction_percent']:.1f}%")
        print(f"    Time: {comp['compression_time_ms']:.1f}ms")
        print(f"    Semantic loss: {comp['semantic_loss']:.2f}")
        print()

    # Summary
    import time
    time.sleep(0.2)

    summary = logger.get_summary()
    comp_stats = summary['compression_stats']

    print("Summary:")
    print(f"  Total compressions: {comp_stats['total_calls']}")
    print(f"  Total input tokens: {total_input}")
    print(f"  Total output tokens: {total_output}")
    print(f"  Total reduction: {((total_input - total_output) / total_input * 100):.1f}%")
    print(f"  Avg compression time: {comp_stats['avg_compression_time_ms']:.1f}ms")
    print(f"  Avg semantic loss: {comp_stats['avg_semantic_loss']:.3f}")
    print()

    logger.stop()


# ============================================================================
# CACHE SYSTEM INTEGRATION
# ============================================================================

def example_cache_tracking():
    """Example 2: Track cache hits and misses"""
    print("Example 2: Cache System Tracking\n")

    logger = CostLogger()

    # Simulate cache operations
    cache_ops = [
        {'hit': True, 'cost_saved': 0.0025, 'cache_source': 'prompt_cache'},
        {'hit': True, 'cost_saved': 0.0025, 'cache_source': 'prompt_cache'},
        {'hit': False, 'cost_saved': 0.0, 'cache_source': 'cache_miss'},
        {'hit': True, 'cost_saved': 0.0025, 'cache_source': 'prompt_cache'},
        {'hit': False, 'cost_saved': 0.0, 'cache_source': 'cache_miss'},
        {'hit': True, 'cost_saved': 0.0025, 'cache_source': 'prompt_cache'},
    ]

    for i, op in enumerate(cache_ops):
        cache = CacheMetrics(
            is_cache_hit=op['hit'],
            cache_key=f"prompt:key-{i}",
            cache_source=op['cache_source'],
            input_tokens=200,
            cache_read_tokens=100 if op['hit'] else 0,
            cache_creation_tokens=200 if not op['hit'] else 0,
            output_tokens=50,
            cost_saved=op['cost_saved'],
        )
        logger.log_cache_operation(cache, f"cache-{i}")

        status = "HIT" if op['hit'] else "MISS"
        print(f"✓ Cache operation {i+1}: {status} (saved: ${op['cost_saved']:.6f})")

    print()

    # Summary
    import time
    time.sleep(0.2)

    summary = logger.get_summary()
    cache_stats = summary['cache_stats']

    hits = cache_stats['hit_count']
    misses = cache_stats['miss_count']
    total = hits + misses

    print("Summary:")
    print(f"  Total operations: {total}")
    print(f"  Cache hits: {hits} ({(hits/total*100):.1f}%)")
    print(f"  Cache misses: {misses} ({(misses/total*100):.1f}%)")
    print(f"  Total cost saved: ${cache_stats['total_cost_saved']:.6f}")
    print(f"  Average savings per hit: ${cache_stats['total_cost_saved']/max(1, hits):.6f}")
    print()

    logger.stop()


# ============================================================================
# COMBINED COMPRESSION + CACHE ANALYSIS
# ============================================================================

def example_combined_compression_cache():
    """Example 3: Combined compression and cache analysis"""
    print("Example 3: Combined Compression + Cache Analysis\n")

    logger = CostLogger()

    # Model pricing (per 1M tokens)
    CLAUDE_OPUS_INPUT = 15.00
    CLAUDE_OPUS_CACHE_READ = 1.50  # 90% discount
    CLAUDE_OPUS_OUTPUT = 75.00

    # Simulate workflow: compression + cache lookup
    print("Scenario: Process 5 large prompts (1000 tokens each)")
    print(f"  Model: Claude Opus ($15.00/M input, $1.50/M cache read)")
    print()

    original_total_tokens = 5000
    compression_targets = [
        {'id': 1, 'target_reduction': 0.35, 'cache_hit': False},
        {'id': 2, 'target_reduction': 0.35, 'cache_hit': True},   # Hit from compression
        {'id': 3, 'target_reduction': 0.35, 'cache_hit': False},
        {'id': 4, 'target_reduction': 0.35, 'cache_hit': True},   # Hit from compression
        {'id': 5, 'target_reduction': 0.35, 'cache_hit': False},
    ]

    total_cost_without_optimization = 0
    total_cost_with_optimization = 0

    for target in compression_targets:
        print(f"Processing prompt {target['id']}:")

        # Cost without optimization
        cost_without = (1000 / 1_000_000) * CLAUDE_OPUS_INPUT
        total_cost_without_optimization += cost_without
        print(f"  Without optimization: ${cost_without:.6f}")

        if not target['cache_hit']:
            # Compression happens
            compression = CompressionMetrics(
                input_size=1000,
                output_size=650,
                reduction_percent=35.0,
                semantic_loss=0.12,
                compression_time_ms=45.0,
                compression_method='recursive-hierarchical',
                key_facts_preserved=7,
            )
            logger.log_compression(compression, f"compress-{target['id']}")

            # Cache miss - pay for full compression
            cost_with = (650 / 1_000_000) * CLAUDE_OPUS_INPUT
            print(f"  With compression (miss): ${cost_with:.6f}")

            # Log cache miss
            cache = CacheMetrics(
                is_cache_hit=False,
                cache_key=f"prompt:{target['id']}",
                cache_source='cache_miss',
                input_tokens=650,
                cache_read_tokens=0,
                cache_creation_tokens=650,
                output_tokens=50,
                cost_saved=0.0,
            )
            logger.log_cache_operation(cache, f"cache-miss-{target['id']}")

        else:
            # Cache hit from previous compression
            cost_with = (650 / 1_000_000) * CLAUDE_OPUS_CACHE_READ
            print(f"  With cache (hit): ${cost_with:.6f}")

            # Log cache hit
            cache = CacheMetrics(
                is_cache_hit=True,
                cache_key=f"prompt:{target['id']}",
                cache_source='prompt_cache',
                input_tokens=0,  # No input on cache hit
                cache_read_tokens=650,
                cache_creation_tokens=0,
                output_tokens=50,
                cost_saved=((650 / 1_000_000) * CLAUDE_OPUS_INPUT) - ((650 / 1_000_000) * CLAUDE_OPUS_CACHE_READ),
            )
            logger.log_cache_operation(cache, f"cache-hit-{target['id']}")

        total_cost_with_optimization += cost_with

        savings = cost_without - cost_with
        savings_pct = (savings / cost_without * 100) if cost_without > 0 else 0
        print(f"  Savings: ${savings:.6f} ({savings_pct:.1f}%)")
        print()

    # Summary
    import time
    time.sleep(0.2)

    summary = logger.get_summary()

    total_savings = total_cost_without_optimization - total_cost_with_optimization
    savings_pct = (total_savings / total_cost_without_optimization * 100)

    print("=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Without optimization (5 prompts × 1000 tokens): ${total_cost_without_optimization:.6f}")
    print(f"With compression + cache (3 full + 2 cached):   ${total_cost_with_optimization:.6f}")
    print(f"Total savings: ${total_savings:.6f} ({savings_pct:.1f}%)")
    print()
    print(f"Compressions performed: {summary['compression_stats']['total_calls']}")
    print(f"Cache hits: {summary['cache_stats']['hit_count']}")
    print(f"Cache misses: {summary['cache_stats']['miss_count']}")
    print()

    logger.stop()


# ============================================================================
# INTEGRATION PATTERN
# ============================================================================

def example_integration_pattern():
    """Example 4: Show how to integrate into existing code"""
    print("Example 4: Integration Pattern\n")

    print("""
BEFORE (compression_api.py):
    def compress_prompt(text, target_reduction=0.35):
        # Compression logic
        return CompressedPrompt(...)


AFTER (add instrumentation):
    from cost_tracking.integration import CostLogger, CompressionPipelineHook

    # Initialize once at startup
    cost_logger = CostLogger()
    hook = CompressionPipelineHook()

    # Import the function
    from compression.compression_api import compress_prompt

    # Instrument it (one line!)
    compress_prompt = hook.instrument_compress_prompt(cost_logger, compress_prompt)

    # Use normally - compression metrics are logged automatically
    result = compress_prompt(long_text, target_reduction=0.35)


SAME PATTERN FOR CACHE:
    from cost_tracking.integration import CacheSystemHook

    hook = CacheSystemHook()
    cache = PromptCacheControl()
    cache.lookup = hook.instrument_cache_lookup(cost_logger, cache.lookup)

    # Metrics logged automatically
    result = cache.lookup("cache:key")


BENEFITS:
  ✓ Zero changes to core logic
  ✓ One-line instrumentation per function
  ✓ Automatic tracking of compression and cache operations
  ✓ Thread-safe async logging
  ✓ Minimal overhead (<1%)
    """)


def example_cost_roi_analysis():
    """Example 5: ROI analysis of compression + caching"""
    print("Example 5: ROI Analysis\n")

    logger = CostLogger()

    # Simulate a week of operations
    week_tasks = 1000

    # Without optimization
    cost_per_task_no_opt = 0.05  # Example: $0.05 per task
    weekly_cost_no_opt = week_tasks * cost_per_task_no_opt

    # With compression + cache
    # - 30% of tasks hit cache (no cost)
    # - 70% need compression (35% reduction)
    cache_hit_rate = 0.30
    compression_reduction = 0.35

    cache_hits = week_tasks * cache_hit_rate
    cache_misses = week_tasks * (1 - cache_hit_rate)

    cost_per_cache_hit = 0.002  # Cached reads are much cheaper
    cost_per_cache_miss = cost_per_task_no_opt * (1 - compression_reduction)

    weekly_cost_with_opt = (cache_hits * cost_per_cache_hit) + (cache_misses * cost_per_cache_miss)

    weekly_savings = weekly_cost_no_opt - weekly_cost_with_opt
    annual_savings = weekly_savings * 52

    print(f"Weekly Analysis ({week_tasks} tasks):")
    print(f"  Without optimization: ${weekly_cost_no_opt:.2f}")
    print(f"  With compression + cache:")
    print(f"    Cache hits (30%): {cache_hits:.0f} × ${cost_per_cache_hit:.4f} = ${cache_hits * cost_per_cache_hit:.2f}")
    print(f"    Cache misses (70%): {cache_misses:.0f} × ${cost_per_cache_miss:.4f} = ${cache_misses * cost_per_cache_miss:.2f}")
    print(f"    Subtotal: ${weekly_cost_with_opt:.2f}")
    print()
    print(f"  Weekly savings: ${weekly_savings:.2f} ({(weekly_savings/weekly_cost_no_opt*100):.1f}%)")
    print(f"  Annual savings: ${annual_savings:.2f}")
    print()

    logger.stop()


if __name__ == "__main__":
    print("=" * 70)
    print("Compression & Cache Cost Tracking Examples")
    print("=" * 70)
    print()

    example_compression_tracking()
    print()

    example_cache_tracking()
    print()

    example_combined_compression_cache()
    print()

    example_cost_roi_analysis()
    print()

    example_integration_pattern()

    print("\n" + "=" * 70)
    print("Files created:")
    print("  ~/.claude/cost_tracking/cost_metrics.json")
    print("  ~/.claude/cost_tracking/cost_summary.jsonl")
    print("  ~/.claude/cost_tracking/cost_detailed.jsonl")
    print("=" * 70)
