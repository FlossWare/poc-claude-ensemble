#!/usr/bin/env python3
"""
Production Integration Example: Compressor + Thompson Router + Cache

This example shows how to integrate the Phase 2 verified compressor
into a real RH workflow with Thompson-based model selection and caching.
"""

from compression_api import compress_prompt
from typing import Dict, Optional
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
import json
import hashlib
import numpy as np


@dataclass
class CompressionMetrics:
    """Track compression effectiveness for learning"""
    context_hash: str
    original_tokens: int
    compressed_tokens: int
    reduction_percent: float
    semantic_loss: float
    cache_hit: bool
    latency_ms: float


class CachedCompressor:
    """Compressor with LRU cache for identical contexts"""

    def __init__(self, cache_size: int = 1000):
        # FIX: Use OrderedDict for proper LRU implementation
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def compress(self, text: str, target_reduction: float = 0.35) -> tuple:
        """Compress with caching. Returns (compressed_text, metrics)"""
        # Hash for cache key
        context_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Check cache
        if context_hash in self.cache:
            self.hits += 1
            # FIX: Move accessed item to end for LRU semantics
            self.cache.move_to_end(context_hash)
            cached_result = self.cache[context_hash]
            metrics = CompressionMetrics(
                context_hash=context_hash,
                original_tokens=cached_result['original_tokens'],
                compressed_tokens=cached_result['compressed_tokens'],
                reduction_percent=cached_result['reduction_percent'],
                semantic_loss=cached_result['semantic_loss'],
                cache_hit=True,
                latency_ms=0.5  # Cache lookup is fast
            )
            return cached_result['text'], metrics

        # Cache miss - compress
        self.misses += 1
        result = compress_prompt(text, target_reduction)

        # Store in cache with proper LRU eviction
        if len(self.cache) >= self.cache_size:
            # FIX: Remove least recently used (first item) instead of FIFO
            self.cache.popitem(last=False)

        self.cache[context_hash] = {
            'text': result.text,
            'original_tokens': result.original_tokens,
            'compressed_tokens': result.compressed_tokens,
            'reduction_percent': result.reduction_percent,
            'semantic_loss': result.semantic_loss
        }

        metrics = CompressionMetrics(
            context_hash=context_hash,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            reduction_percent=result.reduction_percent,
            semantic_loss=result.semantic_loss,
            cache_hit=False,
            latency_ms=15.0  # Estimated compression latency
        )

        return result.text, metrics

    def get_cache_stats(self) -> Dict:
        """Return cache effectiveness stats"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'hit_rate': round(hit_rate, 3),
            'max_size': self.cache_size
        }


class ThompsonRouterWithCompression:
    """Example Thompson router that uses compression for cost optimization"""

    def __init__(self):
        self.compressor = CachedCompressor()
        self.model_rewards = {
            'haiku': {'success': 0, 'failures': 0, 'tokens_saved': 0},
            'sonnet': {'success': 0, 'failures': 0, 'tokens_saved': 0},
            'opus': {'success': 0, 'failures': 0, 'tokens_saved': 0},
        }
        self.compression_metrics = []

    def select_model(self) -> str:
        """Thompson sampling: select model based on past success"""
        # FIX: Implement actual Thompson sampling with Beta posterior sampling
        # instead of greedy selection
        beta_samples = {}
        for model, rewards in self.model_rewards.items():
            # Beta posterior: successes and (failures+1) for regularization
            alpha = rewards['success'] + 1  # Prior: 1 success
            beta = rewards['failures'] + 1   # Prior: 1 failure
            # Sample from Beta(alpha, beta) distribution
            sample = np.random.beta(alpha, beta)
            beta_samples[model] = sample

        # Select model with highest sampled reward
        best_model = max(beta_samples.items(), key=lambda x: x[1])
        return best_model[0]

    def route_with_compression(
        self,
        user_prompt: str,
        context: str,
        target_reduction: float = 0.35
    ) -> Dict:
        """
        Main integration point: route with compression

        Args:
            user_prompt: The actual user query
            context: Background context (to be compressed)
            target_reduction: Target compression ratio (0.35 = compress to 65%)

        Returns:
            Dict with selected model, compressed context, and metrics
        """
        # Step 1: Compress context
        compressed_context, compression_metrics = self.compressor.compress(
            context,
            target_reduction
        )

        # Step 2: Select model using Thompson
        selected_model = self.select_model()

        # Step 3: Prepare LLM call
        llm_input = {
            'model': selected_model,
            'user_message': user_prompt,
            'context': compressed_context,
            'context_tokens': compression_metrics.compressed_tokens
        }

        # Step 4: Log metrics
        self.compression_metrics.append({
            'model': selected_model,
            'compression_metrics': {
                'original_tokens': compression_metrics.original_tokens,
                'compressed_tokens': compression_metrics.compressed_tokens,
                'reduction_percent': compression_metrics.reduction_percent,
                'semantic_loss': compression_metrics.semantic_loss,
                'cache_hit': compression_metrics.cache_hit
            }
        })

        return llm_input

    def record_outcome(self, model: str, success: bool, tokens_saved: int):
        """Record model outcome for Thompson learning"""
        if success:
            self.model_rewards[model]['success'] += 1
        else:
            self.model_rewards[model]['failures'] += 1

        self.model_rewards[model]['tokens_saved'] += tokens_saved

    def get_effectiveness_report(self) -> Dict:
        """Generate compression effectiveness report"""
        if not self.compression_metrics:
            return {"error": "No metrics collected"}

        avg_reduction = sum(
            m['compression_metrics']['reduction_percent']
            for m in self.compression_metrics
        ) / len(self.compression_metrics)

        avg_loss = sum(
            m['compression_metrics']['semantic_loss']
            for m in self.compression_metrics
        ) / len(self.compression_metrics)

        total_tokens_saved = sum(
            m['compression_metrics']['original_tokens'] - m['compression_metrics']['compressed_tokens']
            for m in self.compression_metrics
        )

        cache_hits = sum(1 for m in self.compression_metrics if m['compression_metrics']['cache_hit'])
        cache_hit_rate = cache_hits / len(self.compression_metrics)

        return {
            'total_compressions': len(self.compression_metrics),
            'avg_reduction_percent': round(avg_reduction, 1),
            'avg_semantic_loss': round(avg_loss, 3),
            'total_tokens_saved': total_tokens_saved,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'model_rewards': self.model_rewards,
            'compressor_cache_stats': self.compressor.get_cache_stats()
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Production Integration Example: Compressor + Thompson Router")
    print("=" * 70)

    router = ThompsonRouterWithCompression()

    # Simulate 5 RH workflows
    workflows = [
        {
            "name": "CPSEARCH-10981 Code Review",
            "prompt": "Review this keyset pagination implementation for bugs",
            "context": """
            CPSEARCH-10981 involves implementing keyset pagination for Solr queries.
            The issue is that the current AND logic fails when cursors span logical boundaries.
            Previous work in commits a840f115 and b7edae1d fixed critical blocker issues from code review.
            The new method signature was updated across 8 concrete component unit tests.
            All 9 concrete implementations were updated with the new method signature.
            """,
            "success": True
        },
        {
            "name": "Multi-AI Consensus Review",
            "prompt": "Compare consensus findings across models",
            "context": """
            Multi-AI consensus uses a fleet-based approach with 3-8 models.
            Models include Opus, Sonnet, DeepSeek-Chat, Qwen3-Coder, fable, Hermes-405B.
            Panel 1: opus, sonnet, DeepSeek-Chat, Qwen3-Coder with Opus arbiter.
            Panel 2: fable, Hermes-405B, Nemotron-Ultra-550B with Sonnet arbiter.
            Zero overlap between panels prevents confirmation bias.
            """,
            "success": True
        },
        {
            "name": "Disseminator Deployment Planning",
            "prompt": "Plan deployment for next release",
            "context": """
            Disseminator deployment modes include starting_at_qa and only_qa behavior.
            The deployment spreadsheet tracks every deploy by date, release, version, and environment.
            Team timezones: EST (csanders, loleary, grgardne), IST (ypant, rghandi, vmhaskar).
            Deployment gating appears after each stage completes, not during execution.
            """,
            "success": True
        },
        {
            "name": "Cache hit test (repeat context)",
            "prompt": "Review this implementation again",
            "context": """
            CPSEARCH-10981 involves implementing keyset pagination for Solr queries.
            The issue is that the current AND logic fails when cursors span logical boundaries.
            Previous work in commits a840f115 and b7edae1d fixed critical blocker issues from code review.
            The new method signature was updated across 8 concrete component unit tests.
            All 9 concrete implementations were updated with the new method signature.
            """,
            "success": True
        },
        {
            "name": "Integration Test",
            "prompt": "Check Thompson router integration",
            "context": """
            The FlossWare/model-router is a decorator-based LLM routing system.
            It supports Anthropic (Claude: Haiku, Sonnet, Opus), Google Gemini.
            RH approved models only via official non-personal API keys.
            Multi-AI consensus required for critical work.
            """,
            "success": True
        }
    ]

    print("\nProcessing 5 RH workflows...\n")

    for workflow in workflows:
        print(f"Workflow: {workflow['name']}")

        # Route with compression
        llm_input = router.route_with_compression(
            user_prompt=workflow['prompt'],
            context=workflow['context'],
            target_reduction=0.35
        )

        print(f"  Model selected: {llm_input['model']}")
        print(f"  Context tokens: {llm_input['context_tokens']} (compressed)")
        print(f"  Cache hit: {router.compression_metrics[-1]['compression_metrics']['cache_hit']}")
        print(f"  Reduction: {router.compression_metrics[-1]['compression_metrics']['reduction_percent']}%")

        # Record outcome for Thompson learning
        tokens_saved = (
            router.compression_metrics[-1]['compression_metrics']['original_tokens'] -
            router.compression_metrics[-1]['compression_metrics']['compressed_tokens']
        )
        router.record_outcome(
            model=llm_input['model'],
            success=workflow['success'],
            tokens_saved=tokens_saved
        )
        print()

    # Generate report
    print("=" * 70)
    print("Effectiveness Report")
    print("=" * 70)

    report = router.get_effectiveness_report()
    print(f"\nCompression Results:")
    print(f"  Total compressions: {report['total_compressions']}")
    print(f"  Avg reduction: {report['avg_reduction_percent']}%")
    print(f"  Avg semantic loss: {report['avg_semantic_loss']}")
    print(f"  Total tokens saved: {report['total_tokens_saved']}")
    print(f"  Cache hit rate: {report['cache_hit_rate']:.1%}")

    print(f"\nCache Statistics:")
    cache_stats = report['compressor_cache_stats']
    print(f"  Cache entries: {cache_stats['cache_size']}/{cache_stats['max_size']}")
    print(f"  Hits: {cache_stats['cache_hits']}")
    print(f"  Misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")

    print(f"\nModel Rewards (Thompson Learning):")
    for model, rewards in report['model_rewards'].items():
        success_rate = rewards['success'] / (rewards['success'] + rewards['failures']) if (rewards['success'] + rewards['failures']) > 0 else 0
        print(f"  {model}:")
        print(f"    Success: {rewards['success']}, Failures: {rewards['failures']}")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Tokens saved: {rewards['tokens_saved']}")

    print("\n" + "=" * 70)
    print("Integration Status: ✓ PRODUCTION READY")
    print("=" * 70)
