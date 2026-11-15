#!/usr/bin/env python3
"""
Caching GA Evaluator

Evaluates caching parameters:
- ttl_seconds (60-600): Time-to-live for cache entries
- cache_threshold (0.1-0.9): Minimum token count to cache

Fitness = cache_hit_rate % * token_savings
Constraint: false_negative_rate < 0.05 (invalid cache reliability)

Simulates multi-turn conversations from RH usage patterns.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a multi-turn conversation"""
    turn_id: int
    messages: List[str]
    total_tokens: int
    cacheable_tokens: int
    changes: bool  # True if content changed from previous turn


class CachingEvaluator:
    """Evaluate caching parameters"""

    def __init__(self):
        self.memory_patterns = self._generate_rh_memory_patterns()

    def _generate_rh_memory_patterns(self) -> List[Dict]:
        """Generate realistic RH conversation patterns"""
        patterns = [
            {
                "name": "memory_queries",
                "avg_length": 300,
                "change_rate": 0.1,
                "frequency": 0.3,
            },
            {
                "name": "code_review",
                "avg_length": 1000,
                "change_rate": 0.3,
                "frequency": 0.25,
            },
            {
                "name": "deployment",
                "avg_length": 500,
                "change_rate": 0.05,
                "frequency": 0.15,
            },
            {
                "name": "documentation",
                "avg_length": 800,
                "change_rate": 0.2,
                "frequency": 0.2,
            },
            {
                "name": "debugging",
                "avg_length": 600,
                "change_rate": 0.4,
                "frequency": 0.1,
            },
        ]
        return patterns

    def _generate_synthetic_conversation(self, n_turns: int = 20) -> List[ConversationTurn]:
        """Generate synthetic multi-turn conversation with repeating content"""
        turns = []

        # Create stable content blocks (memory, docs, etc.)
        content_blocks = {
            pattern['name']: f"Content {pattern['name']}\n" + ("Data " * 50)
            for pattern in self.memory_patterns
        }

        for turn_id in range(n_turns):
            # Select pattern
            pattern = np.random.choice(
                self.memory_patterns,
                p=[p['frequency'] for p in self.memory_patterns]
            )

            # Generate content with realistic token distribution
            base_tokens = pattern['avg_length'] + np.random.normal(0, pattern['avg_length'] * 0.2)
            total_tokens = max(100, int(base_tokens))

            # Cacheable portion (e.g., memory, context)
            cacheable_pct = np.random.uniform(0.5, 0.9)
            cacheable_tokens = int(total_tokens * cacheable_pct)

            # Simulate content change (30% chance content changes)
            if random.random() < pattern['change_rate']:
                # Content changed - different message
                message = f"Turn {turn_id}: {pattern['name']} [CHANGED]\n" + content_blocks[pattern['name']]
            else:
                # Same content as some previous turn (cache opportunity)
                message = f"{pattern['name']} [STABLE]\n" + content_blocks[pattern['name']]

            turn = ConversationTurn(
                turn_id=turn_id,
                messages=[message],
                total_tokens=total_tokens,
                cacheable_tokens=cacheable_tokens,
                changes=random.random() < pattern['change_rate'],
            )

            turns.append(turn)

        return turns

    def _simulate_caching(
        self,
        ttl_seconds: float,
        cache_threshold: float,
        conversation: List[ConversationTurn],
    ) -> Tuple[float, float, float, float]:
        """
        Simulate caching performance.

        Returns: (hit_rate, token_savings, false_negative_rate, false_positive_rate)
        """
        cache = {}  # {content_hash: (timestamp, token_count, content_hash)}
        current_time = 0
        cache_timestamp_step = 5  # seconds per turn

        hits = 0
        misses = 0
        token_savings = 0
        false_negatives = 0
        false_positives = 0
        total_cacheable = 0

        for turn in conversation:
            # Advance time
            current_time += cache_timestamp_step

            # Only cache if above threshold (as percentage of total tokens)
            cacheable_ratio = turn.cacheable_tokens / turn.total_tokens if turn.total_tokens > 0 else 0
            if cacheable_ratio < cache_threshold:
                misses += 1
                total_cacheable += turn.cacheable_tokens
                continue

            # Create content hash
            content_hash = hashlib.md5(
                ''.join(turn.messages).encode()
            ).hexdigest()

            # Check cache
            cache_hit = content_hash in cache

            if cache_hit:
                cached_time, cached_tokens, _ = cache[content_hash]

                # Check TTL expiration
                expired = (current_time - cached_time) > ttl_seconds

                if expired:
                    # Cache expired - false negative
                    false_negatives += 1
                    misses += 1
                else:
                    # Cache hit!
                    hits += 1
                    token_savings += cached_tokens

            else:
                # Cache miss
                misses += 1

                # Store in cache
                cache[content_hash] = (current_time, turn.cacheable_tokens, turn.messages)

            total_cacheable += turn.cacheable_tokens

            # Remove expired entries
            expired_keys = [
                k for k, (t, _, _) in cache.items()
                if (current_time - t) > ttl_seconds
            ]
            for k in expired_keys:
                del cache[k]

        # Calculate metrics
        total_accesses = hits + misses
        hit_rate = hits / total_accesses if total_accesses > 0 else 0

        token_savings_rate = token_savings / total_cacheable if total_cacheable > 0 else 0

        false_negative_rate = false_negatives / total_accesses if total_accesses > 0 else 0

        logger.debug(
            f"Caching: ttl={ttl_seconds:.0f}s, threshold={cache_threshold:.2f} -> "
            f"hit_rate={hit_rate:.2%}, savings={token_savings_rate:.2%}, "
            f"false_neg={false_negative_rate:.2%}"
        )

        return hit_rate, token_savings_rate, false_negative_rate, 0.0

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate caching parameters.

        Fitness = hit_rate * token_savings
        Penalize if false_negative_rate > 0.05
        """
        ttl_seconds = parameters['ttl_seconds']
        cache_threshold = parameters['cache_threshold']

        # Simulate multiple conversations for stability
        n_conversations = 5
        total_fitness = 0.0

        for _ in range(n_conversations):
            conversation = self._generate_synthetic_conversation(n_turns=30)

            hit_rate, token_savings, false_neg_rate, _ = self._simulate_caching(
                ttl_seconds, cache_threshold, conversation
            )

            # Fitness components
            # 1. Hit rate contributes to fitness
            hit_component = hit_rate

            # 2. Token savings (normalized to 0-1)
            savings_component = min(1.0, token_savings * 2.0)

            # 3. False negative rate penalty
            if false_neg_rate > 0.05:
                penalty = (false_neg_rate - 0.05) * 10.0
            else:
                penalty = 0

            # Combined fitness
            fitness = (hit_component * 0.5 + savings_component * 0.5) * (1.0 - penalty)

            total_fitness += max(0, fitness)

        avg_fitness = total_fitness / n_conversations

        return np.clip(avg_fitness, 0.0, 1.0)


if __name__ == '__main__':
    # Test evaluator
    evaluator = CachingEvaluator()

    test_params = {
        'ttl_seconds': 300.0,
        'cache_threshold': 0.3,
    }

    fitness = evaluator.evaluate(test_params)
    print(f"Test fitness: {fitness:.6f}")
