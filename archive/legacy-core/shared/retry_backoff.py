#!/usr/bin/env python3
"""
Exponential backoff with jitter for API resilience.
Handles transient failures across model routers with adaptive delays.
"""

import time
import random
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')

class ExponentialBackoff:
    """Exponential backoff strategy with jitter."""

    def __init__(self, base_delay=1.0, max_delay=60.0, jitter=True, max_retries=5):
        """
        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            jitter: Add randomness to avoid thundering herd
            max_retries: Maximum number of retry attempts
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.max_retries = max_retries

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt N (0-indexed)."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with backoff retries."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.get_delay(attempt)
                    time.sleep(delay)

        raise last_error or RuntimeError("Backoff exhausted")

    def decorator(self):
        """Return decorator for automatic retry."""
        def wrapper(func):
            @wraps(func)
            def decorated(*args, **kwargs):
                return self.execute(func, *args, **kwargs)
            return decorated
        return wrapper


def retry_on_error(base_delay=1.0, max_retries=5):
    """Decorator for retry with exponential backoff."""
    backoff = ExponentialBackoff(base_delay=base_delay, max_retries=max_retries)
    return backoff.decorator()


class AdaptiveBackoff:
    """Learns from failure patterns to adjust backoff strategy."""

    def __init__(self, base_delay=1.0, max_retries=5):
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.failure_count = {}
        self.total_failures = 0

    def record_failure(self, model: str, error: str):
        """Track failure by model and error type."""
        key = f"{model}:{error[:50]}"
        self.failure_count[key] = self.failure_count.get(key, 0) + 1
        self.total_failures += 1

    def get_strategy(self, model: str) -> ExponentialBackoff:
        """Select backoff strategy based on failure history."""
        # Count failures for this model
        failures = sum(count for k, count in self.failure_count.items() if k.startswith(f"{model}:"))
        failure_rate = failures / max(self.total_failures, 1)

        # Aggressive backoff for high-failure models
        if failure_rate > 0.5:
            return ExponentialBackoff(base_delay=self.base_delay * 2, max_delay=120, max_retries=self.max_retries + 2)
        elif failure_rate > 0.2:
            return ExponentialBackoff(base_delay=self.base_delay, max_delay=60, max_retries=self.max_retries)
        else:
            return ExponentialBackoff(base_delay=self.base_delay * 0.5, max_delay=30, max_retries=self.max_retries - 1)


if __name__ == '__main__':
    # Demo
    def flaky_api():
        """Simulate API that fails sometimes."""
        if random.random() < 0.7:
            raise ConnectionError("Temporary network failure")
        return "success"

    backoff = ExponentialBackoff(base_delay=0.1, max_delay=5, max_retries=5)
    result = backoff.execute(flaky_api)
    print(f"✅ {result}")
