"""Single source of truth for GA parameter bounds and runtime environment names.

These values describe candidate parameters, not proof that a production consumer exists.
Only explicitly verified runtime adapters may apply them.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = 1

PARAMETER_SCHEMA = {
    "compression": {
        "compression_level": {"env_var": "GA_COMPRESSION_LEVEL", "lower": 0.0, "upper": 5.0, "default": 3.0},
        "target_reduction": {"env_var": "GA_COMPRESSION_TARGET", "lower": 0.2, "upper": 0.7, "default": 0.35},
    },
    "thompson": {
        "alpha_prior": {"env_var": "GA_THOMPSON_ALPHA", "lower": 0.5, "upper": 3.0, "default": 2.0},
        "beta_prior": {"env_var": "GA_THOMPSON_BETA", "lower": 0.5, "upper": 3.0, "default": 1.0},
        "cost_weight": {"env_var": "GA_THOMPSON_COST_WEIGHT", "lower": 0.1, "upper": 0.5, "default": 0.25},
    },
    "caching": {
        "ttl_seconds": {"env_var": "GA_TUNING_CACHE_TTL", "lower": 60.0, "upper": 600.0, "default": 246.87},
        "cache_threshold": {"env_var": "GA_TUNING_CACHE_THRESHOLD", "lower": 0.1, "upper": 0.9, "default": 0.44},
    },
    "matrix": {
        "domain_weight": {"env_var": "GA_MATRIX_DOMAIN_WEIGHT", "lower": 0.1, "upper": 0.5, "default": 0.3},
        "complexity_weight": {"env_var": "GA_MATRIX_COMPLEXITY_WEIGHT", "lower": 0.2, "upper": 0.6, "default": 0.4},
        "task_weight": {"env_var": "GA_MATRIX_TASK_WEIGHT", "lower": 0.1, "upper": 0.5, "default": 0.3},
    },
    "dashboard": {
        "learning_rate": {"env_var": "GA_DASHBOARD_LEARNING_RATE", "lower": 0.01, "upper": 0.2, "default": 0.05},
        "exploration_decay": {"env_var": "GA_DASHBOARD_EXPLORATION_DECAY", "lower": 0.85, "upper": 0.99, "default": 0.95},
        "alert_threshold": {"env_var": "GA_DASHBOARD_ALERT_THRESHOLD", "lower": 0.3, "upper": 0.9, "default": 0.7},
    },
}


def validate_parameter(system: str, name: str, value: Any) -> float:
    """Return a finite in-range numeric candidate value or reject it."""
    import math

    if system not in PARAMETER_SCHEMA or name not in PARAMETER_SCHEMA[system]:
        raise ValueError(f"Unsupported GA parameter: {system}.{name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GA parameter {system}.{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"GA parameter {system}.{name} must be finite")
    spec = PARAMETER_SCHEMA[system][name]
    if not spec["lower"] <= number <= spec["upper"]:
        raise ValueError(
            f"GA parameter {system}.{name}={number} outside "
            f"[{spec['lower']}, {spec['upper']}]"
        )
    return number


def bounds_for_system(system: str) -> dict[str, tuple[float, float]]:
    """Return optimizer bounds derived from the canonical schema."""
    return {
        name: (spec["lower"], spec["upper"])
        for name, spec in PARAMETER_SCHEMA.get(system, {}).items()
    }
