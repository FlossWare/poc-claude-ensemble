"""Where external-source caches and the prior registry live.

Everything is under one directory so an operator can inspect (or delete) all
external learning state in one place::

    ~/.FlossWare/learning/external/
        elo.json            LMSYS Arena ELO snapshot (cached)
        benchmarks.json     Open LLM Leaderboard results (cached per model)
        pricing.json        OpenRouter price snapshot (cached)
        priors.json         external priors registered on Thompson arms

Override the root with ``POC_CLAUDE_ENSEMBLE_EXTERNAL_DIR``.  Tests and the
services conftest redirect it so no test ever writes the operator's cache.
"""

from __future__ import annotations

import os
from pathlib import Path

EXTERNAL_DIR_ENV = "POC_CLAUDE_ENSEMBLE_EXTERNAL_DIR"


def external_dir() -> Path:
    configured = os.environ.get(EXTERNAL_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".FlossWare" / "learning" / "external"


def cache_path(name: str) -> Path:
    return external_dir() / name


def elo_cache_path() -> Path:
    return cache_path("elo.json")


def benchmarks_cache_path() -> Path:
    return cache_path("benchmarks.json")


def pricing_cache_path() -> Path:
    return cache_path("pricing.json")


def priors_path() -> Path:
    return cache_path("priors.json")


def ensure_dir(path: Path | None = None) -> Path:
    """Create the external directory (0700) and return it."""
    target = Path(path).expanduser() if path is not None else external_dir()
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except OSError:
        pass
    return target
