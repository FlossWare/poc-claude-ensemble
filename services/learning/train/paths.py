"""Environment-aware paths for training artifacts and source ledgers.

Tests (``services/conftest.py``) already redirect the ledger variables; the
artifact directory has its own override so a test run never writes models into
the operator's ``~/.FlossWare``.
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_DIR_ENV = "POC_CLAUDE_ENSEMBLE_TRAIN_DIR"
REVIEW_LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"
PROMPT_LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_PROMPT_LEDGER"
RAG_JOURNAL_ENV = "FLOSSWARE_CE_RAG_JOURNAL"


def _home(*parts: str) -> Path:
    return Path(os.path.expanduser("~")).joinpath(*parts)


def train_dir() -> Path:
    override = os.environ.get(ARTIFACT_DIR_ENV)
    if override:
        return Path(os.path.expanduser(override))
    return _home(".FlossWare", "learning", "train")


def artifact_path(name: str) -> Path:
    return train_dir() / f"{name}.joblib"


def review_ledger_path() -> Path:
    value = os.environ.get(REVIEW_LEDGER_ENV) or "~/.FlossWare/ledger/reviews/reviews.jsonl"
    return Path(os.path.expanduser(value))


def prompt_ledger_path() -> Path:
    value = os.environ.get(PROMPT_LEDGER_ENV) or "~/.FlossWare/ledger/prompts/prompts.jsonl"
    return Path(os.path.expanduser(value))


def rag_journal_path() -> Path | None:
    value = os.environ.get(RAG_JOURNAL_ENV)
    if value and value.strip().lower() == "off":
        return None
    if not value:
        value = "~/.FlossWare/ledger/rag/injections.jsonl"
    return Path(os.path.expanduser(value))


def outcome_ledger_path() -> Path:
    value = os.environ.get("POC_CLAUDE_ENSEMBLE_OUTCOME_LEDGER")
    if not value:
        value = "~/.FlossWare/ledger/outcomes/outcomes.jsonl"
    return Path(os.path.expanduser(value))
