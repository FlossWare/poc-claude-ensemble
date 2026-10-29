"""Shared test setup for everything under services/.

Ledgers and state are redirected to temporary directories before any service module is imported,
so tests never write to the live ledger or to production state. Module-level paths are read at
import time, so this must run first.
"""
import os
import sys
import tempfile

SERVICES = os.path.dirname(os.path.abspath(__file__))
if SERVICES not in sys.path:
    sys.path.insert(0, SERVICES)

_TEST_ROOT = tempfile.mkdtemp(prefix="ce-test-")
os.environ["ENSEMBLE_LEDGER_ROOT"] = os.path.join(_TEST_ROOT, "ledger")
os.environ["ENSEMBLE_PROMPT_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "prompts", "prompts.jsonl")
os.environ["ENSEMBLE_SCRAPE_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "scrape", "fetches.jsonl")
os.environ["ENSEMBLE_OUTCOME_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "outcomes", "outcomes.jsonl")
