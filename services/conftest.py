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
os.environ["POC_CLAUDE_ENSEMBLE_LEDGER_ROOT"] = os.path.join(_TEST_ROOT, "ledger")
os.environ["POC_CLAUDE_ENSEMBLE_PROMPT_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "prompts", "prompts.jsonl")
os.environ["POC_CLAUDE_ENSEMBLE_SCRAPE_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "scrape", "fetches.jsonl")
os.environ["POC_CLAUDE_ENSEMBLE_OUTCOME_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "outcomes", "outcomes.jsonl")

# GA state, the Thompson policy file, and the review/selection ledgers must never
# be the live ones during tests.
os.environ["POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "reviews", "reviews.jsonl")
os.environ["POC_CLAUDE_ENSEMBLE_SELECTION_LEDGER"] = os.path.join(_TEST_ROOT, "ledger", "selections", "selections.jsonl")
os.environ["POC_CLAUDE_ENSEMBLE_THOMPSON_STATE_FILE"] = os.path.join(
    _TEST_ROOT, "learning", "thompson-sampling-state.json"
)
os.environ["POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE"] = os.path.join(
    _TEST_ROOT, "learning", "thompson-policy.json"
)
os.environ["POC_CLAUDE_ENSEMBLE_GA_DIR"] = os.path.join(_TEST_ROOT, "learning", "ga")
# learning.train artifacts are read lazily by tools/ledger_to_thompson.py; keep
# tests away from any operator-trained model.
os.environ["POC_CLAUDE_ENSEMBLE_TRAIN_DIR"] = os.path.join(_TEST_ROOT, "learning", "train")
# The corpus allowlist must include the redirected ledger/policy roots.
os.environ["POC_CLAUDE_ENSEMBLE_GA_ALLOWED_ROOTS"] = _TEST_ROOT
os.environ.setdefault("POC_CLAUDE_ENSEMBLE_DEPLOYED_SERVICES_DIR", os.path.join(_TEST_ROOT, "deployed-services"))
