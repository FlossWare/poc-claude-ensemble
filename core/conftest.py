"""Test isolation for the core runtime.

The shadow-selection settings are read when a module is imported, so they are set here, before any test
module imports the arbitration or shared packages. Tests never write the live selection ledger and never
depend on a running Thompson service.
"""
import os
import socket
import tempfile

_TEST_ROOT = tempfile.mkdtemp(prefix="ce-core-test-")
os.environ.setdefault("ENSEMBLE_SELECTION_LEDGER", os.path.join(_TEST_ROOT, "selections.jsonl"))
with socket.socket() as _probe:
    _probe.bind(("127.0.0.1", 0))
    os.environ.setdefault("ENSEMBLE_THOMPSON_URL", f"http://127.0.0.1:{_probe.getsockname()[1]}")

os.environ.setdefault("ENSEMBLE_REVIEW_LEDGER", os.path.join(_TEST_ROOT, "reviews.jsonl"))
