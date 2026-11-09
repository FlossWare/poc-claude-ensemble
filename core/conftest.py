"""Test isolation for the core runtime.

The shadow-selection settings are read when a module is imported, so they are set here, before any test
module imports the arbitration or shared packages. Tests never write the live selection ledger and never
depend on a running Thompson service.
"""
import os
import socket
import sys
import tempfile
from pathlib import Path

_SERVICES_ROOT = Path(__file__).resolve().parents[1] / "services"
_SERVICE_PATHS = [
    _SERVICES_ROOT,
    _SERVICES_ROOT / "memory",
    _SERVICES_ROOT / "learning-service",
    _SERVICES_ROOT / "thompson-service",
    _SERVICES_ROOT / "graph-service",
    _SERVICES_ROOT / "embeddings",
    _SERVICES_ROOT / "learning",
    _SERVICES_ROOT / "server",
    _SERVICES_ROOT / "collaboration",
]
for _service_path in _SERVICE_PATHS:
    if str(_service_path) not in sys.path:
        sys.path.insert(0, str(_service_path))

# Subprocess-based integration tests must inherit the same canonical service paths.
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [*(str(path) for path in _SERVICE_PATHS), *([_existing_pythonpath] if _existing_pythonpath else [])]
)

_TEST_ROOT = tempfile.mkdtemp(prefix="ce-core-test-")
os.environ.setdefault("POC_CLAUDE_ENSEMBLE_SELECTION_LEDGER", os.path.join(_TEST_ROOT, "selections.jsonl"))
Path(os.environ["POC_CLAUDE_ENSEMBLE_SELECTION_LEDGER"]).parent.mkdir(parents=True, exist_ok=True)
Path(os.environ["POC_CLAUDE_ENSEMBLE_SELECTION_LEDGER"]).touch(exist_ok=True)
with socket.socket() as _probe:
    _probe.bind(("127.0.0.1", 0))
    os.environ.setdefault("POC_CLAUDE_ENSEMBLE_THOMPSON_URL", f"http://127.0.0.1:{_probe.getsockname()[1]}")

os.environ.setdefault("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", os.path.join(_TEST_ROOT, "reviews.jsonl"))
