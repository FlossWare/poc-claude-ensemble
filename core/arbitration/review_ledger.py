"""Record worker results and arbiter decisions to the reviews ledger.

Rows use the same shape as the review tool (tools/multi_model_review.py): one row per worker result and one
per arbiter decision, sharing a review_id for the run. Recording never stops a review: write failures are
logged and ignored.
"""

import hashlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_REVIEW_LEDGER", "~/.FlossWare/ledger/reviews/reviews.jsonl"))
MAX_TEXT = 4000


def new_review_id(task_description):
    return hashlib.sha256(f"{task_description}|{time.time_ns()}".encode("utf-8")).hexdigest()[:16]


def _append(row):
    try:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        logger.warning("review ledger write failed: %s", exc)


def record_phase(review_id, task_type, phase, worker_results, arbiter_result):
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base = {"schema_version": 1, "producer": "arbitration", "review_id": review_id,
            "task_type": task_type, "recorded_at": recorded_at}
    for result in worker_results:
        _append(dict(base, kind="worker_result", role="worker", phase=phase, model=result.model,
                     succeeded=result.succeeded, error=result.error,
                     text=(result.analysis or "")[:MAX_TEXT]))
    if arbiter_result is not None:
        _append(dict(base, kind="arbiter_decision", role="arbiter", phase=phase, model=arbiter_result.model,
                     selected_worker=arbiter_result.selected_worker, rationale=(arbiter_result.rationale or "")[:MAX_TEXT],
                     text=(arbiter_result.adjudicated_result or "")[:MAX_TEXT]))
