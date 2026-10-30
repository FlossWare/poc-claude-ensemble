"""Shadow selection: ask Thompson which model it would choose, and record that next to the real choice.

Nothing here changes what runs. The live model is always the one the caller already chose. A shadow row
shows what Thompson proposed for the same slot, so the two can be compared before Thompson is allowed to
decide anything. If Thompson is unreachable, the row is still written with proposal set to null.

Rows go to the selection ledger (producer arbitration, kind selection_shadow).
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

THOMPSON_URL = os.environ.get("ENSEMBLE_THOMPSON_URL", "http://127.0.0.1:8769")
TIMEOUT = 2
LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_SELECTION_LEDGER",
                                           "~/.FlossWare/ledger/selections/selections.jsonl"))


def propose(task_type, required_capability=0.5):
    """Thompson's proposed model for a task type, or None when Thompson cannot answer."""
    body = json.dumps({"action": "select_model", "task_type": task_type,
                       "required_capability": required_capability}).encode("utf-8")
    request = urllib.request.Request(f"{THOMPSON_URL}/thompson", data=body, method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            reply = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    model = reply.get("model") if reply.get("ok") is True else None
    return model if isinstance(model, str) and model else None


def record(role, chosen, eligible, source, slot=None):
    """Write one shadow row for a selection the caller has already made. Never raises."""
    try:
        proposal = propose(f"{source}_{role}")
        selection_id = hashlib.sha256(f"{source}|{role}|{time.time_ns()}".encode("utf-8")).hexdigest()[:24]
        row = {"schema_version": 1, "producer": "arbitration", "kind": "selection_shadow",
               "selection_id": selection_id, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": source, "role": role, "slot": slot, "chosen": list(chosen), "eligible": list(eligible),
               "thompson_proposal": proposal, "proposal_in_eligible": proposal in eligible if proposal else None,
               "agrees_with_chosen": (proposal in chosen) if proposal else None}
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row
    except OSError:
        return None
