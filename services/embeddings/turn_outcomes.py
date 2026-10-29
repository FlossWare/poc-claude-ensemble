"""Turn outcomes from transcripts, recorded to Thompson through its REST route.

Each finished assistant turn (stop_reason end_turn) becomes one outcome:

- arm: the model family, haiku, sonnet, or opus; each gateway model is its own arm.
- success: true unless a tool result in that turn was an error.
- cost: from the central pricing table, including prompt-cache tokens.
- turns that stopped at max_tokens are not recorded.

Every outcome is appended to the ledger before it is sent. The ledger then gets a second row
once Thompson has accepted it. Outcomes that were never accepted are retried on later ingests,
so a Thompson outage delays recording without losing it. Outcome IDs are the message UUIDs,
so a retry never records the same turn twice.
"""

import hashlib
import importlib.util
import json
import os
import urllib.error
import urllib.request

THOMPSON_URL = os.environ.get("ENSEMBLE_THOMPSON_URL", "http://127.0.0.1:8769")
THOMPSON_TIMEOUT = 5
TASK_TYPE = "claude_code_turn"
LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_OUTCOME_LEDGER",
                                           "~/.FlossWare/ledger/outcomes/outcomes.jsonl"))
PRICING_PATH = os.path.expanduser(os.environ.get("ENSEMBLE_PRICING_PATH",
                                                 "~/.FlossWare/poc-claude-ensemble/core/cost_tracking/pricing.py"))
GATEWAY_ARMS = {
    "rits/zai-org/glm-5-3": "glm-5-3",
    "rits/deepseek-ai/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash",
}
_pricing = None


def _load_pricing():
    global _pricing
    if _pricing is None:
        spec = importlib.util.spec_from_file_location("pricing", PRICING_PATH)
        _pricing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_pricing)
    return _pricing


def arm_for(model):
    """Map a transcript model ID to a Thompson arm, or None when the turn is not recorded."""
    if not isinstance(model, str) or not model or model.startswith("<"):
        return None
    lowered = model.lower()
    if lowered in GATEWAY_ARMS:
        return GATEWAY_ARMS[lowered]
    for family in ("haiku", "sonnet", "opus"):
        if family in lowered:
            return family
    return None


def pricing_model(model):
    """The model ID as the pricing table knows it."""
    return model.lower() if model.lower().startswith("rits/") else model


def outcomes_from_lines(lines):
    """Yield one outcome per finished assistant turn, from transcript JSON lines in order."""
    tool_error = False
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                    tool_error = True
            continue
        if message.get("role") != "assistant":
            continue
        stop = message.get("stop_reason")
        if stop != "end_turn":
            if stop == "max_tokens":
                tool_error = False
            continue
        arm = arm_for(message.get("model"))
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        success = not tool_error
        tool_error = False
        if arm is None or not record.get("uuid"):
            continue
        yield {"uuid": record["uuid"], "model": message.get("model"), "arm": arm, "success": success,
               "input_tokens": int(usage.get("input_tokens") or 0),
               "output_tokens": int(usage.get("output_tokens") or 0),
               "cache_write_tokens": int(usage.get("cache_creation_input_tokens") or 0),
               "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
               "timestamp": record.get("timestamp") or ""}


def cost_of(outcome):
    pricing = _load_pricing()
    return round(pricing.calculate_cost(pricing_model(outcome["model"]), outcome["input_tokens"],
                                        outcome["output_tokens"], outcome["cache_write_tokens"],
                                        outcome["cache_read_tokens"]), 6)


def _ledger_rows():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def event_id(uuid):
    return hashlib.sha256(f"turn-outcome|{uuid}".encode("utf-8")).hexdigest()


def post_to_thompson(outcome):
    """Send one outcome to Thompson. Returns True only when Thompson accepted it."""
    body = json.dumps({"action": "record_outcome", "model": outcome["arm"], "task_type": TASK_TYPE,
                       "success": outcome["success"], "cost": outcome["cost"],
                       "tokens": outcome["tokens"]}).encode("utf-8")
    request = urllib.request.Request(f"{THOMPSON_URL}/thompson", data=body, method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=THOMPSON_TIMEOUT) as response:
            reply = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return reply.get("ok") is True


def record_pending(outcomes):
    """Ledger-first recording. Returns (accepted, pending) counts for this batch and older retries."""
    rows = _ledger_rows()
    done = {r["event_id"] for r in rows if r.get("status") == "accepted"}
    seen = {r["event_id"] for r in rows}
    accepted = pending = 0
    for outcome in outcomes:
        eid = event_id(outcome["uuid"])
        if eid in done:
            continue
        outcome = dict(outcome, cost=cost_of(outcome),
                       tokens=outcome["input_tokens"] + outcome["output_tokens"]
                       + outcome["cache_write_tokens"] + outcome["cache_read_tokens"])
        if eid not in seen:
            _append({"schema_version": 1, "producer": "claude-code", "kind": "turn_outcome",
                     "event_id": eid, "status": "pending", "uuid": outcome["uuid"], "arm": outcome["arm"],
                     "model": outcome["model"], "success": outcome["success"], "cost": outcome["cost"],
                     "tokens": outcome["tokens"], "timestamp": outcome["timestamp"]})
            seen.add(eid)
        if post_to_thompson(outcome):
            _append({"schema_version": 1, "producer": "claude-code", "kind": "turn_outcome",
                     "event_id": eid, "status": "accepted", "uuid": outcome["uuid"]})
            done.add(eid)
            accepted += 1
        else:
            pending += 1
    return accepted, pending


def retry_pending():
    """Resend outcomes that were written but never accepted. Safe to call on every ingest."""
    rows = _ledger_rows()
    done = {r["event_id"] for r in rows if r.get("status") == "accepted"}
    retried = failed = 0
    for row in rows:
        if row.get("status") != "pending" or row["event_id"] in done:
            continue
        outcome = {"arm": row["arm"], "success": row["success"], "cost": row["cost"], "tokens": row["tokens"]}
        if post_to_thompson(outcome):
            _append({"schema_version": 1, "producer": "claude-code", "kind": "turn_outcome",
                     "event_id": row["event_id"], "status": "accepted", "uuid": row["uuid"]})
            done.add(row["event_id"])
            retried += 1
        else:
            failed += 1
    return retried, failed
