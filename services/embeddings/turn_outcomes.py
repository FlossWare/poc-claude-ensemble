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
                                                 "~/.FlossWare/poc-claude-ensemble/services/cost_tracking/pricing.py"))
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


CORRECTION_PHRASES = ("wrong", "not right", "isn't right", "not what i", "not what we", "that's not",
                      "that is not", "doesn't work", "does not work", "didn't work", "did not work",
                      "won't work", "wont work", "still broken", "still failing", "try again", "undo",
                      "revert", "incorrect", "you missed", "you forgot", "mistake")
CORRECTION_OPENERS = ("no ", "nope", "nah", "no,", "no.", "no!")
INTERRUPT_MARKER = "[Request interrupted by user"


def _human_text(message):
    """Text typed by the user, or None for tool results and automated messages (system reminders, notifications)."""
    content = message.get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        content = " ".join(texts) if texts else None
    if not isinstance(content, str) or not content.strip():
        return None
    if content.lstrip().startswith("<"):
        return None
    return content


def _result_text(block):
    content = block.get("content")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else ""


def _signal_from_human(text):
    if INTERRUPT_MARKER in text:
        return "interrupted"
    lowered = text[:300].lower()
    if lowered.startswith(CORRECTION_OPENERS) or lowered == "no":
        return "correction"
    if any(phrase in lowered for phrase in CORRECTION_PHRASES):
        return "correction"
    return None


def _finish(outcome, later_signal):
    """Decide success from the first failure signal seen. No failure signal means success."""
    if outcome["tool_error"]:
        signal = "tool_error"
    elif outcome["interrupted"]:
        signal = "interrupted"
    elif later_signal:
        signal = later_signal
    elif outcome["next_message"] is None:
        signal = "provisional"
    else:
        signal = "none"
    result = {k: v for k, v in outcome.items() if k not in ("tool_error", "interrupted", "next_message")}
    # Only objective failures count against a model. A correction is recorded in `signal` for later
    # evaluation, because keyword detection of user pushback is not accurate enough to score models.
    result["success"] = signal not in ("tool_error", "interrupted")
    result["signal"] = signal
    return result


def outcomes_from_lines(lines):
    """Yield one outcome per finished assistant turn, from transcript JSON lines in order.

    A turn is judged by the failure signals in it (tool errors, interruption) and by the user's next
    message (a correction). A turn at the end of the input is judged without the next message and is
    marked provisional.
    """
    pending = None
    tool_error = False
    interrupted = False
    seen = set()   # one API response is written as several records; each message id is one turn
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_error = tool_error or bool(block.get("is_error"))
                        interrupted = interrupted or INTERRUPT_MARKER in _result_text(block)
            text = _human_text(message)
            if text is not None and pending is not None:
                yield _finish(dict(pending, next_message=text), _signal_from_human(text))
                pending = None
            elif text is not None:
                tool_error = interrupted = False
            continue
        if role != "assistant":
            continue
        stop = message.get("stop_reason")
        if stop == "max_tokens":
            tool_error = interrupted = False
            continue
        if stop != "end_turn":
            continue
        key = message.get("id") or record.get("uuid")
        if not key:
            continue
        if pending is not None and pending["key"] == key:
            continue   # another record of the same response
        if key in seen:
            continue
        seen.add(key)
        arm = arm_for(message.get("model"))
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        if pending is not None:
            yield _finish(pending, None)
            pending = None
        if arm is None:
            tool_error = interrupted = False
            continue
        pending = {"key": key, "uuid": record.get("uuid") or key, "model": message.get("model"), "arm": arm,
                   "input_tokens": int(usage.get("input_tokens") or 0),
                   "output_tokens": int(usage.get("output_tokens") or 0),
                   "cache_write_tokens": int(usage.get("cache_creation_input_tokens") or 0),
                   "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
                   "timestamp": record.get("timestamp") or "",
                   "tool_error": tool_error, "interrupted": interrupted, "next_message": None}
        tool_error = interrupted = False
    if pending is not None:
        yield _finish(pending, None)


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


def event_id(key):
    return hashlib.sha256(f"turn-outcome|{key}".encode("utf-8")).hexdigest()


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
        eid = event_id(outcome["key"])
        if eid in done:
            continue
        outcome = dict(outcome, cost=cost_of(outcome),
                       tokens=outcome["input_tokens"] + outcome["output_tokens"]
                       + outcome["cache_write_tokens"] + outcome["cache_read_tokens"])
        if eid not in seen:
            _append({"schema_version": 1, "producer": "claude-code", "kind": "turn_outcome",
                     "event_id": eid, "status": "pending", "uuid": outcome["uuid"], "arm": outcome["arm"],
                     "model": outcome["model"], "success": outcome["success"], "signal": outcome["signal"],
                     "cost": outcome["cost"],
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
