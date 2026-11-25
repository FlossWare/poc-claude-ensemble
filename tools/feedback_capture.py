#!/usr/bin/env python3
"""
Conversation Feedback Capture

Allows manual rating of models based on observed performance or insights
during development work. Feeds back into the Learning service to update
Thompson sampling.

Self-contained by design: talks to the Learning service over REST only
(POST /learning, op=process_outcome) and imports nothing from the
surrounding tree, so the deployment does not need the legacy core/
subsystem copies for this tool to run.

Usage:
  feedback_capture.py --model cursor --rating 5 --task refactoring \
    --context "Excellent code transformations, fast iteration"

  feedback_capture.py --model gemini --rating 2 --task security_audit \
    --context "Missed SQL injection vulnerability"

Environment:
  POC_CLAUDE_ENSEMBLE_LEARNING_URL  Learning service base URL
                                    (default http://127.0.0.1:8768)
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime

DEFAULT_SERVICE_URL = "http://127.0.0.1:8768"
REQUEST_TIMEOUT = 3.0


def send_outcome(service_url, task_id, task_type, model, rating,
                 tokens=0, cost=0.0, context=None, request_id=None):
    """POST one process_outcome request to the Learning service.

    Returns the parsed response dict; {"ok": false, "error": ...} on any
    transport failure (graceful degradation, never raises)."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    payload_out = {
        "op": "process_outcome",
        "task_id": task_id,
        "task_type": task_type,
        "model": model,
        "rating": rating,
        "tokens": tokens,
        "cost": cost,
        "request_id": request_id,
    }
    if context is not None:
        # The service validates known fields and accepts extras, so the
        # notes travel with the request even before server-side
        # persistence of a context field exists.
        payload_out["context"] = context
    body = json.dumps(payload_out).encode("utf-8")
    req = urllib.request.Request(
        service_url.rstrip("/") + "/learning",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    try:
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # An HTML/empty proxy error body must not turn into a
            # confusing json parse message; surface the HTTP status.
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = None
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": f"http_{exc.code}"}
            elif not payload.get("ok") and not payload.get("error"):
                payload = {"ok": False, "error": f"http_{exc.code}"}
        if isinstance(payload, dict):
            return payload
        return {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def capture_feedback(model, rating, task_type, context=None, service_url=None):
    """Record conversation feedback as a learning outcome.

    Returns the task id on success, None on failure."""
    if not 1 <= rating <= 5:
        print(f"Error: rating must be 1-5, got {rating}")
        return None

    if service_url is None:
        service_url = os.environ.get(
            "POC_CLAUDE_ENSEMBLE_LEARNING_URL", DEFAULT_SERVICE_URL)

    task_id = f"feedback_{model}_{datetime.now().isoformat()}"

    result = send_outcome(service_url, task_id=task_id, task_type=task_type,
                          model=model, rating=rating, context=context)

    if result.get("ok"):
        print("✓ Feedback recorded:")
        print(f"  Model: {model}")
        print(f"  Rating: {rating}/5")
        print(f"  Task: {task_type}")
        if context:
            print(f"  Notes: {context}")
        print(f"  Task ID: {task_id}")
        return task_id
    print("✗ Failed to record feedback:", result.get("error", "unknown"))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Capture conversation feedback to update Learning/Thompson")
    parser.add_argument(
        "--model", required=True,
        help="Model being rated (haiku, sonnet, opus, cursor, gemini, etc.)")
    parser.add_argument(
        "--rating", type=int, required=True,
        help="Rating 1-5 (1=poor, 5=excellent)")
    parser.add_argument(
        "--task", required=True, dest="task_type",
        help="Task type (code_review, refactoring, security_audit, etc.)")
    parser.add_argument(
        "--context",
        help="Optional context/notes about the observation")
    args = parser.parse_args()

    task_id = capture_feedback(
        model=args.model,
        rating=args.rating,
        task_type=args.task_type,
        context=args.context,
    )
    return 0 if task_id else 1


if __name__ == "__main__":
    sys.exit(main())
