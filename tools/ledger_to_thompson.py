#!/usr/bin/env python3
"""Feed review-ledger outcomes to the Thompson router.

Reads the review ledger (~/.FlossWare/ledger/reviews/reviews.jsonl), groups
rows by review_id, and records one Thompson outcome per row:

- worker_result: success = produced usable text with no worker error
- arbiter_decision: success = the decision contains a parseable VERDICT

Processed review_ids are tracked in a state file next to the ledger, so
re-running never double-counts. multi_model_review.py calls sync() after each
review; running this script standalone backfills any reviews the tool has not
seen yet.

Usage: ledger_to_thompson.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(REPO, "services", "shared")
if os.path.isdir(SHARED) and SHARED not in sys.path:
    sys.path.insert(0, SHARED)

from thompson_client import ThompsonClient  # noqa: E402

LEDGER = os.path.expanduser(os.environ.get(
    "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", "~/.FlossWare/ledger/reviews/reviews.jsonl"))
STATE = os.path.join(os.path.dirname(LEDGER), "thompson-sync-state.json")

# Ledger model IDs -> Thompson arm names (the arms that already exist in
# thompson-sampling-state.json). Unknown IDs pass through unchanged.
MODEL_ALIASES = {
    "rits/zai-org/glm-5-3": "glm-5-3",
    "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017": "deepseek-v4.1-flash",
    "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4": "muse-spark-open",
    "claude-opus-5-5": "opus",
    "claude-sonnet-5": "sonnet",
    "claude-sonnet-5-5": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    "claude-haiku-5-5": "haiku",
}

VERDICTS = ("approve with fixes", "approve", "reject")

# Event-driven GA trigger: once this many *new* outcome rows are recorded, a
# detached report-only GA run is launched. It never blocks the caller; the GA
# holds an flock, so overlapping triggers are no-ops.
GA_TRIGGER_MIN_ROWS = 4
# Auto-apply is env-gated (POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY=1) per the GA design;
# the trigger passes --apply only when the env var is set, so report-only remains
# the default everywhere the variable is unset.
GA_ARGV = ["-m", "learning.ga", "run", "--target", "thompson_policy"] + (
    ["--apply"] if os.environ.get("POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY") == "1" else ["--report-only"])


def _maybe_trigger_tuning(recorded_rows, *, threshold=GA_TRIGGER_MIN_ROWS, popen=subprocess.Popen):
    """Spawn a detached, report-only GA run when enough new rows were recorded.

    Fixed argv, no shell, no caller data. Returns the Popen handle, or None
    when the threshold is not met or the spawn fails. Never raises.
    """
    if recorded_rows < threshold:
        return None
    services = os.path.join(REPO, "services")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = services + (os.pathsep + existing if existing else "")
    argv = [sys.executable] + GA_ARGV
    try:
        return popen(
            argv,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        print(f"ga tuning trigger skipped: {exc}", file=sys.stderr)
        return None


def verdict_of(text):
    """The verdict on the last line starting with VERDICT:, or 'unclear' if there is none."""
    for line in reversed((text or "").strip().splitlines()):
        if line.strip().upper().startswith("VERDICT:"):
            value = line.split(":", 1)[1].strip().lower().rstrip(".")
            return value if value in VERDICTS else "unclear"
    return "unclear"


def load_processed():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as handle:
            return set(json.load(handle).get("review_ids", []))
    return set()


def save_processed(ids):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"review_ids": sorted(ids)}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def sync(dry_run=False):
    """Record Thompson outcomes for every ledger review not yet processed."""
    if not os.path.exists(LEDGER):
        print(f"no ledger at {LEDGER}; nothing to sync")
        return 0

    rows = []
    with open(LEDGER, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    by_review = {}
    for row in rows:
        review_id = row.get("review_id")
        if review_id:
            by_review.setdefault(review_id, []).append(row)

    processed = load_processed()
    pending = [rid for rid in by_review if rid not in processed]

    client = None if dry_run else ThompsonClient()
    recorded = 0
    for review_id in pending:
        # Quality weighting: parse the arbiter's decision for this review first,
        # so each worker outcome can be weighted by whether the arbiter judged
        # that worker's analysis the strongest. SELECTED names the winner.
        arbiter_rows = [r for r in by_review[review_id] if r.get("kind") == "arbiter_decision"]
        selected_model = None
        for arow in reversed(arbiter_rows):
            verdict_text = arow.get("text") or ""
            for line in reversed(verdict_text.strip().splitlines()):
                if line.strip().upper().startswith("SELECTED:"):
                    selected_model = line.split(":", 1)[1].strip()
                    break
            if selected_model:
                break
        for row in by_review[review_id]:
            kind = row.get("kind")
            if kind == "worker_result":
                success = bool((row.get("text") or "").strip()) and not row.get("error")
                # Quality weighting: Thompson's record_outcome is binary, so we
                # sharpen the signal by recording the arbiter's winner with an
                # additional positive outcome (a bonus success) when the arbiter
                # SELECTED this worker. Models that win arbitration learn faster
                # than models that merely produce usable text.
                model_label = row.get("model", "")
                if selected_model and _label_matches(selected_model, model_label):
                    if dry_run:
                        print(f"would record BONUS: {model_label} (arbiter-selected)")
                    else:
                        client.record_outcome(model=model_label, task_type="code-review",
                                              success=True, cost=0.0, tokens=0,
                                              request_id=f"ledger-quality-{review_id}")
                    recorded += 1
            elif kind == "arbiter_decision":
                success = verdict_of(row.get("text")) in VERDICTS
            else:
                continue
            model = MODEL_ALIASES.get(row.get("model"), row.get("model"))
            tokens = (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
            cost = row.get("cost_usd") or 0.0
            if dry_run:
                print(f"would record: {model} success={success} cost_usd={cost} tokens={tokens} "
                      f"(review {review_id})")
            else:
                client.record_outcome(model=model, task_type="code-review", success=success,
                                      cost=cost, tokens=tokens, request_id=f"ledger-{review_id}")
            recorded += 1
        processed.add(review_id)

    if not dry_run:
        save_processed(processed)
        process = _maybe_trigger_tuning(recorded)
        if process is not None:
            print(f"ga tuning run launched (pid {process.pid})")
    print(f"reviews synced: {len(pending)} (outcome rows: {recorded}); "
          f"total processed reviews: {len(processed)}")
    return recorded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be recorded without touching Thompson state")
    args = parser.parse_args(argv)
    sync(dry_run=args.dry_run)


if __name__ == "__main__":
    main()


def _label_matches(selected: str, model: str) -> bool:
    """Whether an arbiter's SELECTED label refers to this worker's model."""
    aliases = {
        "GLM 5.3": ("glm-5-3", "rits/zai-org/glm-5-3"),
        "DeepSeek V4.1 Flash": ("deepseek", "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        "Muse Spark Open": ("muse", "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4"),
    }
    model_lower = model.lower()
    for label, keys in aliases.items():
        if selected.strip() == label or any(k.lower() in model_lower for k in keys):
            if selected.strip() == label:
                return True
    return False
