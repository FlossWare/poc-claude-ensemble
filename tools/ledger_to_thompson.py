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

# Local Ollama arms use the same label as the Thompson arm name.  Kept in sync
# with crew_select.OLLAMA_ARMS / ARM_MODEL_IDS; the crew-select test asserts
# every crew arm maps back to itself here, so drift fails the suite.
for _ollama_model in (
    "ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b", "ollama/qwen3:4b",
    "ollama/qwen3:1.7b", "ollama/qwen3.5:2b", "ollama/gemma4:e2b",
    "ollama/gemma3:4b", "ollama/phi4-mini", "ollama/llama3.2:3b",
    "ollama/granite3.2-vision:2b",
):
    MODEL_ALIASES.setdefault(_ollama_model, _ollama_model)

# Local OpenVINO serving-layer arms use the same label as the Thompson arm
# name.  Kept in sync with crew_select.LOCAL_ARMS; the crew-select test asserts
# every crew arm maps back to itself here, so drift fails the suite.
for _local_model in ("local/qwen3-coder-30b-a3b", "local/qwen3.6-35b-a3b"):
    MODEL_ALIASES.setdefault(_local_model, _local_model)

VERDICTS = ("approve with fixes", "approve", "reject")

# Event-driven GA trigger: once this many *new* outcome rows are recorded, a
# detached report-only GA run is launched. It never blocks the caller; the GA
# holds an flock, so overlapping triggers are no-ops.
GA_TRIGGER_MIN_ROWS = 4
# Auto-apply is env-gated (POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY=1) per the GA design;
# the trigger passes --apply only when the env var is set, so report-only remains
# the default everywhere the variable is unset.  The argv is built per call so
# env changes after import (tests, long-lived importers) take effect.


def _ga_argv():
    return ["-m", "learning.ga", "run", "--target", "thompson_policy"] + (
        ["--apply"] if os.environ.get("POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY") == "1"
        else ["--report-only"])


# Quality predictor (learning.train): when an artifact exists, its probability
# complements the binary arbiter-SELECTED bonus. The score is recorded as an
# extra positive outcome for workers whose output the model rates highly, so
# Thompson learns from graded quality, not only the winner-take-all selection.
def _quality_bonus_threshold():
    """Parse the env threshold per call; non-numeric values fall back to 0.5."""
    raw = os.environ.get("POC_CLAUDE_ENSEMBLE_QUALITY_BONUS_THRESHOLD", "0.5")
    try:
        value = float(raw)
    except ValueError:
        print(f"invalid POC_CLAUDE_ENSEMBLE_QUALITY_BONUS_THRESHOLD {raw!r}; "
              "using 0.5", file=sys.stderr)
        return 0.5
    return min(1.0, max(0.0, value))
_QUALITY_MODEL = "unset"


def _quality_artifact():
    root = os.environ.get("POC_CLAUDE_ENSEMBLE_TRAIN_DIR")
    root = os.path.expanduser(root) if root else os.path.expanduser("~/.FlossWare/learning/train")
    return os.path.join(root, "quality_predictor.joblib")


def _load_quality_model():
    """Load the trained quality predictor once; ``None`` when unavailable.

    Import errors and corrupt artifacts are non-fatal: review recording must
    never depend on an optional model.
    """
    global _QUALITY_MODEL
    if _QUALITY_MODEL != "unset":
        return _QUALITY_MODEL
    path = _quality_artifact()
    try:
        if not os.path.exists(path):
            _QUALITY_MODEL = None
            return None
        services = os.path.join(REPO, "services")
        if services not in sys.path:
            sys.path.insert(0, services)
        from learning.train.quality_predictor import QualityPredictor

        _QUALITY_MODEL = QualityPredictor.load(path)
    except Exception as exc:
        print(f"quality predictor unavailable: {exc}", file=sys.stderr)
        _QUALITY_MODEL = None
    return _QUALITY_MODEL


def _quality_score(row):
    """The model's P(selected/usable) for one worker row, or ``None``."""
    model = _load_quality_model()
    if model is None:
        return None
    try:
        from learning.train.quality_predictor import worker_features

        return float(model.predict_quality(worker_features(row)))
    except Exception as exc:
        # A persistently broken predictor must not become invisible: report
        # the first failure, then stay quiet (the bonus simply stops firing).
        global _QUALITY_SCORE_FAILURES
        _QUALITY_SCORE_FAILURES += 1
        if _QUALITY_SCORE_FAILURES == 1:
            print(f"quality predictor scoring failed ({exc}); "
                  "quality bonus disabled until it recovers", file=sys.stderr)
        return None


_QUALITY_SCORE_FAILURES = 0


def _maybe_trigger_tuning(recorded_rows, *, threshold=GA_TRIGGER_MIN_ROWS, popen=subprocess.Popen):
    """Spawn a detached GA run when enough new rows were recorded.

    The run applies mutations only when POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY=1;
    otherwise it is report-only. Fixed argv, no shell, no caller data.
    Returns the Popen handle, or None when the threshold is not met or the
    spawn fails. Never raises.
    """
    if recorded_rows < threshold:
        return None
    services = os.path.join(REPO, "services")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = services + (os.pathsep + existing if existing else "")
    argv = [sys.executable] + _ga_argv()
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

    # Arbiter overturn detection: group completed reviews by subject hash;
    # when two reviews cover the same subject and their arbiters selected
    # different workers, the earlier arbiter's judgment was overturned.
    by_subject = {}
    for rid in pending:
        for row in by_review[rid]:
            if row.get("kind") == "arbiter_decision" and row.get("subject_sha256"):
                by_subject.setdefault(row["subject_sha256"], []).append((rid, row))
    overturned_arbiters = set()  # review_ids whose arbiter was overturned
    for subject, reviews in by_subject.items():
        if len(reviews) < 2:
            continue
        # Chronological by review_id (ledger order)
        reviews.sort(key=lambda r: r[0])
        first_selected = _parse_selected(reviews[0][1].get("text") or "")
        for later_rid, later_row in reviews[1:]:
            later_selected = _parse_selected(later_row.get("text") or "")
            if first_selected and later_selected and first_selected != later_selected:
                overturned_arbiters.add(reviews[0][0])
                break  # only the first is "overturned" per subject

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
                # sharpen the signal by recording an additional positive outcome
                # (a bonus success) when either the arbiter SELECTED this worker
                # or the trained quality predictor scores it highly. The two
                # signals complement each other: SELECTED is authoritative but
                # sparse; the predictor generalises to reviews without it.
                model_label = row.get("model", "")
                arbiter_selected = bool(selected_model and _label_matches(selected_model, model_label))
                quality = _quality_score(row) if success else None
                quality_bonus = quality is not None and quality >= _quality_bonus_threshold()
                if success and (arbiter_selected or quality_bonus):
                    if arbiter_selected:
                        reason = "arbiter-selected"
                    else:
                        reason = f"quality={quality:.3f}"
                    if dry_run:
                        print(f"would record BONUS: {model_label} ({reason}, cost={row.get('cost_usd') or 0.0})")
                    else:
                        # Cost-normalized: the bonus carries the worker's real cost
                        # so Thompson's cost-weighted draws prefer efficient winners.
                        client.record_outcome(model=model_label, task_type="code-review",
                                              success=True,
                                              cost=row.get("cost_usd") or 0.0,
                                              tokens=(row.get("input_tokens") or 0) + (row.get("output_tokens") or 0),
                                              request_id=f"ledger-quality-{review_id}-{model_label}")
                    recorded += 1
            elif kind == "arbiter_decision":
                success = verdict_of(row.get("text")) in VERDICTS
                # Overturn penalty: if this review's arbiter was overturned
                # by a later review of the same subject, record a failure.
                if review_id in overturned_arbiters:
                    success = False
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


def _parse_selected(text: str):
    """The model label from the last SELECTED: line, or None."""
    for line in reversed((text or "").strip().splitlines()):
        if line.strip().upper().startswith("SELECTED:"):
            return line.split(":", 1)[1].strip()
    return None
