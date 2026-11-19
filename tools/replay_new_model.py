#!/usr/bin/env python3
"""Cold-start bootstrapping for a new model: replay past review tasks.

A model that has never run has a Beta(1, 1) Thompson posterior -- uniform
uncertainty -- so the router spends budget exploring it blind.  This tool
replays review tasks the pool has already done, runs the new model on the same
source with the same worker prompt the original workers received, has the
existing arbiter adjudicate the new output alongside the historical worker
outputs, and folds the result back into Thompson as real evidence:

* arbiter selects the new model  -> base success + quality-weighted bonus
* arbiter selects a historical worker -> base outcome only (success = the new
  model produced usable text, failure otherwise)
* worker call failed -> base failure

Cost and latency are the replay runs' measured values.  Thompson has no latency
parameter, so latency is recorded in the review ledger and reported here;
Thompson receives the real cost and token counts.

Flow
----
1. Read ``~/.FlossWare/ledger/reviews/reviews.jsonl`` (override:
   ``POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER``), group worker rows by
   ``subject_sha256``, and take the N most recent unique subjects (default 20).
2. Reconstruct the exact source the historical workers saw, verified against
   the ledger's ``subject_sha256``.  A subject whose file is gone, unreadable,
   or whose content no longer hashes to the reviewed bytes is skipped with an
   explicit reason -- replaying a different version of the file would compare
   the new model against outputs written for other code.
3. Run the new model through the same gateway path and prompt as
   ``tools/multi_model_review.py`` (``call`` / ``review_prompt``).
4. Ask the arbiter to adjudicate the new output alongside the historical
   worker outputs.  The arbiter prompt is ``multi_model_review.arbiter_prompt``
   plus one machine-readable line requesting the ``SELECTED:`` label that
   ``tools/ledger_to_thompson.py``, the quality predictor, and the meta
   learning domains already parse.  The live ledger's organic arbiter rows
   predate that line, which is why the replay stamps its own.
5. Record outcomes in Thompson with ``request_id`` prefixed ``replay-`` so a
   replay outcome is distinguishable from an organic one, and mark the replay
   review ids as already synced in ``thompson-sync-state.json`` so the organic
   ``ledger_to_thompson`` pass never double-counts them.
6. Optionally retrain the learners (``--retrain-predictors``): the quality
   predictor through ``learning.train`` and the cost/failure/latency routing
   predictors through ``learning.train.predictors``, then report the new
   model's predicted cost, failure probability, and latency.

Ledger rows
-----------
Replay rows go to the same review ledger (so the learners see them) with
``producer="replay-new-model"``, ``kind`` kept as ``worker_result`` /
``arbiter_decision``, a deterministic ``review_id``
``replay-<model>-<subject12>``, and a ``request_id`` starting with ``replay-``.
The arbiter row carries the raw ``selected`` label and the matched
``selected_model``.

Idempotency
-----------
Replaying the same model on the same subject twice does not double-count:
the replay review id is deterministic, the ledger is checked first, and
``replay-state.json`` (next to the ledger) records which replay review ids
have already had their Thompson outcomes written.  If the outcome write was
interrupted, a re-run folds the stored ledger rows into Thompson without
calling the gateway again.  The Thompson service itself does not deduplicate
``request_id`` values; this bookkeeping is what makes retries safe.

Usage
-----
    replay_new_model.py --model rits/new-model-id --reviews 20
                        [--arbiter opus] [--dry-run] [--retrain-predictors]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS_DIR)
SERVICES = os.path.join(REPO, "services")
SHARED = os.path.join(SERVICES, "shared")
for _path in (SHARED, TOOLS_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import ledger_to_thompson  # noqa: E402  (arm-name namespace, shared with organic outcomes)
import multi_model_review as review_tool  # noqa: E402  (gateway call, prompts, pricing)
from thompson_client import ThompsonClient  # noqa: E402

PRODUCER = "replay-new-model"
REQUEST_PREFIX = "replay-"
LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER"
DEFAULT_LEDGER = "~/.FlossWare/ledger/reviews/reviews.jsonl"
TASK_TYPE = "code-review"
MAX_STORED_CHARS = review_tool.MAX_STORED_CHARS

#: ``multi_model_review``'s default truncation, plus the larger budgets its
#: ``--max-chars`` flag has been used with.  Reconstruction is hash-verified,
#: so the candidate order only decides which hit is reported.
SOURCE_TRUNCATION_CANDIDATES = (6000, 20000, 40000, 100000)

ARBITER_ALIASES = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
    "claude-opus-5-5": "claude-opus-5-5",
    "claude-sonnet-5-5": "claude-sonnet-5-5",
    "claude-haiku-5-5": "claude-haiku-5-5",
}
DEFAULT_ARBITER = "opus"

#: Human labels the arbiter may use in a ``SELECTED:`` line, mapped to the
#: gateway model IDs / Thompson arm names they refer to.
ALIAS_LABELS = {
    "glm 5.3": ("glm-5-3", "rits/zai-org/glm-5-3", "zai-org"),
    "deepseek v4.1 flash": (
        "deepseek-v4.1-flash",
        "rits/deepseek-ai/deepseek-v4.1-flash-1017",
        "deepseek",
    ),
    "muse spark open": (
        "muse-spark-open",
        "rits/meta-models/muse-spark-open-816b-a42b-nvfp4",
        "muse",
    ),
    "claude opus 5.5": ("claude-opus-5-5", "opus"),
    "claude sonnet 5.5": ("claude-sonnet-5-5", "sonnet"),
    "claude haiku 5.5": ("claude-haiku-5-5", "haiku"),
}

SELECTION_INSTRUCTION = (
    "\nFinally, name the single review you found most useful, on one final line "
    "exactly:\nSELECTED: <the model identifier shown in parentheses for that review>"
)

STATUS_REPLAYED = "replayed"
STATUS_FINALIZED = "finalized"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Paths and ledger state
# ---------------------------------------------------------------------------


def ledger_path() -> str:
    """The review ledger, resolved at call time so tests can redirect it."""
    return os.path.abspath(os.path.expanduser(os.environ.get(LEDGER_ENV, DEFAULT_LEDGER)))


def replay_state_path(ledger: str) -> str:
    return os.path.join(os.path.dirname(ledger), "replay-state.json")


def sync_state_path(ledger: str) -> str:
    return os.path.join(os.path.dirname(ledger), "thompson-sync-state.json")


def load_rows(ledger: str) -> List[Dict[str, Any]]:
    """Well-formed ledger rows; corrupt lines are skipped, never fatal."""
    rows: List[Dict[str, Any]] = []
    try:
        handle = open(ledger, encoding="utf-8")
    except FileNotFoundError:
        return rows
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: str, document: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_replay_state(ledger: str) -> Dict[str, Any]:
    state = _read_json(replay_state_path(ledger))
    if not isinstance(state.get("records"), dict):
        state["records"] = {}
    return state


def save_replay_state(ledger: str, state: Dict[str, Any]) -> None:
    state.setdefault("schema_version", 1)
    _write_json(replay_state_path(ledger), state)


def load_synced_review_ids(ledger: str) -> set:
    return set(_read_json(sync_state_path(ledger)).get("review_ids", []))


def save_synced_review_ids(ledger: str, ids: Iterable[str]) -> None:
    _write_json(sync_state_path(ledger), {"review_ids": sorted(set(ids))})


def record_row(ledger: str, row: Dict[str, Any]) -> None:
    """Append one row to the review ledger (0600, fsync'd, like the review tool)."""
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    fd = os.open(ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------


@dataclass
class ReviewSubject:
    """One historical subject, plus the review whose workers it was compared against."""

    subject_sha256: str
    subject: str
    source_review_id: str
    recorded_at: str
    prompt_variant: str
    historical: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.subject_sha256[:12]


def _review_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Worker rows of organic reviews, grouped by review_id (replay rows excluded)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "worker_result":
            continue
        if row.get("producer") == PRODUCER:
            continue
        review_id = row.get("review_id")
        if not review_id:
            continue
        groups.setdefault(review_id, []).append(row)
    return groups


def _variant_of(rows: Sequence[Dict[str, Any]]) -> str:
    for row in rows:
        variant = row.get("prompt_variant")
        if variant in review_tool.PROMPT_VARIANTS:
            return variant
    return review_tool.DEFAULT_PROMPT_VARIANT


def select_subjects(rows: Sequence[Dict[str, Any]], limit: int) -> List[ReviewSubject]:
    """The ``limit`` most recent unique subjects, newest first.

    A subject is identified by ``subject_sha256``, so every review in a group
    covered byte-identical source.  The comparison review is the most recent
    one: its worker rows are the freshest evidence and the arbiter sees them
    alongside the new model's output.
    """
    by_subject: Dict[str, List[Tuple[str, int, str, List[Dict[str, Any]]]]] = {}
    groups = _review_groups(rows)
    for order, (review_id, review_rows) in enumerate(groups.items()):
        sha = review_rows[0].get("subject_sha256")
        subject = review_rows[0].get("subject")
        if not (isinstance(sha, str) and sha and isinstance(subject, str) and subject):
            continue
        recorded_at = max(
            (str(row.get("recorded_at") or row.get("observed_at") or "") for row in review_rows),
            default="",
        )
        by_subject.setdefault(sha, []).append((recorded_at, order, review_id, review_rows))

    chosen: List[Tuple[str, str, int, str, List[Dict[str, Any]]]] = []
    for sha, reviews in by_subject.items():
        recorded_at, order, review_id, review_rows = max(
            reviews, key=lambda item: (item[0], item[1])
        )
        chosen.append((recorded_at, sha, order, recorded_at, review_id, review_rows))

    chosen.sort(key=lambda item: (item[0], item[2]), reverse=True)
    selected = []
    for _recency, sha, _order, recorded_at, review_id, review_rows in chosen[:limit]:
        selected.append(
            ReviewSubject(
                subject_sha256=sha,
                subject=str(review_rows[0].get("subject")),
                source_review_id=review_id,
                recorded_at=recorded_at,
                prompt_variant=_variant_of(review_rows),
                historical=list(review_rows),
            )
        )
    return selected


def truncate_source(text: str, max_chars: int) -> str:
    """The same truncation rule as ``multi_model_review.main``."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[: cut + 1] if cut >= 0 else text[:max_chars]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_source(subject: ReviewSubject) -> Tuple[Optional[str], Optional[str]]:
    """The exact reviewed source, or ``(None, reason)`` when it cannot be reproduced."""
    path = os.path.expanduser(subject.subject)
    if not os.path.isfile(path):
        return None, "source file missing"
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as error:
        return None, f"source unreadable: {type(error).__name__}: {error}"
    candidates = list(SOURCE_TRUNCATION_CANDIDATES) + [len(text)]
    for max_chars in dict.fromkeys(candidates):
        source = truncate_source(text, max_chars)
        if _sha256_text(source) == subject.subject_sha256:
            return source, None
    return None, "source no longer matches the reviewed content (subject hash mismatch)"


# ---------------------------------------------------------------------------
# Selection labels
# ---------------------------------------------------------------------------


def parse_selected(text: Optional[str]) -> Optional[str]:
    """The label on the last ``SELECTED:`` line, or ``None``."""
    for line in reversed((text or "").strip().splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("SELECTED:"):
            value = stripped.split(":", 1)[1].strip()
            return value or None
    return None


def label_matches(selected: str, model: str) -> bool:
    """Whether an arbiter ``SELECTED:`` label refers to *model*.

    Strict on purpose: exact identifier, exact basename, or a known human label
    for a known model.  Fuzzy substring matching would let "GLM 5.3" match a
    different GLM variant and silently mis-attribute the win.
    """
    if not selected or not model:
        return False
    label = " ".join(selected.strip().lower().split())
    model_norm = model.strip().lower()
    if label == model_norm or label == model_norm.rsplit("/", 1)[-1]:
        return True
    # A provider-qualified suffix ("zai-org/glm-5-3" for "rits/zai-org/glm-5-3")
    # is still the same model; anything shorter than a path segment is not.
    if "/" in label and model_norm.endswith("/" + label):
        return True
    for alias, keys in ALIAS_LABELS.items():
        if label == alias and any(key in model_norm for key in keys):
            return True
    return False


def resolve_selected(label: Optional[str], models: Sequence[str]) -> Tuple[Optional[str], str]:
    """Match a ``SELECTED:`` label to one of *models*.

    Returns ``(model, status)`` where status is ``selected``, ``unmatched``,
    ``ambiguous``, or ``no_selected_line``.
    """
    if not label:
        return None, "no_selected_line"
    matches = [model for model in dict.fromkeys(models) if label_matches(label, model)]
    if len(matches) == 1:
        return matches[0], "selected"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "unmatched"


def replay_arbiter_prompt(source: str, path: str, reviews: Sequence[Dict[str, Any]]) -> str:
    """The historical arbiter prompt plus the machine-readable selection line."""
    return review_tool.arbiter_prompt(source, path, reviews) + SELECTION_INSTRUCTION


# ---------------------------------------------------------------------------
# Thompson helpers
# ---------------------------------------------------------------------------


def arm_name(model: str) -> str:
    """The Thompson arm for a gateway model ID, in the organic arm namespace."""
    return ledger_to_thompson.MODEL_ALIASES.get(model, model.rsplit("/", 1)[-1])


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"


def thompson_snapshot(client: Optional[ThompsonClient]) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        return client.get_state()
    except Exception:  # noqa: BLE001 - a snapshot is reporting, never required
        return None


def arm_stats(state: Optional[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    """Posterior summary for one arm; absent arms start at Beta(1, 1)."""
    entry = ((state or {}).get("models") or {}).get(arm) or {}
    successes = int(entry.get("successes") or 0)
    failures = int(entry.get("failures") or 0)
    alpha = successes + 1
    beta = failures + 1
    return {
        "present": bool(entry),
        "successes": successes,
        "failures": failures,
        "calls": int(entry.get("calls") or 0),
        "alpha": alpha,
        "beta": beta,
        "mean": alpha / (alpha + beta),
        "total_cost": float(entry.get("total_cost") or 0.0),
    }


def usable(result: Dict[str, Any]) -> bool:
    """A worker result counts as a success when it produced non-empty text and no error."""
    return bool((result.get("text") or "").strip()) and not result.get("error")


def ensure_cost(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Price a gateway result.

    ``multi_model_review.call`` returns tokens but no cost -- the review tool
    prices the call afterwards in ``main``.  A replay must record the real cost
    of its own runs, so it does the same here.  Unknown models stay ``None``;
    the summary reports those as n/a instead of inventing a number.
    """
    if result.get("cost_usd") is None and not result.get("error"):
        result["cost_usd"] = review_tool.cost_usd(
            model, result.get("input_tokens") or 0, result.get("output_tokens") or 0
        )
    return result


def _tokens(result: Dict[str, Any]) -> int:
    return int(result.get("input_tokens") or 0) + int(result.get("output_tokens") or 0)


def _cost(result: Dict[str, Any]) -> float:
    value = result.get("cost_usd")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def thompson_outcomes(
    *,
    result: Dict[str, Any],
    selected_model: Optional[str],
    model: str,
) -> List[Tuple[str, bool]]:
    """The (kind, success) Thompson records this replay earns.

    ``base`` always reflects the run (usable text or not).  ``bonus`` is the
    PR #71/#72 quality-weighted extra positive outcome, earned only when the
    arbiter selected the new model *and* its output is usable.
    """
    success = usable(result)
    records: List[Tuple[str, bool]] = [("base", success)]
    if success and selected_model == model:
        records.append(("bonus", True))
    return records


def apply_thompson_outcomes(
    client: ThompsonClient,
    *,
    arm: str,
    review_id: str,
    result: Dict[str, Any],
    selected_model: Optional[str],
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Record the replay's outcomes.  Returns status per record; never raises."""
    status: Dict[str, Any] = {}
    records = thompson_outcomes(
        result=result, selected_model=selected_model, model=model
    )
    for kind, success in records:
        request_id = f"{review_id}-{kind}"
        if dry_run:
            status[kind] = "would-record"
            continue
        try:
            # The bonus carries the run's real cost too, matching the
            # ledger_to_thompson PR #71/#72 pattern so the bootstrapped
            # posterior is built from the same kind of samples as organic ones.
            ok = client.record_outcome(
                model=arm,
                task_type=TASK_TYPE,
                success=success,
                cost=_cost(result),
                tokens=_tokens(result),
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001 - reported, never fatal to the replay
            ok = False
        status[kind] = "recorded" if ok else "failed"
    return status


# ---------------------------------------------------------------------------
# Ledger rows
# ---------------------------------------------------------------------------


def base_row(subject: ReviewSubject, model: str, review_id: str,
             result: Dict[str, Any], arbiter: str) -> Dict[str, Any]:
    row = {
        "schema_version": 1,
        "producer": PRODUCER,
        "kind": "worker_result",
        "role": "worker",
        "review_id": review_id,
        "request_id": f"{review_id}-worker",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subject": subject.subject,
        "subject_sha256": subject.subject_sha256,
        "replay": True,
        "source_review_id": subject.source_review_id,
        "replay_model": model,
        "arbiter_model": arbiter,
        "prompt_variant": subject.prompt_variant,
        "model": model,
        "text": (result.get("text") or "")[:MAX_STORED_CHARS],
        "latency_ms": result.get("latency_ms"),
        "stop_reason": result.get("stop_reason"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "cost_usd": result.get("cost_usd"),
    }
    if result.get("error"):
        row["error"] = str(result["error"])
    return row


def arbiter_row(subject: ReviewSubject, model: str, review_id: str,
                arbiter: str, result: Dict[str, Any], selected: Optional[str],
                selected_model: Optional[str], selection_status: str,
                verdict: str) -> Dict[str, Any]:
    row = base_row(subject, model, review_id, result, arbiter)
    row.update(
        {
            "kind": "arbiter_decision",
            "role": "arbiter",
            "request_id": f"{review_id}-arbiter",
            "model": arbiter,
            "selected": selected,
            "selected_model": selected_model,
            "selection_status": selection_status,
            "selected_replay_model": selected_model == model,
            "verdict": verdict,
        }
    )
    return row


def replay_review_id(model: str, subject: ReviewSubject) -> str:
    return f"{REQUEST_PREFIX}{model_slug(model)}-{subject.short_hash}"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _existing_replay_rows(
    rows: Sequence[Dict[str, Any]], model: str, subject_sha256: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    """The replay rows already recorded for one model+subject.

    The arbiter row is found through the worker row's ``review_id`` so an
    arbiter row that only carries the replay review id is still linked.
    """
    candidates = [
        row for row in rows
        if row.get("producer") == PRODUCER and row.get("subject_sha256") == subject_sha256
    ]
    worker = next(
        (row for row in candidates
         if row.get("kind") == "worker_result"
         and (row.get("replay_model") == model or row.get("model") == model)),
        None,
    )
    review_id = (worker or {}).get("review_id")
    arbiter = next(
        (row for row in candidates
         if row.get("kind") == "arbiter_decision"
         and (row.get("review_id") == review_id or row.get("replay_model") == model)),
        None,
    )
    return {"worker": worker, "arbiter": arbiter}


def evaluate_arbiter(result: Dict[str, Any], candidate_models: Sequence[str]) -> Dict[str, Any]:
    """Parse a replay arbiter result into selection + verdict."""
    text = result.get("text") or ""
    label = parse_selected(text)
    selected_model, selection_status = resolve_selected(label, candidate_models)
    if result.get("error"):
        selection_status = "arbiter_error"
    return {
        "selected": label,
        "selected_model": selected_model,
        "selection_status": selection_status,
        "verdict": review_tool.verdict_of(text),
    }


def _summarize_subject(record: Dict[str, Any]) -> str:
    status = record["status"]
    if status == STATUS_SKIPPED:
        return f"skip {record['subject_sha256'][:12]}: {record['reason']}"
    if status == STATUS_FAILED:
        return f"FAIL {record['subject_sha256'][:12]}: {record.get('reason') or 'worker failed'}"
    if record.get("dry_run"):
        return (f"would replay {record['subject_sha256'][:12]}: "
                f"{len(record.get('historical_models') or [])} historical model(s), "
                f"variant {record.get('prompt_variant') or review_tool.DEFAULT_PROMPT_VARIANT}")
    if record.get("selected_model"):
        selection = f"selected {record['selected_model']}"
        if record.get("selected"):
            selection += f" [arbiter wrote SELECTED: {record['selected']}]"
    else:
        selection = f"no selection ({record.get('selection_status')})"
    cost = record.get("cost_usd")
    cost_text = f"${cost:.6f}" if isinstance(cost, (int, float)) else "cost n/a"
    latency = record.get("latency_ms")
    latency_text = f"{latency} ms" if latency is not None else "latency n/a"
    return f"{record['subject_sha256'][:12]}: {cost_text}, {latency_text}, {selection}"


def replay_subject(
    *,
    subject: ReviewSubject,
    model: str,
    arm: str,
    arbiter: str,
    ledger: str,
    client: Optional[ThompsonClient],
    call_model,
    dry_run: bool,
) -> Dict[str, Any]:
    """Replay one subject end to end.  Returns a per-subject result record."""
    record: Dict[str, Any] = {
        "subject_sha256": subject.subject_sha256,
        "subject": subject.subject,
        "source_review_id": subject.source_review_id,
        "recorded_at": subject.recorded_at,
        "review_id": replay_review_id(model, subject),
        "prompt_variant": subject.prompt_variant,
        "historical_models": list(dict.fromkeys(
            str(row.get("model")) for row in subject.historical if row.get("model")
        )),
    }

    source, reason = resolve_source(subject)
    if reason:
        record.update({"status": STATUS_SKIPPED, "reason": reason})
        return record

    comparison = [row for row in subject.historical if row.get("model") != model]
    if not comparison:
        record.update({
            "status": STATUS_SKIPPED,
            "reason": "no historical worker output from another model to adjudicate against",
        })
        return record

    if dry_run:
        record.update({
            "status": STATUS_REPLAYED,
            "reason": "dry-run",
            "selection_status": "dry-run",
            "dry_run": True,
        })
        return record

    result = call_model(model, review_tool.review_prompt(source, subject.subject, subject.prompt_variant))
    result.setdefault("model", model)
    ensure_cost(result, model)
    worker_row = base_row(subject, model, record["review_id"], result, arbiter)
    record_row(ledger, worker_row)

    record.update({
        "usable": usable(result),
        "cost_usd": result.get("cost_usd"),
        "latency_ms": result.get("latency_ms"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "stop_reason": result.get("stop_reason"),
    })

    arbiter_result: Optional[Dict[str, Any]] = None
    if result.get("error"):
        record.update({
            "status": STATUS_FAILED,
            "reason": f"worker call failed: {result['error']}",
            "selected_model": None,
            "selection_status": "worker_failed",
        })
    else:
        reviews = list(comparison) + [result]
        try:
            arbiter_result = call_model(
                arbiter, replay_arbiter_prompt(source, subject.subject, reviews)
            )
            arbiter_result.setdefault("model", arbiter)
            ensure_cost(arbiter_result, arbiter)
        except Exception as error:  # noqa: BLE001 - the worker evidence still counts
            arbiter_result = {
                "model": arbiter,
                "text": "",
                "latency_ms": None,
                "stop_reason": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "error": f"{type(error).__name__}: {error}",
            }
        evaluation = evaluate_arbiter(arbiter_result, [model] + [str(row.get("model")) for row in comparison])
        record.update(evaluation)
        record["status"] = STATUS_REPLAYED
        record["arbiter_cost_usd"] = arbiter_result.get("cost_usd")
        record["arbiter_latency_ms"] = arbiter_result.get("latency_ms")
        record_row(
            ledger,
            arbiter_row(
                subject, model, record["review_id"], arbiter, arbiter_result,
                evaluation["selected"], evaluation["selected_model"],
                evaluation["selection_status"], evaluation["verdict"],
            ),
        )

    if client is None:
        record["thompson"] = {"base": "skipped", "bonus": "skipped"}
        return record

    # Bonus eligibility also requires the arbiter to have run and produced a label.
    status = apply_thompson_outcomes(
        client,
        arm=arm,
        review_id=record["review_id"],
        result=result,
        selected_model=record.get("selected_model"),
        model=model,
    )
    record["thompson"] = status
    record.setdefault("status", STATUS_REPLAYED)
    return record


def finalize_subject(
    *,
    subject: ReviewSubject,
    model: str,
    arm: str,
    rows: Sequence[Dict[str, Any]],
    client: Optional[ThompsonClient],
    dry_run: bool,
) -> Dict[str, Any]:
    """Fold already-recorded replay rows into Thompson without calling the gateway."""
    existing = _existing_replay_rows(rows, model, subject.subject_sha256)
    worker = existing["worker"]
    if worker is None:
        return {
            "subject_sha256": subject.subject_sha256,
            "subject": subject.subject,
            "review_id": replay_review_id(model, subject),
            "status": STATUS_SKIPPED,
            "reason": "replay state is missing and no worker row was found",
        }
    arbiter = existing["arbiter"] or {}
    selected_model = arbiter.get("selected_model")
    if selected_model is None and arbiter.get("selected"):
        selected_model, _status = resolve_selected(
            arbiter.get("selected"),
            [model] + [str(row.get("model")) for row in subject.historical],
        )
    record = {
        "subject_sha256": subject.subject_sha256,
        "subject": subject.subject,
        "review_id": arbiter.get("review_id") or worker.get("review_id") or replay_review_id(model, subject),
        "status": STATUS_FINALIZED,
        "usable": bool((worker.get("text") or "").strip()) and not worker.get("error"),
        "cost_usd": worker.get("cost_usd"),
        "latency_ms": worker.get("latency_ms"),
        "input_tokens": worker.get("input_tokens"),
        "output_tokens": worker.get("output_tokens"),
        "stop_reason": worker.get("stop_reason"),
        "selected": arbiter.get("selected"),
        "selected_model": selected_model,
        "selection_status": arbiter.get("selection_status") or ("selected" if selected_model else "unknown"),
        "verdict": arbiter.get("verdict"),
    }
    if dry_run or client is None:
        record["thompson"] = {"base": "skipped", "bonus": "skipped"}
        record["dry_run"] = bool(dry_run)
        return record
    record["thompson"] = apply_thompson_outcomes(
        client,
        arm=arm,
        review_id=record["review_id"],
        result=worker,
        selected_model=selected_model,
        model=model,
    )
    return record


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------


def _new_stats() -> Dict[str, Any]:
    return {"runs": 0, "usable": 0, "selected": 0, "costs": [], "latencies": []}


def build_comparison(subjects: Sequence[ReviewSubject], records: Sequence[Dict[str, Any]],
                     model: str) -> List[Dict[str, Any]]:
    """Per-model stats over the replayed subjects: new model vs. historical workers."""
    stats: Dict[str, Dict[str, Any]] = {}
    replay_label = f"{model} (replay)"
    by_sha = {subject.subject_sha256: subject for subject in subjects}
    for record in records:
        if record.get("status") not in (STATUS_REPLAYED, STATUS_FINALIZED):
            continue
        if record.get("dry_run"):
            continue
        entry = stats.setdefault(replay_label, _new_stats())
        entry["runs"] += 1
        entry["usable"] += 1 if record.get("usable") else 0
        if isinstance(record.get("cost_usd"), (int, float)):
            entry["costs"].append(float(record["cost_usd"]))
        if isinstance(record.get("latency_ms"), (int, float)):
            entry["latencies"].append(float(record["latency_ms"]))
        if record.get("selected_model"):
            selected = (
                replay_label if record["selected_model"] == model else record["selected_model"]
            )
            stats.setdefault(selected, _new_stats())["selected"] += 1
        subject = by_sha.get(record["subject_sha256"])
        if subject is None:
            continue
        for row in subject.historical:
            historical_model = row.get("model")
            if not historical_model or historical_model == model:
                continue
            historical = stats.setdefault(str(historical_model), _new_stats())
            historical["runs"] += 1
            historical["usable"] += 1 if (row.get("text") or "").strip() and not row.get("error") else 0
            if isinstance(row.get("cost_usd"), (int, float)):
                historical["costs"].append(float(row["cost_usd"]))
            if isinstance(row.get("latency_ms"), (int, float)):
                historical["latencies"].append(float(row["latency_ms"]))

    table = []
    for label, entry in stats.items():
        table.append({
            "model": label,
            "runs": entry["runs"],
            "usable": entry["usable"],
            "selected": entry["selected"],
            "avg_cost_usd": statistics.fmean(entry["costs"]) if entry["costs"] else None,
            "avg_latency_ms": statistics.fmean(entry["latencies"]) if entry["latencies"] else None,
            "replay": label == replay_label,
        })
    table.sort(key=lambda row: (not row["replay"], row["model"]))
    return table


def _fmt_money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _print_comparison(table: Sequence[Dict[str, Any]], out) -> None:
    headers = ("model", "runs", "usable", "selected", "avg_cost_usd", "avg_latency_ms")
    widths = [max(len(headers[0]), *(len(str(row["model"])) for row in table)) if table else len(headers[0]),
              len(headers[1]), len(headers[2]), len(headers[3]), len(headers[4]), len(headers[5])]
    print("\ncomparison (same subjects; historical rows come from the source reviews)", file=out)
    print("  " + "  ".join(header.ljust(width) for header, width in zip(headers, widths)), file=out)
    print("  " + "  ".join("-" * width for width in widths), file=out)
    for row in table:
        cells = (
            row["model"], str(row["runs"]), str(row["usable"]), str(row["selected"]),
            _fmt_money(row["avg_cost_usd"]), _fmt_ms(row["avg_latency_ms"]),
        )
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)), file=out)


def _print_posterior(label: str, before: Dict[str, Any], after: Dict[str, Any], out) -> None:
    print(f"\nThompson state ({label})", file=out)
    for name, stats in (("before", before), ("after", after)):
        present = "" if stats["present"] else " [absent: Beta(1, 1)]"
        print(
            f"  {name:6s} successes={stats['successes']} failures={stats['failures']} "
            f"calls={stats['calls']} alpha={stats['alpha']} beta={stats['beta']} "
            f"mean={stats['mean']:.4f}{present}",
            file=out,
        )
    print(
        f"  delta  successes={after['successes'] - before['successes']:+d} "
        f"failures={after['failures'] - before['failures']:+d} "
        f"mean={after['mean'] - before['mean']:+.4f}",
        file=out,
    )


# ---------------------------------------------------------------------------
# Predictor retraining
# ---------------------------------------------------------------------------


def _median_input_tokens(records: Sequence[Dict[str, Any]]) -> int:
    values = [int(record["input_tokens"]) for record in records
              if isinstance(record.get("input_tokens"), int)]
    return int(statistics.median(values)) if values else 0


def retrain_predictors(model: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Retrain the learners on the updated ledger and predict the new model.

    The quality predictor is the one Thompson's bonus weighting consumes; the
    cost/failure/latency predictors are the routing models.  Everything here is
    additive: a training failure is reported, never fatal -- the replay's
    Thompson outcomes are already recorded.
    """
    if SERVICES not in sys.path:
        sys.path.insert(0, SERVICES)
    report: Dict[str, Any] = {"quality_predictor": {}, "routing": {}, "predictions": {}}
    try:
        from learning.train import cli as train_cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = train_cli.train(["quality_predictor"], force=True)
        report["quality_predictor"] = {
            "exit_code": code,
            "output_tail": buffer.getvalue().strip().splitlines()[-6:],
        }
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        report["quality_predictor"] = {"error": f"{type(error).__name__}: {error}"}

    try:
        from learning.train.predictors import cli as predictors_cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = predictors_cli.train_predictors(train_anomaly=False)
        report["routing"] = {
            "worker_rows": result.get("worker_rows"),
            "models": result.get("models"),
            "trained": {
                kind: bool(entry.get("trained"))
                for kind, entry in (result.get("predictors") or {}).items()
            },
            "output_tail": buffer.getvalue().strip().splitlines()[-8:],
        }
        features = {
            "model": model,
            "task_type": "code",
            "input_tokens": _median_input_tokens(records),
        }
        for kind in ("cost", "failure", "latency"):
            try:
                prediction = predictors_cli.predict_one(kind, features)
                report["predictions"][kind] = {
                    "value": prediction.get("prediction"),
                    "unit": prediction.get("unit"),
                    "warnings": prediction.get("warnings") or [],
                }
            except Exception as error:  # noqa: BLE001 - per-kind reporting
                report["predictions"][kind] = {"error": f"{type(error).__name__}: {error}"}
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        report["routing"] = {"error": f"{type(error).__name__}: {error}"}
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    *,
    model: str,
    reviews: int = 20,
    arbiter: str = DEFAULT_ARBITER,
    dry_run: bool = False,
    retrain: bool = False,
    call_model=None,
    client: Optional[ThompsonClient] = None,
    ledger: Optional[str] = None,
    out=sys.stdout,
) -> Dict[str, Any]:
    """Replay the most recent review subjects with ``model`` and bootstrap Thompson."""
    call_model = call_model or review_tool.call
    ledger_file = ledger or ledger_path()
    arbiter_model = ARBITER_ALIASES.get(arbiter, arbiter)
    arm = arm_name(model)

    if arm_name(arbiter_model) == arm:
        message = (
            f"the replay model and the arbiter both resolve to arm {arm!r}; "
            "a model cannot adjudicate its own replay"
        )
        print(f"error: {message}", file=sys.stderr)
        return {"model": model, "arm": arm, "dry_run": dry_run, "failure": message,
                "records": [], "comparison": [], "exit_code": 2}

    print(f"replay-new-model: {model} (arm {arm}), arbiter {arbiter_model}", file=out)
    print(f"ledger: {ledger_file}", file=out)

    rows = load_rows(ledger_file)
    subjects = select_subjects(rows, reviews)
    print(f"subjects selected: {len(subjects)} (requested {reviews}, most recent first)", file=out)

    if not subjects:
        print("no replayable subjects in the ledger; nothing to do", file=out)
        return {"model": model, "arm": arm, "dry_run": dry_run, "subjects": [],
                "records": [], "skipped": [], "comparison": [],
                "thompson": {"before": arm_stats(None, arm), "after": arm_stats(None, arm)},
                "exit_code": 0}

    print("\nsubjects, newest first:", file=out)
    for index, subject in enumerate(subjects, 1):
        print(
            f"  {index:2d}. {subject.short_hash}  {subject.subject}  "
            f"(review {subject.source_review_id}, {subject.recorded_at or 'no timestamp'}, "
            f"history: {len(subject.historical)} worker row(s))",
            file=out,
        )

    if not dry_run:
        client = client if client is not None else ThompsonClient()
        before = thompson_snapshot(client)
    else:
        client = None
        before = None
    state = load_replay_state(ledger_file)
    records_state = state["records"]

    before_stats = arm_stats(before, arm)
    record_list: List[Dict[str, Any]] = []
    pending_sync: List[str] = []
    pending_state: Dict[str, Dict[str, Any]] = {}
    print("", file=out)

    for index, subject in enumerate(subjects, 1):
        review_id = replay_review_id(model, subject)
        existing = _existing_replay_rows(rows, model, subject.subject_sha256)
        prior = records_state.get(review_id) or {}
        if existing["worker"] is not None and prior.get("outcomes_recorded"):
            record = {
                "subject_sha256": subject.subject_sha256,
                "subject": subject.subject,
                "review_id": review_id,
                "status": STATUS_SKIPPED,
                "reason": "already replayed (ledger rows + Thompson outcomes exist)",
            }
            record_list.append(record)
            print(f"  [{index}/{len(subjects)}] {_summarize_subject(record)}", file=out)
            continue
        if existing["worker"] is not None:
            record = finalize_subject(
                subject=subject, model=model, arm=arm, rows=rows,
                client=client, dry_run=dry_run,
            )
        else:
            record = replay_subject(
                subject=subject, model=model, arm=arm, arbiter=arbiter_model,
                ledger=ledger_file, client=client, call_model=call_model, dry_run=dry_run,
            )
        record_list.append(record)
        outcomes = record.get("thompson") or {}
        if not dry_run and outcomes and all(value == "recorded" for value in outcomes.values()):
            if record.get("status") in (STATUS_REPLAYED, STATUS_FINALIZED):
                # All of this replay's Thompson writes landed: record that, and
                # keep the organic ledger_to_thompson pass off the replay rows.
                pending_state[record["review_id"]] = {
                    "model": model,
                    "arm": arm,
                    "subject_sha256": subject.subject_sha256,
                    "outcomes_recorded": True,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                pending_sync.append(record["review_id"])
        print(f"  [{index}/{len(subjects)}] {_summarize_subject(record)}", file=out)

    if not dry_run:
        if pending_sync:
            save_synced_review_ids(ledger_file, load_synced_review_ids(ledger_file) | set(pending_sync))
        if pending_state:
            state["records"].update(pending_state)
            save_replay_state(ledger_file, state)

    after_stats = before_stats
    if not dry_run and client is not None:
        after_stats = arm_stats(thompson_snapshot(client), arm)

    replayed = [r for r in record_list if r.get("status") in (STATUS_REPLAYED, STATUS_FINALIZED)]
    skipped = [r for r in record_list if r.get("status") == STATUS_SKIPPED]
    failed = [r for r in record_list if r.get("status") == STATUS_FAILED]
    adjudicated = [r for r in replayed if r.get("selected_model")]
    selected_new = [r for r in replayed if r.get("selected_model") == model]
    costs = [float(r["cost_usd"]) for r in replayed if isinstance(r.get("cost_usd"), (int, float))]
    latencies = [float(r["latency_ms"]) for r in replayed if isinstance(r.get("latency_ms"), (int, float))]
    arbiter_costs = [float(r["arbiter_cost_usd"]) for r in replayed
                     if isinstance(r.get("arbiter_cost_usd"), (int, float))]

    print("\nsummary", file=out)
    print(f"  subjects: {len(subjects)} selected, {len(replayed)} replayed, "
          f"{len(failed)} failed, {len(skipped)} skipped", file=out)
    if adjudicated:
        print(f"  selection rate: {len(selected_new)}/{len(adjudicated)} "
              f"({100.0 * len(selected_new) / len(adjudicated):.1f}% of adjudicated subjects)", file=out)
    else:
        print("  selection rate: no adjudicated subjects", file=out)
    if replayed and not dry_run:
        counts: Dict[str, int] = {}
        for record in replayed:
            key = record.get("selection_status") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        print("  arbiter labels: " + ", ".join(
            f"{status}={count}" for status, count in sorted(counts.items())), file=out)
    print(f"  avg cost: {('$%.6f' % statistics.fmean(costs)) if costs else 'n/a'} per replay run"
          f"  (arbiter adjudication: "
          f"{('$%.6f' % statistics.fmean(arbiter_costs)) if arbiter_costs else 'n/a'} per subject)",
          file=out)
    print(f"  avg latency: {('%.0f ms' % statistics.fmean(latencies)) if latencies else 'n/a'} per replay run",
          file=out)
    if dry_run:
        print("  thompson: not queried or written (dry-run)", file=out)
        print("  no API calls, no ledger writes, no Thompson writes", file=out)
    else:
        _print_posterior(arm, before_stats, after_stats, out)

    comparison = build_comparison(subjects, record_list, model)
    if comparison:
        _print_comparison(comparison, out)

    retrain_report: Dict[str, Any] = {}
    if retrain:
        if dry_run:
            print("\npredictor retraining: skipped in dry-run", file=out)
        else:
            print("\npredictor retraining", file=out)
            retrain_report = retrain_predictors(model, replayed)
            quality = retrain_report.get("quality_predictor") or {}
            if quality.get("error"):
                print(f"  quality_predictor: error {quality['error']}", file=out)
            else:
                print(f"  quality_predictor: exit_code={quality.get('exit_code')}", file=out)
                for line in quality.get("output_tail") or []:
                    print(f"    {line}", file=out)
            routing = retrain_report.get("routing") or {}
            if routing.get("error"):
                print(f"  routing predictors: error {routing['error']}", file=out)
            else:
                print(f"  routing predictors: {routing.get('trained')} "
                      f"(worker rows {routing.get('worker_rows')})", file=out)
            for kind, prediction in (retrain_report.get("predictions") or {}).items():
                if prediction.get("error"):
                    print(f"  predicted {kind}: error {prediction['error']}", file=out)
                else:
                    print(f"  predicted {kind} for {model}: {prediction.get('value')} "
                          f"{prediction.get('unit')}", file=out)
                for warning in prediction.get("warnings") or []:
                    print(f"    warning: {warning}", file=out)

    failures = [r for r in replayed if (r.get("thompson") or {}).get("base") == "failed"]
    if failures:
        print(f"\nWARNING: Thompson did not acknowledge {len(failures)} outcome write(s); "
              "re-run to retry them from the stored ledger rows", file=sys.stderr)

    return {
        "model": model,
        "arm": arm,
        "arbiter": arbiter_model,
        "dry_run": dry_run,
        "subjects": [subject.subject_sha256 for subject in subjects],
        "records": record_list,
        "replayed": len(replayed),
        "failed": len(failed),
        "skipped": skipped,
        "selected_new": len(selected_new),
        "adjudicated": len(adjudicated),
        "thompson": {
            "before": before_stats,
            "after": after_stats,
            "outcomes": [
                {"review_id": r.get("review_id"), **(r.get("thompson") or {})} for r in replayed
            ],
        },
        "comparison": comparison,
        "retrain": retrain_report,
        "exit_code": 1 if failures else 0,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay past review subjects with a new model to bootstrap its Thompson prior."
    )
    parser.add_argument("--model", required=True,
                        help="gateway model ID to bootstrap (e.g. rits/new-model-id)")
    parser.add_argument("--reviews", type=int, default=20,
                        help="number of most recent unique subjects to replay (default 20)")
    parser.add_argument("--arbiter", default=DEFAULT_ARBITER,
                        help=f"arbiter model or alias opus/sonnet/haiku (default {DEFAULT_ARBITER})")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the subjects that would be replayed; no API calls, no writes")
    parser.add_argument("--retrain-predictors", action="store_true", dest="retrain",
                        help="retrain the quality and routing predictors on the updated ledger")
    args = parser.parse_args(argv)
    if args.reviews < 1:
        parser.error("--reviews must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run(
        model=args.model,
        reviews=args.reviews,
        arbiter=args.arbiter,
        dry_run=args.dry_run,
        retrain=args.retrain,
    )
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
