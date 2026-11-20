#!/usr/bin/env python3
"""Replay historical reviews with a local Ollama model as the arbiter.

The review ledger records the worker outputs and the historical arbiter
decision for every review.  This tool takes the most recent review subjects,
gives the *same* worker outputs to a local model, and asks it to synthesize the
reviews and SELECT the strongest worker.  That choice is compared with the
historical arbiter's choice, and the local arbiter's outcome is fed back to
Thompson as an arbiter-arm result.

Historical-selection caveat
---------------------------
The live ledger's ``arbiter_decision`` rows predate the ``SELECTED:`` line, so
most have no explicit worker choice (see the review-ledger finding from
2026-11-19).  Baseline selection therefore has three sources:

``auto`` (default)
    Use the historical ``SELECTED:`` line when present; otherwise fall back to
    the documented citation proxy (which worker's review the historical
    arbiter referenced most often).  The proxy is labelled as such in the
    output and is not the same as an explicit selection.
``historic``
    Only explicit historical ``SELECTED:`` lines; subjects without one report
    ``insufficient_data`` and are excluded from the agreement rate.
``proxy``
    Always use the citation proxy.
``fresh``
    Ask the historical arbiter model (Opus by default) to make a fresh
    selection from the same historical worker outputs.  This costs money and is
    opt-in with ``--opus-source fresh``.

Usage
-----
    replay_arbiter.py [--reviews 10] [--model ollama/deepseek-r1:7b]
                      [--opus-source auto|historic|proxy|fresh]
                      [--baseline-model claude-opus-5-5]
                      [--min-workers 2] [--dry-run] [--no-record] [--json]

    replay_arbiter.py --reviews 3 --opus-source proxy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICES = os.path.join(REPO, "services")
SHARED = os.path.join(SERVICES, "shared")
for _path in (SERVICES, SHARED):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_MODEL = "ollama/deepseek-r1:7b"
DEFAULT_BASELINE_MODEL = "claude-opus-5-5"
DEFAULT_REVIEWS = 10
DEFAULT_MIN_WORKERS = 2
MAX_STORED_CHARS = 4000      # multi_model_review truncates arbiter input this way
DEFAULT_MAX_SOURCE_CHARS = 6000
# A local 7B model on CPU generates ~1 token/s. The replay only needs a brief
# synthesis plus two final lines, so cap the answer instead of Ollama's default
# 4000-token budget (which could run for an hour on one subject).
SELECT_MAX_TOKENS = 600
DEFAULT_TIMEOUT = 900.0

LEDGER = os.path.expanduser(os.environ.get(
    "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", "~/.FlossWare/ledger/reviews/reviews.jsonl"))
STATE = os.path.expanduser(os.environ.get(
    "POC_CLAUDE_ENSEMBLE_ARBITER_REPLAY_STATE",
    os.path.join(os.path.dirname(LEDGER), "arbiter-replay-state.json")))

VERDICTS = ("approve with fixes", "approve", "reject")


# ---------------------------------------------------------------------------
# Ledger extraction
# ---------------------------------------------------------------------------

@dataclass
class Review:
    review_id: str
    subject: Optional[str]
    subject_sha256: Optional[str]
    truncated: bool
    recorded_at: Optional[str]
    workers: list[dict[str, Any]] = field(default_factory=list)
    arbiter_text: str = ""
    arbiter_model: Optional[str] = None

    @property
    def usable_workers(self) -> list[dict[str, Any]]:
        return [w for w in self.workers if (w.get("text") or "").strip() and not w.get("error")]


def load_rows(path: str = LEDGER) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def group_reviews(rows: Iterable[dict[str, Any]]) -> list[Review]:
    """Group ``multi_model_review`` rows by review_id, preserving ledger order."""
    grouped: dict[str, Review] = {}
    for row in rows:
        review_id = row.get("review_id")
        kind = row.get("kind")
        if not review_id or kind not in {"worker_result", "arbiter_decision"}:
            continue
        review = grouped.get(review_id)
        if review is None:
            review = Review(
                review_id=str(review_id),
                subject=row.get("subject"),
                subject_sha256=row.get("subject_sha256"),
                truncated=bool(row.get("truncated")),
                recorded_at=row.get("recorded_at"),
            )
            grouped[review_id] = review
        if kind == "worker_result":
            review.workers.append({
                "model": row.get("model") or "unknown",
                "text": row.get("text") or "",
                "error": row.get("error"),
                "stop_reason": row.get("stop_reason"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "cost_usd": row.get("cost_usd"),
            })
        elif kind == "arbiter_decision":
            # Keep the last arbiter row (depth-3 verification can append a
            # second decision); use its text and model as the historical call.
            review.arbiter_text = row.get("text") or ""
            review.arbiter_model = row.get("model")
    return list(grouped.values())


def select_reviews(reviews: Sequence[Review], limit: int = DEFAULT_REVIEWS,
                   min_workers: int = DEFAULT_MIN_WORKERS,
                   require_arbiter: bool = True) -> list[Review]:
    """Most recent unique subjects with enough usable worker output.

    "Unique" is by ``subject_sha256`` when present, else by subject path, so a
    re-review of the same file version is not replayed twice.  Returned in
    ledger (chronological) order.
    """
    chosen: list[Review] = []
    seen: set[str] = set()
    for review in reversed(list(reviews)):
        if len(review.usable_workers) < min_workers:
            continue
        if require_arbiter and not review.arbiter_text.strip():
            continue
        key = review.subject_sha256 or review.subject or review.review_id
        if key in seen:
            continue
        seen.add(key)
        chosen.append(review)
        if len(chosen) >= limit:
            break
    chosen.reverse()
    return chosen


# ---------------------------------------------------------------------------
# Subject source code
# ---------------------------------------------------------------------------

def _truncate_source(source: str, max_chars: int) -> tuple[str, bool]:
    if len(source) <= max_chars:
        return source, False
    cut = source.rfind("\n", 0, max_chars)
    return (source[: cut + 1] if cut >= 0 else source[:max_chars]), True


def read_subject_source(review: Review, max_chars: int = DEFAULT_MAX_SOURCE_CHARS) -> tuple[Optional[str], str]:
    """Return ``(source, reason)``; source is None when it cannot be trusted."""
    path = review.subject
    if not path:
        return None, "no subject path recorded"
    if not os.path.isfile(path):
        return None, "subject file not available"
    try:
        with open(path, encoding="utf-8") as handle:
            full = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"subject unreadable: {type(exc).__name__}"
    source, _ = _truncate_source(full, max_chars)
    if review.subject_sha256:
        actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual != review.subject_sha256:
            return None, "subject changed since the review (sha256 mismatch)"
    return source, "ok"


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------

def _worker_block(index: int, worker: dict[str, Any]) -> str:
    text = worker.get("text") or ""
    flags: list[str] = []
    if worker.get("error"):
        flags.append(f"worker call failed: {worker['error']}")
    elif not text.strip():
        flags.append("empty output")
    if worker.get("stop_reason") == "max_tokens":
        flags.append("hit max_tokens; may be cut off")
    if len(text) > MAX_STORED_CHARS:
        flags.append(f"truncated to {MAX_STORED_CHARS} of {len(text)} chars")
        text = text[:MAX_STORED_CHARS] + "\n[truncated]"
    flag_str = f" [{'; '.join(flags)}]" if flags else ""
    return f"\n--- Review {index} ({worker.get('model', 'unknown')}){flag_str} ---\n{text}"


def build_prompt(review: Review, source: Optional[str], source_reason: str) -> str:
    """Arbiter prompt: same worker evidence, but ask for an explicit choice."""
    workers = review.workers
    parts = [
        f"You are adjudicating {len(workers)} independent reviews of "
        f"{os.path.basename(review.subject or 'the subject')}. Decide which review is the "
        "strongest and most accurate, judging against the code where it is available. "
        "Do not defer to length or confidence; prefer the review whose concrete claims hold up.",
    ]
    if source is not None:
        parts.append("The code under review is included at the end.")
    else:
        parts.append(f"The original code is not available ({source_reason}); judge the reviews on "
                     "internal consistency and specificity alone.")
    for index, worker in enumerate(workers, 1):
        parts.append(_worker_block(index, worker))
    if source is not None:
        parts.append("\n--- Code ---\n```\n" + source + "\n```")
    models = ", ".join(worker.get("model", "unknown") for worker in workers)
    parts.append(
        "\nKeep the synthesis to at most five sentences. End with exactly these two "
        "lines and nothing after them:\n"
        f"SELECTED: <one exact model ID from: {models}>\n"
        "VERDICT: approve, VERDICT: approve with fixes, or VERDICT: reject"
    )
    return "\n".join(parts)


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?(?:</think\s*>|$)", re.IGNORECASE | re.DOTALL)
_SELECTED_LINE = re.compile(
    r"^\s*[*_`#]*\s*SELECTED\s*[*_`]*\s*:\s*[*_`]*\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)
_VERDICT_LINE = re.compile(
    r"^\s*[*_`#]*\s*VERDICT\s*[*_`]*\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)
_REVIEW_REF = re.compile(r"\b(?:reviews?|workers?|r)\s*#?\s*(\d+)\b", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop reasoning-model `` thinking`` blocks before parsing the answer."""
    return _THINK_BLOCK.sub("", text or "")


def parse_selected(text: str, workers: Sequence[dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Map the last ``SELECTED:`` line to a worker model.

    Returns ``(model, raw_value)``.  Accepts an exact/partial model ID,
    ``Review N``, ``Worker N``, or ``R N``.  ``model`` is None when the value
    cannot be resolved to one of the supplied workers.
    """
    clean = strip_thinking(text or "")
    matches = _SELECTED_LINE.findall(clean)
    if not matches:
        return None, None
    raw = matches[-1].strip().strip("*_` ")
    models = [str(worker.get("model", "")) for worker in workers]

    lowered = raw.lower()
    for model in models:
        if lowered == model.lower():
            return model, raw
    # Review/worker number.
    number = _REVIEW_REF.search(raw)
    if number:
        index = int(number.group(1))
        if 1 <= index <= len(models):
            return models[index - 1], raw
    # Unique partial model match (e.g. "GLM" or "qwen2.5-coder"), including
    # display names whose spaces/punctuation differ from the model ID
    # ("Muse Spark Open" vs "Muse-Spark-Open"). Require a few characters so a
    # stray letter cannot select a worker.
    compact_raw = _compact(raw)
    partial = [model for model in models if lowered and lowered in model.lower()]
    if not partial and len(compact_raw) >= 3:
        partial = [model for model in models
                   if compact_raw in _compact(model)]
    if len(partial) == 1:
        return partial[0], raw
    return None, raw


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def verdict_of(text: str) -> str:
    matches = _VERDICT_LINE.findall(strip_thinking(text or ""))
    if not matches:
        return "unclear"
    value = matches[-1].strip().strip("*_` ").lower().rstrip(".")
    return value if value in VERDICTS else "unclear"


def citation_proxy(text: str, workers: Sequence[dict[str, Any]]) -> tuple[Optional[str], dict[str, int]]:
    """Documented proxy for a historical arbiter that did not write SELECTED.

    Count how often the arbiter refers to each review ("Review 2", "R1",
    "worker 3").  The most-referenced worker is the proxy choice.  This is
    attention, not an explicit selection, and is always labelled ``proxy``.
    """
    counts = {str(worker.get("model", "")): 0 for worker in workers}
    for match in _REVIEW_REF.finditer(strip_thinking(text or "")):
        index = int(match.group(1))
        if 1 <= index <= len(workers):
            counts[str(workers[index - 1].get("model", ""))] += 1
    if not counts or max(counts.values()) == 0:
        return None, counts
    best = max(counts, key=lambda model: (counts[model], -list(counts).index(model)))
    return best, counts


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _default_model_client():
    from arbitration.api_client import MultiModelClient
    from providers.ollama import OllamaProvider
    from providers.registry import ProviderRegistry

    # 8192 tokens fit the largest replay prompt (three worker outputs plus the
    # subject code) and the bounded answer; Ollama's 4096 default truncates it.
    return MultiModelClient(
        registry=ProviderRegistry({"ollama": OllamaProvider(num_ctx=8192)}),
        task_name="arbiter-replay")


def call_arbiter(client: Any, model: str, prompt: str, *, temperature: float = 0.0,
                 timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    response = client.call_model_response(
        model=model, prompt=prompt, temperature=temperature,
        max_tokens=SELECT_MAX_TOKENS, timeout=timeout,
    )
    return {
        "model": model,
        "text": response.text,
        "latency_ms": round(response.latency_ms or 0.0),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "provider": response.provider,
    }


def _default_baseline_call(model: str, prompt: str) -> dict[str, Any]:
    """Fresh historical-arbiter call through the review tool's gateway path."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import multi_model_review

    return multi_model_review.call(model, prompt)


# ---------------------------------------------------------------------------
# Thompson state
# ---------------------------------------------------------------------------

def load_processed() -> set[str]:
    if not os.path.exists(STATE):
        return set()
    try:
        with open(STATE, encoding="utf-8") as handle:
            return set(json.load(handle).get("keys", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_processed(keys: Iterable[str]) -> None:
    directory = os.path.dirname(STATE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"keys": sorted(keys)}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def feed_thompson(results: Sequence[dict[str, Any]], client: Any, processed: set[str]) -> dict[str, Any]:
    """Record one arbiter-arm outcome per replay (plus an agreement bonus).

    ``success`` means the local arbiter produced a usable decision (a parseable
    SELECTED line); agreement with the baseline is an additional positive
    signal.  Request ids are prefixed ``replay-arbiter-`` so these are
    distinguishable from organic outcomes.  Returns counters.  A failed
    Thompson write leaves the key unprocessed so a later run retries it.
    """
    recorded = 0
    bonuses = 0
    skipped = 0
    failed = 0
    for result in results:
        key = f"{result['model']}|{result['review_id']}"
        if key in processed:
            skipped += 1
            continue
        if not result.get("local_selected"):
            skipped += 1
            continue
        tokens = int(result.get("input_tokens") or 0) + int(result.get("output_tokens") or 0)
        ok = client.record_outcome(
            model=result["model"], task_type="code-review", success=True,
            cost=0.0, tokens=tokens,
            request_id=f"replay-arbiter-{result['review_id']}",
        )
        if not ok:
            failed += 1
            continue
        recorded += 1
        if result.get("agreement") is True:
            bonus_ok = client.record_outcome(
                model=result["model"], task_type="code-review", success=True,
                cost=0.0, tokens=0,
                request_id=f"replay-arbiter-{result['review_id']}-agree",
            )
            if bonus_ok:
                bonuses += 1
        processed.add(key)
    return {"recorded": recorded, "agreement_bonuses": bonuses,
            "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(*, model: str = DEFAULT_MODEL, limit: int = DEFAULT_REVIEWS,
           min_workers: int = DEFAULT_MIN_WORKERS,
           opus_source: str = "auto", baseline_model: str = DEFAULT_BASELINE_MODEL,
           ledger: str = LEDGER, dry_run: bool = False, record: bool = True,
           require_source: bool = False,
           model_client: Any = None, thompson: Any = None,
           baseline_call: Optional[Callable[[str, str], dict[str, Any]]] = None,
           max_source_chars: int = DEFAULT_MAX_SOURCE_CHARS) -> dict[str, Any]:
    if opus_source not in {"auto", "historic", "proxy", "fresh"}:
        raise ValueError(f"invalid opus_source: {opus_source!r}")

    reviews = group_reviews(load_rows(ledger))
    # With --require-source, scan every eligible subject and keep the most
    # recent ones whose code still matches the ledger, so both arbiters judged
    # the same code rather than stopping at the first source-less subjects.
    scan_limit = len(reviews) if require_source else limit
    selected = select_reviews(reviews, limit=scan_limit, min_workers=min_workers)
    if require_source:
        available = [review for review in selected
                     if read_subject_source(review, max_source_chars)[0] is not None]
        selected = available[-limit:]
    summary: dict[str, Any] = {
        "model": model,
        "ledger": ledger,
        "requested": limit,
        "considered": len(selected),
        "baseline_model": baseline_model,
        "opus_source": opus_source,
        "dry_run": dry_run,
        "results": [],
        "agreement": None,
    }
    if dry_run:
        summary["dry_run_plan"] = [
            {"review_id": review.review_id, "subject": review.subject,
             "workers": [w.get("model") for w in review.workers],
             "historical_arbiter": review.arbiter_model}
            for review in selected
        ]
        return summary

    client = model_client if model_client is not None else _default_model_client()
    thompson_client = thompson if thompson is not None else (None if not record else _thompson_client())
    processed = load_processed()

    for review in selected:
        source, source_reason = read_subject_source(review, max_source_chars)
        prompt = build_prompt(review, source, source_reason)
        result: dict[str, Any] = {
            "review_id": review.review_id,
            "model": model,
            "subject": review.subject,
            "source_status": source_reason,
            "workers": [w.get("model") for w in review.workers],
            "usable_workers": len(review.usable_workers),
            "historical_arbiter": review.arbiter_model,
        }
        try:
            local = call_arbiter(client, model, prompt)
        except Exception as exc:  # noqa: BLE001 - one failed subject must not abort the replay
            result.update({"error": f"{type(exc).__name__}: {exc}",
                           "local_selected": None, "agreement": None})
            summary["results"].append(result)
            continue

        local_selected, local_raw = parse_selected(local["text"], review.workers)
        result.update({
            "local_model": local["model"],
            "local_selected": local_selected,
            "local_selected_raw": local_raw,
            "local_verdict": verdict_of(local["text"]),
            "input_tokens": local.get("input_tokens"),
            "output_tokens": local.get("output_tokens"),
            "cost_usd": local.get("cost_usd"),
            "latency_ms": local.get("latency_ms"),
        })

        # Historical baseline choice.
        baseline_selected: Optional[str] = None
        baseline_source = "none"
        historical_selected, historical_raw = parse_selected(
            review.arbiter_text, review.workers)
        proxy_selected, citation_counts = citation_proxy(
            review.arbiter_text, review.workers)
        result["historical_selected"] = historical_selected
        result["historical_selected_raw"] = historical_raw
        result["proxy_selected"] = proxy_selected
        result["citation_counts"] = citation_counts
        if proxy_selected:
            top = citation_counts.get(proxy_selected, 0)
            tied = [model for model, count in citation_counts.items() if count == top and count > 0]
            result["proxy_tied"] = len(tied) > 1
            result["proxy_tie_models"] = tied if len(tied) > 1 else []

        if opus_source in {"auto", "historic"} and historical_selected:
            baseline_selected, baseline_source = historical_selected, "historic"
        elif opus_source in {"auto", "proxy"}:
            baseline_selected, baseline_source = proxy_selected, "proxy"
        elif opus_source == "fresh":
            try:
                fresh = (baseline_call or _default_baseline_call)(baseline_model, prompt)
                baseline_selected, fresh_raw = parse_selected(
                    fresh.get("text") or "", review.workers)
                result["fresh_baseline_raw"] = fresh_raw
                result["fresh_baseline_model"] = baseline_model
                baseline_source = "fresh"
            except Exception as exc:  # noqa: BLE001 - baseline failure is reported, not fatal
                result["fresh_baseline_error"] = f"{type(exc).__name__}: {exc}"
                baseline_source = "fresh_failed"

        result["baseline_selected"] = baseline_selected
        result["baseline_source"] = baseline_source
        result["agreement"] = (
            None if local_selected is None or baseline_selected is None
            else local_selected == baseline_selected
        )
        summary["results"].append(result)

    comparable = [r for r in summary["results"] if r.get("agreement") is not None]
    agreed = sum(1 for r in comparable if r["agreement"])
    summary["agreement"] = {
        "compared": len(comparable),
        "agreed": agreed,
        "rate": round(agreed / len(comparable), 4) if comparable else None,
        "baseline_sources": sorted({r.get("baseline_source") for r in comparable}),
    }

    if thompson_client is not None and record:
        summary["thompson"] = feed_thompson(summary["results"], thompson_client, processed)
        save_processed(processed)
    else:
        summary["thompson"] = {"recorded": 0, "agreement_bonuses": 0,
                               "skipped": len(summary["results"]), "disabled": True}
    return summary


def _thompson_client():
    from thompson_client import ThompsonClient

    return ThompsonClient()


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------

def format_summary(summary: dict[str, Any]) -> str:
    if summary.get("dry_run"):
        lines = [f"dry run: would replay {summary['considered']} subject(s) "
                 f"with {summary['model']} as arbiter"]
        for item in summary.get("dry_run_plan", []):
            lines.append(f"  {os.path.basename(item['subject'] or '?')} "
                         f"[{item['review_id']}] workers={len(item['workers'])} "
                         f"historical_arbiter={item['historical_arbiter']}")
        return "\n".join(lines)

    lines = [f"arbiter replay: {summary['model']} vs {summary['baseline_model']} "
             f"(baseline source: {summary['opus_source']})",
             f"subjects: {summary['considered']} of {summary['requested']} requested"]
    for index, result in enumerate(summary["results"], 1):
        name = os.path.basename(result.get("subject") or "?")
        lines.append(f"  {index}. {name} [{result['review_id']}] "
                     f"workers={result['usable_workers']}/{len(result['workers'])} "
                     f"source={result['source_status']}")
        if result.get("error"):
            lines.append(f"     local arbiter FAILED: {result['error']}")
            continue
        lines.append(f"     local   SELECTED: {result.get('local_selected')} "
                     f"({result.get('local_verdict')}, {result.get('latency_ms')} ms)")
        tie = (" (citation tie: " + ", ".join(result["proxy_tie_models"]) + ")"
               if result.get("proxy_tied") else "")
        lines.append(f"     baseline SELECTED: {result.get('baseline_selected')} "
                     f"[{result.get('baseline_source')}]{tie}")
        agreement = result.get("agreement")
        lines.append(f"     agreement: {'yes' if agreement else 'no' if agreement is False else 'n/a'}")
    stats = summary.get("agreement") or {}
    if stats.get("rate") is None:
        lines.append("agreement: insufficient data (no comparable selections)")
    else:
        lines.append(f"agreement: {stats['agreed']}/{stats['compared']} "
                     f"({stats['rate'] * 100:.1f}%) "
                     f"(sources: {', '.join(stats['baseline_sources'])})")
    thompson = summary.get("thompson") or {}
    disabled = " (disabled)" if thompson.get("disabled") else ""
    lines.append(f"thompson: recorded={thompson.get('recorded', 0)} "
                 f"agreement_bonuses={thompson.get('agreement_bonuses', 0)} "
                 f"skipped={thompson.get('skipped', 0)} "
                 f"failed={thompson.get('failed', 0)}{disabled}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay historical reviews with a local Ollama arbiter and compare "
                    "its SELECTED choices with the historical arbiter.")
    parser.add_argument("--reviews", type=int, default=DEFAULT_REVIEWS,
                        help=f"number of recent subjects to replay (default {DEFAULT_REVIEWS})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"local arbiter model (default {DEFAULT_MODEL})")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL,
                        help=f"historical arbiter model for --opus-source fresh "
                             f"(default {DEFAULT_BASELINE_MODEL})")
    parser.add_argument("--opus-source", choices=("auto", "historic", "proxy", "fresh"),
                        default="auto",
                        help="where the baseline choice comes from (default auto)")
    parser.add_argument("--min-workers", type=int, default=DEFAULT_MIN_WORKERS,
                        help=f"minimum usable worker outputs per subject (default {DEFAULT_MIN_WORKERS})")
    parser.add_argument("--require-source", action="store_true",
                        help="only replay subjects whose file still hashes to the ledger's "
                             "subject_sha256 (same code for both arbiters)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the replay plan without calling any model or writing state")
    parser.add_argument("--no-record", action="store_true",
                        help="do not feed the local arbiter's outcomes to Thompson")
    parser.add_argument("--think", choices=("auto", "on", "off"), default="off",
                        help="local reasoning control; 'off' (default) stops CPU "
                             "reasoning models from spending the budget before SELECTED")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    if args.reviews < 1:
        parser.error("--reviews must be at least 1")
    if args.min_workers < 1:
        parser.error("--min-workers must be at least 1")

    # The provider reads this at construction time (lazily, during the first
    # call), so the flag must be applied before replay() builds the client.
    from providers.ollama import OLLAMA_THINK_ENV

    if args.think == "off":
        os.environ[OLLAMA_THINK_ENV] = "0"
    elif args.think == "on":
        os.environ[OLLAMA_THINK_ENV] = "1"
    else:
        os.environ.pop(OLLAMA_THINK_ENV, None)

    summary = replay(model=args.model, limit=args.reviews, min_workers=args.min_workers,
                     opus_source=args.opus_source, baseline_model=args.baseline_model,
                     require_source=args.require_source,
                     dry_run=args.dry_run, record=not args.no_record)
    print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
