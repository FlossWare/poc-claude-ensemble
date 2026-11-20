#!/usr/bin/env python3
"""User-satisfaction signal: classify the message a user sends *after* a response.

Claude Ensemble records plenty of per-review outcomes, but nothing captures
whether the person on the other end was satisfied with a response.  This module
fills that gap with a deterministic, model-free classifier plus a ledger and
three consumers:

1. **Classification** -- :func:`classify` scores the next user message with
   regular-expression / keyword patterns into ``SATISFIED``, ``CORRECTION``,
   or ``UNCLEAR``.  Every pattern is documented in :data:`PATTERNS` (and by
   ``python3 tools/satisfaction_signal.py patterns``).
2. **Journal** -- one row per classified turn in
   ``~/.FlossWare/ledger/satisfaction/satisfaction.jsonl`` (override with
   ``POC_CLAUDE_ENSEMBLE_SATISFACTION_LEDGER``), mode 0600, append-once.
3. **RAG linkage** -- the row records the injection-journal row
   (``~/.FlossWare/ledger/rag/injections.jsonl``) whose prompt produced the
   response being judged, matched by timestamp proximity within 60 s.  The
   ``rag-outcomes`` command turns that join into a derived ledger carrying the
   ``helped`` label that ``learning.train.retrieval_relevance`` (GA Target B's
   training signal) already knows how to read.
4. **Thompson** -- ``SATISFIED`` / ``CORRECTION`` become positive / negative
   outcomes for the arm that produced the judged response.  ``UNCLEAR`` and
   low-confidence rows are journaled but not fed to Thompson.

The hook (``Stop`` recommended, ``PostToolUse`` also supported) is fail-open:
every error exits 0 with a one-line stderr message, it never writes to stdout,
it reads only a bounded tail of the transcript, and it performs no network
I/O.  Thompson recording happens in a detached child process, so the hook
stays well under 100 ms.

Install (deliberate, into a profile; this repo does not modify settings):

  {"hooks": {"Stop": [{"hooks": [{"type": "command",
     "command": "python3 /path/to/tools/satisfaction_signal.py hook"}]}]}}

Design notes
------------

* **Which message is classified?**  The hook runs at ``Stop`` of turn *N*.
  The transcript then holds ``... assistant(N-1), user(N), assistant(N)``.
  ``user(N)`` is the first thing the user said *after* ``assistant(N-1)``, so
  it is the satisfaction signal for the response that ``assistant(N-1)``
  produced.  Its model is attributed to that response.
* **Which injection is linked?**  The injection journal row is written when a
  prompt is submitted, so the context that produced response *N-1* was
  journaled next to ``user(N-1)``.  The satisfaction row therefore links to
  the injection nearest to the *previous* human prompt timestamp (within
  ``RAG_LINK_WINDOW_SECONDS``).  Linking to the classified message's own
  injection would be one turn ahead -- it conditions the response that has not
  been judged yet.
* **No text is stored.**  Rows carry lengths, pattern ids, UUIDs, model ids and
  injection metadata only; never the user message, the response, or retrieved
  context.
* **Corrections count against a model here**, which reverses the earlier
  ``services/embeddings/turn_outcomes.py`` decision to keep keyword pushback
  out of ``success``.  That decision was about *turn success*; this signal is
  an explicit, confidence-gated quality observation.  The keyword detector is
  not precise enough to count every ambiguous reply, so a correction scores a
  model only when it is confident (default 0.8), not a question, and names a
  concrete problem (see :func:`thompson_eligible`).  A bare leading "no" stays
  journaled but never counts.
* **Harness automation is not the user.**  Subagent hand-backs, task
  notifications, reminders, and command wrappers are skipped in
  ``_human_text``; model output must never become satisfaction feedback.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
PRODUCER = "satisfaction-signal"
KIND = "satisfaction"
SYNC_KIND = "satisfaction_thompson_sync"
RAG_OUTCOME_KIND = "rag_outcome"

SATISFIED = "SATISFIED"
CORRECTION = "CORRECTION"
UNCLEAR = "UNCLEAR"

DEFAULT_LEDGER = "~/.FlossWare/ledger/satisfaction/satisfaction.jsonl"
LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_LEDGER"
DEFAULT_SYNC_LEDGER = "~/.FlossWare/ledger/satisfaction/thompson-sync.jsonl"
SYNC_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_SYNC_LEDGER"
DEFAULT_RAG_OUTCOMES = "~/.FlossWare/ledger/satisfaction/rag-outcomes.jsonl"
RAG_OUTCOMES_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_RAG_OUTCOMES"
DEFAULT_RAG_JOURNAL = "~/.FlossWare/ledger/rag/injections.jsonl"
RAG_JOURNAL_ENV = "FLOSSWARE_CE_RAG_JOURNAL"
THOMPSON_URL_ENV = "POC_CLAUDE_ENSEMBLE_THOMPSON_URL"
DEFAULT_THOMPSON_URL = "http://127.0.0.1:8769"
THOMPSON_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_THOMPSON"
MIN_CONFIDENCE_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_MIN_CONFIDENCE"
DEFAULT_MIN_CONFIDENCE = 0.6
CORRECTION_MIN_CONFIDENCE_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_CORRECTION_MIN_CONFIDENCE"
DEFAULT_CORRECTION_MIN_CONFIDENCE = 0.8
RAG_WINDOW_ENV = "POC_CLAUDE_ENSEMBLE_SATISFACTION_RAG_WINDOW"
RAG_LINK_WINDOW_SECONDS = 60.0
TASK_TYPE = "satisfaction_signal"
MAX_ROW_BYTES = 16 * 1024
MAX_STDIN_BYTES = 1024 * 1024
#: Bounded tail read: a Stop hook must not parse a 40 MB transcript.
TRANSCRIPT_TAIL_BYTES = 1024 * 1024
#: Bounded tail scan for append-once / injection lookup.
JOURNAL_TAIL_BYTES = 256 * 1024

INTERRUPT_MARKER = "[Request interrupted by user"
_TOO_SHORT_CHARS = 2
_NEUTRAL_CONTINUATION_CHARS = 15
#: A long message is usually an instruction or a pasted log, where reaction
#: words appear incidentally.  For those, reaction rules only see the opening.
_LONG_MESSAGE_CHARS = 200
_LONG_MESSAGE_WINDOW = 120
_RESTATEMENT_JACCARD = 0.75
_CONFLICT_MARGIN = 0.15

#: Satisfaction rules strong enough to survive a later incidental reaction word.
_STRONG_SATISFACTION = frozenset(
    {"thanks", "praise", "approval", "proceed", "no_problem_guard", "bare_good", "we_good"}
)
#: Contrastive markers mean the user is taking the praise back.
_QUESTION_OPENER = re.compile(
    r"^(?:what|whats|what'?s|why|how|when|where|who|which|did|does|is|are|"
    r"was|were|can|could|should|would|will)\b",
    re.IGNORECASE,
)
_CONTRAST = re.compile(
    r"\b(?:but|however|though|although|except|still|one\s+issue|one\s+problem|"
    r"only\s+issue|only\s+problem)\b",
    re.IGNORECASE,
)
#: Correction rules that are unambiguous even after opening praise.
_UNAMBIGUOUS_CORRECTION = frozenset(
    {"interrupted", "not_that", "meant", "missed", "already_asked", "not_ok",
     "confusion", "frustration"}
)

_REQUEST_MARKER = re.compile(
    r"\b(?:please|resolve|fix|run|add|update|change|remove|use|make|do|"
    r"re-?run|retry|try)\b",
    re.IGNORECASE,
)

#: Messages the harness injects that are not the human: task notifications,
#: subagent hand-backs, reminders.  They must never become a satisfaction row
#: (model output is not user feedback).
_AUTOMATED_PREFIXES = (
    "another claude session sent a message",
    "<agent-message",
    "<task-notification",
    "<system-reminder",
    "<command-name",
    "<local-command",
    "caveat: the messages below were generated",
)

_STOPWORDS = frozenset("""
a an and are as at be been but by can could do does did for from had has have
how i if in into is it its just me my no not of on or our please should so
some that the their them then there these they this to too us was we were what
when where which who why will with would you your
""".split())


@dataclass(frozen=True)
class Pattern:
    """One documented classification rule."""

    id: str
    classification: str
    weight: float
    regex: re.Pattern
    description: str
    guard: bool = False
    #: Only search the first N characters.  Reaction words ("failed", "wrong")
    #: are reliable near the start of a reply but appear incidentally in long
    #: pasted instructions and logs.
    window: int | None = None


# ---------------------------------------------------------------------------
# Pattern table.  ``weight`` is the rule's standalone confidence; the
# classifier picks the strongest rule for the winning class.
# ---------------------------------------------------------------------------

PATTERNS: tuple[Pattern, ...] = (
    # ---- SATISFIED ------------------------------------------------------
    Pattern(
        "interrupted",
        CORRECTION,
        0.95,
        re.compile(re.escape(INTERRUPT_MARKER), re.IGNORECASE),
        "The transcript marks the turn as interrupted by the user.",
    ),
    Pattern(
        "thanks",
        SATISFIED,
        0.92,
        re.compile(r"\b(?:thanks|thank\s+you|thx)\b", re.IGNORECASE),
        "Explicit gratitude: 'thanks', 'thank you', 'thx'.",
    ),
    Pattern(
        "praise",
        SATISFIED,
        0.92,
        re.compile(
            r"\b(?:perfect|great|excellent|awesome|amazing|fantastic|brilliant|"
            r"wonderful|nice\s+work|good\s+job|well\s+done|love\s+it)\b",
            re.IGNORECASE,
        ),
        "Explicit praise: 'perfect', 'great', 'nice work', 'love it', ...",
    ),
    Pattern(
        "approval",
        SATISFIED,
        0.85,
        re.compile(
            r"\b(?:lgtm|looks\s+good(?:\s+to\s+me)?|works\s+for\s+me|"
            r"that\s+works|this\s+works|makes\s+sense|got\s+it)\b",
            re.IGNORECASE,
        ),
        "Approval idiom: 'lgtm', 'looks good', 'that works', 'makes sense'.",
    ),
    Pattern(
        "affirm_opener",
        SATISFIED,
        0.80,
        re.compile(
            r"^(?:yes|yeah|yep|yup|sure|okay|ok|alright|correct|exactly|right|"
            r"agreed|indeed|fine)\b",
            re.IGNORECASE,
        ),
        "Affirmation at the start: 'yes', 'ok', 'sure', 'correct', 'fine'.",
    ),
    Pattern(
        "proceed",
        SATISFIED,
        0.80,
        re.compile(
            r"\b(?:proceed|continue|go\s+ahead|carry\s+on|keep\s+going|"
            r"move\s+on|next|do\s+it|please\s+do|go\s+for\s+it|ship\s+it|"
            r"sounds\s+good|will\s+do|on\s+it|roger)\b",
            re.IGNORECASE,
        ),
        "Continuation instruction: 'proceed', 'go ahead', 'next', 'do it'.",
    ),
    Pattern(
        "negated_proceed",
        SATISFIED,
        0.78,
        re.compile(
            r"^(?:no|nah|nope)[,\s]+(?:proceed|continue|go\s+ahead|carry\s+on)\b",
            re.IGNORECASE,
        ),
        "'nah, proceed' / 'no, continue': declining an option, not the response; "
        "suppresses the leading-'no' correction.",
        guard=True,
    ),
    Pattern(
        "we_good",
        SATISFIED,
        0.70,
        re.compile(r"\b(?:we'?re|we\s+are|we|all)\s+good\b", re.IGNORECASE),
        "Reassurance: 'we good', 'all good'.",
        window=200,
    ),
    Pattern(
        "bare_good",
        SATISFIED,
        0.85,
        re.compile(r"^(?:good|great|nice|cool|awesome)[.!]?$", re.IGNORECASE),
        "A one-word approval: 'good', 'nice', 'cool'.",
    ),
    Pattern(
        "no_problem_guard",
        SATISFIED,
        0.85,
        re.compile(
            r"^(?:no\s+(?:problem|worries|need|rush)|not\s+bad|no\s+complaints)\b",
            re.IGNORECASE,
        ),
        "'no problem' / 'not bad' idioms; suppresses the leading-'no' correction.",
        guard=True,
    ),
    Pattern(
        "reassurance",
        SATISFIED,
        0.70,
        re.compile(r"\b(?:it'?s|its|that'?s)\s+(?:ok(?:ay)?|fine|good)\b",
                   re.IGNORECASE),
        "Reassurance: \"it's ok\", \"that's fine\".",
        window=200,
    ),
    Pattern(
        "imperative_continuation",
        SATISFIED,
        0.50,
        re.compile(
            r"^(?:do|run|apply|use|make|add|fix|update|include|remove|change|"
            r"check|verify|review)\b",
            re.IGNORECASE,
        ),
        "A new imperative instruction ('do them all', 'run the tests', "
        "'update the docs') is a continuation.  Below the Thompson floor.",
    ),
    Pattern(
        "question",
        SATISFIED,
        0.45,
        re.compile(r"\?\s*$"),
        "A question with no correction cue is a weak proxy for moving on to a "
        "new topic.  Low confidence, so it is journaled but not fed to Thompson.",
    ),
    # ---- CORRECTION -----------------------------------------------------
    Pattern(
        "negation_opener",
        CORRECTION,
        0.88,
        re.compile(r"^(?:no|nope|nah|wrong|incorrect)\b", re.IGNORECASE),
        "Message opens with 'no', 'nope', 'nah', 'wrong', 'incorrect'.",
    ),
    Pattern(
        "not_that",
        CORRECTION,
        0.92,
        re.compile(
            r"\b(?:not\s+that|that'?s\s+not|that\s+is\s+not|"
            r"not\s+what\s+(?:i|we|you)|isn'?t\s+what|wasn'?t\s+what)\b",
            re.IGNORECASE,
        ),
        "Disidentification: 'not that', \"that's not\", 'not what I ...'.",
        window=200,
    ),
    Pattern(
        "meant",
        CORRECTION,
        0.90,
        re.compile(
            r"\b(?:i\s+meant|i\s+said|i\s+asked|as\s+i\s+said|like\s+i\s+said|"
            r"what\s+i\s+mean(?:t)?)\b",
            re.IGNORECASE,
        ),
        "Explicit intent correction: 'I meant', 'I said', 'as I said'.",
        window=200,
    ),
    Pattern(
        "failure",
        CORRECTION,
        0.85,
        re.compile(
            r"(?:doesn'?t\s+work|does\s+not\s+work|didn'?t\s+work|"
            r"did\s+not\s+work|won'?t\s+work|not\s+working|"
            r"still\s+(?:broken|failing|fails?)|(?:it|that|this)\s+failed|"
            r"failed\s+again|didn'?t\s+make\s+it|"
            r"(?:the\s+)?(?:build|test|tests|pipeline|command|script|deploy|"
            r"deployment|job|run)\s+failed|"
            r"(?:build|test|pipeline|deploy)\s+failure|broken)",
            re.IGNORECASE,
        ),
        "A response/attempt failed: \"doesn't work\", 'still broken', "
        "'the test failed', 'it failed', 'build failure'.",
        window=200,
    ),
    Pattern(
        "failure_mention",
        CORRECTION,
        0.50,
        re.compile(r"\b(?:fails?|failed|failure|crash(?:ed|es)?)\b", re.IGNORECASE),
        "A bare mention of 'fail'/'failure'/'crash' -- often part of a new "
        "instruction or a question.  Below the Thompson floor.",
        window=200,
    ),
    Pattern(
        "wrong",
        CORRECTION,
        0.85,
        re.compile(
            r"\b(?:wrong|incorrect|mistaken|that'?s\s+a\s+mistake)\b",
            re.IGNORECASE,
        ),
        "Explicit error word: 'wrong', 'incorrect', 'mistake'.",
        window=200,
    ),
    Pattern(
        "retry",
        CORRECTION,
        0.82,
        re.compile(
            r"\b(?:try\s+again|re-?do|run\s+it\s+again|one\s+more\s+time|"
            r"start\s+over|try\s+once\s+more)\b",
            re.IGNORECASE,
        ),
        "Retry request: 'try again', 'redo', 'one more time', 'start over'.",
        window=200,
    ),
    Pattern(
        "missed",
        CORRECTION,
        0.88,
        re.compile(
            r"\b(?:you\s+missed|you\s+forgot|you\s+didn'?t|you\s+ignored|"
            r"missed\s+the|forgot\s+to)\b",
            re.IGNORECASE,
        ),
        "Omission accusation: 'you missed', 'you forgot', 'you didn't'.",
        window=200,
    ),
    Pattern(
        "stop_interjection",
        CORRECTION,
        0.85,
        re.compile(r"\bstop\s*[-\u2013\u2014,:]", re.IGNORECASE),
        "Mid-message halt before a clarification: 'OK stop - what I mean is ...'.",
        window=200,
    ),
    Pattern(
        "already_asked",
        CORRECTION,
        0.80,
        re.compile(
            r"\b(?:already\s+asked|i\s+(?:already\s+)?asked|asked\s+for\s+(?:it|this|that))\b",
            re.IGNORECASE,
        ),
        "Repeat-request reminder: 'I already asked', 'asked for it'.",
        window=200,
    ),
    Pattern(
        "frustration",
        CORRECTION,
        0.80,
        re.compile(
            r"\b(?:wtf|what\s+the\s+(?:hell|heck|fuck)|seriously|come\s+on)\b",
            re.IGNORECASE,
        ),
        "Frustration interjection: 'wtf', 'what the hell', 'seriously', "
        "'come on'.",
        window=200,
    ),
    Pattern(
        "confusion",
        CORRECTION,
        0.75,
        re.compile(
            r"\b(?:confused|i\s+don'?t\s+understand|not\s+sure\s+what|"
            r"what\s+are\s+you\s+(?:doing|building)|why\s+are\s+you)\b",
            re.IGNORECASE,
        ),
        "Reported confusion: 'I'm confused', \"I don't understand\", "
        "'what are you building'.",
        window=200,
    ),
    Pattern(
        "not_ok",
        CORRECTION,
        0.80,
        re.compile(
            r"\b(?:not|isn'?t|wasn'?t)\s+(?:ok(?:ay)?|fine|good)\b",
            re.IGNORECASE,
        ),
        "Negated reassurance: 'not ok', 'isn't fine', 'not good'.",
        window=200,
    ),
    Pattern(
        "wait",
        CORRECTION,
        0.55,
        re.compile(
            r"^(?:wait|hold\s+on|hang\s+on)\b|^stop\s*[-\u2013\u2014,:]|^stop\s*[.!]?$",
            re.IGNORECASE,
        ),
        "Halt interjection: 'wait', 'stop', 'hold on'.  Weak/ambiguous, so it "
        "is below the default Thompson floor.",
    ),
    Pattern(
        "actually",
        CORRECTION,
        0.55,
        re.compile(r"^(?:actually|instead|rather)\b", re.IGNORECASE),
        "'actually' / 'instead' at the start.  Weak because it is also a "
        "discourse filler before a new request; below the Thompson floor.",
    ),
)

def documented_patterns() -> list[dict]:
    """The pattern table as JSON-serialisable documentation."""
    return [
        {
            "id": pattern.id,
            "classification": pattern.classification,
            "weight": pattern.weight,
            "regex": pattern.regex.pattern,
            "description": pattern.description,
            "guard": pattern.guard,
            "window": pattern.window,
        }
        for pattern in PATTERNS
    ]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _result(classification: str, confidence: float, patterns: list[str],
            question: bool = False) -> dict:
    helpful = True if classification == SATISFIED else False if classification == CORRECTION else None
    return {
        "classification": classification,
        "confidence": round(max(0.0, min(0.95, confidence)), 2),
        "trigger_patterns": patterns,
        "response_was_helpful": helpful,
        "is_question": question,
    }


def classify(text: str, previous_text: str | None = None) -> dict:
    """Classify one user message.

    ``previous_text`` is the user message that prompted the response being
    judged; it enables the restatement rule.  Returns a dict with
    ``classification``, ``confidence``, ``trigger_patterns``,
    ``response_was_helpful`` and ``is_question``.
    """
    if not isinstance(text, str):
        return _result(UNCLEAR, 0.10, ["non_text"])
    stripped = text.strip()
    if len(stripped) < _TOO_SHORT_CHARS:
        return _result(UNCLEAR, 0.10, ["too_short"])

    normalized = re.sub(r"\s+", " ", stripped)
    is_question = bool(
        re.search(r"\?\s*$", normalized) or _QUESTION_OPENER.match(normalized)
    )
    # A numbered or bulleted reply is usually an answer or a list of
    # requirements, where "failed"/"wrong" are topics rather than reactions.
    is_list = bool(re.match(r"^\s*(?:\d+[.)]?|[-*\u2022])\s", normalized))

    matched: list[Pattern] = []
    guards: set[str] = set()
    long_message = len(normalized) > _LONG_MESSAGE_CHARS
    for pattern in PATTERNS:
        if is_list and pattern.id in {"failure", "wrong"}:
            continue
        window = pattern.window
        if window and long_message:
            window = min(window, _LONG_MESSAGE_WINDOW)
        haystack = normalized[:window] if window else normalized
        if pattern.regex.search(haystack):
            matched.append(pattern)
            if pattern.guard:
                guards.add(pattern.id)

    # "no problem" / "not bad" are satisfactions, not the leading "no" correction.
    if guards & {"no_problem_guard", "negated_proceed"}:
        matched = [p for p in matched if p.id != "negation_opener"]

    # Restating an earlier request (high lexical overlap) is a correction signal,
    # but only when the message reads like another request and is not a question:
    # short conversational follow-ups overlap heavily without being re-asks.
    if previous_text and not is_question:
        overlap = _jaccard(_tokens(normalized), _tokens(previous_text))
        if (overlap >= _RESTATEMENT_JACCARD and len(normalized) >= 12
                and _REQUEST_MARKER.search(normalized)):
            matched.append(
                Pattern(
                    "restated_request",
                    CORRECTION,
                    0.58,
                    re.compile(r".*"),
                    "High lexical overlap with the previous user request "
                    f"(Jaccard >= {_RESTATEMENT_JACCARD}); treated as a weak "
                    "re-ask.  The matched text is not stored.",
                )
            )

    if not matched:
        if is_question and len(normalized) >= 8:
            return _result(SATISFIED, 0.45, ["question"], question=True)
        if len(normalized) >= _NEUTRAL_CONTINUATION_CHARS:
            return _result(SATISFIED, 0.40, ["neutral_continuation"])
        return _result(UNCLEAR, 0.25, ["too_short"] if len(normalized) < 12 else ["no_signal"])

    correction = [p for p in matched if p.classification == CORRECTION]
    satisfied = [p for p in matched if p.classification == SATISFIED]
    best_correction = max(correction, key=lambda p: p.weight) if correction else None
    best_satisfied = max(satisfied, key=lambda p: p.weight) if satisfied else None

    # A question that merely mentions failure ("what's wrong with X?",
    # "did it fail again?") is asking for status, not registering pushback.
    correction_ids = {p.id for p in correction}
    if (is_question and correction_ids
            and correction_ids <= {"failure_mention", "wrong", "question"}):
        matched = [p for p in matched if p.classification == SATISFIED]
        if not matched:
            return _result(SATISFIED, 0.45, ["question"], question=True)
        best_correction = None

    # Opening praise plus a later incidental reaction word is still praise
    # unless the user contrasts it ("thanks, but ...").
    if best_correction is not None and best_satisfied is not None:
        opener = normalized[:40]
        strong_open = any(
            p.id in _STRONG_SATISFACTION and p.regex.search(opener)
            for p in satisfied
        )
        if (strong_open and not _CONTRAST.search(normalized)
                and best_correction.id not in _UNAMBIGUOUS_CORRECTION):
            best_correction = None

    if best_correction is None:
        chosen, other = best_satisfied, None
        winner = SATISFIED
    elif best_satisfied is None:
        chosen, other = best_correction, None
        winner = CORRECTION
    elif best_correction.weight >= 0.80:
        # An explicit failure beats a courtesy "yes" ("yes, but it doesn't work").
        chosen, other = best_correction, best_satisfied
        winner = CORRECTION
    elif best_satisfied.weight > best_correction.weight:
        chosen, other = best_satisfied, best_correction
        winner = SATISFIED
    elif best_correction.weight > best_satisfied.weight:
        chosen, other = best_correction, best_satisfied
        winner = CORRECTION
    else:
        patterns = sorted({p.id for p in matched})
        return _result(UNCLEAR, 0.35, patterns, question=is_question)

    confidence = chosen.weight
    conflicting = other is not None and abs(chosen.weight - other.weight) <= _CONFLICT_MARGIN
    if conflicting and not _CONTRAST.search(normalized):
        confidence -= 0.10
    patterns = sorted({p.id for p in matched})
    return _result(winner, confidence, patterns, question=is_question)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _seconds_between(left: datetime, right: datetime) -> float:
    return (left - right).total_seconds()


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


def _read_jsonl_tail(path: Path, max_bytes: int = JOURNAL_TAIL_BYTES) -> list[dict]:
    """Parse complete rows from a bounded tail; malformed rows are skipped."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    rows: list[dict] = []
    lines = data.split(b"\n")
    if start > 0 and lines:
        lines = lines[1:]  # first line may be a partial row
    for raw in lines:
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _existing_event_ids(path: Path) -> set[str]:
    return {
        str(row.get("event_id"))
        for row in _read_jsonl_tail(path)
        if row.get("event_id")
    }


def _append_row(path: Path, row: dict) -> bool:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if len(line.encode("utf-8")) > MAX_ROW_BYTES:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (line + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _event_id(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# RAG injection linkage
# ---------------------------------------------------------------------------


def _rag_journal_path(value: str | None = None) -> Path | None:
    raw = value if value is not None else os.environ.get(RAG_JOURNAL_ENV, DEFAULT_RAG_JOURNAL)
    if isinstance(raw, str) and raw.strip().lower() == "off":
        return None
    return Path(os.path.expanduser(raw or DEFAULT_RAG_JOURNAL))


def link_injection(
    anchor_ts: str,
    session_id: str,
    *,
    journal: Path | None = None,
    disabled: bool = False,
    window_seconds: float = RAG_LINK_WINDOW_SECONDS,
) -> tuple[dict | None, str]:
    """Find the injection row nearest ``anchor_ts`` for this session.

    Returns ``(link, status)`` where status is ``linked``,
    ``no_injection_in_window``, ``journal_unavailable`` or ``disabled``.
    Only metadata from the injection row is copied -- never injected text.
    """
    if disabled:
        return None, "disabled"
    path = journal if journal is not None else _rag_journal_path()
    if path is None:
        return None, "disabled"
    if not path.exists():
        return None, "journal_unavailable"
    anchor = _parse_ts(anchor_ts)
    if anchor is None:
        return None, "no_injection_in_window"
    best: tuple[float, dict, datetime] | None = None
    for row in _read_jsonl_tail(path):
        if row.get("kind") != "rag_injection":
            continue
        if session_id and str(row.get("session_id") or "") != session_id:
            continue
        when = _parse_ts(row.get("timestamp"))
        if when is None:
            continue
        delta = _seconds_between(when, anchor)
        if abs(delta) > window_seconds:
            continue
        if best is None or abs(delta) < abs(best[0]):
            best = (delta, row, when)
    if best is None:
        return None, "no_injection_in_window"
    delta, row, when = best
    link = {
        "event_id": row.get("event_id"),
        "timestamp": row.get("timestamp"),
        "delta_seconds": round(delta, 3),
        "direction": "before" if delta <= 0 else "after",
        "context_chars": row.get("context_chars"),
        "source_count": len(row.get("source_ids") or []),
        "truncated": row.get("truncated"),
    }
    return link, "linked"


# ---------------------------------------------------------------------------
# Thompson arm mapping / feeding
# ---------------------------------------------------------------------------

# Mirrors tools/ledger_to_thompson.py MODEL_ALIASES and
# services/embeddings/turn_outcomes.py arm_for(); kept local so the hook has no
# import path setup and no third-party imports.
MODEL_ALIASES = {
    "rits/zai-org/glm-5-3": "glm-5-3",
    "rits/deepseek-ai/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash",
    "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017": "deepseek-v4.1-flash",
    "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4": "muse-spark-open",
    "claude-opus-5-5": "opus",
    "claude-sonnet-5": "sonnet",
    "claude-sonnet-5-5": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    "claude-haiku-5-5": "haiku",
}


def thompson_arm(model: str | None) -> str | None:
    """Map a transcript model id to a Thompson arm, or ``None``."""
    if not isinstance(model, str) or not model or model.startswith("<"):
        return None
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    lowered = model.lower()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]
    for family in ("haiku", "sonnet", "opus"):
        if family in lowered:
            return family
    return None


def min_confidence() -> float:
    raw = os.environ.get(MIN_CONFIDENCE_ENV)
    if raw:
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    return DEFAULT_MIN_CONFIDENCE


def correction_min_confidence() -> float:
    raw = os.environ.get(CORRECTION_MIN_CONFIDENCE_ENV)
    if raw:
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    return DEFAULT_CORRECTION_MIN_CONFIDENCE


#: A correction is fed to Thompson only when it names a concrete failure or
#: intent mismatch.  A bare leading "no" and a restated request are journaled
#: but excluded: on real transcripts they are usually mid-task clarifications
#: or answers to a question, not dissatisfaction with the response.
THOMPSON_STRONG_CORRECTION = frozenset({
    "not_that", "meant", "failure", "wrong", "retry", "missed",
    "already_asked", "confusion", "frustration", "not_ok", "stop_interjection",
})


def thompson_eligible(row: dict) -> bool:
    """Whether one journal row is precise enough to score the model."""
    helpful = row.get("response_was_helpful")
    if not isinstance(helpful, bool):
        return False
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    if helpful:
        return confidence >= min_confidence()
    if confidence < max(min_confidence(), correction_min_confidence()):
        return False
    if row.get("is_question") is True:
        return False
    patterns = set(row.get("trigger_patterns") or [])
    return bool(patterns & THOMPSON_STRONG_CORRECTION)


def thompson_enabled() -> bool:
    raw = os.environ.get(THOMPSON_ENV)
    return not (isinstance(raw, str) and raw.strip().lower() in ("0", "false", "off", "no", "disabled"))


def thompson_url() -> str:
    return (os.environ.get(THOMPSON_URL_ENV) or DEFAULT_THOMPSON_URL).rstrip("/")


def _post_outcome(url: str, row: dict, timeout: float) -> bool:
    payload = {
        "action": "record_outcome",
        "model": row["thompson_arm"],
        "task_type": TASK_TYPE,
        "success": bool(row["response_was_helpful"]),
        "cost": 0.0,
        "tokens": 0,
        "request_id": row["event_id"],
    }
    request = urllib.request.Request(
        url + "/thompson",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(body, dict) and body.get("ok") is True


def sync_thompson(
    ledger: Path,
    sync_ledger: Path,
    *,
    url: str | None = None,
    dry_run: bool = False,
    timeout: float = 1.5,
    quiet: bool = False,
) -> dict:
    """Feed journaled SATISFIED/CORRECTION rows to Thompson once each.

    Rows below :func:`min_confidence`, ``UNCLEAR`` rows, and rows without a
    known Thompson arm are skipped.  A successful POST is recorded in the sync
    ledger so a later run never double-counts it.  A failed POST is not
    recorded, so the next run retries it.
    """
    summary = {"candidates": 0, "sent": 0, "failed": 0, "skipped": 0, "dry_run": dry_run}
    if not ledger.exists():
        return summary
    sent_ids = _existing_event_ids(sync_ledger)
    target = (url or thompson_url()).rstrip("/")

    lock_fd = None
    lock_path = sync_ledger.with_suffix(sync_ledger.suffix + ".lock")
    if not dry_run:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if lock_fd is not None:
                os.close(lock_fd)
            summary["skipped"] = 0
            summary["locked"] = True  # another sync process already holds the lock
            return summary

    try:
        for row in _read_jsonl_tail(ledger, max_bytes=8 * JOURNAL_TAIL_BYTES):
            if row.get("kind") != KIND:
                continue
            event_id = row.get("event_id")
            if not event_id or event_id in sent_ids:
                continue
            helpful = row.get("response_was_helpful")
            if not thompson_eligible(row):
                continue
            arm = thompson_arm(row.get("model"))
            if not arm:
                continue
            row = dict(row, thompson_arm=arm)
            summary["candidates"] += 1
            if dry_run:
                summary["sent"] += 1
                continue
            ok = _post_outcome(target, row, timeout)
            if ok:
                summary["sent"] += 1
                sent_ids.add(event_id)
                _append_row(
                    sync_ledger,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "producer": PRODUCER,
                        "kind": SYNC_KIND,
                        "event_id": event_id,
                        "sent_at": _utc_now(),
                        "model": row.get("thompson_arm"),
                        "success": helpful,
                        "request_id": event_id,
                    },
                )
            else:
                summary["failed"] += 1
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    if not quiet:
        print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# Transcript reading
# ---------------------------------------------------------------------------


def _human_text(message: dict) -> str | None:
    """Text typed by the user; ``None`` for tool results and automated input."""
    content = message.get("content")
    if isinstance(content, list):
        if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return None
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = " ".join(texts) if texts else None
    if not isinstance(content, str) or not content.strip():
        return None
    stripped = content.lstrip()
    lowered = stripped[:200].lower()
    if stripped.startswith("<") or lowered.startswith(_AUTOMATED_PREFIXES):
        # System reminders, task notifications, subagent hand-backs, wrappers.
        return None
    return content


def _tail_records(transcript_path: Path, max_bytes: int = TRANSCRIPT_TAIL_BYTES) -> list[dict]:
    try:
        with open(transcript_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if start > 0 and lines:
        lines = lines[1:]
    records: list[dict] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def extract_turn(transcript_path: Path) -> dict | None:
    """Locate the last human message, its response model, and the prior prompt.

    Returns ``None`` when the transcript has no human message preceded by an
    assistant response (the first turn of a session, or an unreadable file).
    """
    records = _tail_records(transcript_path)
    last_human: tuple[dict, str] | None = None
    previous_human: tuple[dict, str] | None = None
    assistant: tuple[dict, str] | None = None

    for record in reversed(records):
        if record.get("isSidechain") is True or record.get("isMeta") is True:
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = _human_text(message)
            if text is None:
                continue
            if last_human is None:
                last_human = (record, text)
            elif previous_human is None:
                previous_human = (record, text)
                break
            continue
        if role == "assistant" and last_human is not None and assistant is None:
            model = message.get("model")
            if isinstance(model, str) and model and not model.startswith("<"):
                assistant = (record, model)

    if last_human is None or assistant is None:
        return None
    record, text = last_human
    assistant_record, model = assistant
    return {
        "message": text,
        "message_uuid": str(record.get("uuid") or ""),
        "message_ts": str(record.get("timestamp") or ""),
        "previous_message": previous_human[1] if previous_human else None,
        "previous_message_ts": str(previous_human[0].get("timestamp") or "") if previous_human else "",
        "assistant_message_id": str((assistant_record.get("message") or {}).get("id")
                                    or assistant_record.get("uuid") or ""),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Row construction / hook
# ---------------------------------------------------------------------------


def build_row(
    *,
    session_id: str,
    turn: dict,
    verdict: dict,
    hook_event: str = "",
    anchor_ts: str | None = None,
    journal: Path | None = None,
    rag_disabled: bool = False,
    window_seconds: float = RAG_LINK_WINDOW_SECONDS,
) -> dict:
    """Build one satisfaction ledger row (metadata only)."""
    message_ts = turn.get("message_ts") or _utc_now()
    # The injection that produced the judged response was journaled next to the
    # *previous* prompt.  Fall back to this message's timestamp for CLI use.
    rag_anchor = anchor_ts or turn.get("previous_message_ts") or message_ts
    rag_link, rag_status = link_injection(
        rag_anchor, session_id, journal=journal, disabled=rag_disabled,
        window_seconds=window_seconds,
    )
    model = turn.get("model")
    identity_text = turn.get("message_uuid") or hashlib.sha256(
        (turn.get("message") or "").encode("utf-8")
    ).hexdigest()
    event_id = _event_id(
        "satisfaction",
        session_id,
        identity_text,
        turn.get("assistant_message_id") or str(model or ""),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "kind": KIND,
        "event_id": event_id,
        "ts": _utc_now(),
        "session_id": session_id,
        "classification": verdict["classification"],
        "confidence": verdict["confidence"],
        "trigger_patterns": list(verdict["trigger_patterns"]),
        "response_was_helpful": verdict["response_was_helpful"],
        "is_question": bool(verdict.get("is_question")),
        "message_chars": len(turn.get("message") or ""),
        "message_ts": message_ts,
        "message_id": turn.get("message_uuid") or None,
        "judged_response_message_id": turn.get("assistant_message_id") or None,
        "model": model,
        "thompson_arm": thompson_arm(model),
        "hook_event": hook_event or None,
        "rag_injection": rag_link,
        "rag_link_status": rag_status,
        "rag_anchor_ts": rag_anchor,
    }


def _spawn_thompson_sync(ledger: Path, sync_ledger: Path, url: str) -> bool:
    """Start the detached Thompson feeder.  Never raises, never blocks."""
    try:
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--ledger", str(ledger),
                "--sync-ledger", str(sync_ledger),
                "--thompson-url", url,
                "--quiet",
                "thompson-sync",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except Exception:  # noqa: BLE001 - a hook must never block
        return False


def handle_hook(payload: dict, args) -> dict | None:
    """Classify the turn described by a hook payload; return the row or None."""
    session_id = str(payload.get("session_id") or "")
    transcript_path = payload.get("transcript_path")
    hook_event = str(payload.get("hook_event_name") or "")
    if not session_id or not transcript_path:
        return None
    turn = extract_turn(Path(os.path.expanduser(str(transcript_path))))
    if turn is None:
        return None
    verdict = classify(turn["message"], turn.get("previous_message"))
    ledger = Path(os.path.expanduser(args.ledger))
    window = _window_seconds(args.rag_window)
    journal = None if args.no_rag else _rag_journal_path(args.injection_journal)
    row = build_row(
        session_id=session_id,
        turn=turn,
        verdict=verdict,
        hook_event=hook_event,
        journal=journal,
        rag_disabled=bool(args.no_rag),
        window_seconds=window,
    )
    if row["event_id"] in _existing_event_ids(ledger):
        return None
    if not _append_row(ledger, row):
        return None
    if (
        not args.no_thompson
        and thompson_enabled()
        and thompson_eligible(row)
        and row["thompson_arm"]
    ):
        _spawn_thompson_sync(
            ledger,
            Path(os.path.expanduser(args.sync_ledger)),
            args.thompson_url or thompson_url(),
        )
    return row


def _window_seconds(value) -> float:
    if value is None:
        raw = os.environ.get(RAG_WINDOW_ENV)
        if raw:
            try:
                parsed = float(raw)
                if 0 < parsed <= 3600:
                    return parsed
            except ValueError:
                pass
        return RAG_LINK_WINDOW_SECONDS
    return float(value)


# ---------------------------------------------------------------------------
# Derived RAG outcome ledger (the label retrieval_relevance reads)
# ---------------------------------------------------------------------------


def build_rag_outcomes(satisfaction_rows: list[dict], injection_rows: list[dict]) -> list[dict]:
    """Join satisfaction links to injection rows and attach the ``helped`` label.

    A satisfaction row labels the injection that produced the judged response.
    When several rows link the same injection, the most recent classification
    wins.
    """
    injections = {
        row.get("event_id"): row
        for row in injection_rows
        if isinstance(row, dict) and row.get("kind") == "rag_injection"
    }
    labels: dict[str, tuple[str, dict]] = {}
    for row in satisfaction_rows:
        if not isinstance(row, dict) or row.get("kind") != KIND:
            continue
        helpful = row.get("response_was_helpful")
        if not isinstance(helpful, bool):
            continue
        link = row.get("rag_injection")
        if not isinstance(link, dict):
            continue
        event_id = link.get("event_id")
        if not event_id or event_id not in injections:
            continue
        when = str(row.get("ts") or "")
        if event_id not in labels or when >= labels[event_id][0]:
            labels[event_id] = (when, row)
    out: list[dict] = []
    for event_id, (_, row) in labels.items():
        derived = dict(injections[event_id])
        derived["kind"] = RAG_OUTCOME_KIND
        derived["helped"] = bool(row["response_was_helpful"])
        derived["satisfaction_event_id"] = row.get("event_id")
        derived["satisfaction_classification"] = row.get("classification")
        derived["satisfaction_confidence"] = row.get("confidence")
        derived["label_source"] = "satisfaction_signal"
        out.append(derived)
    out.sort(key=lambda item: str(item.get("timestamp") or ""))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_stdin(limit: int = MAX_STDIN_BYTES) -> str:
    try:
        return sys.stdin.read(limit)
    except Exception:  # noqa: BLE001
        return ""


def _cmd_hook(args) -> int:
    payload_text = _read_stdin()
    try:
        payload = json.loads(payload_text)
    except ValueError as exc:
        if not args.quiet:
            print(f"satisfaction hook: ignored unreadable input ({type(exc).__name__})",
                  file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        if not args.quiet:
            print("satisfaction hook: ignored non-object input", file=sys.stderr)
        return 0
    if payload.get("stop_hook_active") is True:
        return 0
    try:
        handle_hook(payload, args)
    except Exception as exc:  # noqa: BLE001 - a hook must never block
        if not args.quiet:
            print(f"satisfaction hook: ignored error ({type(exc).__name__})", file=sys.stderr)
    return 0


def _cmd_classify(args) -> int:
    text = args.text if args.text is not None else _read_stdin()
    verdict = classify(text, args.previous)
    if args.json:
        print(json.dumps(verdict, sort_keys=True))
    else:
        print(f"{verdict['classification']} {verdict['confidence']:.2f} "
              f"{','.join(verdict['trigger_patterns'])}")
    return 0


def _cmd_sync(args) -> int:
    try:
        sync_thompson(
            Path(os.path.expanduser(args.ledger)),
            Path(os.path.expanduser(args.sync_ledger)),
            url=args.url or args.thompson_url or thompson_url(),
            dry_run=args.dry_run,
            timeout=args.timeout,
            quiet=args.quiet,
        )
    except Exception as exc:  # noqa: BLE001 - never block the caller
        if not args.quiet:
            print(f"satisfaction thompson-sync: failed ({type(exc).__name__})", file=sys.stderr)
    return 0


def _cmd_rag_outcomes(args) -> int:
    ledger = Path(os.path.expanduser(args.ledger))
    output = Path(os.path.expanduser(args.output))
    journal = _rag_journal_path(args.injection_journal)
    if journal is None:
        if not args.quiet:
            print("satisfaction rag-outcomes: injection journal disabled", file=sys.stderr)
        return 0
    satisfaction_rows = []
    if ledger.exists():
        with open(ledger, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    satisfaction_rows.append(row)
    injection_rows = []
    if journal.exists():
        with open(journal, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    injection_rows.append(row)
    rows = build_rag_outcomes(satisfaction_rows, injection_rows)
    _write_jsonl_atomic(output, rows)
    if not args.quiet:
        print(json.dumps({"output": str(output), "rows": len(rows)}, sort_keys=True),
              file=sys.stderr)
    return 0


def _cmd_patterns(_args) -> int:
    print(json.dumps(documented_patterns(), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", default=os.environ.get(LEDGER_ENV, DEFAULT_LEDGER))
    parser.add_argument("--sync-ledger",
                        default=os.environ.get(SYNC_ENV, DEFAULT_SYNC_LEDGER))
    parser.add_argument("--injection-journal", default=None)
    parser.add_argument("--rag-window", type=float, default=None)
    parser.add_argument("--thompson-url", default=None)
    parser.add_argument("--no-thompson", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command")

    hook = sub.add_parser("hook", help="PostToolUse/Stop hook (reads stdin)")
    hook.set_defaults(func=_cmd_hook)

    classify_parser = sub.add_parser("classify", help="classify one message")
    classify_parser.add_argument("--text", default=None)
    classify_parser.add_argument("--previous", default=None)
    classify_parser.add_argument("--json", action="store_true")
    classify_parser.set_defaults(func=_cmd_classify)

    sync = sub.add_parser("thompson-sync", help="feed journaled rows to Thompson")
    sync.add_argument("--url", default=None)
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--timeout", type=float, default=1.5)
    sync.set_defaults(func=_cmd_sync)

    rag = sub.add_parser("rag-outcomes", help="join satisfaction to the RAG journal")
    rag.add_argument("--output",
                     default=os.environ.get(RAG_OUTCOMES_ENV, DEFAULT_RAG_OUTCOMES))
    rag.set_defaults(func=_cmd_rag_outcomes)

    patterns = sub.add_parser("patterns", help="print the documented pattern table")
    patterns.set_defaults(func=_cmd_patterns)
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # Bare invocation with a JSON payload on stdin is the hook contract.
    if not getattr(args, "command", None):
        args.func = _cmd_hook
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - last-resort fail-open
        if not args.quiet:
            print(f"satisfaction signal: ignored error ({type(exc).__name__})", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
