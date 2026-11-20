#!/usr/bin/env python3
"""Tests for tools/satisfaction_signal.py (classification, journal, RAG, Thompson).

Run with pytest from the repository root:

    python3 -m pytest tests/test_satisfaction_signal.py -q
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import satisfaction_signal as signal  # noqa: E402

TOOL = TOOLS_DIR / "satisfaction_signal.py"

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

#: Hand-labelled messages covering every documented pattern and the tricky
#: combinations (courtesy "no", "yes ... but", retries, terse input).
LABELLED = [
    # SATISFIED: gratitude, praise, affirmation, continuation, new question
    ("SATISFIED", "thanks!"),
    ("SATISFIED", "Perfect, that's exactly it."),
    ("SATISFIED", "Great work"),
    ("SATISFIED", "yes"),
    ("SATISFIED", "yep, proceed"),
    ("SATISFIED", "ok"),
    ("SATISFIED", "sure, go ahead"),
    ("SATISFIED", "continue"),
    ("SATISFIED", "sounds good"),
    ("SATISFIED", "lgtm"),
    ("SATISFIED", "looks good to me"),
    ("SATISFIED", "that works"),
    ("SATISFIED", "makes sense, next"),
    ("SATISFIED", "no problem, proceed"),
    ("SATISFIED", "not bad at all"),
    ("SATISFIED", "Thanks — now do the same for the review ledger."),
    ("SATISFIED", "Great, now update the README."),
    ("SATISFIED", "How does the Thompson router pick a model?"),
    ("SATISFIED", "Please also update the docs."),
    ("SATISFIED", "what about the GA target?"),
    # CORRECTION: negation, disidentification, failure, retry, omission
    ("CORRECTION", "no"),
    ("CORRECTION", "nope"),
    ("CORRECTION", "that's wrong"),
    ("CORRECTION", "No, that's not what I meant."),
    ("CORRECTION", "I meant the other file"),
    ("CORRECTION", "I said remove the flag"),
    ("CORRECTION", "it doesn't work"),
    ("CORRECTION", "still failing"),
    ("CORRECTION", "the test failed"),
    ("CORRECTION", "that's broken"),
    ("CORRECTION", "try again"),
    ("CORRECTION", "redo it"),
    ("CORRECTION", "you missed the tests"),
    ("CORRECTION", "you forgot to update the schema"),
    ("CORRECTION", "incorrect"),
    ("CORRECTION", "yes, but it doesn't work"),
    ("CORRECTION", "thanks, but the build failed"),
    ("CORRECTION", "not what I asked for"),
    ("CORRECTION", "I already asked for the tests"),
    ("CORRECTION", "Can you redo that?"),
    ("CORRECTION", "[Request interrupted by user]"),
    ("CORRECTION", "wait"),
    ("CORRECTION", "actually, that's not right"),
    # UNCLEAR: too short / no signal
    ("UNCLEAR", "hmm"),
    ("UNCLEAR", "k"),
    ("UNCLEAR", "?"),
    ("UNCLEAR", "maybe"),
    ("UNCLEAR", "..."),
    ("UNCLEAR", "what"),
]


def write_transcript(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def turn(message: str, *, uuid: str = "u-next", ts: str = "2026-11-19T10:00:20.000Z",
         model: str = "rits/zai-org/glm-5-3", assistant_id: str = "msg-prev",
         previous: str | None = "build it") -> list[dict]:
    """A minimal transcript: previous prompt, response, next user message."""
    rows = [
        {"type": "user", "uuid": "u-prev", "timestamp": "2026-11-19T10:00:00.000Z",
         "isSidechain": False, "sessionId": "S1",
         "message": {"role": "user", "content": previous}},
        {"type": "assistant", "uuid": "a-prev", "timestamp": "2026-11-19T10:00:05.000Z",
         "isSidechain": False,
         "message": {"role": "assistant", "id": assistant_id, "model": model,
                     "stop_reason": "end_turn",
                     "content": [{"type": "text", "text": "response text"}]}},
        {"type": "user", "uuid": uuid, "timestamp": ts,
         "isSidechain": False, "sessionId": "S1",
         "message": {"role": "user", "content": message}},
        {"type": "assistant", "uuid": "a-now", "timestamp": "2026-11-19T10:00:30.000Z",
         "isSidechain": False,
         "message": {"role": "assistant", "id": "msg-now", "model": model,
                     "stop_reason": "end_turn",
                     "content": [{"type": "text", "text": "current response"}]}},
    ]
    return rows


def injection(event_id: str, ts: str, *, session: str = "S1",
              context_chars: int = 100, sources: list[str] | None = None,
              truncated: bool = False) -> dict:
    return {
        "schema_version": 1, "producer": "claude-code", "kind": "rag_injection",
        "event_id": event_id, "timestamp": ts, "session_id": session,
        "query_hash": "abc", "prompt_chars": 10, "context_chars": context_chars,
        "truncated": truncated, "source_ids": sources or ["memory:notes.md#Deploy"],
        "sections": {"memory": {"status": "ok", "results": 2, "chars": context_chars}},
    }


def run_hook(payload, *extra, env=None):
    command = [sys.executable, str(TOOL), *extra, "hook"]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        command,
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True, text=True, check=False, timeout=30, env=full_env,
    )


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def args_for(tmp_path: Path, **overrides):
    values = {
        "ledger": str(tmp_path / "satisfaction.jsonl"),
        "sync_ledger": str(tmp_path / "thompson-sync.jsonl"),
        "injection_journal": "off",
        "rag_window": None,
        "thompson_url": None,
        "no_thompson": True,
        "no_rag": True,
        "quiet": True,
        "command": "hook",
    }
    values.update(overrides)
    return type("Args", (), values)()


class FakeThompson:
    """Loopback stand-in speaking the same /thompson protocol as the daemon."""

    def __init__(self, replies=None):
        self.replies = list(replies) if replies else None
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _handler(fake):  # noqa: N805 - closure factory
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(body)
                ok = True if fake.replies is None else fake.replies.pop(0)
                raw = json.dumps({"ok": ok, "request_id": body.get("request_id")}).encode()
                self.send_response(200 if ok else 500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args):
                return

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------
# Pattern matching accuracy
# ---------------------------------------------------------------------------


def test_pattern_accuracy_on_labelled_messages():
    misses = []
    for expected, text in LABELLED:
        verdict = signal.classify(text)
        if verdict["classification"] != expected:
            misses.append((expected, text, verdict["classification"], verdict["trigger_patterns"]))
    accuracy = (len(LABELLED) - len(misses)) / len(LABELLED)
    assert accuracy >= 0.90, f"accuracy {accuracy:.2f}; misses={misses}"


@pytest.mark.parametrize("expected,text", LABELLED)
def test_each_labelled_message(expected, text):
    assert signal.classify(text)["classification"] == expected


def test_courtesy_no_is_not_a_correction():
    verdict = signal.classify("no problem, proceed")
    assert verdict["classification"] == "SATISFIED"
    assert verdict["response_was_helpful"] is True
    assert "no_problem_guard" in verdict["trigger_patterns"]


def test_explicit_failure_beats_courtesy_yes():
    verdict = signal.classify("yes, but it doesn't work")
    assert verdict["classification"] == "CORRECTION"
    assert verdict["response_was_helpful"] is False


def test_weak_ambiguous_signals_stay_below_thompson_floor():
    # "actually" and "wait" are documented correction cues but are ambiguous
    # enough that the default 0.6 Thompson floor must exclude them.
    for text in ("Actually can you also add tests?", "wait"):
        verdict = signal.classify(text)
        assert verdict["confidence"] < signal.DEFAULT_MIN_CONFIDENCE


def test_restated_request_is_a_correction():
    verdict = signal.classify(
        "add tests for the satisfaction signal",
        "please add tests for the satisfaction signal",
    )
    assert verdict["classification"] == "CORRECTION"
    assert "restated_request" in verdict["trigger_patterns"]
    assert verdict["confidence"] < signal.DEFAULT_MIN_CONFIDENCE


def test_confusion_and_frustration_are_corrections():
    for text in ("waait im confused what are you building and why?",
                 "wtf", "I don't understand what this does"):
        verdict = signal.classify(text)
        assert verdict["classification"] == "CORRECTION", text
        assert verdict["response_was_helpful"] is False


def test_reassurance_is_satisfied():
    assert signal.classify("its ok for now")["classification"] == "SATISFIED"
    assert signal.classify("that's not ok")["classification"] == "CORRECTION"


def test_long_instruction_is_not_a_correction():
    # Reaction words deep inside pasted instructions are not user pushback.
    text = (
        "Check run 32 workers for MR !1117 review (bundle /tmp/review/run32). "
        "Workers: w_glm.md, w_deepseek.md, w_muse.md (if its output is empty, "
        "garbage, or off-format, note that it failed the smoke test and exclude "
        "it from the arbiter bundle, reporting the failure to the user)."
    )
    verdict = signal.classify(text)
    assert verdict["classification"] == "SATISFIED"


def test_questions_are_not_restatements():
    verdict = signal.classify("so may I exit this session",
                              "may I exit this session now")
    assert verdict["classification"] != "CORRECTION"


def test_harness_automation_is_not_human():
    agent_message = {
        "role": "user",
        "content": "Another Claude session sent a message:\n<agent-message from=\"x\">\n"
                   "[Subagent hand-back] model output, not the user.\n  The test failed.\n",
    }
    assert signal._human_text(agent_message) is None
    notification = {"role": "user", "content": "<task-notification>done</task-notification>"}
    assert signal._human_text(notification) is None
    reminder = {"role": "user", "content": "<system-reminder>note</system-reminder>"}
    assert signal._human_text(reminder) is None
    tool_result = {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}
    assert signal._human_text(tool_result) is None
    assert signal._human_text({"role": "user", "content": "thanks"}) == "thanks"


def test_incidental_failure_words_do_not_override_opening_praise():
    for text in (
        "sounds good...and if it fails, we need an email sent out explaining the failure",
        "perfect - that did work!  now how can we test where we fail at QA",
    ):
        assert signal.classify(text)["classification"] == "SATISFIED", text
    # A contrastive "but" takes the praise back.
    assert signal.classify("thanks, but the build failed")["classification"] == "CORRECTION"


def test_questions_that_mention_failure_are_not_pushback():
    for text in ("whats wrong with lib runs", "did it fail again"):
        verdict = signal.classify(text)
        assert verdict["classification"] == "SATISFIED", text
        assert verdict["is_question"] is True


def test_numbered_answers_are_not_confident_corrections():
    verdict = signal.classify(
        "1 no. 2 if the lib fails, thats a pipeline failure. 3 yes. 4 update"
    )
    assert verdict["confidence"] < signal.DEFAULT_MIN_CONFIDENCE


def test_stop_interjection_is_a_correction_but_stop_the_service_is_not():
    assert signal.classify(
        "OK stop - what I mean is we had a set of documents copied over: that is perfect"
    )["classification"] == "CORRECTION"
    assert signal.classify("stop the service")["classification"] != "CORRECTION"


def test_unclear_is_null_helpfulness():
    verdict = signal.classify("hmm")
    assert verdict["classification"] == "UNCLEAR"
    assert verdict["response_was_helpful"] is None


def test_non_text_input_is_unclear():
    assert signal.classify(None)["classification"] == "UNCLEAR"


def test_pattern_table_is_documented():
    ids = [pattern.id for pattern in signal.PATTERNS]
    assert len(ids) == len(set(ids))
    for pattern in signal.PATTERNS:
        assert pattern.description.strip()
        assert pattern.regex.pattern
        assert pattern.classification in (signal.SATISFIED, signal.CORRECTION)
        assert 0.0 <= pattern.weight <= 1.0
    documented = signal.documented_patterns()
    assert {row["id"] for row in documented} == set(ids)


# ---------------------------------------------------------------------------
# Journal schema / hook
# ---------------------------------------------------------------------------


def test_hook_journals_required_schema_and_no_text(tmp_path):
    transcript = write_transcript(tmp_path / "t.jsonl", turn("thanks, perfect"))
    ledger = tmp_path / "ledger" / "satisfaction.jsonl"
    result = run_hook(
        {"hook_event_name": "Stop", "session_id": "S1", "transcript_path": str(transcript)},
        "--ledger", str(ledger), "--no-thompson", "--no-rag",
    )
    assert result.returncode == 0, result.stderr
    rows = read_rows(ledger)
    assert len(rows) == 1
    row = rows[0]
    for field in ("ts", "session_id", "classification", "confidence",
                  "trigger_patterns", "response_was_helpful"):
        assert field in row
    assert row["classification"] == "SATISFIED"
    assert row["response_was_helpful"] is True
    assert row["model"] == "rits/zai-org/glm-5-3"
    assert row["thompson_arm"] == "glm-5-3"
    assert row["judged_response_message_id"] == "msg-prev"
    assert row["message_id"] == "u-next"
    assert row["kind"] == "satisfaction"
    assert row["schema_version"] == 1
    # Metadata only: neither the user message nor the response text is stored.
    text = ledger.read_text(encoding="utf-8")
    assert "thanks, perfect" not in text
    assert "response text" not in text
    assert stat.S_IMODE(os.stat(ledger).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(ledger.parent).st_mode) == 0o700


def test_hook_is_idempotent_for_the_same_turn(tmp_path):
    transcript = write_transcript(tmp_path / "t.jsonl", turn("no, that's wrong"))
    ledger = tmp_path / "satisfaction.jsonl"
    payload = {"hook_event_name": "Stop", "session_id": "S1", "transcript_path": str(transcript)}
    for _ in range(3):
        assert run_hook(payload, "--ledger", str(ledger), "--no-thompson", "--no-rag").returncode == 0
    assert len(read_rows(ledger)) == 1


def test_post_tool_use_and_stop_share_one_row(tmp_path):
    transcript = write_transcript(tmp_path / "t.jsonl", turn("perfect"))
    ledger = tmp_path / "satisfaction.jsonl"
    for event in ("PostToolUse", "Stop"):
        run_hook({"hook_event_name": event, "session_id": "S1",
                  "transcript_path": str(transcript)},
                 "--ledger", str(ledger), "--no-thompson", "--no-rag")
    assert len(read_rows(ledger)) == 1


def test_first_turn_has_no_row(tmp_path):
    # A user message with no preceding assistant response is not feedback.
    transcript = write_transcript(tmp_path / "t.jsonl", [
        {"type": "user", "uuid": "u1", "timestamp": "2026-11-19T10:00:00.000Z",
         "message": {"role": "user", "content": "first prompt"}},
    ])
    ledger = tmp_path / "satisfaction.jsonl"
    assert run_hook({"hook_event_name": "Stop", "session_id": "S1",
                     "transcript_path": str(transcript)},
                    "--ledger", str(ledger), "--no-thompson", "--no-rag").returncode == 0
    assert not ledger.exists()


def test_tool_results_and_system_reminders_are_skipped(tmp_path):
    rows = turn("ok proceed")
    # Inject synthetic noise after the human message: tool results, reminders.
    rows.insert(2, {"type": "user", "uuid": "tr", "timestamp": "2026-11-19T10:00:10.000Z",
                    "message": {"role": "user",
                                "content": [{"type": "tool_result", "content": "big output"}]}})
    rows.insert(3, {"type": "user", "uuid": "rem", "timestamp": "2026-11-19T10:00:11.000Z",
                    "isMeta": True,
                    "message": {"role": "user", "content": "<system-reminder>noise</system-reminder>"}})
    transcript = write_transcript(tmp_path / "t.jsonl", rows)
    ledger = tmp_path / "satisfaction.jsonl"
    run_hook({"hook_event_name": "Stop", "session_id": "S1",
              "transcript_path": str(transcript)},
             "--ledger", str(ledger), "--no-thompson", "--no-rag")
    row = read_rows(ledger)[0]
    assert row["message_id"] == "u-next"
    assert row["classification"] == "SATISFIED"


def test_hook_skips_stop_hook_active(tmp_path):
    transcript = write_transcript(tmp_path / "t.jsonl", turn("perfect"))
    ledger = tmp_path / "satisfaction.jsonl"
    assert run_hook({"hook_event_name": "Stop", "session_id": "S1",
                     "transcript_path": str(transcript), "stop_hook_active": True},
                    "--ledger", str(ledger), "--no-thompson", "--no-rag").returncode == 0
    assert not ledger.exists()


def test_hook_fails_open(tmp_path):
    ledger = tmp_path / "satisfaction.jsonl"
    cases = [
        "not json",
        json.dumps({"session_id": "S1"}),
        json.dumps({"session_id": "S1", "transcript_path": str(tmp_path / "missing.jsonl")}),
        json.dumps(["not", "an", "object"]),
    ]
    for payload in cases:
        result = run_hook(payload, "--ledger", str(ledger), "--no-thompson", "--no-rag")
        assert result.returncode == 0
        assert result.stdout == ""
    assert not ledger.exists()


def test_hook_reads_a_bounded_tail_of_a_large_transcript(tmp_path):
    # 6 MB of tool-result noise before the real turn; the hook must not parse it all.
    noise = json.dumps({
        "type": "user", "uuid": "noise", "timestamp": "2026-11-19T09:00:00.000Z",
        "message": {"role": "user", "content": [{"type": "tool_result", "content": "x" * 600_000}]},
    })
    transcript = tmp_path / "t.jsonl"
    with open(transcript, "w", encoding="utf-8") as handle:
        for _ in range(10):
            handle.write(noise + "\n")
        for row in turn("perfect"):
            handle.write(json.dumps(row) + "\n")
    ledger = tmp_path / "satisfaction.jsonl"
    started = time.monotonic()
    result = run_hook({"hook_event_name": "Stop", "session_id": "S1",
                       "transcript_path": str(transcript)},
                      "--ledger", str(ledger), "--no-thompson", "--no-rag")
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert read_rows(ledger)[0]["classification"] == "SATISFIED"
    assert elapsed < 1.0, f"hook took {elapsed:.3f}s"
    assert len(signal._tail_records(transcript)) < 10


# ---------------------------------------------------------------------------
# RAG linkage
# ---------------------------------------------------------------------------


def test_link_injection_picks_the_nearest_row_in_window(tmp_path):
    journal = tmp_path / "injections.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in [
        injection("near", "2026-11-19T10:00:01.000Z"),
        injection("far", "2026-11-19T09:58:00.000Z", context_chars=9),
        injection("other-session", "2026-11-19T10:00:01.000Z", session="S2"),
    ]) + "\n", encoding="utf-8")
    link, status = signal.link_injection("2026-11-19T10:00:00.000Z", "S1", journal=journal)
    assert status == "linked"
    assert link["event_id"] == "near"
    assert link["delta_seconds"] == 1.0
    assert link["direction"] == "after"
    assert link["context_chars"] == 100


def test_link_injection_outside_window(tmp_path):
    journal = tmp_path / "injections.jsonl"
    journal.write_text(json.dumps(injection("old", "2026-11-19T09:00:00.000Z")) + "\n",
                       encoding="utf-8")
    link, status = signal.link_injection("2026-11-19T10:00:00.000Z", "S1", journal=journal)
    assert link is None
    assert status == "no_injection_in_window"


def test_link_injection_missing_and_disabled(tmp_path, monkeypatch):
    missing = tmp_path / "missing.jsonl"
    assert signal.link_injection("2026-11-19T10:00:00.000Z", "S1", journal=missing) == (
        None, "journal_unavailable")
    monkeypatch.setenv(signal.RAG_JOURNAL_ENV, "off")
    assert signal.link_injection("2026-11-19T10:00:00.000Z", "S1") == (
        None, "disabled")


def test_no_rag_flag_reports_disabled(tmp_path):
    transcript = write_transcript(tmp_path / "t.jsonl", turn("thanks"))
    ledger = tmp_path / "satisfaction.jsonl"
    run_hook({"hook_event_name": "Stop", "session_id": "S1",
              "transcript_path": str(transcript)},
             "--ledger", str(ledger), "--no-thompson", "--no-rag")
    assert read_rows(ledger)[0]["rag_link_status"] == "disabled"


def test_hook_links_the_injection_that_conditioned_the_judged_response(tmp_path):
    # Previous prompt at 10:00:00 produced the response; its RAG injection was
    # at 10:00:01.  The classified prompt's own injection at 10:00:20 must not
    # be chosen: that context conditions the *next* response.
    journal = tmp_path / "injections.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in [
        injection("judged", "2026-11-19T10:00:01.000Z", context_chars=111),
        injection("current", "2026-11-19T10:00:20.500Z", context_chars=222),
    ]) + "\n", encoding="utf-8")
    transcript = write_transcript(tmp_path / "t.jsonl", turn("thanks"))
    ledger = tmp_path / "satisfaction.jsonl"
    run_hook({"hook_event_name": "Stop", "session_id": "S1",
              "transcript_path": str(transcript)},
             "--ledger", str(ledger), "--no-thompson",
             "--injection-journal", str(journal))
    row = read_rows(ledger)[0]
    assert row["rag_link_status"] == "linked"
    assert row["rag_injection"]["event_id"] == "judged"
    assert row["rag_anchor_ts"] == "2026-11-19T10:00:00.000Z"


def test_rag_outcomes_attach_the_helped_label(tmp_path):
    journal = tmp_path / "injections.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in [
        injection("inj-1", "2026-11-19T10:00:01.000Z", context_chars=111),
        injection("inj-2", "2026-11-19T10:05:01.000Z", context_chars=222),
    ]) + "\n", encoding="utf-8")
    ledger = tmp_path / "satisfaction.jsonl"
    rows = [
        {"schema_version": 1, "producer": "satisfaction-signal", "kind": "satisfaction",
         "event_id": "s1", "ts": "2026-11-19T10:06:00.000Z", "classification": "SATISFIED",
         "confidence": 0.9, "response_was_helpful": True,
         "rag_injection": {"event_id": "inj-1"}},
        {"schema_version": 1, "producer": "satisfaction-signal", "kind": "satisfaction",
         "event_id": "s2", "ts": "2026-11-19T10:06:30.000Z", "classification": "CORRECTION",
         "confidence": 0.9, "response_was_helpful": False,
         "rag_injection": {"event_id": "inj-1"}},
        {"schema_version": 1, "producer": "satisfaction-signal", "kind": "satisfaction",
         "event_id": "s3", "ts": "2026-11-19T10:07:00.000Z", "classification": "UNCLEAR",
         "confidence": 0.2, "response_was_helpful": None,
         "rag_injection": {"event_id": "inj-2"}},
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "rag-outcomes.jsonl"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--ledger", str(ledger),
         "--injection-journal", str(journal), "--quiet", "rag-outcomes",
         "--output", str(output)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    derived = read_rows(output)
    assert len(derived) == 1  # the UNCLEAR row contributes no label
    row = derived[0]
    assert row["event_id"] == "inj-1"
    assert row["kind"] == "rag_outcome"
    assert row["helped"] is False  # latest classification wins
    assert row["satisfaction_event_id"] == "s2"
    assert row["label_source"] == "satisfaction_signal"
    assert row["context_chars"] == 111


# ---------------------------------------------------------------------------
# Thompson integration
# ---------------------------------------------------------------------------


def satisfaction_row(event_id, classification, confidence, model, helpful, *,
                     ts=None, patterns=None, is_question=False):
    return {
        "schema_version": 1, "producer": "satisfaction-signal", "kind": "satisfaction",
        "event_id": event_id, "ts": ts or "2026-11-19T10:00:00.000Z",
        "session_id": "S1", "classification": classification, "confidence": confidence,
        "trigger_patterns": list(patterns or []), "response_was_helpful": helpful,
        "is_question": is_question,
        "model": model, "thompson_arm": signal.thompson_arm(model),
    }


def test_thompson_arm_mapping():
    assert signal.thompson_arm("rits/zai-org/glm-5-3") == "glm-5-3"
    assert signal.thompson_arm("rits/deepseek-ai/DeepSeek-V4.1-Flash-1017") == "deepseek-v4.1-flash"
    assert signal.thompson_arm("claude-sonnet-5-5") == "sonnet"
    assert signal.thompson_arm("<synthetic>") is None
    assert signal.thompson_arm(None) is None


def test_thompson_records_positive_and_negative_outcomes(tmp_path):
    fake = FakeThompson()
    try:
        ledger = tmp_path / "satisfaction.jsonl"
        rows = [
            satisfaction_row("e1", "SATISFIED", 0.92, "rits/zai-org/glm-5-3", True),
            satisfaction_row("e2", "CORRECTION", 0.88, "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017", False,
                             patterns=["failure"]),
        ]
        ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        sync = tmp_path / "sync.jsonl"
        summary = signal.sync_thompson(ledger, sync, url=fake.url, quiet=True)
        assert summary["sent"] == 2
        assert len(fake.requests) == 2
        by_model = {request["model"]: request for request in fake.requests}
        assert by_model["glm-5-3"]["success"] is True
        assert by_model["deepseek-v4.1-flash"]["success"] is False
        assert by_model["glm-5-3"]["task_type"] == signal.TASK_TYPE
        assert by_model["glm-5-3"]["cost"] == 0.0
        assert by_model["glm-5-3"]["request_id"] == "e1"
        assert len(read_rows(sync)) == 2
        # A second sync does not double-count.
        again = signal.sync_thompson(ledger, sync, url=fake.url, quiet=True)
        assert again["sent"] == 0
        assert len(fake.requests) == 2
    finally:
        fake.close()


def test_thompson_skips_unclear_low_confidence_and_unknown_arms(tmp_path):
    fake = FakeThompson()
    try:
        ledger = tmp_path / "satisfaction.jsonl"
        rows = [
            satisfaction_row("u", "UNCLEAR", 0.9, "rits/zai-org/glm-5-3", None),
            satisfaction_row("low", "CORRECTION", 0.55, "rits/zai-org/glm-5-3", False,
                             patterns=["failure"]),
            satisfaction_row("unknown", "SATISFIED", 0.9, "some-unmapped-model", True),
            satisfaction_row("negation", "CORRECTION", 0.88, "rits/zai-org/glm-5-3", False,
                             patterns=["negation_opener"]),
            satisfaction_row("restated", "CORRECTION", 0.9, "rits/zai-org/glm-5-3", False,
                             patterns=["restated_request"]),
            satisfaction_row("question", "CORRECTION", 0.9, "rits/zai-org/glm-5-3", False,
                             patterns=["failure"], is_question=True),
            satisfaction_row("interrupt", "CORRECTION", 0.95, "rits/zai-org/glm-5-3", False,
                             patterns=["interrupted"]),
        ]
        ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        summary = signal.sync_thompson(ledger, tmp_path / "sync.jsonl", url=fake.url, quiet=True)
        assert summary["sent"] == 0
        assert fake.requests == []
    finally:
        fake.close()


def test_sync_derives_the_arm_when_the_row_lacks_it(tmp_path):
    fake = FakeThompson()
    try:
        row = satisfaction_row("e1", "SATISFIED", 0.9, "rits/zai-org/glm-5-3", True)
        del row["thompson_arm"]
        ledger = tmp_path / "satisfaction.jsonl"
        ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary = signal.sync_thompson(ledger, tmp_path / "sync.jsonl", url=fake.url, quiet=True)
        assert summary["sent"] == 1
        assert fake.requests[0]["model"] == "glm-5-3"
    finally:
        fake.close()


def test_thompson_precision_gate():
    def row(classification, confidence, patterns, helpful, is_question=False):
        return {"classification": classification, "confidence": confidence,
                "trigger_patterns": patterns, "response_was_helpful": helpful,
                "is_question": is_question}

    assert signal.thompson_eligible(row("SATISFIED", 0.7, ["we_good"], True))
    assert not signal.thompson_eligible(row("SATISFIED", 0.5, ["imperative_continuation"], True))
    assert signal.thompson_eligible(row("CORRECTION", 0.85, ["failure"], False))
    assert signal.thompson_eligible(row("CORRECTION", 0.9, ["meant"], False))
    assert not signal.thompson_eligible(row("CORRECTION", 0.88, ["negation_opener"], False))
    assert not signal.thompson_eligible(row("CORRECTION", 0.9, ["restated_request"], False))
    assert not signal.thompson_eligible(row("CORRECTION", 0.9, ["failure"], False, True))
    assert not signal.thompson_eligible(row("CORRECTION", 0.95, ["interrupted"], False))
    assert not signal.thompson_eligible(row("UNCLEAR", 0.9, [], None))


def test_thompson_failed_post_is_retried(tmp_path):
    fake = FakeThompson(replies=[False, True])
    try:
        ledger = tmp_path / "satisfaction.jsonl"
        ledger.write_text(json.dumps(
            satisfaction_row("e1", "SATISFIED", 0.9, "rits/zai-org/glm-5-3", True)) + "\n",
            encoding="utf-8")
        sync = tmp_path / "sync.jsonl"
        first = signal.sync_thompson(ledger, sync, url=fake.url, quiet=True)
        assert first["failed"] == 1
        assert read_rows(sync) == []
        second = signal.sync_thompson(ledger, sync, url=fake.url, quiet=True)
        assert second["sent"] == 1
        assert len(read_rows(sync)) == 1
    finally:
        fake.close()


def test_sync_cli_accepts_the_detached_spawn_argv(tmp_path):
    """The exact argv _spawn_thompson_sync uses must parse and send."""
    fake = FakeThompson()
    try:
        ledger = tmp_path / "satisfaction.jsonl"
        ledger.write_text(json.dumps(
            satisfaction_row("e1", "SATISFIED", 0.9, "rits/zai-org/glm-5-3", True)) + "\n",
            encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(TOOL),
             "--ledger", str(ledger),
             "--sync-ledger", str(tmp_path / "sync.jsonl"),
             "--thompson-url", fake.url,
             "--quiet", "thompson-sync"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert [request["model"] for request in fake.requests] == ["glm-5-3"]
        assert fake.requests[0]["success"] is True
    finally:
        fake.close()


def test_thompson_unreachable_is_not_fatal(tmp_path):
    ledger = tmp_path / "satisfaction.jsonl"
    ledger.write_text(json.dumps(
        satisfaction_row("e1", "SATISFIED", 0.9, "rits/zai-org/glm-5-3", True)) + "\n",
        encoding="utf-8")
    summary = signal.sync_thompson(ledger, tmp_path / "sync.jsonl",
                                   url="http://127.0.0.1:1", timeout=0.2, quiet=True)
    assert summary["sent"] == 0
    assert summary["failed"] == 1


def test_thompson_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv(signal.THOMPSON_ENV, "0")
    assert signal.thompson_enabled() is False
    monkeypatch.setenv(signal.THOMPSON_ENV, "1")
    assert signal.thompson_enabled() is True


def test_hook_spawns_sync_only_for_helpful_confident_rows(tmp_path, monkeypatch):
    spawned = []

    def fake_spawn(ledger, sync, url):
        spawned.append((ledger, sync, url))
        return True

    monkeypatch.setattr(signal, "_spawn_thompson_sync", fake_spawn)
    monkeypatch.setattr(signal, "thompson_enabled", lambda: True)
    monkeypatch.delenv(signal.MIN_CONFIDENCE_ENV, raising=False)

    transcript = write_transcript(tmp_path / "t.jsonl", turn("thanks, perfect"))
    args = args_for(tmp_path, no_thompson=False, no_rag=True)
    row = signal.handle_hook({"hook_event_name": "Stop", "session_id": "S1",
                              "transcript_path": str(transcript)}, args)
    assert row["classification"] == "SATISFIED"
    assert len(spawned) == 1

    unclear = write_transcript(tmp_path / "u.jsonl", turn("hmm", uuid="u-2"))
    row = signal.handle_hook({"hook_event_name": "Stop", "session_id": "S1",
                              "transcript_path": str(unclear)}, args)
    assert row["classification"] == "UNCLEAR"
    assert len(spawned) == 1  # no Thompson feed for UNCLEAR


def test_hook_does_not_spawn_when_disabled(tmp_path, monkeypatch):
    spawned = []

    def fake_spawn(*args, **kwargs):
        spawned.append(args)
        return True

    monkeypatch.setattr(signal, "_spawn_thompson_sync", fake_spawn)
    transcript = write_transcript(tmp_path / "t.jsonl", turn("perfect"))
    args = args_for(tmp_path, no_thompson=True, no_rag=True)
    signal.handle_hook({"hook_event_name": "Stop", "session_id": "S1",
                        "transcript_path": str(transcript)}, args)
    assert spawned == []
