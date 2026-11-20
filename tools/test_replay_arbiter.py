#!/usr/bin/env python3
"""Tests for tools/replay_arbiter.py (local arbiter replay + Opus comparison)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import replay_arbiter  # noqa: E402

WORKER_A = "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
WORKER_B = "rits/zai-org/glm-5-3"
WORKER_C = "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Never touch the live replay state file."""
    monkeypatch.setattr(replay_arbiter, "STATE",
                        str(tmp_path / "arbiter-replay-state.json"))


def write_review(tmp_path, review_id, *, workers, arbiter_text,
                 arbiter_model="claude-opus-5-5", subject_name=None,
                 subject_text="def add(a, b):\n    return a + b\n", sha_override=None):
    subject = tmp_path / (subject_name or f"{review_id}.py")
    subject.write_text(subject_text, encoding="utf-8")
    sha = sha_override or hashlib.sha256(subject_text.encode("utf-8")).hexdigest()
    rows = []
    for model, text in workers:
        rows.append({"schema_version": 1, "producer": "multi-model-review",
                     "review_id": review_id, "kind": "worker_result", "role": "worker",
                     "subject": str(subject), "subject_sha256": sha,
                     "truncated": False, "model": model, "text": text,
                     "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 5,
                     "cost_usd": 0.01})
    rows.append({"schema_version": 1, "producer": "multi-model-review",
                 "review_id": review_id, "kind": "arbiter_decision", "role": "arbiter",
                 "subject": str(subject), "subject_sha256": sha,
                 "truncated": False, "model": arbiter_model, "text": arbiter_text,
                 "stop_reason": "end_turn", "input_tokens": 50, "output_tokens": 20,
                 "cost_usd": 0.1})
    return rows


def write_ledger(tmp_path, rows):
    path = tmp_path / "reviews.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class FakeModelClient:
    def __init__(self, text="SELECTED: " + WORKER_B + "\nVERDICT: approve"):
        self.text = text
        self.calls = []

    def call_model_response(self, *, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return SimpleNamespace(text=self.text, latency_ms=12.0, input_tokens=100,
                               output_tokens=7, cost_usd=0.0, provider="ollama")


class FakeThompson:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def record_outcome(self, **kwargs):
        self.calls.append(kwargs)
        return self.ok


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_group_reviews_and_select_recent_unique_subjects(tmp_path):
    rows = []
    rows += write_review(tmp_path, "r1", workers=[(WORKER_A, "a1"), (WORKER_B, "b1")],
                         arbiter_text="Review 1 is strongest", subject_name="one.py",
                         subject_text="one = 1\n")
    rows += write_review(tmp_path, "r2", workers=[(WORKER_A, "a2"), (WORKER_B, "b2")],
                         arbiter_text="Review 2 is strongest", subject_name="two.py",
                         subject_text="two = 2\n")
    # Same content hash as r2 -> a re-review, must not be replayed twice.
    rows += write_review(tmp_path, "r3", workers=[(WORKER_A, "a3"), (WORKER_B, "b3")],
                         arbiter_text="Review 1 wins", subject_name="two-copy.py",
                         subject_text="two = 2\n")
    rows += write_review(tmp_path, "r4", workers=[(WORKER_A, "a4")],
                         arbiter_text="Review 1", subject_name="three.py",
                         subject_text="three = 3\n")

    reviews = replay_arbiter.group_reviews(rows)
    selected = replay_arbiter.select_reviews(reviews, limit=5, min_workers=2)

    assert [review.review_id for review in selected] == ["r1", "r3"]
    assert [len(review.workers) for review in selected] == [2, 2]


def test_missing_subject_file_is_reported(tmp_path):
    rows = write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 2 wins")
    review = replay_arbiter.group_reviews(rows)[0]
    Path(review.subject).unlink()

    source, reason = replay_arbiter.read_subject_source(review)
    prompt = replay_arbiter.build_prompt(review, source, reason)

    assert source is None
    assert "not available" in reason
    assert "original code is not available" in prompt
    assert "SELECTED:" in prompt


def test_require_source_scans_past_stale_subjects(tmp_path):
    rows = []
    rows += write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                         arbiter_text="Review 1", subject_name="one.py",
                         subject_text="one = 1\n")
    rows += write_review(tmp_path, "r2", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                         arbiter_text="Review 1", subject_name="two.py",
                         subject_text="two = 2\n", sha_override="0" * 64)
    rows += write_review(tmp_path, "r3", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                         arbiter_text="Review 1", subject_name="three.py",
                         subject_text="three = 3\n")
    ledger = write_ledger(tmp_path, rows)

    summary = replay_arbiter.replay(
        ledger=str(ledger), limit=2, require_source=True,
        model_client=FakeModelClient(), thompson=FakeThompson())

    assert [r["review_id"] for r in summary["results"]] == ["r1", "r3"]
    assert all(r["source_status"] == "ok" for r in summary["results"])


def test_changed_subject_is_not_used_as_evidence(tmp_path):
    rows = write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 2 wins",
                        sha_override="0" * 64)
    review = replay_arbiter.group_reviews(rows)[0]

    source, reason = replay_arbiter.read_subject_source(review)

    assert source is None
    assert "sha256 mismatch" in reason


# ---------------------------------------------------------------------------
# Parsing / proxy
# ---------------------------------------------------------------------------

WORKERS = [{"model": WORKER_A}, {"model": WORKER_B}, {"model": WORKER_C}]


@pytest.mark.parametrize("text,expected", [
    (f"analysis\nSELECTED: {WORKER_B}\nVERDICT: approve", WORKER_B),
    ("SELECTED: Review 2\nVERDICT: approve", WORKER_B),
    ("SELECTED: R3\nVERDICT: reject", WORKER_C),
    ("SELECTED: worker 1", WORKER_A),
    ("**SELECTED:** " + WORKER_C, WORKER_C),
    ("<think>ignore SELECTED: " + WORKER_A + "</think>\nSELECTED: " + WORKER_B, WORKER_B),
    ("SELECTED: glm\nVERDICT: approve", WORKER_B),
    ("SELECTED: GLM 5.3", WORKER_B),
    ("SELECTED: Muse Spark Open", WORKER_A),
    ("SELECTED: DeepSeek V4.1 Flash", WORKER_C),
    ("no selection here", None),
])
def test_parse_selected(text, expected):
    assert replay_arbiter.parse_selected(text, WORKERS)[0] == expected


def test_parse_selected_returns_last_line():
    text = f"SELECTED: {WORKER_A}\nmid\nSELECTED: {WORKER_C}\n"
    assert replay_arbiter.parse_selected(text, WORKERS)[0] == WORKER_C


def test_citation_proxy_picks_most_referenced_review():
    text = ("Review 1 claims X. R1 also claims Y. Review 2 claims Z. Review 3 agrees. "
            "Reviews 1 and 2 disagree.")
    selected, counts = replay_arbiter.citation_proxy(text, WORKERS)

    assert selected == WORKER_A
    assert counts[WORKER_A] == 3  # "Review 1", "R1", and "Reviews 1"
    assert counts[WORKER_B] == 1
    assert counts[WORKER_C] == 1


def test_citation_proxy_without_references_is_none():
    assert replay_arbiter.citation_proxy("approve with fixes", WORKERS) == (
        None, {WORKER_A: 0, WORKER_B: 0, WORKER_C: 0})


# ---------------------------------------------------------------------------
# Replay flow
# ---------------------------------------------------------------------------

def test_replay_compares_with_proxy_and_feeds_thompson(tmp_path):
    rows = write_review(
        tmp_path, "r1",
        workers=[(WORKER_A, "review A text"), (WORKER_B, "review B text"),
                 (WORKER_C, "review C text")],
        # No SELECTED line (the live ledger's shape); Review 1 is referenced twice.
        arbiter_text="Review 1 found the real bug. R1 is right. Review 2 missed it.",
    )
    ledger = write_ledger(tmp_path, rows)
    model_client = FakeModelClient(f"SELECTED: {WORKER_A}\nVERDICT: approve with fixes")
    thompson = FakeThompson()

    summary = replay_arbiter.replay(
        ledger=str(ledger), limit=10, opus_source="auto",
        model_client=model_client, thompson=thompson)

    result = summary["results"][0]
    assert result["historical_selected"] is None
    assert result["proxy_selected"] == WORKER_A
    assert result["baseline_source"] == "proxy"
    assert result["local_selected"] == WORKER_A
    assert result["agreement"] is True
    assert summary["agreement"] == {"compared": 1, "agreed": 1, "rate": 1.0,
                                    "baseline_sources": ["proxy"]}
    assert model_client.calls[0]["model"] == replay_arbiter.DEFAULT_MODEL
    assert model_client.calls[0]["max_tokens"] == replay_arbiter.SELECT_MAX_TOKENS
    assert "review A text" in model_client.calls[0]["prompt"]
    assert "SELECTED:" in model_client.calls[0]["prompt"]

    # One base outcome + one agreement bonus, both $0 and tagged replay-.
    assert [call["request_id"] for call in thompson.calls] == [
        "replay-arbiter-r1", "replay-arbiter-r1-agree"]
    assert all(call["cost"] == 0.0 for call in thompson.calls)
    assert all(call["model"] == replay_arbiter.DEFAULT_MODEL for call in thompson.calls)
    assert summary["thompson"]["recorded"] == 1
    assert summary["thompson"]["agreement_bonuses"] == 1


def test_proxy_tie_is_reported(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 1 and Review 2 both found it.")
    ledger = write_ledger(tmp_path, rows)

    summary = replay_arbiter.replay(ledger=str(ledger), model_client=FakeModelClient(),
                                    thompson=FakeThompson())

    result = summary["results"][0]
    assert result["proxy_tied"] is True
    assert set(result["proxy_tie_models"]) == {WORKER_A, WORKER_B}
    assert "(citation tie:" in replay_arbiter.format_summary(summary)


def test_replay_is_idempotent_for_thompson(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 2 is strongest")
    ledger = write_ledger(tmp_path, rows)
    thompson = FakeThompson()

    first = replay_arbiter.replay(ledger=str(ledger), model_client=FakeModelClient(),
                                  thompson=thompson)
    second = replay_arbiter.replay(ledger=str(ledger), model_client=FakeModelClient(),
                                   thompson=thompson)

    assert first["thompson"]["recorded"] == 1
    assert second["thompson"]["recorded"] == 0
    assert second["thompson"]["skipped"] == 1
    assert len(thompson.calls) == 2  # base + agreement bonus only once


def test_failed_thompson_write_is_retried_later(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 2 is strongest")
    ledger = write_ledger(tmp_path, rows)
    thompson = FakeThompson(ok=False)

    summary = replay_arbiter.replay(ledger=str(ledger), model_client=FakeModelClient(),
                                    thompson=thompson)
    assert summary["thompson"]["failed"] == 1
    assert replay_arbiter.load_processed() == set()


def test_historic_source_reports_insufficient_data(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="approve with fixes")  # no SELECTED, no citations
    ledger = write_ledger(tmp_path, rows)

    summary = replay_arbiter.replay(ledger=str(ledger), opus_source="historic",
                                    model_client=FakeModelClient(), thompson=FakeThompson())

    assert summary["results"][0]["baseline_selected"] is None
    assert summary["results"][0]["agreement"] is None
    assert summary["agreement"]["rate"] is None


def test_explicit_historic_selected_is_preferred(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text=f"SELECTED: {WORKER_B}\nVERDICT: approve")
    ledger = write_ledger(tmp_path, rows)

    summary = replay_arbiter.replay(ledger=str(ledger), opus_source="auto",
                                    model_client=FakeModelClient(
                                        f"SELECTED: {WORKER_A}\nVERDICT: approve"),
                                    thompson=FakeThompson())

    result = summary["results"][0]
    assert result["baseline_source"] == "historic"
    assert result["agreement"] is False
    assert summary["agreement"]["rate"] == 0.0


def test_fresh_baseline_uses_injected_call(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="no selection")
    ledger = write_ledger(tmp_path, rows)
    seen = {}

    def baseline(model, prompt):
        seen["model"] = model
        return {"model": model, "text": f"SELECTED: {WORKER_B}\nVERDICT: approve"}

    summary = replay_arbiter.replay(
        ledger=str(ledger), opus_source="fresh", baseline_call=baseline,
        model_client=FakeModelClient(f"SELECTED: {WORKER_B}\nVERDICT: approve"),
        thompson=FakeThompson())

    assert seen["model"] == replay_arbiter.DEFAULT_BASELINE_MODEL
    assert summary["results"][0]["baseline_source"] == "fresh"
    assert summary["results"][0]["agreement"] is True


def test_fresh_baseline_failure_is_reported_not_fatal(tmp_path):
    rows = write_review(tmp_path, "r1",
                        workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="no selection")
    ledger = write_ledger(tmp_path, rows)

    def baseline(model, prompt):
        raise RuntimeError("gateway down")

    summary = replay_arbiter.replay(
        ledger=str(ledger), opus_source="fresh", baseline_call=baseline,
        model_client=FakeModelClient(), thompson=FakeThompson())

    result = summary["results"][0]
    assert result["baseline_source"] == "fresh_failed"
    assert "gateway down" in result["fresh_baseline_error"]
    assert summary["agreement"]["rate"] is None


def test_local_arbiter_failure_does_not_abort_other_subjects(tmp_path):
    rows = []
    rows += write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                         arbiter_text="Review 1", subject_name="one.py",
                         subject_text="one = 1\n")
    rows += write_review(tmp_path, "r2", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                         arbiter_text="Review 1", subject_name="two.py",
                         subject_text="two = 2\n")
    ledger = write_ledger(tmp_path, rows)

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def call_model_response(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ollama is not reachable")
            return SimpleNamespace(text=f"SELECTED: {WORKER_A}\nVERDICT: approve",
                                   latency_ms=1, input_tokens=1, output_tokens=1,
                                   cost_usd=0.0, provider="ollama")

    summary = replay_arbiter.replay(ledger=str(ledger), model_client=FlakyClient(),
                                    thompson=FakeThompson())

    assert len(summary["results"]) == 2
    assert "not reachable" in summary["results"][0]["error"]
    assert summary["results"][1]["local_selected"] == WORKER_A


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path):
    rows = write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 1")
    ledger = write_ledger(tmp_path, rows)

    class Explode:
        def call_model_response(self, **kwargs):
            raise AssertionError("dry run must not call a model")

        def record_outcome(self, **kwargs):
            raise AssertionError("dry run must not write Thompson state")

    summary = replay_arbiter.replay(ledger=str(ledger), dry_run=True,
                                    model_client=Explode(), thompson=Explode())

    assert summary["dry_run"] is True
    assert summary["considered"] == 1
    assert summary["dry_run_plan"][0]["review_id"] == "r1"
    assert not Path(replay_arbiter.STATE).exists()


def test_format_summary_mentions_agreement_and_thompson(tmp_path):
    rows = write_review(tmp_path, "r1", workers=[(WORKER_A, "a"), (WORKER_B, "b")],
                        arbiter_text="Review 1")
    ledger = write_ledger(tmp_path, rows)

    summary = replay_arbiter.replay(
        ledger=str(ledger), model_client=FakeModelClient(
            f"SELECTED: {WORKER_A}\nVERDICT: approve"),
        thompson=FakeThompson())
    text = replay_arbiter.format_summary(summary)

    assert "arbiter replay:" in text
    assert "agreement: 1/1 (100.0%)" in text
    assert "thompson: recorded=1" in text
