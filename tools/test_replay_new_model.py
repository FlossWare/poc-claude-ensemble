#!/usr/bin/env python3
"""Tests for tools/replay_new_model.py.

The Thompson stand-in is a real loopback HTTP server speaking the same
``/thompson`` JSON protocol as the live daemon, so the production
``ThompsonClient`` path (get_state, record_outcome, circuit breaker) is what
runs here.  The gateway is a deterministic in-process callable, so no test
touches the network beyond loopback.

Run with pytest, or standalone: python3 tools/test_replay_new_model.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import replay_new_model as replay  # noqa: E402
from thompson_client import ThompsonClient  # noqa: E402

NEW_MODEL = "rits/new-model-id"
GLM = "rits/zai-org/glm-5-3"
MUSE = "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
ARBITER = "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class FakeThompson:
    """Loopback HTTP Thompson stand-in that keeps real success/failure counters."""

    def __init__(self, state=None):
        self.state = state if state is not None else {"last_updated": "2026-11-19T00:00:00Z", "models": {}}
        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _handler(fake):  # noqa: N805 - closure factory, not a method
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(body)
                raw = json.dumps(fake.reply(body)).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        return Handler

    def reply(self, body):
        action = body.get("action")
        if action == "ping":
            return {"ok": True, "message": "pong"}
        if action == "get_state":
            return {"ok": True, "state": self.state}
        if action == "record_outcome":
            model = body["model"]
            entry = self.state["models"].setdefault(
                model, {"model_name": model, "successes": 0, "failures": 0, "calls": 0,
                        "total_cost": 0.0, "total_tokens": 0}
            )
            entry["successes" if body["success"] else "failures"] += 1
            entry["calls"] += 1
            entry["total_cost"] += body.get("cost") or 0.0
            entry["total_tokens"] += body.get("tokens") or 0
            return {"ok": True}
        return {"ok": False, "error": f"unknown action {action!r}"}

    def client(self):
        return ThompsonClient(service_url=self.url, timeout=2.0)

    def records(self):
        return [request for request in self.requests if request.get("action") == "record_outcome"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def fake_thompson():
    fake = FakeThompson()
    yield fake
    fake.close()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def worker_row(review_id, subject, source, model, *, recorded_at="2026-11-01T00:00:00Z",
               text=None, **overrides):
    row = {
        "schema_version": 1,
        "producer": "multi-model-review",
        "kind": "worker_result",
        "role": "worker",
        "review_id": review_id,
        "recorded_at": recorded_at,
        "subject": str(subject),
        "subject_sha256": sha(source),
        "subject_review_marker": True,
        "model": model,
        "text": text if text is not None else f"review from {model}",
        "latency_ms": 1000,
        "stop_reason": "end_turn",
        "input_tokens": 100,
        "output_tokens": 500,
        "cost_usd": 0.01,
    }
    row.update(overrides)
    return row


def write_ledger(tmp_path, rows):
    path = tmp_path / "ledger" / "reviews" / "reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_source(tmp_path, name="target.py", text="print('hello')\n"):
    path = tmp_path / "src" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def ledger_bytes(path):
    return path.read_bytes()


class Gateway:
    """Deterministic gateway: records calls and returns scripted results."""

    def __init__(self, *, worker_text="new model findings", selected=NEW_MODEL,
                 worker_error=None):
        self.worker_text = worker_text
        self.selected = selected
        self.worker_error = worker_error
        self.calls = []

    def __call__(self, model, prompt):
        self.calls.append((model, prompt))
        if model != ARBITER:
            if self.worker_error:
                return {"model": model, "text": "", "latency_ms": None, "stop_reason": None,
                        "input_tokens": None, "output_tokens": None, "cost_usd": None,
                        "error": self.worker_error}
            return {"model": model, "text": self.worker_text, "latency_ms": 2500,
                    "stop_reason": "end_turn", "input_tokens": 100, "output_tokens": 300,
                    "cost_usd": 0.02}
        text = f"confirmed defects...\nSELECTED: {self.selected}\nVERDICT: approve"
        return {"model": ARBITER, "text": text, "latency_ms": 800, "stop_reason": "end_turn",
                "input_tokens": 900, "output_tokens": 200, "cost_usd": 0.05}

    def worker_calls(self):
        return [call for call in self.calls if call[0] != ARBITER]


def one_subject_ledger(tmp_path, source_text="print('hello')\n", history_models=(GLM, MUSE),
                       review_id="abc123", recorded_at="2026-11-01T00:00:00Z"):
    source_path = write_source(tmp_path, text=source_text)
    rows = [
        worker_row(review_id, source_path, source_text, model, recorded_at=recorded_at)
        for model in history_models
    ]
    ledger = write_ledger(tmp_path, rows)
    return ledger, source_path, rows


def invoke(tmp_path, ledger, gateway=None, client=None, **kwargs):
    kwargs.setdefault("model", NEW_MODEL)
    out = io.StringIO()
    summary = replay.run(ledger=str(ledger), call_model=gateway, client=client, out=out, **kwargs)
    return summary, out.getvalue()


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------


def test_subjects_group_by_hash_and_use_the_most_recent_review(tmp_path):
    source = "same bytes\n"
    path = write_source(tmp_path, text=source)
    rows = [
        worker_row("older", path, source, GLM, recorded_at="2026-11-01T00:00:00Z"),
        worker_row("newer", path, source, GLM, recorded_at="2026-11-05T00:00:00Z"),
        worker_row("newer", path, source, MUSE, recorded_at="2026-11-05T00:00:00Z"),
    ]
    other = write_source(tmp_path, name="other.py", text="other\n")
    rows.append(worker_row("other", other, "other\n", GLM, recorded_at="2026-11-03T00:00:00Z"))

    subjects = replay.select_subjects(rows, 10)

    assert [subject.subject for subject in subjects] == [str(path), str(other)]
    first = subjects[0]
    assert first.source_review_id == "newer"
    assert first.recorded_at == "2026-11-05T00:00:00Z"
    assert {row["model"] for row in first.historical} == {GLM, MUSE}
    assert len(replay.select_subjects(rows, 1)) == 1


def test_replay_rows_never_become_subjects(tmp_path):
    source = "x\n"
    path = write_source(tmp_path, text=source)
    rows = [worker_row("organic", path, source, GLM)]
    rows.append({
        "producer": replay.PRODUCER, "kind": "worker_result", "review_id": "replay-x",
        "subject": str(path), "subject_sha256": sha(source), "model": NEW_MODEL,
        "replay_model": NEW_MODEL, "recorded_at": "2026-11-09T00:00:00Z", "text": "replayed",
    })
    rows.append(worker_row("organic-2", path, "different\n", GLM))
    subjects = replay.select_subjects(rows, 10)
    # The replay row's review is ignored; only the organic review of this source remains.
    assert len(subjects) == 2
    assert all(subject.source_review_id != "replay-x" for subject in subjects)


def test_missing_and_drifted_sources_are_skipped_with_explicit_reasons(tmp_path):
    present = write_source(tmp_path, name="present.py", text="present\n")
    missing = tmp_path / "src" / "gone.py"
    drifted = write_source(tmp_path, name="drifted.py", text="changed since review\n")
    rows = [
        worker_row("r-present", present, "present\n", GLM),
        worker_row("r-missing", missing, "was here\n", GLM),
        worker_row("r-drifted", drifted, "the reviewed bytes\n", GLM),
    ]

    summary, output = invoke(tmp_path, write_ledger(tmp_path, rows), dry_run=True)

    reasons = {record["subject"]: record.get("reason") for record in summary["records"]}
    assert reasons[str(missing)] == "source file missing"
    assert "hash mismatch" in reasons[str(drifted)]
    assert reasons.get(str(present)) == "dry-run"
    assert sum(record["status"] == replay.STATUS_SKIPPED for record in summary["records"]) == 2
    assert "source file missing" in output


def test_source_reconstruction_reproduces_the_original_prompt_input(tmp_path):
    full = ("line\n" * 4000)  # 20000 chars, longer than the 6000 default budget
    historical_source = replay.truncate_source(full, 6000)
    path = write_source(tmp_path, text=full)
    row = worker_row("r", path, historical_source, GLM)
    subject = replay.select_subjects([row], 1)[0]

    source, reason = replay.resolve_source(subject)

    assert reason is None
    assert source == historical_source
    assert hashlib.sha256(source.encode()).hexdigest() == row["subject_sha256"]
    # The new model receives the same prompt the historical workers received.
    from multi_model_review import review_prompt

    assert review_prompt(source, str(path)) == review_prompt(historical_source, str(path))


# ---------------------------------------------------------------------------
# Replay flow and Thompson
# ---------------------------------------------------------------------------


def test_replay_selected_records_base_and_bonus(tmp_path, fake_thompson):
    ledger, source_path, _rows = one_subject_ledger(tmp_path)
    gateway = Gateway(selected=NEW_MODEL)

    summary, output = invoke(tmp_path, ledger, gateway=gateway, client=fake_thompson.client(),
                             reviews=5)

    assert summary["replayed"] == 1
    assert summary["selected_new"] == 1
    assert summary["adjudicated"] == 1
    # base + quality-weighted bonus, both on the new model's arm.
    arm = fake_thompson.state["models"]["new-model-id"]
    assert (arm["successes"], arm["failures"], arm["calls"]) == (2, 0, 2)
    assert summary["thompson"]["before"]["present"] is False
    assert summary["thompson"]["before"]["mean"] == 0.5
    assert summary["thompson"]["after"]["successes"] == 2
    assert summary["thompson"]["after"]["mean"] == pytest.approx(3 / 4)
    review_id = f"replay-{replay.model_slug(NEW_MODEL)}-{summary['subjects'][0][:12]}"
    request_ids = [record["request_id"] for record in fake_thompson.records()]
    assert request_ids == [f"{review_id}-base", f"{review_id}-bonus"]
    assert all(request_id.startswith("replay-") for request_id in request_ids)
    # The worker prompt is the multi-model-review prompt for this source.
    worker_model, worker_prompt = gateway.worker_calls()[0]
    assert worker_model == NEW_MODEL
    assert source_path.read_text() in worker_prompt
    assert "review from" not in worker_prompt
    # Both ledger rows are present, tagged and linked.
    rows = replay.load_rows(str(ledger))
    replay_rows = [row for row in rows if row.get("producer") == replay.PRODUCER]
    assert {row["kind"] for row in replay_rows} == {"worker_result", "arbiter_decision"}
    worker = next(row for row in replay_rows if row["kind"] == "worker_result")
    arbiter = next(row for row in replay_rows if row["kind"] == "arbiter_decision")
    assert worker["request_id"].startswith("replay-")
    assert worker["review_id"] == worker["request_id"].rsplit("-worker", 1)[0]
    assert worker["subject_sha256"] == sha(source_path.read_text())
    assert arbiter["selected"] == NEW_MODEL
    assert arbiter["selected_model"] == NEW_MODEL
    assert arbiter["selected_replay_model"] is True
    assert arbiter["verdict"] == "approve"
    assert "SELECTED:" in output


def test_replay_not_selected_records_base_outcome_only(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    gateway = Gateway(selected="GLM 5.3")

    summary, _output = invoke(tmp_path, ledger, gateway=gateway, client=fake_thompson.client())

    assert summary["replayed"] == 1
    assert summary["selected_new"] == 0
    assert summary["adjudicated"] == 1
    arm = fake_thompson.state["models"]["new-model-id"]
    assert (arm["successes"], arm["failures"], arm["calls"]) == (1, 0, 1)
    arbiter = next(row for row in replay.load_rows(str(ledger))
                   if row.get("kind") == "arbiter_decision" and row.get("producer") == replay.PRODUCER)
    assert arbiter["selected_model"] == GLM
    assert arbiter["selected_replay_model"] is False


def test_gateway_cost_is_priced_when_the_call_omits_it(tmp_path, fake_thompson, monkeypatch):
    """multi_model_review.call returns tokens only; the replay must price its own runs."""
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    monkeypatch.setattr(replay.review_tool, "cost_usd", lambda model, inputs, outputs: 0.123)

    class NoCostGateway(Gateway):
        def __call__(self, model, prompt):
            result = super().__call__(model, prompt)
            result.pop("cost_usd")
            return result

    summary, _output = invoke(tmp_path, ledger, gateway=NoCostGateway(selected=NEW_MODEL),
                              client=fake_thompson.client())

    worker = next(row for row in replay.load_rows(str(ledger))
                  if row.get("kind") == "worker_result" and row.get("producer") == replay.PRODUCER)
    arbiter = next(row for row in replay.load_rows(str(ledger))
                   if row.get("kind") == "arbiter_decision" and row.get("producer") == replay.PRODUCER)
    assert worker["cost_usd"] == pytest.approx(0.123)
    assert arbiter["cost_usd"] == pytest.approx(0.123)
    assert summary["comparison"][0]["avg_cost_usd"] == pytest.approx(0.123)
    recorded = fake_thompson.records()
    assert all(request["cost"] == pytest.approx(0.123) for request in recorded)


def test_worker_failure_records_a_thompson_failure(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    gateway = Gateway(worker_error="gateway exploded")

    summary, _output = invoke(tmp_path, ledger, gateway=gateway, client=fake_thompson.client())

    assert summary["failed"] == 1
    assert summary["replayed"] == 0
    arm = fake_thompson.state["models"]["new-model-id"]
    assert (arm["successes"], arm["failures"]) == (0, 1)
    rows = [row for row in replay.load_rows(str(ledger)) if row.get("producer") == replay.PRODUCER]
    assert [row["kind"] for row in rows] == ["worker_result"]
    assert rows[0]["error"] == "gateway exploded"
    assert summary["exit_code"] == 0


def test_arbiter_failure_keeps_the_worker_outcome(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)

    def gateway(model, prompt):
        if model == ARBITER:
            raise RuntimeError("arbiter down")
        return {"model": model, "text": "findings", "latency_ms": 100, "stop_reason": "end_turn",
                "input_tokens": 10, "output_tokens": 10, "cost_usd": 0.001}

    summary, _output = invoke(tmp_path, ledger, gateway=gateway, client=fake_thompson.client())

    assert summary["records"][0]["selection_status"] == "arbiter_error"
    assert fake_thompson.state["models"]["new-model-id"]["successes"] == 1
    arbiter = next(row for row in replay.load_rows(str(ledger))
                   if row.get("kind") == "arbiter_decision" and row.get("producer") == replay.PRODUCER)
    assert arbiter["error"] == "RuntimeError: arbiter down"


def test_dry_run_makes_no_calls_and_no_writes(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    before = ledger_bytes(ledger)

    def explode(model, prompt):
        raise AssertionError("dry-run must not call the gateway")

    summary, output = invoke(tmp_path, ledger, gateway=explode, dry_run=True)

    assert summary["replayed"] == 1
    assert summary["dry_run"] is True
    assert ledger_bytes(ledger) == before
    assert fake_thompson.requests == []
    assert not (ledger.parent / "replay-state.json").exists()
    assert not (ledger.parent / "thompson-sync-state.json").exists()
    assert "no API calls, no ledger writes, no Thompson writes" in output
    assert "thompson: not queried or written (dry-run)" in output


def test_second_run_is_idempotent(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    gateway = Gateway(selected=NEW_MODEL)
    client = fake_thompson.client()

    first, _output = invoke(tmp_path, ledger, gateway=gateway, client=client)
    rows_after_first = ledger_bytes(ledger)
    calls_after_first = len(gateway.calls)
    state_after_first = json.loads(json.dumps(fake_thompson.state["models"]))

    second, output = invoke(tmp_path, ledger, gateway=gateway, client=client)

    assert first["replayed"] == 1
    assert second["replayed"] == 0
    assert second["skipped"][0]["reason"].startswith("already replayed")
    assert len(gateway.calls) == calls_after_first          # no new gateway calls
    assert ledger_bytes(ledger) == rows_after_first          # no new ledger rows
    assert fake_thompson.state["models"] == state_after_first  # no double-counted outcomes
    assert "already replayed" in output


def test_interrupted_outcome_write_is_finalized_from_the_ledger(tmp_path, fake_thompson):
    """A crash after the ledger write leaves the outcomes pending; the re-run retries them."""
    ledger, source_path, _rows = one_subject_ledger(tmp_path)
    subject_sha = sha(source_path.read_text())
    review_id = f"replay-{replay.model_slug(NEW_MODEL)}-{subject_sha[:12]}"
    with open(ledger, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema_version": 1, "producer": replay.PRODUCER, "kind": "worker_result",
            "review_id": review_id, "request_id": f"{review_id}-worker",
            "recorded_at": "2026-11-02T00:00:00Z", "subject": str(source_path),
            "subject_sha256": subject_sha, "replay": True, "replay_model": NEW_MODEL,
            "model": NEW_MODEL, "text": "stored replay output", "latency_ms": 1234,
            "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 20,
            "cost_usd": 0.002,
        }) + "\n")
        handle.write(json.dumps({
            "schema_version": 1, "producer": replay.PRODUCER, "kind": "arbiter_decision",
            "review_id": review_id, "request_id": f"{review_id}-arbiter", "model": ARBITER,
            "subject": str(source_path), "subject_sha256": subject_sha,
            "selected": NEW_MODEL, "selected_model": NEW_MODEL,
            "selection_status": "selected", "verdict": "approve",
        }) + "\n")

    def explode(model, prompt):
        raise AssertionError("a finalize run must not call the gateway")

    summary, _output = invoke(tmp_path, ledger, gateway=explode, client=fake_thompson.client())

    assert summary["records"][0]["status"] == replay.STATUS_FINALIZED
    arm = fake_thompson.state["models"]["new-model-id"]
    assert (arm["successes"], arm["failures"]) == (2, 0)  # base + bonus from stored rows
    synced = json.loads((ledger.parent / "thompson-sync-state.json").read_text())["review_ids"]
    assert review_id in synced


def test_replay_review_ids_are_marked_synced(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    invoke(tmp_path, ledger, gateway=Gateway(selected="GLM 5.3"), client=fake_thompson.client())

    synced = json.loads((ledger.parent / "thompson-sync-state.json").read_text())["review_ids"]
    assert len(synced) == 1
    assert synced[0].startswith("replay-")


def test_thompson_unavailable_reports_failure_and_keeps_rows(tmp_path):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    dead = ThompsonClient(service_url="http://127.0.0.1:9", timeout=0.2)

    summary, _output = invoke(tmp_path, ledger, gateway=Gateway(selected=NEW_MODEL), client=dead)

    assert summary["exit_code"] == 1
    assert summary["records"][0]["thompson"]["base"] == "failed"
    assert any(row.get("producer") == replay.PRODUCER for row in replay.load_rows(str(ledger)))
    assert not (ledger.parent / "replay-state.json").exists()


# ---------------------------------------------------------------------------
# Comparison table and predictors
# ---------------------------------------------------------------------------


def test_comparison_table_compares_new_model_with_historical_workers(tmp_path, fake_thompson):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    summary, output = invoke(tmp_path, ledger, gateway=Gateway(selected=NEW_MODEL),
                             client=fake_thompson.client())

    table = {row["model"]: row for row in summary["comparison"]}
    assert f"{NEW_MODEL} (replay)" in table
    replay_row = table[f"{NEW_MODEL} (replay)"]
    assert (replay_row["runs"], replay_row["usable"], replay_row["selected"]) == (1, 1, 1)
    assert replay_row["avg_cost_usd"] == pytest.approx(0.02)
    assert replay_row["avg_latency_ms"] == pytest.approx(2500)
    assert table[GLM]["selected"] == 0
    assert table[GLM]["avg_cost_usd"] == pytest.approx(0.01)
    assert "comparison" in output and "avg_latency_ms" in output


def test_retrain_predictors_runs_only_when_requested(tmp_path, fake_thompson, monkeypatch):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    calls = []

    def fake_retrain(model, records):
        calls.append((model, [record["review_id"] for record in records]))
        return {"quality_predictor": {"exit_code": 0}, "routing": {}, "predictions": {}}

    monkeypatch.setattr(replay, "retrain_predictors", fake_retrain)

    invoke(tmp_path, ledger, gateway=Gateway(selected=NEW_MODEL), client=fake_thompson.client())
    assert calls == []

    invoke(tmp_path, ledger, gateway=Gateway(selected=NEW_MODEL), client=fake_thompson.client(),
           retrain=True)
    # The second run skipped the already-replayed subject, so there is nothing
    # new to retrain on -- but the flag still reaches the retrainer.
    assert calls == [(NEW_MODEL, [])]


# ---------------------------------------------------------------------------
# Selection parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,models,expected",
    [
        ("GLM 5.3", [GLM, MUSE], GLM),
        ("rits/zai-org/glm-5-3", [GLM, MUSE], GLM),
        ("zai-org/glm-5-3", [GLM, MUSE], GLM),
        ("glm-5-3", [GLM, MUSE], GLM),
        ("DeepSeek V4.1 Flash", ["rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"],
         "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        ("rits/new-model-id", [NEW_MODEL, GLM], NEW_MODEL),
        ("new-model-id", [NEW_MODEL, GLM], NEW_MODEL),
        ("GLM 5.3", ["rits/fork/glm-5-3", "other/glm-5-3-beta"], None),
        ("zai-org/glm-5-3", [GLM, "other/zai-org/glm-5-3"], None),
        ("something else", [GLM, MUSE], None),
        (None, [GLM], None),
    ],
)
def test_resolve_selected(label, models, expected):
    model, status = replay.resolve_selected(label, models)
    assert model == expected
    if expected is None and label:
        assert status in ("unmatched", "ambiguous")
    if not label:
        assert status == "no_selected_line"


def test_parse_selected_uses_the_last_line():
    text = "SELECTED: GLM 5.3\nnoise\nSELECTED: DeepSeek V4.1 Flash\nVERDICT: approve"
    assert replay.parse_selected(text) == "DeepSeek V4.1 Flash"
    assert replay.parse_selected("VERDICT: approve") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_model_that_is_also_the_arbiter_is_rejected(tmp_path):
    ledger, _source, _rows = one_subject_ledger(tmp_path)
    summary, _output = invoke(tmp_path, ledger, model=ARBITER, arbiter="opus", dry_run=True)
    assert summary["exit_code"] == 2
    assert "cannot adjudicate its own replay" in summary["failure"]


def test_main_requires_a_model():
    with pytest.raises(SystemExit) as error:
        replay.main([])
    assert error.value.code == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
