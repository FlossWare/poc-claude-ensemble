#!/usr/bin/env python3
"""Tests for tools/crew_select.py and the multi_model_review --thompson flag.

The fake Thompson service is a real loopback HTTP server speaking the same
/Thompson JSON protocol as the live daemon, so the production ThompsonClient
path (ping, get_state, select_model, circuit breaker) is what runs here.
Draws are a fixed queue, which makes rejection sampling deterministic.

Run with pytest, or standalone: python3 tools/test_crew_select.py
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Keep the posterior snapshot on the default prior and away from any live
# policy file, regardless of the invoking shell.
os.environ.setdefault("POC_CLAUDE_ENSEMBLE_THOMPSON_POLICY_FILE",
                      "/nonexistent/crew-select-tests/thompson-policy.json")

import crew_select  # noqa: E402
from thompson_client import ThompsonClient  # noqa: E402

# Arm counters shaped like the live thompson-sampling-state.json, scaled down
# where the exact numbers do not matter.
STATE = {
    "last_updated": "2026-11-19T05:40:47.956724+00:00",
    "models": {
        "deepseek-v4.1-flash": {"model_name": "deepseek-v4.1-flash", "successes": 59, "failures": 11,
                                "calls": 70, "total_cost": 71.225601, "total_tokens": 4332182},
        "glm-5-3": {"model_name": "glm-5-3", "successes": 51, "failures": 13,
                    "calls": 64, "total_cost": 70.102863, "total_tokens": 3911565},
        "haiku": {"model_name": "haiku", "successes": 3005, "failures": 388,
                  "calls": 3393, "total_cost": 113.726052, "total_tokens": 799170109},
        "muse-spark-open": {"model_name": "muse-spark-open", "successes": 16, "failures": 0,
                            "calls": 16, "total_cost": 0.958995, "total_tokens": 21405},
        "opus": {"model_name": "opus", "successes": 34, "failures": 11,
                 "calls": 45, "total_cost": 15.966806, "total_tokens": 1733598},
        "sonnet": {"model_name": "sonnet", "successes": 35, "failures": 4,
                   "calls": 39, "total_cost": 5.512225, "total_tokens": 1510511},
    },
}


class FakeThompson:
    """Loopback HTTP Thompson stand-in with a scripted draw sequence."""

    def __init__(self, draws, state=None, fail_select_at=None):
        self.draws = list(draws)
        self.state = state if state is not None else STATE
        self.fail_select_at = fail_select_at
        self.requests = []
        self._draw_index = 0
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
                payload = fake.reply(body)
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200 if payload.get("ok") else 500)
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
        if action == "select_model":
            count = sum(1 for request in self.requests if request.get("action") == "select_model")
            if self.fail_select_at is not None and count >= self.fail_select_at:
                return {"ok": False, "error": "fake failure"}
            if not self.draws:
                raise AssertionError("fake Thompson ran out of scripted draws")
            index = min(self._draw_index, len(self.draws) - 1)
            self._draw_index += 1
            return {"ok": True, "model": self.draws[index]}
        return {"ok": False, "error": f"unknown action {action!r}"}

    def select_requests(self):
        return [request for request in self.requests if request.get("action") == "select_model"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def make_fake():
    fakes = []

    def build(*args, **kwargs):
        fake = FakeThompson(*args, **kwargs)
        fakes.append(fake)
        return fake

    yield build
    for fake in fakes:
        fake.close()


def select(fake, tmp_path, **kwargs):
    kwargs.setdefault("count", 3)
    kwargs.setdefault("arbiter", "claude-opus-5-5")
    kwargs.setdefault("task_type", "code-review")
    ledger_dir = kwargs.pop("ledger_dir", tmp_path / "crews")
    return crew_select.select_crew(
        client=ThompsonClient(service_url=fake.url, timeout=2.0),
        ledger_dir=ledger_dir,
        **kwargs,
    )


def ledger_rows(tmp_path):
    files = sorted((tmp_path / "crews").glob("*.jsonl"))
    rows = []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, f"expected one row per selection in {path}"
        rows.append(json.loads(lines[0]))
    return files, rows


# ---------------------------------------------------------------------------
# Draw without replacement
# ---------------------------------------------------------------------------

def test_duplicate_draws_are_rejected_until_arms_are_distinct(make_fake, tmp_path):
    fake = make_fake(["glm-5-3", "glm-5-3", "deepseek-v4.1-flash", "glm-5-3", "muse-spark-open"])

    record = select(fake, tmp_path)

    assert record["status"] == "thompson"
    assert [worker["arm"] for worker in record["workers"]] == [
        "glm-5-3", "deepseek-v4.1-flash", "muse-spark-open"]
    assert record["models"] == [
        "rits/zai-org/glm-5-3",
        "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    ]
    assert all(worker["source"] == "thompson_draw" for worker in record["workers"])
    assert record["draws"] == {"total": 5, "collisions": 2, "per_slot": [1, 2, 2]}
    assert record["fill_used"] is False


def test_arbiter_is_never_drawn_into_the_crew(make_fake, tmp_path):
    fake = make_fake(["opus", "opus", "glm-5-3", "opus",
                      "deepseek-v4.1-flash", "muse-spark-open"])

    record = select(fake, tmp_path)

    arms = [worker["arm"] for worker in record["workers"]]
    assert arms == ["glm-5-3", "deepseek-v4.1-flash", "muse-spark-open"]
    assert "opus" not in arms
    assert record["excluded"] == ["opus"]
    assert record["arbiter"] == {"configured": "claude-opus-5-5", "arm": "opus", "excluded": True}
    assert record["draws"]["collisions"] == 3
    # The rejection is client-side: every draw still went through select_model.
    assert len(fake.select_requests()) == 6


def test_posterior_fill_when_one_arm_keeps_winning(make_fake, tmp_path):
    fake = make_fake(["glm-5-3"])  # exhausts to the last entry: always glm

    record = select(fake, tmp_path, count=2, max_draws_per_slot=3)

    # muse-spark-open has the highest posterior mean among the arms still available.
    assert [worker["arm"] for worker in record["workers"]] == ["glm-5-3", "muse-spark-open"]
    assert record["workers"][1]["source"] == "posterior_fill"
    assert record["fill_used"] is True
    assert record["draws"] == {"total": 4, "collisions": 3, "per_slot": [1, 3]}


def test_crew_is_short_rather_than_duplicated_when_arms_run_out(make_fake, tmp_path):
    small_state = {"last_updated": "2026-11-19T00:00:00Z", "models": {
        "glm-5-3": {"successes": 5, "failures": 0, "calls": 5, "total_cost": 0.1},
        "opus": {"successes": 1, "failures": 1, "calls": 2, "total_cost": 0.2},
    }}
    fake = make_fake(["glm-5-3"], state=small_state)

    record = select(fake, tmp_path, count=3, max_draws_per_slot=2)

    assert [worker["arm"] for worker in record["workers"]] == ["glm-5-3"]
    assert record["status"] == "thompson"
    assert record["reason"] == "insufficient_eligible_arms"
    assert record["truncated"] is True
    assert record["eligible"] == ["glm-5-3"]  # opus excluded as arbiter


# ---------------------------------------------------------------------------
# Fallback is documented, logged, and recorded
# ---------------------------------------------------------------------------

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_unavailable_thompson_uses_logged_default_lineup(tmp_path, caplog):
    client = ThompsonClient(service_url=f"http://127.0.0.1:{free_port()}", timeout=0.5)
    caplog.set_level(logging.WARNING, logger="crew_select")

    record = crew_select.select_crew(task_type="code-review", count=3,
                                     arbiter="claude-opus-5-5", client=client,
                                     ledger_dir=tmp_path / "crews")

    assert record["status"] == "fallback"
    assert record["reason"] == "thompson_unavailable"
    assert record["models"] == crew_select.DEFAULT_LINEUP
    assert [worker["source"] for worker in record["workers"]] == ["fallback_default"] * 3
    assert record["posterior"] is None
    warning = " ".join(message.getMessage() for message in caplog.records)
    assert "thompson_unavailable" in warning
    assert "documented default crew" in warning

    files, rows = ledger_rows(tmp_path)
    assert len(files) == 1
    assert rows[0]["status"] == "fallback"
    assert rows[0]["reason"] == "thompson_unavailable"

    # The fallback still keeps a named arbiter out of the crew.
    assert "opus" not in [worker["arm"] for worker in record["workers"]]


def test_failure_mid_selection_falls_back_instead_of_mixing(make_fake, tmp_path, caplog):
    fake = make_fake(["muse-spark-open"], fail_select_at=2)
    caplog.set_level(logging.WARNING, logger="crew_select")

    record = select(fake, tmp_path)

    assert record["status"] == "fallback"
    assert record["reason"] == "thompson_error"
    assert record["models"] == crew_select.DEFAULT_LINEUP  # no partial Thompson crew
    assert record["draws"]["total"] == 2
    assert "thompson_error" in " ".join(message.getMessage() for message in caplog.records)


def test_partial_crew_failure_still_falls_back_to_full_default(make_fake, tmp_path):
    fake = make_fake(["muse-spark-open"], fail_select_at=2)

    record = select(fake, tmp_path, count=2)

    assert len(record["models"]) == 2
    assert record["models"] == crew_select.DEFAULT_LINEUP[:2]


# ---------------------------------------------------------------------------
# Exposure ledger
# ---------------------------------------------------------------------------

def test_selection_is_recorded_with_sources_and_posterior(make_fake, tmp_path):
    fake = make_fake(["muse-spark-open", "glm-5-3", "deepseek-v4.1-flash"])

    record = select(fake, tmp_path, correlation_id="review-abc123",
                    required_capability=0.5, max_cost=float("inf"))

    files, rows = ledger_rows(tmp_path)
    assert len(files) == 1
    assert files[0].name.endswith(".jsonl")
    row = rows[0]
    assert row == json.loads(json.dumps(record))  # the returned row is the written row
    assert row["kind"] == "crew_selection"
    assert row["correlation_id"] == "review-abc123"
    assert row["task_type"] == "code-review"
    assert row["arbiter"]["arm"] == "opus"
    assert row["max_cost"] is None  # infinity is recorded as "unlimited", not invalid JSON
    assert [worker["arm"] for worker in row["workers"]] == [
        "muse-spark-open", "glm-5-3", "deepseek-v4.1-flash"]
    assert [worker["source"] for worker in row["workers"]] == ["thompson_draw"] * 3

    posterior = row["posterior"]
    prior = crew_select._prior_strength("code-review")
    assert posterior["prior_strength"] == prior
    assert posterior["state_last_updated"] == STATE["last_updated"]
    assert posterior["arms"]["deepseek-v4.1-flash"] == {
        "successes": 59, "failures": 11, "calls": 70,
        "avg_cost": round(71.225601 / 70, 8),
        "alpha": 59 + prior, "beta": 11 + prior,
        "posterior_mean": round((59 + prior) / (70 + 2 * prior), 6),
    }

    mode = files[0].stat().st_mode & 0o777
    assert mode == 0o600


def test_arm_names_close_the_loop_with_ledger_to_thompson():
    """Every crew arm maps to the same arm name the outcome sync records."""
    import ledger_to_thompson

    for arm, model in crew_select.ARM_MODEL_IDS.items():
        assert ledger_to_thompson.MODEL_ALIASES[model] == arm
        assert crew_select.arm_of(model) == arm


# ---------------------------------------------------------------------------
# Ollama arms
# ---------------------------------------------------------------------------

def test_ollama_models_are_registered_as_callable_arms():
    expected = {
        "ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b", "ollama/qwen3:4b",
        "ollama/qwen3:1.7b", "ollama/qwen3.5:2b", "ollama/gemma4:e2b",
        "ollama/gemma3:4b", "ollama/phi4-mini", "ollama/llama3.2:3b",
        "ollama/granite3.2-vision:2b",
    }
    assert set(crew_select.OLLAMA_ARMS) == expected
    for arm in expected:
        assert crew_select.ARM_MODEL_IDS[arm] == arm
        assert crew_select.model_of(arm) == arm
        assert crew_select.arm_of(arm) == arm


def test_crew_can_draw_local_ollama_arms(make_fake, tmp_path):
    state = json.loads(json.dumps(STATE))
    for name in ("ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b"):
        state["models"][name] = {"model_name": name, "successes": 8, "failures": 1,
                                 "calls": 9, "total_cost": 0.0, "total_tokens": 1234}
    fake = make_fake(["ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b", "glm-5-3"],
                     state=state)

    record = select(fake, tmp_path)

    assert record["status"] == "thompson"
    assert [worker["arm"] for worker in record["workers"]] == [
        "ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b", "glm-5-3"]
    assert record["models"][:2] == ["ollama/qwen2.5-coder:3b", "ollama/deepseek-r1:7b"]
    assert all(worker["mapped"] for worker in record["workers"])
    assert record["posterior"]["arms"]["ollama/deepseek-r1:7b"]["avg_cost"] == 0.0
    assert "ollama/qwen2.5-coder:3b" in record["eligible"]


def test_register_ollama_arms_is_idempotent_and_reports_created():
    class FakeClient:
        def __init__(self):
            self.known = set()
            self.calls = []

        def ensure_arms(self, models):
            self.calls.append(list(models))
            created = [model for model in models if model not in self.known]
            self.known.update(created)
            return {"ok": True, "created": created}

    client = FakeClient()
    first = crew_select.register_ollama_arms(client)
    assert first["ok"] is True
    assert first["created"] == crew_select.OLLAMA_ARMS
    assert client.calls == [crew_select.OLLAMA_ARMS]

    absent = crew_select.register_ollama_arms(client)
    assert absent == {"ok": True, "created": [], "models": crew_select.OLLAMA_ARMS}


def test_register_ollama_arms_reports_unsupported_service(caplog):
    class OldService:
        def ensure_arms(self, models):
            return None

    caplog.set_level(logging.WARNING, logger="crew_select")
    result = crew_select.register_ollama_arms(OldService())

    assert result["ok"] is False
    assert result["created"] == []
    assert "ensure_arms" in " ".join(message.getMessage() for message in caplog.records)


def test_ledger_failure_is_reported_not_swallowed(make_fake, tmp_path, monkeypatch):
    fake = make_fake(["muse-spark-open", "glm-5-3", "deepseek-v4.1-flash"])
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    record = select(fake, tmp_path, ledger_dir=blocker / "crews")

    assert "ledger_error" in record
    assert record["status"] == "thompson"
    assert record["models"]


# ---------------------------------------------------------------------------
# Task-type inference and CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("description,expected", [
    ("review the retry path for defects", "code-review"),
    ("refactor the ingest loop", "refactoring"),
    ("add pytest coverage for the parser", "testing"),
    ("find the root cause of the crash", "bug_analysis"),
    ("update the README", "documentation"),
    ("", "code-review"),
    (None, "code-review"),
])
def test_infer_task_type(description, expected):
    assert crew_select.infer_task_type(description) == expected


def test_cli_prints_lineup_and_posteriors(make_fake, tmp_path, capsys, monkeypatch):
    fake = make_fake(["muse-spark-open", "glm-5-3", "deepseek-v4.1-flash"])
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR", str(tmp_path / "crews"))
    original = ThompsonClient.__init__

    def pointed_init(self, *args, **kwargs):
        kwargs["service_url"] = fake.url
        original(self, *args, **kwargs)

    ThompsonClient.__init__ = pointed_init
    try:
        code = crew_select.main(["review of parser.py", "--task-type", "code-review",
                                 "--workers", "3", "--json"])
    finally:
        ThompsonClient.__init__ = original

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "thompson"
    assert printed["ledger_path"]


# ---------------------------------------------------------------------------
# multi_model_review --thompson integration
# ---------------------------------------------------------------------------

@pytest.fixture
def review_tool(tmp_path, monkeypatch):
    import multi_model_review

    monkeypatch.setattr(multi_model_review, "LEDGER", str(tmp_path / "reviews" / "reviews.jsonl"))
    monkeypatch.setattr(multi_model_review, "PRICING_PATH", str(tmp_path / "missing-pricing.py"))

    fake_shared = types.SimpleNamespace(thompson_shadow=types.SimpleNamespace(record=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "shared", fake_shared)
    import ledger_to_thompson
    monkeypatch.setattr(ledger_to_thompson, "sync", lambda *args, **kwargs: 0)

    calls = []

    def fake_call(model, prompt):
        calls.append(model)
        text = "VERDICT: approve" if model == multi_model_review.ARBITER else "found a defect"
        return {"model": model, "text": text, "latency_ms": 5, "stop_reason": "end_turn",
                "input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(multi_model_review, "call", fake_call)
    return multi_model_review, calls, tmp_path


def make_subject(tmp_path):
    path = tmp_path / "subject.py"
    path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return path


def test_thompson_flag_uses_drawn_crew_and_keeps_arbiter(review_tool, monkeypatch, capsys):
    review_tool_mod, calls, tmp_path = review_tool
    drawn = ["rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
             "rits/zai-org/glm-5-3",
             "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"]
    seen = {}

    def fake_select_crew(**kwargs):
        seen.update(kwargs)
        return {"status": "thompson", "reason": None, "selection_id": "sel-1",
                "ledger_path": str(tmp_path / "crews" / "x.jsonl"),
                "models": drawn,
                "workers": [{"slot": i, "arm": arm, "model": model}
                            for i, (arm, model) in enumerate(zip(
                                ["muse-spark-open", "glm-5-3", "deepseek-v4.1-flash"], drawn))]}

    monkeypatch.setattr(crew_select, "select_crew", fake_select_crew)

    review_tool_mod.main([str(make_subject(tmp_path)), "--thompson"])

    assert seen["task_type"] == "code-review"
    assert seen["count"] == 3
    assert seen["arbiter"] == review_tool_mod.ARBITER
    assert seen["correlation_id"]
    assert calls == drawn + [review_tool_mod.ARBITER]

    ledger = Path(review_tool_mod.LEDGER)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    worker_rows = [row for row in rows if row["kind"] == "worker_result"]
    assert [row["model"] for row in worker_rows] == drawn
    arbiter_rows = [row for row in rows if row["kind"] == "arbiter_decision"]
    assert [row["model"] for row in arbiter_rows] == [review_tool_mod.ARBITER]
    assert "crew (thompson):" in capsys.readouterr().out


def test_default_mode_keeps_hardcoded_workers(review_tool, monkeypatch):
    review_tool_mod, calls, tmp_path = review_tool

    def explode(**kwargs):
        raise AssertionError("crew_select must not run without --thompson")

    monkeypatch.setattr(crew_select, "select_crew", explode)

    review_tool_mod.main([str(make_subject(tmp_path))])

    assert calls == list(review_tool_mod.WORKERS) + [review_tool_mod.ARBITER]


if __name__ == "__main__":
    # Standalone entry point: run this file through pytest (repo convention,
    # see requirements-dev.txt) and mirror pytest's exit status.
    sys.exit(pytest.main([__file__, "-v"]))
