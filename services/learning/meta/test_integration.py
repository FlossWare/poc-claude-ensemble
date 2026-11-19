"""Integration tests: MetaLearner feeds the tools that consume each domain."""

from __future__ import annotations

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


class FakeThompson:
    """Small loopback Thompson stand-in: ping, get_state, scripted draws."""

    STATE = {
        "last_updated": "2026-11-19T00:00:00Z",
        "models": {
            "arm-a": {"successes": 5, "failures": 1, "calls": 6, "total_cost": 1.0},
            "arm-b": {"successes": 4, "failures": 1, "calls": 5, "total_cost": 1.0},
            "arm-c": {"successes": 3, "failures": 1, "calls": 4, "total_cost": 1.0},
            "arm-d": {"successes": 2, "failures": 1, "calls": 3, "total_cost": 1.0},
            "arm-e": {"successes": 1, "failures": 1, "calls": 2, "total_cost": 1.0},
        },
    }

    def __init__(self, draws):
        self.draws = list(draws)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handler(fake):  # noqa: N805
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                action = body.get("action")
                if action == "ping":
                    payload = {"ok": True}
                elif action == "get_state":
                    payload = {"ok": True, "state": FakeThompson.STATE}
                elif action == "select_model":
                    payload = {"ok": True,
                               "model": fake.draws.pop(0) if fake.draws else "arm-a"}
                else:
                    payload = {"ok": False, "error": "unsupported"}
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                return

        return Handler


class StubLearner:
    def __init__(self, *, size=5, depth=2):
        self.size = size
        self.depth = depth
        self.records = []

    def decide(self, domain, context=None, **kwargs):
        choice = self.size if domain == "crew_size" else self.depth
        return {"arm": str(choice), "choice": choice, "source": "thompson",
                "posterior": {"context_scope": "global", "state_path": "stub",
                              "arms": {str(choice): {"posterior_mean": 0.6}}}}

    def record(self, *args, **kwargs):
        self.records.append((args, kwargs))
        return {"recorded": True}


# ---------------------------------------------------------------------------
# crew_select --learn-size
# ---------------------------------------------------------------------------

def test_learn_size_overrides_requested_count(tmp_path):
    import crew_select

    fake = FakeThompson(["arm-a", "arm-b", "arm-c", "arm-d", "arm-e"])
    learner = StubLearner(size=5)
    record = crew_select.select_crew(
        "code-review", count=3, learn_size=True, learner=learner,
        client=crew_select.ThompsonClient(service_url=fake.url),
        record_selection=False,
    )
    assert record["requested_workers"] == 5
    assert record["learned_size"]["choice"] == 5
    assert record["size_source"] == "meta_learner"
    assert len(record["workers"]) == 5


def test_without_learn_size_the_requested_count_wins(tmp_path):
    import crew_select

    fake = FakeThompson(["arm-a", "arm-b", "arm-c"])
    record = crew_select.select_crew(
        "code-review", count=3, client=crew_select.ThompsonClient(service_url=fake.url),
        record_selection=False,
    )
    assert record["requested_workers"] == 3
    assert record["learned_size"] is None
    assert record["size_source"] == "requested"


# ---------------------------------------------------------------------------
# multi_model_review --learn-depth / --depth / --prompt-variant
# ---------------------------------------------------------------------------

@pytest.fixture
def review_tool(tmp_path, monkeypatch):
    import multi_model_review

    monkeypatch.setattr(multi_model_review, "LEDGER", str(tmp_path / "reviews" / "reviews.jsonl"))
    monkeypatch.setattr(multi_model_review, "PRICING_PATH", str(tmp_path / "missing-pricing.py"))
    monkeypatch.setitem(sys.modules, "shared",
                        types.SimpleNamespace(thompson_shadow=types.SimpleNamespace(
                            record=lambda *a, **k: None)))
    import ledger_to_thompson
    monkeypatch.setattr(ledger_to_thompson, "sync", lambda *a, **k: 0)

    calls = []

    def fake_call(model, prompt):
        calls.append((model, prompt))
        text = "VERDICT: approve" if model == multi_model_review.ARBITER else "found a defect"
        return {"model": model, "text": text, "latency_ms": 5, "stop_reason": "end_turn",
                "input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(multi_model_review, "call", fake_call)
    subject = tmp_path / "subject.py"
    subject.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return multi_model_review, calls, subject


def read_ledger(module):
    path = Path(module.LEDGER)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_depth_one_skips_the_arbiter(review_tool):
    module, calls, subject = review_tool
    module.main([str(subject), "--depth", "1"])
    assert [model for model, _ in calls] == module.WORKERS
    rows = read_ledger(module)
    assert {row["kind"] for row in rows} == {"worker_result"}
    assert {row["review_depth"] for row in rows} == {1}
    assert {row["review_depth_source"] for row in rows} == {"explicit"}


def test_depth_three_adds_a_verification_pass(review_tool):
    module, calls, subject = review_tool
    module.main([str(subject), "--depth", "3"])
    assert [model for model, _ in calls] == module.WORKERS + [module.ARBITER, module.ARBITER]
    rows = read_ledger(module)
    assert [row["kind"] for row in rows].count("arbiter_verification") == 1
    assert calls[-1][1].startswith("You previously adjudicated reviews")


def test_learn_depth_uses_the_domain_draw(review_tool, monkeypatch):
    module, calls, subject = review_tool
    stub = StubLearner(depth=3)
    monkeypatch.setattr(module, "meta_learner", lambda: stub)
    module.main([str(subject), "--learn-depth"])
    assert len(calls) == len(module.WORKERS) + 2
    assert {row["review_depth_source"] for row in read_ledger(module)} == {"meta_learner"}


def test_prompt_variant_is_recorded_on_worker_rows(review_tool):
    module, calls, subject = review_tool
    module.main([str(subject), "--prompt-variant", "risk-first-v1"])
    rows = read_ledger(module)
    worker_rows = [row for row in rows if row["kind"] == "worker_result"]
    assert {row["prompt_variant"] for row in worker_rows} == {"risk-first-v1"}
    assert any("most dangerous first" in prompt for _, prompt in calls)


def test_thompson_with_learned_size_folds_review_evidence_back_in(review_tool, monkeypatch):
    module, calls, subject = review_tool
    import crew_select

    synced = []
    monkeypatch.setattr(module, "sync_meta_outcomes", lambda: synced.append(True))
    monkeypatch.setattr(crew_select, "select_crew", lambda **kwargs: {
        "status": "thompson", "reason": None, "selection_id": "sel-1",
        "ledger_path": "x", "learned_size": {"choice": 4, "arm": "4"},
        "models": list(module.WORKERS),
        "workers": [{"slot": i, "arm": f"arm-{i}", "model": model}
                    for i, model in enumerate(module.WORKERS)],
    })
    module.main([str(subject), "--thompson", "--learn-size"])
    assert synced == [True]


def test_without_learn_size_meta_state_is_untouched(review_tool, monkeypatch):
    module, calls, subject = review_tool
    import crew_select

    def explode():
        raise AssertionError("meta backfill must not run without --learn-size")

    monkeypatch.setattr(module, "sync_meta_outcomes", explode)
    monkeypatch.setattr(crew_select, "select_crew", lambda **kwargs: {
        "status": "thompson", "reason": None, "models": list(module.WORKERS),
        "workers": [{"slot": i, "arm": f"arm-{i}", "model": model}
                    for i, model in enumerate(module.WORKERS)],
    })
    module.main([str(subject), "--thompson"])


# ---------------------------------------------------------------------------
# bulk_ingest --learn-batch
# ---------------------------------------------------------------------------

def test_learn_batch_sets_the_run_budget_and_records_the_outcome(tmp_path, monkeypatch):
    import bulk_ingest

    item = tmp_path / "item.json"
    item.write_text(json.dumps({"title": "t", "text": "body"}), encoding="utf-8")

    stub = StubLearner(size=25)

    def decide(domain, context=None, **kwargs):
        assert domain == "batch_size"
        return {"arm": "25", "choice": 25, "source": "thompson",
                "posterior": {"context_scope": "global", "state_path": "stub",
                              "arms": {"25": {"posterior_mean": 0.5}}}}

    stub.decide = decide
    monkeypatch.setattr(bulk_ingest, "meta_learner", lambda: stub)

    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"source": "test", "origin": None, "format": "json", "gateway": "x",
                "ledger": str(tmp_path / "bulk.jsonl"), "input_path": str(item),
                "dry_run": False, "ingested": 3, "skipped_duplicate": 0,
                "skipped_malformed": 0, "failed": 0, "skipped_reasons": {}, "errors": []}

    monkeypatch.setattr(bulk_ingest, "run", fake_run)
    argv = ["bulk_ingest.py", str(item), "--source", "test",
            "--resume-state", str(tmp_path / "bulk.jsonl"), "--learn-batch"]
    monkeypatch.setattr(sys, "argv", argv)
    bulk_ingest.main()

    assert seen["limit"] == 25  # the draw became the run budget
    assert stub.records, "batch_size outcome was not recorded"
    args, kwargs = stub.records[0]
    assert (args[0], args[1]) == ("batch_size", 25)
    assert args[2] == 1.0
    assert kwargs["context"] == {"op": "bulk_ingest"}


def test_learn_batch_dry_run_is_not_evidence(tmp_path, monkeypatch):
    import bulk_ingest

    item = tmp_path / "item.json"
    item.write_text(json.dumps({"title": "t", "text": "body"}), encoding="utf-8")
    stub = StubLearner(size=25)
    stub.decide = lambda domain, context=None, **kwargs: {
        "arm": "25", "choice": 25, "source": "thompson",
        "posterior": {"context_scope": "global", "state_path": "stub",
                      "arms": {"25": {"posterior_mean": 0.5}}}}
    monkeypatch.setattr(bulk_ingest, "meta_learner", lambda: stub)
    monkeypatch.setattr(bulk_ingest, "run", lambda **kwargs: {
        "source": "test", "origin": None, "format": "json", "gateway": "x",
        "ledger": "x", "input_path": str(item), "dry_run": True, "ingested": 1,
        "skipped_duplicate": 0, "skipped_malformed": 0, "failed": 0,
        "skipped_reasons": {}, "errors": []})
    monkeypatch.setattr(sys, "argv", ["bulk_ingest.py", str(item), "--dry-run",
                                      "--source", "test", "--learn-batch"])
    bulk_ingest.main()
    assert stub.records == []
