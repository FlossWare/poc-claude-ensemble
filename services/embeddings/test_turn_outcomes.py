"""Turn-outcome tests: derivation from transcript lines, ledger-first recording, retries, and idempotency.

Thompson is a fake route on an ephemeral port, so no real Thompson state is changed.
"""
import http.server
import json
import os
import tempfile
import threading
import unittest

import turn_outcomes as to


def assistant(uuid, model, stop="end_turn", inp=10, out=5, cw=0, cr=0):
    return json.dumps({"uuid": uuid, "timestamp": "2026-10-29T10:00:00Z", "message": {
        "role": "assistant", "model": model, "stop_reason": stop,
        "content": [{"type": "text", "text": "answer"}],
        "usage": {"input_tokens": inp, "output_tokens": out,
                  "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}}})


def user_tool_result(is_error):
    return json.dumps({"uuid": "u", "message": {"role": "user", "content": [
        {"type": "tool_result", "is_error": is_error, "content": "x"}]}})


class FakeThompson:
    def __init__(self, accept=True):
        self.accept = accept
        self.received = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.received.append(body)
                reply = {"ok": outer.accept, "request_id": "t"}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                return

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class DerivationTests(unittest.TestCase):
    def test_end_turn_becomes_one_outcome_with_family_arm(self):
        outs = list(to.outcomes_from_lines([assistant("a1", "claude-opus-5")]))
        self.assertEqual([(o["uuid"], o["arm"], o["success"]) for o in outs], [("a1", "opus", True)])

    def test_gateway_models_are_their_own_arms(self):
        self.assertEqual(to.arm_for("rits/zai-org/glm-5-3"), "glm-5-3")
        self.assertEqual(to.arm_for("rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"), "deepseek-v4.1-flash")

    def test_family_names_for_claude_models(self):
        self.assertEqual(to.arm_for("claude-haiku-4-5-20251001"), "haiku")
        self.assertEqual(to.arm_for("claude-sonnet-4-5-20250929"), "sonnet")
        self.assertEqual(to.arm_for("claude-opus-5-5"), "opus")

    def test_synthetic_and_unknown_models_are_not_recorded(self):
        self.assertIsNone(to.arm_for("<synthetic>"))
        self.assertIsNone(to.arm_for("mystery-model"))

    def test_max_tokens_turns_are_skipped(self):
        self.assertEqual(list(to.outcomes_from_lines([assistant("m1", "claude-opus-5", stop="max_tokens")])), [])

    def test_tool_error_in_the_turn_makes_it_a_failure(self):
        lines = [user_tool_result(True), assistant("a2", "claude-haiku-5-5")]
        self.assertFalse(list(to.outcomes_from_lines(lines))[0]["success"])

    def test_tool_error_does_not_leak_into_the_next_turn(self):
        lines = [user_tool_result(True), assistant("a3", "claude-haiku-5-5"),
                 assistant("a4", "claude-haiku-5-5")]
        outs = list(to.outcomes_from_lines(lines))
        self.assertEqual([o["success"] for o in outs], [False, True])

    def test_malformed_lines_are_ignored(self):
        self.assertEqual(len(list(to.outcomes_from_lines(["{bad", "[1]", assistant("ok", "claude-haiku-5-5")]))), 1)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = (to.LEDGER, to.THOMPSON_URL)
        to.LEDGER = os.path.join(self.dir, "outcomes.jsonl")
        self.fake = FakeThompson(accept=True)
        to.THOMPSON_URL = self.fake.url

    def tearDown(self):
        self.fake.close()
        to.LEDGER, to.THOMPSON_URL = self._saved

    def outcomes(self, *lines):
        return list(to.outcomes_from_lines(lines))

    def test_accepted_outcome_is_sent_with_cost_and_tokens(self):
        accepted, pending = to.record_pending(self.outcomes(assistant("r1", "claude-opus-5", inp=1_000_000, out=0)))
        self.assertEqual((accepted, pending), (1, 0))
        sent = self.fake.received[0]
        self.assertEqual(sent["action"], "record_outcome")
        self.assertEqual(sent["model"], "opus")
        self.assertEqual(sent["task_type"], "claude_code_turn")
        self.assertAlmostEqual(sent["cost"], 15.0)
        self.assertEqual(sent["tokens"], 1_000_000)

    def test_cache_tokens_are_priced_and_counted(self):
        to.record_pending(self.outcomes(assistant("r2", "claude-opus-5", inp=0, out=0, cw=1_000_000, cr=1_000_000)))
        self.assertAlmostEqual(self.fake.received[0]["cost"], 15.0 * 1.25 + 15.0 * 0.1)
        self.assertEqual(self.fake.received[0]["tokens"], 2_000_000)

    def test_same_turn_is_never_recorded_twice(self):
        line = assistant("r3", "claude-haiku-5-5")
        to.record_pending(self.outcomes(line))
        accepted, _ = to.record_pending(self.outcomes(line))
        self.assertEqual(accepted, 0)
        self.assertEqual(len(self.fake.received), 1)

    def test_thompson_outage_leaves_pending_and_retry_completes_it(self):
        self.fake.close()
        to.THOMPSON_URL = "http://127.0.0.1:1"
        accepted, pending = to.record_pending(self.outcomes(assistant("r4", "claude-sonnet-5")))
        self.assertEqual((accepted, pending), (0, 1))
        self.fake = FakeThompson(accept=True)
        to.THOMPSON_URL = self.fake.url
        self.assertEqual(to.retry_pending(), (1, 0))
        self.assertEqual(to.retry_pending(), (0, 0))

    def test_thompson_rejection_is_not_marked_accepted(self):
        self.fake.accept = False
        accepted, pending = to.record_pending(self.outcomes(assistant("r5", "claude-haiku-5-5")))
        self.assertEqual((accepted, pending), (0, 1))
        rows = [json.loads(l) for l in open(to.LEDGER)]
        self.assertEqual([r["status"] for r in rows], ["pending"])




class IngestIntegrationTests(unittest.TestCase):
    """The ingest path reads only complete lines past the stored offset, then records outcomes."""

    def setUp(self):
        import hybrid_service as hs
        self.hs = hs
        self.dir = tempfile.mkdtemp()
        self._saved = (to.LEDGER, to.THOMPSON_URL)
        to.LEDGER = os.path.join(self.dir, "outcomes.jsonl")
        self.fake = FakeThompson(accept=True)
        to.THOMPSON_URL = self.fake.url
        self.path = os.path.join(self.dir, "session.jsonl")

    def tearDown(self):
        self.fake.close()
        to.LEDGER, to.THOMPSON_URL = self._saved

    def test_only_new_complete_lines_are_recorded(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(assistant("old", "claude-haiku-5-5") + "\n")
        offset = os.path.getsize(self.path)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(assistant("new", "claude-sonnet-5") + "\n")
            handle.write('{"uuid": "partial"')  # still being written; must wait for the next run
        result = self.hs.record_turn_outcomes(self.path, offset)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual([r["model"] for r in self.fake.received], ["sonnet"])


if __name__ == "__main__":
    unittest.main()
