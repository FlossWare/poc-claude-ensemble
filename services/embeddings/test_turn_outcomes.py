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


def human(text):
    return json.dumps({"uuid": "h", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})


class SignalTests(unittest.TestCase):
    def signals(self, *lines):
        return [(o["uuid"], o["success"], o["signal"]) for o in to.outcomes_from_lines(lines)]

    def test_correction_is_recorded_but_does_not_count_against_the_model(self):
        self.assertEqual(self.signals(assistant("c1", "claude-opus-5"), human("That's wrong, the file is elsewhere")),
                         [("c1", True, "correction")])

    def test_neutral_next_message_is_a_success(self):
        self.assertEqual(self.signals(assistant("n1", "claude-opus-5"), human("thanks, now add a test")),
                         [("n1", True, "none")])

    def test_interrupted_turn_is_a_failure(self):
        self.assertEqual(self.signals(assistant("i1", "claude-opus-5"), human("[Request interrupted by user]")),
                         [("i1", False, "interrupted")])

    def test_turn_at_the_end_of_input_is_a_provisional_success(self):
        self.assertEqual(self.signals(assistant("p1", "claude-opus-5")), [("p1", True, "provisional")])

    def test_tool_error_is_a_failure_with_its_own_signal(self):
        self.assertEqual(self.signals(user_tool_result(True), assistant("t1", "claude-haiku-5-5"), human("ok")),
                         [("t1", False, "tool_error")])

    def test_a_later_correction_does_not_change_an_earlier_turn(self):
        lines = [assistant("e1", "claude-opus-5"), human("thanks"), assistant("e2", "claude-opus-5"), human("no, that's wrong")]
        self.assertEqual(self.signals(*lines), [("e1", True, "none"), ("e2", True, "correction")])


def response_record(uuid, msg_id, model="claude-opus-5", stop="end_turn", inp=10, out=5):
    return json.dumps({"uuid": uuid, "timestamp": "2026-10-29T10:00:00Z", "message": {
        "id": msg_id, "role": "assistant", "model": model, "stop_reason": stop,
        "content": [{"type": "text", "text": "answer"}],
        "usage": {"input_tokens": inp, "output_tokens": out}}})


class DuplicateRecordTests(unittest.TestCase):
    def test_one_response_written_as_several_records_is_one_outcome(self):
        lines = [response_record("r1", "msg_A"), response_record("r2", "msg_A"), response_record("r3", "msg_A")]
        self.assertEqual([o["key"] for o in to.outcomes_from_lines(lines)], ["msg_A"])

    def test_distinct_responses_are_distinct_outcomes(self):
        lines = [response_record("r1", "msg_A"), human("thanks"), response_record("r2", "msg_B")]
        self.assertEqual([o["key"] for o in to.outcomes_from_lines(lines)], ["msg_A", "msg_B"])

    def test_retry_across_batches_does_not_double_record(self):
        dir_ = tempfile.mkdtemp()
        saved = (to.LEDGER, to.THOMPSON_URL)
        to.LEDGER = os.path.join(dir_, "o.jsonl")
        fake = FakeThompson(accept=True)
        to.THOMPSON_URL = fake.url
        try:
            to.record_pending(to.outcomes_from_lines([response_record("r1", "msg_X")]))
            accepted, _ = to.record_pending(to.outcomes_from_lines([response_record("r2", "msg_X")]))
            self.assertEqual(accepted, 0)
            self.assertEqual(len(fake.received), 1)
        finally:
            fake.close()
            to.LEDGER, to.THOMPSON_URL = saved


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


class RebuildTests(unittest.TestCase):
    def test_rebuild_records_each_turn_once_across_files(self):
        import rebuild_outcomes as ro
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "proj"))
        with open(os.path.join(root, "proj", "a.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(response_record("r1", "msg_1") + "\n" + human("thanks") + "\n")
        with open(os.path.join(root, "proj", "b.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(response_record("r2", "msg_1") + "\n" + response_record("r3", "msg_2") + "\n")
        saved = (to.LEDGER, to.THOMPSON_URL)
        to.LEDGER = os.path.join(tempfile.mkdtemp(), "o.jsonl")
        fake = FakeThompson(accept=True)
        to.THOMPSON_URL = fake.url
        try:
            totals = ro.rebuild(root)
            self.assertEqual(totals["files"], 2)
            self.assertEqual(totals["accepted"], 2)  # msg_1 once, msg_2 once
            self.assertEqual(len(fake.received), 2)
        finally:
            fake.close()
            to.LEDGER, to.THOMPSON_URL = saved


if __name__ == "__main__":
    unittest.main()
