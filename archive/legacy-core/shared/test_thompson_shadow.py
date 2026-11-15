"""Shadow selection tests: proposals, recorded rows, and the guarantee that the live choice is never changed."""
import http.server
import json
import os
import socket
import tempfile
import threading
import unittest

from shared import thompson_shadow as ts


class FakeThompson:
    def __init__(self, model):
        outer = self
        self.model = model
        self.requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                data = json.dumps({"ok": True, "model": outer.model}).encode()
                self.send_response(200)
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


def closed_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self._saved = (ts.THOMPSON_URL, ts.LEDGER)
        self.ledger = os.path.join(tempfile.mkdtemp(), "selections.jsonl")
        ts.LEDGER = self.ledger
        self.fake = FakeThompson("opus")
        ts.THOMPSON_URL = self.fake.url

    def tearDown(self):
        self.fake.close()
        ts.THOMPSON_URL, ts.LEDGER = self._saved

    def rows(self):
        with open(self.ledger, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_propose_returns_thompson_model(self):
        self.assertEqual(ts.propose("arbitration_worker"), "opus")
        self.assertEqual(self.fake.requests[0]["action"], "select_model")

    def test_propose_returns_none_when_thompson_is_down(self):
        ts.THOMPSON_URL = closed_url()
        self.assertIsNone(ts.propose("arbitration_worker"))

    def test_record_writes_proposal_next_to_the_live_choice(self):
        row = ts.record("worker", ["sonnet", "haiku"], ["sonnet", "haiku", "opus"], source="arbitration", slot=0)
        self.assertEqual(row["thompson_proposal"], "opus")
        self.assertFalse(row["agrees_with_chosen"])
        self.assertTrue(row["proposal_in_eligible"])
        saved = self.rows()[0]
        self.assertEqual((saved["kind"], saved["chosen"], saved["source"]), ("selection_shadow", ["sonnet", "haiku"], "arbitration"))

    def test_record_marks_missing_proposal_as_null(self):
        ts.THOMPSON_URL = closed_url()
        row = ts.record("arbiter", ["opus"], ["opus"], source="arbitration")
        self.assertIsNone(row["thompson_proposal"])
        self.assertIsNone(row["agrees_with_chosen"])

    def test_record_never_raises_when_the_ledger_is_unwritable(self):
        ts.LEDGER = "/proc/ce-no-such-dir/selections.jsonl"
        self.assertIsNone(ts.record("worker", ["haiku"], ["haiku"], source="arbitration"))


if __name__ == "__main__":
    unittest.main()
