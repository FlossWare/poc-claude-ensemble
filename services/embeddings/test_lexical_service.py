"""Tests for the lexical FTS5 service: ranked search, incremental ingest, validation, persistence, and HTTP."""
import json
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import lexical_service as lx

_SERVICES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SERVICES, "tokenization"))
import tokenization_service as tk  # noqa: E402

# Same layout as the migrated transcript database: messages.analyzed feeds an external-content FTS5 table.
_FTS_LITERAL = lx.FTS_TOKENIZE.replace("'", "''")
SCHEMA = f"""
CREATE TABLE sessions(
    session_id TEXT, source TEXT, project TEXT, title TEXT,
    first_ts TEXT, last_ts TEXT, file TEXT PRIMARY KEY,
    is_subagent INTEGER, bytes INTEGER);
CREATE TABLE messages(
    id INTEGER PRIMARY KEY, session_id TEXT, uuid TEXT, role TEXT,
    kind TEXT, ts TEXT, text TEXT, file TEXT, analyzed TEXT);
CREATE INDEX messages_session ON messages(session_id);
CREATE INDEX messages_file ON messages(file);
CREATE VIRTUAL TABLE messages_fts USING fts5(
    analyzed, content='messages', content_rowid='id', tokenize='{_FTS_LITERAL}');
"""

# One real Tokenization service for the whole test run.
_TOKENIZER = tk.create_server("127.0.0.1", 0)
TOKENIZATION_URL = f"http://127.0.0.1:{_TOKENIZER.server_address[1]}"
threading.Thread(target=_TOKENIZER.serve_forever, daemon=True).start()


def session(session_id="s1", file="/work/proj/s1.jsonl", **extra):
    return dict({"session_id": session_id, "project": "proj", "file": file}, **extra)


def message(uuid, text, role="user", kind="text"):
    return {"uuid": uuid, "role": role, "kind": kind, "text": text, "ts": "2026-10-29T10:00:00Z"}


def post(port, path, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def refused_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


class LexicalTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "transcripts.db")
        conn = sqlite3.connect(self.db)
        conn.executescript(SCHEMA)
        conn.close()
        self._saved = (lx.DB_PATH, lx.TOKENIZATION_URL)
        lx.DB_PATH = self.db
        lx.TOKENIZATION_URL = TOKENIZATION_URL

    def tearDown(self):
        lx.DB_PATH, lx.TOKENIZATION_URL = self._saved


class SearchTests(LexicalTestCase):
    def test_finds_message_by_keyword(self):
        lx.ingest_document(session(), [message("u1", "The parser rejects trailing commas"),
                                       message("u2", "Unrelated note about deployment")])
        results = lx.search("parser", 10)
        self.assertEqual([r["text"] for r in results], ["The parser rejects trailing commas"])

    def test_result_carries_source_fields(self):
        lx.ingest_document(session("abc", project="ce", is_subagent=True),
                           [message("u1", "migration checklist item", role="assistant")])
        (result,) = lx.search("checklist", 5)
        self.assertEqual((result["session_id"], result["project"], result["role"], result["is_subagent"]),
                         ("abc", "ce", "assistant", True))
        self.assertEqual(result["kind"], "text")
        self.assertEqual(len(result["text_sha"]), 64)

    def test_query_terms_are_or_combined(self):
        lx.ingest_document(session(), [message("u1", "only alpha here please"),
                                       message("u2", "only beta here please")])
        texts = {r["text"] for r in lx.search("alpha beta", 10)}
        self.assertEqual(texts, {"only alpha here please", "only beta here please"})

    def test_porter_stemming_matches_inflections(self):
        lx.ingest_document(session(), [message("u1", "the service keeps running overnight")])
        self.assertEqual(len(lx.search("run", 5)), 1)

    def test_identical_text_across_sessions_is_returned_once(self):
        lx.ingest_document(session("s1", "/a/s1.jsonl"), [message("u1", "repeated pasted stack trace")])
        lx.ingest_document(session("s2", "/a/s2.jsonl"), [message("u9", "repeated pasted stack trace")])
        self.assertEqual(len(lx.search("stack", 10)), 1)

    def test_top_k_limits_results(self):
        lx.ingest_document(session(), [message(f"u{i}", f"keyword number {i}") for i in range(8)])
        self.assertEqual(len(lx.search("keyword", 3)), 3)

    def test_query_without_word_characters_returns_nothing(self):
        lx.ingest_document(session(), [message("u1", "some ordinary text here")])
        self.assertEqual(lx.search("!!! ... ???", 5), [])

    def test_quotes_in_query_do_not_break_match_syntax(self):
        lx.ingest_document(session(), [message("u1", 'the "quoted" phrase appears here')])
        self.assertEqual(len(lx.search('say "quoted" now', 5)), 1)

    def test_no_match_returns_empty_list(self):
        lx.ingest_document(session(), [message("u1", "something else entirely")])
        self.assertEqual(lx.search("zebra", 5), [])

    def test_results_survive_reopening_the_database(self):
        lx.ingest_document(session(), [message("u1", "durable keyword survives restart")])
        self.assertEqual(len(lx.search("durable", 5)), 1)
        lx.DB_PATH = self.db  # a new connection on every call; nothing is cached
        self.assertEqual(len(lx.search("restart", 5)), 1)


class IngestTests(LexicalTestCase):
    def test_counts_inserted_and_already_present(self):
        first = lx.ingest_document(session(), [message("u1", "first message text"),
                                               message("u2", "second message text")])
        self.assertEqual((first["inserted"], first["already_present"]), (2, 0))
        again = lx.ingest_document(session(), [message("u1", "first message text"),
                                               message("u3", "third message text")])
        self.assertEqual((again["inserted"], again["already_present"]), (1, 1))

    def test_same_text_with_different_uuid_is_a_new_row(self):
        lx.ingest_document(session(), [message("u1", "identical body text")])
        result = lx.ingest_document(session(), [message("u2", "identical body text")])
        self.assertEqual(result["inserted"], 1)

    def test_same_text_with_different_kind_is_a_new_row(self):
        lx.ingest_document(session(), [message("u1", "body text for both kinds")])
        result = lx.ingest_document(session(), [message("u1", "body text for both kinds", kind="tool_result")])
        self.assertEqual(result["inserted"], 1)

    def test_session_row_is_replaced_on_reingest(self):
        lx.ingest_document(session(title="old title"), [])
        lx.ingest_document(session(title="new title"), [])
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute("SELECT title FROM sessions WHERE file = ?", (session()["file"],)).fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [("new title",)])

    def test_default_kind_is_text(self):
        lx.ingest_document(session(), [{"uuid": "u1", "role": "user", "text": "no explicit kind here"}])
        self.assertEqual(lx.search("explicit", 5)[0]["kind"], "text")

    def test_tool_result_kind_is_stored_and_searchable(self):
        lx.ingest_document(session(), [message("t1", "command output with marker zxq", kind="tool_result")])
        self.assertEqual(lx.search("marker", 5)[0]["kind"], "tool_result")


class AnalysisTests(LexicalTestCase):
    """Index and query text go through the same Tokenization profile."""

    def test_compound_identifier_matches_whole_and_by_part(self):
        lx.ingest_document(session(), [message("u1", "the memory-service restarted after the crash")])
        self.assertEqual(len(lx.search("memory-service", 5)), 1)
        self.assertEqual(len(lx.search("memory", 5)), 1)

    def test_file_path_matches_as_one_identifier(self):
        lx.ingest_document(session(), [message("u1", "edited /src/app/tokenizer.py during the review")])
        self.assertEqual(len(lx.search("/src/app/tokenizer.py", 5)), 1)

    def test_uuid_matches_exactly(self):
        uuid = "3f2a9c1b-7d4e-4f10-8a2b-9c0d1e2f3a4b"
        lx.ingest_document(session(), [message("u1", f"trace id {uuid} was logged")])
        self.assertEqual(len(lx.search(uuid, 5)), 1)

    def test_analyzed_terms_are_stored_with_the_message(self):
        lx.ingest_document(session(), [message("u1", "the memory-service restarted")])
        conn = sqlite3.connect(self.db)
        try:
            (analyzed,) = conn.execute("SELECT analyzed FROM messages").fetchone()
        finally:
            conn.close()
        self.assertEqual(set(analyzed.split()), {"memory-service", "memory", "service", "restarted", "the"})

    def test_search_fails_closed_when_tokenization_is_down(self):
        lx.TOKENIZATION_URL = refused_url()
        with self.assertRaises(lx.AnalysisUnavailable):
            lx.search("anything", 5)

    def test_ingest_writes_nothing_when_tokenization_is_down(self):
        lx.TOKENIZATION_URL = refused_url()
        with self.assertRaises(lx.AnalysisUnavailable):
            lx.ingest_document(session(), [message("u1", "this must not be stored")])
        conn = sqlite3.connect(self.db)
        try:
            counts = (conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                      conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        finally:
            conn.close()
        self.assertEqual(counts, (0, 0))

    def test_match_expression_quotes_and_deduplicates(self):
        self.assertEqual(lx.match_expression(["alpha", "beta", "alpha"]), '"alpha" OR "beta"')
        self.assertEqual(lx.match_expression(['a"b']), '"a""b"')

    def test_analysis_failure_over_http_is_503_and_nothing_is_written(self):
        server = lx.create_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        lx.TOKENIZATION_URL = refused_url()
        try:
            status, body = post(port, "/documents", {
                "session": session("h", "/h/x.jsonl"),
                "messages": [message("hu", "http write while analysis is down")]})
            self.assertEqual(status, 503)
            self.assertEqual(body["error"]["code"], "service_unavailable")
            status, _ = post(port, "/search", {"query": "anything", "top_k": 5})
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()


class ValidationTests(unittest.TestCase):
    def assertRejected(self, body, fragment):
        with self.assertRaises(ValueError) as ctx:
            lx._validate_document(body)
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_document_passes(self):
        session_obj, messages = lx._validate_document({"session": session(), "messages": [message("u", "some text")]})
        self.assertEqual(session_obj["session_id"], "s1")
        self.assertEqual(len(messages), 1)

    def test_session_must_be_object(self):
        self.assertRejected({"session": "s1", "messages": []}, "session must be an object")

    def test_session_fields_are_required(self):
        for key in ("session_id", "project", "file"):
            broken = session()
            broken[key] = ""
            self.assertRejected({"session": broken, "messages": []}, f"session.{key}")

    def test_messages_must_be_list(self):
        self.assertRejected({"session": session(), "messages": {}}, "messages must be a list")

    def test_message_limit_is_enforced(self):
        many = [message(f"u{i}", "text") for i in range(lx.MAX_MESSAGES_PER_DOCUMENT + 1)]
        self.assertRejected({"session": session(), "messages": many}, "at most")

    def test_message_must_have_uuid_and_known_role(self):
        self.assertRejected({"session": session(), "messages": [{"role": "user", "text": "x"}]}, "uuid")
        self.assertRejected({"session": session(), "messages": [message("u", "x", role="system")]}, "uuid and a role")

    def test_unknown_kind_is_rejected(self):
        self.assertRejected({"session": session(), "messages": [message("u", "x", kind="image")]}, "kind must be")

    def test_empty_or_blank_text_is_rejected(self):
        self.assertRejected({"session": session(), "messages": [message("u", "   ")]}, "non-empty text")

    def test_text_length_limit_is_enforced(self):
        big = "x" * (lx.MAX_MESSAGE_CHARS + 1)
        self.assertRejected({"session": session(), "messages": [message("u", big)]}, "limited to")

    def test_non_dict_message_is_rejected(self):
        self.assertRejected({"session": session(), "messages": ["text"]}, "each message must be an object")


class HttpTests(LexicalTestCase):
    """Each test gets a fresh database from LexicalTestCase; the server reads DB_PATH per request."""

    @classmethod
    def setUpClass(cls):
        cls.server = lx.create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_reports_backend(self):
        status, body = get(self.port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual((body["service"], body["backend"]), ("lexical", "sqlite-fts5"))

    def test_unknown_path_is_404(self):
        self.assertEqual(get(self.port, "/nope")[0], 404)
        self.assertEqual(post(self.port, "/nope", {})[0], 404)

    def test_search_returns_ranked_results_over_http(self):
        lx.ingest_document(session("http-s", "/h/http.jsonl"), [message("hu1", "http boundary keyword sentinel")])
        status, body = post(self.port, "/search", {"query": "sentinel", "top_k": 5})
        self.assertEqual(status, 200)
        self.assertTrue(any(r["session_id"] == "http-s" for r in body["results"]))

    def test_search_rejects_bad_requests(self):
        cases = [
            ({"query": "", "top_k": 5}, 400),
            ({"query": 7, "top_k": 5}, 400),
            ({"query": "ok", "top_k": 0}, 400),
            ({"query": "ok", "top_k": lx.MAX_TOP_K + 1}, 400),
            ({"query": "ok", "top_k": "5"}, 400),
            ({"query": "x" * (lx.MAX_QUERY_CHARS + 1), "top_k": 5}, 413),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(post(self.port, "/search", body)[0], expected)

    def test_malformed_json_and_non_object_bodies_are_400(self):
        self.assertEqual(post(self.port, "/search", raw=b"{not json")[0], 400)
        self.assertEqual(post(self.port, "/search", raw=b"[1, 2]")[0], 400)
        self.assertEqual(post(self.port, "/search", raw=b"\xff\xfe")[0], 400)

    def test_empty_body_is_400(self):
        self.assertEqual(post(self.port, "/search", raw=b"")[0], 400)

    def test_oversized_body_is_refused_and_service_stays_up(self):
        oversized = b"{" + b" " * lx.MAX_BODY_BYTES + b"}"
        try:
            status = post(self.port, "/search", raw=oversized)[0]
            self.assertEqual(status, 400)
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # the server may close the socket without reading the body
        self.assertEqual(get(self.port, "/health")[0], 200)

    def test_documents_endpoint_ingests_and_reports_counts(self):
        status, body = post(self.port, "/documents", {
            "session": session("doc-http", "/d/doc.jsonl"),
            "messages": [message("du1", "posted over http document text")],
        })
        self.assertEqual(status, 200)
        self.assertEqual((body["inserted"], body["already_present"]), (1, 0))

    def test_documents_endpoint_rejects_invalid_document(self):
        status, body = post(self.port, "/documents", {"session": session(), "messages": [message("u", "")]})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_error_responses_do_not_expose_stack_traces(self):
        _, body = post(self.port, "/search", {"query": "", "top_k": 5})
        self.assertEqual(set(body), {"ok", "error"})
        self.assertEqual(set(body["error"]), {"code", "message"})


class BindTests(unittest.TestCase):
    def test_non_loopback_bind_is_refused(self):
        with self.assertRaises(ValueError):
            lx.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
