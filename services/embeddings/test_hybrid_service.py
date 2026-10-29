"""Tests for the hybrid search service: fusion, transcript parsing, prompt capture, ingest state, and HTTP.

Dependency failures use closed loopback ports, so they are real refused connections. The partial-success
test runs the real lexical, ranking, intent, and extraction services in-process on ephemeral ports.
"""
import json
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(_HERE)
for _name in ("ranking", "intent", "extraction"):
    sys.path.insert(0, os.path.join(_SERVICES, _name))

import hybrid_service as hs  # noqa: E402
import lexical_service as lx  # noqa: E402
import test_lexical_service as lexical_fixture  # noqa: E402
import extraction_service  # noqa: E402
import intent_service  # noqa: E402
import ranking_service  # noqa: E402


def closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def post(port, body=None, raw=None, path="/search"):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def record(key, **extra):
    return dict({"key": key, "text": f"text for {key}"}, **extra)


class FuseTests(unittest.TestCase):
    def test_item_ranked_high_in_both_lists_wins(self):
        fused = hs.fuse([[record("a"), record("b")], [record("b"), record("a")]])
        self.assertEqual([r["key"] for r in fused], ["a", "b"])

    def test_scores_follow_reciprocal_rank_fusion(self):
        fused = hs.fuse([[record("a")], [record("a")]])
        self.assertEqual(fused[0]["score"], round(2 / (hs.RRF_K + 1), 6))

    def test_single_list_keeps_its_order(self):
        keys = [r["key"] for r in hs.fuse([[record("x"), record("y"), record("z")]])]
        self.assertEqual(keys, ["x", "y", "z"])

    def test_first_occurrence_supplies_the_record_fields(self):
        fused = hs.fuse([[record("d", source="vector")], [record("d", source="lexical")]])
        self.assertEqual(fused[0]["source"], "vector")

    def test_empty_inputs(self):
        self.assertEqual(hs.fuse([]), [])
        self.assertEqual(hs.fuse([[], []]), [])


class TranscriptParsingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "sess-1.jsonl")

    def write(self, records, trailing=b""):
        with open(self.path, "wb") as handle:
            for item in records:
                handle.write((json.dumps(item) + "\n").encode("utf-8"))
            handle.write(trailing)

    def test_extracts_text_thinking_tool_use_and_tool_result(self):
        self.write([
            {"uuid": "u1", "sessionId": "sess-1", "timestamp": "2026-10-29T10:00:00Z",
             "message": {"role": "user", "content": [{"type": "text", "text": "Please fix the parser bug today"}]}},
            {"uuid": "a1", "timestamp": "2026-10-29T10:00:05Z",
             "message": {"role": "assistant", "content": [
                 {"type": "thinking", "thinking": "Need to look at the tokenizer first"},
                 {"type": "text", "text": "I will read the tokenizer file now."},
                 {"type": "tool_use", "name": "Read", "input": {"file_path": "/src/tokenizer.py"}}]}},
            {"uuid": "r1", "timestamp": "2026-10-29T10:00:06Z",
             "message": {"role": "user", "content": [{"type": "tool_result",
                                                      "content": [{"type": "text", "text": "file contents here"}]}]}},
        ])
        parsed = hs._parse_transcript(self.path)
        kinds = [(m["kind"], m["role"]) for m in parsed["messages"]]
        self.assertEqual(kinds, [("text", "user"), ("thinking", "assistant"), ("text", "assistant"),
                                 ("tool_use", "assistant"), ("tool_result", "user")])
        self.assertEqual(parsed["session_id"], "sess-1")
        self.assertEqual(parsed["touches"], [{"path": "/src/tokenizer.py", "tool": "Read",
                                              "ts": "2026-10-29T10:00:05Z"}])
        self.assertEqual((parsed["first_ts"], parsed["last_ts"]), ("2026-10-29T10:00:00Z", "2026-10-29T10:00:06Z"))

    def test_short_and_blank_text_is_skipped(self):
        self.write([
            {"uuid": "s1", "message": {"role": "user", "content": [{"type": "text", "text": "ok thanks"}]}},
            {"uuid": "s2", "message": {"role": "user", "content": [{"type": "text", "text": "   "}]}},
        ])
        self.assertEqual(hs._parse_transcript(self.path)["messages"], [])

    def test_invalid_json_lines_and_non_dict_records_are_ignored(self):
        with open(self.path, "wb") as handle:
            handle.write(b"not json at all\n")
            handle.write(b"[1, 2, 3]\n")
            handle.write(json.dumps({"uuid": "ok", "message": {"role": "user", "content": [
                {"type": "text", "text": "a perfectly valid long enough message"}]}}).encode() + b"\n")
        self.assertEqual(len(hs._parse_transcript(self.path)["messages"]), 1)

    def test_partial_last_line_is_left_for_the_next_run(self):
        complete = {"uuid": "c", "message": {"role": "user", "content": [
            {"type": "text", "text": "a complete message that is long enough"}]}}
        self.write([complete], trailing=b'{"uuid": "partial", "message": ')
        parsed = hs._parse_transcript(self.path)
        self.assertEqual(len(parsed["messages"]), 1)
        self.assertEqual(parsed["end_offset"], os.path.getsize(self.path) - len(b'{"uuid": "partial", "message": '))

    def test_resuming_from_offset_reads_only_new_lines(self):
        first = {"uuid": "1", "message": {"role": "user", "content": [
            {"type": "text", "text": "first message in the transcript"}]}}
        self.write([first])
        end = hs._parse_transcript(self.path)["end_offset"]
        with open(self.path, "ab") as handle:
            handle.write((json.dumps({"uuid": "2", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "second message appended later"}]}}) + "\n").encode())
        resumed = hs._parse_transcript(self.path, start=end)
        self.assertEqual([m["uuid"] for m in resumed["messages"]], ["2"])

    def test_tool_output_and_thinking_are_clipped(self):
        big = "z" * (hs.TOOL_TEXT_CHARS + 500)
        self.write([{"uuid": "t", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": big}]}}])
        self.assertEqual(len(hs._parse_transcript(self.path)["messages"][0]["text"]), hs.TOOL_TEXT_CHARS)


class TranscriptValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        self._saved = hs.TRANSCRIPT_ROOTS
        hs.TRANSCRIPT_ROOTS = [self.root]

    def tearDown(self):
        hs.TRANSCRIPT_ROOTS = self._saved

    def test_transcript_under_root_is_accepted(self):
        path = os.path.join(self.root, "ok.jsonl")
        open(path, "w").close()
        self.assertEqual(hs._validate_transcript(path), os.path.realpath(path))

    def test_path_outside_roots_is_refused(self):
        path = os.path.join(self.outside, "elsewhere.jsonl")
        open(path, "w").close()
        with self.assertRaises(ValueError):
            hs._validate_transcript(path)

    def test_non_jsonl_and_missing_files_are_refused(self):
        text = os.path.join(self.root, "notes.txt")
        open(text, "w").close()
        with self.assertRaises(ValueError):
            hs._validate_transcript(text)
        with self.assertRaises(ValueError):
            hs._validate_transcript(os.path.join(self.root, "missing.jsonl"))

    def test_symlink_escaping_the_root_is_refused(self):
        target = os.path.join(self.outside, "secret.jsonl")
        open(target, "w").close()
        link = os.path.join(self.root, "link.jsonl")
        os.symlink(target, link)
        with self.assertRaises(ValueError):
            hs._validate_transcript(link)

    def test_prefix_sibling_directory_is_not_treated_as_inside(self):
        sibling = self.root + "-evil"
        os.makedirs(sibling, exist_ok=True)
        path = os.path.join(sibling, "x.jsonl")
        open(path, "w").close()
        try:
            with self.assertRaises(ValueError):
                hs._validate_transcript(path)
        finally:
            os.remove(path)
            os.rmdir(sibling)


class PromptLedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = hs.PROMPT_LEDGER
        hs.PROMPT_LEDGER = os.path.join(self.dir, "prompts.jsonl")

    def tearDown(self):
        hs.PROMPT_LEDGER = self._saved

    def rows(self):
        with open(hs.PROMPT_LEDGER, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_stored_prompt_is_written_with_owner_only_permissions(self):
        result = hs.store_prompt("sess-a", "Ship the release notes", "2026-10-29T10:00:00Z", "/work/ce")
        self.assertEqual(result["status"], "stored")
        self.assertEqual(os.stat(hs.PROMPT_LEDGER).st_mode & 0o777, 0o600)
        self.assertEqual(self.rows()[0]["prompt"], "Ship the release notes")

    def test_retry_with_same_fields_is_a_duplicate(self):
        hs.store_prompt("sess-a", "same prompt", "2026-10-29T10:00:00Z", "/work")
        result = hs.store_prompt("sess-a", "same prompt", "2026-10-29T10:00:00Z", "/work")
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(len(self.rows()), 1)

    def test_different_timestamp_is_a_new_row(self):
        hs.store_prompt("sess-a", "same text", "2026-10-29T10:00:00Z", "/work")
        result = hs.store_prompt("sess-a", "same text", "2026-10-29T10:00:01Z", "/work")
        self.assertEqual(result["status"], "stored")
        self.assertEqual(len(self.rows()), 2)

    def test_prompt_is_not_claimed_as_indexed(self):
        result = hs.store_prompt("sess-a", "not indexed on capture", "2026-10-29T10:00:00Z", "")
        self.assertFalse(result["vector_indexed"])
        self.assertFalse(result["lexical_indexed"])

    def test_concurrent_identical_captures_write_one_row(self):
        # Pause after the ledger's existence check, so every thread reads "not yet stored" before any
        # append. Without the ledger lock, each thread would then append its own copy.
        real_exists = os.path.exists

        def slow_exists(path):
            found = real_exists(path)
            if path == hs.PROMPT_LEDGER:
                time.sleep(0.02)
            return found

        results = []

        def capture():
            results.append(hs.store_prompt("sess-race", "racing prompt text", "2026-10-29T11:00:00Z", ""))

        threads = [threading.Thread(target=capture) for _ in range(8)]
        with mock.patch.object(os.path, "exists", slow_exists):
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(sorted(r["status"] for r in results).count("stored"), 1)


class IngestStateTests(unittest.TestCase):
    """Only accepted or duplicate revisions may advance the stored offset."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.root = os.path.join(self.dir, "projects")
        os.makedirs(self.root)
        self.path = os.path.join(self.root, "sess.jsonl")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"uuid": "1", "message": {"role": "user", "content": [
                {"type": "text", "text": "a transcript line that is long enough"}]}}) + "\n")
        self._saved = (hs.INGEST_STATE, hs.INGEST_LOG, hs.ingest_session, hs.TRANSCRIPT_ROOTS)
        hs.INGEST_STATE = os.path.join(self.dir, "state.json")
        hs.INGEST_LOG = os.path.join(self.dir, "ingest.log")
        hs.TRANSCRIPT_ROOTS = [self.root]

    def tearDown(self):
        hs.INGEST_STATE, hs.INGEST_LOG, hs.ingest_session, hs.TRANSCRIPT_ROOTS = self._saved

    def state(self):
        with open(hs.INGEST_STATE, encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepted_ingest_records_the_new_offset(self):
        hs.ingest_session = lambda real, start=0, known=None: {
            "ok": True, "end_offset": os.path.getsize(real), "first_ts": None}
        hs._run_ingest(os.path.realpath(self.path))
        self.assertEqual(self.state()[os.path.realpath(self.path)]["offset"], os.path.getsize(self.path))

    def test_skipped_decision_does_not_advance_the_offset(self):
        hs.ingest_session = lambda real, start=0, known=None: {
            "ok": True, "skipped": "tombstoned", "end_offset": os.path.getsize(real), "first_ts": None}
        hs._run_ingest(os.path.realpath(self.path))
        self.assertFalse(os.path.exists(hs.INGEST_STATE))
        with open(hs.INGEST_LOG, encoding="utf-8") as handle:
            self.assertIn("tombstoned", handle.read())

    def test_unchanged_transcript_is_skipped_without_work(self):
        real = os.path.realpath(self.path)
        with open(hs.INGEST_STATE, "w", encoding="utf-8") as handle:
            json.dump({real: {"offset": os.path.getsize(real), "first_ts": None}}, handle)
        called = []
        hs.ingest_session = lambda *a, **k: called.append(a) or {}
        hs._run_ingest(real)
        self.assertEqual(called, [])
        with open(hs.INGEST_LOG, encoding="utf-8") as handle:
            self.assertIn("unchanged", handle.read())

    def test_failures_are_logged_not_raised(self):
        def boom(*a, **k):
            raise RuntimeError("dependency exploded")

        hs.ingest_session = boom
        hs._run_ingest(os.path.realpath(self.path))
        with open(hs.INGEST_LOG, encoding="utf-8") as handle:
            entry = json.loads(handle.readline())
        self.assertFalse(entry["ok"])
        self.assertIn("dependency exploded", entry["error"])


class HttpValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = hs.create_server("127.0.0.1", 0)
        cls.port = serve(cls.server)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = (hs.PROMPT_LEDGER, hs.TRANSCRIPT_ROOTS, hs.INGEST_LOG, hs.INGEST_STATE)
        hs.PROMPT_LEDGER = os.path.join(self.dir, "prompts.jsonl")
        hs.TRANSCRIPT_ROOTS = [self.dir]
        hs.INGEST_LOG = os.path.join(self.dir, "ingest.log")
        hs.INGEST_STATE = os.path.join(self.dir, "state.json")

    def tearDown(self):
        hs.PROMPT_LEDGER, hs.TRANSCRIPT_ROOTS, hs.INGEST_LOG, hs.INGEST_STATE = self._saved

    def test_prompts_endpoint_stores_valid_capture(self):
        status, body = post(self.port, {"session_id": "s", "prompt": "hello ledger", "timestamp": "2026-10-29T12:00:00Z",
                                        "cwd": "/w"}, path="/prompts")
        self.assertEqual((status, body["status"]), (200, "stored"))

    def test_prompts_endpoint_validation(self):
        base = {"session_id": "s", "prompt": "p", "timestamp": "t", "cwd": ""}
        cases = [
            (dict(base, session_id=""), 400),
            (dict(base, prompt="   "), 400),
            (dict(base, prompt="x" * (hs.MAX_PROMPT_CHARS + 1)), 413),
            (dict(base, timestamp=""), 400),
            (dict(base, cwd=5), 400),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(post(self.port, body, path="/prompts")[0], expected)

    def test_prompts_body_limit_is_enforced(self):
        raw = b"{" + b" " * hs.MAX_PROMPT_BODY_BYTES + b"}"
        try:
            self.assertEqual(post(self.port, raw=raw, path="/prompts")[0], 400)
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=10) as response:
            self.assertEqual(response.status, 200)

    def test_search_validation(self):
        cases = [
            ({"query": "", "top_k": 5}, 400),
            ({"query": "ok", "top_k": 0}, 400),
            ({"query": "ok", "top_k": hs.MAX_TOP_K + 1}, 400),
            ({"query": "ok", "top_k": 5, "rerank": "yes"}, 400),
            ({"query": "q" * (hs.MAX_QUERY_CHARS + 1), "top_k": 5}, 413),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(post(self.port, body, path="/search")[0], expected)

    def test_ingest_endpoint_rejects_paths_outside_roots(self):
        outside = tempfile.mkdtemp()
        path = os.path.join(outside, "x.jsonl")
        open(path, "w").close()
        status, body = post(self.port, {"transcript_path": path}, path="/sessions/ingest")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_ingest_endpoint_accepts_transcript_under_root(self):
        path = os.path.join(self.dir, "accepted.jsonl")
        open(path, "w").close()
        status, body = post(self.port, {"transcript_path": path}, path="/sessions/ingest")
        self.assertEqual((status, body["status"]), (202, "accepted"))

    def test_unknown_path_is_404(self):
        self.assertEqual(post(self.port, {}, path="/elsewhere")[0], 404)


class DependencyFailureTests(unittest.TestCase):
    """Refused connections to every dependency: the search must fail clearly, not return empty success."""

    def setUp(self):
        self._saved = {k: getattr(hs, k) for k in (
            "EMBEDDING_URL", "VECTOR_URL", "LEXICAL_URL", "RANKING_URL", "INTENT_URL", "GRAPH_URL",
            "RERANK_URL", "RERANK_ENABLED")}
        dead = f"http://127.0.0.1:{closed_port()}"
        hs.EMBEDDING_URL = hs.VECTOR_URL = hs.LEXICAL_URL = dead
        hs.RANKING_URL = hs.INTENT_URL = hs.GRAPH_URL = hs.RERANK_URL = dead
        hs.RERANK_ENABLED = False
        hs._SEARCH_CACHE.clear()

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(hs, key, value)
        hs._SEARCH_CACHE.clear()

    def test_all_candidate_sources_down_is_a_dependency_error(self):
        with self.assertRaises(hs.DependencyError) as ctx:
            hs._hybrid_search_uncached("anything at all", 5)
        self.assertIn("no candidate source succeeded", str(ctx.exception))

    def test_all_down_over_http_is_503(self):
        server = hs.create_server("127.0.0.1", 0)
        port = serve(server)
        try:
            status, body = post(port, {"query": "anything at all", "top_k": 5})
            self.assertEqual(status, 503)
            self.assertEqual(body["error"]["code"], "service_unavailable")
        finally:
            server.shutdown()
            server.server_close()

    def test_failed_search_is_never_cached(self):
        with self.assertRaises(hs.DependencyError):
            hs.hybrid_search("cache me if you can", 5)
        self.assertEqual(len(hs._SEARCH_CACHE), 0)


class PartialSearchTests(unittest.TestCase):
    """Real lexical, ranking, intent, and extraction services; embedding and vector are down."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.db = os.path.join(cls.dir, "transcripts.db")
        conn = sqlite3.connect(cls.db)
        conn.executescript(lexical_fixture.SCHEMA)
        conn.close()
        cls._saved_db = (lx.DB_PATH, lx.TOKENIZATION_URL)
        lx.DB_PATH = cls.db
        lx.TOKENIZATION_URL = lexical_fixture.TOKENIZATION_URL
        lx.ingest_document(lexical_fixture.session("partial-s", "/p/partial.jsonl"),
                           [lexical_fixture.message("pu1", "investigate the zanzibar cache eviction bug")])
        cls.servers = {
            "lexical": lx.create_server("127.0.0.1", 0),
            "ranking": ranking_service.create_server("127.0.0.1", 0),
            "intent": intent_service.create_server("127.0.0.1", 0),
            "extraction": extraction_service.create_server("127.0.0.1", 0),
        }
        cls.ports = {name: serve(server) for name, server in cls.servers.items()}

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers.values():
            server.shutdown()
            server.server_close()
        lx.DB_PATH, lx.TOKENIZATION_URL = cls._saved_db

    def setUp(self):
        self._saved = {k: getattr(hs, k) for k in (
            "EMBEDDING_URL", "VECTOR_URL", "LEXICAL_URL", "RANKING_URL", "INTENT_URL", "EXTRACTION_URL",
            "GRAPH_URL", "RERANK_ENABLED")}
        dead = f"http://127.0.0.1:{closed_port()}"
        hs.EMBEDDING_URL = hs.VECTOR_URL = hs.GRAPH_URL = dead
        hs.LEXICAL_URL = f"http://127.0.0.1:{self.ports['lexical']}"
        hs.RANKING_URL = f"http://127.0.0.1:{self.ports['ranking']}"
        hs.INTENT_URL = f"http://127.0.0.1:{self.ports['intent']}"
        hs.EXTRACTION_URL = f"http://127.0.0.1:{self.ports['extraction']}"
        hs.RERANK_ENABLED = False
        hs._SEARCH_CACHE.clear()

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(hs, key, value)

    def test_keyword_results_survive_when_vector_side_is_down(self):
        result = hs._hybrid_search_uncached("zanzibar eviction", 5)
        self.assertEqual(result["results"][0]["session_id"], "partial-s")
        self.assertEqual(result["results"][0]["source"], "lexical")
        # The vector leg needs the query embedding, so it is not attempted and is not listed separately.
        self.assertEqual(result["degraded"], ["embedding"])

    def test_ranked_output_comes_from_the_real_ranking_service(self):
        result = hs._hybrid_search_uncached("zanzibar eviction", 5)
        self.assertIn("contributions", result["results"][0])
        self.assertFalse(result["reranked"])

    def test_intent_is_classified_by_the_real_intent_service(self):
        result = hs._hybrid_search_uncached("why does the zanzibar cache fail", 5)
        self.assertIsNotNone(result["intent"])
        self.assertIn("label", result["intent"])

    def test_partial_result_is_not_cached(self):
        hs.hybrid_search("zanzibar eviction", 5)
        self.assertEqual(len(hs._SEARCH_CACHE), 0)


if __name__ == "__main__":
    unittest.main()
