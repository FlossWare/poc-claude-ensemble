"""HTTP contract tests for the read-only corpus export API.

The fixtures write through the Knowledge service's own schema helper
(``knowledge_service._connect``) so the tests fail if the export view and the
ingestion store ever drift apart. All HTTP assertions go through the real
gateway server on an ephemeral loopback port.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SERVICES = Path(__file__).resolve().parents[1]
# The gateway's runtime unit sets PYTHONPATH=services; its own directory is
# added here so this test file does not depend on another test module having
# inserted it earlier in the collection order.
for _path in (SERVICES, SERVICES / "server", SERVICES / "knowledge"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import knowledge_service  # noqa: E402
from corpus_export import CorpusExportStore, decode_cursor, encode_cursor  # noqa: E402


def create_store(path: Path) -> None:
    """Create a Knowledge-schema database at *path* without writing to the live one."""
    original = knowledge_service.DB_PATH
    knowledge_service.DB_PATH = str(path)
    try:
        knowledge_service._connect().close()
    finally:
        knowledge_service.DB_PATH = original


def insert_record(
    path: Path,
    source_id: str,
    *,
    origin: str = "default-origin",
    sequence: int = 1,
    title: str | None = None,
    content: str = "body",
    locator: str = "https://example.org/doc",
    tombstoned: int = 0,
) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO records (source_id, revision_sequence, revision_id, title, content,"
                " content_sha256, origin, locator, profiles, tombstoned) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    source_id,
                    sequence,
                    f"rev-{source_id}",
                    title or f"Title {source_id}",
                    content,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    origin,
                    locator,
                    json.dumps({"chunking": "window-v1@1"}),
                    tombstoned,
                ),
            )
    finally:
        conn.close()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    db_path = tmp_path / "knowledge.sqlite3"
    create_store(db_path)
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CORPUS_DB", str(db_path))
    return db_path


@pytest.fixture
def gateway_factory(corpus, monkeypatch):
    """Start real gateways; extra env must be set before calling the factory."""
    servers: list[tuple[object, threading.Thread]] = []

    def start():
        from server.ensemble_server import create_server

        server = create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield start

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(2)


def get(server, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def send(server, method: str, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}", method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def walk(server, path: str) -> list[dict]:
    records: list[dict] = []
    cursor = None
    for _ in range(20):
        page_path = path + (f"&cursor={cursor}" if cursor else "")
        status, body = get(server, page_path)
        assert status == 200, body
        records.extend(body["records"])
        cursor = body["next_cursor"]
        if cursor is None:
            return records
    raise AssertionError("pagination did not terminate")


def test_pagination_is_stable_while_the_store_changes(gateway_factory, corpus):
    for index in range(1, 6):
        insert_record(corpus, f"source-{index}", origin="rfc" if index % 2 else "book")
    server = gateway_factory()

    status, first = get(server, "/api/v1/corpus/records?limit=2")
    assert status == 200
    assert [record["source_id"] for record in first["records"]] == ["source-1", "source-2"]
    assert first["next_cursor"]
    assert first["total_hint"] == 5

    # A new record sorting *before* the cursor must not shift the next page,
    # which is exactly what an offset cursor would do.
    insert_record(corpus, "source-0", origin="rfc")
    status, second = get(server, f"/api/v1/corpus/records?limit=2&cursor={first['next_cursor']}")
    assert status == 200
    assert [record["source_id"] for record in second["records"]] == ["source-3", "source-4"]
    # total_hint is a live count, not a snapshot count.
    assert second["total_hint"] == 6

    full = walk(server, "/api/v1/corpus/records?limit=2")
    assert [record["source_id"] for record in full] == sorted(
        record["source_id"] for record in full
    )
    assert len({record["source_id"] for record in full}) == len(full) == 6


def test_filters_apply_before_pagination(gateway_factory, corpus):
    insert_record(corpus, "a-1", origin="rfc")
    insert_record(corpus, "a-2", origin="book")
    insert_record(corpus, "b-1", origin="rfc")
    insert_record(corpus, "b-2", origin="book")
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/records?origin=rfc")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["a-1", "b-1"]
    assert body["total_hint"] == 2

    status, body = get(server, "/api/v1/corpus/records?source_id=b-2")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["b-2"]
    assert body["records"][0]["origin"] == "book"

    status, body = get(server, "/api/v1/corpus/records?origin=rfc&source_id=a-1")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["a-1"]

    status, body = get(server, "/api/v1/corpus/records?origin=rfc&limit=1")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["a-1"]
    status, body = get(server, f"/api/v1/corpus/records?origin=rfc&limit=1&cursor={body['next_cursor']}")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["b-1"]
    assert body["next_cursor"] is None


def test_empty_results_are_a_successful_page(gateway_factory, corpus):
    insert_record(corpus, "only-one", origin="rfc")
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/records?origin=missing")
    assert status == 200
    assert body == {
        "ok": True,
        "records": [],
        "next_cursor": None,
        "total_hint": 0,
        "limit": 100,
        "returned": 0,
        "content_budget_exhausted": False,
    }


def test_limit_bounds_and_parameter_validation(gateway_factory, corpus):
    insert_record(corpus, "record-1")
    server = gateway_factory()
    bad_paths = (
        "/api/v1/corpus/records?limit=0",
        "/api/v1/corpus/records?limit=1001",
        "/api/v1/corpus/records?limit=-1",
        "/api/v1/corpus/records?limit=abc",
        "/api/v1/corpus/records?limit=",
        "/api/v1/corpus/records?limit=1&limit=2",
        "/api/v1/corpus/records?offset=1",
        "/api/v1/corpus/records?origin=",
        "/api/v1/corpus/records?origin=rfc&origin=book",
        "/api/v1/corpus/records?source_id=" + "x" * 513,
    )
    for path in bad_paths:
        status, body = get(server, path)
        assert status == 400, (path, body)
        assert body["ok"] is False
        assert body["error_code"] == "invalid_request"
        assert body["error"]


def test_default_limit_is_100_and_maximum_is_1000(gateway_factory, corpus):
    for index in range(101):
        insert_record(corpus, f"record-{index:04d}")
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/records")
    assert status == 200
    assert body["limit"] == 100
    assert body["returned"] == 100
    assert len(body["records"]) == 100
    assert body["next_cursor"] is not None
    assert body["total_hint"] == 101

    status, body = get(server, "/api/v1/corpus/records?limit=1000")
    assert status == 200
    assert body["returned"] == 101
    assert body["next_cursor"] is None


@pytest.mark.parametrize(
    "cursor",
    (
        "not-a-cursor",
        "!!!",
        "eyJ2IjoxfQ",  # base64 of {"v":1}: missing keys
        "eyJ2IjoyLCJzb3VyY2VfaWQiOiJhIiwicmV2aXNpb25fc2VxdWVuY2UiOjF9",  # wrong version
        "eyJ2IjoxLCJzb3VyY2VfaWQiOiJhIiwicmV2aXNpb25fc2VxdWVuY2UiOjB9",  # sequence 0
    ),
)
def test_malformed_cursors_are_rejected(gateway_factory, corpus, cursor):
    insert_record(corpus, "record-1")
    server = gateway_factory()
    status, body = get(server, f"/api/v1/corpus/records?cursor={cursor}")
    assert status == 400, body
    assert body["ok"] is False
    assert body["error_code"] == "invalid_request"
    assert "cursor" in body["error"]


def test_cursor_round_trips_arbitrary_source_ids():
    for source_id in ("plain", "with/slash", "with space", "ünïcode-✓", "x" * 512):
        assert decode_cursor(encode_cursor(source_id, 1_700_000_000_000_000_000)) == (
            source_id,
            1_700_000_000_000_000_000,
        )


def test_content_is_truncated_with_an_explicit_flag(gateway_factory, corpus, monkeypatch):
    content = "abcdefghijklmnop"
    insert_record(corpus, "long-record", content=content)
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CORPUS_MAX_CONTENT_CHARS", "10")
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/records")
    assert status == 200
    record = body["records"][0]
    assert record["content"] == content[:10]
    assert record["content_length"] == len(content)
    assert record["content_truncated"] is True
    # The digest always describes the full stored content, never the excerpt.
    assert record["content_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_page_content_budget_stops_a_page_and_resumes(gateway_factory, corpus, monkeypatch):
    for index in range(3):
        insert_record(corpus, f"record-{index}", content="123456")
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CORPUS_MAX_CONTENT_CHARS", "10")
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CORPUS_MAX_PAGE_CONTENT_CHARS", "10")
    server = gateway_factory()

    status, first = get(server, "/api/v1/corpus/records?limit=3")
    assert status == 200
    assert [record["source_id"] for record in first["records"]] == ["record-0"]
    assert first["content_budget_exhausted"] is True
    assert first["next_cursor"] is not None

    seen = [record["source_id"] for record in first["records"]]
    cursor = first["next_cursor"]
    for _ in range(3):
        status, page = get(server, f"/api/v1/corpus/records?limit=3&cursor={cursor}")
        assert status == 200
        seen.extend(record["source_id"] for record in page["records"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == ["record-0", "record-1", "record-2"]


def test_tombstoned_records_are_hidden(gateway_factory, corpus):
    insert_record(corpus, "live", origin="rfc")
    insert_record(corpus, "deleted", origin="private", tombstoned=1)
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/records")
    assert status == 200
    assert [record["source_id"] for record in body["records"]] == ["live"]
    assert body["total_hint"] == 1

    status, body = get(server, "/api/v1/corpus/origins")
    assert status == 200
    assert body == {"ok": True, "origins": [{"origin": "rfc", "count": 1}]}


def test_origins_lists_counts_sorted_by_origin(gateway_factory, corpus):
    insert_record(corpus, "b-1", origin="zebra")
    insert_record(corpus, "a-1", origin="alpha")
    insert_record(corpus, "b-2", origin="zebra")
    server = gateway_factory()

    status, body = get(server, "/api/v1/corpus/origins")
    assert status == 200
    assert body["origins"] == [
        {"origin": "alpha", "count": 1},
        {"origin": "zebra", "count": 2},
    ]

    status, body = get(server, "/api/v1/corpus/origins?limit=1")
    assert status == 400
    assert body["error_code"] == "invalid_request"


def test_corpus_routes_are_get_only_and_unknown_paths_are_404(gateway_factory, corpus):
    insert_record(corpus, "record-1")
    server = gateway_factory()
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        status, body = send(server, method, "/api/v1/corpus/records")
        assert status == 405, (method, body)
        assert body == {"ok": False, "error": "method not allowed"}
    for path in ("/api/v1/corpus", "/api/v1/corpus/records/extra", "/api/v1/corpus/unknown"):
        status, body = get(server, path)
        assert status == 404, (path, body)
        assert body["ok"] is False


def test_read_only_connection_rejects_writes(corpus):
    insert_record(corpus, "record-1")
    store = CorpusExportStore(str(corpus))
    with store.read_only_connection() as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM records")
    # The row is still there after the failed write.
    payload = CorpusExportStore(str(corpus)).records()
    assert payload["ok"] is True and len(payload["records"]) == 1


def test_missing_store_is_an_explicit_unavailable_error(gateway_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("POC_CLAUDE_ENSEMBLE_CORPUS_DB", str(tmp_path / "absent.sqlite3"))
    server = gateway_factory()
    status, body = get(server, "/api/v1/corpus/records")
    assert status == 503
    assert body["ok"] is False
    assert body["error_code"] == "corpus_unavailable"
