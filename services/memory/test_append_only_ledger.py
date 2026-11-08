import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from append_only_ledger import JsonlAppendOnlyLedger, digest


def event(event_id, payload=None, event_type="prompt", source="poc"):
    return {
        "event_id": event_id, "event_type": event_type, "source": source,
        "timestamp": "2026-11-07T12:00:00Z", "correlation_id": "corr-1",
        "payload": payload if payload is not None else {"id": event_id},
    }


def test_append_duplicate_conflict_pull_and_verify(tmp_path):
    ledger = JsonlAppendOnlyLedger(tmp_path / "events.jsonl")
    first = ledger.append(event("a"))
    second = ledger.append(event("b"))
    assert first["entry"]["sequence"] == 1
    assert second["entry"]["sequence"] == 2
    assert ledger.append(event("a"))["status"] == "duplicate"
    try:
        ledger.append(event("a", {"changed": True}))
        assert False, "expected event ID conflict"
    except ValueError as exc:
        assert "different content" in str(exc)
    page = ledger.pull(limit=1)
    assert [item["event_id"] for item in page["entries"]] == ["a"]
    assert page["next_after_sequence"] == 1 and page["has_more"]
    page = ledger.pull(after_sequence=1, limit=1)
    assert [item["event_id"] for item in page["entries"]] == ["b"]
    filtered = ledger.pull(event_type="absent", limit=1)
    assert filtered["entries"] == []
    assert filtered["next_after_sequence"] == 1 and filtered["has_more"]
    assert ledger.get("b")["sequence"] == 2
    assert ledger.verify()["ok"]


def test_filtered_cursor_is_bounded_and_advances(tmp_path):
    ledger = JsonlAppendOnlyLedger(tmp_path / "events.jsonl")
    for index in range(9):
        ledger.append(event(str(index), event_type="match" if index == 8 else "skip"))
    page = ledger.pull(limit=3, event_type="match")
    assert page["entries"] == []
    assert page["next_after_sequence"] == 3
    assert page["has_more"]
    page = ledger.pull(after_sequence=page["next_after_sequence"], limit=3, event_type="match")
    assert page["entries"] == []
    assert page["next_after_sequence"] == 6
    page = ledger.pull(after_sequence=page["next_after_sequence"], limit=3, event_type="match")
    assert [item["event_id"] for item in page["entries"]] == ["8"]
    assert page["next_after_sequence"] == 9


def test_legacy_prefix_remains_readable_and_assurance_is_explicit(tmp_path):
    path = tmp_path / "events.jsonl"
    legacy = {"event_id": "legacy", "kind": "session_end", "producer": "claude-code",
              "timestamp": "2026-11-06T12:00:00Z", "payload": {"session": "old"}}
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    ledger = JsonlAppendOnlyLedger(path)
    assert ledger.pull()["entries"][0] == legacy
    assert ledger.verify()["ok"]
    assert "legacy_prefix_unverified" in ledger.verify()["assurance"]
    added = ledger.append(event("new"))
    assert added["entry"]["sequence"] == 2
    assert ledger.verify()["ok"]


def test_partial_verification_requires_external_checkpoint(tmp_path):
    ledger = JsonlAppendOnlyLedger(tmp_path / "events.jsonl")
    one = ledger.append(event("one"))["entry"]
    ledger.append(event("two"))
    assert ledger.verify(from_sequence=2)["failure_category"] == "missing_checkpoint"
    checkpoint = {"ledger_id": ledger.ledger_id, "sequence": 1, "entry_sha256": one["entry_sha256"]}
    assert ledger.verify(from_sequence=2, checkpoint=checkpoint)["ok"]


def test_concurrent_append_sequences_are_contiguous(tmp_path):
    ledger = JsonlAppendOnlyLedger(tmp_path / "events.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: ledger.append(event(f"parallel-{i}")), range(24)))
    assert len(results) == 24
    entries = ledger.pull(limit=100)["entries"]
    assert [item["sequence"] for item in entries] == list(range(1, 25))


def test_import_is_idempotent_and_conflicts_are_preflighted(tmp_path):
    ledger = JsonlAppendOnlyLedger(tmp_path / "events.jsonl")
    source = JsonlAppendOnlyLedger(tmp_path / "source.jsonl")
    first = source.append(event("source-1"))["entry"]
    second = source.append(event("source-2"))["entry"]
    batch = [first, second]
    result = ledger.import_batch(source.ledger_id, batch)
    assert result["imported"] == ["source-1", "source-2"]
    assert ledger.import_batch(source.ledger_id, batch)["duplicates"] == ["source-1", "source-2"]
    before = ledger.status()["head_sequence"]
    conflict = dict(second, payload={"changed": True})
    conflict["payload_sha256"] = digest(conflict["payload"])
    try:
        ledger.import_batch(source.ledger_id, [event("would-be-prefix"), conflict])
        assert False, "expected conflict"
    except ValueError as exc:
        assert "conflict" in str(exc)
    assert ledger.status()["head_sequence"] == before


def test_tampered_payload_is_detected(tmp_path):
    path = tmp_path / "events.jsonl"
    ledger = JsonlAppendOnlyLedger(path)
    ledger.append(event("tamper"))
    record = json.loads(path.read_text().strip())
    record["payload"]["id"] = "changed"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = ledger.verify()
    assert not result["ok"]
    assert result["failure_category"] == "payload_hash_mismatch"

def test_shared_contract_fixture(tmp_path):
    fixture_path = Path(__file__).resolve().parents[2] / "parity" / "ledger-contract.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    ledger = JsonlAppendOnlyLedger(tmp_path / "contract.jsonl")
    for item in fixture["events"]:
        ledger.append(item)
    page = ledger.pull(limit=fixture["expectations"]["max_page_size"])
    assert [item["sequence"] for item in page["entries"]] == [1, 2]
    assert page["next_after_sequence"] == 2 and page["has_more"]
    filtered = ledger.pull(event_type=fixture["expectations"]["filter_event_type"], limit=2)
    assert [item["event_id"] for item in filtered["entries"]] == fixture["expectations"]["filtered_event_ids"]
    assert [item["event_id"] for item in ledger.replay(limit=10)["entries"]] == fixture["expectations"]["event_ids"]
    assert ledger.verify()["ok"]
