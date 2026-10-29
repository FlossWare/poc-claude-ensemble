"""Tests for the canonical cost record contract."""

from cost_tracking.schema import CostRecord


def test_canonical_record_round_trip():
    record = CostRecord(
        timestamp="2026-09-29T12:00:00+00:00",
        model="sonnet",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.0105,
        task_name="test",
        provider="anthropic",
        metadata={"workflow_id": "ci"},
    )
    payload = record.to_dict()
    restored = CostRecord.from_dict(payload)

    assert restored == record
    assert payload["total_tokens"] == 1500
    assert payload["cost_usd"] == 0.0105


def test_legacy_aliases_are_read():
    record = CostRecord.from_dict(
        {
            "timestamp": "2026-09-29T12:00:00+00:00",
            "model": "sonnet",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_cost_usd": 0.0105,
            "task": "legacy",
        }
    )

    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    assert record.total_tokens == 1500
    assert record.cost_usd == 0.0105
    assert record.task_name == "legacy"
