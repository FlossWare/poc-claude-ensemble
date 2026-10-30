import json

import pytest

from learning.portable_artifacts import LearningArtifact, LearningArtifactStore


def test_teaching_signal_payload_round_trips_independently():
    artifact = LearningArtifact.create(
        "teaching-signal",
        {
            "phase": 1,
            "rationale": "Prefer the evidence-backed implementation.",
            "supporting_evidence": ["test-suite", "review"],
            "rejected_alternatives": ["unverified shortcut"],
            "next_phase_questions": ["Does the fix generalize?"],
        },
        provenance={"experiment": "arbiter-teaching-signal", "strategy_version": "0.1"},
        created_at="2026-10-04T00:00:00+00:00",
    )
    wire = artifact.to_json()
    restored = LearningArtifact.from_json(wire)
    assert json.loads(wire)["schema"] == "learning-artifact"
    assert restored == artifact
    assert restored.payload["phase"] == 1
    assert restored.provenance["experiment"] == "arbiter-teaching-signal"


def test_artifact_store_persists_and_reads_independently(tmp_path):
    store = LearningArtifactStore(tmp_path / "artifacts.jsonl")
    first = LearningArtifact.create("outcome", {"status": "success"})
    second = LearningArtifact.create("teaching-signal", {"phase": 2})
    store.record(first)
    store.record(second)
    raw = (tmp_path / "artifacts.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n") == 2
    restored = LearningArtifactStore(tmp_path / "artifacts.jsonl").read(limit=1)
    assert len(restored) == 1
    assert restored[0].artifact_type == "teaching-signal"
    assert restored[0].payload == {"phase": 2}


def test_artifact_rejects_unsupported_schema_or_version():
    base = LearningArtifact.create("outcome", {"status": "success"}).to_dict()
    with pytest.raises(ValueError, match="unsupported artifact schema"):
        LearningArtifact.from_dict(dict(base, schema="other"))
    with pytest.raises(ValueError, match="unsupported artifact version"):
        LearningArtifact.from_dict(dict(base, version="9.9"))
    with pytest.raises(ValueError, match="unsupported artifact schema"):
        LearningArtifact(
            "outcome", "test", {}, schema="other", version="0.1"
        )
    with pytest.raises(ValueError, match="unsupported artifact version"):
        LearningArtifact(
            "outcome", "test", {}, schema="learning-artifact", version="9.9"
        )


def test_artifact_rejects_non_json_values():
    with pytest.raises(TypeError, match="unsupported artifact value type"):
        LearningArtifact.create("outcome", {"value": object()})


def test_artifact_snapshots_nested_input():
    payload = {"nested": {"values": [1, 2]}}
    artifact = LearningArtifact.create("outcome", payload)
    payload["nested"]["values"].append(3)
    assert artifact.payload["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        artifact.payload["nested"]["values"][0] = 4


def test_artifact_store_limit_is_bounded(tmp_path):
    store = LearningArtifactStore(tmp_path / "artifacts.jsonl")
    for index in range(5):
        store.record(LearningArtifact.create("outcome", {"index": index}))
    restored = store.read(limit=2)
    assert [artifact.payload["index"] for artifact in restored] == [3, 4]
