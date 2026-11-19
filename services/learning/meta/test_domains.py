"""Per-domain tests: arm definitions, derivation from ledgers, backfill."""

from __future__ import annotations

import json
import random

import pytest

from learning.meta import MetaLearner
from learning.meta.domains import batch_size, chunking, crew_size, graph_traversal
from learning.meta.domains import prompt_strategy, rag_scope, review_depth, tool_selection
from learning.meta.domains.base import DomainError


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(review_id, subject, kind, *, model="m", text="", **extra):
    data = {"review_id": review_id, "subject_sha256": subject, "kind": kind,
            "model": model, "text": text, "cost_usd": 0.1}
    data.update(extra)
    return data


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain,values", [
    (crew_size.CrewSize, [1, 5, "3"]),
    (review_depth.ReviewDepth, [1, 3, "2"]),
    (batch_size.BatchSize, [25, 500, "100"]),
    (graph_traversal.GraphTraversal, [1, 5, "3"]),
])
def test_numeric_domains_normalize_and_value(domain, values):
    for value in values:
        arm = domain.normalize(value)
        assert arm in domain.arms
        assert domain.value(arm) == int(arm)
    with pytest.raises(DomainError):
        domain.normalize(9 if domain is not review_depth.ReviewDepth else 9)
    assert domain.describe_arm(domain.default_arm)


def test_rag_scope_arms_and_mapping():
    assert len(rag_scope.RagScope.arms) == 12
    arm = rag_scope.RagScope.normalize({"memory.max_results": 10,
                                        "knowledge.max_queries": 2,
                                        "graph.enabled": True})
    assert arm == "m10-k2-g1"
    assert rag_scope.RagScope.value(arm) == {
        "memory.max_results": 10, "knowledge.max_queries": 2, "graph.enabled": True}
    nested = rag_scope.RagScope.normalize({"memory": {"max_results": 5},
                                           "knowledge": {"max_queries": 1},
                                           "graph": {"enabled": False}})
    assert nested == "m5-k1-g0"
    assert rag_scope.RagScope.normalize("m5-k3-g0") == rag_scope.RagScope.default_arm
    with pytest.raises(DomainError):
        rag_scope.RagScope.normalize({"memory.max_results": 7})


def test_label_domains_normalize():
    assert prompt_strategy.PromptStrategy.normalize("risk-first-v1") == "risk-first-v1"
    assert prompt_strategy.PromptStrategy.normalize("TERSE-V1") == "terse-v1"
    assert chunking.Chunking.normalize("window-v1@1") == "window-v1"
    assert tool_selection.ToolSelection.normalize("MultiEdit") == "edit"
    assert tool_selection.ToolSelection.normalize("SomeFutureTool") == "other"


# ---------------------------------------------------------------------------
# crew_size / review_depth from the review ledger
# ---------------------------------------------------------------------------

def test_crew_size_derivation_uses_actual_workers_and_overturns(tmp_path):
    ledger = write_jsonl(tmp_path / "reviews.jsonl", [
        row("r1", "subj-a", "worker_result", model="m1"),
        row("r1", "subj-a", "worker_result", model="m2"),
        row("r1", "subj-a", "worker_result", model="m3"),
        row("r1", "subj-a", "arbiter_decision", model="arb", text="VERDICT: approve"),
        row("r2", "subj-a", "worker_result", model="m1"),
        row("r2", "subj-a", "worker_result", model="m2"),
        row("r2", "subj-a", "worker_result", model="m3"),
        row("r2", "subj-a", "arbiter_decision", model="arb", text="VERDICT: reject"),
        row("r3", "subj-b", "worker_result", model="m1"),
        row("r3", "subj-b", "worker_result", model="m2"),
        row("r3", "subj-b", "arbiter_decision", model="arb", text="VERDICT: approve with fixes"),
    ])
    records = crew_size.derive_outcomes(ledger)
    by_review = {record["metadata"]["review_id"]: record for record in records}
    assert by_review["r1"]["arm"] == "3"
    assert by_review["r1"]["outcome"] == 0.0  # later review flipped approve -> reject
    assert by_review["r1"]["metadata"]["overturned"] is True
    assert by_review["r2"]["outcome"] == 1.0  # last review of its subject
    assert by_review["r3"]["arm"] == "2"

    learner = MetaLearner(tmp_path / "meta", rng=random.Random(1))
    first = crew_size.backfill(learner, ledger_path=ledger)
    second = crew_size.backfill(learner, ledger_path=ledger)
    assert (first["recorded"], first["duplicates"]) == (3, 0)
    assert (second["recorded"], second["duplicates"]) == (0, 3)
    assert learner.posterior("crew_size")["arms"]["3"]["calls"] == 2


def test_review_depth_derivation_marks_re_reviewed_depth_one(tmp_path):
    ledger = write_jsonl(tmp_path / "reviews.jsonl", [
        row("d1", "subj-a", "worker_result", model="m1"),
        row("d1", "subj-a", "arbiter_decision", model="arb", text="VERDICT: approve"),
        row("d2", "subj-a", "worker_result", model="m1"),
        row("d2", "subj-a", "arbiter_decision", model="arb", text="VERDICT: reject"),
        row("d3", "subj-b", "worker_result", model="m1"),
        row("d4", "subj-b", "worker_result", model="m1", review_depth=3),
        row("d4", "subj-b", "arbiter_decision", model="arb", text="VERDICT: approve",
            review_depth=3),
        row("d4", "subj-b", "arbiter_verification", model="arb", text="VERDICT: approve",
            review_depth=3),
    ])
    records = review_depth.derive_outcomes(ledger)
    by_review = {record["metadata"]["review_id"]: record for record in records}
    assert by_review["d1"]["arm"] == "2"
    assert by_review["d1"]["outcome"] == 0.0
    assert by_review["d3"]["arm"] == "1"
    assert by_review["d3"]["outcome"] == 0.0
    assert by_review["d3"]["metadata"]["reason"] == "depth-1 review was re-reviewed"
    assert "d2" not in by_review and "d4" not in by_review  # last of their subject


# ---------------------------------------------------------------------------
# batch_size from ops telemetry
# ---------------------------------------------------------------------------

def test_batch_size_derivation_from_telemetry_and_join(tmp_path):
    ops = tmp_path / "ops"
    ops.mkdir()
    write_jsonl(ops / "embedding_batch.jsonl", [
        {"ts": "2026-11-19T06:00:00Z", "op": "embedding_batch", "n_texts": 30,
         "status": 200, "latency_ms": 100.0},
        {"ts": "2026-11-19T06:00:01Z", "op": "embedding_batch", "n_texts": 500,
         "status": 503, "latency_ms": 50.0},
        {"ts": "2026-11-19T06:00:02Z", "op": "embedding_batch", "n_texts": 40,
         "status": 200, "latency_ms": 300.0},
    ])
    write_jsonl(ops / "gateway_forward.jsonl", [
        {"ts": "2026-11-19T06:10:00Z", "op": "gateway_forward",
         "path": "/ingestion/ingest", "status": 200, "latency_ms": 60.0},
    ])
    bulk = tmp_path / "bulk"
    write_jsonl(bulk / "pubmed.jsonl", [
        {"timestamp": "2026-11-19T06:10:00Z", "chunks": 100, "content_sha256": "abc",
         "status": "succeeded"},
    ])
    records = batch_size.derive_outcomes(ops, bulk)
    by_source = {}
    for record in records:
        by_source.setdefault(record["metadata"]["source"], []).append(record)
    embedding = by_source["embedding_batch"]
    assert [record["arm"] for record in embedding] == ["25", "500", "25"]
    assert embedding[0]["outcome"] > 0.0
    assert embedding[1]["outcome"] == 0.0  # 503
    assert embedding[2]["outcome"] < embedding[0]["outcome"]  # slower than median
    joined = by_source["gateway_forward+bulk-ingest"][0]
    assert joined["arm"] == "100"
    assert joined["outcome"] > 0.0


# ---------------------------------------------------------------------------
# rag_scope from the injection journal
# ---------------------------------------------------------------------------

def journal_row(event_id, memory_results, knowledge_results, **config):
    return {
        "schema_version": 1, "producer": "rag-hook", "kind": "rag_injection",
        "event_id": event_id, "timestamp": "2026-11-19T07:00:00Z",
        "config": {
            "sha256": "cfg",
            "memory": {"enabled": config.get("memory_enabled", True),
                       "max_results": config.get("memory_max_results", 5)},
            "knowledge": {"enabled": config.get("knowledge_enabled", True),
                          "max_queries": config.get("knowledge_max_queries", 3)},
            "graph": {"enabled": config.get("graph_enabled", False)},
        },
        "sections": {
            "memory": {"status": "ok" if memory_results else "empty", "results": memory_results},
            "knowledge": {"status": "ok" if knowledge_results else "empty",
                          "results": knowledge_results},
        },
        "context_chars": 1234,
        "truncated": False,
        "source_ids": ["knowledge:a#0"],
    }


def test_rag_scope_derivation_and_readiness(tmp_path):
    journal = write_jsonl(tmp_path / "injections.jsonl", [
        journal_row("e1", 2, 0, memory_max_results=10, knowledge_max_queries=2),
        journal_row("e2", 0, 3, memory_max_results=10, knowledge_max_queries=2),
        journal_row("e3", 0, 0, memory_enabled=False, knowledge_enabled=False),
    ])
    records = rag_scope.derive_outcomes(journal)
    assert len(records) == 2  # the all-sources-disabled row is not an outcome
    assert records[0]["arm"] == "m10-k2-g0"
    assert records[0]["outcome"] == 0.5  # one of two enabled sources yielded
    assert records[0]["cost"] == pytest.approx(1.234)
    assert records[1]["outcome"] == 0.5

    report = rag_scope.readiness(journal, minimum=2)
    assert report["ready"] is True
    assert report["arms_observed"] == ["m10-k2-g0"]
    assert rag_scope.readiness(tmp_path / "missing.jsonl")["observations"] == 0

    learner = MetaLearner(tmp_path / "meta")
    summary = rag_scope.backfill(learner, journal_path=journal)
    assert summary["recorded"] == 2
    assert learner.posterior("rag_scope")["arms"]["m10-k2-g0"]["calls"] == 2


# ---------------------------------------------------------------------------
# Instrumentation domains
# ---------------------------------------------------------------------------

def test_prompt_strategy_derivation(tmp_path):
    ledger = write_jsonl(tmp_path / "reviews.jsonl", [
        row("p1", "s", "worker_result", model="rits/zai-org/glm-5-3",
            text="defect", prompt_variant="risk-first-v1"),
        row("p1", "s", "worker_result", model="rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
            text="", prompt_variant="terse-v1"),
        row("p1", "s", "arbiter_decision", model="arb",
            text="SELECTED: GLM 5.3\nVERDICT: approve"),
    ])
    records = prompt_strategy.derive_outcomes(ledger)
    by_arm = {record["arm"]: record for record in records}
    assert by_arm["risk-first-v1"]["outcome"] == 1.0
    assert by_arm["risk-first-v1"]["metadata"]["arbiter_selected"] is True
    assert by_arm["terse-v1"]["outcome"] == 0.0


def test_chunking_derivation(tmp_path):
    bulk = tmp_path / "bulk"
    write_jsonl(bulk / "pubmed.jsonl", [
        {"status": "succeeded", "ingest_status": "completed", "chunks": 3,
         "chunk_profile": "markdown-v1@1", "source": "pubmed", "content_sha256": "a"},
        {"status": "failed", "stage": "ingest", "chunk_profile": "window-v1@1",
         "source": "pubmed", "content_sha256": "b"},
        {"status": "succeeded", "ingest_status": "rejected", "chunks": 0,
         "chunk_profile": "message-v1@1", "source": "pubmed", "content_sha256": "c"},
        {"status": "succeeded", "chunks": 1, "source": "pubmed", "content_sha256": "d"},
    ])
    records = chunking.derive_outcomes(bulk)
    by_arm = {record["arm"]: record for record in records}
    assert by_arm["markdown-v1"]["outcome"] == 1.0
    assert by_arm["markdown-v1"]["cost"] == 3.0
    assert by_arm["window-v1"]["outcome"] == 0.0
    assert by_arm["message-v1"]["outcome"] == 0.5
    assert len(records) == 3  # the row with no profile identity is skipped


def test_tool_selection_derivation(tmp_path):
    ledger = write_jsonl(tmp_path / "tools.jsonl", [
        {"tool": "Bash", "success": True, "task_type": "testing", "tool_use_id": "t1"},
        {"tool": "Read", "success": False, "task_type": "testing", "tool_use_id": "t2"},
    ])
    records = tool_selection.derive_outcomes(ledger)
    by_arm = {record["arm"]: record for record in records}
    assert by_arm["bash"]["outcome"] is True
    assert by_arm["read"]["outcome"] is False
    assert by_arm["bash"]["context"] == {"task_type": "testing"}


def test_graph_traversal_derivation(tmp_path):
    ops = tmp_path / "ops"
    write_jsonl(ops / "graph_traverse.jsonl", [
        {"depth": 3, "direction": "out", "status": "ok", "result_count": 4,
         "node_count": 4, "edge_count": 3, "latency_ms": 10.0, "ts": "t1", "start": "n1"},
        {"depth": 10, "direction": "in", "status": "ok", "result_count": 0,
         "latency_ms": 5.0, "ts": "t2", "start": "n2"},
        {"depth": 2, "direction": "out", "status": "error", "timeout": True,
         "result_count": 0, "latency_ms": 5000.0, "ts": "t3", "start": "n3"},
    ])
    records = graph_traversal.derive_outcomes(ops)
    assert [record["arm"] for record in records] == ["3", "5", "2"]
    assert [record["outcome"] for record in records] == [1.0, 0.25, 0.0]
