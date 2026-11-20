"""PR 5 retrieval verification: per-query lexical, vector, hybrid and graph results."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import dataset_ingest
from ingestion.datasets import graph, verify
from ingestion.datasets.config import Config
from ingestion.datasets.tests.helpers import (
    FakeServicesBackend,
    fake_services_server,
    service_env,
)

ENTRY_ID = "apd:Agriculture/Wheat-Prices.yml"
SOURCE_ID = f"{ENTRY_ID}/src/abc123/shard/0000"


def integration_row(
    entry_id: str = ENTRY_ID,
    *,
    title: str = "Wheat Prices",
    category: str = "Agriculture",
    status: str = "indexed",
    records: list[dict[str, object]] | None = None,
    metadata_status: str | None = None,
) -> dict[str, object]:
    records = records if records is not None else [
        {
            "source_id": f"{entry_id}/src/abc123/shard/0000",
            "kind": "content",
            "status": "indexed",
            "content_sha256": "d" * 64,
        }
    ]
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "title": title,
        "category": category,
        "metadata_sha256": "a" * 64,
        "catalog": {"repo": "awesomedata/apd-core", "commit": "c" * 40},
        "integration": {
            "status": status,
            "metadata": {
                "source_id": entry_id,
                "kind": "metadata",
                "status": metadata_status or ("indexed" if status == "indexed" else "failed"),
            },
            "content": {"records": records},
        },
    }


def config_for(base: str, tmp_path: Path) -> Config:
    return Config.load(data_root=tmp_path / "data", ledger_enabled=False).replace(
        graph_url=base,
        knowledge_url=base,
        chunking_url=base,
        embedding_url=base,
        lexical_url=base,
        vector_url=base,
        index_url=base,
        hybrid_url=base,
        verify_timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_build_queries_uses_indexed_entries_only():
    rows = [
        integration_row(),
        integration_row("apd:Test/b.yml", status="deferred", records=[]),
        integration_row(
            "apd:Test/c.yml",
            title="Barley",
            status="metadata_indexed",
            records=[],
            metadata_status="indexed",
        ),
    ]
    specs = verify.build_queries(rows, limit=10)
    # c.yml is metadata-only: its metadata record is indexed, so it is
    # verifiable even though it has no content shards.
    assert [spec.entry_id for spec in specs] == [ENTRY_ID, "apd:Test/c.yml"]
    assert specs[0].text == "Wheat Prices"
    assert SOURCE_ID in specs[0].expected_sources
    assert ENTRY_ID in specs[0].expected_sources  # the metadata record


def test_build_queries_strips_a_parenthetical_aside():
    rows = [integration_row(title="Indoor Light for Houseplants (meter readings)")]
    specs = verify.build_queries(rows, limit=5)
    assert specs[0].text == "Indoor Light for Houseplants"


def test_build_queries_respects_category_and_limit():
    rows = [
        integration_row("apd:A/a.yml", title="A", category="Agriculture"),
        integration_row("apd:B/b.yml", title="B", category="Biology"),
    ]
    assert [spec.text for spec in verify.build_queries(rows, limit=5, categories=["Biology"])] == ["B"]
    assert len(verify.build_queries(rows, limit=1)) == 1


def test_build_queries_skips_entries_without_an_indexed_record():
    rows = [
        integration_row(
            records=[{"source_id": SOURCE_ID, "kind": "content", "status": "failed"}],
            metadata_status="failed",
        )
    ]
    assert verify.build_queries(rows, limit=5) == []


def test_matches_accepts_the_entry_and_its_shards():
    spec = verify.QuerySpec(
        name="q1", text="x", entry_id=ENTRY_ID, expected_sources=(SOURCE_ID,)
    )
    assert spec.matches({"url": SOURCE_ID}) is True
    assert spec.matches({"url": ENTRY_ID}) is True
    assert spec.matches({"url": f"{ENTRY_ID}/src/other/shard/0000"}) is True
    assert spec.matches({"url": "apd:Other/x.yml"}) is False
    assert spec.matches({}) is False


# ---------------------------------------------------------------------------
# Legs against the fake services
# ---------------------------------------------------------------------------


def _spec(text: str = "Wheat Prices") -> verify.QuerySpec:
    return verify.QuerySpec(
        name="q1", text=text, entry_id=ENTRY_ID, expected_sources=(SOURCE_ID,)
    )


def test_lexical_leg_reports_a_match(tmp_path):
    with fake_services_server() as (base, backend):
        backend.chunks["c1"] = {
            "chunk_id": "c1",
            "url": SOURCE_ID,
            "source": "apd-core@c",
            "title": "Wheat Prices",
            "text": "wheat prices by year",
            "content_sha256": "d" * 64,
        }
        result = verify.lexical_leg(_spec(), config_for(base, tmp_path), top_k=5)
        assert result.status == "ok"
        assert result.matched is True
        assert result.hits[0]["url"] == SOURCE_ID


def test_lexical_leg_reports_no_match_for_unrelated_content(tmp_path):
    with fake_services_server() as (base, backend):
        backend.chunks["c1"] = {
            "chunk_id": "c1",
            "url": "apd:Other/x.yml",
            "text": "completely unrelated",
        }
        result = verify.lexical_leg(_spec(), config_for(base, tmp_path), top_k=5)
        assert result.status == "ok"
        assert result.matched is False


def test_lexical_leg_is_unavailable_when_the_service_fails(tmp_path):
    with fake_services_server() as (base, backend):
        backend.fail_paths = {"/knowledge/search": 503}
        result = verify.lexical_leg(_spec(), config_for(base, tmp_path), top_k=5)
        assert result.status == "unavailable"
        assert result.reason_code == "dependency_unavailable"
        assert result.matched is False


def test_vector_leg_embeds_the_query_then_queries_the_collection(tmp_path):
    with fake_services_server() as (base, backend):
        backend.vectors["v1"] = {
            "url": SOURCE_ID,
            "title": "Wheat Prices",
            "chunk_id": "c1",
            "content_sha256": "d" * 64,
            "text": "wheat prices",
        }
        result = verify.vector_leg(_spec(), config_for(base, tmp_path), top_k=5)
        assert result.status == "ok"
        assert result.matched is True
        assert backend.count("/embeddings") == 1
        assert backend.count("/collections/knowledge_chunks/query") == 1


def test_vector_leg_is_unavailable_when_embedding_fails(tmp_path):
    with fake_services_server() as (base, backend):
        backend.fail_paths = {"/embeddings": 503}
        result = verify.vector_leg(_spec(), config_for(base, tmp_path), top_k=5)
        assert result.status == "unavailable"
        assert result.reason_code == "dependency_unavailable"


def test_hybrid_leg_reports_degradation(tmp_path, monkeypatch):
    with fake_services_server() as (base, _backend):
        config = config_for(base, tmp_path)
        original = verify._call_json

        def fake_call(base_url, path, payload, **kwargs):
            if path == "/knowledge/search":
                return {
                    "ok": True,
                    "results": [{"url": SOURCE_ID, "text": "wheat prices"}],
                    "degraded": ["knowledge:vector"],
                }
            return original(base_url, path, payload, **kwargs)

        monkeypatch.setattr(verify, "_call_json", fake_call)
        result = verify.hybrid_leg(_spec(), config, top_k=5)
        assert result.status == "degraded"
        assert result.reason_code == "degraded_leg"
        assert result.matched is True


# ---------------------------------------------------------------------------
# Vector store health
# ---------------------------------------------------------------------------


def test_vector_store_is_functional_when_it_holds_vectors(tmp_path):
    with fake_services_server() as (base, backend):
        backend.vectors["v1"] = {"url": SOURCE_ID}
        health = verify.vector_store_health(config_for(base, tmp_path))
        assert health["functional"] is True
        assert health["count"] == 1
        assert health["reason_code"] is None


def test_empty_vector_store_is_not_functional(tmp_path):
    with fake_services_server() as (base, _backend):
        health = verify.vector_store_health(config_for(base, tmp_path))
        assert health["functional"] is False
        assert health["reason_code"] == "empty_collection"


def test_unreachable_vector_store_is_reported_not_raised(tmp_path):
    config = Config.load(data_root=tmp_path / "data", ledger_enabled=False).replace(
        vector_url="http://127.0.0.1:9", verify_timeout=2.0
    )
    health = verify.vector_store_health(config)
    assert health["functional"] is False
    assert health["reason_code"] in {"dependency_unavailable", "timeout"}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _query(lexical="ok", vector="ok", hybrid="ok", matched=True) -> dict[str, object]:
    return {
        "name": "q1",
        "query": "wheat",
        "entry_id": ENTRY_ID,
        "lexical": {"status": lexical, "matched": matched, "hits": 1, "top": []},
        "vector": {"status": vector, "matched": matched, "hits": 1, "top": []},
        "hybrid": {"status": hybrid, "matched": matched, "hits": 1, "top": []},
    }


def test_summary_passes_when_lexical_matches_and_vector_is_unavailable():
    summary = verify.summarize(
        [_query(vector="unavailable")],
        vector_health={"functional": False},
        graph_results=[{"status": "ok", "datasets": 3}],
    )
    assert summary["ok"] is True
    assert summary["legs"]["lexical"]["status"] == "verified"
    assert summary["legs"]["vector"]["status"] == "unavailable"
    assert summary["graph"]["verified"] is True


def test_summary_fails_when_the_lexical_leg_does_not_match():
    summary = verify.summarize(
        [_query(matched=False)],
        vector_health={"functional": True},
        graph_results=[],
    )
    assert summary["ok"] is False
    assert "lexical:failed" in summary["problems"]


def test_summary_fails_when_a_functional_vector_store_misses():
    summary = verify.summarize(
        [_query(vector="ok", matched=False)],
        vector_health={"functional": True},
        graph_results=[],
    )
    assert "vector:failed" in summary["problems"]


def test_summary_fails_when_graph_traversal_fails():
    summary = verify.summarize(
        [_query()],
        vector_health={"functional": True},
        graph_results=[{"status": "missing", "datasets": 0}],
    )
    assert "graph:traversal_failed" in summary["problems"]


# ---------------------------------------------------------------------------
# Graph traversal verification
# ---------------------------------------------------------------------------


def _seed_graph(backend: FakeServicesBackend, config: Config) -> None:
    """Project one entry into the fake graph with the real projection code."""
    from ingestion.datasets.tests.test_graph import COMMIT, row

    client = graph.graph_client(config)
    graph.project_entries(
        [row()],
        commit=COMMIT,
        config=config,
        client=client,
        selected={"apd:Agriculture/Wheat-Prices.yml"},
    )


def test_graph_traversal_returns_related_datasets(tmp_path):
    with fake_services_server() as (base, backend):
        config = config_for(base, tmp_path)
        _seed_graph(backend, config)
        client = graph.graph_client(config)
        results = verify.verify_graph(client, ["Agriculture"], max_depth=2)
        assert results[0]["status"] == "ok"
        assert results[0]["datasets"] == 1
        assert results[0]["sample"] == ["apd:Agriculture/Wheat-Prices.yml"]
        assert results[0]["related_formats"] >= 1
        assert results[0]["related_licenses"] >= 1


def test_graph_traversal_reports_a_missing_category(tmp_path):
    with fake_services_server() as (base, _backend):
        config = config_for(base, tmp_path)
        client = graph.graph_client(config)
        results = verify.verify_graph(client, ["Nonexistent"])
        assert results[0]["status"] == "missing"
        assert results[0]["reason_code"] == "category_node_missing"


def test_graph_queries_prefer_the_busiest_categories():
    rows = [
        integration_row("apd:A/a.yml", category="Agriculture"),
        integration_row("apd:A/b.yml", category="Agriculture"),
        integration_row("apd:B/b.yml", category="Biology"),
    ]
    assert verify.graph_queries(rows, limit=1) == ["Agriculture"]
    assert verify.graph_queries(rows, limit=5, categories=["Biology"]) == ["Biology"]


# ---------------------------------------------------------------------------
# Report and CLI
# ---------------------------------------------------------------------------


def test_markdown_report_lists_every_query_and_leg(tmp_path):
    with fake_services_server() as (base, backend):
        backend.chunks["c1"] = {"chunk_id": "c1", "url": SOURCE_ID, "text": "wheat prices"}
        config = config_for(base, tmp_path)
        specs = [_spec()]
        results = verify.run_queries(specs, config, top_k=5)
        payload = verify.verification_report(
            queries=specs,
            results=results,
            vector_health=verify.vector_store_health(config),
            graph_results=[],
            run_id="r",
            generated_at="2026-01-01T00:00:00Z",
            duration_ms=1,
            top_k=5,
            exit_code=0,
        )
        text = verify.render_markdown(payload)
        assert "# Dataset retrieval verification report" in text
        assert "Wheat Prices" in text
        assert "lexical: status=ok matched=True" in text


def test_cli_verify_end_to_end(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    (data_root / "manifest").mkdir(parents=True)
    manifest_path = data_root / "manifest" / ("c" * 40 + ".integration.jsonl")
    manifest_path.write_text(json.dumps(integration_row()) + "\n", encoding="utf-8")

    with fake_services_server() as (base, backend):
        backend.chunks["c1"] = {
            "chunk_id": "c1",
            "url": SOURCE_ID,
            "source": "apd-core@c",
            "title": "Wheat Prices",
            "text": "wheat prices by year and country",
            "content_sha256": "d" * 64,
        }
        backend.vectors["v1"] = {
            "url": SOURCE_ID,
            "title": "Wheat Prices",
            "chunk_id": "c1",
            "content_sha256": "d" * 64,
            "text": "wheat prices",
        }
        for key, value in service_env(base).items():
            monkeypatch.setenv(key, value)
        _seed_graph(backend, config_for(base, tmp_path))

        code = dataset_ingest.main(["verify", "--data-root", str(data_root), "--no-ledger"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "verify: queries=1" in out
        assert "lexical=verified" in out
        assert "vector=verified" in out
        assert f"graph { 'Agriculture' }" in out or "graph Agriculture" in out

        report = json.loads(next((data_root / "runs").glob("*.json")).read_text())
        assert report["report_type"] == "verification"
        assert report["summary"]["ok"] is True
        assert report["vector_store"]["functional"] is True
        assert report["queries"][0]["lexical"]["hits"] >= 1


def test_cli_verify_without_ingested_content_exits_one(tmp_path, capsys):
    data_root = tmp_path / "data"
    (data_root / "manifest").mkdir(parents=True)
    (data_root / "manifest" / "x.integration.jsonl").write_text(
        json.dumps(integration_row(status="deferred", records=[])) + "\n", encoding="utf-8"
    )
    code = dataset_ingest.main(
        ["verify", "--data-root", str(data_root), "--no-ledger", "--no-graph"]
    )
    assert code == 1
    assert "no indexed content to verify" in capsys.readouterr().err


def test_cli_verify_without_a_manifest_exits_one(tmp_path, capsys):
    code = dataset_ingest.main(
        ["verify", "--data-root", str(tmp_path / "empty"), "--no-ledger"]
    )
    assert code == 1
    assert "no integration manifest" in capsys.readouterr().err


@pytest.mark.parametrize("top_k", [0, -1])
def test_cli_verify_rejects_a_non_positive_top_k(tmp_path, capsys, top_k):
    code = dataset_ingest.main(
        ["verify", "--data-root", str(tmp_path), "--top-k", str(top_k), "--no-ledger"]
    )
    assert code == 1


def test_ad_hoc_queries_are_informational_not_passed_or_failed(tmp_path):
    with fake_services_server() as (base, _backend):
        config = config_for(base, tmp_path)
        spec = verify.QuerySpec(name="q1", text="anything", entry_id="", expected_sources=())
        results = verify.run_queries([spec], config, top_k=3)
        summary = verify.summarize(
            results, vector_health={"functional": False}, graph_results=[]
        )
        assert summary["legs"]["lexical"]["status"] == "informational"
        assert summary["legs"]["lexical"]["queries"] == 0
        assert summary["legs"]["lexical"]["informational"] == 1
        assert summary["ok"] is True


def test_cli_verify_can_check_only_graph_traversal(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    (data_root / "manifest").mkdir(parents=True)
    (data_root / "manifest" / "x.integration.jsonl").write_text(
        json.dumps(integration_row(status="deferred", records=[])) + "\n", encoding="utf-8"
    )
    with fake_services_server() as (base, backend):
        for key, value in service_env(base).items():
            monkeypatch.setenv(key, value)
        _seed_graph(backend, config_for(base, tmp_path))
        code = dataset_ingest.main(
            [
                "verify", "--data-root", str(data_root), "--no-ledger",
                "--category", "Agriculture",
            ]
        )
        captured = capsys.readouterr()
        assert code == 0, captured.out + captured.err
        assert "verifying graph traversal only" in captured.err
        report = json.loads(next((data_root / "runs").glob("*.json")).read_text())
        assert report["query_count"] == 0
        assert report["summary"]["legs"]["lexical"]["status"] == "not_requested"
        assert report["summary"]["graph"]["verified"] is True
        assert report["summary"]["graph"]["datasets"] == 1


def test_transient_dependency_failure_is_retried_once(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "VERIFY_RETRY_DELAY_SECONDS", 0.0)
    with fake_services_server() as (base, backend):
        backend.vectors["v1"] = {
            "url": SOURCE_ID,
            "title": "Wheat Prices",
            "chunk_id": "c1",
            "content_sha256": "d" * 64,
            "text": "wheat prices",
        }
        backend.fail_once = {"/embeddings": 1}
        result = verify.run_leg(
            _spec(), config_for(base, tmp_path), "vector", top_k=5
        )
        assert result.status == "ok"
        assert result.matched is True
        assert backend.count("/embeddings") == 2


def test_a_genuine_miss_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "VERIFY_RETRY_DELAY_SECONDS", 0.0)
    with fake_services_server() as (base, backend):
        backend.vectors["v1"] = {
            "url": "apd:Other/x.yml",
            "title": "Other",
            "chunk_id": "c9",
            "content_sha256": "e" * 64,
            "text": "unrelated",
        }
        result = verify.run_leg(_spec(), config_for(base, tmp_path), "vector", top_k=5)
        assert result.status == "ok"
        assert result.matched is False
        assert backend.count("/embeddings") == 1
