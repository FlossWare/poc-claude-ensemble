"""PR 5 graph projection: identity, provenance, idempotency and tombstones."""
from __future__ import annotations

import json

import pytest

from ingestion.datasets import graph, resume
from ingestion.datasets.config import Config
from ingestion.datasets.tests.helpers import FakeServicesBackend, fake_services_server

COMMIT = "c" * 40


def row(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "entry_id": "apd:Agriculture/Wheat-Prices.yml",
        "catalog": {
            "repo": "awesomedata/apd-core",
            "commit": COMMIT,
            "meta_url": "https://github.com/awesomedata/apd-core/tree/master/core/Agriculture/Wheat-Prices.yml",
        },
        "title": "Wheat Prices",
        "category": "Agriculture",
        "homepage": "https://example.org/wheat",
        "metadata_sha256": "a" * 64,
        "organization": [{"name": "World Bank", "web": None}],
        "publisher": [{"name": "A Person", "web": None}],
        "sources": [
            {"name": None, "access_url": "https://github.com/acme/wheat/blob/main/prices.csv"},
            {"name": None, "access_url": "https://zenodo.org/records/12345"},
        ],
        "license": {
            "catalog": "MIT",
            "metadata_repo": "GPL-3.0",
            "dataset": {"family": "cc-by", "policy": "index_content", "reason_code": None},
        },
        "classes": {"format": ["csv", "unknown"], "access": "open"},
        "disposition": "index_content",
        "status": "deferred",
        "reason_code": "resolver_not_implemented",
    }
    base.update(overrides)
    return base


def config_for(base_url: str, tmp_path) -> Config:
    return Config.load(data_root=tmp_path / "data", ledger_enabled=False).replace(
        graph_url=base_url, resume_base_delay_seconds=0.0, resume_max_delay_seconds=0.0
    )


# ---------------------------------------------------------------------------
# Pure projection
# ---------------------------------------------------------------------------


def test_projection_is_deterministic():
    first = graph.project_row(row(), commit=COMMIT)
    second = graph.project_row(row(), commit=COMMIT)
    assert [node["id"] for node in first.nodes] == [node["id"] for node in second.nodes]
    assert [edge["id"] for edge in first.edges] == [edge["id"] for edge in second.edges]
    assert first.ok and second.ok


def test_projection_covers_every_relationship_type():
    projection = graph.project_row(row(), commit=COMMIT)
    types = {str(node["type"]) for node in projection.nodes}
    assert types == {
        graph.NODE_DATASET,
        graph.NODE_VERSION,
        graph.NODE_CATEGORY,
        graph.NODE_ORGANIZATION,
        graph.NODE_REPOSITORY,
        graph.NODE_FORMAT,
        graph.NODE_LICENSE,
    }
    edge_types = {str(edge["type"]) for edge in projection.edges}
    assert edge_types == {
        graph.EDGE_HAS_VERSION,
        graph.EDGE_IN_CATEGORY,
        graph.EDGE_PUBLISHED_BY,
        graph.EDGE_HOSTED_AT,
        graph.EDGE_HAS_FORMAT,
        graph.EDGE_LICENSED_UNDER,
    }


def test_node_properties_hold_no_run_variant_state():
    """A commit or timestamp in a node's properties would break immutability."""
    projection = graph.project_row(row(), commit=COMMIT)
    blob = json.dumps([node["properties"] for node in projection.nodes])
    assert COMMIT not in blob
    assert "meta_url" not in blob
    assert "status" not in blob
    assert "reason_code" not in blob


def test_every_edge_carries_provenance():
    projection = graph.project_row(row(), commit=COMMIT)
    for edge in projection.edges:
        properties = edge["properties"]
        assert isinstance(properties, dict)
        assert properties["origin"] == f"awesomedata/apd-core@{COMMIT}"
        assert properties["catalog_commit"] == COMMIT
        assert properties["metadata_sha256"] == "a" * 64
        assert properties["entry_id"] == "apd:Agriculture/Wheat-Prices.yml"
        assert "locator" in properties


def test_new_metadata_hash_creates_a_new_version_node_and_edge():
    first = graph.project_row(row(), commit=COMMIT)
    changed = graph.project_row(row(metadata_sha256="b" * 64), commit=COMMIT)
    first_versions = {
        node["id"] for node in first.nodes if node["type"] == graph.NODE_VERSION
    }
    changed_versions = {
        node["id"] for node in changed.nodes if node["type"] == graph.NODE_VERSION
    }
    assert first_versions != changed_versions
    assert not (first_versions & changed_versions)
    first_edges = {edge["id"] for edge in first.edges}
    assert not (first_edges & {edge["id"] for edge in changed.edges})


def test_unknown_license_and_format_produce_no_relationship():
    projection = graph.project_row(
        row(
            license={"dataset": {"family": "unknown", "policy": "metadata_only_review"}},
            classes={"format": ["unknown"]},
        ),
        commit=COMMIT,
    )
    types = {str(node["type"]) for node in projection.nodes}
    assert graph.NODE_LICENSE not in types
    assert graph.NODE_FORMAT not in types


def test_row_from_another_commit_is_rejected():
    projection = graph.project_row(row(), commit="d" * 40)
    assert projection.ok is False
    assert projection.reason_code == "catalog_mismatch"


def test_missing_metadata_hash_is_rejected():
    projection = graph.project_row(row(metadata_sha256=None), commit=COMMIT)
    assert projection.reason_code == "no_metadata"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/acme/wheat/blob/main/prices.csv", ("github.com", "acme", "wheat")),
        ("https://raw.githubusercontent.com/acme/wheat/main/x.csv", ("raw.githubusercontent.com", "acme", "wheat")),
        ("https://huggingface.co/datasets/acme/wheat", ("huggingface.co", "acme", "wheat")),
        ("https://zenodo.org/records/12345", ("zenodo.org", "records", "12345")),
        ("https://data.example.org/files/x.csv", ("data.example.org", "", "")),
        ("ftp://files.example.org/x.csv", ("files.example.org", "", "")),
    ],
)
def test_repository_for_extracts_host_owner_and_repo(url, expected):
    canonical = graph.repository_for(url)
    assert canonical is not None
    assert (canonical["host"], canonical["owner"], canonical["repo"]) == expected


def test_repository_for_rejects_non_urls():
    assert graph.repository_for(None) is None
    assert graph.repository_for("not a url") is None


def test_named_ids_do_not_collide_after_slugging():
    assert graph.named_id("organization", "Café") != graph.named_id("organization", "Cafe")
    assert graph.named_id("organization", "World Bank") == graph.named_id("organization", "World Bank")
    assert graph.named_id("organization", "  ").startswith("organization:unnamed-")


def test_homepage_is_only_used_when_no_source_exists():
    with_source = graph.project_row(row(), commit=COMMIT)
    without_source = graph.project_row(row(sources=[]), commit=COMMIT)
    kinds = {
        edge["properties"]["kind"]
        for edge in with_source.edges
        if edge["type"] == graph.EDGE_HOSTED_AT
    }
    assert kinds == {"source"}
    homepage_kinds = {
        edge["properties"]["kind"]
        for edge in without_source.edges
        if edge["type"] == graph.EDGE_HOSTED_AT
    }
    assert homepage_kinds == {"homepage"}


# ---------------------------------------------------------------------------
# Against a real HTTP Graph service (fake backend, real transport)
# ---------------------------------------------------------------------------


def _project(rows, base_url, tmp_path, *, selected=None, **kwargs):
    config = config_for(base_url, tmp_path)
    client = graph.graph_client(config)
    return graph.project_entries(
        rows,
        commit=COMMIT,
        config=config,
        client=client,
        selected=selected if selected is not None else {str(item["entry_id"]) for item in rows},
        **kwargs,
    )


def test_projection_writes_nodes_then_edges(tmp_path):
    with fake_services_server() as (base, backend):
        results, context = _project([row()], base, tmp_path)
        assert results[0].status == "indexed"
        assert results[0].nodes_created == 8  # one Dataset, its version, five shared
        assert results[0].edges_created == 7  # two hosted_at edges
        assert context.snapshot.mode == "prefetch"
        assert len(backend.nodes) == 8
        assert len(backend.edges) == 7
        assert backend.count("/graph/add-node") == 8
        assert backend.count("/graph/add-edge") == 7


def test_second_projection_writes_nothing(tmp_path):
    rows = [row(), row(entry_id="apd:Agriculture/Barley.yml", metadata_sha256="b" * 64)]
    with fake_services_server() as (base, backend):
        first, _ = _project(rows, base, tmp_path)
        assert {item.status for item in first} == {"indexed"}
        backend.reset()
        second, _ = _project(rows, base, tmp_path)
        assert {item.status for item in second} == {"unchanged"}
        assert {item.reason_code for item in second} == {"graph_unchanged"}
        assert backend.count("/graph/add-node") == 0
        assert backend.count("/graph/add-edge") == 0
        # Shared Category/Format/License nodes were reused, not duplicated.
        assert len(backend.nodes) == len({node["id"] for node in backend.nodes.values()})
        assert all(item.nodes_created == 0 for item in second)
        assert all(item.nodes_present > 0 for item in second)


def test_interrupted_projection_never_duplicates_on_retry(tmp_path):
    rows = [
        row(entry_id=f"apd:Agriculture/Crop-{index}.yml", metadata_sha256=f"{index:064d}")
        for index in range(5)
    ]
    with fake_services_server() as (base, backend):
        # Simulate an interruption after two entries: only they were written.
        config = config_for(base, tmp_path)
        client = graph.graph_client(config)
        partial = graph.project_entries(
            rows[:2],
            commit=COMMIT,
            config=config,
            client=client,
            selected={str(item["entry_id"]) for item in rows},
        )[0]
        assert [item.status for item in partial[:2]] == ["indexed", "indexed"]
        nodes_before = dict(backend.nodes)
        edges_before = dict(backend.edges)

        # Restart with no checkpoints at all: the deterministic ids are the
        # only protection, and they must be enough.
        results, _ = _project(rows, base, tmp_path)
        assert len(backend.nodes) == len(set(backend.nodes))
        assert len(backend.edges) == len(set(backend.edges))
        for node_id, node in nodes_before.items():
            assert backend.nodes[node_id] == node
        for edge_id, edge in edges_before.items():
            assert backend.edges[edge_id] == edge
        by_entry = {item.entry_id: item for item in results}
        assert by_entry[rows[0]["entry_id"]].status == "unchanged"
        assert by_entry[rows[4]["entry_id"]].status == "indexed"


def test_checkpoint_resume_skips_completed_entries(tmp_path):
    rows = [row(), row(entry_id="apd:Agriculture/Barley.yml", metadata_sha256="b" * 64)]
    with fake_services_server() as (base, backend):
        config = config_for(base, tmp_path)
        client = graph.graph_client(config)
        journal = graph.checkpoint_log(config)
        first, _ = graph.project_entries(
            rows, commit=COMMIT, config=config, client=client,
            selected={str(item["entry_id"]) for item in rows}, journal=journal,
        )
        assert [item.status for item in first] == ["indexed", "indexed"]
        backend.reset()

        resumed_log = graph.checkpoint_log(config)
        second, _ = graph.project_entries(
            rows, commit=COMMIT, config=config, client=client,
            selected={str(item["entry_id"]) for item in rows}, journal=resumed_log,
        )
        assert [item.status for item in second] == ["unchanged", "unchanged"]
        assert all(item.resumed for item in second)
        assert all(item.reason_code == "checkpoint_resume" for item in second)
        # Only the eight prefetch reads happened; no writes at all.
        assert backend.count("/graph/add-node") == 0
        assert backend.count("/graph/add-edge") == 0


def test_changed_metadata_invalidates_the_checkpoint(tmp_path):
    with fake_services_server() as (base, backend):
        config = config_for(base, tmp_path)
        client = graph.graph_client(config)
        journal = graph.checkpoint_log(config)
        graph.project_entries(
            [row()], commit=COMMIT, config=config, client=client,
            selected={"apd:Agriculture/Wheat-Prices.yml"}, journal=journal,
        )
        changed = row(metadata_sha256="b" * 64, title="Wheat Prices 2026")
        results, _ = graph.project_entries(
            [changed], commit=COMMIT, config=config, client=client,
            selected={"apd:Agriculture/Wheat-Prices.yml"}, journal=graph.checkpoint_log(config),
        )
        assert results[0].status == "indexed"
        assert results[0].resumed is False
        assert results[0].nodes_created == 1  # only the new DatasetVersion node


def test_retry_is_bounded_and_reports_the_reason(tmp_path):
    with fake_services_server() as (base, backend):
        backend.fail_paths = {"/graph/add-node": 500}
        policy = resume.RetryPolicy(
            max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0
        )
        results, context = _project([row()], base, tmp_path, policy=policy)
        assert results[0].status == "deferred"
        assert results[0].reason_code == "graph_failed"
        assert results[0].retryable is True
        assert results[0].attempts == 2
        assert context.graph_errors == 1


def test_conflicting_existing_node_is_a_failure_not_a_crash(tmp_path):
    with fake_services_server() as (base, backend):
        projection = graph.project_row(row(), commit=COMMIT)
        dataset = next(node for node in projection.nodes if node["type"] == graph.NODE_DATASET)
        backend.nodes[str(dataset["id"])] = {
            "id": dataset["id"],
            "type": dataset["type"],
            "properties": {"entry_id": "different"},
        }
        results, _ = _project([row()], base, tmp_path)
        assert results[0].status == "failed"
        assert results[0].reason_code == "graph_conflict"


def test_tombstoned_entries_are_not_projected(tmp_path):
    with fake_services_server() as (base, backend):
        results, _ = _project(
            [row()], base, tmp_path, tombstoned={"apd:Agriculture/Wheat-Prices.yml"}
        )
        assert results[0].status == "skipped"
        assert results[0].reason_code == "source_tombstoned"
        assert backend.count("/graph/add-node") == 0


def test_tombstones_are_read_from_the_previous_integration_manifest():
    artifacts = {
        "apd:Test/a.yml": {"metadata": {"status": "skipped", "reason_code": "source_tombstoned"}},
        "apd:Test/b.yml": {
            "content": {"records": [{"status": "skipped", "reason_code": "source_tombstoned"}]}
        },
        "apd:Test/c.yml": {"status": "indexed", "content": {"records": []}},
    }
    assert graph.tombstones_from_manifest(artifacts) == {"apd:Test/a.yml", "apd:Test/b.yml"}


def test_unselected_entries_still_have_a_terminal_status(tmp_path):
    rows = [row(), row(entry_id="apd:Agriculture/Barley.yml", metadata_sha256="b" * 64)]
    with fake_services_server() as (base, _backend):
        results, _ = _project(
            rows, base, tmp_path, selected={"apd:Agriculture/Wheat-Prices.yml"}
        )
        by_entry = {item.entry_id: item for item in results}
        assert by_entry["apd:Agriculture/Barley.yml"].status == "deferred"
        assert by_entry["apd:Agriculture/Barley.yml"].reason_code == "not_selected"


def test_accounting_matches_discovery(tmp_path):
    rows = [row()]
    with fake_services_server() as (base, _backend):
        results, _ = _project(rows, base, tmp_path)
        accounting = graph.check_accounting(rows, results, 1)
        assert accounting.ok is True
        assert accounting.status_counts == {"indexed": 1}


def test_graph_unavailable_defers_every_entry(tmp_path):
    config = Config.load(data_root=tmp_path / "data", ledger_enabled=False).replace(
        graph_url="http://127.0.0.1:9", resume_base_delay_seconds=0.0, resume_max_delay_seconds=0.0
    )
    client = graph.graph_client(config)
    results, context = graph.project_entries(
        [row()], commit=COMMIT, config=config, client=client,
        selected={"apd:Agriculture/Wheat-Prices.yml"},
    )
    assert results[0].status == "deferred"
    assert results[0].reason_code in {"graph_unavailable", "graph_timeout"}
    assert context.snapshot.mode == "blind"


def test_prefetch_blind_mode_is_still_idempotent(tmp_path):
    """Even without a readable graph, deterministic ids keep writes idempotent."""
    with fake_services_server() as (base, backend):
        config = config_for(base, tmp_path)
        client = graph.graph_client(config)
        monkey = client.prefetch
        client.prefetch = lambda: graph.GraphSnapshot(mode="blind")  # type: ignore[assignment]
        for _ in range(2):
            results, _ = graph.project_entries(
                [row()], commit=COMMIT, config=config, client=client,
                selected={"apd:Agriculture/Wheat-Prices.yml"},
            )
            assert results[0].status in {"indexed", "unchanged"}
        assert len(backend.nodes) == 8
        assert len(backend.edges) == 7
        client.prefetch = monkey  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_prune_is_a_dry_run_by_default(tmp_path):
    with fake_services_server() as (base, backend):
        _project([row()], base, tmp_path)
        client = graph.graph_client(config_for(base, tmp_path))
        preview = graph.prune_tombstoned(client, ["apd:Agriculture/Wheat-Prices.yml"])
        assert preview["dry_run"] is True
        assert preview["would_remove"] == 2  # Dataset plus its DatasetVersion
        assert backend.nodes  # nothing removed


def test_prune_removes_the_dataset_and_its_versions(tmp_path):
    with fake_services_server() as (base, backend):
        _project([row()], base, tmp_path)
        client = graph.graph_client(config_for(base, tmp_path))
        result = graph.prune_tombstoned(
            client, ["apd:Agriculture/Wheat-Prices.yml"], dry_run=False
        )
        assert result["removed"] == 2
        remaining = {node["type"] for node in backend.nodes.values()}
        assert graph.NODE_DATASET not in remaining
        assert graph.NODE_VERSION not in remaining


# ---------------------------------------------------------------------------
# CLI-level reporting
# ---------------------------------------------------------------------------


def test_graph_manifest_rows_are_deterministic(tmp_path):
    with fake_services_server() as (base, _backend):
        results, _ = _project([row()], base, tmp_path)
        artifact = graph.manifest_row(row(), results[0])
        assert artifact["graph"]["status"] == "indexed"
        assert artifact["graph"]["nodes_created"] == 8
        first = graph.graph_manifest_bytes([artifact])
        second = graph.graph_manifest_bytes([artifact])
        assert first == second


def test_outcome_rows_are_deduplicated_by_event_id(tmp_path):
    with fake_services_server() as (base, _backend):
        results, _ = _project([row()], base, tmp_path)
        outcomes = graph.outcome_rows(
            [row()], results, run_id="r", catalog_commit=COMMIT, retrieved_at="t"
        )
        repeated = graph.outcome_rows(
            [row()], results, run_id="r", catalog_commit=COMMIT, retrieved_at="t"
        )
        assert [item["event_id"] for item in outcomes] == [
            item["event_id"] for item in repeated
        ]
        assert outcomes[0]["stage"] == "graph"
        assert outcomes[0]["attempts"] == 1


def test_resume_stats_count_reused_and_exhausted_results(tmp_path):
    results = [
        graph.EntryGraphResult(entry_id="a", status="unchanged", resumed=True),
        graph.EntryGraphResult(
            entry_id="b",
            status="deferred",
            reason_code=resume.RETRY_EXHAUSTED_REASON,
            attempts=12,
            resumed=True,
        ),
        graph.EntryGraphResult(entry_id="c", status="deferred", reason_code="not_selected"),
    ]
    stats = graph.resume_stats(results, None)
    assert stats["reused"] == 2
    assert stats["retry_exhausted"] == 1
    assert stats["attempted"] == 0


def test_duplicate_entry_ids_still_get_one_status_each(tmp_path):
    """A catalog with duplicate ids must keep ``discovered == statuses``."""
    rows = [row(), row()]
    with fake_services_server() as (base, _backend):
        results, _context = _project(rows, base, tmp_path)
        assert len(results) == 2
        accounting = graph.check_accounting(rows, results, 2)
        assert accounting.ok is True
        assert accounting.status_total == 2
