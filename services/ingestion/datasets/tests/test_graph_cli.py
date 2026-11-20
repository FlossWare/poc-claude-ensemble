"""PR 5 CLI: graph projection, resume after a kill, and duplicate-free recovery.

The interruption tests are real: the CLI runs in a child process, the fake
services deliberately block mid-stage, and the child is killed with SIGKILL.
What survives is only what the pipeline actually persisted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ingestion.datasets import catalog, manifest
from ingestion.datasets.config import Config
from ingestion.datasets.tests.helpers import (
    FakeServicesBackend,
    FakeStreamResponse,
    FakeStreamTransport,
    fake_services_server,
    make_catalog,
    service_env,
)
from ingestion.datasets.tests.test_normalize_cli import _seed
from ingestion.datasets import acquire
from ingestion.datasets.resolver import ResolvedSource

REPO_ROOT = Path(__file__).resolve().parents[4]
CSV_URL = "https://data.example.org/table.csv"
CSV_BODY = b"name,value\nwheat,1\nbarley,2\n"

ENTRIES = (
    "full.yml",
    "license-list.yml",
    "sparse.yml",
    "restricted.yml",
    "proprietary.yml",
    "typo-name.yml",
    "hhttps.yml",
)


def _catalog(tmp_path: Path) -> Path:
    return make_catalog(tmp_path, names=ENTRIES)


def _env(base: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(service_env(base))
    env["PYTHONPATH"] = str(REPO_ROOT / "services")
    env["POC_CLAUDE_ENSEMBLE_DATASET_LEDGER_ENABLED"] = "0"
    env["POC_CLAUDE_ENSEMBLE_DATASET_RESUME_BASE_DELAY_SECONDS"] = "0"
    env["POC_CLAUDE_ENSEMBLE_DATASET_RESUME_MAX_DELAY_SECONDS"] = "0"
    env.pop("POC_CLAUDE_ENSEMBLE_DATASET_DATA_ROOT", None)
    return env


def _run(args: list[str], *, base: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ingestion.dataset_ingest", *args],
        cwd=REPO_ROOT,
        env=_env(base),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _spawn(args: list[str], *, base: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ingestion.dataset_ingest", *args],
        cwd=REPO_ROOT,
        env=_env(base),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(path: Path, *, timeout: float = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def _checkpoint_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("entry_id"), str):
            ids.append(row["entry_id"])
    return ids


def _manifest_rows(data_root: Path, suffix: str) -> list[dict[str, object]]:
    path = next((data_root / "manifest").glob(f"*{suffix}"), None)
    assert path is not None, f"no {suffix} manifest under {data_root / 'manifest'}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Projection through the CLI
# ---------------------------------------------------------------------------


def test_graph_cli_projects_the_whole_catalog(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    with fake_services_server() as (base, backend):
        result = _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"graph: discovered={len(ENTRIES)}" in result.stdout
        assert "invariant=ok" in result.stdout
        rows = _manifest_rows(data_root, ".graph.jsonl")
        assert len(rows) == len(ENTRIES)
        assert {row["graph"]["status"] for row in rows} == {"indexed"}
        for row in rows:
            assert row["graph"]["nodes_created"] > 0

        second = _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        assert second.returncode == 0, second.stderr
        assert f"'unchanged': {len(ENTRIES)}" in second.stdout
        backend.reset()
        third = _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        assert third.returncode == 0, third.stderr
        assert backend.count("/graph/add-node") == 0
        assert backend.count("/graph/add-edge") == 0


def test_graph_cli_dry_run_writes_nothing(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    with fake_services_server() as (base, backend):
        result = _run(
            [
                "graph", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--dry-run",
            ],
            base=base,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "dry-run: no graph writes" in result.stdout
        assert backend.nodes == {}
        assert backend.edges == {}
        assert not (data_root / "manifest").exists()
        assert not (data_root / "runs").exists()


def test_graph_cli_limit_marks_the_rest_not_selected(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    with fake_services_server() as (base, _backend):
        result = _run(
            [
                "graph", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--limit", "2",
            ],
            base=base,
        )
        assert result.returncode == 3  # budget cap, as the other stages report it
        rows = _manifest_rows(data_root, ".graph.jsonl")
        assert sum(1 for row in rows if row["graph"]["status"] == "indexed") == 2
        assert (
            sum(1 for row in rows if row["graph"]["reason_code"] == "not_selected")
            == len(ENTRIES) - 2
        )


def test_graph_cli_prune_previews_then_applies(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    with fake_services_server() as (base, backend):
        assert _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        ).returncode == 0

        # Mark one entry tombstoned in the integration manifest the graph stage reads.
        entry = next(
            row["entry_id"]
            for row in _manifest_rows(data_root, ".graph.jsonl")
            if row["graph"]["status"] == "indexed"
        )
        integration_path = (
            data_root / "manifest" / next((data_root / "manifest").glob("*.graph.jsonl")).name.replace(
                ".graph.jsonl", ".integration.jsonl"
            )
        )
        integration_path.write_text(
            json.dumps(
                {
                    "entry_id": entry,
                    "integration": {
                        "status": "skipped",
                        "metadata": {"status": "skipped", "reason_code": "source_tombstoned"},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        nodes_before = len(backend.nodes)
        preview = _run(
            [
                "graph", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--prune-tombstoned",
            ],
            base=base,
        )
        assert preview.returncode == 0, preview.stderr
        assert "prune: dry_run=True" in preview.stdout
        assert len(backend.nodes) == nodes_before

        applied = _run(
            [
                "graph", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--prune-tombstoned", "--apply",
            ],
            base=base,
        )
        assert applied.returncode == 0, applied.stderr
        assert "prune: dry_run=False" in applied.stdout
        assert len(backend.nodes) < nodes_before


def test_graph_cli_reports_a_dead_service_honestly(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    env = _env("http://127.0.0.1:9")
    result = subprocess.run(
        [
            sys.executable, "-m", "ingestion.dataset_ingest", "graph",
            "--catalog-dir", str(root), "--data-root", str(data_root),
            "--no-ledger", "--max-attempts", "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "deferred" in result.stdout
    assert "graph_unavailable" in result.stdout or "graph_timeout" in result.stdout


# ---------------------------------------------------------------------------
# Interruption and recovery
# ---------------------------------------------------------------------------


def test_graph_cli_survives_sigkill_and_resumes_without_duplicates(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    marker = tmp_path / "blocked"
    with fake_services_server() as (base, backend):
        # Block the first write of the *second* entry: compute the exact number
        # of node writes the first entry makes, so the kill always lands after
        # one entry has been fully checkpointed.
        block_after = _first_entry_node_writes(root, data_root)
        backend.block_path = "/graph/add-node"
        backend.block_after = block_after
        backend.block_marker = marker
        child = _spawn(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        try:
            assert _wait_for(marker), "the child never reached the blocking write"
            child.kill()  # SIGKILL: no cleanup, no flush, no atexit
            child.wait(timeout=30)
        finally:
            backend.block_path = None
            backend.release.set()
            if child.poll() is None:
                child.kill()

        checkpoint = data_root / "cache" / "checkpoints" / "graph.jsonl"
        assert checkpoint.exists(), "completed entries must be checkpointed before the kill"
        assert _checkpoint_ids(checkpoint), "the checkpoint should name the finished entries"
        assert child.returncode != 0

        # Restart the same command.  The checkpoint covers the completed
        # entries and the deterministic ids cover everything already written.
        backend.reset()
        second = _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        assert second.returncode == 0, second.stdout + second.stderr
        assert "invariant=ok" in second.stdout

        rows = _manifest_rows(data_root, ".graph.jsonl")
        assert len(rows) == len(ENTRIES)
        statuses = [row["graph"]["status"] for row in rows]
        assert set(statuses) <= {"indexed", "unchanged", "skipped"}
        assert "failed" not in statuses
        assert all(
            row["graph"]["reason_code"] != "graph_conflict" for row in rows
        )

        # The restart asserted nothing that was already present, and produced
        # no conflicting write: the deterministic ids did the deduplication.
        assert backend.repeated_node_writes == 0
        assert backend.repeated_edge_writes == 0
        assert any(row["graph"].get("resumed") for row in rows)

        # A third run is a pure no-op: nothing is created at all.
        backend.reset()
        third = _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        )
        assert third.returncode == 0, third.stderr
        assert backend.count("/graph/add-node") == 0
        assert backend.count("/graph/add-edge") == 0

        # Outcome rows are deduplicated by deterministic event id.
        outcomes = [
            json.loads(line)
            for line in (data_root / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        graph_events = [row["event_id"] for row in outcomes if row["stage"] == "graph"]
        assert len(graph_events) == len(set(graph_events))


def test_normalize_cli_survives_sigkill_and_resumes_without_duplicates(tmp_path, monkeypatch):
    root, data_root, config, build = _setup_acquire(tmp_path, monkeypatch)
    marker = tmp_path / "blocked"
    with fake_services_server() as (base, backend):
        # Entry one writes metadata plus two content shards; block on the first
        # write of entry two, so entry one is fully finished when the kill lands.
        backend.block_path = "/records"
        backend.block_after = 4
        backend.block_marker = marker
        child = _spawn(
            [
                "normalize", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--include-metadata-only",
            ],
            base=base,
        )
        try:
            assert _wait_for(marker), "the child never reached the blocking write"
            child.kill()
            child.wait(timeout=30)
        finally:
            backend.block_path = None
            backend.release.set()
            if child.poll() is None:
                child.kill()

        checkpoint = data_root / "cache" / "checkpoints" / "integrate.jsonl"
        assert checkpoint.exists()
        completed = set(_checkpoint_ids(checkpoint))
        assert completed, "the finished entry must be checkpointed before the kill"
        assert len(completed) < len(ENTRIES)

        backend.reset()
        second = _run(
            [
                "normalize", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--include-metadata-only",
            ],
            base=base,
        )
        assert second.returncode == 0, second.stdout + second.stderr
        assert "invariant=ok" in second.stdout
        assert "resumed=" in second.stdout

        rows = _manifest_rows(data_root, ".integration.jsonl")
        assert len(rows) == len(ENTRIES)
        for row in rows:
            assert row["integration"]["status"] in {
                "indexed",
                "metadata_indexed",
                "unchanged",
                "skipped",
                "deferred",
                "failed",
            }
        # Every checkpointed entry was reused, not re-sent to Knowledge.
        resumed = {row["entry_id"] for row in rows if row["integration"].get("resumed")}
        assert completed <= resumed

        # No duplicate outcome rows, and one Knowledge revision per source.
        outcomes = [
            json.loads(line)
            for line in (data_root / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        integrate_events = [row["event_id"] for row in outcomes if row["stage"] == "integrate"]
        assert len(integrate_events) == len(set(integrate_events))
        # One accepted revision per source: the resumed entry was not re-sent.
        accepted = [
            source_id
            for source_id, decision in backend.decisions
            if decision == "accepted"
        ]
        assert len(accepted) == len(set(accepted))
        assert backend.knowledge


# ---------------------------------------------------------------------------
# Acquisition setup for the normalize interruption test
# ---------------------------------------------------------------------------


def _setup_acquire(tmp_path: Path, monkeypatch):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    config = Config.load(catalog_dir=root, data_root=data_root, ledger_enabled=False)
    snapshot = catalog.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)
    source = ResolvedSource(
        uri=CSV_URL,
        kind="direct_file",
        method="direct_file",
        media_type="text/csv",
        member="table.csv",
    )
    # Give two entries the same resolved source so both acquire real content.
    for entry_id in ("apd:Test/full.yml", "apd:Test/license-list.yml"):
        _seed(config, build.rows, entry_id, [source])
    transport = FakeStreamTransport(
        {
            CSV_URL: FakeStreamResponse(
                CSV_BODY, headers={"content-length": str(len(CSV_BODY))}
            )
        }
    )
    monkeypatch.setattr(acquire, "build_transport", lambda config: transport)
    from ingestion import dataset_ingest

    assert (
        dataset_ingest.main(
            [
                "acquire", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger",
            ]
        )
        == 0
    )
    return root, data_root, config, build


def _first_entry_node_writes(root: Path, data_root: Path) -> int:
    """How many ``add-node`` calls the first entry projects, plus one.

    Derived from the same projection code the CLI runs, which keeps the
    interruption test deterministic instead of hoping a fixed count lands in
    the right entry.
    """
    from ingestion.datasets import graph

    config = Config.load(catalog_dir=root, data_root=data_root, ledger_enabled=False)
    snapshot = catalog.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)
    first = sorted(build.rows, key=lambda item: str(item["entry_id"]))[0]
    projection = graph.project_row(first, commit=snapshot.commit)
    return len(projection.nodes) + 1


def test_graph_cli_dry_run_never_prunes(tmp_path):
    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    with fake_services_server() as (base, backend):
        assert _run(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"],
            base=base,
        ).returncode == 0
        nodes_before = len(backend.nodes)
        integration_path = data_root / "manifest" / next(
            (data_root / "manifest").glob("*.graph.jsonl")
        ).name.replace(".graph.jsonl", ".integration.jsonl")
        integration_path.write_text(
            json.dumps(
                {
                    "entry_id": "apd:Test/full.yml",
                    "integration": {
                        "status": "skipped",
                        "metadata": {"status": "skipped", "reason_code": "source_tombstoned"},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = _run(
            [
                "graph", "--catalog-dir", str(root), "--data-root", str(data_root),
                "--no-ledger", "--dry-run", "--prune-tombstoned", "--apply",
            ],
            base=base,
        )
        assert result.returncode == 0, result.stderr
        assert "prune: dry_run=True" in result.stdout
        assert len(backend.nodes) == nodes_before


def test_graph_cli_prefers_the_persisted_inventory_manifest(tmp_path, monkeypatch):
    """A projection resumes from the PR 1 manifest instead of re-parsing YAML."""
    from ingestion import dataset_ingest

    root = _catalog(tmp_path)
    data_root = tmp_path / "data"
    config = Config.load(catalog_dir=root, data_root=data_root, ledger_enabled=False)
    snapshot = catalog.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)
    inventory_path = data_root / "manifest" / dataset_ingest._manifest_name(snapshot)
    manifest.write_manifest(build.rows, inventory_path)

    # Keep the same snapshot identity but make every entry unparseable: only
    # the persisted manifest can still describe the catalog.
    monkeypatch.setattr(dataset_ingest.catalog_module, "load_catalog", lambda config: snapshot)
    for path in (root / "core" / "Test").iterdir():
        path.write_text("not: [valid", encoding="utf-8")

    with fake_services_server() as (base, _backend):
        for key, value in service_env(base).items():
            monkeypatch.setenv(key, value)
        code = dataset_ingest.main(
            ["graph", "--catalog-dir", str(root), "--data-root", str(data_root), "--no-ledger"]
        )
        assert code == 0
        report = json.loads(next((data_root / "runs").glob("*.json")).read_text())
        assert report["graph"]["inventory_source"] == "manifest"
        assert report["funnel"]["discovered"] == len(ENTRIES)
        assert report["funnel"]["created"] == len(ENTRIES)
