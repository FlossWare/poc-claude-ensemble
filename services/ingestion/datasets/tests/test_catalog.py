"""Catalog walking, README reconciliation and tarball safety."""
from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

import pytest

from ingestion.datasets import catalog, manifest

from ingestion.datasets.tests.helpers import (
    ENTRY_FIXTURES,
    FIXTURES,
    device,
    directory,
    hardlink,
    make_catalog,
    make_config,
    regular,
    symlink,
    tar_bytes,
)


def test_walk_finds_every_suffix(tmp_path):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    paths = {item.relpath for item in snapshot.files}
    assert len(paths) == len(ENTRY_FIXTURES)
    assert "Test/full.yml" in paths
    assert "Test/OpenNEURO" in paths
    assert "Test/LocationData.wml" in paths
    assert "Test/San Diego, CA.yml" in paths
    assert snapshot.commit == "92d1b210bfba28055f672e6331eb5e5fa27227c5" or len(snapshot.commit) in {40, 64}
    assert snapshot.catalog_license == "MIT"
    assert snapshot.metadata_repo_license == "GPL-3.0"


def test_walk_includes_yaml_suffix_and_yml_only_would_lose_entries(tmp_path):
    root = make_catalog(tmp_path)
    shutil.copyfile(
        root / "core" / "Test" / "full.yml",
        root / "core" / "Test" / "alternate.yaml",
    )
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    paths = {item.relpath for item in snapshot.files}
    assert "Test/alternate.yaml" in paths
    yml_only = {path for path in paths if path.endswith(".yml")}
    # A *.yml-only walk silently loses .yaml, extensionless and .wml entries.
    assert yml_only != paths


def test_readme_reconciliation_uses_generated_readme(tmp_path):
    root = make_catalog(
        tmp_path, statuses={"full.yml": "OK", "sparse.yml": "FIXME"}
    )
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    assert snapshot.readme_path is not None
    assert snapshot.readme_path.name == "README.rst"
    assert snapshot.readme_meta_total == len(ENTRY_FIXTURES)
    assert snapshot.readme_entries["Test/full.yml"]["status"] == "OK"
    assert snapshot.readme_entries["Test/sparse.yml"]["status"] == "FIXME"
    assert snapshot.readme_warnings == []


def test_readme_parse_rejects_unsafe_and_duplicate_paths():
    text = "\n".join(
        [
            "* |OK_ICON| `x <https://example.org>`_ [`Meta <https://github.com/o/r/tree/m/core//A/../evil.yml>`_]",
            "* |OK_ICON| `x <https://example.org>`_ [`Meta <https://github.com/o/r/tree/m/core//A/ok.yml>`_]",
            "* |FIXME_ICON| `x <https://example.org>`_ [`Meta <https://github.com/o/r/tree/m/core//A/ok.yml>`_]",
            "* |OK_ICON| `x <https://example.org>`_ [`Meta <https://elsewhere.example/x>`_]",
        ]
    )
    entries, total, warnings = catalog.parse_readme(text)
    assert total == 4
    assert entries == {"A/ok.yml": {"status": "OK", "url": "https://github.com/o/r/tree/m/core//A/ok.yml"}}
    assert "readme_meta_unsafe_path" in warnings
    assert "readme_duplicate_meta:A/ok.yml" in warnings
    assert "readme_meta_outside_core" in warnings


def test_partial_failure_file_is_still_inventoried(tmp_path):
    root = make_catalog(tmp_path)
    (root / "core" / "Test" / "bad.yml").write_text("[unclosed\n", encoding="utf-8")
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    assert "Test/bad.yml" in {item.relpath for item in snapshot.files}


def test_symlinked_file_and_directory_are_refused(tmp_path):
    root = make_catalog(tmp_path)
    outside = tmp_path / "outside.yml"
    outside.write_text("title: outside\n", encoding="utf-8")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    os.symlink(outside, root / "core" / "Test" / "link.yml")
    os.symlink(outside_dir, root / "core" / "Test" / "linkdir")
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    unsafe = {
        item.relpath: item.unsafe_reason
        for item in snapshot.files
        if item.unsafe_reason
    }
    assert unsafe["Test/link.yml"] == "symlink"
    assert unsafe["Test/linkdir"] == "symlink"


def test_file_cap_is_a_snapshot_failure(tmp_path):
    root = make_catalog(tmp_path)
    config = make_config(tmp_path, root).replace(catalog_max_files=2)
    with pytest.raises(catalog.CatalogError) as error:
        catalog.snapshot_from_directory(root, config)
    assert error.value.code == "catalog_snapshot_failed"


def test_oversized_entry_is_marked_not_dropped(tmp_path):
    root = make_catalog(tmp_path)
    (root / "core" / "Test" / "big.yml").write_text("x" * 200, encoding="utf-8")
    config = make_config(tmp_path, root).replace(catalog_max_file_bytes=50)
    snapshot = catalog.snapshot_from_directory(root, config)
    record = next(item for item in snapshot.files if item.relpath == "Test/big.yml")
    assert record.unsafe_reason == "oversized"


def test_valid_tarball_is_expanded_member_by_member(tmp_path):
    config = make_config(tmp_path)
    data = (FIXTURES / "catalog.tar.gz").read_bytes()
    destination = catalog.extract_tarball(
        data, tmp_path / "out", expected_roots={"apd-core-test"}, config=config
    )
    assert (destination / "core" / "Test" / "full.yml").is_file()
    assert (destination / "LICENSE").is_file()
    assert not (destination / "core" / "Test" / "full.yml").is_symlink()


def _entries(*pairs):
    return [(info, data) for info, data in pairs]


def _extract(data, tmp_path, config, expected_roots={"root"}):
    return catalog.extract_tarball(
        data, tmp_path / "out", expected_roots=expected_roots, config=config
    )


@pytest.mark.parametrize(
    "entries",
    [
        [(directory("root/"), None), (regular("root/../evil", b"x"), b"x")],
        [(directory("root/"), None), (regular("/root/evil", b"x"), b"x")],
        [(directory("root/"), None), (regular("C:/evil", b"x"), b"x")],
        [(directory("root/"), None), (regular("root/a\\b", b"x"), b"x")],
        [(directory("root/"), None), (symlink("root/link", "../../etc/passwd"), None)],
        [(directory("root/"), None), (hardlink("root/hard", "root/target"), None)],
        [(directory("root/"), None), (device("root/null"), None)],
        [
            (directory("root/"), None),
            (regular("root/core/x", b"a"), b"a"),
            (regular("root/core/x", b"b"), b"b"),
        ],
        [(directory("root/"), None), (regular("root/C:evil", b"x"), b"x")],
    ],
)
def test_unsafe_tar_members_are_rejected(entries, tmp_path):
    config = make_config(tmp_path)
    data = tar_bytes(entries)
    with pytest.raises(catalog.CatalogError):
        _extract(data, tmp_path, config)


def test_tar_root_must_match_pinned_commit(tmp_path):
    config = make_config(tmp_path)
    data = tar_bytes(
        [(directory("other/"), None), (regular("other/core/x.yml", b"title: x\n"), b"title: x\n")]
    )
    with pytest.raises(catalog.CatalogError):
        _extract(data, tmp_path, config, expected_roots={"apd-core-test"})


def test_tar_must_have_one_root(tmp_path):
    config = make_config(tmp_path)
    data = tar_bytes(
        [
            (regular("a/x.yml", b"title: x\n"), b"title: x\n"),
            (regular("b/y.yml", b"title: y\n"), b"title: y\n"),
        ]
    )
    with pytest.raises(catalog.CatalogError):
        _extract(data, tmp_path, config)


def test_tar_member_cap(tmp_path):
    config = make_config(tmp_path).replace(catalog_max_file_bytes=4)
    data = tar_bytes(
        [(directory("root/"), None), (regular("root/core/big.yml", b"12345"), b"12345")]
    )
    with pytest.raises(catalog.CatalogError):
        _extract(data, tmp_path, config)


def test_truncated_tar_is_rejected(tmp_path):
    config = make_config(tmp_path)
    data = (FIXTURES / "catalog.tar.gz").read_bytes()
    with pytest.raises(catalog.CatalogError):
        _extract(data[: len(data) // 2], tmp_path, config, expected_roots={"apd-core-test"})


@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_NETWORK_TESTS") != "1",
    reason="set RUN_DATASET_NETWORK_TESTS=1 with a local catalog checkout",
)
def test_real_pinned_snapshot(tmp_path):
    catalog_dir = os.environ.get("POC_CLAUDE_ENSEMBLE_DATASET_TEST_CATALOG_DIR")
    if not catalog_dir:
        pytest.skip("POC_CLAUDE_ENSEMBLE_DATASET_TEST_CATALOG_DIR is not set")
    root = Path(catalog_dir)
    config = make_config(tmp_path, root)
    snapshot = catalog.snapshot_from_directory(root, config)
    assert len(snapshot.files) == 1142
    assert len({item.category for item in snapshot.files}) == 36
    assert snapshot.readme_meta_total == 1142
    assert sum(1 for item in snapshot.files if item.unsafe_reason) == 0
    build = manifest.build_rows(snapshot, config)
    assert build.unhandled == 0
    assert sum(1 for row in build.rows if row["status"] == "failed") == 0
    # Definition pinned for PR 1: "license missing" means no usable `license`
    # value in the entry YAML (the free-text `copyrights` field is not evidence).
    assert sum(
        1
        for row in build.rows
        if row["license"]["dataset"]["family"] == "unknown"
    ) == 644
