"""Target A adapter: policy application, rollback, and stub refusals."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx
from learning.ga import journal
from learning.ga import targets
from learning.ga.targets import thompson_policy as adapter
from shared import thompson_policy as policy
from shared.thompson_policy import default_params, load_policy_file


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def _write_prior(path: Path, policies: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(policy.policy_document(policies), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    policy.clear_cache()
    return encoded


def test_baseline_defaults_when_no_policy_file(isolated):
    policies = adapter.baseline_policies()
    assert policies["default"] == default_params()
    assert policies["code-review"] == default_params()


def test_baseline_reads_the_live_policy_file(isolated):
    path = Path(policy.policy_file_path())
    _write_prior(path, {"default": dict(default_params(), cost_weight=0.4)})
    assert adapter.baseline_policies()["default"]["cost_weight"] == 0.4


def test_invalid_policy_file_is_ignored(isolated):
    path = Path(policy.policy_file_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ definitely not json", encoding="utf-8")
    policy.clear_cache()
    assert adapter.baseline_policies()["default"] == default_params()


def test_apply_then_rollback_restores_bytes(isolated):
    path = Path(policy.policy_file_path())
    prior = _write_prior(path, {"default": default_params(), "code-review": default_params()})

    candidate = {"default": dict(default_params(), cost_weight=0.75, prior_strength=2.0)}
    policies = {"default": candidate["default"], "code-review": default_params()}
    applied = adapter.apply_policies(policies, {"change_id": "change-1"})

    assert path.exists()
    loaded = load_policy_file(path)
    assert loaded["default"]["cost_weight"] == 0.75
    assert loaded["default"]["prior_strength"] == 2.0

    entry = {"kind": "apply", "change_id": "change-1"}
    entry.update(applied.journal_fields())
    restored = adapter.rollback_change(entry)

    assert path.read_bytes() == prior
    assert restored["removed"] is False
    assert restored["sha256"] == applied.previous_sha256
    assert load_policy_file(path)["default"] == default_params()


def test_rollback_removes_a_file_that_did_not_exist(isolated):
    path = Path(policy.policy_file_path())
    assert not path.exists()
    applied = adapter.apply_policies({"default": dict(default_params(), cost_weight=0.5)}, {})
    assert path.exists()
    entry = {"kind": "apply", "change_id": "change-2"}
    entry.update(applied.journal_fields())
    adapter.rollback_change(entry)
    assert not path.exists()


def test_apply_rejects_out_of_bounds(isolated):
    with pytest.raises(ValueError):
        adapter.apply_policies({"default": dict(default_params(), cost_weight=3.0)}, {})
    assert not Path(policy.policy_file_path()).exists()


def test_stub_targets_refuse():
    for name in ("rag_config", "review_pipeline"):
        stub = targets.get_target(name)
        assert stub.PROMOTABLE is False
        with pytest.raises(NotImplementedError):
            stub.evaluate({})
        with pytest.raises(NotImplementedError):
            stub.apply({})
        with pytest.raises(NotImplementedError):
            stub.load_corpora()
        assert stub.param_specs()


def test_unknown_target_is_a_key_error():
    with pytest.raises(KeyError):
        targets.get_target("not-a-target")


FORBIDDEN_ROOTS = {"urllib", "requests", "http", "socket", "thompson_client", "ssl", "ftplib"}


def test_adapter_modules_have_no_network_imports():
    """Test 16: no network imports in any adapter module."""
    for module in (adapter, targets.rag_config, targets.review_pipeline):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert not (roots & FORBIDDEN_ROOTS), (module.__name__, roots & FORBIDDEN_ROOTS)


def test_code_hash_and_deployed_status(isolated):
    digest = adapter.code_hash()
    assert len(digest) == 64
    status = adapter.deployed_status()
    assert status["repo_sha256"] is not None
    assert status["deployed_sha256"] is None  # no deployed copy in the test environment
    assert status["matches"] is None
