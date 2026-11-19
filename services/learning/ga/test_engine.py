"""Replay correctness and search determinism (design test plan 6-10)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from learning.ga import _fixtures as fx
from learning.ga import corpus as corpus_mod
from learning.ga import engine
from learning.ga.schema import canonical_json, default_candidate, make_candidate


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def _turn_corpus(arms_rows):
    corpus = corpus_mod.Corpus(kind="turn")
    corpus.rows = list(arms_rows)
    corpus.universe = sorted({row.arm for row in corpus.rows})
    return corpus


def _review_corpus():
    rows = [
        corpus_mod.Row("r.jsonl", 1, "r1a", fx.EPOCH, "alpha", True, 1.0, 10, "r1", "code-review"),
        corpus_mod.Row("r.jsonl", 2, "r1b", fx.EPOCH, "beta", False, 1.0, 10, "r1", "code-review"),
        corpus_mod.Row("r.jsonl", 3, "r2a", fx.EPOCH + 1, "alpha", True, 1.0, 10, "r2", "code-review"),
        corpus_mod.Row("r.jsonl", 4, "r2b", fx.EPOCH + 1, "beta", True, 1.0, 10, "r2", "code-review"),
        corpus_mod.Row("r.jsonl", 5, "r3a", fx.EPOCH + 2, "alpha", False, 3.0, 10, "r3", "code-review"),
        corpus_mod.Row("r.jsonl", 6, "r3b", fx.EPOCH + 2, "beta", True, 1.0, 10, "r3", "code-review"),
    ]
    corpus = corpus_mod.Corpus(kind="review", rows=rows, universe=["alpha", "beta"])
    corpus.as_of = fx.EPOCH + 2
    return corpus


def test_no_lookahead_in_turn_replay():
    """Test 6: the state at step i contains only rows strictly before i."""
    rows = [
        corpus_mod.Row("o.jsonl", i, f"e{i}", fx.EPOCH + i, "alpha" if i % 2 else "beta",
                       i % 3 != 0, 0.1, 10, None, None)
        for i in range(60)
    ]
    corpus = _turn_corpus(rows)
    cfg = engine.SearchConfig(holdout_fraction=0.3)
    result = engine.replay_turn(default_candidate(), corpus, "train", seed=11, cfg=cfg, trace=True)
    positions = {row.row_key: index for index, row in enumerate(corpus.rows)}
    assert result.trace
    for entry in result.trace:
        assert entry["prior_total"] == positions[entry["row_key"]]
        assert entry["prior_calls"] <= entry["prior_total"]


def test_chronological_split_is_disjoint(isolated):
    """Test 7: replayed train and holdout decisions never share a row."""
    fx.write_jsonl(isolated / "outcomes" / "outcomes.jsonl", fx.passing_turn_rows(days=10, per_day=6))
    corpus = corpus_mod.ingest_turn_corpus()
    train, holdout = corpus_mod.split_rows(corpus.rows, 0.30)
    assert {row.row_key for row in train}.isdisjoint({row.row_key for row in holdout})
    assert max(row.ts for row in train) < min(row.ts for row in holdout)


def test_replay_is_deterministic(isolated):
    """Test 8: the same corpus, seed, and parameters give byte-identical results."""
    fx.write_jsonl(isolated / "outcomes" / "outcomes.jsonl", fx.passing_turn_rows(days=10, per_day=6))
    corpus = corpus_mod.ingest_turn_corpus()
    cfg = engine.SearchConfig()
    first = engine.replay_turn(default_candidate(), corpus, "holdout", seed=5, cfg=cfg)
    second = engine.replay_turn(default_candidate(), corpus, "holdout", seed=5, cfg=cfg)
    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())
    assert first.contributions == second.contributions
    assert first.block_keys == second.block_keys


def test_full_information_review_replay_hand_computed():
    """Test 9: a 3-round fixture matches a hand-computed holdout fitness."""
    corpus = _review_corpus()
    params = dict(default_candidate(), capability_margin=1.0)
    cfg = engine.SearchConfig(holdout_fraction=0.30)
    result = engine.replay_review(params, corpus, "holdout", seed=3, cfg=cfg, trace=True)
    # Round 3 is the holdout; alpha (2/2 before the round) is the only arm above
    # the 1.0 margin, so the candidate picks alpha and is scored on its realized
    # row: success False, cost 3.0. cost_ref = mean holdout cost = (3.0+1.0)/2.
    assert result.decisions == 1
    assert result.skipped_steps == 0
    assert result.succ_hat == 0.0
    assert result.cost_hat == pytest.approx(3.0)
    assert result.cost_ref == pytest.approx(2.0)
    assert result.fitness == pytest.approx(-0.375)
    assert result.coverage == 1.0
    assert result.unsupported_fraction == 0.0
    assert result.per_arm["alpha"]["decisions"] == 1
    assert result.per_arm["alpha"]["logged_rows"] == 1
    assert result.per_arm["alpha"]["logged_success"] == 0.0
    assert result.trace[0]["arm"] == "alpha"
    assert result.trace[0]["prior_calls"] == 2
    assert result.estimator == "full_information_replay"


def test_review_train_window_hand_computed():
    corpus = _review_corpus()
    params = dict(default_candidate(), capability_margin=1.0)
    result = engine.replay_review(params, corpus, "train", seed=3, cfg=engine.SearchConfig(holdout_fraction=0.30))
    # Round 1 has no priors (skipped); round 2 is the only train decision.
    assert result.skipped_steps == 1
    assert result.decisions == 1
    assert result.succ_hat == 1.0
    assert result.cost_hat == pytest.approx(1.0)
    assert result.fitness == pytest.approx(0.75)


def test_unsupported_penalty_and_coverage():
    """Test 10: an arm absent from the holdout is unsupported and uncovered."""
    train = [
        corpus_mod.Row("o.jsonl", i, f"t{i}", fx.EPOCH + i, "common" if i % 2 else "rare",
                       True, 1.0 if i % 2 else 0.0, 10, None, None)
        for i in range(100)
    ]
    holdout = [
        corpus_mod.Row("o.jsonl", 100 + i, f"h{i}", fx.EPOCH + 100 + i, "common", True, 1.0, 10, None, None)
        for i in range(40)
    ]
    corpus = _turn_corpus(train + holdout)
    params = dict(default_candidate(), cost_weight=1.0)
    cfg = engine.SearchConfig(holdout_fraction=len(holdout) / len(train + holdout))
    result = engine.replay_turn(params, corpus, "holdout", seed=1, cfg=cfg)
    assert result.decisions == 40
    assert result.unsupported_fraction == 1.0
    assert result.coverage == 0.0
    assert result.per_arm["rare"]["logged_rows"] == 0

    sparse = [
        corpus_mod.Row("o.jsonl", 100 + i, f"h{i}", fx.EPOCH + 100 + i,
                       "rare" if i < 5 else "common", True, 0.0 if i < 5 else 1.0, 10, None, None)
        for i in range(40)
    ]
    sparse_corpus = _turn_corpus(train + sparse)
    sparse_result = engine.replay_turn(params, sparse_corpus, "holdout", seed=1, cfg=cfg)
    assert sparse_result.unsupported_fraction == 1.0
    assert sparse_result.coverage == 1.0  # the arm did run in the holdout, just not enough
    assert sparse_result.per_arm["rare"]["logged_rows"] == 5


def test_search_is_deterministic_and_bounded(isolated):
    fx.write_jsonl(isolated / "outcomes" / "outcomes.jsonl", fx.passing_turn_rows())
    corpus = corpus_mod.ingest_turn_corpus()
    cfg = engine.SearchConfig(population=8, generations=2, max_evals=16, elites=2)
    first = engine.search_turn(default_candidate(), corpus, cfg, seed=1234)
    second = engine.search_turn(default_candidate(), corpus, cfg, seed=1234)
    assert first.best_params == second.best_params
    assert first.evaluations == second.evaluations
    assert first.generations == second.generations
    assert first.evaluations <= 16
    for name, value in first.best_params.items():
        spec = engine.PARAM_BOUNDS[name]
        assert spec.low <= value <= spec.high


def test_search_retains_the_default_candidate():
    corpus = _turn_corpus(
        [corpus_mod.Row("o.jsonl", i, f"e{i}", fx.EPOCH + i, "a", True, 0.1, 1, None, None) for i in range(20)]
    )
    cfg = engine.SearchConfig(population=4, generations=2, max_evals=8)
    outcome = engine.search_turn(default_candidate(), corpus, cfg, seed=1)
    # The default candidate is always evaluated (and can win).
    assert outcome.evaluations >= 1
    assert make_candidate(outcome.best_params) == outcome.best_params


def test_fitness_weights_are_not_genes():
    from learning.ga.schema import GENES

    assert set(GENES) == {"prior_strength", "cost_weight", "exploration_epsilon", "capability_margin"}
    assert "cost_lambda" not in GENES
    assert "unsupported_mu" not in GENES
    assert engine.COST_LAMBDA == 0.25
    assert engine.UNSUPPORTED_MU == 0.10
    cfg = engine.SearchConfig()
    assert cfg.cost_lambda == 0.25 and cfg.unsupported_mu == 0.10
