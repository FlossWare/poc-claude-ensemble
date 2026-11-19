"""Corpus ingestion: taxonomy, projection, ordering, splits, and rechecks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from learning.ga import corpus as corpus_mod
from learning.ga import _fixtures as fx


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    return fx.configure_paths(monkeypatch, tmp_path)


def test_outcome_taxonomy_and_projection(isolated):
    path = isolated / "outcomes" / "outcomes.jsonl"
    rows = [
        fx.outcome_row("good-1", "haiku", True, 0.01, fx.EPOCH),
        fx.outcome_row("good-1", "haiku", False, 0.02, fx.EPOCH + 1),  # duplicate key
        fx.outcome_row("done-1", "haiku", True, 0.01, fx.EPOCH, status="accepted"),
        fx.outcome_row("wrong-kind", "haiku", True, 0.01, fx.EPOCH, kind="something_else"),
        "{not json",
        json.dumps([1, 2, 3]),
        json.dumps({"kind": "turn_outcome", "status": "pending", "success": True, "cost": 1.0,
                    "timestamp": fx.iso(fx.EPOCH)}),  # missing arm
        json.dumps({**fx.outcome_row("nan", "a", True, 1.0, fx.EPOCH), "cost": float("nan")}),
        json.dumps({**fx.outcome_row("inf", "a", True, 1.0, fx.EPOCH), "cost": float("inf")}),
        json.dumps({**fx.outcome_row("neg", "a", True, 1.0, fx.EPOCH), "cost": -1.0}),
        json.dumps({**fx.outcome_row("badarm", "not a valid arm!", True, 1.0, fx.EPOCH)}),
        json.dumps({**fx.outcome_row("nots", "a", True, 1.0, fx.EPOCH), "timestamp": "whenever"}),
        json.dumps({**fx.outcome_row("badsucc", "a", 1, 1.0, fx.EPOCH)}),
        json.dumps({**fx.outcome_row("badtok", "a", True, 1.0, fx.EPOCH), "tokens": -5}),
        json.dumps({**fx.outcome_row("float-tokens", "a", True, 1.0, fx.EPOCH), "tokens": 1.5}),
        fx.outcome_row("good-2", "sonnet", False, 0.5, fx.EPOCH + 2),
        "",  # blank lines are skipped, not rejected
    ]
    rows.insert(5, '{"a":' * 40 + "1" + "}" * 40)  # depth bomb
    fx.write_jsonl(path, rows)

    corpus = corpus_mod.ingest_turn_corpus()
    counts = corpus.counts
    assert counts["accepted"] == 2
    assert counts["duplicate"] == 1
    assert counts["skipped_status"] == 1
    assert counts["skipped_kind"] == 1
    assert counts["rejected_malformed_json"] == 3  # bad JSON, non-object, depth
    assert counts["rejected_schema"] == 7  # arm, neg cost, bad arm, ts, success, tokens x2
    assert counts["rejected_non_finite"] == 2
    assert corpus.total == 2
    assert [row.arm for row in corpus.rows] == ["haiku", "sonnet"]
    assert corpus.rows[0].task_type is None
    assert corpus.as_of == pytest.approx(fx.EPOCH + 2)
    # The raw payload never survives projection.
    assert not hasattr(corpus.rows[0], "text")
    assert "not a valid arm!" not in json.dumps([row._asdict() for row in corpus.rows])


def test_oversize_line_rejected_and_counted(isolated):
    path = isolated / "outcomes" / "outcomes.jsonl"
    huge = json.dumps({"kind": "turn_outcome", "status": "pending", "arm": "a", "success": True,
                       "cost": 1.0, "timestamp": fx.iso(fx.EPOCH), "text": "x" * (corpus_mod.MAX_LINE_BYTES + 10)})
    fx.write_jsonl(path, [huge, fx.outcome_row("ok", "a", True, 1.0, fx.EPOCH)])
    corpus = corpus_mod.ingest_turn_corpus()
    assert corpus.counts["rejected_oversize_line"] == 1
    assert corpus.counts["accepted"] == 1


def test_review_taxonomy_and_incomplete_rounds(isolated):
    path = isolated / "reviews" / "reviews.jsonl"
    rows = []
    for index in range(3):
        ts = fx.EPOCH + index * fx.DAY
        rows.append(fx.review_row(f"complete-{index}", "alpha", True, 0.1, ts))
        rows.append(fx.review_row(f"complete-{index}", "beta", True, 0.2, ts))
    # A round missing beta: excluded from full-information replay.
    rows.append(fx.review_row("incomplete", "alpha", True, 0.1, fx.EPOCH + 10 * fx.DAY))
    rows.append(fx.review_row("archived", "alpha", True, 0.1, fx.EPOCH, producer="archive-transcripts"))
    rows.append(fx.review_row("arbiter", "alpha", True, 0.1, fx.EPOCH, kind="arbiter_decision"))
    fx.write_jsonl(path, rows)

    corpus = corpus_mod.ingest_review_corpus()
    assert corpus.counts["accepted"] == 6
    assert corpus.counts["incomplete_round"] == 1
    assert corpus.counts["skipped_producer"] == 1
    assert corpus.counts["skipped_kind"] == 1
    assert len(corpus.rounds()) == 3
    assert corpus.universe == ["alpha", "beta"]
    assert all(row.task_type == "code-review" for row in corpus.rows)


def test_reviews_success_definition(isolated):
    path = isolated / "reviews" / "reviews.jsonl"
    rows = [
        fx.review_row("r", "alpha", True, 0.1, fx.EPOCH),
        fx.review_row("r", "beta", False, 0.1, fx.EPOCH),
        {**fx.review_row("r", "gamma", True, 0.1, fx.EPOCH), "error": "boom"},
    ]
    fx.write_jsonl(path, rows)
    corpus = corpus_mod.ingest_review_corpus()
    assert corpus.counts["incomplete_round"] == 0  # all three arms ran in the round
    by_arm = {row.arm: row.success for row in corpus.rows}
    assert by_arm == {"alpha": True, "beta": False, "gamma": False}  # error makes gamma unusable


def test_splits_are_chronological_and_disjoint():
    rows = [
        corpus_mod.Row("ledger.jsonl", i, f"e{i}", fx.EPOCH + i, "a", True, 0.1, 1, None, None)
        for i in range(10)
    ]
    train, holdout = corpus_mod.split_rows(rows, 0.30)
    assert len(train) == 7 and len(holdout) == 3
    assert not ({row.row_key for row in train} & {row.row_key for row in holdout})
    assert max(row.ts for row in train) < min(row.ts for row in holdout)
    assert corpus_mod.split_rows([], 0.3) == ([], [])
    single = corpus_mod.split_rows(rows[:1], 0.3)
    assert len(single[0]) == 1 and single[1] == []


def test_recheck_detects_changes(isolated):
    path = isolated / "outcomes" / "outcomes.jsonl"
    fx.write_jsonl(path, [fx.outcome_row("e", "a", True, 0.1, fx.EPOCH)])
    corpus = corpus_mod.ingest_turn_corpus()
    corpus_mod.recheck(corpus)
    fx.write_jsonl(path, [fx.outcome_row("e", "a", True, 0.2, fx.EPOCH)])
    with pytest.raises(corpus_mod.CorpusChangedError):
        corpus_mod.recheck(corpus)
    path.unlink()
    with pytest.raises(corpus_mod.CorpusChangedError):
        corpus_mod.recheck(corpus)


def test_missing_ledger_is_an_empty_corpus(isolated):
    corpus = corpus_mod.ingest_turn_corpus()
    assert corpus.rows == []
    assert corpus.ledgers[0].exists is False
    assert corpus.ledgers[0].sha256 == corpus_mod.hashlib.sha256(b"").hexdigest()


def test_gz_and_directory_inputs_are_refused(isolated, tmp_path):
    with pytest.raises(corpus_mod.CorpusConfigError):
        corpus_mod.ingest_turn_corpus(tmp_path / "ledger.jsonl.gz")
    directory = tmp_path / "a-directory.jsonl"
    directory.mkdir()
    with pytest.raises(corpus_mod.CorpusConfigError):
        corpus_mod.ingest_turn_corpus(directory)


def test_fingerprint_is_stable_and_path_aware(isolated):
    fx.write_tiny_corpora(isolated)
    first = corpus_mod.ingest_turn_corpus().fingerprint()
    second = corpus_mod.ingest_turn_corpus().fingerprint()
    assert first == second
    assert first["accepted"] == 40
    assert first["counts"]["accepted"] == 40


def test_max_rows_cap(isolated, monkeypatch):
    monkeypatch.setattr(corpus_mod, "MAX_ROWS", 3)
    path = isolated / "outcomes" / "outcomes.jsonl"
    fx.write_jsonl(path, [fx.outcome_row(f"e{i}", "a", True, 0.1, fx.EPOCH + i) for i in range(5)])
    with pytest.raises(corpus_mod.CorpusConfigError):
        corpus_mod.ingest_turn_corpus()
