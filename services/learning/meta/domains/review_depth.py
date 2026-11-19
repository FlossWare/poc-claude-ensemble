"""review_depth: how many review phases a subject gets (arms 1, 2, 3).

The outcome is *survival*: a review whose verdict (or arbiter-selected worker)
changed in a later review of the same subject was too shallow for its subject.
Reviews with no later review yet are pending and derived as nothing.

Depth is recorded on new ledger rows (``review_depth``); historical rows are
inferred, where worker-only is depth 1 and worker+arbiter is depth 2 -- the
behaviour every pre-existing review had.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_int_label


class ReviewDepth(Domain):
    name = "review_depth"
    arms = ("1", "2", "3")
    default_arm = "2"
    description = "Review phases: workers (1), + arbiter (2), + verification pass (3)"
    instrumentation = False
    outcome_source = "review ledger: later reviews of the same subject that changed the verdict"
    value_kind = "int"

    @classmethod
    def normalize(cls, value: Any) -> str:
        return coerce_int_label(value, cls.arms, what="review depth")

    @classmethod
    def value(cls, arm: str) -> int:
        return int(arm)

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        phases = ["workers", "arbiter", "arbiter-verification"][: int(arm)]
        return {"phases": int(arm), "steps": phases}


def _changed(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    """Whether a later arbiter pass materially changed this review's decision."""
    earlier_text = earlier.get("text") or ""
    later_text = later.get("text") or ""
    earlier_selected = _ledger.selected_of(earlier_text)
    later_selected = _ledger.selected_of(later_text)
    if earlier_selected and later_selected and earlier_selected != later_selected:
        return True
    earlier_verdict = _ledger.verdict_of(earlier_text)
    later_verdict = _ledger.verdict_of(later_text)
    if earlier_verdict in _ledger.VERDICTS and later_verdict in _ledger.VERDICTS:
        return earlier_verdict != later_verdict
    return False


def derive_outcomes(ledger_path=None) -> List[Dict[str, Any]]:
    path = ledger_path or _ledger.reviews_ledger()
    groups = _ledger.review_groups(_ledger.read_jsonl(path))
    by_subject = _ledger.subjects(groups)
    records: List[Dict[str, Any]] = []
    for subject, review_ids in by_subject.items():
        for position, review_id in enumerate(review_ids):
            if position + 1 >= len(review_ids):
                continue  # no later review yet: pending, not an outcome
            group = groups[review_id]
            depth = _ledger.review_depth(group)
            arbiter = _ledger.final_arbiter_row(group)
            later_ids = review_ids[position + 1:]
            task_type = next((row.get("task_type") for row in group
                              if isinstance(row.get("task_type"), str)), "code-review")
            if arbiter is None:
                # Depth 1 has no adjudicated decision to survive.  A later
                # review of the same subject is the only signal available, and
                # it means the shallow pass had to be redone.
                if depth == 1 and later_ids:
                    records.append({
                        "arm": "1",
                        "outcome": 0.0,
                        "cost": _ledger.total_cost(group),
                        "context": {"task_type": task_type},
                        "metadata": {
                            "outcome_id": _ledger.stable_id("review_depth", review_id),
                            "review_id": review_id,
                            "subject_sha256": subject,
                            "depth": 1,
                            "verdict": "none",
                            "overturned": True,
                            "reason": "depth-1 review was re-reviewed",
                            "later_reviews": later_ids,
                        },
                    })
                continue
            overturned = any(
                _changed(arbiter, _ledger.final_arbiter_row(groups[later_id]) or {})
                for later_id in later_ids
                if _ledger.final_arbiter_row(groups[later_id]) is not None
            )
            # A review whose arbiter produced nothing at all is a failure; a
            # review that survived every later pass is a success.
            completed = bool((arbiter.get("text") or "").strip()) and not arbiter.get("error")
            reward = 0.0 if overturned or not completed else 1.0
            records.append({
                "arm": str(depth),
                "outcome": reward,
                "cost": _ledger.total_cost(group),
                "context": {"task_type": task_type},
                "metadata": {
                    "outcome_id": _ledger.stable_id("review_depth", review_id),
                    "review_id": review_id,
                    "subject_sha256": subject,
                    "depth": depth,
                    "verdict": _ledger.verdict_of(arbiter.get("text") or ""),
                    "overturned": overturned,
                    "later_reviews": later_ids,
                },
            })
    return records


def backfill(learner: Any, *, ledger_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, ReviewDepth.name, derive_outcomes(ledger_path), limit=limit)


__all__ = ["ReviewDepth", "backfill", "derive_outcomes"]
