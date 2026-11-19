"""crew_size: how many workers a review crew should have (arms 1-5).

Outcome evidence comes from the review ledger, not from a new signal: for each
recorded review the *exposure* is how many distinct worker models actually ran,
the *reward* is whether the arbiter delivered a usable decision that survived a
later review of the same subject, and the *cost* is the review's total
recorded spend.  ``tools/crew_select.py --learn-size`` consumes the draw.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_int_label


class CrewSize(Domain):
    name = "crew_size"
    arms = ("1", "2", "3", "4", "5")
    default_arm = "3"
    description = "Number of review workers in a Thompson-drawn crew"
    instrumentation = False
    outcome_source = "review ledger: workers that ran, arbiter verdict survival, total cost"
    value_kind = "int"

    @classmethod
    def normalize(cls, value: Any) -> str:
        return coerce_int_label(value, cls.arms, what="crew size")

    @classmethod
    def value(cls, arm: str) -> int:
        return int(arm)

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        return {"workers": int(arm)}


def _overturned(review_ids: List[str], groups: Mapping[str, List[Dict[str, Any]]]) -> bool:
    """Whether a later review of the same subject changed the decision."""
    if len(review_ids) < 2:
        return False
    first = groups[review_ids[0]]
    first_row = _ledger.final_arbiter_row(first)
    if first_row is None:
        return False
    first_verdict = _ledger.verdict_of(first_row.get("text") or "")
    first_selected = _ledger.selected_of(first_row.get("text") or "")
    for later_id in review_ids[1:]:
        later_row = _ledger.final_arbiter_row(groups[later_id])
        if later_row is None:
            continue
        later_verdict = _ledger.verdict_of(later_row.get("text") or "")
        later_selected = _ledger.selected_of(later_row.get("text") or "")
        if first_selected and later_selected and first_selected != later_selected:
            return True
        if first_verdict in _ledger.VERDICTS and later_verdict in _ledger.VERDICTS:
            if first_verdict != later_verdict:
                return True
    return False


def outcome_id_for(review_id: str) -> str:
    """Stable outcome id shared by the live review tool and the ledger backfill."""
    return _ledger.stable_id("crew_size", review_id)


def derive_outcomes(ledger_path=None) -> List[Dict[str, Any]]:
    """One outcome per review, attributed to the crew size that actually ran."""
    path = ledger_path or _ledger.reviews_ledger()
    rows = _ledger.read_jsonl(path)
    groups = _ledger.review_groups(rows)
    by_subject = _ledger.subjects(groups)
    records: List[Dict[str, Any]] = []
    for subject, review_ids in by_subject.items():
        for position, review_id in enumerate(review_ids):
            group = groups[review_id]
            models = _ledger.worker_models(group)
            if not models:
                continue
            arbiter = _ledger.final_arbiter_row(group)
            if arbiter is None:
                # No adjudication: the review's quality was never measured.
                continue
            text = arbiter.get("text") or ""
            verdict = _ledger.verdict_of(text)
            overturned = _overturned(review_ids[position:], groups) if position + 1 < len(review_ids) else False
            if overturned:
                reward = 0.0
            elif verdict in _ledger.VERDICTS:
                reward = 1.0
            elif text.strip():
                reward = 0.5  # adjudicated, but no machine-readable verdict
            else:
                reward = 0.0  # empty or failed arbiter decision
            task_type = next((row.get("task_type") for row in group
                              if isinstance(row.get("task_type"), str)), "code-review")
            records.append({
                "arm": _ledger.nearest_arm(len(models), CrewSize.arms, floor=True),
                "outcome": reward,
                "cost": _ledger.total_cost(group),
                "context": {"task_type": task_type},
                "metadata": {
                    "outcome_id": outcome_id_for(review_id),
                    "review_id": review_id,
                    "subject_sha256": subject,
                    "workers": models,
                    "worker_count": len(models),
                    "verdict": verdict,
                    "overturned": overturned,
                    "later_reviews": review_ids[position + 1:],
                },
            })
    return records


def backfill(learner: Any, *, ledger_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, CrewSize.name, derive_outcomes(ledger_path), limit=limit)


__all__ = ["CrewSize", "backfill", "derive_outcomes"]
