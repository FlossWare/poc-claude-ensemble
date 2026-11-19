"""prompt_strategy: prompt variant used for review workers (signal capture).

Arms are prompt variants implemented by ``tools/multi_model_review.py``.  Every
worker ledger row now carries a ``prompt_variant`` field, so an A/B comparison
is possible without changing what is recorded today.  This domain is
instrumentation-only: nothing draws from it automatically yet.

Reward per worker row: 1.0 when the arbiter SELECTED that worker, 0.5 when the
worker produced usable text, 0.0 when the call failed or returned nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_label


class PromptStrategy(Domain):
    name = "prompt_strategy"
    arms = ("defects-v1", "risk-first-v1", "terse-v1")
    default_arm = "defects-v1"
    description = "Review worker prompt variant (A/B instrumented, not yet tuned)"
    instrumentation = True
    outcome_source = "review ledger: worker rows labelled with prompt_variant and arbiter selection"
    value_kind = "label"

    @classmethod
    def normalize(cls, value: Any) -> str:
        return coerce_label(value, cls.arms, what="prompt variant")


def _matches(selected: Optional[str], model: str) -> bool:
    if not selected:
        return False
    label = selected.strip().lower()
    model_lower = (model or "").lower()
    mapping = {
        "glm 5.3": ("glm",),
        "deepseek v4.1 flash": ("deepseek",),
        "muse spark open": ("muse",),
    }
    if label in mapping:
        return any(key in model_lower for key in mapping[label])
    return label == model_lower


def derive_outcomes(ledger_path=None) -> List[Dict[str, Any]]:
    path = ledger_path or _ledger.reviews_ledger()
    groups = _ledger.review_groups(_ledger.read_jsonl(path))
    records: List[Dict[str, Any]] = []
    for review_id, group in groups.items():
        arbiter = _ledger.final_arbiter_row(group) or {}
        selected = _ledger.selected_of(arbiter.get("text") or "")
        for index, row in enumerate(group):
            if row.get("kind") != "worker_result":
                continue
            model = row.get("model") or ""
            variant = row.get("prompt_variant") or PromptStrategy.default_arm
            try:
                arm = PromptStrategy.normalize(variant)
            except Exception:  # noqa: BLE001 - unknown label means other tooling wrote it
                continue
            usable = bool((row.get("text") or "").strip()) and not row.get("error")
            if usable and _matches(selected, model):
                reward = 1.0
            elif usable:
                reward = 0.5
            else:
                reward = 0.0
            task_type = row.get("task_type") or "code-review"
            records.append({
                "arm": arm,
                "outcome": reward,
                "cost": float(row.get("cost_usd") or 0.0),
                "context": {"task_type": task_type},
                "metadata": {
                    "outcome_id": _ledger.stable_id("prompt_strategy", review_id, index, model),
                    "review_id": review_id,
                    "model": model,
                    "prompt_variant": variant,
                    "arbiter_selected": _matches(selected, model),
                },
            })
    return records


def backfill(learner: Any, *, ledger_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, PromptStrategy.name, derive_outcomes(ledger_path), limit=limit)


__all__ = ["PromptStrategy", "backfill", "derive_outcomes"]
