"""Record derived outcomes into a domain, idempotently."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional


def apply(
    learner: Any,
    domain: str,
    records: Iterable[Mapping[str, Any]],
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Feed derived ``{arm, outcome, cost, metadata, context}`` rows to a learner.

    Rows without a stable ``outcome_id`` are recorded once per derivation run,
    so every derived row here carries one.  Invalid rows are counted, not
    raised: a single malformed ledger line must not abort a backfill.
    """
    counts: Counter = Counter()
    by_arm: Counter = Counter()
    first_error: Optional[str] = None
    derived = 0
    for record in records:
        if limit is not None and derived >= limit:
            break
        derived += 1
        try:
            result = learner.record(
                domain,
                record["arm"],
                record["outcome"],
                record.get("cost", 0.0),
                dict(record.get("metadata") or {}),
                context=record.get("context"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            counts["invalid"] += 1
            if first_error is None:
                first_error = str(exc)
            continue
        if result.get("recorded"):
            counts["recorded"] += 1
            by_arm[record["arm"]] += 1
        else:
            counts["duplicates"] += 1
    return {
        "domain": domain,
        "derived": derived,
        "recorded": counts["recorded"],
        "duplicates": counts["duplicates"],
        "invalid": counts["invalid"],
        "first_error": first_error,
        "by_arm": dict(sorted(by_arm.items())),
    }
