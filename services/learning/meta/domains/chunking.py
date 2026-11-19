"""chunking: chunk-profile identity per ingest (signal capture).

The chunking service declares every profile as ``name@version`` (see
``services/chunking/chunking_service.py`` and the ``profiles`` map the
ingestion service passes downstream).  Bulk-ingest ledger rows now record the
chunking profile that produced them, so chunks can be compared by profile.

Reward: 1.0 for a completed ingest that stored chunks, 0.5 for a refused
duplicate, 0.0 for a failure.  Cost is the chunk count -- downstream embedding
and indexing work -- not money.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, coerce_label

# Mirrors PROFILES in the chunking service contract.  Kept explicit so a new
# profile is a deliberate domain change, not an import-time surprise.
PROFILE_ARMS = ("window-v1", "markdown-v1", "message-v1")


def _profile_name(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().split("@", 1)[0]


class Chunking(Domain):
    name = "chunking"
    arms = PROFILE_ARMS
    default_arm = "window-v1"
    description = "Chunking profile identity per ingest (instrumented, not yet tuned)"
    instrumentation = True
    outcome_source = "bulk-ingest ledger rows carrying chunk_profile and ingest outcome"
    value_kind = "label"

    @classmethod
    def normalize(cls, value: Any) -> str:
        name = _profile_name(value)
        if name is None:
            raise ValueError(f"chunk profile must be one of {cls.arms}, got {value!r}")
        return coerce_label(name, cls.arms, what="chunk profile")


def derive_outcomes(bulk_dir=None) -> List[Dict[str, Any]]:
    root = Path(bulk_dir) if bulk_dir is not None else _ledger.bulk_ingest_dir()
    records: List[Dict[str, Any]] = []
    if not root.is_dir():
        return records
    for path in sorted(root.glob("*.jsonl")):
        for index, row in enumerate(_ledger.read_jsonl(path)):
            response = row.get("response") if isinstance(row.get("response"), dict) else {}
            profile = (row.get("chunk_profile")
                       or _profile_name((response.get("profiles") or {}).get("chunking")))
            try:
                arm = Chunking.normalize(profile)
            except Exception:  # noqa: BLE001 - rows written before instrumentation
                continue
            status = row.get("status")
            ingest_status = row.get("ingest_status")
            chunks = row.get("chunks")
            if status == "succeeded" and isinstance(chunks, (int, float)) and chunks > 0:
                reward = 1.0
            elif status == "succeeded" and ingest_status == "rejected":
                reward = 0.5
            elif status in ("failed", "skipped"):
                reward = 0.0
            else:
                continue
            records.append({
                "arm": arm,
                "outcome": reward,
                "cost": float(chunks or 0),
                "context": {"source": str(row.get("source") or "unknown")},
                "metadata": {
                    "outcome_id": _ledger.stable_id("chunking", path.name, index,
                                                    row.get("content_sha256")),
                    "source": row.get("source"),
                    "ingest_status": ingest_status,
                    "chunks": chunks,
                    "characters": row.get("characters"),
                    "chunk_profile": profile,
                },
            })
    return records


def backfill(learner: Any, *, bulk_dir=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, Chunking.name, derive_outcomes(bulk_dir), limit=limit)


__all__ = ["Chunking", "PROFILE_ARMS", "backfill", "derive_outcomes"]
