"""rag_scope: prompt-time retrieval knob combinations (GA Target B's adapter).

Arms are the cross-product the RAG hook understands:

* ``memory.max_results`` in {5, 10}
* ``knowledge.max_queries`` in {1, 2, 3}
* ``graph.enabled`` in {off, on}

Outcome evidence is the injection journal PR #69 added
(``~/.FlossWare/ledger/rag/injections.jsonl``, override with
``FLOSSWARE_CE_RAG_JOURNAL``).  Each row records the config that produced the
injected context and whether each retrieval source returned anything.  The
reward here is *retrieval yield* -- the share of enabled sources that returned
at least one result -- not usefulness.  There is still no explicit usefulness
rating, so this domain can inform a choice but the GA target must stay
non-promotable until one exists (see :func:`readiness`).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from learning.meta.domains import _ledger
from learning.meta.domains._backfill import apply
from learning.meta.domains.base import Domain, DomainError, coerce_label

MEMORY_MAX_RESULTS = (5, 10)
KNOWLEDGE_MAX_QUERIES = (1, 2, 3)
GRAPH_ENABLED = (False, True)

#: A future GA fitness pass should not treat a handful of rows as evidence.
MIN_GA_OBSERVATIONS = 200


def _label(max_results: int, max_queries: int, graph: bool) -> str:
    return f"m{max_results}-k{max_queries}-g{1 if graph else 0}"


ARMS: Tuple[str, ...] = tuple(
    _label(memory, queries, graph)
    for memory in MEMORY_MAX_RESULTS
    for queries in KNOWLEDGE_MAX_QUERIES
    for graph in GRAPH_ENABLED
)


def _pick(value: Any, allowed: Iterable[Any], what: str) -> Any:
    if isinstance(value, bool):
        allowed_list = list(allowed)
        if value in allowed_list:
            return value
        raise DomainError(f"{what} must be one of {allowed_list}, got {value!r}")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if value in allowed:
        return value
    raise DomainError(f"{what} must be one of {list(allowed)}, got {value!r}")


class RagScope(Domain):
    name = "rag_scope"
    arms = ARMS
    default_arm = _label(5, 3, False)  # today's documented rag.json defaults
    description = "Prompt-time RAG knobs: memory.max_results, knowledge.max_queries, graph.enabled"
    instrumentation = False
    outcome_source = "RAG injection journal: per-source retrieval yield and context size"
    value_kind = "mapping"

    @classmethod
    def normalize(cls, value: Any) -> str:
        if isinstance(value, str):
            return coerce_label(value, cls.arms, what="RAG scope")
        if not isinstance(value, Mapping):
            raise DomainError(f"RAG scope must be an arm label or a config mapping, got {value!r}")
        memory = value.get("memory") if isinstance(value.get("memory"), Mapping) else {}
        knowledge = value.get("knowledge") if isinstance(value.get("knowledge"), Mapping) else {}
        graph = value.get("graph") if isinstance(value.get("graph"), Mapping) else {}
        max_results = value.get("memory.max_results", value.get("memory_max_results",
                                                                 memory.get("max_results", 5)))
        max_queries = value.get("knowledge.max_queries", value.get("knowledge_max_queries",
                                                                   knowledge.get("max_queries", 3)))
        enabled = value.get("graph.enabled", value.get("graph_enabled",
                                                       graph.get("enabled", False)))
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
        label = _label(
            _pick(int(max_results) if not isinstance(max_results, bool) else max_results,
                  MEMORY_MAX_RESULTS, "memory.max_results"),
            _pick(int(max_queries) if not isinstance(max_queries, bool) else max_queries,
                  KNOWLEDGE_MAX_QUERIES, "knowledge.max_queries"),
            _pick(bool(enabled), GRAPH_ENABLED, "graph.enabled"),
        )
        if label not in cls.arms:
            raise DomainError(f"RAG scope {label!r} is not a defined arm")
        return label

    @classmethod
    def value(cls, arm: str) -> Dict[str, Any]:
        memory, queries, graph = _parse_label(arm)
        return {
            "memory.max_results": memory,
            "knowledge.max_queries": queries,
            "graph.enabled": graph,
        }

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        return cls.value(arm)


def _parse_label(arm: str) -> Tuple[int, int, bool]:
    if not isinstance(arm, str) or not arm.startswith("m") or "-k" not in arm or "-g" not in arm:
        raise DomainError(f"invalid RAG arm label {arm!r}")
    try:
        memory_part, rest = arm[1:].split("-k", 1)
        queries_part, graph_part = rest.split("-g", 1)
        value = (int(memory_part), int(queries_part), graph_part == "1")
    except ValueError as exc:
        raise DomainError(f"invalid RAG arm label {arm!r}") from exc
    if _label(*value) != arm:
        raise DomainError(f"invalid RAG arm label {arm!r}")
    return value


def _yield_from_sections(config: Mapping[str, Any], sections: Mapping[str, Any]) -> Tuple[float, int]:
    """(reward, enabled_source_count) from one journal row."""
    memory_config = config.get("memory") if isinstance(config.get("memory"), Mapping) else {}
    knowledge_config = config.get("knowledge") if isinstance(config.get("knowledge"), Mapping) else {}
    enabled = 0
    yielded = 0
    for name, cfg in (("memory", memory_config), ("knowledge", knowledge_config)):
        if cfg.get("enabled") is False:
            continue
        section = sections.get(name) if isinstance(sections.get(name), Mapping) else {}
        enabled += 1
        results = section.get("results")
        if isinstance(results, (int, float)) and not isinstance(results, bool) and results > 0:
            yielded += 1
        elif section.get("status") == "ok":
            yielded += 1
    if enabled == 0:
        return 0.0, 0
    return round(yielded / enabled, 6), enabled


def derive_outcomes(journal_path=None) -> List[Dict[str, Any]]:
    path = journal_path if journal_path is not None else _ledger.rag_journal()
    if path is None:
        return []
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(_ledger.read_jsonl(path)):
        if row.get("kind") != "rag_injection":
            continue
        config = row.get("config") if isinstance(row.get("config"), Mapping) else {}
        memory = config.get("memory") if isinstance(config.get("memory"), Mapping) else {}
        knowledge = config.get("knowledge") if isinstance(config.get("knowledge"), Mapping) else {}
        graph = config.get("graph") if isinstance(config.get("graph"), Mapping) else {}
        try:
            arm = RagScope.normalize({
                "memory.max_results": memory.get("max_results", 5),
                "knowledge.max_queries": knowledge.get("max_queries", 3),
                "graph.enabled": graph.get("enabled", False),
            })
        except DomainError:
            continue
        sections = row.get("sections") if isinstance(row.get("sections"), Mapping) else {}
        reward, enabled = _yield_from_sections(config, sections)
        if enabled == 0:
            # Every retrieval source was disabled: the row says nothing about
            # the knobs this arm controls, so it is not an outcome.
            continue
        context_chars = row.get("context_chars")
        cost = (float(context_chars) / 1000.0
                if isinstance(context_chars, (int, float)) and not isinstance(context_chars, bool)
                else 0.0)
        records.append({
            "arm": arm,
            "outcome": reward,
            "cost": round(cost, 6),
            "context": {"producer": str(row.get("producer") or "rag_hook")},
            "metadata": {
                "outcome_id": row.get("event_id") or _ledger.stable_id("rag_scope", index,
                                                                       row.get("timestamp")),
                "config_sha256": config.get("sha256"),
                "enabled_sources": enabled,
                "memory_results": (sections.get("memory") or {}).get("results"),
                "knowledge_results": (sections.get("knowledge") or {}).get("results"),
                "context_chars": context_chars,
                "truncated": row.get("truncated"),
                "source_count": len(row.get("source_ids") or []),
            },
        })
    return records


def readiness(journal_path=None, minimum: int = MIN_GA_OBSERVATIONS) -> Dict[str, Any]:
    """Whether the journal holds enough observations for GA Target B fitness."""
    path = journal_path if journal_path is not None else _ledger.rag_journal()
    records = derive_outcomes(path)
    observed = sorted({record["arm"] for record in records})
    return {
        "domain": RagScope.name,
        "journal": str(path) if path is not None else None,
        "observations": len(records),
        "minimum": int(minimum),
        "ready": len(records) >= int(minimum),
        "arms_observed": observed,
        "arms_defined": len(RagScope.arms),
        "note": "retrieval yield is a proxy; an explicit usefulness rating is still required for promotion",
    }


def backfill(learner: Any, *, journal_path=None, limit: Optional[int] = None) -> Dict[str, Any]:
    return apply(learner, RagScope.name, derive_outcomes(journal_path), limit=limit)


__all__ = [
    "MIN_GA_OBSERVATIONS",
    "RagScope",
    "backfill",
    "derive_outcomes",
    "readiness",
]
