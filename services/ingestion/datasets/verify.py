"""PR 5: prove that ingested content is retrievable, from the live services.

A manifest that says ``indexed`` is a claim.  This module checks the claim
against the services that would actually answer a user's query:

* the **lexical** leg queries the Lexical service's knowledge index directly;
* the **vector** leg embeds the query with the Embedding service and queries the
  Vector collection directly;
* the **hybrid** leg asks the hybrid service, which fuses both legs with RRF;
* the **graph** leg traverses the projected relationships and reports the
  related datasets a category query returns.

Every query is derived from a record the integration manifest says is indexed,
so a "match" means the exact source the pipeline stored came back.  A leg that
cannot run (for example a broken ChromaDB) is reported as ``unavailable`` with
its reason rather than being counted as a pass or a failure: the task requires
vector verification only when the vector store is functional, and requires that
the report say so explicitly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import graph, indexing, model
from .config import Config

VERIFY_VERSION = 1

LEXICAL = "lexical"
VECTOR = "vector"
HYBRID = "hybrid"
LEGS = (LEXICAL, VECTOR, HYBRID)

MAX_HITS_REPORTED = 5
MAX_TEXT_CHARS = 300
#: An explicit ``--category`` list is honoured up to this many traversals.
MAX_EXPLICIT_GRAPH_QUERIES = 20
#: A read-only verification leg is retried once when a dependency is
#: transiently unavailable; a genuine miss is never retried.
VERIFY_ATTEMPTS = 2
VERIFY_RETRY_DELAY_SECONDS = 0.25
#: Related organizations/formats/licenses are counted over at most this many
#: datasets per category (each needs its own traversal); the counts are a
#: sample, not a census.
RELATED_SAMPLE_DATASETS = 10


@dataclass(frozen=True)
class QuerySpec:
    """One verification query and the sources that should answer it."""

    name: str
    text: str
    entry_id: str
    expected_sources: tuple[str, ...] = ()

    def matches(self, hit: dict[str, object]) -> bool:
        url = hit.get("url") or hit.get("source_id")
        if not isinstance(url, str) or not url:
            return False
        if url in self.expected_sources:
            return True
        return bool(self.entry_id) and (
            url == self.entry_id or url.startswith(self.entry_id + "/")
        )


@dataclass
class LegResult:
    """What one retrieval leg returned for one query."""

    leg: str
    status: str = "ok"
    reason_code: str | None = None
    matched: bool = False
    #: False for an ad-hoc ``--query`` with no known source to match against.
    checked: bool = True
    hits: list[dict[str, object]] = field(default_factory=list)
    latency_ms: int = 0
    detail: str | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "leg": self.leg,
            "status": self.status,
            "reason_code": self.reason_code,
            "matched": self.matched,
            "checked": self.checked,
            "hits": len(self.hits),
            "top": self.hits[:MAX_HITS_REPORTED],
            "latency_ms": self.latency_ms,
            "detail": self.detail[:300] if self.detail else None,
        }


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def _title_query(title: str) -> str:
    """The title with its parenthetical aside removed: a real user's query."""
    text = str(title or "").strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text


def build_queries(
    integration_rows: Sequence[dict[str, object]],
    *,
    limit: int,
    categories: Iterable[str] | None = None,
) -> list[QuerySpec]:
    """Derive queries from entries the integration manifest says are indexed."""
    wanted_categories = {value for value in (categories or ()) if value}
    queries: list[QuerySpec] = []
    for row in sorted(integration_rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        if not entry_id:
            continue
        if wanted_categories and str(row.get("category") or "") not in wanted_categories:
            continue
        integration = row.get("integration")
        if not isinstance(integration, dict):
            continue
        if integration.get("status") not in {"indexed", "metadata_indexed", "unchanged"}:
            continue
        sources: list[str] = []
        records: list[dict[str, object]] = []
        metadata = integration.get("metadata")
        if isinstance(metadata, dict):
            records.append(metadata)
        content = integration.get("content")
        if isinstance(content, dict) and isinstance(content.get("records"), list):
            records.extend(item for item in content["records"] if isinstance(item, dict))
        for record in records:
            if record.get("status") not in {"indexed", "unchanged"}:
                continue
            source_id = record.get("source_id")
            if isinstance(source_id, str) and source_id:
                sources.append(source_id)
        if not sources:
            continue
        text = _title_query(str(row.get("title") or ""))
        if not text:
            continue
        queries.append(
            QuerySpec(
                name=f"q{len(queries) + 1}",
                text=text,
                entry_id=entry_id,
                expected_sources=tuple(sorted(set(sources))),
            )
        )
        if len(queries) >= limit:
            break
    return queries


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class VerificationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


def _service_error(
    body: dict[str, Any], *, fallback_code: str, fallback_message: str
) -> VerificationError:
    """Turn a service's error envelope into a reason code and a real message.

    The hybrid service reports a degraded dependency as
    ``{"ok": false, "error": {"code", "message"}}``.  Surface it verbatim:
    "embedding: connection refused" is the useful fact, not "the request was
    rejected".
    """
    error = body.get("error")
    code = fallback_code
    message = fallback_message
    if isinstance(error, dict):
        if isinstance(error.get("code"), str) and error["code"]:
            code = error["code"]
        if isinstance(error.get("message"), str) and error["message"]:
            message = error["message"]
    elif isinstance(error, str) and error:
        message = error
    return VerificationError(code, message[:300])


def _call_json(
    base: str,
    path: str,
    payload: dict[str, Any] | None,
    *,
    method: str = "POST",
    timeout: float,
    post: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if post is not None and payload is not None:
        return post(base, path, payload, timeout)
    try:
        status, body = indexing.request_json(
            base.rstrip("/") + path, method=method, payload=payload, timeout=timeout
        )
    except indexing.IndexingError as exc:
        raise VerificationError(exc.code, str(exc)) from exc
    if status >= 400:
        raise _service_error(
            body,
            fallback_code="dependency_unavailable" if status >= 500 else "request_rejected",
            fallback_message=f"{base} returned HTTP {status}",
        )
    if body.get("ok") is not True:
        raise _service_error(
            body,
            fallback_code="invalid_downstream_response",
            fallback_message=f"{base} rejected the request",
        )
    return body


# ---------------------------------------------------------------------------
# Individual legs
# ---------------------------------------------------------------------------


def _normalize_lexical(row: dict[str, object]) -> dict[str, object]:
    return {
        "url": row.get("url") or row.get("source"),
        "title": row.get("title"),
        "chunk_id": row.get("chunk_id"),
        "content_sha256": row.get("content_sha256"),
        "score": row.get("score"),
        "text": str(row.get("text") or "")[:MAX_TEXT_CHARS],
    }


def _normalize_vector(row: dict[str, object]) -> dict[str, object]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    assert isinstance(metadata, dict)
    return {
        "url": metadata.get("url") or metadata.get("source"),
        "title": metadata.get("title"),
        "chunk_id": metadata.get("chunk_id"),
        "content_sha256": metadata.get("content_sha256"),
        "score": row.get("score"),
        "text": str(row.get("document") or "")[:MAX_TEXT_CHARS],
    }


def lexical_leg(
    spec: QuerySpec, config: Config, *, top_k: int, post=None
) -> LegResult:
    started = time.monotonic()
    try:
        body = _call_json(
            config.lexical_url,
            "/knowledge/search",
            {"query": spec.text, "top_k": top_k},
            timeout=config.verify_timeout,
            post=post,
        )
    except VerificationError as exc:
        return LegResult(
            leg=LEXICAL,
            status="unavailable",
            reason_code=exc.code,
            checked=bool(spec.expected_sources),
            latency_ms=int((time.monotonic() - started) * 1000),
            detail=exc.detail,
        )
    rows = body.get("results")
    hits = [
        _normalize_lexical(row) for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
    ]
    return LegResult(
        leg=LEXICAL,
        matched=any(spec.matches(hit) for hit in hits),
        checked=bool(spec.expected_sources),
        hits=hits,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def vector_leg(spec: QuerySpec, config: Config, *, top_k: int, post=None) -> LegResult:
    started = time.monotonic()
    try:
        embedded = _call_json(
            config.embedding_url,
            "/embeddings",
            {"texts": [spec.text], "mode": "query"},
            timeout=config.verify_timeout,
            post=post,
        )
        vectors = embedded.get("embeddings")
        if not isinstance(vectors, list) or not vectors:
            raise VerificationError(
                "invalid_downstream_response", "embedding service returned no vector"
            )
        body = _call_json(
            config.vector_url,
            "/collections/knowledge_chunks/query",
            {"embedding": vectors[0], "top_k": top_k},
            timeout=config.verify_timeout,
            post=post,
        )
    except VerificationError as exc:
        return LegResult(
            leg=VECTOR,
            status="unavailable",
            reason_code=exc.code,
            checked=bool(spec.expected_sources),
            latency_ms=int((time.monotonic() - started) * 1000),
            detail=exc.detail,
        )
    rows = body.get("results")
    hits = [
        _normalize_vector(row) for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
    ]
    return LegResult(
        leg=VECTOR,
        matched=any(spec.matches(hit) for hit in hits),
        checked=bool(spec.expected_sources),
        hits=hits,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def hybrid_leg(spec: QuerySpec, config: Config, *, top_k: int, post=None) -> LegResult:
    started = time.monotonic()
    try:
        body = _call_json(
            config.hybrid_url,
            "/knowledge/search",
            {"query": spec.text, "top_k": top_k},
            timeout=config.verify_timeout,
            post=post,
        )
    except VerificationError as exc:
        return LegResult(
            leg=HYBRID,
            status="unavailable",
            reason_code=exc.code,
            checked=bool(spec.expected_sources),
            latency_ms=int((time.monotonic() - started) * 1000),
            detail=exc.detail,
        )
    rows = body.get("results")
    hits = [
        _normalize_lexical(row) for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
    ]
    degraded = body.get("degraded")
    detail = ",".join(str(item) for item in degraded) if isinstance(degraded, list) and degraded else None
    return LegResult(
        leg=HYBRID,
        status="degraded" if detail else "ok",
        reason_code="degraded_leg" if detail else None,
        matched=any(spec.matches(hit) for hit in hits),
        checked=bool(spec.expected_sources),
        hits=hits,
        latency_ms=int((time.monotonic() - started) * 1000),
        detail=detail,
    )


LEG_FUNCTIONS = {LEXICAL: lexical_leg, VECTOR: vector_leg, HYBRID: hybrid_leg}


# ---------------------------------------------------------------------------
# Vector store health
# ---------------------------------------------------------------------------


def vector_store_health(config: Config) -> dict[str, object]:
    """Is the vector store functional at all?  Reported, never assumed."""
    payload: dict[str, object] = {
        "functional": False,
        "backend": None,
        "collection": "knowledge_chunks",
        "count": 0,
        "reason_code": None,
    }
    try:
        health = _call_json(
            config.vector_url, "/health", None, method="GET", timeout=config.verify_timeout
        )
        payload["backend"] = health.get("backend")
        collections = _call_json(
            config.vector_url, "/collections", None, method="GET", timeout=config.verify_timeout
        )
    except VerificationError as exc:
        payload["reason_code"] = exc.code
        payload["detail"] = exc.detail
        return payload
    entries = collections.get("collections")
    count = 0
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == "knowledge_chunks":
                value = entry.get("count")
                if isinstance(value, int) and not isinstance(value, bool):
                    count = value
    payload["count"] = count
    payload["functional"] = count > 0
    payload["reason_code"] = None if count > 0 else "empty_collection"
    return payload


# ---------------------------------------------------------------------------
# Graph traversal verification
# ---------------------------------------------------------------------------


def graph_queries(
    rows: Sequence[dict[str, object]],
    *,
    limit: int,
    categories: Iterable[str] | None = None,
) -> list[str]:
    """Category names to traverse, busiest first unless the caller chose.

    Only categories that actually contain integrated entries are candidates:
    traversing a category nothing was projected into proves nothing and would
    report a failure that is really a mismatched query.
    """
    explicit = [value for value in (categories or ()) if value]
    if explicit:
        # An operator who names categories gets them all, up to a bound.
        return explicit[:MAX_EXPLICIT_GRAPH_QUERIES]
    counts: dict[str, int] = {}
    for row in rows:
        integration = row.get("integration")
        if not isinstance(integration, dict):
            continue
        if integration.get("status") not in {"indexed", "metadata_indexed", "unchanged"}:
            continue
        category = row.get("category")
        if isinstance(category, str) and category:
            counts[category] = counts.get(category, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ordered[:limit]]


def verify_graph(
    client: graph.GraphClient,
    categories: Sequence[str],
    *,
    max_depth: int = 2,
) -> list[dict[str, object]]:
    """Traverse from each category and report the related datasets."""
    results: list[dict[str, object]] = []
    for category in categories:
        node_id = graph.category_node_id(category)
        entry: dict[str, object] = {
            "category": category,
            "node_id": node_id,
            "status": "ok",
            "reason_code": None,
            "datasets": 0,
            "related_organizations": 0,
            "related_formats": 0,
            "related_licenses": 0,
            "sample": [],
        }
        try:
            walked = client.traverse(
                node_id, direction="in", max_depth=max_depth, edge_type=graph.EDGE_IN_CATEGORY
            )
        except graph.GraphError as exc:
            reason = "category_node_missing" if exc.code == "graph_rejected" else exc.code
            entry["status"] = "unavailable" if exc.retryable else "missing"
            entry["reason_code"] = reason
            entry["detail"] = exc.detail
            results.append(entry)
            continue
        datasets: list[str] = []
        for item in walked:
            node = item.get("node") if isinstance(item.get("node"), dict) else {}
            assert isinstance(node, dict)
            properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
            assert isinstance(properties, dict)
            if node.get("type") == graph.NODE_DATASET and isinstance(properties.get("entry_id"), str):
                datasets.append(str(properties["entry_id"]))
        # Depth 2 from a category reaches each dataset's versions, and from
        # there the organizations/formats/licenses those versions assert.
        entry["datasets"] = len(datasets)
        entry["sample"] = sorted(set(datasets))[:5]
        if max_depth >= 2:
            related = _related_from_datasets(
                client, sorted(set(datasets))[:RELATED_SAMPLE_DATASETS]
            )
            entry["related_organizations"] = related["organizations"]
            entry["related_formats"] = related["formats"]
            entry["related_licenses"] = related["licenses"]
        if not datasets:
            entry["status"] = "empty"
            entry["reason_code"] = "no_related_datasets"
        results.append(entry)
    return results


def _related_from_datasets(
    client: graph.GraphClient, entry_ids: Sequence[str]
) -> dict[str, int]:
    counts = {"organizations": 0, "formats": 0, "licenses": 0}
    edges = (
        (graph.EDGE_PUBLISHED_BY, "organizations"),
        (graph.EDGE_HAS_FORMAT, "formats"),
        (graph.EDGE_LICENSED_UNDER, "licenses"),
    )
    for entry_id in entry_ids:
        dataset_id = graph.dataset_node_id(entry_id)
        try:
            version_hits = client.traverse(
                dataset_id, direction="out", max_depth=1, edge_type=graph.EDGE_HAS_VERSION
            )
        except graph.GraphError:
            continue
        versions: list[str] = []
        for item in version_hits:
            node = item.get("node")
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                versions.append(node["id"])
        for version_id in versions[:1]:
            for edge_type, key in edges:
                try:
                    hits = client.traverse(
                        version_id, direction="out", max_depth=1, edge_type=edge_type
                    )
                except graph.GraphError:
                    continue
                counts[key] += sum(
                    1
                    for item in hits
                    if isinstance(item.get("node"), dict)
                    and item["node"].get("type")
                    in {graph.NODE_ORGANIZATION, graph.NODE_FORMAT, graph.NODE_LICENSE}
                )
    return counts


# ---------------------------------------------------------------------------
# Whole verification
# ---------------------------------------------------------------------------


def summarize(
    queries: Sequence[dict[str, object]],
    *,
    vector_health: dict[str, object],
    graph_results: Sequence[dict[str, object]],
    min_matches: int = 1,
    retrieval_checked: bool = True,
) -> dict[str, object]:
    """Pass/fail per leg, with the reason a leg could not be judged."""
    legs: dict[str, dict[str, object]] = {}
    for leg in LEGS:
        if not retrieval_checked:
            # Graph-only verification: do not claim a retrieval result nobody
            # asked the services for.
            legs[leg] = {
                "queries": 0,
                "matched": 0,
                "unavailable": 0,
                "informational": 0,
                "verified": False,
                "status": "not_requested",
            }
            continue
        total = 0
        matched = 0
        unavailable = 0
        informational = 0
        for query in queries:
            result = query.get(leg)
            if not isinstance(result, dict):
                continue
            if result.get("checked") is False:
                informational += 1
                continue
            total += 1
            if result.get("status") == "unavailable":
                unavailable += 1
            elif result.get("matched"):
                matched += 1
        verified = matched >= min_matches
        if total == 0 and informational:
            status = "informational"
        else:
            status = (
                "verified"
                if verified
                else ("unavailable" if unavailable == total and total else "failed")
            )
        if leg == VECTOR and not vector_health.get("functional") and not verified:
            status = "unavailable"
        legs[leg] = {
            "queries": total,
            "matched": matched,
            "unavailable": unavailable,
            "informational": informational,
            "verified": verified,
            "status": status,
        }
    graph_ok = all(item.get("status") == "ok" for item in graph_results) if graph_results else None
    problems: list[str] = []
    if retrieval_checked and legs[LEXICAL]["status"] not in {"verified", "informational"}:
        problems.append(f"lexical:{legs[LEXICAL]['status']}")
    if retrieval_checked and legs[VECTOR]["status"] == "failed":
        problems.append("vector:failed")
    if graph_ok is False:
        problems.append("graph:traversal_failed")
    return {
        "legs": legs,
        "graph": {
            "queries": len(graph_results),
            "verified": graph_ok,
            "datasets": sum(int(item.get("datasets") or 0) for item in graph_results),
        },
        "ok": not problems,
        "problems": problems,
        "min_matches": min_matches,
    }


def verification_report(
    *,
    queries: Sequence[QuerySpec],
    results: Sequence[dict[str, object]],
    vector_health: dict[str, object],
    graph_results: Sequence[dict[str, object]],
    run_id: str,
    generated_at: str,
    duration_ms: int,
    top_k: int,
    exit_code: int,
    retrieval_checked: bool = True,
) -> dict[str, object]:
    summary = summarize(
        results,
        vector_health=vector_health,
        graph_results=graph_results,
        retrieval_checked=retrieval_checked,
    )
    return {
        "schema_version": 1,
        "report_type": "verification",
        "verify_version": VERIFY_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "top_k": top_k,
        "queries": list(results),
        "query_count": len(queries),
        "vector_store": vector_health,
        "graph": list(graph_results),
        "summary": summary,
    }


def render_markdown(report: dict[str, object]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    assert isinstance(summary, dict)
    legs = summary.get("legs") if isinstance(summary.get("legs"), dict) else {}
    assert isinstance(legs, dict)
    graph_summary = summary.get("graph") if isinstance(summary.get("graph"), dict) else {}
    vector_store = report.get("vector_store") if isinstance(report.get("vector_store"), dict) else {}
    assert isinstance(graph_summary, dict)
    assert isinstance(vector_store, dict)
    lines = [
        "# Dataset retrieval verification report",
        "",
        f"- Run: `{report.get('run_id')}`",
        f"- Queries: {report.get('query_count')}",
        f"- Top-k: {report.get('top_k')}",
        f"- Result: {'PASS' if summary.get('ok') else 'FAIL'}",
        "",
        "## Retrieval legs",
        "",
        "| Leg | Queries | Matched | Unavailable | Status |",
        "|---|---|---|---|---|",
    ]
    for leg in LEGS:
        value = legs.get(leg) if isinstance(legs.get(leg), dict) else {}
        assert isinstance(value, dict)
        lines.append(
            f"| {leg} | {value.get('queries', 0)} | {value.get('matched', 0)} | "
            f"{value.get('unavailable', 0)} | {value.get('status', 'n/a')} |"
        )
    lines += [
        "",
        "## Vector store",
        "",
        f"- Functional: {bool(vector_store.get('functional'))}",
        f"- Backend: {vector_store.get('backend')}",
        f"- Collection: {vector_store.get('collection')} ({vector_store.get('count')} vectors)",
        f"- Reason: {vector_store.get('reason_code') or '-'}",
        "",
        "## Graph traversal",
        "",
        f"- Category queries: {graph_summary.get('queries', 0)}",
        f"- Related datasets: {graph_summary.get('datasets', 0)}",
        f"- Verified: {graph_summary.get('verified')}",
        "",
        "| Category | Datasets | Organizations* | Formats* | Licenses* | Status |",
        "|---|---|---|---|---|---|",
    ]
    for item in report.get("graph") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('category')} | {item.get('datasets')} | "
            f"{item.get('related_organizations')} | {item.get('related_formats')} | "
            f"{item.get('related_licenses')} | {item.get('status')} |"
        )
    lines += [
        "",
        "\\* sampled over at most "
        f"{RELATED_SAMPLE_DATASETS} datasets in the category.",
        "",
        "## Per-query results",
        "",
    ]
    for query in report.get("queries") or []:
        if not isinstance(query, dict):
            continue
        lines.append(f"### {query.get('name')}: `{query.get('query')}`")
        lines.append("")
        lines.append(f"- Entry: `{query.get('entry_id')}`")
        for leg in LEGS:
            value = query.get(leg) if isinstance(query.get(leg), dict) else {}
            assert isinstance(value, dict)
            top = value.get("top") or []
            first = top[0] if top and isinstance(top[0], dict) else {}
            assert isinstance(first, dict)
            lines.append(
                f"- {leg}: status={value.get('status')} matched={value.get('matched')} "
                f"hits={value.get('hits')} top_url={first.get('url') or '-'} "
                f"score={first.get('score')}"
            )
        lines.append("")
    problems = summary.get("problems") or []
    if problems:
        lines += ["## Problems", ""] + [f"- {problem}" for problem in problems] + [""]
    return "\n".join(lines)


def render_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


RETRYABLE_LEG_REASONS = frozenset(
    {"dependency_unavailable", "service_unavailable", "timeout"}
)


def run_leg(spec: QuerySpec, config: Config, leg: str, *, top_k: int, post=None) -> LegResult:
    """Run one leg, retrying once if a dependency was transiently unavailable."""
    function = LEG_FUNCTIONS[leg]
    result = function(spec, config, top_k=top_k, post=post)
    remaining = VERIFY_ATTEMPTS - 1
    while (
        remaining > 0
        and result.status == "unavailable"
        and result.reason_code in RETRYABLE_LEG_REASONS
    ):
        remaining -= 1
        if VERIFY_RETRY_DELAY_SECONDS:
            time.sleep(VERIFY_RETRY_DELAY_SECONDS)
        result = function(spec, config, top_k=top_k, post=post)
    return result


def run_queries(
    specs: Sequence[QuerySpec],
    config: Config,
    *,
    top_k: int,
    post=None,
    legs: Sequence[str] = LEGS,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for spec in specs:
        entry: dict[str, object] = {
            "name": spec.name,
            "query": spec.text,
            "entry_id": spec.entry_id,
            "expected_sources": list(spec.expected_sources[:5]),
            "expected_source_count": len(spec.expected_sources),
        }
        for leg in legs:
            entry[leg] = run_leg(spec, config, leg, top_k=top_k, post=post).to_artifact()
        results.append(entry)
    return results


def verification_run_id(
    commit: str, queries: Sequence[QuerySpec], *, graph_categories: Sequence[str] = ()
) -> str:
    return model.sha256_hex(
        model.canonical_json(
            {
                "schema_version": 1,
                "verify_version": VERIFY_VERSION,
                "catalog_commit": commit,
                "queries": [spec.text for spec in queries],
                "graph_categories": list(graph_categories),
            }
        )
    )[:16]
