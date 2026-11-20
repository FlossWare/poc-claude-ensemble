"""PR 5: the catalog's relationships as a Graph source projection.

The graph is a *projection*, not a second system of record.  Everything here is
derived from the PR 1 inventory rows (which are themselves derived from the
pinned catalog commit), and nothing here is consulted to decide what to ingest.
Deleting the graph and re-running ``graph`` rebuilds it exactly.

Model (versioned, because the Graph service never mutates a node):

===============  =====================================================
Node             Identity
===============  =====================================================
``Dataset``      ``dataset:{sha256(entry_id)[:32]}``
``DatasetVersion`` ``dataset_version:{metadata_sha256}``
``Category``     catalog category (path-derived)
``Organization`` organization name (metadata-derived)
``Repository``   ``host/owner/repo`` from a source URL, else the host
``Format``       a classified format
``License``      a normalized license family
===============  =====================================================

Edges always originate at ``Dataset`` (stable facts: category) or
``DatasetVersion`` (metadata facts), and every edge carries the provenance of
the assertion: catalog commit, meta URL and the metadata hash it came from.

Two consequences of the service's immutability rule drive the design:

* Node properties contain only values intrinsic to the identity.  A commit, a
  timestamp or an outcome in a node's properties would make an unchanged
  dataset conflict with itself on the next commit.
* Provenance therefore lives on the **edge**, whose id is derived from
  (source, target, type, properties).  A new commit produces a new edge id and
  a new provenance assertion; the identical commit produces the identical edge
  id, which the service returns as a no-op.  That is what makes projection
  idempotent.

Tombstones are respected twice: a tombstoned entry is never projected (it is
reported ``skipped`` with ``source_tombstoned``), and ``prune`` removes the
dataset's nodes through ``/graph/remove-nodes`` with a dry-run default.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import indexing, manifest, model, resume
from .config import Config

GRAPH_VERSION = 1

NODE_DATASET = "Dataset"
NODE_VERSION = "DatasetVersion"
NODE_CATEGORY = "Category"
NODE_ORGANIZATION = "Organization"
NODE_REPOSITORY = "Repository"
NODE_FORMAT = "Format"
NODE_LICENSE = "License"

EDGE_HAS_VERSION = "has_version"
EDGE_IN_CATEGORY = "in_category"
EDGE_PUBLISHED_BY = "published_by"
EDGE_HOSTED_AT = "hosted_at"
EDGE_HAS_FORMAT = "has_format"
EDGE_LICENSED_UNDER = "licensed_under"

ALL_NODE_TYPES = (
    NODE_DATASET,
    NODE_VERSION,
    NODE_CATEGORY,
    NODE_ORGANIZATION,
    NODE_REPOSITORY,
    NODE_FORMAT,
    NODE_LICENSE,
)
#: Every projected edge starts at one of these two node types.
EDGE_SOURCE_TYPES = (NODE_DATASET, NODE_VERSION)

#: Reasons that justify another attempt without operator action.
RETRYABLE_REASONS = frozenset(
    {
        "graph_unavailable",
        "graph_timeout",
        "graph_failed",
        "invalid_downstream_response",
    }
)

CODEX_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org", "codeberg.org"})

#: Reason codes that mean a graph operation failed (as opposed to a skipped row).
GRAPH_FAILURE_REASONS = frozenset(
    {
        "graph_unavailable",
        "graph_timeout",
        "graph_failed",
        "graph_rejected",
        "graph_conflict",
        "invalid_downstream_response",
    }
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class GraphError(RuntimeError):
    """A graph operation failed with a stable reason code."""

    def __init__(self, code: str, detail: str, *, status: int | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_REASONS


# ---------------------------------------------------------------------------
# Deterministic identity
# ---------------------------------------------------------------------------


def _digest(value: object) -> str:
    return hashlib.sha256(model.canonical_json(value).encode("utf-8")).hexdigest()


def slug(value: str, *, max_length: int = 80) -> str:
    """A readable, bounded, deterministic fragment of a display value."""
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text[:max_length] or "unnamed"


def named_id(prefix: str, value: str) -> str:
    """Identity for a name-derived node: readable, and collision-free.

    The short hash is what makes two different display names that slugify the
    same (``Café`` and ``Cafe``) stay distinct nodes; without it the immutable
    store would reject the second one as a conflicting write.
    """
    return f"{prefix}:{slug(value)}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def dataset_node_id(entry_id: str) -> str:
    return f"dataset:{hashlib.sha256(entry_id.encode('utf-8')).hexdigest()[:32]}"


def category_node_id(name: str) -> str:
    """The one place the Category id prefix is defined."""
    return named_id("category", name)


def organization_node_id(name: str) -> str:
    return named_id("organization", name)


def format_node_id(name: str) -> str:
    return named_id("format", name)


def license_node_id(family: str) -> str:
    return named_id("license", family)


def version_node_id(metadata_sha256: str) -> str:
    return f"dataset_version:{metadata_sha256}"


def edge_id(source: str, target: str, edge_type: str, properties: dict[str, object]) -> str:
    return "edge:" + _digest(
        {
            "source": source,
            "target": target,
            "type": edge_type,
            "properties": properties,
        }
    )[:32]


def repository_for(url: object) -> dict[str, str] | None:
    """Canonical ``host/owner/repo`` for a source URL.

    A code host contributes owner and repo.  Anything else contributes its host
    only, which is still an honest answer to "where is this hosted?".
    """
    sanitized = model.sanitize_url(url)
    if not sanitized:
        return None
    try:
        parsed = urllib.parse.urlsplit(sanitized)
    except ValueError:
        return None
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None
    segments = [part for part in parsed.path.split("/") if part]
    owner = repo = ""
    if host in CODEX_HOSTS and len(segments) >= 2:
        owner, repo = segments[0], segments[1]
    elif host == "huggingface.co" and len(segments) >= 3 and segments[0] == "datasets":
        owner, repo = segments[1], segments[2]
    elif host == "raw.githubusercontent.com" and len(segments) >= 3:
        owner, repo = segments[0], segments[1]
    elif host == "zenodo.org" and len(segments) >= 2 and segments[0] == "records":
        owner, repo = "records", segments[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    owner = slug(owner) if owner else ""
    repo = slug(repo) if repo else ""
    return {"host": host, "owner": owner, "repo": repo}


def repository_node_id(canonical: dict[str, str]) -> str:
    parts = [canonical["host"]]
    if canonical.get("owner"):
        parts.append(canonical["owner"])
    if canonical.get("repo"):
        parts.append(canonical["repo"])
    return "repository:" + "/".join(parts)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """The nodes and edges one catalog entry contributes."""

    entry_id: str
    nodes: tuple[dict[str, object], ...] = ()
    edges: tuple[dict[str, object], ...] = ()
    reason_code: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason_code is None


def _node(node_type: str, node_id: str, properties: dict[str, object]) -> dict[str, object]:
    return {"id": node_id, "type": node_type, "properties": properties}


def _edge(
    source: str,
    target: str,
    edge_type: str,
    properties: dict[str, object],
) -> dict[str, object]:
    return {
        "id": edge_id(source, target, edge_type, properties),
        "source": source,
        "target": target,
        "type": edge_type,
        "properties": properties,
    }


def provenance(row: dict[str, object], commit: str) -> dict[str, object]:
    catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
    assert isinstance(catalog, dict)
    origin_repo = catalog.get("repo") if isinstance(catalog.get("repo"), str) else "apd-core"
    locator = catalog.get("meta_url")
    return {
        "origin": f"{origin_repo}@{commit}",
        "catalog_commit": commit,
        "locator": locator if isinstance(locator, str) else None,
        "entry_id": row.get("entry_id"),
        "metadata_sha256": row.get("metadata_sha256"),
    }


def _license_family(row: dict[str, object]) -> str | None:
    block = row.get("license") if isinstance(row.get("license"), dict) else {}
    assert isinstance(block, dict)
    dataset = block.get("dataset") if isinstance(block.get("dataset"), dict) else {}
    assert isinstance(dataset, dict)
    family = dataset.get("family")
    if isinstance(family, str) and family and family != "unknown":
        return family
    return None


def _formats(row: dict[str, object]) -> list[str]:
    classes = row.get("classes") if isinstance(row.get("classes"), dict) else {}
    assert isinstance(classes, dict)
    values = classes.get("format")
    if not isinstance(values, list):
        return []
    return sorted(
        {str(value) for value in values if isinstance(value, str) and value and value != "unknown"}
    )


def _organizations(row: dict[str, object]) -> list[str]:
    values = row.get("organization")
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for item in values:
        name = item.get("name") if isinstance(item, dict) else None
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names))


def _source_urls(row: dict[str, object]) -> list[tuple[str, str]]:
    """``(url, kind)`` pairs, sources first, the homepage only as a fallback."""
    pairs: list[tuple[str, str]] = []
    sources = row.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            url = model.sanitize_url(source.get("access_url"))
            if url:
                pairs.append((url, "source"))
    if not pairs:
        url = model.sanitize_url(row.get("homepage"))
        if url:
            pairs.append((url, "homepage"))
    return pairs


def project_row(row: dict[str, object], *, commit: str) -> Projection:
    """Build one entry's nodes and edges.  Pure: no I/O, no clock, no config."""
    entry_id = row.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        return Projection(entry_id="", reason_code="entry_invalid", detail="missing entry id")
    metadata_sha256 = row.get("metadata_sha256")
    if not isinstance(metadata_sha256, str) or not metadata_sha256:
        return Projection(
            entry_id=entry_id, reason_code="no_metadata", detail="entry has no metadata hash"
        )
    catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
    assert isinstance(catalog, dict)
    if catalog.get("commit") and catalog.get("commit") != commit:
        return Projection(
            entry_id=entry_id,
            reason_code="catalog_mismatch",
            detail="row belongs to a different catalog commit",
        )

    prov = provenance(row, commit)
    dataset_id = dataset_node_id(entry_id)
    version_id = version_node_id(metadata_sha256)

    title = row.get("title") if isinstance(row.get("title"), str) else None
    category = row.get("category") if isinstance(row.get("category"), str) else None
    family = _license_family(row)

    nodes: list[dict[str, object]] = [
        _node(NODE_DATASET, dataset_id, {"entry_id": entry_id}),
        _node(
            NODE_VERSION,
            version_id,
            {
                "entry_id": entry_id,
                "metadata_sha256": metadata_sha256,
                "title": title,
                "category": category,
                "license_family": family,
            },
        ),
    ]
    edges: list[dict[str, object]] = [
        _edge(dataset_id, version_id, EDGE_HAS_VERSION, dict(prov)),
    ]

    if category:
        category_id = category_node_id(category)
        nodes.append(_node(NODE_CATEGORY, category_id, {"name": category}))
        edges.append(_edge(dataset_id, category_id, EDGE_IN_CATEGORY, dict(prov)))

    for name in _organizations(row):
        organization_id = organization_node_id(name)
        nodes.append(_node(NODE_ORGANIZATION, organization_id, {"name": name}))
        edges.append(_edge(version_id, organization_id, EDGE_PUBLISHED_BY, dict(prov)))

    seen_repositories: dict[str, dict[str, str]] = {}
    for url, kind in _source_urls(row):
        canonical = repository_for(url)
        if canonical is None:
            continue
        repository_id = repository_node_id(canonical)
        seen_repositories.setdefault(repository_id, canonical)
        edge_properties = dict(prov)
        edge_properties["kind"] = kind
        edge_properties["source_url"] = url
        edges.append(_edge(version_id, repository_id, EDGE_HOSTED_AT, edge_properties))
    for repository_id, canonical in seen_repositories.items():
        nodes.append(_node(NODE_REPOSITORY, repository_id, dict(canonical)))

    for name in _formats(row):
        format_id = format_node_id(name)
        nodes.append(_node(NODE_FORMAT, format_id, {"name": name}))
        edges.append(_edge(version_id, format_id, EDGE_HAS_FORMAT, dict(prov)))

    if family:
        license_id = license_node_id(family)
        nodes.append(_node(NODE_LICENSE, license_id, {"family": family}))
        edges.append(_edge(version_id, license_id, EDGE_LICENSED_UNDER, dict(prov)))

    # Deterministic order, and a node id is submitted at most once per entry.
    unique_nodes: dict[str, dict[str, object]] = {}
    for node in nodes:
        unique_nodes.setdefault(str(node["id"]), node)
    unique_edges: dict[str, dict[str, object]] = {}
    for edge in edges:
        unique_edges.setdefault(str(edge["id"]), edge)
    return Projection(
        entry_id=entry_id,
        nodes=tuple(unique_nodes[key] for key in sorted(unique_nodes)),
        edges=tuple(unique_edges[key] for key in sorted(unique_edges)),
    )


def fingerprint(
    row: dict[str, object], *, commit: str, config: Config
) -> str:
    """Inputs that would change what this entry projects."""
    return resume.fingerprint(
        "graph",
        entry_id=str(row.get("entry_id") or ""),
        inputs={
            "catalog_commit": commit,
            "metadata_sha256": row.get("metadata_sha256"),
            "graph_version": GRAPH_VERSION,
            "license_family": _license_family(row),
            "category": row.get("category"),
            "formats": _formats(row),
            "organizations": _organizations(row),
            "sources": [url for url, _kind in _source_urls(row)],
            "allow_families": sorted(config.license_allow_families),
        },
    )


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------


@dataclass
class GraphSnapshot:
    """What the graph already holds, so writes can be counted and skipped."""

    nodes: dict[str, dict[str, object]] = field(default_factory=dict)
    edges: dict[str, dict[str, object]] = field(default_factory=dict)
    mode: str = "blind"
    prefetch_calls: int = 0
    error: str | None = None


class GraphClient:
    """Minimal client for the Graph service's documented REST surface."""

    def __init__(self, base: str, *, timeout: float = 30.0, post=None, get=None) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._post = post or indexing.post_json
        self._get = get

    # -- transport ---------------------------------------------------------
    def _write(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            return self._post(self.base, path, payload, self.timeout)
        except indexing.IndexingError as exc:
            code = "graph_timeout" if exc.code == "dependency_unavailable" else "graph_failed"
            raise GraphError(code, str(exc), status=exc.status) from exc

    def _write_raw(self, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        """A write whose HTTP status matters (400 means immutable conflict)."""
        try:
            return indexing.request_json(
                self.base + path, method="POST", payload=payload, timeout=self.timeout
            )
        except indexing.IndexingError as exc:
            code = "graph_timeout" if exc.code == "dependency_unavailable" else "graph_failed"
            raise GraphError(code, str(exc), status=exc.status) from exc

    def _read(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        status, body = self._write_raw(path, payload)
        if status == 404:
            return {}
        if status >= 400:
            raise GraphError(
                "graph_unavailable" if status >= 500 else "graph_rejected",
                f"graph service returned HTTP {status}",
                status=status,
            )
        if body.get("ok") is not True:
            raise GraphError("invalid_downstream_response", "graph service rejected the request")
        return body

    # -- operations --------------------------------------------------------
    def add_node(self, node: dict[str, object]) -> str:
        status, body = self._write_raw(
            "/graph/add-node",
            {
                "id": node["id"],
                "type": node["type"],
                "properties": node.get("properties") or {},
            },
        )
        if status == 200 and body.get("ok") is True:
            return "present"
        if status == 400 and "already exists with different content" in str(body.get("error", "")):
            raise GraphError(
                "graph_conflict",
                f"node {node['id']} exists with different content",
                status=status,
            )
        raise GraphError(
            "graph_rejected" if 400 <= status < 500 else "graph_failed",
            f"graph service returned HTTP {status}: {str(body.get('error') or '')[:200]}",
            status=status,
        )

    def add_edge(self, edge: dict[str, object]) -> str:
        status, body = self._write_raw(
            "/graph/add-edge",
            {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "properties": edge.get("properties") or {},
            },
        )
        if status == 200 and body.get("ok") is True:
            return "present"
        if status == 400 and "already exists with different content" in str(body.get("error", "")):
            raise GraphError(
                "graph_conflict",
                f"edge {edge['id']} exists with different content",
                status=status,
            )
        raise GraphError(
            "graph_rejected" if 400 <= status < 500 else "graph_failed",
            f"graph service returned HTTP {status}: {str(body.get('error') or '')[:200]}",
            status=status,
        )

    def query_nodes(self, node_type: str, properties: dict[str, object] | None = None) -> list[dict[str, object]]:
        body = self._read("/graph/query", {"type": node_type, "properties": properties or {}})
        nodes = body.get("nodes")
        if not isinstance(nodes, list):
            raise GraphError("invalid_downstream_response", "graph service returned no node list")
        return [node for node in nodes if isinstance(node, dict)]

    def query_edges(self, from_type: str) -> list[dict[str, object]]:
        body = self._read(
            "/graph/query", {"type": "edges", "from_type": from_type}
        )
        edges = body.get("edges")
        if not isinstance(edges, list):
            raise GraphError("invalid_downstream_response", "graph service returned no edge list")
        result: list[dict[str, object]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            result.append(
                {
                    "id": edge.get("id"),
                    "source": edge.get("from_id"),
                    "target": edge.get("to_id"),
                    "type": edge.get("relationship"),
                    "properties": edge.get("properties") or {},
                }
            )
        return result

    def traverse(
        self,
        start: str,
        *,
        direction: str = "out",
        max_depth: int = 1,
        edge_type: str | None = None,
    ) -> list[dict[str, object]]:
        payload: dict[str, object] = {
            "start": start,
            "direction": direction,
            "max_depth": max_depth,
        }
        if edge_type:
            payload["edge_type"] = edge_type
        body = self._read("/graph/traverse", payload)
        results = body.get("results")
        if not isinstance(results, list):
            raise GraphError("invalid_downstream_response", "graph service returned no results")
        return [item for item in results if isinstance(item, dict)]

    def remove_nodes(self, ids: Sequence[str], *, dry_run: bool = True) -> dict[str, object]:
        body = self._read(
            "/graph/remove-nodes", {"ids": list(ids), "dry_run": bool(dry_run)}
        )
        return body

    # -- prefetch ----------------------------------------------------------
    def prefetch(self) -> GraphSnapshot:
        """Read the existing nodes and edges once so writes are counted exactly.

        Bounded: one query per node type plus one per edge-source type.  A
        failure degrades to ``blind`` mode rather than failing the run — the
        deterministic ids still make every write idempotent.
        """
        snapshot = GraphSnapshot(mode="prefetch")
        try:
            for node_type in ALL_NODE_TYPES:
                snapshot.prefetch_calls += 1
                for node in self.query_nodes(node_type):
                    node_id = node.get("id")
                    if isinstance(node_id, str):
                        snapshot.nodes[node_id] = node
            for source_type in EDGE_SOURCE_TYPES:
                snapshot.prefetch_calls += 1
                for edge in self.query_edges(source_type):
                    edge_key = edge.get("id")
                    if isinstance(edge_key, str):
                        snapshot.edges[edge_key] = edge
        except GraphError as exc:
            return GraphSnapshot(mode="blind", prefetch_calls=snapshot.prefetch_calls, error=exc.code)
        return snapshot


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryGraphResult:
    """What happened to one entry's projection."""

    entry_id: str
    status: str
    reason_code: str | None = None
    detail: str | None = None
    retryable: bool = False
    attempts: int = 1
    resumed: bool = False
    nodes_created: int = 0
    nodes_present: int = 0
    edges_created: int = 0
    edges_present: int = 0
    pruned: tuple[str, ...] = ()

    def to_artifact(self) -> dict[str, object]:
        return {
            "graph_version": GRAPH_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail[:300] if self.detail else None,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "resumed": self.resumed,
            "nodes_created": self.nodes_created,
            "nodes_present": self.nodes_present,
            "edges_created": self.edges_created,
            "edges_present": self.edges_present,
            "pruned": list(self.pruned),
        }

    @classmethod
    def from_artifact(
        cls,
        entry_id: str,
        artifact: dict[str, object],
        *,
        resumed: bool = False,
        as_unchanged: bool = False,
    ) -> "EntryGraphResult":
        def _int(key: str) -> int:
            value = artifact.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        reason = artifact.get("reason_code")
        detail = artifact.get("detail")
        status = str(artifact.get("status") or "failed")
        retryable = bool(artifact.get("retryable"))
        if as_unchanged and status in resume.SUCCESS_STATUSES:
            status, reason, retryable = "unchanged", "checkpoint_resume", False
        return cls(
            entry_id=entry_id,
            status=status,
            reason_code=reason if isinstance(reason, str) else None,
            detail=detail if isinstance(detail, str) else None,
            retryable=retryable,
            attempts=_int("attempts") or 1,
            resumed=resumed,
            nodes_created=_int("nodes_created"),
            nodes_present=_int("nodes_present"),
            edges_created=_int("edges_created"),
            edges_present=_int("edges_present"),
        )


@dataclass
class ProjectionContext:
    """Counters shared by every entry in one projection run."""

    snapshot: GraphSnapshot
    nodes_created: int = 0
    nodes_present: int = 0
    edges_created: int = 0
    edges_present: int = 0
    graph_errors: int = 0
    #: Exact ids this run created, so a retried entry cannot inflate the totals.
    created_node_ids: set[str] = field(default_factory=set)
    created_edge_ids: set[str] = field(default_factory=set)


def _record_equal(existing: dict[str, object], planned: dict[str, object]) -> bool:
    """Whether an existing graph record is exactly the one we would assert."""
    return (
        existing.get("id") == planned.get("id")
        and existing.get("type") == planned.get("type")
        and existing.get("properties") == planned.get("properties")
        and existing.get("source", planned.get("source")) == planned.get("source")
        and existing.get("target", planned.get("target")) == planned.get("target")
    )


def _apply(
    projection: Projection,
    context: ProjectionContext,
    client: GraphClient | None,
    *,
    dry_run: bool,
) -> EntryGraphResult:
    """Write one projection nodes-first, counting created versus present."""
    created_nodes = present_nodes = created_edges = present_edges = 0
    try:
        for node in projection.nodes:
            node_id = str(node["id"])
            existing = context.snapshot.nodes.get(node_id)
            if existing is not None:
                # Prefetch is an optimisation, not a licence to trust: an
                # existing record with different content is the same conflict
                # the service would return.
                if not _record_equal(existing, node):
                    raise GraphError(
                        "graph_conflict",
                        f"node {node_id} exists with different content",
                    )
                present_nodes += 1
                continue
            if not dry_run:
                assert client is not None
                client.add_node(node)
            context.snapshot.nodes[node_id] = node
            context.created_node_ids.add(node_id)
            created_nodes += 1
        for edge in projection.edges:
            key = str(edge["id"])
            existing_edge = context.snapshot.edges.get(key)
            if existing_edge is not None:
                if not _record_equal(existing_edge, edge):
                    raise GraphError(
                        "graph_conflict",
                        f"edge {key} exists with different content",
                    )
                present_edges += 1
                continue
            if not dry_run:
                assert client is not None
                client.add_edge(edge)
            context.snapshot.edges[key] = edge
            context.created_edge_ids.add(key)
            created_edges += 1
    except GraphError as exc:
        retryable = exc.retryable
        return EntryGraphResult(
            entry_id=projection.entry_id,
            status="deferred" if retryable else "failed",
            reason_code=exc.code,
            detail=exc.detail,
            retryable=retryable,
            nodes_created=created_nodes,
            nodes_present=present_nodes,
            edges_created=created_edges,
            edges_present=present_edges,
        )
    context.nodes_created += created_nodes
    context.nodes_present += present_nodes
    context.edges_created += created_edges
    context.edges_present += present_edges
    if created_nodes or created_edges:
        status, reason = "indexed", None
    else:
        status, reason = "unchanged", "graph_unchanged"
    return EntryGraphResult(
        entry_id=projection.entry_id,
        status=status,
        reason_code=reason,
        nodes_created=created_nodes,
        nodes_present=present_nodes,
        edges_created=created_edges,
        edges_present=present_edges,
    )


def project_entries(
    rows: Sequence[dict[str, object]],
    *,
    commit: str,
    config: Config,
    client: GraphClient | None,
    selected: Iterable[str],
    journal: resume.CheckpointLog | None = None,
    policy: resume.RetryPolicy | None = None,
    tombstoned: Iterable[str] = (),
    dry_run: bool = False,
    sleep=None,
) -> tuple[list[EntryGraphResult], ProjectionContext]:
    """Project every selected entry, resuming from checkpoints when trusted.

    ``selected`` is the set of entry ids this run may write.  Entries outside
    it are still returned (``deferred``/``not_selected``) so the accounting
    invariant ``discovered == statuses`` keeps holding.
    """
    policy = policy or resume.RetryPolicy()
    tombstoned_set = set(tombstoned)
    context = ProjectionContext(snapshot=GraphSnapshot(mode="blind"))
    # A prefetch also serves ``--dry-run``: it is how the dry run can report
    # exactly which nodes and edges would be created.
    if client is not None:
        context.snapshot = client.prefetch()

    # One result per row (not per entry id): a catalog with duplicate ids still
    # has to satisfy ``discovered == count(statuses)``.
    results: list[EntryGraphResult] = []
    by_entry: dict[str, EntryGraphResult] = {}

    def record(result: EntryGraphResult) -> EntryGraphResult:
        results.append(result)
        by_entry[result.entry_id] = result
        return result

    for row in rows:
        entry_id = str(row.get("entry_id") or "")
        if entry_id not in selected:
            record(
                EntryGraphResult(
                    entry_id=entry_id,
                    status="deferred",
                    reason_code="not_selected",
                    detail="entry was outside the bounded selection for this run",
                    retryable=True,
                )
            )
            continue
        digest = fingerprint(row, commit=commit, config=config)
        if entry_id in tombstoned_set:
            skipped = record(
                EntryGraphResult(
                    entry_id=entry_id,
                    status="skipped",
                    reason_code="source_tombstoned",
                    detail="the entry's Knowledge source is tombstoned",
                )
            )
            if journal is not None:
                journal.record_result(
                    entry_id,
                    digest,
                    status="skipped",
                    reason_code="source_tombstoned",
                    retryable=False,
                    attempts=0,
                    result=skipped.to_artifact(),
                )
            continue
        decision = (
            journal.decide(entry_id, digest, policy=policy)
            if journal is not None
            else resume.ResumeDecision(reuse=False, reason="disabled")
        )
        if decision.exhausted and decision.checkpoint is not None:
            artifact = dict(decision.checkpoint.result or {})
            artifact.update(
                {
                    "status": "deferred",
                    "reason_code": resume.RETRY_EXHAUSTED_REASON,
                    "retryable": False,
                    "attempts": decision.checkpoint.attempts,
                }
            )
            record(EntryGraphResult.from_artifact(entry_id, artifact, resumed=True))
            continue
        if decision.reuse and decision.checkpoint is not None and decision.checkpoint.result:
            # Nothing was written this run, so a recorded success is reported
            # as ``unchanged``; recorded counts stay visible on the entry row.
            record(
                EntryGraphResult.from_artifact(
                    entry_id,
                    decision.checkpoint.result,
                    resumed=True,
                    as_unchanged=True,
                )
            )
            continue

        projection = project_row(row, commit=commit)
        if not projection.ok:
            record(
                EntryGraphResult(
                    entry_id=entry_id,
                    status="skipped",
                    reason_code=projection.reason_code,
                    detail=projection.detail,
                )
            )
        else:
            attempt = resume.with_retries(
                lambda: _apply(projection, context, client, dry_run=dry_run),
                policy=policy,
                retryable=lambda result: result.retryable,
                **({"sleep": sleep} if sleep is not None else {}),
            )
            result = attempt.value
            prior = decision.checkpoint.attempts if decision.checkpoint else 0
            record(
                EntryGraphResult(
                    entry_id=result.entry_id,
                    status=result.status,
                    reason_code=result.reason_code,
                    detail=result.detail,
                    retryable=result.retryable,
                    attempts=prior + attempt.attempts,
                    nodes_created=result.nodes_created,
                    nodes_present=result.nodes_present,
                    edges_created=result.edges_created,
                    edges_present=result.edges_present,
                )
            )
        final = by_entry[entry_id]
        if final.reason_code in GRAPH_FAILURE_REASONS:
            context.graph_errors += 1
        if journal is not None and not dry_run:
            prior = decision.checkpoint.attempts if decision.checkpoint else 0
            journal.record_result(
                entry_id,
                digest,
                status=final.status,
                reason_code=final.reason_code,
                retryable=final.retryable,
                attempts=max(1, final.attempts - prior),
                prior_attempts=prior,
                detail=final.detail,
                result=final.to_artifact(),
            )
    return results, context


# ---------------------------------------------------------------------------
# Tombstone pruning
# ---------------------------------------------------------------------------


def prune_tombstoned(
    client: GraphClient,
    entries: Sequence[str],
    *,
    dry_run: bool = True,
    max_removals: int = 500,
) -> dict[str, object]:
    """Remove the nodes of tombstoned datasets, keeping their version history.

    The ``Dataset`` node and every ``DatasetVersion`` node that records this
    entry are removed.  The service removes incident edges with the nodes, so a
    traversal stops yielding the withdrawn dataset.  ``dry_run`` is the default
    because this is the only destructive graph operation in the pipeline.
    """
    planned = 0
    removed = 0
    sample: list[str] = []
    missing: list[str] = []
    failures: list[dict[str, str]] = []
    for entry_id in list(entries)[:max_removals]:
        ids = [dataset_node_id(entry_id)]
        try:
            versions = client.query_nodes(NODE_VERSION, {"entry_id": entry_id})
        except GraphError as exc:
            failures.append({"entry_id": entry_id, "reason_code": exc.code})
            continue
        for version in versions:
            node_id = version.get("id")
            if isinstance(node_id, str):
                ids.append(node_id)
        try:
            result = client.remove_nodes(ids, dry_run=dry_run)
        except GraphError as exc:
            failures.append({"entry_id": entry_id, "reason_code": exc.code})
            continue
        if dry_run:
            planned += int(result.get("would_remove_nodes") or 0)
            sample.extend(str(item) for item in result.get("sample_ids") or [])
        else:
            removed += int(result.get("removed_nodes") or 0)
            sample.extend(ids)
        for item in result.get("missing") or []:
            if isinstance(item, str):
                missing.append(item)
    return {
        "dry_run": bool(dry_run),
        "requested": len(list(entries)[:max_removals]),
        "would_remove": planned,
        "removed": removed,
        "sample": sample[:10],
        "missing": sorted(set(missing))[:10],
        "failures": failures[:10],
    }


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def check_accounting(
    rows: Sequence[dict[str, object]],
    results: Sequence[EntryGraphResult],
    discovered: int,
):
    """Reuse the pipeline-wide invariant: one status per entry, reasons present."""
    from . import integrate

    return integrate.check_accounting(rows, results, discovered)


def select_entries(
    rows: Sequence[dict[str, object]],
    *,
    entries: Sequence[str] | None = None,
    sample: int | None = None,
    limit: int | None = None,
) -> tuple[set[str], list[str], bool]:
    """Choose which entries to project, mirroring the other stages' semantics."""
    known = {str(row.get("entry_id") or "") for row in rows}
    candidates = sorted(known)
    if entries:
        wanted = list(dict.fromkeys(entries))
        unknown = [entry_id for entry_id in wanted if entry_id not in known]
        return {entry_id for entry_id in wanted if entry_id in known}, unknown, False
    if sample is not None:
        chosen = candidates[: max(0, sample)]
        return set(chosen), [], len(chosen) < len(candidates)
    if limit is not None:
        chosen = candidates[: max(0, limit)]
        return set(chosen), [], len(chosen) < len(candidates)
    return set(candidates), [], False


def run_id_for(commit: str, snapshot_sha256: str, selection: Iterable[str]) -> str:
    return model.sha256_hex(
        model.canonical_json(
            {
                "schema_version": 1,
                "catalog_commit": commit,
                "snapshot_sha256": snapshot_sha256,
                "graph_version": GRAPH_VERSION,
                "selection": sorted(selection),
            }
        )
    )[:16]


def outcome_rows(
    rows: Sequence[dict[str, object]],
    results: Sequence[EntryGraphResult],
    *,
    run_id: str,
    catalog_commit: str,
    retrieved_at: str,
) -> list[dict[str, object]]:
    """One append-only outcome row per entry, deduplicated by ``event_id``."""
    result_by_entry = {item.entry_id: item for item in results}
    outcomes: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("entry_id"))):
        entry_id = str(row.get("entry_id") or "")
        result = result_by_entry.get(entry_id)
        if result is None:
            continue
        catalog = row.get("catalog") if isinstance(row.get("catalog"), dict) else {}
        assert isinstance(catalog, dict)
        artifact = result.to_artifact()
        event_id = model.sha256_hex(
            model.canonical_json(
                {
                    "entry_id": entry_id,
                    "stage": "graph",
                    "input_hash": row.get("metadata_sha256"),
                    "outcome_code": f"{result.status}:{result.reason_code or ''}",
                    "attempt": 1,
                }
            )
        )
        outcomes.append(
            {
                "schema_version": 1,
                "event_id": event_id,
                "run_id": run_id,
                "entry_id": entry_id,
                "catalog_commit": catalog_commit,
                "stage": "graph",
                "status": result.status,
                "reason_code": result.reason_code,
                "detail": result.detail,
                "source_uri": catalog.get("meta_url"),
                "source_sha256": row.get("metadata_sha256"),
                "bytes": None,
                "etag": None,
                "last_modified": None,
                "retrieved_at": retrieved_at,
                "attempts": result.attempts,
                "retryable": result.retryable,
                "next_retry_at": None,
                "knowledge_source_id": None,
                "revision_id": None,
                "revision_sequence": None,
                "knowledge_decision": None,
                "record_kind": "graph_projection",
                "shard_index": None,
                "shard_count": None,
                "chunks": None,
                "vectors": None,
                "nodes_created": artifact["nodes_created"],
                "nodes_present": artifact["nodes_present"],
                "edges_created": artifact["edges_created"],
                "edges_present": artifact["edges_present"],
                "resumed": artifact["resumed"],
            }
        )
    return outcomes


def checkpoint_log(config: Config, *, enabled: bool = True) -> resume.CheckpointLog:
    return resume.CheckpointLog(
        resume.checkpoint_root(config), "graph", enabled=enabled
    )


def retry_policy(config: Config) -> resume.RetryPolicy:
    return resume.RetryPolicy(
        max_attempts=config.resume_max_attempts,
        base_delay_seconds=config.resume_base_delay_seconds,
        max_delay_seconds=config.resume_max_delay_seconds,
        max_total_attempts=config.resume_max_total_attempts,
    )


def resume_stats(
    results: Sequence[EntryGraphResult], checkpoint: resume.CheckpointLog | None
) -> dict[str, object]:
    """What resume and retry did, in the shape the report and ledger expect."""
    selected = [item for item in results if item.reason_code != "not_selected"]
    return {
        "checkpoint": (
            checkpoint.stats()
            if checkpoint is not None
            else {"enabled": False, "stage": "graph", "entries": 0, "by_status": {}}
        ),
        "reused": sum(1 for item in selected if item.resumed),
        "attempted": sum(1 for item in selected if not item.resumed),
        "retried": sum(1 for item in selected if item.attempts > 1),
        "retry_exhausted": sum(
            1 for item in selected if item.reason_code == resume.RETRY_EXHAUSTED_REASON
        ),
        "manifest_reused": 0,
    }


def tombstones_from_manifest(artifacts: dict[str, dict[str, object]]) -> set[str]:
    """Entry ids a previous integration run recorded as tombstoned.

    Free, offline tombstone evidence: no service call, and it is exactly the
    state the pipeline last observed.
    """
    tombstoned: set[str] = set()
    for entry_id, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict) and metadata.get("reason_code") == "source_tombstoned":
            tombstoned.add(entry_id)
            continue
        content = artifact.get("content")
        if isinstance(content, dict):
            for record in content.get("records") or []:
                if isinstance(record, dict) and record.get("reason_code") == "source_tombstoned":
                    tombstoned.add(entry_id)
                    break
    return tombstoned


def graph_client(config: Config) -> GraphClient:
    return GraphClient(config.graph_url, timeout=config.graph_timeout)


def graph_manifest_path(config: Config, manifest_name: str) -> Path:
    return Path(config.data_root) / "manifest" / manifest_name.replace(
        ".jsonl", ".graph.jsonl"
    )


def graph_manifest_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    ordered = sorted(rows, key=lambda item: str(item.get("entry_id")))
    return "".join(model.canonical_json(row) + "\n" for row in ordered).encode("utf-8")


def write_graph_manifest(
    rows: Sequence[dict[str, object]], path: Path
) -> tuple[bool, Path]:
    data = graph_manifest_bytes(rows)
    try:
        existing = path.read_bytes()
    except OSError:
        existing = None
    if existing == data:
        return False, path
    manifest.atomic_write(path, data)
    return True, path


def manifest_row(
    row: dict[str, object], result: EntryGraphResult
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entry_id": row.get("entry_id"),
        "catalog": row.get("catalog"),
        "category": row.get("category"),
        "title": row.get("title"),
        "metadata_sha256": row.get("metadata_sha256"),
        "graph": result.to_artifact(),
    }
