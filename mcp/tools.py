"""MCP tool definitions for the Claude Ensemble gateway.

Each tool has a Pydantic input model (advertised as the MCP ``inputSchema``)
and an output model (advertised as ``outputSchema`` and used to validate every
successful result before it is returned).  Handlers are synchronous and are
run in a worker thread by the server, so blocking HTTP never stalls the MCP
event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mcp import load_impl
from pydantic import BaseModel, ConfigDict, Field

load_impl("client")
from mcp.ce_client import GatewayClient  # noqa: E402
from mcp.types import Tool, ToolAnnotations  # noqa: E402


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Output(BaseModel):
    # Services regularly add fields; preserving them is more useful than
    # dropping data just because this schema predates the field.
    model_config = ConfigDict(extra="allow")


class SearchInput(_Input):
    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="Natural-language or keyword query (at most 2000 characters).",
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return (1-100).")


class GetRecordInput(_Input):
    source_id: str = Field(
        min_length=1,
        max_length=512,
        description="Corpus source_id, as returned by a search result or corpus export.",
    )


class ListOriginsInput(_Input):
    """list_origins takes no arguments."""


class GraphQueryInput(_Input):
    query_type: str = Field(
        min_length=1,
        description=(
            "Node type to match (for example 'model'), or 'edges' to query "
            "relationships."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Exact property filters for node queries; from_type/to_type/"
            "relationship for query_type='edges'."
        ),
    )


class GraphTraverseInput(_Input):
    start_node: str = Field(min_length=1, description="Node id to start from.")
    depth: int = Field(default=1, ge=1, le=100, description="Maximum traversal depth (1-100).")
    limits: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional bounds: direction ('out'|'in'|'both'), edge_type, and "
            "max_results (enforced client-side)."
        ),
    )


class ModelStatusInput(_Input):
    """get_model_status takes no arguments."""


class DrawCrewInput(_Input):
    task_type: str = Field(
        min_length=1, description="Thompson task type, for example 'code-review'."
    )
    workers: int = Field(default=3, ge=1, le=10, description="Number of distinct crew arms.")
    arbiter: str | None = Field(
        default=None,
        description=(
            "Model id or arm that must not appear in the crew. Defaults to "
            "crew_select's configured arbiter."
        ),
    )


class LearningStatusInput(_Input):
    """get_learning_status takes no arguments."""


class HealthCheckInput(_Input):
    """health_check takes no arguments."""


# -- outputs ---------------------------------------------------------------


class SearchOutput(_Output):
    ok: bool
    tool: str
    query: str
    limit: int
    count: int
    results: list[dict[str, Any]]
    method: str | None = None
    knowledge: list[dict[str, Any]] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    reranked: bool | None = None
    related_sessions: list[Any] = Field(default_factory=list)
    intent: Any = None


class RecordOutput(_Output):
    ok: bool
    tool: str
    source_id: str
    record: dict[str, Any]
    revision_count: int
    page_truncated: bool


class OriginsOutput(_Output):
    ok: bool
    tool: str
    count: int
    origins: list[dict[str, Any]]


class GraphQueryOutput(_Output):
    ok: bool
    tool: str
    query_type: str
    count: int
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class GraphTraverseOutput(_Output):
    ok: bool
    tool: str
    start: str
    depth: int
    direction: str
    count: int
    truncated: bool
    results: list[dict[str, Any]]


class ModelStatusOutput(_Output):
    ok: bool
    tool: str
    state_last_updated: str | None
    prior_strength: float
    arm_count: int
    arms: dict[str, dict[str, Any]]


class CrewOutput(_Output):
    ok: bool
    tool: str
    selection: dict[str, Any]


class LearningStatusOutput(_Output):
    ok: bool
    tool: str
    report: str | None
    total_outcomes: int | None
    recent_outcomes: list[Any]
    errors: list[str]


class HealthCheckOutput(_Output):
    ok: bool
    tool: str
    gateway: dict[str, Any]
    ready: dict[str, Any] | None
    counts: dict[str, Any]
    degraded: list[dict[str, Any]]
    services: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]
    read_only: bool = True
    idempotent: bool = True

    def to_tool(self) -> Tool:
        return Tool(
            name=self.name,
            title=self.title,
            description=self.description,
            inputSchema=self.input_model.model_json_schema(),
            outputSchema=self.output_model.model_json_schema(),
            annotations=ToolAnnotations(
                title=self.title,
                readOnlyHint=self.read_only,
                destructiveHint=False,
                idempotentHint=self.idempotent,
                openWorldHint=False,
            ),
        )


def build_tool_specs(client: GatewayClient) -> list[ToolSpec]:
    """Return the tool catalogue bound to ``client``."""
    return [
        ToolSpec(
            name="search_knowledge",
            title="Search knowledge (lexical)",
            description=(
                "Exact-term (SQLite FTS5) search over the ingested knowledge "
                "corpus. Returns chunks with source, url, title and provenance. "
                "Use this for names, ids and phrases; use search_semantic for "
                "meaning-based recall."
            ),
            input_model=SearchInput,
            output_model=SearchOutput,
            handler=lambda args: client.search_knowledge(args.query, args.limit),
        ),
        ToolSpec(
            name="search_semantic",
            title="Search knowledge (semantic)",
            description=(
                "Dense-vector similarity search over the knowledge corpus. The "
                "query is embedded by the local embedding service and matched "
                "against the knowledge_chunks vector collection. Returns "
                "cosine-style scores."
            ),
            input_model=SearchInput,
            output_model=SearchOutput,
            handler=lambda args: client.search_semantic(args.query, args.limit),
        ),
        ToolSpec(
            name="search_hybrid",
            title="Search (hybrid)",
            description=(
                "Fused transcript + knowledge retrieval through the hybrid "
                "service (vector and lexical legs, reciprocal-rank fusion, "
                "optional reranking). Reports degraded legs explicitly."
            ),
            input_model=SearchInput,
            output_model=SearchOutput,
            handler=lambda args: client.search_hybrid(args.query, args.limit),
        ),
        ToolSpec(
            name="get_record",
            title="Get corpus record",
            description=(
                "Fetch the newest ingested revision for one corpus source_id, "
                "including content (subject to the gateway's per-record "
                "truncation budget) and provenance."
            ),
            input_model=GetRecordInput,
            output_model=RecordOutput,
            handler=lambda args: client.get_record(args.source_id),
        ),
        ToolSpec(
            name="list_origins",
            title="List corpus origins",
            description="List every live corpus origin with its record count.",
            input_model=ListOriginsInput,
            output_model=OriginsOutput,
            handler=lambda args: client.list_origins(),
        ),
        ToolSpec(
            name="graph_query",
            title="Query graph",
            description=(
                "Query knowledge-graph nodes by type and exact property "
                "filters, or relationships (query_type='edges') by endpoint "
                "type and relationship name."
            ),
            input_model=GraphQueryInput,
            output_model=GraphQueryOutput,
            handler=lambda args: client.graph_query(args.query_type, args.params),
        ),
        ToolSpec(
            name="graph_traverse",
            title="Traverse graph",
            description=(
                "Breadth-first traversal from a node with a bounded depth and "
                "optional direction, edge-type and result-count limits."
            ),
            input_model=GraphTraverseInput,
            output_model=GraphTraverseOutput,
            handler=lambda args: client.graph_traverse(args.start_node, args.depth, args.limits),
        ),
        ToolSpec(
            name="get_model_status",
            title="Thompson model status",
            description=(
                "Thompson-sampling arms with observed successes/failures, "
                "derived Beta alpha/beta, posterior mean, call count and "
                "average cost."
            ),
            input_model=ModelStatusInput,
            output_model=ModelStatusOutput,
            handler=lambda args: client.get_model_status(),
        ),
        ToolSpec(
            name="draw_crew",
            title="Draw reviewer crew",
            description=(
                "Draw a distinct-arm worker crew for a task type using the "
                "existing Thompson crew_select workflow, excluding the arbiter. "
                "Writes a local exposure row to the crew ledger."
            ),
            input_model=DrawCrewInput,
            output_model=CrewOutput,
            handler=lambda args: client.draw_crew(args.task_type, args.workers, args.arbiter),
            read_only=False,
            idempotent=False,
        ),
        ToolSpec(
            name="get_learning_status",
            title="Learning stack status",
            description=(
                "Learning-service report plus recent task outcomes, including "
                "per-call partial failures."
            ),
            input_model=LearningStatusInput,
            output_model=LearningStatusOutput,
            handler=lambda args: client.get_learning_status(),
        ),
        ToolSpec(
            name="health_check",
            title="Claude Ensemble health",
            description=(
                "Gateway liveness, readiness (graph + memory) and per-service "
                "health, with a list of degraded services."
            ),
            input_model=HealthCheckInput,
            output_model=HealthCheckOutput,
            handler=lambda args: client.health_check(),
        ),
    ]
