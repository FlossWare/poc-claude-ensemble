# Claude Ensemble MCP server

A stdio Model Context Protocol server that exposes the Claude Ensemble
gateway as native tools for Claude Code (or any MCP client).

All traffic goes to the loopback gateway (`http://127.0.0.1:8080` by default).
The server holds no provider credentials and never calls a backing service
directly.

## Tools

| Tool | Gateway route(s) | Notes |
| --- | --- | --- |
| `search_knowledge(query, limit)` | `POST /api/v1/lexical/knowledge/search` | Exact-term (FTS5) knowledge search. |
| `search_semantic(query, limit)` | `POST /api/v1/embedding/embeddings`, then `POST /api/v1/vector/collections/knowledge_chunks/query` | Dense-vector knowledge search. |
| `search_hybrid(query, limit)` | `POST /api/v1/hybrid/search` | Transcript + knowledge fusion; reports `degraded` legs. |
| `get_record(source_id)` | `GET /api/v1/corpus/records` | Returns the newest revision for the source id. |
| `list_origins()` | `GET /api/v1/corpus/origins` | Live origins and counts. |
| `graph_query(query_type, params)` | `POST /api/v1/graph/query` | Node filters, or `query_type="edges"`. |
| `graph_traverse(start_node, depth, limits)` | `POST /api/v1/graph/traverse` | `limits` supports `direction`, `edge_type`, `max_results`. |
| `get_model_status()` | `POST /api/v1/thompson` (`get_state`) | Arms with Beta alpha/beta and posterior mean. |
| `draw_crew(task_type, workers, arbiter)` | `POST /api/v1/thompson` via `tools/crew_select.py` | Draws distinct arms, excludes the arbiter, writes an exposure ledger row. |
| `get_learning_status()` | `POST /api/v1/learning` (`get_report`, `get_recent_outcomes`) | Partial failures are reported in `errors`. |
| `health_check()` | `GET /api/v1/health`, `/ready`, `/services` | Lists degraded services. |

### Route deviations from the original brief

The brief named `/api/v1/lexical/search` and `/api/v1/vector/search`. Neither
is the correct knowledge-search route in this deployment:

* `/api/v1/lexical/search` forwards to the lexical service's *transcript*
  search (`messages_fts`). The knowledge corpus is served by
  `/api/v1/lexical/knowledge/search`.
* The vector service has no `/search` endpoint; it exposes collection
  operations only. `search_semantic` therefore embeds the query through the
  gateway and runs a vector query, which is the same path the hybrid service
  uses. Set `POC_CLAUDE_ENSEMBLE_MCP_SEMANTIC_PATH` to a single route if one is
  added later.

Graph calls use the gateway's public `/api/v1/graph/*` paths; the backing graph
service itself owns `/graph/*`.

## Run

From the repository root:

```sh
python3 -m mcp
```

Or from any directory, using the launcher:

```sh
python3 /path/to/poc-claude-ensemble/mcp/run_server.py
```

Diagnostics go to stderr; stdout carries only MCP protocol messages.

## Claude Code integration

Copy `mcp/mcp.json.example` to `.mcp.json` in the repository root and replace
the path with this checkout's `mcp/run_server.py`:

```json
{
  "mcpServers": {
    "claude-ensemble": {
      "command": "python3",
      "args": ["/absolute/path/to/poc-claude-ensemble/mcp/run_server.py"]
    }
  }
}
```

Or register it directly:

```sh
claude mcp add claude-ensemble -- python3 /absolute/path/to/poc-claude-ensemble/mcp/run_server.py
```

The launcher adds the repository root to `sys.path`, so the client's working
directory does not matter. If you prefer `python3 -m mcp`, run it with the
repository root as the working directory.

The server does not need to be started separately. Claude Code starts it on
demand and speaks MCP over the process's stdin/stdout.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `POC_CLAUDE_ENSEMBLE_GATEWAY_URL` | `http://127.0.0.1:8080` | Gateway base URL. Loopback only. |
| `POC_CLAUDE_ENSEMBLE_MCP_ALLOW_REMOTE_GATEWAY` | unset | Set to `1` to permit a non-loopback gateway (not recommended). |
| `POC_CLAUDE_ENSEMBLE_MCP_TIMEOUT_SECONDS` | `15` | Per-request timeout. |
| `POC_CLAUDE_ENSEMBLE_MCP_RATE_PER_SECOND` | `8` | Token-bucket refill rate per server process. |
| `POC_CLAUDE_ENSEMBLE_MCP_RATE_BURST` | `8` | Token-bucket burst. |
| `POC_CLAUDE_ENSEMBLE_MCP_LOG_LEVEL` | `WARNING` | stderr log level. |
| `POC_CLAUDE_ENSEMBLE_MCP_KNOWLEDGE_COLLECTION` | `knowledge_chunks` | Collection used by `search_semantic`. |
| `POC_CLAUDE_ENSEMBLE_MCP_PRIOR_STRENGTH` | `1.0` | Beta prior strength used for posterior display. |
| `POC_CLAUDE_ENSEMBLE_COLLABORATION_AUTH_TOKEN` | unset | Optional bearer token forwarded to the gateway. |
| `POC_CLAUDE_ENSEMBLE_CREW_LEDGER_DIR` | `~/.FlossWare/ledger/crews` | Where `draw_crew` writes exposure rows. |

## Security

* The gateway URL must be loopback unless explicitly overridden.
* Tool paths are fixed `/api/v1/...` strings; arguments are never used to
  build URLs, so tool input cannot redirect a request.
* Every request passes through a token-bucket rate limiter, including the
  Thompson draws used by `draw_crew`.
* Gateway and service errors become structured MCP errors
  (`{"ok": false, "tool": ..., "error": {"code", "message", "status", "details"}}`),
  never raw tracebacks.
* `draw_crew` is the only tool with a side effect, and it is local: one
  exposure row in the crew ledger. It is annotated `readOnlyHint: false`.

## Tests

```sh
python3 -m pytest mcp/test_mcp.py -q
```

The suite starts a loopback mock gateway and drives the real MCP server through
the SDK's in-memory transport: tool listing, input/output schemas, every tool,
structured errors, loopback enforcement and rate limiting.

## Package-name note

The directory is named `mcp` so `python3 -m mcp` works, but that shadows the
installed `mcp` SDK. `mcp/__init__.py` resolves the collision by putting the
installed SDK first on the package `__path__` and loading the local modules as
`mcp.ce_client`, `mcp.ce_tools` and `mcp.ce_server`. The public file names stay
as documented.
