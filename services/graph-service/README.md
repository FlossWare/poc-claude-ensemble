# Claude Ensemble Graph Service

The Graph service is the Graph capability's HTTP service boundary, backed by a durable JSON store. The verified recovery baseline has no separate Ensemble application router, so this service is the canonical Graph REST boundary for #103.

## Boundary

Default listener:

`127.0.0.1:8766`

Health:

`GET /health`

Graph operations:

- `POST /graph/add-node`
- `POST /graph/add-edge`
- `GET /graph/node/{id}`
- `POST /graph/query`
- `POST /graph/traverse`

The service uses only Python's standard library. It does not depend on a graph database and does not claim success before a mutation has been durably persisted.

## Persistence

The default store is:

`~/.local/share/claude-ensemble/graph.json`

Set `POC_CLAUDE_ENSEMBLE_GRAPH_STORE` to use another location.

Mutations are serialized under an in-process reentrant lock and persisted by writing a temporary file, flushing it with `fsync`, and atomically replacing the store. A new process reconstructs the graph from that file.

Automatically generated node and edge IDs are SHA-256 identifiers over canonical JSON content, so the same logical object gets the same ID after a restart.

## Configuration

- `POC_CLAUDE_ENSEMBLE_GRAPH_HOST`, default `127.0.0.1`
- `POC_CLAUDE_ENSEMBLE_GRAPH_PORT`, default `8766`
- `POC_CLAUDE_ENSEMBLE_GRAPH_STORE`, default `~/.local/share/claude-ensemble/graph.json`

The service is loopback-only. Remote federation and authentication are intentionally outside this issue. Systemd installer/lifecycle integration is provided by `claude-graph.service.template` and `install.sh`. The service is independently owned by systemd and must not create or supervise sibling processes.

## Graphify extraction

Graphify is an optional in-process extraction engine. The Graph service remains
the only REST boundary and the existing NetworkX graph plus durable JSON store
remain canonical. Graphify does not run as a sibling service and its own
`graph.json` is not used as a second system of record.

Install the pinned Graphify dependency into the same Python environment used by
the service:

```bash
python3 -m pip install -r services/graph-service/requirements-graphify.txt
```

Configure the filesystem root the service is allowed to inspect:

```ini
POC_CLAUDE_ENSEMBLE_GRAPHIFY_ROOT=/path/to/local/source-corpus
```

The service rejects extraction targets outside that resolved root, including
symlink escapes. The Graphify cache is stored beside the Graph service's
configured graph store, not inside the source corpus.

Run extraction using a path relative to the configured root:

```bash
curl -fsS http://127.0.0.1:8766/graph/extract \\
  -H 'content-type: application/json' \\
  -d '{"path":"my-project"}' | jq
```

This integration invokes Graphify's deterministic local extraction API
(`graphify.extract.collect_files` and `graphify.extract.extract`) directly.
It does not invoke a model backend, Graphify Cloud, URL ingestion, or any
network service. The endpoint is disabled unless the allowed root is configured.
Only file types supported by Graphify's deterministic extractor are processed;
this endpoint does not perform model-based semantic extraction of arbitrary
PDFs or other documents.

Graphify node IDs are namespaced as `graphify:<id>`. Its source metadata,
locations, relationship direction, confidence tier, and other extraction
attributes are retained in GraphStore node/edge properties. Batch ingestion
validates the complete batch and persists it with one atomic store replacement.
Repeated identical extraction is idempotent.

The optional dependency must be installed before enabling extraction. Initial
package installation requires access to the package index; subsequent
extraction is local and can be run with network access disabled. Existing
Graph REST operations continue to work without Graphify installed.

## Test

From the repository root:

`python graph-service/test_graph_service.py`

The test starts the actual HTTP server, performs mutations and reads through HTTP, exercises negative paths and an injected persistence failure, shuts it down, starts a fresh server against the same store, and verifies the data and deterministic IDs survive the restart.
