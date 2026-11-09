# poc-claude-ensemble

Proof-of-concept services for Claude Ensemble, the local REST services behind prompt-time memory, transcript search, and ingestion. This repository is the baseline the next Ensemble work branches from. The previous implementation is archived at FlossWare-Archives/claude-ensemble-legacy.

## Services

Each service is a standard-library HTTP server bound to loopback, with its own tests.

| Service | Port | Contract |
|---|---:|---|
| Embedding (`services/embeddings`) | 8770 | nomic-embed-text-v1.5 vectors, 768 dimensions |
| Vector (`services/embeddings`) | 8771 | ChromaDB-backed exact search |
| Lexical (`services/embeddings`) | 8772 | SQLite FTS5 keyword search, incremental writes |
| Hybrid (`services/embeddings`) | 8773 | Search orchestration and transcript ingestion |
| Reranking (`services/embeddings`) | 8774 | Local cross-encoder |
| Chunking (`services/chunking`) | 8775 | Versioned, deterministic chunk profiles |
| Tokenization (`services/tokenization`) | 8776 | Versioned analyzers |
| Index (`services/index`) | 8777 | Source manifest: revision order and tombstones |
| Extraction (`services/extraction`) | 8778 | Exact identifiers, with uncertainty flagged |
| Intent (`services/intent`) | 8779 | Query intent with evidence |
| Ranking (`services/ranking`) | 8780 | Explainable reciprocal rank fusion |
| Knowledge (`services/knowledge`) | 8781 | Sourced records; revisions checked against Index |
| Memory (`services/memory`) | 8767 | Durable episodic memory and the session-events ledger |
| Ingestion (`services/ingestion`) | 8782 | URI, uploaded document, and inline-text ingestion |

## Source layout

`services/` is the authoritative implementation of the POC's REST services and
their tests. Do not implement service changes independently in `core/`.

`core/` is retained as a compatibility mirror for the historical package
layout and for repository-local reviewer tooling (`core/reviewer-mcp`,
`core/claude-code-skills`, and related tools). When a module exists in both
trees, update `services/` first and copy the exact file to `core/`. CI checks
that all shared non-test Python modules remain byte-identical; the independent
test configuration files are intentionally excluded.

The retired Alert and Messenger service implementations are archived under
`archive/legacy-core/`. They are not part of the active POC service set.

## Session history API

The Memory service exposes the existing append-only session-events ledger
through read-only REST endpoints. The API is part of this POC and reads the
existing JSONL source of truth; it does not introduce a separate Ledger
service or repository. The POC deliberately has no REST write endpoint for
the ledger.

- `GET /history` lists events with filtering and offset pagination.
- `GET /history/{event_id}` returns one event by its event ID.

The default page size is 50 and the maximum is 100. Supported filters are
`event_type`, `source`, `correlation_id`, `since`, and `until`.
Pagination uses the returned `pagination.next_cursor` as the next request's
`cursor` parameter. See
[`services/docs/SESSION_HISTORY_API.md`](services/docs/SESSION_HISTORY_API.md)
for the complete contract, response examples, error behavior, and curl usage.

## Embedding inference backends

The embedding REST contract is unchanged. The service still applies the Nomic
`search_document:` / `search_query:` prefixes, masked mean pooling, and L2
normalization, and returns 768-dimensional vectors. The inference runtime is
selectable at service startup:

| `POC_CLAUDE_ENSEMBLE_EMBED_BACKEND` | Runtime | Graph | Precision |
|---|---|---|---|
| `onnxruntime-cpu` (default) | ONNX Runtime | `model.onnx` | FP32 |
| `openvino-cpu` | OpenVINO CPU | `model.onnx` | FP32 |
| `openvino-gpu` | OpenVINO GPU | `model_fp16.onnx` | FP16 |

CPU ONNX Runtime remains the default and fallback. OpenVINO is optional; it
does not become a required dependency for CPU deployments. Install it in the
service's Python environment only when selecting an OpenVINO backend:

```bash
python3 -m pip install 'openvino>=2026.2'
```

Select the backend before starting the embedding service:

```bash
# Existing behavior: ONNX Runtime CPU FP32
export POC_CLAUDE_ENSEMBLE_EMBED_BACKEND=onnxruntime-cpu

# Optional Intel GPU path: OpenVINO GPU FP16
export POC_CLAUDE_ENSEMBLE_EMBED_BACKEND=openvino-gpu
export POC_CLAUDE_ENSEMBLE_OPENVINO_PERFORMANCE_HINT=THROUGHPUT
python3 services/embeddings/embedding_service.py
```

Supported OpenVINO performance hints are `THROUGHPUT` (default),
`LATENCY`, and `CUMULATIVE_THROUGHPUT`. `POC_CLAUDE_ENSEMBLE_EMBED_THREADS` applies
to ONNX Runtime CPU. OpenVINO selects the device explicitly and reports its
actual execution device(s), runtime, graph precision, OpenVINO inference precision
hint when exposed by the runtime, backend, and pinned model revision from
`GET /model` and `GET /health`. Graph precision describes the ONNX file; the
compiled inference precision hint is separate and is not a per-operation
execution trace.

**GPU selection fails closed.** If OpenVINO is not installed, the FP16 graph is
missing, or no GPU device is available, model loading fails and `/health`
returns 503. The service never silently falls back to CPU when
`openvino-gpu` was requested.

The runtime choice does not change the model ID, revision, vector dimensions,
prefixes, pooling, normalization, REST response shape, or ChromaDB collection
identity. Existing vectors do not need migration, but FP32 and FP16 inference can
produce slightly different vectors. Keep one backend and precision consistent
for all writes to a collection. If changing precision for an existing collection,
plan a deliberate full re-embedding/rebuild rather than mixing old and new
vectors. Keep one backend selected for a running embedding service; do not
switch inference backends while that process is serving requests.

To run the hardware-dependent GPU parity tests explicitly:

```bash
cd services/embeddings && FLOSSWARE_TEST_OPENVINO=1 python3 -m unittest -v test_embedding_service
```

The opt-in tests verify the selected GPU device, FP16 graph, vector shape and
normalization, and cosine similarity against CPU FP32 (each tested vector pair
must exceed 0.9999). They are skipped by
default so ordinary CI does not require Intel GPU hardware. Ordinary CI also
runs mocked OpenVINO failure-path tests for a missing dependency, unavailable
GPU, missing FP16 graph, invalid backend, and invalid performance hint; these
tests do not require OpenVINO or GPU hardware.

To smoke-test the actual REST process without touching the deployed service,
start a second instance on a separate loopback port:

```bash
POC_CLAUDE_ENSEMBLE_EMBED_BACKEND=openvino-gpu \
POC_CLAUDE_ENSEMBLE_EMBED_PORT=18770 \
python3 services/embeddings/embedding_service.py
```

In another terminal, inspect the selected runtime and send a real embedding
request:

```bash
curl -fsS http://127.0.0.1:18770/health | jq
curl -fsS http://127.0.0.1:18770/model | jq
curl -fsS http://127.0.0.1:18770/embeddings \
  -H 'content-type: application/json' \
  -d '{"texts":["TLS certificate rotation validation"],"mode":"document"}' \
  | jq '{ok, model, dimensions, vectors:(.embeddings|length), vector_dimensions:(.embeddings[0]|length)}'
```

The model endpoint should identify `openvino-gpu`, a `GPU.*` device, and
`FP16`; the embedding response should contain one 768-dimensional vector.
Stop this test process with Ctrl-C when finished.

## Running the tests

From the repository root, run the whole suite with pytest:

    python3 -m pytest services -q

Each service directory also runs on its own, for example
`cd services/chunking && python3 -m unittest -v test_chunking_service`.
The embeddings tests load the real Nomic and cross-encoder models from
`~/.FlossWare/models` and skip only when those files are absent. Tests that
cross a service boundary start the real services in-process on ephemeral
loopback ports.

## Status

The services run from `shared/` in the working deployment and are copied here as they stand.

- Keyword analysis: index text and queries go through the Tokenization service (`lexical-v1`). Existing databases are upgraded with `services/embeddings/migrate_lexical_tokenization.py`, run with the lexical service stopped.
- Sourced documents: `services/embeddings/build_knowledge_chunks.py` stores each document as a Knowledge record, chunks it through Chunking, and writes vectors to `knowledge_chunks` and exact-match rows to the knowledge lexical index. Transcript sessions are indexed in the transcript lexical index.
- Transcript outcomes: finished turns are recorded to Thompson during ingest (`services/embeddings/turn_outcomes.py`). Send `"record_outcomes": false` in the `POST /sessions/ingest` body to skip this.
- Collaboration execution events are appended to the existing session-events ledger; the Control Center exposes historical and live model interaction review.


## Graphical Control Center

The gateway defaults to loopback-only. Optional home-LAN access requires
`POC_CLAUDE_ENSEMBLE_HTTP_HOST=0.0.0.0` and an explicit
`POC_CLAUDE_ENSEMBLE_ALLOWED_NETWORKS=192.168.1.0/24,192.168.2.0/24` (replace with the
trusted LAN CIDR). The gateway enforces this client allowlist and always
permits loopback. Configure the host firewall to allow TCP 8080 only from
that subnet. This exposes the REST API as well as the dashboard; do not
port-forward it or expose it to the public internet.


The gateway serves a read-only browser interface at
`http://127.0.0.1:8080/dashboard` (also available at `/`). It uses the
operational dashboard REST API below and refreshes the service inventory and
storage snapshot every 30 seconds.

The Control Center includes:
- Overview of healthy, degraded, and unavailable services.
- Service inventory with HTTP health status and probe latency.
- ChromaDB collection and vector counts when exposed by the vector service.
- A documented processing-stage map.

The pipeline view is currently a capability map, not a live execution trace.
Graph database and Thompson metrics are not yet included in this dashboard API.
Operational views are read-only and do not restart services or mutate stored data. The separate Ingest workbench writes only after a successful preview and explicit operator confirmation.

## Installing the local services

The server installer installs and enables both the canonical REST gateway and
the URI/document ingestion service as systemd user units. Run it from the
deployed `shared/` directory (or from this repository's `services/`
directory when testing a source checkout):

```bash
bash server/install.sh
```

The installer writes `claude-ensemble.service` and
`claude-ingestion.service` under `~/.config/systemd/user`, reloads systemd,
starts ingestion and the gateway, and waits for both the gateway readiness
endpoint and ingestion `/health` endpoint. The ingestion unit is loopback-only,
runs with a read-only home and system filesystem, and uses a private temporary
directory for bounded PDF extraction. It does not enable anonymous FTP.

To install and enable the units without starting them (for a top-level
installer that controls startup ordering), set
`POC_CLAUDE_ENSEMBLE_DEFER_SERVICE_START=1`. The top-level installer must
then start `claude-ingestion.service` and `claude-ensemble.service` after
their dependencies are available.

Inspect the service with:

```bash
systemctl --user status claude-ingestion.service
journalctl --user -u claude-ingestion.service -f
curl -fsS http://127.0.0.1:8782/health | jq
```

The service is also runnable directly for development:

```bash
python3 services/ingestion/ingestion_service.py
```

## Operational dashboard API

The gateway exposes a read-only operational view of the POC service fleet at
`http://127.0.0.1:8080`. It probes each service's `GET /health` endpoint and
reports health state and probe latency. Service URLs can be overridden with
`POC_CLAUDE_ENSEMBLE_<SERVICE>_URL` (for example,
`POC_CLAUDE_ENSEMBLE_VECTOR_URL`). The original `ENSEMBLE_<SERVICE>_URL`
override remains supported for compatibility.

| Endpoint | Description |
|---|---|
| `GET /api/v1/dashboard` | Aggregated service health and storage snapshot |
| `GET /api/v1/services` | Service inventory and health |
| `GET /api/v1/services/{id}` | One service's metadata and health |
| `GET /api/v1/services/{id}/health` | One service's health probe |
| `GET /api/v1/services/{id}/metrics` | Service-specific metrics where supported |

The inventory covers embedding, vector, lexical, hybrid, reranking, chunking,
tokenization, index, extraction, intent, ranking, knowledge, memory, and ingestion.
Vector metrics report collection names and counts from the existing
`GET /collections` endpoint. Services without a metrics endpoint report
metrics as unavailable. Ingestion history is exposed separately from the
append-only session-events ledger; aggregate ingestion metrics remain unavailable
rather than being inferred from logs or process state.

Example:

```bash
curl -fsS http://127.0.0.1:8080/api/v1/dashboard | jq
curl -fsS http://127.0.0.1:8080/api/v1/services | jq
curl -fsS http://127.0.0.1:8080/api/v1/services/vector/metrics | jq
```

The inventory probes services concurrently with a bounded worker pool and
short per-service timeout. It is observational only: it does not restart
services, write to stores, or change live configuration.

## Source ingestion

The POC gateway exposes a first-class source ingestion API through the Ingestion
service on loopback port 8782. The request contract accepts a remote `uri`,
an uploaded file (`filename` plus `content_base64`), or inline `text`.
Transport resolution is separate from content extraction.

The POC connector registry supports HTTP and HTTPS on their standard ports.
Anonymous FTP is also available as an explicit opt-in because it transmits data
without encryption. Enable it with
`POC_CLAUDE_ENSEMBLE_INGESTION_ENABLE_FTP=true`; URI credentials are never
accepted. The FTP connector pins both the control and passive data connections
to the validated public peer address. SFTP and object-storage schemes remain
unavailable until their connectors are implemented and tested.

Remote fetching rejects credentials in URIs, non-public DNS results, unsafe
redirect targets, oversized responses, and unsupported schemes. HTTP(S) is
restricted to standard ports. PDF extraction uses `pdftotext` from Poppler (`poppler-utils` on Debian/Fedora-derived Linux package sets); it is time- and output-bounded. HTML extraction omits script/style content. PDF image OCR is not included. The health endpoint reports only transports that
are actually enabled.

- `GET /api/v1/ingestion/health` reports the supported schemes and extractors.
- `POST /api/v1/ingestion/preview` resolves and extracts a source without persisting it. It returns a short-lived `preview_id` tied to the exact extracted content.
- `POST /api/v1/ingestion/ingest` accepts that `preview_id`, so it ingests exactly what was previewed rather than refetching a potentially changed remote resource. It stores the source in Knowledge, chunks it,
  embeds it, and updates the lexical and vector knowledge indexes.
- `GET /api/v1/ingestion/history` returns bounded ingestion lifecycle history (started, completed, rejected, and failed) from
  the existing append-only session-events ledger.

Example preview:

```bash
curl -fsS http://127.0.0.1:8080/api/v1/ingestion/preview \
  -H 'content-type: application/json' \
  -d '{"uri":"https://example.org/document.pdf"}' | jq
```

Start the ingestion service with:

```bash
python3 services/ingestion/ingestion_service.py
```

The service is loopback-only and uses the existing Knowledge, Chunking,
Embedding, Lexical, Vector, and Memory services. It does not create a parallel
document store. Ingestion lifecycle events are appended as metadata-only records to the
existing session-events ledger; event persistence is best-effort and never changes the ingestion result.

## Model interaction history

The collaboration orchestrator records metadata-only lifecycle events for
Arbiter decisions, solver workers, reviewers, and model provider exchanges.
Events are appended idempotently through the Memory service to the existing
append-only session-events ledger. Prompt and response bodies, credentials,
authorization headers, and raw provider payloads are not captured.

- `GET /api/v1/interactions?limit=50&cursor=0` reads historical events.
- `GET /api/v1/interactions/stream` streams new events using Server-Sent Events.
- SSE sequence cursors support reconnect through `Last-Event-ID`.

Open `http://127.0.0.1:8080/dashboard` and select **Interactions** to inspect
the historical timeline and watch new activity. Select **Ingest** to preview
and submit supported sources.

## Knowledge retrieval

Knowledge (books, RFCs, papers) is searched beside transcripts, never merged into
their ranking. `POST /search` returns the transcript results unchanged, plus a
separate `knowledge` list. Each knowledge hit carries `source_type: "knowledge"`,
origin, source URL, document title, chunk ID and position, and content SHA-256.
Knowledge hits come from two legs fused with reciprocal-rank fusion:

- **Vector:** the `knowledge_chunks` collection, for meaning-based matches.
- **Exact match:** the knowledge lexical index (`POC_CLAUDE_ENSEMBLE_KNOWLEDGE_LEXICAL_DB`), analyzed
  with the same analyzer as transcripts, so identifiers such as RFC numbers match exactly.

Knowledge-only retrieval:

```bash
curl -s http://127.0.0.1:8773/knowledge/search \
  -H 'content-type: application/json' \
  -d '{"query":"RFC 8259","top_k":10}'
```

Set `POC_CLAUDE_ENSEMBLE_KNOWLEDGE_SEARCH=0` to turn off the `knowledge` section in `/search`.
Knowledge failures are reported in `degraded` as `knowledge:<leg>` and never affect
the transcript results.

Search runs its independent legs concurrently. The lexical, intent, related-session and
knowledge legs start while the query embedding and transcript vector steps are in progress.
The `/knowledge/search` endpoint runs its vector and exact-match legs at the same time. The
gain on a live service was about 11% on `/search` (mean 3.87s to 3.46s over 15 queries). Most
of the remaining time is the embedding and reranking calls, which each hold a model lock and
handle one request at a time. Concurrency is tested in `services/embeddings/test_hybrid_concurrency.py`.

Chunks stored before exact-match rows existed are copied into the index with
`services/embeddings/backfill_knowledge_lexical.py`. The copy is idempotent. Rows
copied before the provenance columns existed (chunk position, chunk count, content
SHA-256) show those fields as `null` until the backfill is rerun, which fills them in
without overwriting existing values. The backfill log is written to
`~/.FlossWare/poc-claude-ensemble/state/logs/backfill-knowledge-lexical.log`.

Options: `--workers N` sets how many batches are posted to the lexical service at once
(default 4). `--start-offset N` resumes a run at a record offset, for example after an
interrupted run. `--limit N` stops early for trial runs.
