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

## Session history API

The Memory service exposes the existing append-only session-events store as a
read-only REST resource. This is implemented inside this POC repository; it
does not require a separate Ledger service or repository.

List events (default page size 50, maximum 100):

```bash
curl -fsS 'http://127.0.0.1:8767/history?limit=50' | jq
```

Supported filters are `event_type`, `source`, `correlation_id`, `since`,
and `until` (ISO-8601 timestamps). Pagination uses the returned
`pagination.next_cursor` value:

```bash
curl -fsS 'http://127.0.0.1:8767/history?event_type=session_end&limit=25' | jq
curl -fsS 'http://127.0.0.1:8767/history?correlation_id=request-123' | jq
curl -fsS 'http://127.0.0.1:8767/history?limit=25&cursor=25' | jq
curl -fsS 'http://127.0.0.1:8767/history/<event-id>' | jq
```

The collection response contains `entries` and `pagination` metadata
(`limit`, `returned`, `total`, and nullable `next_cursor`). A missing
entry returns 404; invalid filters or limits return 400. History reads fail
closed with 500 if the JSONL source contains a malformed record, rather than
silently omitting potentially important events. The endpoint is read-only;
existing append/idempotency behavior remains unchanged.

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
- Sourced documents: `services/embeddings/build_knowledge_chunks.py` stores each document as a Knowledge record, chunks it through Chunking, and writes vectors to `knowledge_chunks`. Lexical indexing covers transcript sessions only.
- Not yet built: Thompson outcome recording from transcripts, and worker/arbiter quality monitoring.


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

Chunks stored before exact-match rows existed are copied into the index with
`services/embeddings/backfill_knowledge_lexical.py`. The copy is idempotent.
