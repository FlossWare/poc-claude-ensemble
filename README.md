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

## Embedding inference backends

The embedding REST contract is unchanged. The service still applies the Nomic
`search_document:` / `search_query:` prefixes, masked mean pooling, and L2
normalization, and returns 768-dimensional vectors. The inference runtime is
selectable at service startup:

| `ENSEMBLE_EMBED_BACKEND` | Runtime | Graph | Precision |
|---|---|---|---|
| `onnxruntime-cpu` (default) | ONNX Runtime | `model.onnx` | FP32 |
| `openvino-cpu` | OpenVINO CPU | `model.onnx` | FP32 |
| `openvino-gpu` | OpenVINO GPU | `model_fp16.onnx` | FP16 |

CPU ONNX Runtime remains the default and fallback. OpenVINO is optional; it
does not become a required dependency for CPU deployments. Install it in the
service's Python environment only when selecting an OpenVINO backend:

```bash
python3 -m pip install openvino
```

Select the backend before starting the embedding service:

```bash
# Existing behavior: ONNX Runtime CPU FP32
export ENSEMBLE_EMBED_BACKEND=onnxruntime-cpu

# Optional Intel GPU path: OpenVINO GPU FP16
export ENSEMBLE_EMBED_BACKEND=openvino-gpu
export ENSEMBLE_OPENVINO_PERFORMANCE_HINT=THROUGHPUT
python3 services/embeddings/embedding_service.py
```

Supported OpenVINO performance hints are `THROUGHPUT` (default),
`LATENCY`, and `CUMULATIVE_THROUGHPUT`. `ENSEMBLE_EMBED_THREADS` applies
to ONNX Runtime CPU. OpenVINO selects the device explicitly and reports its
actual device, runtime, precision, backend, and pinned model revision from
`GET /model` and `GET /health`.

**GPU selection fails closed.** If OpenVINO is not installed, the FP16 graph is
missing, or no GPU device is available, model loading fails and `/health`
returns 503. The service never silently falls back to CPU when
`openvino-gpu` was requested.

The runtime choice does not change the model ID, revision, vector dimensions,
prefixes, pooling, normalization, REST response shape, or ChromaDB collection
identity. Existing vectors do not need migration. CPU FP32 remains available
as the fallback. Keep one backend selected for a running embedding service;
do not switch inference backends while that process is serving requests.

To run the hardware-dependent GPU parity tests explicitly:

```bash
FLOSSWARE_TEST_OPENVINO=1 python3 -m unittest services/embeddings/test_embedding_service.py
```

The opt-in tests verify the selected GPU device, FP16 graph, vector shape and
normalization, and cosine similarity against CPU FP32. They are skipped by
default so ordinary CI does not require Intel GPU hardware.

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
