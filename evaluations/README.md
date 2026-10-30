# Evaluations

This directory contains repeatable quality, correctness, and performance evaluations for the poc-claude-ensemble services and local inference runtimes. It is deliberately separate from service unit tests: unit tests protect contracts; evaluations compare implementations and measure behavior on representative workloads.

## Principles

- Do not change production service behavior just to make an evaluation pass.
- Keep the current implementation as the baseline for every comparison.
- Separate quality from speed. A faster implementation is not an improvement if retrieval quality regresses.
- Record hardware, software versions, model revisions, provider configuration, and corpus revision.
- Use isolated temporary stores and collections. Never benchmark against or mutate live CE data.
- Warm up models before timed runs; report cold-start separately.
- Run repeated trials and report p50/p95 latency, throughput, and sample count.
- Make optional hardware providers explicit. Missing OpenVINO/GPU support must be reported as unavailable, not silently treated as CPU.
- Keep all model artifacts local; evaluations must not download models or access hosted inference APIs.

## Evaluation tracks

| Track | Scope | Primary outputs |
|---|---|---|
| embedding | ONNX Runtime CPU vs OpenVINO CPU/GPU; batch sizes and thread counts | vectors/sec, latency, RAM, provider, vector quality |
| vector | Exact NumPy search vs ChromaDB HNSW | Recall@K, p50/p95, build time, memory |
| reranking | Current MiniLM CPU vs ONNX/OpenVINO and quantized variants | NDCG@10, MRR, latency, pairs/sec |
| hybrid | Vector + FTS5 + RRF, optional reranking | Recall@K, MRR, NDCG@10, p50/p95 |
| chunking | Existing versioned profiles and candidate changes | throughput, chunk count, boundary/coverage checks |
| tokenization | Existing lexical-v1 and candidate analyzers | chars/sec, tokens/sec, correctness |
| llm | Local Ollama models and runtime settings | prompt/generation tokens/sec, latency, quality, RAM/GPU/power |
| pipeline | End-to-end ingestion and search | documents/sec, latency, retrieval quality |

## Quick start

See PLAN.md for the evaluation design. Test fixtures live in fixtures/.

The existing service test suite remains:

    python3 -m pytest services -q

Evaluation runners should be independent commands under this directory and must write machine-readable JSON results under an explicitly supplied output directory. They must not be included in the normal unit-test command.

## Result format

Every run should emit JSON with schema version, run ID and UTC timestamp, track and case, git commit and dirty state, host/CPU/RAM/GPU/driver/OS, runtime/library versions, model ID and immutable revision, quantization, provider and actual execution device, warmup and measured iteration counts, batch size and concurrency, raw durations and p50/p95, throughput and quality metrics, and errors/skips/unavailable capabilities.

Never compare results if model revision, corpus, workload, or hardware configuration differs without explicitly identifying that difference.
