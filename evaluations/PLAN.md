# Evaluation plan

## Goal

Establish a reproducible baseline for every performance-sensitive component in poc-claude-ensemble, then compare alternative libraries, providers, models, and algorithms without changing service contracts or mixing performance results with retrieval-quality results.

## Phase 0: reproducibility and safety

1. Capture git commit and working-tree state, OS/kernel, CPU model, physical RAM, GPU, driver/runtime versions, and power profile.
2. Record exact versions of Python, NumPy, ONNX Runtime, ChromaDB, Sentence Transformers, PyTorch, tokenizers, and Ollama when used.
3. Resolve model artifact paths and revisions from local metadata. A directory name alone does not prove artifact revision.
4. Verify all stores point to temporary evaluation directories.
5. Warm up each runtime, then run at least 5 measured repetitions for microbenchmarks and 3 full repetitions for end-to-end tests.
6. Save raw measurements and summary JSON. Never overwrite previous runs.

## Track 1: embeddings

Baseline: services/embeddings/embed_lib.py, Nomic Embed Text v1.5, ONNX Runtime CPUExecutionProvider.

Compare ONNX Runtime CPU with thread counts 1, 2, 4, and 8; OpenVINO Execution Provider on CPU when installed and selected; OpenVINO Execution Provider on GPU when installed and selected; and candidate ONNX graph variants only when matching artifacts exist. Test batch sizes 1, 4, 8, 16, 32, and 64.

Measure cold model-load time, warm latency, texts/sec, tokens/sec, peak RSS, vector dimension, finite values, L2 norm, and provider/device. Check query/document prefix behavior and semantic ranking on the retrieval fixture. Record provider fallback explicitly.

Do not change production Embedder until the benchmark proves the provider works and quality checks pass. OpenVINO GPU and Ollama Vulkan are separate backends; success in one does not imply the other works.

## Track 2: vector retrieval

Baseline: exact dot-product search in vector_service.py over normalized vectors.

Compare existing exact NumPy, exact NumPy with argpartition top-K selection (same ranking/score semantics, tie handling documented), and ChromaDB HNSW using identical collection contents and query vectors.

Use deterministic synthetic collections at 1K, 10K, and 100K vectors, plus a real evaluation corpus where available. Test top-K values 5, 10, 20, and 50. Measure index build time, query p50/p95, throughput, memory, and Recall@K against exact search. Exact search is the ground truth for ANN recall.

Never use the live ChromaDB directory. Each strategy gets a fresh temporary collection/store.

## Track 3: reranking

Baseline: local cross-encoder/ms-marco-MiniLM-L6-v2 through Sentence Transformers on CPU.

Compare existing PyTorch/Sentence Transformers CPU, ONNX Runtime CPU when an equivalent exported model exists locally, OpenVINO CPU/GPU when supported, and INT8 only when a validated quantized artifact is available.

Use labeled query/document relevance judgments. Report MRR@10 and NDCG@10 before and after reranking, pair throughput, batch-size sensitivity, p50/p95 latency, load time, and peak RSS. A reranker must demonstrate a retrieval-quality gain to justify enabling it in hybrid search.

## Track 4: hybrid retrieval

Compare vector-only, lexical-only, RRF fusion, and RRF plus reranking. Use a fixed corpus and fixed query/relevance set. Report Recall@5/10/20, MRR@10, NDCG@10, p50/p95 latency, and degraded dependency behavior. Keep candidate counts and RRF parameters explicit.

## Track 5: chunking and tokenization

For chunking, compare window-v1, markdown-v1, and message-v1 on fixed Markdown, code, transcript, and long-document fixtures. Measure chars/sec, chunks/sec, chunk count, coverage, overlap, deterministic IDs, and whether headings/code fences remain sensibly associated with chunks.

For tokenization, compare lexical-v1 with candidate implementations on identifiers, paths, UUIDs, Unicode, punctuation, and prose. Measure chars/sec and tokens/sec. Correctness and exact source offsets are hard requirements; speed cannot compensate for changed semantics unless a new analyzer version is explicitly introduced.

## Track 6: local LLMs

Use Ollama as a separately measured inference runtime. Candidate models must be pulled and pinned before benchmarking; the runner itself must never pull models.

Initial candidates, subject to confirming exact tags exist locally:
- qwen3.5:2b baseline
- granite4.2:3b
- qwen3.5:4b
- gemma4:e2b
- gemma4:e4b
- qwen3.5:9b stretch candidate

For each model, record exact model digest, quantization, context length, Ollama processor allocation, runner logs, prompt-eval tokens/sec, generation tokens/sec, time-to-first-token, total latency, peak RSS, GPU utilization, and power when sensors are available.

Use fixed prompts for structured extraction, intent classification, concise summarization, and JSON-schema compliance. Score exact/schema validity and task quality separately from speed. Compare Vulkan GPU against CPU-only only through an isolated, documented configuration; do not infer GPU offload solely from GPU discovery.

## Track 7: end-to-end pipeline

Measure transcript/document ingestion and representative search requests across real REST service boundaries. Report documents/sec, chunks/sec, embeddings/sec, indexing time, search p50/p95, result quality, and per-service contribution. Run cold-start and warm steady-state scenarios. Record failures and partial/degraded responses.

## Acceptance gates

A candidate is eligible to replace a baseline only when it passes correctness and contract tests; reports the intended provider/device without silent fallback; is non-inferior on fixed quality data with thresholds documented per track; shows repeatable performance improvement beyond measurement noise; has acceptable resource use and operational complexity; and includes reproducible commands, artifact revisions, and raw output.

No aggregate score should hide a regression. Publish the metric vector and explain tradeoffs.

## Suggested implementation order

1. Common result schema and host/runtime metadata collector.
2. Embedding benchmark and semantic fixture.
3. Vector exact-vs-HNSW benchmark.
4. Reranking quality/performance benchmark.
5. Hybrid retrieval quality benchmark.
6. Chunking/tokenization throughput benchmarks.
7. Ollama LLM benchmark.
8. End-to-end pipeline benchmark and consolidated report.
