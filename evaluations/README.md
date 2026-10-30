# Evaluations

This directory contains repeatable quality, correctness, and performance evaluations for poc-claude-ensemble. It is deliberately separate from service unit tests: unit tests protect contracts; evaluations compare implementations and measure behavior on representative workloads.

## Principles

- Do not change production service behavior just to make an evaluation pass.
- Keep the current implementation as the baseline for every comparison.
- Separate quality from speed. A faster implementation is not an improvement if retrieval quality regresses.
- Record hardware, software versions, model revisions, provider configuration, and corpus revision.
- Use isolated temporary stores and collections. Never benchmark against or mutate live CE data.
- Warm up models before timed runs; report cold-start separately.
- Run repeated trials and report p50/p95 latency, throughput, and sample count.
- Missing optional providers must be reported as unavailable, not silently treated as CPU.
- Keep model artifacts local; evaluations must not download models or access hosted inference APIs.

## Current runners

Run from the repository root. Results are written to `evaluations/results/` by default; pass `--output` to change that location.

| Track | Command | What it measures |
|---|---|---|
| Embedding | `python3 evaluations/embedding/run.py` | Current Nomic ONNX CPU baseline, batch sizes, latency, throughput, vector validity and a small semantic smoke check |
| Vector | `python3 evaluations/vector/run.py` | Exact NumPy vs argpartition vs Chroma HNSW; Recall@K against exact ground truth |
| Reranking | `python3 evaluations/reranking/run.py` | Pinned local MiniLM quality (MRR/NDCG) and batch-size latency/throughput |
| Hybrid | `python3 evaluations/hybrid/run.py` | Production RRF implementation against fixed lexical/vector relevance judgments |
| Chunking | `python3 evaluations/chunking/run.py` | Existing profiles, throughput, coverage, offsets and deterministic output |
| Tokenization | `python3 evaluations/tokenization/run.py` | lexical-v1 throughput, deterministic output and source-offset correctness |
| Local LLM | `python3 evaluations/llm/run_ollama.py --models qwen3.5:2b` | Already-installed Ollama models, prompt/generation rates, response latency and JSON validity |

The vector runner defaults to 1K and 10K vectors. Add `--sizes 1000 10000 100000` for the larger case. Embedding and reranking evaluations require their pinned local model artifacts. The Ollama runner checks `/api/tags` and skips models that are not already installed. No runner downloads models.

The existing service test suite remains:

    python3 -m pytest services -q

Evaluation runners are independent and are not included in the normal unit-test command. Each writes a unique JSON result containing host and git metadata, raw/aggregate timings, configuration, and quality/correctness metrics.

## Not yet implemented

OpenVINO provider comparison for embeddings/reranking, a full end-to-end pipeline benchmark against an isolated temporary deployment, and production-corpus relevance judgments. These require validating the installed runtime/provider and preparing a representative, privacy-safe evaluation corpus. They are intentionally not faked with synthetic timing claims.
