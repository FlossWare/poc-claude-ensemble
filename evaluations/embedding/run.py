"""Benchmark the deployed Embedder without changing production service behavior.

Run from the repository root: python3 evaluations/embedding/run.py --help
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import package_versions, summarize_ms, write_result

TEXTS = [
    "How do I renew the TLS certificate on the load balancer?",
    "The JSON parser rejects trailing commas and reports the offending byte offset.",
    "The vector cache doubles its NumPy buffer capacity when the number of rows reaches the current allocation.",
    "Chunk IDs include the profile name and version, source ID, and character offsets.",
    "SQLite FTS5 provides lexical full-text search over indexed transcript text.",
    "The embedding service converts query and document text into normalized vectors.",
    "OpenVINO can execute compatible ONNX graphs on supported CPU and GPU devices.",
    "Reciprocal rank fusion combines ranked lists without requiring score calibration.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="model.onnx", help="ONNX graph filename under the model onnx directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="evaluations/results")
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 0 or args.iterations < 1 or min(args.batch_sizes) < 1:
        parser.error("threads and batch sizes must be positive; warmups must be non-negative; iterations must be positive")

    from embed_lib import Embedder, MODEL_DIR, MODEL_ID, MODEL_REVISION, DIMENSIONS
    model_path = Path(MODEL_DIR) / "onnx" / args.variant
    if not model_path.is_file():
        parser.error(f"local model graph not found: {model_path}; evaluations never download models")
    start = time.perf_counter()
    embedder = Embedder(variant=args.variant, threads=args.threads)
    load_ms = (time.perf_counter() - start) * 1000
    cases = []
    for batch_size in args.batch_sizes:
        texts = (TEXTS * ((batch_size + len(TEXTS) - 1) // len(TEXTS)))[:batch_size]
        fn = embedder.embed_documents
        for _ in range(args.warmups):
            fn(texts)
        samples = []
        vectors = None
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            vectors = fn(texts)
            samples.append((time.perf_counter() - t0) * 1000)
        norms = __import__("numpy").linalg.norm(vectors, axis=1)
        cases.append({"batch_size": batch_size, "timing": summarize_ms(samples),
                      "texts_per_second": batch_size / (sum(samples) / 1000 / len(samples)),
                      "dimensions": int(vectors.shape[1]), "finite": bool(__import__("numpy").isfinite(vectors).all()),
                      "mean_l2_norm": float(norms.mean()), "min_l2_norm": float(norms.min()),
                      "max_l2_norm": float(norms.max())})

    # Small paired semantic sanity check: matching query/document text should outrank unrelated fixtures.
    query_vectors = embedder.embed_queries(TEXTS[:4])
    doc_vectors = embedder.embed_documents(TEXTS)
    similarities = query_vectors @ doc_vectors.T
    correct_top1 = sum(int(similarities[i].argmax() == i) for i in range(4))
    result = write_result("embedding", {"model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                      "directory": MODEL_DIR, "graph": str(model_path), "dimensions": DIMENSIONS},
                      "runtime": {"providers": embedder.session.get_providers(), "threads": args.threads,
                                  "packages": package_versions(["numpy", "onnxruntime", "tokenizers"])},
                      "load_ms": load_ms, "warmups": args.warmups, "iterations": args.iterations,
                      "batch_cases": cases, "semantic_smoke": {"queries": 4, "correct_top1": correct_top1,
                      "similarity_matrix": similarities.tolist()}}, args.output)
    print(result)

if __name__ == "__main__":
    main()
