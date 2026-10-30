"""Compare exact NumPy search, argpartition top-K, and Chroma HNSW.

Uses synthetic normalized vectors and a temporary Chroma store only. The exact result is
used as ground truth for ANN recall. Run: python3 evaluations/vector/run.py --help
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import package_versions, summarize_ms, write_result


def exact(matrix, query, k):
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]


def partial(matrix, query, k):
    scores = matrix @ query
    if k >= len(scores):
        candidates = np.arange(len(scores))
    else:
        candidates = np.argpartition(-scores, k - 1)[:k]
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order, scores[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000], help="include 100000 for the larger run")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", default="evaluations/results")
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.dimensions < 1 or args.queries < 1 or min(args.top_k) < 1:
        parser.error("sizes, dimensions, queries, and top-k values must be positive")
    try:
        import chromadb
    except ImportError:
        parser.error("ChromaDB is required for the HNSW comparison")

    rng = np.random.default_rng(args.seed)
    all_cases = []
    for size in args.sizes:
        matrix = rng.normal(size=(size, args.dimensions)).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        queries = rng.normal(size=(args.queries, args.dimensions)).astype(np.float32)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        with tempfile.TemporaryDirectory(prefix="poc-ce-eval-chroma-") as temp:
            client = chromadb.PersistentClient(path=temp)
            collection = client.create_collection("eval_vectors", metadata={"hnsw:space": "cosine"})
            ids = [f"v{i}" for i in range(size)]
            build_start = time.perf_counter()
            for start in range(0, size, 500):
                end = min(start + 500, size)
                collection.add(ids=ids[start:end], embeddings=matrix[start:end].tolist())
            build_ms = (time.perf_counter() - build_start) * 1000
            for k in args.top_k:
                k = min(k, size)
                exact_times, partial_times, hnsw_times = [], [], []
                recalls, exact_partial_agreement = [], []
                for query in queries:
                    t0 = time.perf_counter(); exact_ids, _ = exact(matrix, query, k); exact_times.append((time.perf_counter()-t0)*1000)
                    t0 = time.perf_counter(); partial_ids, _ = partial(matrix, query, k); partial_times.append((time.perf_counter()-t0)*1000)
                    exact_partial_agreement.append(len(set(exact_ids.tolist()) & set(partial_ids.tolist())) / k)
                    t0 = time.perf_counter()
                    hits = collection.query(query_embeddings=[query.tolist()], n_results=k, include=["distances"])
                    hnsw_times.append((time.perf_counter()-t0)*1000)
                    ann = {int(v[1:]) for v in hits["ids"][0]}
                    recalls.append(len(set(exact_ids.tolist()) & ann) / k)
                all_cases.append({"vectors": size, "dimensions": args.dimensions, "top_k": k,
                    "queries": args.queries, "hnsw_build_ms": build_ms,
                    "exact_numpy": summarize_ms(exact_times), "argpartition": summarize_ms(partial_times),
                    "chroma_hnsw": summarize_ms(hnsw_times), "argpartition_topk_agreement": float(np.mean(exact_partial_agreement)),
                    "hnsw_recall_at_k": float(np.mean(recalls)),
                    "hnsw_recall_min": float(np.min(recalls))})
    result = write_result("vector", {"seed": args.seed, "runtime": {"packages": package_versions(["numpy", "chromadb"])},
                      "cases": all_cases}, args.output)
    print(result)

if __name__ == "__main__":
    main()
