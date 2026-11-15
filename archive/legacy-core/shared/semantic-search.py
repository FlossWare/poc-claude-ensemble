#!/usr/bin/env python3
"""
Semantic Search Prototype - Test advanced search concepts

Borrows from semantic-search-ai:
- Hybrid search (semantic + keyword + RRF)
- Reranking (bi-encoder → cross-encoder)
- Advanced filtering (MongoDB-style operators)

Usage:
    from semantic_search import HybridSearch, Reranker, AdvancedFilter

    # Hybrid search
    hybrid = HybridSearch()
    results = hybrid.merge(vector_results, keyword_results, top_k=10)

    # Reranking
    reranker = Reranker()
    top = reranker.rerank(query, candidates, top_k=5)

    # Advanced filtering
    filtered = AdvancedFilter.apply(results, {"type": {"$eq": "feedback"}})
"""

from typing import List, Dict, Any, Optional
import re


class HybridSearch:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF)

    Combines semantic (vector) and keyword (text) search results.
    """

    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3, k: int = 60):
        """
        Initialize hybrid search

        Args:
            semantic_weight: Weight for semantic results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            k: RRF parameter (default 60, from original paper)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.k = k

    def merge(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge semantic and keyword results using RRF

        Args:
            semantic_results: Results from vector search (with 'id' and 'score')
            keyword_results: Results from keyword search (with 'id' and 'score')
            top_k: Number of results to return

        Returns:
            Merged and ranked results
        """
        # RRF formula: score = sum(1 / (k + rank))
        scores = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result.get('id')
            rrf_score = self.semantic_weight / (self.k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get('id')
            rrf_score = self.keyword_weight / (self.k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Combine all results by ID
        all_docs = {}
        for result in semantic_results + keyword_results:
            doc_id = result.get('id')
            if doc_id not in all_docs:
                all_docs[doc_id] = result

        # Sort by RRF score
        merged = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            doc = all_docs[doc_id].copy()
            doc['hybrid_score'] = score
            merged.append(doc)

        return merged


class Reranker:
    """
    Two-stage retrieval: bi-encoder → cross-encoder

    Stage 1 (bi-encoder): Fast vector search, get many candidates
    Stage 2 (cross-encoder): Accurate reranking, get top results
    """

    def __init__(self, model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize reranker

        Args:
            model: Cross-encoder model for reranking
        """
        self.model_name = model
        self.model = None  # Lazy load

    def _load_model(self):
        """Lazy load cross-encoder model"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed, using mock scoring")
                self.model = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder

        Args:
            query: Query text
            candidates: Candidate documents (from bi-encoder search)
            top_k: Number of top results to return

        Returns:
            Reranked top results
        """
        self._load_model()

        if not candidates:
            return []

        # If model not available, return top candidates as-is
        if self.model is False:
            return candidates[:top_k]

        # Extract texts for scoring
        texts = [c.get('document', c.get('text', '')) for c in candidates]

        # Score with cross-encoder
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        # Combine and sort
        reranked = []
        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result['rerank_score'] = float(score)
            reranked.append(result)

        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]


class AdvancedFilter:
    """
    MongoDB-style filtering for search results

    Supports:
    - Comparison: $eq, $ne, $gt, $gte, $lt, $lte
    - Logical: $and, $or, $not
    - Array: $in, $nin
    - String: $regex, $glob
    """

    @staticmethod
    def apply(results: List[Dict[str, Any]], filter_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply filter query to results

        Args:
            results: List of result documents
            filter_query: MongoDB-style filter

        Returns:
            Filtered results

        Examples:
            # Exact match
            filter = {"type": "feedback"}
            filter = {"type": {"$eq": "feedback"}}

            # Comparison
            filter = {"score": {"$gte": 0.8}}

            # Logical
            filter = {"$and": [{"type": "feedback"}, {"score": {"$gte": 0.8}}]}

            # Array
            filter = {"type": {"$in": ["feedback", "user"]}}

            # Regex
            filter = {"name": {"$regex": "^multi-"}}
        """
        filtered = []
        for result in results:
            if AdvancedFilter._match(result.get('metadata', {}), filter_query):
                filtered.append(result)
        return filtered

    @staticmethod
    def _match(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if document matches query"""
        for key, value in query.items():
            # Logical operators
            if key == '$and':
                if not all(AdvancedFilter._match(doc, q) for q in value):
                    return False
            elif key == '$or':
                if not any(AdvancedFilter._match(doc, q) for q in value):
                    return False
            elif key == '$not':
                if AdvancedFilter._match(doc, value):
                    return False

            # Field operators
            else:
                field_value = doc.get(key)
                if isinstance(value, dict):
                    # Operator-based comparison
                    for op, op_value in value.items():
                        if not AdvancedFilter._match_operator(field_value, op, op_value):
                            return False
                else:
                    # Direct equality
                    if field_value != value:
                        return False

        return True

    @staticmethod
    def _match_operator(field_value: Any, operator: str, op_value: Any) -> bool:
        """Match single operator"""
        if operator == '$eq':
            return field_value == op_value
        elif operator == '$ne':
            return field_value != op_value
        elif operator == '$gt':
            return field_value > op_value
        elif operator == '$gte':
            return field_value >= op_value
        elif operator == '$lt':
            return field_value < op_value
        elif operator == '$lte':
            return field_value <= op_value
        elif operator == '$in':
            return field_value in op_value
        elif operator == '$nin':
            return field_value not in op_value
        elif operator == '$regex':
            return bool(re.search(op_value, str(field_value)))
        elif operator == '$glob':
            import fnmatch
            return fnmatch.fnmatch(str(field_value), op_value)
        else:
            return False


if __name__ == '__main__':
    print("Testing Semantic Search Components\n")

    # Test 1: Hybrid Search
    print("1. Hybrid Search (RRF)")
    hybrid = HybridSearch()

    semantic_results = [
        {'id': 'doc1', 'score': 0.95, 'text': 'Multi-model consensus'},
        {'id': 'doc2', 'score': 0.85, 'text': 'Arbiter pattern'},
        {'id': 'doc3', 'score': 0.75, 'text': 'Worker validation'}
    ]

    keyword_results = [
        {'id': 'doc2', 'score': 0.9, 'text': 'Arbiter pattern'},
        {'id': 'doc4', 'score': 0.8, 'text': 'Consensus voting'},
        {'id': 'doc1', 'score': 0.7, 'text': 'Multi-model consensus'}
    ]

    merged = hybrid.merge(semantic_results, keyword_results, top_k=3)
    print(f"  Merged {len(merged)} results:")
    for r in merged:
        print(f"    {r['id']}: {r['hybrid_score']:.3f} - {r['text']}")

    # Test 2: Advanced Filtering
    print("\n2. Advanced Filtering")
    results = [
        {'id': '1', 'metadata': {'type': 'feedback', 'score': 0.9}},
        {'id': '2', 'metadata': {'type': 'user', 'score': 0.7}},
        {'id': '3', 'metadata': {'type': 'feedback', 'score': 0.6}}
    ]

    filtered = AdvancedFilter.apply(results, {"type": "feedback", "score": {"$gte": 0.8}})
    print(f"  Filtered {len(filtered)} results (type=feedback, score>=0.8):")
    for r in filtered:
        print(f"    {r['id']}: {r['metadata']}")

    print("\n✓ Tests complete!")
