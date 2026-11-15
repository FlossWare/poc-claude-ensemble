"""
Compression API Module - Production wrapper for WORKER 1 Summarizer

Usage:
    from compression_api import CompressedPrompt, compress_prompt

    result = compress_prompt(
        text="long RH prompt...",
        target_reduction=0.35,
        preserve_code_refs=True
    )

    print(f"Reduced by {result.reduction_percent}%")
    print(f"Semantic loss: {result.semantic_loss}")
    print(f"Compressed: {result.text}")
"""

from dataclasses import dataclass
from typing import Optional
from summarizer import RecursiveSummarizer, TokenEstimator, SummaryStats


@dataclass
class CompressedPrompt:
    """Compression result with metrics"""
    text: str
    original_tokens: int
    compressed_tokens: int
    reduction_percent: float
    semantic_loss: float
    key_facts_preserved: int
    compression_method: str


def compress_prompt(
    text: str,
    target_reduction: float = 0.35,
    preserve_code_refs: bool = True,
    min_semantic_threshold: float = 0.3
) -> CompressedPrompt:
    """
    Compress a prompt/context while preserving semantic meaning.

    Args:
        text: Input text to compress
        target_reduction: Target reduction ratio (0.35 = compress to 65%)
        preserve_code_refs: If True, always keep code references
        min_semantic_threshold: Max acceptable semantic loss (0-1)

    Returns:
        CompressedPrompt with text and metrics
    """
    summarizer = RecursiveSummarizer()
    compressed, stats = summarizer.summarize_with_stats(text, target_reduction)

    # Verify semantic threshold
    if stats.semantic_loss_score > min_semantic_threshold:
        # Fallback: be more conservative
        compressed, stats = summarizer.summarize_with_stats(text, target_reduction * 0.7)

    return CompressedPrompt(
        text=compressed,
        original_tokens=stats.original_tokens,
        compressed_tokens=stats.compressed_tokens,
        reduction_percent=stats.reduction_percent,
        semantic_loss=stats.semantic_loss_score,
        key_facts_preserved=stats.key_facts_preserved,
        compression_method="recursive-hierarchical"
    )


def batch_compress(
    prompts: list[str],
    target_reduction: float = 0.35
) -> list[CompressedPrompt]:
    """Compress multiple prompts"""
    return [compress_prompt(p, target_reduction) for p in prompts]


def measure_effectiveness(results: list[CompressedPrompt]) -> dict:
    """Measure compression effectiveness across batch"""
    avg_reduction = sum(r.reduction_percent for r in results) / len(results)
    avg_loss = sum(r.semantic_loss for r in results) / len(results)
    total_tokens_saved = sum(r.original_tokens - r.compressed_tokens for r in results)

    return {
        "batch_size": len(results),
        "avg_reduction_percent": round(avg_reduction, 1),
        "avg_semantic_loss": round(avg_loss, 3),
        "total_tokens_saved": total_tokens_saved,
        "achieves_target": 30 <= avg_reduction <= 50,
        "semantic_acceptable": avg_loss <= 0.3
    }


if __name__ == "__main__":
    # Example usage
    print("Compression API Module - WORKER 1\n")

    sample = """
    The CPSEARCH-10981 issue involves implementing keyset pagination for Solr queries.
    The problem occurs when AND logic fails at cursor logical boundaries.
    Previous commits (a840f115, b7edae1d) fixed critical blocker issues.
    All 8 concrete component tests and 9 concrete implementations updated.
    The processor test assertion used incorrect Solr bracket syntax, now fixed.
    Test mismatches and logging corrected from code review feedback.
    This affects keyset pagination with AND vs OR logic, similar to cursorMark alternative.
    """

    result = compress_prompt(sample, target_reduction=0.35)

    print(f"Original: {result.original_tokens} tokens")
    print(f"Compressed: {result.compressed_tokens} tokens")
    print(f"Reduction: {result.reduction_percent}%")
    print(f"Semantic Loss: {result.semantic_loss}")
    print(f"\nCompressed Text:\n{result.text}")
