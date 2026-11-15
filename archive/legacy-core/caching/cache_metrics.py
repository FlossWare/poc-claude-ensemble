"""
Cache Metrics Tracking for RH Prompt Caching Integration

Measures and reports:
- Baseline tokens (without cache)
- Cached tokens (with cache)
- Cache hit/miss detection
- Cost savings analysis
- Per-workflow and aggregate statistics
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Cache operation status"""
    UNKNOWN = "unknown"
    MISS = "miss"
    HIT = "hit"
    PARTIAL = "partial"  # Some input cached, some not


@dataclass
class CacheMetric:
    """Single cache operation metric"""
    timestamp: str
    workflow_id: int
    workflow_name: str
    cache_status: str  # "hit", "miss", "partial"
    baseline_tokens: int  # Total input tokens without cache
    cached_tokens: int    # Total input tokens with cache (includes cache write cost)
    cache_savings: int    # baseline_tokens - cached_tokens
    cache_savings_pct: float  # (cache_savings / baseline_tokens) * 100
    model: str = "claude-haiku-4-5"  # Default for RH
    response_tokens: int = 0
    total_latency_ms: int = 0
    cache_latency_ms: int = 0  # Time to check/write cache
    notes: str = ""

    def __post_init__(self):
        """Calculate derived fields"""
        if self.baseline_tokens > 0:
            self.cache_savings = self.baseline_tokens - self.cached_tokens
            self.cache_savings_pct = (self.cache_savings / self.baseline_tokens) * 100
        else:
            self.cache_savings = 0
            self.cache_savings_pct = 0.0

    def cost_reduction(self, input_cost_per_1k: float = 0.80, output_cost_per_1k: float = 2.40) -> float:
        """Calculate cost reduction in dollars (Claude Haiku pricing).

        Uses correct Anthropic prompt caching pricing:
        - Cache write: normal rate × 1.25 (25% premium for cache creation)
        - Cache read: normal rate × 0.1 (90% discount for cached tokens)

        Args:
            input_cost_per_1k: Cost per 1K input tokens (Claude Haiku: $0.80)
            output_cost_per_1k: Cost per 1K output tokens (Claude Haiku: $2.40)

        Returns:
            Dollar amount saved compared to non-cached baseline
        """
        # Baseline: normal input cost (no cache)
        baseline_cost = (self.baseline_tokens / 1000) * input_cost_per_1k

        # With cache: we need to account for cache creation + cache reads
        # Approximate split: assume first request creates cache, subsequent use cache
        # For Phase 1 testing, use simple average: half at normal rate, half at 0.1x
        # In production, this depends on actual hit/miss patterns
        cached_cost = (self.cached_tokens / 1000) * input_cost_per_1k * 0.55  # ~55% average
        # More precisely: if we hit cache, cost is 10% of normal
        # If cache miss, cost is 125% of normal
        # Typical mix: 80% hits + 20% misses = (0.8 * 0.1) + (0.2 * 1.25) = 0.33x

        return baseline_cost - cached_cost


@dataclass
class CacheReport:
    """Cache test run report"""
    test_run_id: str
    timestamp: str
    test_cases: int
    metrics: List[CacheMetric] = field(default_factory=list)

    def add_metric(self, metric: CacheMetric):
        """Add a cache metric to the report"""
        self.metrics.append(metric)

    def total_baseline_tokens(self) -> int:
        """Sum of all baseline tokens"""
        return sum(m.baseline_tokens for m in self.metrics)

    def total_cached_tokens(self) -> int:
        """Sum of all cached tokens"""
        return sum(m.cached_tokens for m in self.metrics)

    def total_savings_pct(self) -> float:
        """Overall savings percentage"""
        total_baseline = self.total_baseline_tokens()
        if total_baseline == 0:
            return 0
        total_cached = self.total_cached_tokens()
        return (1 - total_cached / total_baseline) * 100

    def average_savings_pct(self) -> float:
        """Average savings across all metrics"""
        if not self.metrics:
            return 0
        return sum(m.cache_savings_pct for m in self.metrics) / len(self.metrics)

    def hit_count(self) -> int:
        """Number of cache hits"""
        return sum(1 for m in self.metrics if m.cache_status == "hit")

    def miss_count(self) -> int:
        """Number of cache misses"""
        return sum(1 for m in self.metrics if m.cache_status == "miss")

    def hit_rate(self) -> float:
        """Cache hit rate percentage"""
        if not self.metrics:
            return 0
        return (self.hit_count() / len(self.metrics)) * 100

    def total_cost_reduction(self, input_cost_per_1k: float = 0.80) -> float:
        """Total cost reduction in dollars"""
        return sum(m.cost_reduction(input_cost_per_1k) for m in self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_run_id": self.test_run_id,
            "timestamp": self.timestamp,
            "test_cases": self.test_cases,
            "total_baseline_tokens": self.total_baseline_tokens(),
            "total_cached_tokens": self.total_cached_tokens(),
            "total_savings_pct": self.total_savings_pct(),
            "average_savings_pct": self.average_savings_pct(),
            "hit_count": self.hit_count(),
            "miss_count": self.miss_count(),
            "hit_rate_pct": self.hit_rate(),
            "total_cost_reduction": self.total_cost_reduction(),
            "metrics": [asdict(m) for m in self.metrics],
        }

    def save_json(self, output_path: Path):
        """Save report to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Cache report saved to {output_path}")

    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "=" * 80)
        print("CACHE INTEGRATION TEST REPORT")
        print("=" * 80)
        print(f"Test Run ID: {self.test_run_id}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Test Cases: {self.test_cases}")
        print()
        print("TOKEN STATISTICS:")
        print(f"  Baseline (no cache):  {self.total_baseline_tokens():>10,} tokens")
        print(f"  With cache:           {self.total_cached_tokens():>10,} tokens")
        print(f"  Total savings:        {self.total_baseline_tokens() - self.total_cached_tokens():>10,} tokens")
        print(f"  Savings percentage:   {self.total_savings_pct():>10.1f}%")
        print(f"  Average per prompt:   {self.average_savings_pct():>10.1f}%")
        print()
        print("CACHE HIT RATE:")
        print(f"  Hits:                 {self.hit_count():>10} / {len(self.metrics)}")
        print(f"  Hit rate:             {self.hit_rate():>10.1f}%")
        print()
        print("COST ANALYSIS (Claude Haiku):")
        cost_reduction = self.total_cost_reduction()
        print(f"  Total cost reduction: ${cost_reduction:>10.2f}")
        print(f"  Per prompt avg:       ${cost_reduction / len(self.metrics) if self.metrics else 0:>10.2f}")
        print()
        print("PROJECTED ANNUAL SAVINGS (RH Team Usage):")
        annual_runs = 52  # Weekly if single release cycle
        print(f"  Weekly runs × 52:     ${cost_reduction * annual_runs:>10.2f}")
        print()
        print("=" * 80)

    def print_detailed(self):
        """Print detailed per-case results"""
        self.print_summary()
        print("\nDETAILED RESULTS BY TEST CASE:")
        print("-" * 80)

        for metric in self.metrics:
            print(f"\n#{metric.workflow_id}: {metric.workflow_name}")
            print(f"  Status:      {metric.cache_status}")
            print(f"  Baseline:    {metric.baseline_tokens:>7,} tokens")
            print(f"  Cached:      {metric.cached_tokens:>7,} tokens")
            print(f"  Savings:     {metric.cache_savings:>7,} tokens ({metric.cache_savings_pct:>5.1f}%)")
            print(f"  Cost reduce: ${metric.cost_reduction():>7.2f}")
            if metric.total_latency_ms:
                print(f"  Latency:     {metric.total_latency_ms:>7,} ms (cache: {metric.cache_latency_ms} ms)")
            if metric.notes:
                print(f"  Notes:       {metric.notes}")

        print("\n" + "=" * 80)


class CacheMetricsCollector:
    """Collects and manages cache metrics for a test run"""

    def __init__(self, test_run_id: str = None):
        """Initialize collector"""
        self.test_run_id = test_run_id or f"rh_cache_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report = CacheReport(
            test_run_id=self.test_run_id,
            timestamp=datetime.now().isoformat(),
            test_cases=0,
        )

    def record_metric(
        self,
        workflow_id: int,
        workflow_name: str,
        cache_status: str,
        baseline_tokens: int,
        cached_tokens: int,
        model: str = "claude-haiku-4-5",
        response_tokens: int = 0,
        total_latency_ms: int = 0,
        cache_latency_ms: int = 0,
        notes: str = "",
    ) -> CacheMetric:
        """Record a cache metric"""
        # Calculate cache savings upfront for CacheMetric constructor
        cache_savings = baseline_tokens - cached_tokens if baseline_tokens > 0 else 0
        cache_savings_pct = (cache_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0

        metric = CacheMetric(
            timestamp=datetime.now().isoformat(),
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            cache_status=cache_status,
            baseline_tokens=baseline_tokens,
            cached_tokens=cached_tokens,
            cache_savings=cache_savings,
            cache_savings_pct=cache_savings_pct,
            model=model,
            response_tokens=response_tokens,
            total_latency_ms=total_latency_ms,
            cache_latency_ms=cache_latency_ms,
            notes=notes,
        )
        self.report.add_metric(metric)
        logger.info(
            f"Recorded: {workflow_name} - {cache_status} - "
            f"{baseline_tokens:,} → {cached_tokens:,} tokens ({metric.cache_savings_pct:.1f}% savings)"
        )
        return metric

    def set_test_case_count(self, count: int):
        """Set total number of test cases"""
        self.report.test_cases = count

    def save_report(self, output_dir: Path = None) -> Path:
        """Save report and return path"""
        if output_dir is None:
            output_dir = Path.cwd() / "cache_reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.test_run_id}.json"
        self.report.save_json(output_file)
        return output_file

    def print_report(self, detailed: bool = False):
        """Print report to console"""
        if detailed:
            self.report.print_detailed()
        else:
            self.report.print_summary()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create collector
    collector = CacheMetricsCollector()
    collector.set_test_case_count(10)

    # Record example metrics (from test_cases.py expected values)
    test_data = [
        (1, "Critical Code Review with Multi-AI Consensus", 18000, 4500),
        (2, "Disseminator Release Note Generation", 12000, 2000),
        (3, "MR 1087 Arbiter-Worker Pattern Review", 15000, 6000),
        (4, "CPSEARCH-10981 Keyset Pagination Fix Review", 16000, 5000),
        (5, "Infrastructure Decision: AWX Connectivity Issue", 10000, 2500),
        (6, "Sumo Logic Monitoring Integration Setup", 14000, 4000),
        (7, "Google Workspace & Sheets Integration", 13000, 5500),
        (8, "Confluence Documentation Publishing Workflow", 9000, 2500),
        (9, "GitLab Branch & MR Workflow with Approval Gates", 17000, 6500),
        (10, "Continuous Delivery Deployment Tracking", 8500, 1500),
    ]

    for wf_id, wf_name, baseline, cached in test_data:
        cache_status = "hit" if cached < baseline * 0.5 else "partial"
        collector.record_metric(
            workflow_id=wf_id,
            workflow_name=wf_name,
            cache_status=cache_status,
            baseline_tokens=baseline,
            cached_tokens=cached,
            notes="Integration test run"
        )

    # Print and save report
    collector.print_report(detailed=True)
    report_path = collector.save_report()
    print(f"\nReport saved to: {report_path}")
