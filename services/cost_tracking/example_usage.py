"""
Example usage of the CostAggregator module.

Demonstrates common patterns for:
- Loading and querying cost data
- Generating reports
- Analyzing compression and cache metrics
- Tracking costs by model
- Exporting data
"""

from aggregator import CostAggregator
from datetime import datetime
import json


def example_1_basic_usage():
    """Example 1: Load existing data and generate basic reports."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    agg = CostAggregator()
    stats = agg.get_summary_stats()
    print(f"\nTotal Cost Entries: {stats['total_entries']}")
    print(f"Total Cost: ${stats['total_cost_usd']:.2f}")
    print(f"Models Used: {', '.join(stats['models_used'])}")


def example_2_daily_breakdown():
    """Example 2: Analyze daily costs by model."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Daily Cost Breakdown")
    print("="*80)

    agg = CostAggregator()
    daily = agg.daily_summary()

    recent_day = max(daily['daily_summaries'].keys())
    today = daily['daily_summaries'][recent_day]

    print(f"\nDate: {recent_day}")
    print(f"Total Cost: ${today['total_cost']:.4f}")
    print(f"Total Calls: {today['total_calls']}")

    print("\nCost Breakdown by Model:")
    for model, data in sorted(today['models'].items(),
                              key=lambda x: x[1]['total_cost'],
                              reverse=True):
        print(f"  {model}: ${data['total_cost']:.4f}  ({data['calls']} calls)")


def example_3_savings_analysis():
    """Example 3: Comprehensive savings report."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Savings Analysis")
    print("="*80)

    agg = CostAggregator()
    savings = agg.savings_report()

    cost_summary = savings['cost_summary']

    print(f"\nCost Summary:")
    print(f"  Actual Cost:         ${cost_summary['total_actual_cost_usd']:.6f}")
    print(f"  Uncompressed:        ${cost_summary['estimated_uncompressed_cost_usd']:.6f}")
    print(f"  Compression Savings: ${cost_summary['compression_savings_usd']:.6f}")
    print(f"  Cache Hit Savings:   ${cost_summary['cache_hit_savings_usd']:.6f}")
    print(f"  Total Savings:       ${cost_summary['total_savings_usd']:.6f}")
    print(f"  Savings Rate:        {cost_summary['savings_percentage']:.2f}%")


def example_4_model_comparison():
    """Example 4: Compare costs across models."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Cost Comparison")
    print("="*80)

    agg = CostAggregator()
    savings = agg.savings_report()

    print(f"\n{'Model':<30s} {'Cost':>12s} {'Calls':>8s} {'Cache Hit':>12s}")
    print("─" * 65)

    for model in sorted(savings['model_breakdown'].keys(),
                       key=lambda m: savings['model_breakdown'][m]['total_cost_usd'],
                       reverse=True):
        data = savings['model_breakdown'][model]
        print(f"{model:<30s} ${data['total_cost_usd']:>10.4f} {data['calls']:>8d} "
              f"{data['cache_hit_rate']:>11.1%}")


def example_5_weekly_trends():
    """Example 5: Analyze weekly spending trends."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Weekly Spending Trends")
    print("="*80)

    agg = CostAggregator()
    weekly = agg.weekly_summary()

    print(f"\n{'Week':<15s} {'Total Cost':>15s} {'Calls':>10s}")
    print("─" * 45)

    total_cost = 0
    for week in sorted(weekly['weekly_summaries'].keys()):
        data = weekly['weekly_summaries'][week]
        total_cost += data['total_cost']
        print(f"{week:<15s} ${data['total_cost']:>13.4f} {data['total_calls']:>10d}")

    print("─" * 45)
    print(f"{'Total':<15s} ${total_cost:>13.4f}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("COST AGGREGATOR - USAGE EXAMPLES")
    print("="*80)

    try:
        example_1_basic_usage()
        example_2_daily_breakdown()
        example_3_savings_analysis()
        example_4_model_comparison()
        example_5_weekly_trends()

        print("\n" + "="*80)
        print("EXAMPLES COMPLETED")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
