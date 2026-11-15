#!/usr/bin/env python3
"""
Feedback Loops - Verify learnings are improving performance over time
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'memory-service'))

from memory_client import MemoryClient


def analyze_improvement():
    """Analyze if Thompson rankings improve over time"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    perf = client.read('thompson_performance')
    if not perf:
        print("No performance history yet")
        return

    print("\n📈 Thompson Improvement Tracking:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in perf.split('\n') if line.strip()]

        if len(entries) < 2:
            print("Need at least 2 sessions to measure improvement")
            return

        # Compare first vs last
        first = entries[0]
        last = entries[-1]

        first_best = first.get('best_model', 'unknown')
        last_best = last.get('best_model', 'unknown')

        print(f"First session best: {first_best}")
        print(f"Latest session best: {last_best}")
        print(f"Sessions analyzed: {len(entries)}")

        # Check if best model is stable (good sign - confident in ranking)
        best_model_stability = sum(1 for e in entries if e.get('best_model') == last_best) / len(entries)
        print(f"Stability of {last_best}: {best_model_stability:.0%}")

        if best_model_stability > 0.8:
            print("✓ Learning is converging (good)")
        elif best_model_stability > 0.5:
            print("⚠ Learning is oscillating (unstable)")
        else:
            print("✗ Learning hasn't converged yet (needs more data)")

    except Exception as e:
        print(f"Error: {e}")


def analyze_cost_savings():
    """Analyze if Thompson is saving money"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    costs = client.read('cost_patterns')
    if not costs:
        print("No cost history yet")
        return

    print("\n💰 Cost Improvement Tracking:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in costs.split('\n') if line.strip()]

        if len(entries) < 2:
            print("Need at least 2 sessions to measure savings")
            return

        costs_list = [e.get('total_cost', 0) for e in entries]

        first_avg = sum(costs_list[:len(costs_list)//2]) / (len(costs_list)//2 + 1)
        last_avg = sum(costs_list[len(costs_list)//2:]) / (len(costs_list) - len(costs_list)//2 + 1)

        savings = ((first_avg - last_avg) / first_avg) * 100 if first_avg > 0 else 0

        print(f"First half avg: ${first_avg:.2f}")
        print(f"Second half avg: ${last_avg:.2f}")
        print(f"Savings: {savings:.1f}%")

        if savings > 10:
            print("✓ Thompson is reducing costs")
        elif savings > 0:
            print("⚠ Modest savings, needs more optimization")
        else:
            print("✗ Costs are increasing, Thompson may need tuning")

    except Exception as e:
        print(f"Error: {e}")


def main():
    analyze_improvement()
    analyze_cost_savings()


if __name__ == '__main__':
    main()
