#!/usr/bin/env python3
"""
Memory Analytics - Query and analyze captured memory categories
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'memory-service'))

from memory_client import MemoryClient


def show_cost_trends():
    """Show spending trends"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    costs = client.read('cost_patterns')
    if not costs:
        print("No cost data yet")
        return

    print("\n💰 Cost Patterns:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in costs.split('\n') if line.strip()]
        total = sum(e.get('total_cost', 0) for e in entries)
        avg = total / len(entries) if entries else 0

        print(f"Sessions recorded: {len(entries)}")
        print(f"Total spent: ${total:.2f}")
        print(f"Average per session: ${avg:.2f}")

        if entries:
            last = entries[-1]
            print(f"\nLatest session:")
            print(f"  Cost: ${last.get('total_cost', 0):.2f}")
            print(f"  Models: {', '.join(last.get('models_used', []))}")
    except:
        pass


def show_thompson_stats():
    """Show model performance rankings"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    perf = client.read('thompson_performance')
    if not perf:
        print("No Thompson performance data yet")
        return

    print("\n🎯 Thompson Model Rankings:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in perf.split('\n') if line.strip()]

        if entries:
            latest = entries[-1]
            scores = latest.get('model_scores', {})

            if scores:
                sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for i, (model, score) in enumerate(sorted_models[:5], 1):
                    print(f"  {i}. {model}: {score:.3f}")

            print(f"\nBest performing: {latest.get('best_model', 'unknown')}")
    except:
        pass


def show_learning_progress():
    """Show autonomous learning progress"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    learning = client.read('learning_dataset')
    if not learning:
        print("No learning data yet")
        return

    print("\n🧠 Autonomous Learning Progress:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in learning.split('\n') if line.strip()]

        if entries:
            latest = entries[-1]
            print(f"Outcomes analyzed: {latest.get('outcome_count', 0)}")
            print(f"Average quality: {latest.get('average_quality', 0):.2%}")

            models = latest.get('models_analyzed', [])
            if models:
                print(f"Models studied: {', '.join(models)}")
    except:
        pass


def show_integration_health():
    """Show service health status"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    status = client.read('integration_status')
    if not status:
        print("No integration data yet")
        return

    print("\n🔧 Integration Status:")
    print("─" * 70)

    try:
        entries = [json.loads(line) for line in status.split('\n') if line.strip()]

        if entries:
            latest = entries[-1]
            services = latest.get('service_status', {})

            for service, state in services.items():
                symbol = '✓' if state == 'active' else '✗'
                print(f"  {symbol} {service}: {state}")
    except:
        pass


def main():
    if len(sys.argv) < 2:
        print("Memory Analytics - View captured session data")
        print()
        print("Usage:")
        print("  memory-analytics.py costs    - Show spending trends")
        print("  memory-analytics.py thompson - Show model performance")
        print("  memory-analytics.py learning - Show learning progress")
        print("  memory-analytics.py health   - Show service health")
        print("  memory-analytics.py all      - Show everything")
        return 1

    command = sys.argv[1]

    if command == 'costs':
        show_cost_trends()
    elif command == 'thompson':
        show_thompson_stats()
    elif command == 'learning':
        show_learning_progress()
    elif command == 'health':
        show_integration_health()
    elif command == 'all':
        show_cost_trends()
        show_thompson_stats()
        show_learning_progress()
        show_integration_health()
    else:
        print(f"Unknown command: {command}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
