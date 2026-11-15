#!/usr/bin/env python3
"""
Memory Synthesis - Auto-generate insights from all captured data
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'memory-service'))

from memory_client import MemoryClient


def synthesize_insights():
    """Generate comprehensive insights from all memory categories"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service")
        return

    print("\n🧠 Session Insights Summary")
    print("=" * 70)

    # Learnings
    learnings = client.read('session_learnings')
    if learnings:
        try:
            entries = [json.loads(line) for line in learnings.split('\n') if line.strip()]
            print(f"\n📚 Learnings Captured: {len(entries)}")
            for e in entries[-3:]:  # Last 3
                print(f"   • {e.get('title', 'learning')}")
        except:
            pass

    # Costs
    costs = client.read('cost_patterns')
    if costs:
        try:
            entries = [json.loads(line) for line in costs.split('\n') if line.strip()]
            if entries:
                latest = entries[-1]
                print(f"\n💰 Spending: ${latest.get('total_cost', 0):.2f}")
                print(f"   Models: {', '.join(latest.get('models_used', []))}")
                print(f"   Avg per call: ${latest.get('avg_cost_per_call', 0):.4f}")
        except:
            pass

    # Thompson
    thompson = client.read('thompson_performance')
    if thompson:
        try:
            entries = [json.loads(line) for line in thompson.split('\n') if line.strip()]
            if entries:
                latest = entries[-1]
                print(f"\n🎯 Model Performance:")
                print(f"   Best: {latest.get('best_model', 'unknown')}")
                scores = latest.get('model_scores', {})
                if scores:
                    for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"   • {model}: {score:.3f}")
        except:
            pass

    # Learning
    learning = client.read('learning_dataset')
    if learning:
        try:
            entries = [json.loads(line) for line in learning.split('\n') if line.strip()]
            if entries:
                latest = entries[-1]
                print(f"\n🧬 Autonomous Learning:")
                print(f"   Outcomes: {latest.get('outcome_count', 0)}")
                print(f"   Quality: {latest.get('average_quality', 0):.2%}")
                models = latest.get('models_analyzed', [])
                if models:
                    print(f"   Analyzed: {', '.join(models)}")
        except:
            pass

    # Architecture
    arch = client.read('architecture_decisions')
    if arch:
        try:
            entries = [json.loads(line) for line in arch.split('\n') if line.strip()]
            if entries:
                print(f"\n🏗️  Architecture Decisions: {len(entries)}")
                for e in entries[-2:]:
                    print(f"   • {e.get('decision', 'decision')[:60]}")
        except:
            pass

    # Integration
    integration = client.read('integration_status')
    if integration:
        try:
            entries = [json.loads(line) for line in integration.split('\n') if line.strip()]
            if entries:
                latest = entries[-1]
                status = latest.get('service_status', {})
                active = sum(1 for v in status.values() if v == 'active')
                print(f"\n🔧 Services: {active}/{len(status)} active")
        except:
            pass

    print("\n" + "=" * 70)
    print("✓ Synthesis complete")


if __name__ == '__main__':
    synthesize_insights()
