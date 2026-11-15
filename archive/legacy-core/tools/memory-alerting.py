#!/usr/bin/env python3
"""
Memory Alerting - Detect anomalies and service issues
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import math

sys.path.insert(0, str(Path(__file__).parent.parent / 'memory-service'))

from memory_client import MemoryClient


def check_cost_anomalies():
    """Alert on unusual spending"""
    client = MemoryClient()
    if not client.connect():
        return

    costs = client.read('cost_patterns')
    if not costs:
        return

    try:
        entries = [json.loads(line) for line in costs.split('\n') if line.strip()]

        if len(entries) < 3:
            return  # Need baseline

        costs_list = [e.get('total_cost', 0) for e in entries]

        # Calculate mean and std dev
        mean = sum(costs_list) / len(costs_list)
        variance = sum((x - mean) ** 2 for x in costs_list) / len(costs_list)
        std_dev = math.sqrt(variance)

        # Check latest
        latest = costs_list[-1]

        if latest > mean + (2 * std_dev):
            print(f"⚠️  ALERT: Cost spike detected!")
            print(f"   Current: ${latest:.2f} (expected: ${mean:.2f} ± ${std_dev:.2f})")
            print(f"   Deviation: {((latest - mean) / mean * 100):.1f}% above normal")
            return True

    except:
        pass

    return False


def check_service_health():
    """Alert on service failures"""
    client = MemoryClient()
    if not client.connect():
        print("⚠️  ALERT: Memory service disconnected!")
        return True

    status = client.read('integration_status')
    if not status:
        return False

    try:
        entries = [json.loads(line) for line in status.split('\n') if line.strip()]

        if entries:
            latest = entries[-1]
            services = latest.get('service_status', {})

            failed = [s for s, state in services.items() if state != 'active']

            if failed:
                print(f"⚠️  ALERT: Services down!")
                for service in failed:
                    print(f"   - {service}")
                return True

    except:
        pass

    return False


def check_learning_stagnation():
    """Alert if learning isn't improving"""
    client = MemoryClient()
    if not client.connect():
        return

    learning = client.read('learning_dataset')
    if not learning:
        return

    try:
        entries = [json.loads(line) for line in learning.split('\n') if line.strip()]

        if len(entries) < 3:
            return

        # Check if quality is improving
        qualities = [e.get('average_quality', 0) for e in entries]

        # Simple trend: is latest worse than first?
        if qualities[-1] < qualities[0]:
            decline = ((qualities[0] - qualities[-1]) / qualities[0] * 100)
            print(f"⚠️  ALERT: Learning quality declining!")
            print(f"   First: {qualities[0]:.2%} → Latest: {qualities[-1]:.2%}")
            print(f"   Decline: {decline:.1f}%")
            return True

    except:
        pass

    return False


def main():
    alerts = []

    alerts.append(("Cost anomaly", check_cost_anomalies()))
    alerts.append(("Service health", check_service_health()))
    alerts.append(("Learning stagnation", check_learning_stagnation()))

    triggered = [name for name, result in alerts if result]

    if triggered:
        print(f"\n🚨 {len(triggered)} alert(s) triggered")
        return 1
    else:
        print("✓ All systems nominal")
        return 0


if __name__ == '__main__':
    sys.exit(main())
