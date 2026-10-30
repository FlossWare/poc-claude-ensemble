#!/usr/bin/env python3
"""Cost dashboard backed by the canonical cost-tracking log."""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from cost_tracking.schema import CANONICAL_LOG_PATH, CostRecord

def read_cost_log(log_path: Path = CANONICAL_LOG_PATH) -> list[CostRecord]:
    entries = []
    if not log_path.exists(): return entries
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try: entries.append(CostRecord.from_dict(json.loads(line)))
                except (ValueError, TypeError, json.JSONDecodeError): print("Warning: skipping malformed cost record", file=sys.stderr)
    return entries

def aggregate(entries, key):
    result = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for entry in entries: result[key(entry)]["calls"] += 1; result[key(entry)]["cost"] += entry.cost_usd
    return dict(result)

def time_bucket(timestamp: str) -> str:
    try: dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError: return "unknown"
    return dt.strftime("%Y-%m-%d")

def print_report(entries: list[CostRecord]) -> None:
    if not entries: print("No cost entries found yet."); return
    total = sum(e.cost_usd for e in entries)
    print("\n" + "=" * 80); print("  COST DASHBOARD"); print("=" * 80)
    print(f"\nTotal Cost: \${total:.4f}"); print(f"Total Calls: {len(entries)}"); print(f"Avg Cost/Call: \${total / len(entries):.6f}")
    groups = [("COST BY MODEL", lambda e: e.model), ("COST BY PROVIDER", lambda e: e.provider), ("COST BY WORKFLOW", lambda e: e.metadata.get("workflow_id", "default"))]
    for title, key in groups:
        print("\n" + "-" * 80); print(title); print("-" * 80)
        for name, value in sorted(aggregate(entries, key).items()): print(f"{name:40s} Calls: {value['calls']:4d} Cost: \${value['cost']:10.4f}")
    daily = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for entry in entries: daily[time_bucket(entry.timestamp)]["calls"] += 1; daily[time_bucket(entry.timestamp)]["cost"] += entry.cost_usd
    print("\n" + "-" * 80); print("COST BY DAY"); print("-" * 80)
    for name, value in sorted(daily.items()): print(f"{name} Calls: {value['calls']:4d} Cost: \${value['cost']:10.4f}")

if __name__ == "__main__": print_report(read_cost_log())