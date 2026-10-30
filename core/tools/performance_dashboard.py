#!/usr/bin/env python3
"""Performance dashboard backed by the canonical cost-tracking log."""
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from cost_tracking.schema import CANONICAL_LOG_PATH, CostRecord

class PerformanceDashboard:
    def __init__(self, log_file=None, hours=24):
        self.log_file = Path(log_file) if log_file else CANONICAL_LOG_PATH
        self.hours = hours
        self.cutoff = datetime.now().astimezone() - timedelta(hours=hours)
        self.data = self._load_logs()

    def _load_logs(self):
        data = []
        if not self.log_file.exists(): return data
        with self.log_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                try:
                    record = CostRecord.from_dict(json.loads(line)); ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                    if ts.tzinfo is None: ts = ts.astimezone()
                    if ts >= self.cutoff: data.append(record)
                except (ValueError, TypeError, json.JSONDecodeError): continue
        return data

    def display(self):
        if not self.data: print(f"No data in last {self.hours} hours"); return
        by_model = defaultdict(lambda: {"count": 0, "cost": 0.0, "tokens": 0})
        for record in self.data:
            stats = by_model[record.model]; stats["count"] += 1; stats["cost"] += record.cost_usd; stats["tokens"] += record.total_tokens
        print("\n" + "=" * 70); print("PERFORMANCE DASHBOARD"); print("=" * 70)
        print(f"Total API Calls: {len(self.data)}"); print(f"Total Cost: \${sum(r.cost_usd for r in self.data):.6f}"); print(f"Total Tokens: {sum(r.total_tokens for r in self.data):,}")
        for model, stats in sorted(by_model.items()): print(f"  {model:30s}: {stats['count']:3d} calls | \${stats['cost']:9.6f} | {stats['tokens']:,} tokens")
        print(f"\nData source: {self.log_file}"); print(f"Time range: Last {self.hours} hours\n")

if __name__ == "__main__":
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    PerformanceDashboard(hours=hours).display()