"""Caching bridge subprocess entry point.

Reads one JSON request from stdin and writes one JSON response to stdout.
Request data is never interpolated into Python source or a shell command.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "caching"))

from cache_metrics import CacheMetricsTracker


def main():
    data = json.load(sys.stdin)
    operation = data["operation"]

    if operation == "initialize":
        CacheMetricsTracker(data["workflow_name"])
        result = {"status": "initialized", "file": data["metrics_file"]}
    elif operation == "log":
        tracker = CacheMetricsTracker()
        if data["cache_hit"]:
            tracker.log_cached_request(
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                cache_read_tokens=data["cache_tokens"],
            )
        else:
            tracker.log_baseline_request(
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
            )
        result = {"logged": True}
    elif operation == "report":
        report = CacheMetricsTracker(data["workflow_name"]).generate_report()
        result = {
            "cache_hits": report.get("cache_hits", 0),
            "cache_misses": report.get("cache_misses", 0),
            "total_tokens_saved": report.get("total_tokens_saved", 0),
            "total_cost_saved": report.get("total_cost_saved", 0),
        }
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
