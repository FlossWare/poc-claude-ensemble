#!/usr/bin/env python3
"""
Sync actual API costs from GCP Billing to api_costs.jsonl

Prerequisites:
  1. GCP billing data exported to BigQuery
  2. Service account with BigQuery read access
  3. GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS set

Usage:
  ./sync_from_gcp.py --date 2026-09-26
  ./sync_from_gcp.py --days 7  # Last 7 days
  ./sync_from_gcp.py --month 2026-09
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    from google.cloud import bigquery
except ImportError:
    print("ERROR: google-cloud-bigquery not installed")
    print("Install with: pip install google-cloud-bigquery")
    exit(1)


class GCPBillingSync:
    """Sync costs from GCP Billing to JSONL"""

    # Map GCP SKU patterns to RH models + pricing
    SKU_MAPPING = {
        # Claude Haiku
        "Claude Haiku 4 5 — Input Cache Write": {
            "model": "haiku",
            "token_type": "input_cache_write",
            "rate_per_token": 0.00000125,
        },
        "Claude Haiku 4 5 — Input Cache Read": {
            "model": "haiku",
            "token_type": "input_cache_read",
            "rate_per_token": 0.000000100,
        },
        "Claude Haiku 4 5 — Output Tokens": {
            "model": "haiku",
            "token_type": "output",
            "rate_per_token": 0.000005,
        },
        # Claude Sonnet
        "Claude Sonnet 4.5 — Input Cache Write": {
            "model": "sonnet",
            "token_type": "input_cache_write",
            "rate_per_token": 0.00000375,
        },
        "Claude Sonnet 4.5 — Output Tokens": {
            "model": "sonnet",
            "token_type": "output",
            "rate_per_token": 0.000015,
        },
        # Claude Opus 5
        "Claude Opus 5 — Input Cache Write": {
            "model": "opus",
            "token_type": "input_cache_write",
            "rate_per_token": 0.00000625,
        },
        "Claude Opus 5 — Input Cache Read": {
            "model": "opus",
            "token_type": "input_cache_read",
            "rate_per_token": 0.0000005,
        },
        "Claude Opus 5 — Output Tokens": {
            "model": "opus",
            "token_type": "output",
            "rate_per_token": 0.000025,
        },
    }

    def __init__(self, project_id: str = None):
        """Initialize BigQuery client"""
        if project_id is None:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not set")

        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.log_file = Path(__file__).parent / "api_costs.jsonl"

    def query_costs(self, date_str: str = None, days: int = None, month: str = None):
        """Query GCP billing for Claude API costs"""

        # Determine date range
        if date_str:
            start_date = datetime.strptime(date_str, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)
        elif days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        elif month:
            start_date = datetime.strptime(month, "%Y-%m")
            end_date = (start_date + timedelta(days=32)).replace(day=1)
        else:
            raise ValueError("Specify --date, --days, or --month")

        query = f"""
        SELECT
            TIMESTAMP(DATE(usage_time)) as timestamp,
            sku.description as sku,
            usage.amount as usage_amount,
            cost as cost_usd,
            project.name as project_name
        FROM `{self.project_id}.billing_export.gcp_billing_export_v1_*`
        WHERE
            (_TABLE_SUFFIX BETWEEN
                FORMAT_DATE('%Y%m%d', DATE('{start_date.date()}'))
                AND
                FORMAT_DATE('%Y%m%d', DATE('{(end_date - timedelta(days=1)).date()}'))
            )
            AND (sku.description LIKE '%Claude%'
                 OR sku.description LIKE '%Gemini%'
                 OR sku.description LIKE '%Cursor%')
        ORDER BY usage_time DESC
        """

        print(f"Querying GCP billing from {start_date.date()} to {end_date.date()}...")
        results = self.client.query(query).result()

        return list(results)

    def parse_sku(self, sku_desc: str) -> Optional[Dict]:
        """Parse SKU description to get model + rate info"""
        for sku_pattern, info in self.SKU_MAPPING.items():
            if sku_pattern in sku_desc:
                return info
        return None

    def build_cost_entry(self, row) -> Optional[Dict]:
        """Convert GCP billing row to JSONL cost entry"""
        sku_info = self.parse_sku(row["sku"])

        if not sku_info:
            # Unknown SKU - log but skip
            print(f"  ⚠ Unknown SKU: {row['sku'][:50]}...")
            return None

        model = sku_info["model"]
        token_type = sku_info["token_type"]
        tokens = int(row["usage_amount"])
        cost = float(row["cost_usd"])

        # Categorize by token type
        if token_type == "output":
            input_tokens, output_tokens = 0, tokens
        elif "cache_write" in token_type:
            input_tokens, output_tokens = tokens, 0
        elif "cache_read" in token_type:
            input_tokens, output_tokens = 0, 0  # Cache reads don't cost output
        else:
            input_tokens, output_tokens = tokens, 0

        entry = {
            "timestamp": row["timestamp"].isoformat() + "+00:00",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": tokens,
            "cost_usd": cost,
            "task_name": "gcp_billing_sync",
            "source": "gcp_billing",
            "metadata": {
                "sku": row["sku"],
                "token_type": token_type,
                "project": row["project_name"],
                "source": "Synced from GCP Billing export",
            },
        }

        return entry

    def sync(self, date_str: str = None, days: int = None, month: str = None):
        """Sync costs to JSONL"""
        try:
            rows = self.query_costs(date_str=date_str, days=days, month=month)
        except Exception as e:
            print(f"✗ Query failed: {e}")
            print("\nMake sure:")
            print("  1. GOOGLE_CLOUD_PROJECT env var is set")
            print("  2. GOOGLE_APPLICATION_CREDENTIALS points to service account JSON")
            print("  3. Billing export is enabled in GCP")
            print("  4. Service account has BigQuery read access")
            return

        if not rows:
            print("No results found")
            return

        print(f"Found {len(rows)} billing entries")

        # Parse and append to JSONL
        added = 0
        skipped = 0

        for row in rows:
            entry = self.build_cost_entry(row)
            if entry:
                # Append to JSONL (no duplicates check for now)
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                added += 1
                print(f"  ✓ {entry['model']:10s} {entry['total_tokens']:6d} tokens ${entry['cost_usd']:.6f}")
            else:
                skipped += 1

        print(f"\n✅ Synced {added} entries, skipped {skipped}")
        print(f"   Written to: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync actual costs from GCP Billing to api_costs.jsonl"
    )
    parser.add_argument("--date", help="Sync specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Sync last N days")
    parser.add_argument("--month", help="Sync specific month (YYYY-MM)")
    parser.add_argument("--project", help="GCP project ID (or set GOOGLE_CLOUD_PROJECT)")

    args = parser.parse_args()

    if not (args.date or args.days or args.month):
        parser.print_help()
        print("\nExample:")
        print("  ./sync_from_gcp.py --date 2026-09-26")
        print("  ./sync_from_gcp.py --days 7")
        print("  ./sync_from_gcp.py --month 2026-09")
        exit(1)

    sync = GCPBillingSync(project_id=args.project)
    sync.sync(date_str=args.date, days=args.days, month=args.month)


if __name__ == "__main__":
    main()
