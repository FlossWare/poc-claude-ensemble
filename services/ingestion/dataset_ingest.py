#!/usr/bin/env python3
"""Dataset inventory CLI (PR 1: inventory and classification).

Usage::

    PYTHONPATH=services python3 -m ingestion.dataset_ingest inventory \
        [--catalog-ref master | --catalog-dir PATH] [--probe] [--limit N] \
        [--no-ledger]

    PYTHONPATH=services python3 -m ingestion.dataset_ingest report \
        [--format md|json]

Exit codes: 0 accounting invariant holds, 1 unhandled per-entry errors,
2 invariant violation, 3 budget/limit cap reached.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ingestion.datasets import catalog as catalog_module
from ingestion.datasets import classify, ledger as ledger_module, manifest, probe, report
from ingestion.datasets.config import Config


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _manifest_name(snapshot: "catalog_module.CatalogSnapshot") -> str:
    commit = snapshot.commit
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        commit = snapshot.snapshot_sha256
    return re.sub(r"[^A-Za-z0-9._-]", "_", commit)[:120] + ".jsonl"


def _inventory(args: argparse.Namespace) -> int:
    config = Config.load(
        catalog_dir=args.catalog_dir,
        catalog_ref=args.catalog_ref,
        readme_path=args.readme,
        probe_enabled=True if args.probe else None,
        limit=args.limit,
        ledger_enabled=False if args.no_ledger else None,
        data_root=args.data_root,
    )
    if config.limit is not None and config.limit < 0:
        print("--limit must be non-negative", file=sys.stderr)
        return 1

    started = time.monotonic()
    generated_at = _utcnow()
    snapshot = catalog_module.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=config.limit)

    probe_checked = 0
    if config.probe_enabled:
        targets = [
            row
            for row in build.rows
            if row.get("status") != "failed"
            and row.get("reason_code") != "run_budget_exhausted"
        ]
        probe_results = probe.probe_entries(targets, config)
        probe_checked = len(probe_results)
        for row in targets:
            result = probe_results.get(str(row.get("entry_id")))
            if result is not None:
                classify.classify_row(row, config, result)
        accounting = manifest.check_accounting(build.rows, build.discovered)
    else:
        accounting = manifest.check_accounting(build.rows, build.discovered)

    run_id = manifest.run_id_for(config, snapshot)
    manifest_path = config.data_root / "manifest" / _manifest_name(snapshot)
    changed, manifest_path = manifest.write_manifest(build.rows, manifest_path)
    outcomes_added = manifest.append_outcomes(
        build.rows,
        config.data_root / "outcomes.jsonl",
        run_id=run_id,
        catalog_commit=snapshot.commit,
        retrieved_at=generated_at,
    )

    if not accounting.ok:
        exit_code = 2
    elif build.unhandled > 0:
        exit_code = 1
    elif build.truncated:
        exit_code = 3
    else:
        exit_code = 0

    duration_ms = int((time.monotonic() - started) * 1000)
    report_payload = report.build_report(
        build.rows,
        run_id=run_id,
        snapshot=snapshot,
        accounting=accounting,
        generated_at=generated_at,
        duration_ms=duration_ms,
        exit_code=exit_code,
        manifest_path=str(manifest_path),
        manifest_changed=changed,
        outcomes_added=outcomes_added,
        ledger_result={"enabled": config.ledger_enabled, "posted": False},
        probe_checked=probe_checked,
        probe_enabled=config.probe_enabled,
        limit=config.limit,
        unhandled=build.unhandled,
        truncated=build.truncated,
    )
    event_id = ledger_module.summary_event_id(config, snapshot)
    event_entry = ledger_module.build_summary_entry(
        event_id=event_id,
        run_id=run_id,
        snapshot=snapshot,
        report=report_payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        manifest_changed=changed,
        outcomes_added=outcomes_added,
        probe_checked=probe_checked,
    )
    ledger_result = ledger_module.post_summary(
        config, event_id=event_id, entry=event_entry
    )
    report_payload["ledger"] = ledger_result
    report_path = manifest.write_run_report(report_payload, config.data_root, run_id)

    print(
        "inventory: discovered={discovered} statuses={statuses} "
        "invariant={invariant} exit={exit_code}".format(
            discovered=accounting.discovered,
            statuses=accounting.status_counts,
            invariant="ok" if accounting.ok else "violated",
            exit_code=exit_code,
        )
    )
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return exit_code


def _report(args: argparse.Namespace) -> int:
    config = Config.load(data_root=args.data_root)
    payload = manifest.latest_run(config.data_root)
    if payload is None:
        print(f"no run report found under {config.data_root / 'runs'}", file=sys.stderr)
        return 1
    text = report.render_json(payload) if args.format == "json" else report.render_markdown(payload)
    if args.out:
        manifest.atomic_write(Path(args.out), text.encode("utf-8"))
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.dataset_ingest",
        description="Awesome Public Datasets inventory and classification (PR 1)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="inventory and classify the catalog")
    source = inventory.add_mutually_exclusive_group()
    source.add_argument("--catalog-ref", help="catalog git ref to fetch (default: master)")
    source.add_argument("--catalog-dir", help="local catalog checkout to read")
    inventory.add_argument("--probe", action="store_true", help="run the bounded accessibility probe")
    inventory.add_argument("--limit", type=int, help="classify at most N entries (budget cap)")
    inventory.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    inventory.add_argument("--readme", help="explicit README.rst path for OK/FIXME reconciliation")
    inventory.add_argument("--data-root", help="override the dataset data root")

    report_parser = subparsers.add_parser("report", help="render the latest inventory report")
    report_parser.add_argument("--format", choices=("md", "json"), default="md")
    report_parser.add_argument("--out", help="write the report to a file")
    report_parser.add_argument("--data-root", help="override the dataset data root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            return _inventory(args)
        if args.command == "report":
            return _report(args)
        parser.print_help(sys.stderr)
        return 1
    except catalog_module.CatalogError as exc:
        print(f"catalog error ({exc.code}): {exc.detail}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"I/O error: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
