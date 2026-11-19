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
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ingestion.datasets import acquire as acquire_module
from ingestion.datasets import adapters as adapter_module
from ingestion.datasets import archive as archive_module
from ingestion.datasets import catalog as catalog_module
from ingestion.datasets import (
    classify,
    ledger as ledger_module,
    manifest,
    probe,
    report,
    resolver,
)
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


def _resolve(args: argparse.Namespace) -> int:
    if args.limit is not None and args.sample is not None:
        print("--limit and --sample are mutually exclusive", file=sys.stderr)
        return 1
    if (args.limit is not None and args.limit < 0) or (
        args.sample is not None and args.sample < 0
    ):
        print("--limit/--sample must be non-negative", file=sys.stderr)
        return 1
    config = Config.load(
        catalog_dir=args.catalog_dir,
        catalog_ref=args.catalog_ref,
        readme_path=args.readme,
        limit=args.limit,
        sample=args.sample,
        resolve_cache_enabled=False if args.no_cache else None,
        ledger_enabled=False if args.no_ledger else None,
        data_root=args.data_root,
    )

    started = time.monotonic()
    generated_at = _utcnow()
    snapshot = catalog_module.load_catalog(config)
    # Build the full inventory (no probe, no limit) so every discovered entry
    # has exactly one resolution row and the accounting invariant can hold.
    build = manifest.build_rows(snapshot, config, limit=None)

    selected_ids, truncated = resolver.select_entries(
        build.rows, sample=config.sample, limit=config.limit
    )
    resolutions, unhandled, context = resolver.resolve_entries(
        build.rows, selected_ids, config, now=generated_at
    )
    completed = resolver.complete_resolutions(
        build.rows, resolutions, now=generated_at, config=config
    )
    accounting = resolver.check_resolution_accounting(completed, build.discovered)

    run_id = resolver.resolution_run_id(
        config,
        snapshot,
        selected_ids=selected_ids,
        sample=config.sample,
        limit=config.limit,
    )
    manifest_path = config.data_root / "manifest" / (
        _manifest_name(snapshot).replace(".jsonl", ".resolution.jsonl")
    )
    changed, manifest_path = resolver.write_resolution_manifest(
        build.rows, completed, manifest_path
    )
    outcomes_added = resolver.append_resolution_outcomes(
        build.rows,
        completed,
        config.data_root / "outcomes.jsonl",
        run_id=run_id,
        catalog_commit=snapshot.commit,
        retrieved_at=generated_at,
    )

    if not accounting.ok:
        exit_code = 2
    elif unhandled > 0:
        exit_code = 1
    elif truncated:
        exit_code = 3
    else:
        exit_code = 0

    duration_ms = int((time.monotonic() - started) * 1000)
    report_payload = report.build_resolution_report(
        build.rows,
        completed,
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
        resolver_version=resolver.RESOLVER_VERSION,
        sample=config.sample,
        limit=config.limit,
        unhandled=unhandled,
        truncated=truncated,
        cache_hits=context.cache_hits,
        cache_misses=context.cache_misses,
    )
    event_id = ledger_module.resolution_event_id(
        config,
        snapshot,
        resolver_version=resolver.RESOLVER_VERSION,
        selection_hash=resolver.selection_hash(selected_ids),
        sample=config.sample,
        limit=config.limit,
    )
    event_entry = ledger_module.build_resolution_entry(
        event_id=event_id,
        run_id=run_id,
        snapshot=snapshot,
        report=report_payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        manifest_changed=changed,
        outcomes_added=outcomes_added,
        resolver_version=resolver.RESOLVER_VERSION,
    )
    ledger_result = ledger_module.post_summary(
        config, event_id=event_id, entry=event_entry
    )
    report_payload["ledger"] = ledger_result
    report_path = manifest.write_run_report(report_payload, config.data_root, run_id)

    print(
        "resolve: discovered={discovered} statuses={statuses} "
        "cache_hits={hits} invariant={invariant} exit={exit_code}".format(
            discovered=accounting.discovered,
            statuses=accounting.status_counts,
            hits=context.cache_hits,
            invariant="ok" if accounting.ok else "violated",
            exit_code=exit_code,
        )
    )
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return exit_code


def _acquire(args: argparse.Namespace) -> int:
    if args.entry and (args.limit is not None or args.sample is not None):
        print("--entry cannot be combined with --limit/--sample", file=sys.stderr)
        return 1
    if args.limit is not None and args.sample is not None:
        print("--limit and --sample are mutually exclusive", file=sys.stderr)
        return 1
    if (args.limit is not None and args.limit < 0) or (
        args.sample is not None and args.sample < 0
    ):
        print("--limit/--sample must be non-negative", file=sys.stderr)
        return 1
    config = Config.load(
        catalog_dir=args.catalog_dir,
        catalog_ref=args.catalog_ref,
        readme_path=args.readme,
        limit=args.limit,
        sample=args.sample,
        ledger_enabled=False if args.no_ledger else None,
        data_root=args.data_root,
        acquire_resume_enabled=False if args.no_resume else None,
        acquire_reuse_enabled=False if args.no_reuse else None,
        acquire_resolve_missing=False if args.no_resolve else None,
        acquire_max_files_per_dataset=args.max_files,
        max_file_bytes=args.max_bytes,
    )
    config.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    started = time.monotonic()
    generated_at = _utcnow()
    snapshot = catalog_module.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)
    if args.entry:
        selected_ids, unknown = acquire_module.select_named(build.rows, args.entry)
        truncated = False
        for entry_id in unknown:
            print(f"not an acquirable entry id: {entry_id}", file=sys.stderr)
    else:
        selected_ids, truncated = acquire_module.select_entries(
            build.rows, sample=config.sample, limit=config.limit
        )

    # Resolution evidence comes from the resolve stage's cache; missing entries
    # are resolved lazily through the same PR 2 code path (bounded, cached).
    resolutions: dict[str, object] = {}
    resolution_cache = resolver.ResolutionCache(config)
    rows_by_id = {str(row.get("entry_id")): row for row in build.rows}
    for entry_id in sorted(selected_ids):
        row = rows_by_id.get(entry_id)
        if row is None:
            continue
        cached = resolution_cache.get(
            entry_id, str(row.get("metadata_sha256") or ""), generated_at
        )
        if cached is not None:
            resolutions[entry_id] = cached

    if args.dry_run:
        plan = acquire_module.build_plan(build.rows, resolutions, selected_ids, config)
        payload = {
            "catalog": {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "entries": len(build.rows),
            },
            "selection": {
                "sample": config.sample,
                "limit": config.limit,
                "selected": len(selected_ids),
                "truncated": truncated,
            },
            "cached_resolutions": len(resolutions),
            "plan": plan,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    resolved_during_run = 0
    missing = selected_ids - set(resolutions)
    if missing and config.acquire_resolve_missing:
        resolved, resolve_unhandled, _ctx = resolver.resolve_entries(
            build.rows, missing, config, now=generated_at
        )
        for item in resolved:
            resolutions[item.entry_id] = item
            if item.status == "resolved":
                resolved_during_run += 1
        if resolve_unhandled:
            print(
                f"resolve: {resolve_unhandled} unhandled per-entry error(s)",
                file=sys.stderr,
            )

    acquisitions, unhandled, context = acquire_module.acquire_entries(
        build.rows, resolutions, selected_ids, config
    )
    completed = acquire_module.complete_acquisitions(
        build.rows, acquisitions, now=generated_at, config=config
    )
    accounting = acquire_module.check_acquisition_accounting(completed, build.discovered)

    run_id = acquire_module.acquisition_run_id(
        config,
        snapshot,
        selected_ids=selected_ids,
        sample=config.sample,
        limit=config.limit,
    )
    manifest_path = config.data_root / "manifest" / (
        _manifest_name(snapshot).replace(".jsonl", ".acquisition.jsonl")
    )
    changed, manifest_path = acquire_module.write_acquisition_manifest(
        build.rows, completed, manifest_path
    )
    outcomes_added = acquire_module.append_acquisition_outcomes(
        build.rows,
        completed,
        config.data_root / "outcomes.jsonl",
        run_id=run_id,
        catalog_commit=snapshot.commit,
        retrieved_at=generated_at,
    )

    if not accounting.ok:
        exit_code = 2
    elif unhandled > 0:
        exit_code = 1
    elif truncated:
        exit_code = 3
    else:
        exit_code = 0

    duration_ms = int((time.monotonic() - started) * 1000)
    budget = {
        "bytes_used": context.budget.bytes_used,
        "datasets_used": context.budget.datasets_used,
        "max_run_bytes": config.max_run_bytes,
        "max_run_datasets": config.max_run_datasets,
        "max_file_bytes": config.max_file_bytes,
        "max_dataset_bytes": config.max_dataset_bytes,
    }
    report_payload = report.build_acquisition_report(
        build.rows,
        completed,
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
        acquire_version=acquire_module.ACQUIRE_VERSION,
        adapter_version=adapter_module.ADAPTER_VERSION,
        archive_version=archive_module.ARCHIVE_VERSION,
        sample=config.sample,
        limit=config.limit,
        unhandled=unhandled,
        truncated=truncated,
        cache_hits=context.cache_hits,
        cache_misses=context.cache_misses,
        budget=budget,
        resolutions_used=len(resolutions),
        resolved_during_run=resolved_during_run,
    )
    event_id = ledger_module.acquisition_event_id(
        config,
        snapshot,
        acquire_version=acquire_module.ACQUIRE_VERSION,
        selection_hash=resolver.selection_hash(selected_ids),
        sample=config.sample,
        limit=config.limit,
    )
    event_entry = ledger_module.build_acquisition_entry(
        event_id=event_id,
        run_id=run_id,
        snapshot=snapshot,
        report=report_payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        manifest_changed=changed,
        outcomes_added=outcomes_added,
        acquire_version=acquire_module.ACQUIRE_VERSION,
        adapter_version=adapter_module.ADAPTER_VERSION,
    )
    ledger_result = ledger_module.post_summary(
        config, event_id=event_id, entry=event_entry
    )
    report_payload["ledger"] = ledger_result
    report_path = manifest.write_run_report(report_payload, config.data_root, run_id)

    funnel = report_payload["funnel"]
    print(
        "acquire: discovered={discovered} statuses={statuses} files={files} "
        "bytes={bytes} records={records} invariant={invariant} exit={exit_code}".format(
            discovered=accounting.discovered,
            statuses=accounting.status_counts,
            files=funnel["files"],
            bytes=funnel["bytes"],
            records=funnel["records"],
            invariant="ok" if accounting.ok else "violated",
            exit_code=exit_code,
        )
    )
    for entry in report_payload["entries"]:
        if entry.get("status") not in {"acquired", "unchanged"}:
            continue
        digests = ",".join(
            str(item.get("sha256") or "")[:16]
            for item in entry.get("files") or []
            if item.get("sha256")
        )
        adapters = ",".join(
            f"{key}:{value}" for key, value in (entry.get("adapters") or {}).items()
        )
        print(
            "  {status} {entry} files={files} bytes={bytes} records={records} "
            "parse={parse} adapters={adapters} sha256={sha256}".format(
                status=entry.get("status"),
                entry=entry.get("entry_id"),
                files=entry.get("files_acquired"),
                bytes=entry.get("bytes_total"),
                records=entry.get("records_total"),
                parse=entry.get("parse_status"),
                adapters=adapters or "-",
                sha256=digests or "-",
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

    resolve_parser = subparsers.add_parser(
        "resolve", help="resolve entries to their authoritative dataset sources (PR 2)"
    )
    resolve_source = resolve_parser.add_mutually_exclusive_group()
    resolve_source.add_argument("--catalog-ref", help="catalog git ref to fetch (default: master)")
    resolve_source.add_argument("--catalog-dir", help="local catalog checkout to read")
    resolve_parser.add_argument(
        "--sample",
        type=int,
        help="resolve a deterministic stratified sample of N entries across dispositions",
    )
    resolve_parser.add_argument("--limit", type=int, help="resolve at most N entries")
    resolve_parser.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    resolve_parser.add_argument(
        "--no-cache", action="store_true", help="ignore and do not write the resolution cache"
    )
    resolve_parser.add_argument("--readme", help="explicit README.rst path for OK/FIXME reconciliation")
    resolve_parser.add_argument("--data-root", help="override the dataset data root")

    acquire_parser = subparsers.add_parser(
        "acquire", help="download resolved dataset content and parse it (PR 3)"
    )
    acquire_source = acquire_parser.add_mutually_exclusive_group()
    acquire_source.add_argument("--catalog-ref", help="catalog git ref to fetch (default: master)")
    acquire_source.add_argument("--catalog-dir", help="local catalog checkout to read")
    acquire_parser.add_argument(
        "--sample",
        type=int,
        help="acquire a deterministic stratified sample of N index_content entries across formats",
    )
    acquire_parser.add_argument("--limit", type=int, help="acquire at most N entries")
    acquire_parser.add_argument(
        "--entry",
        action="append",
        help="acquire one specific entry id (repeatable; overrides --sample/--limit)",
    )
    acquire_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the acquisition plan; no network, no writes",
    )
    acquire_parser.add_argument(
        "--no-resume", action="store_true", help="do not resume partial downloads"
    )
    acquire_parser.add_argument(
        "--no-reuse", action="store_true", help="ignore the per-entry acquisition cache"
    )
    acquire_parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="do not resolve entries missing resolution evidence",
    )
    acquire_parser.add_argument(
        "--max-files", type=int, help="files to acquire per dataset"
    )
    acquire_parser.add_argument(
        "--max-bytes", type=int, help="per-file byte cap"
    )
    acquire_parser.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    acquire_parser.add_argument("--readme", help="explicit README.rst path for OK/FIXME reconciliation")
    acquire_parser.add_argument("--data-root", help="override the dataset data root")

    report_parser = subparsers.add_parser("report", help="render the latest inventory or resolution report")
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
        if args.command == "resolve":
            return _resolve(args)
        if args.command == "acquire":
            return _acquire(args)
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
