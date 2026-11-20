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
    graph as graph_module,
    indexing,
    integrate,
    ledger as ledger_module,
    manifest,
    normalize,
    probe,
    report,
    resolver,
    verify as verify_module,
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


def _integration_row(
    row: dict[str, object], integration: integrate.EntryIntegration
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entry_id": row.get("entry_id"),
        "catalog": row.get("catalog"),
        "category": row.get("category"),
        "title": row.get("title"),
        "license_policy": normalize.license_policy(row),
        "metadata_sha256": row.get("metadata_sha256"),
        "integration": integration.to_artifact(),
    }


def _normalize(args: argparse.Namespace) -> int:
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
        ledger_enabled=False if args.no_ledger else None,
        data_root=args.data_root,
        resume_checkpoints_enabled=False if args.no_checkpoints else None,
        resume_max_attempts=args.max_attempts,
    ).replace(integration_metadata_only=bool(args.include_metadata_only))
    if args.limit is not None:
        config = config.replace(limit=args.limit)
    if args.sample is not None:
        config = config.replace(sample=args.sample)

    started = time.monotonic()
    generated_at = _utcnow()
    snapshot = catalog_module.load_catalog(config)
    build = manifest.build_rows(snapshot, config, limit=None)

    acquisition_path = config.data_root / "manifest" / (
        _manifest_name(snapshot).replace(".jsonl", ".acquisition.jsonl")
    )
    acquisitions, acquisition_commit = integrate.load_acquisitions(acquisition_path)
    if not acquisitions:
        print(
            f"no acquisition manifest at {acquisition_path}; run the acquire stage first",
            file=sys.stderr,
        )
        return 1
    if acquisition_commit and acquisition_commit != snapshot.commit:
        print(
            f"acquisition manifest is for commit {acquisition_commit}, "
            f"catalog resolves to {snapshot.commit}",
            file=sys.stderr,
        )
        return 1

    selected, unknown, truncated = integrate.select_entries(
        build.rows,
        acquisitions,
        entries=args.entry,
        sample=config.sample,
        limit=config.limit,
        include_metadata_only=config.integration_metadata_only,
    )
    for entry_id in unknown:
        print(f"not an acquirable entry id: {entry_id}", file=sys.stderr)

    if args.dry_run:
        plan = []
        for row in build.rows:
            entry_id = str(row.get("entry_id") or "")
            if entry_id not in selected:
                continue
            normalization = normalize.normalize_entry(
                row,
                acquisitions.get(entry_id),
                commit=snapshot.commit,
                profiles=indexing.PROFILES,
                config=config,
            )
            plan.append(
                {
                    "entry_id": entry_id,
                    "license_policy": normalize.license_policy(row),
                    "metadata_record": normalization.metadata.to_artifact()
                    if normalization.metadata
                    else None,
                    "content_records": [
                        record.to_artifact() for record in normalization.records
                    ],
                    "files": [item.to_artifact() for item in normalization.files],
                    "records_total": normalization.records_total,
                    "content_reason": normalization.content_reason,
                }
            )
        print(
            json.dumps(
                {
                    "catalog": {
                        "repo": snapshot.repo,
                        "commit": snapshot.commit,
                        "entries": len(build.rows),
                    },
                    "selection": {
                        "selected": len(selected),
                        "truncated": truncated,
                        "include_metadata_only": config.integration_metadata_only,
                    },
                    "plan": plan,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    endpoints = indexing.Endpoints.from_config(config)
    journal = indexing.CompletionJournal(
        config.data_root / "cache" / "integrations",
        enabled=config.integration_journal_enabled and not args.no_journal,
    )
    checkpoints = integrate.checkpoint_log(
        config,
        enabled=config.resume_checkpoints_enabled and not args.no_checkpoints,
    )
    manifest_path = config.data_root / "manifest" / (
        _manifest_name(snapshot).replace(".jsonl", ".integration.jsonl")
    )
    # Two resume sources: the checkpoint log (exact input fingerprints) and the
    # previous manifest (stage versions and metadata hash only).  Both are read
    # before the run so an interruption never costs completed work.
    previous_manifest = integrate.load_integration_manifest(manifest_path)
    integration_run = integrate.integrate_entries(
        build.rows,
        acquisitions,
        selected,
        commit=snapshot.commit,
        config=config,
        endpoints=endpoints,
        journal=journal,
        checkpoint=checkpoints,
        previous_manifest=previous_manifest,
        policy=integrate.retry_policy(config),
    )
    integrations = integration_run.integrations
    resume_state = integrate.resume_artifact(integration_run, checkpoint=checkpoints)

    accounting = integrate.check_accounting(build.rows, integrations, build.discovered)
    run_id = integrate.run_id_for(
        config, snapshot.commit, snapshot.snapshot_sha256, selected
    )
    integrations_by_entry = {item.entry_id: item for item in integrations}
    artifact_rows = [
        _integration_row(row, integrations_by_entry[str(row.get("entry_id"))])
        for row in build.rows
        if str(row.get("entry_id")) in integrations_by_entry
    ]
    changed, manifest_path = integrate.write_integration_manifest(
        artifact_rows, manifest_path
    )
    outcomes = integrate.integration_outcome_rows(
        build.rows,
        integrations,
        run_id=run_id,
        catalog_commit=snapshot.commit,
        retrieved_at=generated_at,
    )
    outcomes_added = manifest.append_outcome_rows(
        outcomes, config.data_root / "outcomes.jsonl"
    )

    if not accounting.ok:
        exit_code = 2
    elif integration_run.unhandled > 0:
        exit_code = 1
    elif truncated:
        exit_code = 3
    else:
        exit_code = 0

    duration_ms = int((time.monotonic() - started) * 1000)
    journal_state = {"enabled": journal.enabled, "entries": journal.count()}
    report_payload = report.build_integration_report(
        build.rows,
        integrations,
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
        integrate_version=integrate.INTEGRATE_VERSION,
        normalize_version=normalize.NORMALIZE_VERSION,
        indexing_version=indexing.INDEXING_VERSION,
        adapter_version=adapter_module.ADAPTER_VERSION,
        sample=config.sample,
        limit=config.limit,
        truncated=truncated,
        include_metadata_only=config.integration_metadata_only,
        journal=journal_state,
        resume_state=resume_state,
    )
    event_id = ledger_module.integration_event_id(
        config,
        snapshot,
        integrate_version=integrate.INTEGRATE_VERSION,
        selection_hash=resolver.selection_hash(selected),
        sample=config.sample,
        limit=config.limit,
        include_metadata_only=config.integration_metadata_only,
    )
    event_entry = ledger_module.build_integration_entry(
        event_id=event_id,
        run_id=run_id,
        snapshot=snapshot,
        report=report_payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        manifest_changed=changed,
        outcomes_added=outcomes_added,
        integrate_version=integrate.INTEGRATE_VERSION,
        normalize_version=normalize.NORMALIZE_VERSION,
        indexing_version=indexing.INDEXING_VERSION,
        adapter_version=adapter_module.ADAPTER_VERSION,
    )
    ledger_result = ledger_module.post_summary(
        config, event_id=event_id, entry=event_entry
    )
    report_payload["ledger"] = ledger_result
    report_path = manifest.write_run_report(report_payload, config.data_root, run_id)

    funnel = report_payload["funnel"]
    print(
        "normalize: discovered={discovered} statuses={statuses} records={records} "
        "shards={shards} chunks={chunks} vectors={vectors} resumed={resumed} "
        "retried={retried} invariant={invariant} exit={exit_code}".format(
            discovered=accounting.discovered,
            statuses=accounting.status_counts,
            records=funnel["records"],
            shards=funnel["shards"],
            chunks=funnel["chunks"],
            vectors=funnel["vectors"],
            resumed=funnel["resumed"],
            retried=funnel["retried"],
            invariant="ok" if accounting.ok else "violated",
            exit_code=exit_code,
        )
    )
    for entry in report_payload["entries"]:
        if entry.get("status") in {"deferred"} and entry.get("reason_code") == "not_selected":
            continue
        print(
            "  {status} {entry} records={records} shards={shards} chunks={chunks} "
            "vectors={vectors} attempts={attempts} resumed={resumed} reason={reason}".format(
                status=entry.get("status"),
                entry=entry.get("entry_id"),
                records=entry.get("records_total"),
                shards=entry.get("shards"),
                chunks=entry.get("chunks"),
                vectors=entry.get("vectors"),
                attempts=entry.get("attempts"),
                resumed=entry.get("resumed"),
                reason=entry.get("reason_code") or "-",
            )
        )
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return exit_code


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return rows
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _latest_manifest(data_root: Path, suffix: str) -> Path | None:
    directory = data_root / "manifest"
    try:
        candidates = sorted(
            directory.glob(f"*{suffix}"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _tombstoned_from_index(config: Config, selected: set[str]) -> tuple[set[str], int]:
    """Ask the Index service which selected entries are tombstoned.

    Opt-in and bounded by the selection: one GET per selected entry, errors
    ignored (a tombstone check must never fail the projection).
    """
    tombstoned: set[str] = set()
    checked = 0
    for entry_id in sorted(selected):
        try:
            state = indexing.source_state(config.index_url, entry_id)
        except indexing.IndexingError:
            continue
        checked += 1
        if state is not None and state.get("tombstoned"):
            tombstoned.add(entry_id)
    return tombstoned, checked


def _graph(args: argparse.Namespace) -> int:
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
    if args.apply and not args.prune_tombstoned:
        print("--apply only applies to --prune-tombstoned", file=sys.stderr)
        return 1

    config = Config.load(
        catalog_dir=args.catalog_dir,
        catalog_ref=args.catalog_ref,
        readme_path=args.readme,
        ledger_enabled=False if args.no_ledger else None,
        data_root=args.data_root,
        resume_checkpoints_enabled=False if args.no_checkpoints else None,
        resume_max_attempts=args.max_attempts,
        graph_check_tombstones=True if args.check_tombstones else None,
    ).replace(sample=args.sample, limit=args.limit)

    started = time.monotonic()
    generated_at = _utcnow()
    snapshot = catalog_module.load_catalog(config)
    # Resume from the persisted inventory when it still describes this
    # snapshot; otherwise rebuild the same rows from the catalog.
    inventory_path = config.data_root / "manifest" / _manifest_name(snapshot)
    persisted = manifest.load_rows(inventory_path)
    if persisted is not None and manifest.rows_match_snapshot(persisted, snapshot):
        build = manifest.BuildResult(
            rows=persisted, discovered=len(persisted), unhandled=0, truncated=False
        )
        inventory_source = "manifest"
    else:
        build = manifest.build_rows(snapshot, config, limit=None)
        inventory_source = "catalog"
    selected, unknown, truncated = graph_module.select_entries(
        build.rows, entries=args.entry, sample=config.sample, limit=config.limit
    )
    for entry_id in unknown:
        print(f"not an acquirable entry id: {entry_id}", file=sys.stderr)

    integration_path = config.data_root / "manifest" / (
        _manifest_name(snapshot).replace(".jsonl", ".integration.jsonl")
    )
    tombstones = graph_module.tombstones_from_manifest(
        integrate.load_integration_manifest(integration_path)
    )
    tombstones_checked = 0
    if config.graph_check_tombstones and selected:
        from_index, tombstones_checked = _tombstoned_from_index(config, selected)
        tombstones |= from_index
    tombstoned_selected = tombstones & selected

    client = graph_module.graph_client(config)
    checkpoints = graph_module.checkpoint_log(
        config,
        enabled=config.resume_checkpoints_enabled and not args.no_checkpoints,
    )
    results, context = graph_module.project_entries(
        build.rows,
        commit=snapshot.commit,
        config=config,
        client=client,
        selected=selected,
        journal=checkpoints,
        policy=graph_module.retry_policy(config),
        tombstoned=tombstones,
        dry_run=args.dry_run,
    )
    accounting = graph_module.check_accounting(build.rows, results, build.discovered)
    run_id = graph_module.run_id_for(
        snapshot.commit, snapshot.snapshot_sha256, selected
    )

    result_by_entry = {item.entry_id: item for item in results}
    artifact_rows = [
        graph_module.manifest_row(row, result_by_entry[str(row.get("entry_id"))])
        for row in build.rows
        if str(row.get("entry_id")) in result_by_entry
    ]
    manifest_path = graph_module.graph_manifest_path(config, _manifest_name(snapshot))
    if args.dry_run:
        changed = False
    else:
        changed, manifest_path = graph_module.write_graph_manifest(
            artifact_rows, manifest_path
        )
    outcomes_added = 0
    if not args.dry_run:
        outcomes_added = manifest.append_outcome_rows(
            graph_module.outcome_rows(
                build.rows,
                results,
                run_id=run_id,
                catalog_commit=snapshot.commit,
                retrieved_at=generated_at,
            ),
            config.data_root / "outcomes.jsonl",
        )

    prune = None
    if args.prune_tombstoned and not config.graph_prune_enabled:
        print(
            "prune: disabled by GRAPH_PRUNE_ENABLED; skipping",
            file=sys.stderr,
        )
    elif args.prune_tombstoned:
        prune = graph_module.prune_tombstoned(
            client,
            sorted(tombstones),
            # ``--dry-run`` wins over ``--apply``: a dry run never prunes.
            dry_run=args.dry_run or not args.apply,
        )

    successful = [
        item for item in results if item.entry_id in selected and item.status in {"indexed", "unchanged"}
    ]
    if not accounting.ok:
        exit_code = 2
    elif context.graph_errors and not successful:
        # Every selected entry failed against the graph: that is a run failure,
        # not a budget cap.
        exit_code = 1
    elif truncated:
        exit_code = 3
    else:
        exit_code = 0

    duration_ms = int((time.monotonic() - started) * 1000)
    resume_state = graph_module.resume_stats(results, checkpoints)
    graph_state = {
        "url": config.graph_url,
        "prefetch_mode": context.snapshot.mode,
        "prefetch_calls": context.snapshot.prefetch_calls,
        "prefetch_error": context.snapshot.error,
        "inventory_source": inventory_source,
        "tombstoned": len(tombstoned_selected),
        "tombstones_checked": tombstones_checked,
        "graph_errors": context.graph_errors,
        # Exact unique ids this run created; per-entry counts can undercount
        # when an entry was retried after a partial write.
        "nodes_created_unique": len(context.created_node_ids),
        "edges_created_unique": len(context.created_edge_ids),
    }
    report_payload = report.build_graph_report(
        build.rows,
        results,
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
        graph_version=graph_module.GRAPH_VERSION,
        sample=config.sample,
        limit=config.limit,
        truncated=truncated,
        dry_run=args.dry_run,
        resume_state=resume_state,
        graph_state=graph_state,
        prune=prune,
    )
    event_id = ledger_module.graph_event_id(
        config,
        snapshot,
        graph_version=graph_module.GRAPH_VERSION,
        selection_hash=resolver.selection_hash(selected),
        sample=config.sample,
        limit=config.limit,
        dry_run=args.dry_run,
    )
    event_entry = ledger_module.build_graph_entry(
        event_id=event_id,
        run_id=run_id,
        snapshot=snapshot,
        report=report_payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        graph_version=graph_module.GRAPH_VERSION,
        outcomes_added=outcomes_added,
        dry_run=args.dry_run,
    )
    report_path: Path | None = None
    if args.dry_run:
        # A dry run mutates nothing, including the run-report directory: it
        # prints the plan and leaves the "latest run" pointer alone.
        report_payload["ledger"] = {"enabled": config.ledger_enabled, "posted": False}
    else:
        report_payload["ledger"] = ledger_module.post_summary(
            config, event_id=event_id, entry=event_entry
        )
        report_path = manifest.write_run_report(
            report_payload, config.data_root, run_id
        )
    if args.out:
        text = (
            report.render_json(report_payload)
            if args.format == "json"
            else report.render_graph_markdown(report_payload)
        )
        manifest.atomic_write(Path(args.out), text.encode("utf-8"))

    funnel = report_payload["funnel"]
    print(
        "graph: discovered={discovered} statuses={statuses} nodes={nodes} edges={edges} "
        "tombstoned={tombstoned} prefetch={prefetch} dry_run={dry_run} "
        "invariant={invariant} exit={exit_code}".format(
            discovered=accounting.discovered,
            statuses=accounting.status_counts,
            nodes=(
                f"{graph_state['nodes_created_unique']}"
                f"/{funnel['nodes_present']}"
            ),
            edges=(
                f"{graph_state['edges_created_unique']}"
                f"/{funnel['edges_present']}"
            ),
            tombstoned=len(tombstoned_selected),
            prefetch=context.snapshot.mode,
            dry_run=args.dry_run,
            invariant="ok" if accounting.ok else "violated",
            exit_code=exit_code,
        )
    )
    if prune is not None:
        print(
            "prune: dry_run={dry_run} requested={requested} would_remove={would} "
            "removed={removed} failures={failures}".format(
                dry_run=prune["dry_run"],
                requested=prune["requested"],
                would=prune["would_remove"],
                removed=prune["removed"],
                failures=len(prune["failures"]),
            )
        )
    for entry in report_payload["entries"]:
        if entry.get("status") == "deferred" and entry.get("reason_code") == "not_selected":
            continue
        print(
            "  {status} {entry} nodes=+{nc}/{np} edges=+{ec}/{ep} attempts={attempts} "
            "resumed={resumed} reason={reason}".format(
                status=entry.get("status"),
                entry=entry.get("entry_id"),
                nc=entry.get("nodes_created"),
                np=entry.get("nodes_present"),
                ec=entry.get("edges_created"),
                ep=entry.get("edges_present"),
                attempts=entry.get("attempts"),
                resumed=entry.get("resumed"),
                reason=entry.get("reason_code") or "-",
            )
        )
    if args.dry_run:
        print("dry-run: no graph writes, no manifest, no ledger event")
        return exit_code
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return exit_code


def _verify(args: argparse.Namespace) -> int:
    config = Config.load(data_root=args.data_root, ledger_enabled=False if args.no_ledger else None)
    if args.top_k is not None:
        if args.top_k < 1:
            print("--top-k must be positive", file=sys.stderr)
            return 1
        config = config.replace(verify_top_k=args.top_k)

    started = time.monotonic()
    generated_at = _utcnow()
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else _latest_manifest(config.data_root, ".integration.jsonl")
    )
    if manifest_path is None:
        print(
            "no integration manifest found; run the normalize stage first",
            file=sys.stderr,
        )
        return 1
    rows = _load_jsonl(manifest_path)
    if not rows:
        print(f"integration manifest is empty: {manifest_path}", file=sys.stderr)
        return 1
    catalog_block = rows[0].get("catalog") if isinstance(rows[0].get("catalog"), dict) else {}
    assert isinstance(catalog_block, dict)
    catalog_commit = str(catalog_block.get("commit") or "unknown")

    if args.query:
        specs = [
            verify_module.QuerySpec(
                name=f"q{index + 1}", text=text, entry_id="", expected_sources=()
            )
            for index, text in enumerate(args.query)
        ]
    else:
        specs = verify_module.build_queries(
            rows, limit=config.verify_queries, categories=args.category
        )
    if not specs and args.no_graph:
        print(
            "no indexed content to verify in "
            f"{manifest_path}; integrate at least one entry first",
            file=sys.stderr,
        )
        return 1
    if not specs:
        print(
            f"no indexed content in {manifest_path}; verifying graph traversal only",
            file=sys.stderr,
        )

    results = (
        verify_module.run_queries(specs, config, top_k=config.verify_top_k)
        if specs
        else []
    )
    vector_health = verify_module.vector_store_health(config)
    graph_results: list[dict[str, object]] = []
    graph_categories: list[str] = []
    if not args.no_graph:
        client = graph_module.graph_client(config)
        graph_categories = verify_module.graph_queries(
            rows, limit=3, categories=args.category
        )
        graph_results = verify_module.verify_graph(
            client, graph_categories, max_depth=2
        )
    summary = verify_module.summarize(
        results,
        vector_health=vector_health,
        graph_results=graph_results,
        retrieval_checked=bool(specs),
    )
    exit_code = 0 if summary["ok"] else 2
    duration_ms = int((time.monotonic() - started) * 1000)
    run_id = verify_module.verification_run_id(
        catalog_commit, specs, graph_categories=graph_categories
    )
    payload = verify_module.verification_report(
        queries=specs,
        results=results,
        vector_health=vector_health,
        graph_results=graph_results,
        run_id=run_id,
        generated_at=generated_at,
        duration_ms=duration_ms,
        top_k=config.verify_top_k,
        exit_code=exit_code,
        retrieval_checked=bool(specs),
    )
    payload["catalog"] = {"commit": catalog_commit, "entries": len(rows)}
    payload["manifest"] = {"path": str(manifest_path)}
    payload["ledger"] = {"enabled": config.ledger_enabled, "posted": False}
    event_id = ledger_module.verification_event_id(
        config,
        catalog_commit=catalog_commit,
        verify_version=verify_module.VERIFY_VERSION,
        query_hash=model_hash(
            [spec.text for spec in specs]
            + [f"graph:{name}" for name in graph_categories]
        ),
        top_k=config.verify_top_k,
    )
    event_entry = ledger_module.build_verification_entry(
        event_id=event_id,
        run_id=run_id,
        catalog_commit=catalog_commit,
        report=payload,
        exit_code=exit_code,
        duration_ms=duration_ms,
        verify_version=verify_module.VERIFY_VERSION,
    )
    payload["ledger"] = ledger_module.post_summary(
        config, event_id=event_id, entry=event_entry
    )
    report_path = manifest.write_run_report(payload, config.data_root, run_id)
    text = (
        verify_module.render_json(payload)
        if args.format == "json"
        else verify_module.render_markdown(payload)
    )
    if args.out:
        manifest.atomic_write(Path(args.out), text.encode("utf-8"))

    legs = summary["legs"]
    print(
        "verify: queries={queries} lexical={lexical} vector={vector} hybrid={hybrid} "
        "vector_store={functional} graph={graph} result={result} exit={exit_code}".format(
            queries=len(specs),
            lexical=f"{legs['lexical']['status']}({legs['lexical']['matched']}/{legs['lexical']['queries']})",
            vector=f"{legs['vector']['status']}({legs['vector']['matched']}/{legs['vector']['queries']})",
            hybrid=f"{legs['hybrid']['status']}({legs['hybrid']['matched']}/{legs['hybrid']['queries']})",
            functional=vector_health["functional"],
            graph=summary["graph"]["verified"],
            result="pass" if summary["ok"] else "fail",
            exit_code=exit_code,
        )
    )
    for query in payload["queries"]:
        print(
            "  {name} query={query!r} entry={entry} lexical={l} vector={v} hybrid={h}".format(
                name=query["name"],
                query=query["query"],
                entry=query["entry_id"] or "-",
                l=query["lexical"]["matched"],
                v=query["vector"]["matched"],
                h=query["hybrid"]["matched"],
            )
        )
    for item in graph_results:
        print(
            "  graph {category}: datasets={datasets} orgs={orgs} formats={formats} "
            "licenses={licenses} status={status}".format(
                category=item.get("category"),
                datasets=item.get("datasets"),
                orgs=item.get("related_organizations"),
                formats=item.get("related_formats"),
                licenses=item.get("related_licenses"),
                status=item.get("status"),
            )
        )
    for problem in summary["problems"]:
        print(f"  problem: {problem}", file=sys.stderr)
    print(f"report: {report_path}")
    return exit_code


def model_hash(values: list[str]) -> str:
    """Stable hash of the query set for the verification ledger event."""
    import hashlib

    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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

    integrate_parser = subparsers.add_parser(
        "normalize",
        help="normalize acquired content and index it in Knowledge and the retrieval stack (PR 4)",
    )
    integrate_source = integrate_parser.add_mutually_exclusive_group()
    integrate_source.add_argument("--catalog-ref", help="catalog git ref to fetch (default: master)")
    integrate_source.add_argument("--catalog-dir", help="local catalog checkout to read")
    integrate_parser.add_argument(
        "--sample", type=int, help="integrate a deterministic sample of N eligible entries"
    )
    integrate_parser.add_argument("--limit", type=int, help="integrate at most N entries")
    integrate_parser.add_argument(
        "--entry",
        action="append",
        help="integrate one specific entry id (repeatable; overrides --sample/--limit)",
    )
    integrate_parser.add_argument(
        "--include-metadata-only",
        action="store_true",
        help="also integrate metadata records for license-unknown entries (no content)",
    )
    integrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="normalize and print the record plan; no service calls, no writes",
    )
    integrate_parser.add_argument(
        "--no-journal",
        action="store_true",
        help="do not read or write the local indexing-completion journal",
    )
    integrate_parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="ignore the resume checkpoints for this stage",
    )
    integrate_parser.add_argument(
        "--max-attempts",
        type=int,
        help="per-entry attempt budget for retryable failures (default: 3)",
    )
    integrate_parser.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    integrate_parser.add_argument("--readme", help="explicit README.rst path for OK/FIXME reconciliation")
    integrate_parser.add_argument("--data-root", help="override the dataset data root")

    graph_parser = subparsers.add_parser(
        "graph", help="project catalog relationships into the Graph service (PR 5)"
    )
    graph_source = graph_parser.add_mutually_exclusive_group()
    graph_source.add_argument("--catalog-ref", help="catalog git ref to fetch (default: master)")
    graph_source.add_argument("--catalog-dir", help="local catalog checkout to read")
    graph_parser.add_argument("--sample", type=int, help="project a deterministic sample of N entries")
    graph_parser.add_argument("--limit", type=int, help="project at most N entries")
    graph_parser.add_argument(
        "--entry",
        action="append",
        help="project one specific entry id (repeatable; overrides --sample/--limit)",
    )
    graph_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written; no graph writes, no manifest",
    )
    graph_parser.add_argument(
        "--check-tombstones",
        action="store_true",
        help="also ask the Index service which selected entries are tombstoned",
    )
    graph_parser.add_argument(
        "--prune-tombstoned",
        action="store_true",
        help="remove graph nodes for tombstoned entries (dry run unless --apply)",
    )
    graph_parser.add_argument(
        "--apply", action="store_true", help="perform the tombstone prune instead of previewing it"
    )
    graph_parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="ignore the resume checkpoints for the graph stage",
    )
    graph_parser.add_argument(
        "--max-attempts",
        type=int,
        help="per-entry attempt budget for retryable failures (default: 3)",
    )
    graph_parser.add_argument(
        "--format", choices=("md", "json"), default="md", help="format for --out"
    )
    graph_parser.add_argument("--out", help="write the report to a file")
    graph_parser.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    graph_parser.add_argument("--readme", help="explicit README.rst path for OK/FIXME reconciliation")
    graph_parser.add_argument("--data-root", help="override the dataset data root")

    verify_parser = subparsers.add_parser(
        "verify", help="verify ingested content is searchable (PR 5)"
    )
    verify_parser.add_argument(
        "--manifest",
        help="integration manifest to verify (default: the most recent one)",
    )
    verify_parser.add_argument(
        "--query",
        action="append",
        help="ad-hoc query text (repeatable; informational, no expectation)",
    )
    verify_parser.add_argument(
        "--category",
        action="append",
        help="restrict derived queries and graph traversal to a category (repeatable)",
    )
    verify_parser.add_argument("--top-k", type=int, help="results per query (default: 5)")
    verify_parser.add_argument(
        "--no-graph", action="store_true", help="skip graph traversal verification"
    )
    verify_parser.add_argument(
        "--format", choices=("md", "json"), default="md", help="format for --out"
    )
    verify_parser.add_argument("--out", help="write the report to a file")
    verify_parser.add_argument("--no-ledger", action="store_true", help="do not post the summary event")
    verify_parser.add_argument("--data-root", help="override the dataset data root")

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
        if args.command == "normalize":
            return _normalize(args)
        if args.command == "graph":
            return _graph(args)
        if args.command == "verify":
            return _verify(args)
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
