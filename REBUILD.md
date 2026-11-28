# Development halted — POC completed its assignment

**Date:** 2026-11-28
**Decision:** POC data-path development is permanently stopped. This
repository is archived as a frozen reference.

## Why

This POC existed to validate the ensemble design — service contracts,
worker/arbiter flows, memory/knowledge/learning loops — at small scale.
It did that successfully. On 2026-11-27/28 the first full-corpus
ingestion campaign (~1M items) promoted it to production load, and it
correctly found the ceilings it was never built to cross:

- The in-service work queue retained every item (including completed
  ones) in RAM with no eviction — 482K enqueued items OOM-killed the
  host and dependent services.
- The chromadb vector backend held the full index in memory (~20 GB)
  and required ~25-minute full reloads on restart.
- The embedding service deadlocked under concurrent load (handler
  threads never returned; /health stayed green while /embeddings hung).

These are architecture properties, not code bugs — the 259-finding
code-review register is a separate record and most of it is now
historical. Fixing them in place would be rebuilding the POC into the
real system, which is what the successor does properly.

## What replaces this

- **[FlossWare/ensemble](https://github.com/FlossWare/ensemble)** — the
  real implementation: one FastAPI application, SQLite persistence
  throughout (FTS5 lexical, vector collections), Alembic migrations, and
  a capability completion plan that already covers most of the load
  properties this POC could not provide.
- **Rebuild spec:** `~/.FlossWare/docs/ensemble-rebuild-spec.md` (local)
  — the campaign's non-negotiables (bounded memory, warm reload,
  concurrency safety, backpressure, crash-observability) as ensemble
  acceptance criteria, plus the gap analysis and work queue.
- **Roadmap:** ensemble proves the product under load → capabilities
  graduate into the standalone FlossWare component repos → Loom
  (loom/loom-ai) is the language-neutral contract layer → per-client
  adapters (ensemble-claude/codex/crush/cursor).

## What still runs from this repo's deployment

The deployed copy at `~/.FlossWare/poc-claude-ensemble/` continues to
run the **fleet functions** — gateway, memory, learning, thompson,
graph — which power Claude Code context hooks and review tooling. These
stay running as-is (urgent-breakage fixes only) until ensemble's
equivalents cut over, at which point this deployment retires.

## Register

The 259-finding register and the fix campaign (#94–98 closed at
9ac18e2) remain in this repo's history and issues as historical record.
Data-path findings are superseded as rebuild inputs per the spec;
relabeling of those issues was interrupted and not completed — the spec
is authoritative for disposition.
