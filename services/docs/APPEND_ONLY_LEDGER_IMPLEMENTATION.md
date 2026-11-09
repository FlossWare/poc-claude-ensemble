# Append-only ledger implementation

The shared contract is in [APPEND_ONLY_LEDGER_CONTRACT.md](APPEND_ONLY_LEDGER_CONTRACT.md). This document records the POC adapter details without changing the existing session-history API.

## Storage and compatibility

The POC ledger is layered over the existing session-events JSONL file:

- Default: `~/.FlossWare/ledger/session-events/session-events.jsonl`
- Override: `POC_CLAUDE_ENSEMBLE_LEDGER_ROOT`
- Stable ledger identity: sibling `session-events.jsonl.ledger-meta.json`
- Cross-process writer/read coordination: sibling `.lock` file using shared/exclusive `flock`
- Appends and imported records are flushed and fsynced.

Existing JSONL rows are not rewritten or migrated in place. `GET /history` and `GET /history/{event_id}` continue to return their original record representation and retain their existing strict malformed-record behavior.

New ledger pulls normalize legacy rows into a portable schema-version-0 envelope. The original row is retained in `provenance.legacy_record`; a deterministic synthetic event ID is derived from the original row bytes only when the legacy row has no event ID. The exported legacy row has no entry hash and is explicitly marked `legacy_unverified`. The computed payload digest is an export integrity aid, not evidence that the original legacy row was tamper-evident.

New appends and imported rows use schema version 1 and the shared canonical JSON SHA-256 hash chain.

## REST

The memory HTTP service exposes:

- `POST /api/v1/ledger/entries`
- `GET /api/v1/ledger/entries`
- `GET /api/v1/ledger/entries/{event_id}`
- `GET /api/v1/ledger/status`
- `POST /api/v1/ledger/verify`
- `POST /api/v1/ledger/import`

The legacy `/history` endpoints are unchanged. GET pull scans at most `limit` source records, advances the cursor across non-matching records, and returns records in sequence order. Set `through_sequence` to the status head for a finite traversal.

A full verification starts at sequence 1. Partial verification requires a checkpoint retained by the caller. A hash-chain checkpoint has `ledger_id`, `sequence`, and `entry_sha256`. A checkpoint at the end of a legacy prefix instead has `ledger_id`, `sequence`, `legacy_prefix_sha256`, and `legacy_prefix_sequence_count`; it preserves the explicit legacy-unverified assurance.

## Import and retry

Imports are fully validated and conflict-checked before the first append. Each accepted JSONL row is flushed and fsynced separately. If an I/O error interrupts the batch, the API reports `committed_prefix` and `retry_after_source_sequence`; retrying the same ordered batch is safe because already-present identical event IDs are duplicates. A conflicting ID is rejected.

The source ledger is only read. Export and transfer do not consume, truncate, or modify it.

## Cross-repository transfer test

The same `parity/ledger-contract.json` fixture is used by both adapters. The optional live HTTP transfer test is skipped unless both service URLs are configured.

Run against live services:

```sh
POC_LEDGER_URL=http://127.0.0.1:8767 \
ENSEMBLE_LEDGER_URL=http://127.0.0.1:8080 \
python parity/ledger_http_transfer.py
```

The runner captures each source head, transfers bounded pages in both directions, and verifies that source ledger identity, head sequence, and head hash are unchanged.
