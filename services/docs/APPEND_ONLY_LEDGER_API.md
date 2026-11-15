# Append-only Ledger API

This document defines the POC's language-neutral ledger contract. The JSONL
file is authoritative; REST is an adapter over that file, not a second store.
The implementation is in `services/memory/append_only_ledger.py`.

## Storage and identity

The session-events ledger is stored at:

```text
$POC_CLAUDE_ENSEMBLE_LEDGER_ROOT/session-events/session-events.jsonl
```

The default root is `~/.FlossWare/ledger`. A sidecar file named
`session-events.jsonl.ledger-meta.json` holds a stable `ledger_id`.
Each non-empty JSONL record occupies one 1-based sequence position. Records
are append-only; no API operation deletes or rewrites an existing entry.

## Event envelope

New entries use schema version 1 and contain:

| Field | Meaning |
|---|---|
| `schema_version` | Envelope version, currently `1` |
| `ledger_id` | Stable identity of the source ledger |
| `sequence` | 1-based append order |
| `event_id` | Stable, unique event identity |
| `event_type` | Event kind |
| `source` | Producer/source name |
| `timestamp` | Optional event time (timezone-aware ISO-8601) |
| `captured_at` | Time the ledger accepted the event |
| `correlation_id` | Optional request/workflow correlation |
| `payload` | Canonical-JSON event data |
| `payload_sha256` | SHA-256 of canonical payload JSON |
| `previous_entry_sha256` | Previous v1 envelope hash, or legacy-prefix digest |
| `entry_sha256` | SHA-256 of the canonical envelope excluding this field |
| `provenance` | Import/source provenance; empty for locally appended events |

Canonical JSON is UTF-8 with object keys sorted, compact separators, and
non-finite numbers rejected. SHA-256 values are lowercase hexadecimal.

The first v1 entry has a null predecessor when the ledger has no legacy
prefix. Each subsequent v1 entry links to the prior v1 `entry_sha256`.
When pre-contract legacy records exist, their exact raw JSONL bytes (without
line terminators) are folded into a SHA-256 prefix digest; the first v1 entry
links to that digest. Legacy records remain readable but are explicitly
reported as unverified: they do not gain cryptographic assurance retroactively.

## REST contract

Base URL: `http://127.0.0.1:8767` (Memory service).

### Status and checkpoint

`GET /api/v1/ledger/status` returns `ledger_id`, `head_sequence`,
`head_hash`, legacy-prefix metadata, and a `checkpoint` suitable for the
incremental verification endpoint.

A checkpoint contains `ledger_id` and `sequence`, plus either
`entry_sha256` for a hashed head or `legacy_prefix_sha256` and
`legacy_prefix_sequence_count` for a legacy-only head. Keep the checkpoint
with the sequence cursor it describes; do not reuse it for another ledger.

### Pull, pagination, and export

`GET /api/v1/ledger/entries` supports:

| Query parameter | Meaning |
|---|---|
| `after_sequence` | Exclusive lower cursor; defaults to `0` |
| `through_sequence` | Inclusive upper bound; defaults to the current head |
| `limit` | Maximum records scanned per page; 1–1000, default 100 |
| `event_id` | Exact event ID |
| `event_type` | Exact event type |
| `source` | Exact producer/source |
| `correlation_id` | Exact correlation ID |
| `since`, `until` | Inclusive timezone-aware ISO-8601 event-time bounds |

The response includes `entries`, `next_after_sequence`, `head_sequence`,
and `has_more`. The cursor advances over scanned sequence positions, not
only matching entries; this prevents filtered pulls from becoming stuck on
non-matching records. The upper bound is fixed for a transfer by reading
status once and supplying that `through_sequence` on every page. Appends
after that bound are picked up by the next transfer and do not shift earlier
pages.

For a full export, pull without filters. `export()` and `replay()` are
in-process aliases over the same ordered pull operation; the REST pull
endpoint is the portable export surface.

`GET /api/v1/ledger/entries/{event_id}` retrieves one event by stable ID.

### Append

`POST /api/v1/ledger/entries` accepts an event object with `event_id`,
`event_type`, `source`, optional `timestamp` and `correlation_id`, and
`payload`. The server assigns the destination ledger ID, sequence,
capture time, payload hash, predecessor hash, and entry hash.

- HTTP 201: newly stored.
- HTTP 200: an identical event ID/content already exists (idempotent retry).
- HTTP 400: invalid event.
- HTTP 409: event ID already exists with different content.

### Verification

`POST /api/v1/ledger/verify` accepts:

```json
{
  "from_sequence": 1,
  "through_sequence": null,
  "checkpoint": null
}
```

A full verification starts at sequence 1. Incremental verification starts at
the next sequence after a previously retained checkpoint, passing that
checkpoint in the request. Partial verification without a matching
`ledger_id` and immediately preceding sequence checkpoint fails closed.

Verification detects malformed JSONL, missing/reordered sequence positions,
duplicate event IDs, payload hash mismatches, entry hash mismatches, and
broken predecessor links. The result reports `first_failing_sequence`,
`failure_category`, `verified_count`, and `assurance`. Legacy-prefix
assurance remains explicitly marked as unverified.

### Import and portable synchronization

`POST /api/v1/ledger/import` accepts `source_ledger_id` and up to 1000
ordered source `entries`. The destination assigns its own ledger ID and
sequence while retaining source ledger ID, source sequence, source entry
hash, source assurance, and source provenance in the destination envelope's
`provenance`.

- Identical event IDs and content are no-ops on retry.
- Reusing an event ID with different content is a conflict.
- Source sequences in a batch must be strictly ascending and contiguous.
- Each incoming v1 envelope is self-hash validated; predecessor hashes are
  checked between adjacent hashed entries in the batch.
- Imports are preflighted before the first write. JSONL cannot make a
  multi-record batch transactional: each fsynced record is a committed
  prefix. An I/O failure reports `committed_prefix` and
  `retry_after_source_sequence`; retry the batch because already committed
  entries are idempotent.

A portable pull/push cycle is non-destructive: read from either ledger,
import into the other, and never consume or delete source entries. Pin the
source head sequence for the duration of a transfer. For a complete
bidirectional parity exercise, use `parity/ledger_http_transfer.py` with
`POC_LEDGER_URL` and `ENSEMBLE_LEDGER_URL` set to the two live API base
URLs.

## Error behavior

| HTTP status | Meaning |
|---:|---|
| 200 | Read/verify/import success, duplicate append, or verification result (inspect `ok`) |
| 201 | New append |
| 400 | Invalid query, event, sequence, checkpoint, or envelope |
| 404 | Event ID not found |
| 409 | Event identity/content conflict |
| 500 | Corrupt local ledger or I/O failure; import I/O errors include committed-prefix details |

The POC ledger API is loopback-only through the Memory service. It does not
introduce authentication, a separate ledger daemon, or a second persistence
backend.
