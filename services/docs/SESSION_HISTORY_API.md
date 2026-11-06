# Session History REST API

The POC Memory service exposes the existing append-only session-events ledger
through a read-only HTTP API. The API reads the JSONL file configured by
`POC_CLAUDE_ENSEMBLE_LEDGER_ROOT`; by default, the ledger root is
`~/.FlossWare/ledger`.

This is a retrieval interface, not a second ledger implementation. The POC
does not provide a REST write endpoint. Existing event producers and the
Memory service's append/idempotency behavior remain responsible for writes.

## Base URL

The default Memory HTTP listener is:

```text
http://127.0.0.1:8767
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/history` | List events, optionally filtered and paginated |
| `GET` | `/history/{event_id}` | Retrieve the first matching event by ID |

### List events

```http
GET /history
```

Query parameters:

| Parameter | Required | Default | Description |
|---|---:|---:|---|
| `limit` | No | `50` | Page size, integer from 1 through 100 |
| `cursor` | No | `0` | Non-negative numeric offset into the filtered result set |
| `event_type` | No | | Exact event type match. Reads `event_type`, falling back to `kind` |
| `source` | No | | Exact source match. Reads `source`, falling back to `producer` |
| `correlation_id` | No | | Exact correlation ID match |
| `since` | No | | Inclusive lower timestamp bound, ISO-8601 |
| `until` | No | | Inclusive upper timestamp bound, ISO-8601 |

Each parameter may be supplied only once. Unknown parameters are rejected.
Timestamp values may use `Z` or an explicit UTC offset. A timestamp without
a timezone is interpreted as UTC. When both bounds are supplied, `since`
must not be later than `until`.

Time filtering uses each record's `timestamp`, falling back to
`captured_at`. Records without a timestamp are excluded when a time bound
is active.

### Pagination

Pagination is offset-based, applied after filtering. The response's
`pagination.next_cursor` is either the numeric offset to use in the next
request or `null` when there are no more results.

Example:

```bash
curl -fsS 'http://127.0.0.1:8767/history?event_type=session_end&limit=25' | jq
curl -fsS 'http://127.0.0.1:8767/history?limit=25&cursor=25' | jq
```

The cursor is not an opaque snapshot token. If events are appended between
page requests, the underlying result set can change; clients should treat
pagination as a best-effort traversal rather than a snapshot-consistent scan.

### Retrieve one event

```http
GET /history/{event_id}
```

The event ID is URL-decoded before matching. This endpoint does not accept
query parameters. If duplicate IDs are present in the source file, the first
matching record in append order is returned.

## Response format

A successful collection request returns HTTP 200:

```json
{
  "ok": true,
  "entries": [
    {
      "schema_version": 1,
      "producer": "claude-code",
      "kind": "session_end",
      "event_id": "event-123",
      "timestamp": "2026-11-05T18:00:00Z",
      "captured_at": "2026-11-05T18:00:01Z",
      "payload_sha256": "<sha256>"
    }
  ],
  "pagination": {
    "limit": 50,
    "next_cursor": null,
    "returned": 1,
    "total": 1
  }
}
```

The event object is returned as stored. The envelope fields shown above
describe the current session-events ledger format; producer-specific event
payload fields may vary.

A successful single-event request returns HTTP 200:

```json
{
  "ok": true,
  "entry": {
    "event_id": "event-123",
    "kind": "session_end"
  }
}
```

## Examples

List the first page:

```bash
curl -fsS 'http://127.0.0.1:8767/history' | jq
```

Filter by event type and source:

```bash
curl -fsS 'http://127.0.0.1:8767/history?event_type=session_end&source=claude-code' | jq
```

Filter by correlation ID:

```bash
curl -fsS 'http://127.0.0.1:8767/history?correlation_id=request-123' | jq
```

Filter by an inclusive time range:

```bash
curl -fsS 'http://127.0.0.1:8767/history?since=2026-11-05T00:00:00Z&until=2026-11-06T00:00:00Z' | jq
```

Retrieve a specific event:

```bash
curl -fsS 'http://127.0.0.1:8767/history/event-123' | jq
```

## Errors

| HTTP status | Meaning |
|---:|---|
| `400` | Invalid query parameter, duplicate parameter, invalid limit/cursor/timestamp, reversed time range, or invalid event ID path |
| `404` | Event ID does not exist, or route is not found |
| `500` | History could not be read safely |

Malformed JSONL records cause the request to fail with HTTP 500. The service
does not silently skip corrupt records, because doing so would make a
successful response look complete when it is not. Invalid event timestamps
also fail a time-filtered request rather than silently producing an
incomplete time range.

## Write boundary

There is intentionally no `POST /history`, `PUT /history`, or
`DELETE /history` endpoint in this POC. The REST API is read-only. This
keeps retrieval separate from event production and preserves the existing
append-only ledger as the source of truth.
