# Corpus Export REST API

The canonical Ensemble gateway exposes ingested Knowledge records through a
read-only, paginated HTTP API. External applications can pull the corpus
without a SQLite handle, a filesystem path, or access to the ingestion
pipeline.

The gateway does **not** own corpus state. Records are written by the
Knowledge service; this API is a bounded read view over the same
`knowledge.sqlite3` database. There is no write, update, tombstone, or delete
endpoint here, and the gateway does not proxy corpus requests to another
Ensemble instance.

## Base URL

```text
http://127.0.0.1:8080
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/corpus/records` | List records with filters and keyset pagination |
| `GET` | `/api/v1/corpus/origins` | List each live origin with its record count |

Every other method returns `405 method not allowed`; unknown sub-paths return
`404`. Errors use the gateway's shared envelope:

```json
{"ok": false, "error_code": "invalid_request", "error": "limit must be between 1 and 1000"}
```

Clients that send `Accept: application/problem+json` receive the same envelope
adapted to RFC 9457, as with every other gateway error.

## List records

```http
GET /api/v1/corpus/records
```

| Parameter | Required | Default | Description |
|---|---:|---:|---|
| `limit` | No | `100` | Page size, integer from 1 through 1000 |
| `cursor` | No | | Opaque keyset cursor returned as `next_cursor` |
| `origin` | No | | Exact origin match |
| `source_id` | No | | Exact source ID match |

Each parameter may be supplied only once. Unknown parameters are rejected.
Blank filter values are rejected.

Records are ordered by `source_id`, then `revision_sequence`, and tombstoned
sources are excluded. Ordering and filtering are applied before the page is
cut, so a filtered walk is stable.

### Response

```json
{
  "ok": true,
  "records": [
    {
      "source_id": "rfc-8259",
      "revision_sequence": 1761000000000000000,
      "title": "JSON: The JavaScript Object Notation",
      "origin": "rfc-editor",
      "locator": "https://www.rfc-editor.org/rfc/rfc8259.txt",
      "content_sha256": "…",
      "content": "…",
      "content_length": 48213,
      "content_truncated": true
    }
  ],
  "next_cursor": "eyJ2IjoxLCJzb3VyY2VfaWQiOiJyZmMtODI1OSIsInJldmlzaW9uX3NlcXVlbmNlIjoxNzYxMDAwMDAwMDAwMDAwMDAwfQ",
  "total_hint": 4213,
  "limit": 100,
  "returned": 1,
  "content_budget_exhausted": false
}
```

* `next_cursor` is `null` on the last page.
* `total_hint` is the exact number of live records matching the filters at
  request time, ignoring the cursor. It is a hint, not a snapshot guarantee:
  ingestion can change it between pages.
* `content_length` is the full stored length in characters. `content` may be a
  prefix of that when `content_truncated` is `true`. `content_sha256` always
  describes the full stored content, never the prefix.
* `retrieved` is included when the store schema carries a `retrieved` or
  `retrieved_at` column. The current Knowledge schema has neither, so the field
  is omitted rather than reported as unknown.

### Pagination

Pagination is keyset-based, not offset-based. The cursor encodes the last
returned `(source_id, revision_sequence)` pair, so inserting or removing other
records between page requests cannot duplicate or skip rows the client is
already walking. Cursors are opaque: clients must pass the returned value back
verbatim and must not construct or modify one. A malformed cursor is rejected
with `400` and `error_code: "invalid_request"`.

```bash
curl -fsS 'http://127.0.0.1:8080/api/v1/corpus/records?limit=100' | jq
curl -fsS 'http://127.0.0.1:8080/api/v1/corpus/records?origin=rfc-editor&limit=25' | jq
curl -fsS 'http://127.0.0.1:8080/api/v1/corpus/records?limit=25&cursor=<next_cursor>' | jq
```

The cursor is a traversal position, not a consistent snapshot. A record that
is rewritten at a higher revision while a walk is in progress may be returned
again later in the same walk, because its key sorts after the cursor.

### Bounded responses

An export endpoint must not let one request read unbounded content, so each
record's `content` is truncated to
`POC_CLAUDE_ENSEMBLE_CORPUS_MAX_CONTENT_CHARS` characters (default 65536) and
flagged with `content_truncated`. Truncation was chosen over rejecting
oversized records because the Knowledge schema accepts content up to 4,000,000
characters: rejecting would make a page fail or silently omit rows based on
content size, and page composition would change as content changed. A flag
keeps the page shape stable and makes the loss explicit.

A page also stops early once its accumulated content reaches
`POC_CLAUDE_ENSEMBLE_CORPUS_MAX_PAGE_CONTENT_CHARS` characters (default
4,194,304, and never less than the per-record cap). When that happens,
`content_budget_exhausted` is `true` and `next_cursor` points at the first
omitted record, so a full export is still possible by following cursors. A
page always returns at least one record when one matches.

## List origins

```http
GET /api/v1/corpus/origins
```

Returns each live origin and its record count, sorted by origin. This endpoint
does not accept query parameters.

```json
{
  "ok": true,
  "origins": [
    {"origin": "arxiv", "count": 812},
    {"origin": "rfc-editor", "count": 4213}
  ]
}
```

## Read path and isolation

* The database is opened per request with a `mode=ro` SQLite URI. The
  connection cannot write, and the gateway never creates or migrates the
  schema. A missing or incompatible store returns `503` with
  `error_code: "corpus_unavailable"`.
* The Knowledge service keeps its own connection for ingestion. Export reads
  take only shared read locks for the duration of a bounded page query (2
  second busy timeout), so an export cannot hold a write lock or leave a
  partially read transaction open.
* The gateway process needs read permission on the database file and its
  directory. It does not need write permission.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `POC_CLAUDE_ENSEMBLE_CORPUS_DB` | `POC_CLAUDE_ENSEMBLE_KNOWLEDGE_DB`, else `~/.FlossWare/poc-claude-ensemble/data/sqlite/knowledge.sqlite3` | Store to export |
| `POC_CLAUDE_ENSEMBLE_CORPUS_MAX_CONTENT_CHARS` | `65536` | Per-record content cap, in characters |
| `POC_CLAUDE_ENSEMBLE_CORPUS_MAX_PAGE_CONTENT_CHARS` | `4194304` | Per-page accumulated content cap, in characters |

Invalid non-positive values for the two caps fail gateway startup rather than
silently falling back.
