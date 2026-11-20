# Dataset ingestion: inventory, resolution, acquisition, indexing, graph and verification (PR 1–PR 5)

This package implements sections 5 (PR 1), the resolution stage (PR 2), the
acquisition/archive/adapter stage (PR 3), the normalization/integration
stage (PR 4) and the graph projection, resume/retry hardening and retrieval
verification stage (PR 5) of the arbitrated Awesome Public Datasets ingestion
design:

- **PR 1**: full-catalog inventory, license/format/scheme classification, an
  optional bounded accessibility probe, a manifest, a report and one summary
  ledger event.
- **PR 2**: resolution of every catalog entry to its authoritative dataset
  source, with explicit reason codes, a resumable cache and an accounting
  invariant. Resolution never downloads dataset bodies.
- **PR 3**: streaming, bounded, resumable acquisition of `index_content`
  entries whose license permits it, into a content-addressed store; hostile
  archive inspection and member-by-member extraction; and format adapters for
  CSV/TSV, JSON, JSONL/NDJSON, XML, text/Markdown and (optionally) Parquet.
- **PR 4**: deterministic normalization into the canonical record envelope,
  bounded text projection and sharding, submission to Knowledge, chunking,
  embedding, lexical/vector indexing, per-record outcomes and content-hash
  idempotency.
- **PR 5**: versioned graph projection of the catalog's relationships
  (datasets, categories, organizations, source repositories, formats,
  licenses) into the Graph service; crash-safe per-entry checkpoints with
  bounded retries; and verification that ingested content is retrievable
  through the lexical and vector services and that graph traversal answers a
  category query.

Still out of scope (and deliberately not built here): nested archive
expansion, historical content retention beyond the revision high-water mark,
and entity extraction into the graph.

## Commands

```bash
PYTHONPATH=services python3 -m ingestion.dataset_ingest inventory \
    [--catalog-ref master | --catalog-dir PATH] [--probe] [--limit N] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest resolve \
    [--catalog-ref master | --catalog-dir PATH] [--sample N | --limit N] \
    [--no-cache] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest acquire \
    [--catalog-ref master | --catalog-dir PATH] [--sample N | --limit N] \
    [--dry-run] [--max-files N] [--max-bytes N] \
    [--no-resume] [--no-reuse] [--no-resolve] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest normalize \
    [--catalog-ref master | --catalog-dir PATH] [--sample N | --limit N | --entry ID] \
    [--include-metadata-only] [--dry-run] [--no-journal] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest graph \
    [--catalog-ref master | --catalog-dir PATH] [--sample N | --limit N | --entry ID] \
    [--dry-run] [--check-tombstones] [--prune-tombstoned [--apply]] \
    [--no-checkpoints] [--max-attempts N] [--format md|json] [--out FILE] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest verify \
    [--manifest PATH] [--query TEXT]... [--category NAME]... [--top-k N] \
    [--no-graph] [--format md|json] [--out FILE] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest report [--format md|json]
```

`report` renders the newest run (inventory, resolution, acquisition,
integration, graph or verification) from `runs/`.

For `normalize`, the default selection is every entry PR 3 acquired
(`acquired`/`unchanged`). `--include-metadata-only` also integrates a metadata
record for license-unknown entries whose policy is `metadata_only_review`, so
they become discoverable without indexing their content. `--dry-run` prints
the full record plan (metadata record, content shards, files and reasons) and
makes no service calls and no writes. Every discovered entry still appears in
the accounting: entries outside the selection are `deferred` /
`not_selected`.

For `acquire`, only entries whose `disposition` is `index_content` and whose
license policy allows content indexing are candidates. `--sample N` is a
deterministic round-robin sample stratified by primary format, so a bounded run
spans the format classes; `--limit N` takes the first `N` candidates by
`entry_id`. Missing resolution evidence is resolved in-process through the PR 2
code path unless `--no-resolve` is given. `--dry-run` prints the plan (as JSON)
and makes no network requests and no writes.

A second `acquire` run is offline: resolutions come from the resolution cache,
content comes from the per-entry acquisition cache and the content store, and
the run reports `unchanged`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Accounting invariant holds |
| 1 | Unhandled per-entry errors or a global failure |
| 2 | Accounting invariant violation (fail closed) |
| 3 | Budget/`--limit` cap reached |

`--limit N` classifies at most `N` entries; the rest are still represented in
the manifest as `deferred` / `run_budget_exhausted`, so the accounting
invariant always holds. `--probe` is opt-in and uses HEAD with a
`GET Range: bytes=0-0` fallback, global concurrency 4 and at most one request
per second per host. No response bodies are stored.

For `resolve`, `--sample N` is a deterministic round-robin sample stratified by
disposition (`index_content`, `metadata_only_review`, `blocked`), so a bounded
acceptance run always spans every disposition class. `--limit N` takes the
first `N` processable entries by `entry_id`. Rows outside the sample are still
written with `deferred` / `run_budget_exhausted`, and catalog-level failures
keep their own reason, so `discovered == count(resolution statuses)` always
holds. `--no-cache` disables both cache reads and writes.

## Graph projection (PR 5)

`graph` derives relationships from the PR 1 inventory rows and asserts them in
the Graph service over its documented REST surface (`/graph/add-node`,
`/graph/add-edge`, `/graph/query`, `/graph/traverse`, `/graph/remove-nodes`).
It writes nothing that is not derived from the pinned catalog commit, and it is
a projection: deleting the graph and re-running the command rebuilds it.

Because the Graph service never mutates a node, the model is versioned:

| Node | Id |
|---|---|
| `Dataset` | `dataset:{sha256(entry_id)[:32]}` |
| `DatasetVersion` | `dataset_version:{metadata_sha256}` |
| `Category` | `category:{slug}-{sha256(name)[:8]}` |
| `Organization` | `organization:{slug}-{hash}` |
| `Repository` | `repository:{host}/{owner}/{repo}` |
| `Format` | `format:{slug}-{hash}` |
| `License` | `license:{family}` |

Edges (`has_version`, `in_category`, `published_by`, `hosted_at`,
`has_format`, `licensed_under`) originate at the `Dataset` node for stable,
path-derived facts and at the `DatasetVersion` node for metadata-derived facts.
**Node properties contain only values intrinsic to their identity**; every edge
carries the provenance of the assertion (`origin`, `catalog_commit`,
`locator`, `entry_id`, `metadata_sha256`). A new commit therefore produces new
edge ids, and re-running the same commit produces identical ids that the
service returns as a no-op.

Idempotency is the composition of three things:

1. Deterministic node and edge ids, so a retry can never duplicate a record.
2. A prefetched snapshot of the existing graph, so writes that are already
   present are skipped and counted instead of re-sent. The snapshot is also
   compared, so an existing record with *different* content is reported as
   `graph_conflict` rather than silently accepted.
3. A per-entry checkpoint (see below), so a resumed run does not even look at
   entries that already finished.

Tombstones are honoured in both directions: an entry a previous integration run
recorded as tombstoned is never projected (`skipped` / `source_tombstoned`), and
`--prune-tombstoned` removes the `Dataset` node and its `DatasetVersion` nodes
through `/graph/remove-nodes` — a dry run unless `--apply` is given.
`--check-tombstones` additionally asks the Index service, one bounded GET per
selected entry.

`--dry-run` builds the projection, reads the existing graph and prints exactly
what would be created; it writes no nodes, no manifest, no run report and no
ledger event.

## Resume and retries (PR 5)

Every stage writes its result at the end of the run, so a process killed
mid-run used to lose the per-entry progress it had already made. PR 5 adds
`cache/checkpoints/<stage>.jsonl`: an append-only, fsynced line per completed
entry, written the moment the entry reaches a terminal state.

- Each line carries a **fingerprint** of the inputs that produced it (stage
  versions, catalog commit, metadata hash, license policy, acquisition
  evidence). A checkpoint is reused only when the fingerprint still matches;
  changed inputs re-process the entry instead of replaying a stale result.
- A reused success is reported as `unchanged` with reason `checkpoint_resume`,
  because nothing was written in the current run. Recorded failures keep their
  original status. The previous stage manifest is a second resume source (with
  weaker evidence: stage versions and the metadata hash only), which is what
  lets work done before checkpoints existed still be reused.
- `--max-attempts N` bounds attempts **per run** for outcomes flagged
  retryable, with exponential backoff (`RESUME_BASE_DELAY_SECONDS`,
  `RESUME_MAX_DELAY_SECONDS`). A lifetime budget (`RESUME_MAX_TOTAL_ATTEMPTS`,
  default 12) stops an entry that fails identically across runs; it is then
  `deferred` / `retry_budget_exhausted`.
- Retried operations are idempotent by construction: Knowledge is a revision
  gate, the vector/lexical ids are content-derived, and graph ids are
  deterministic. A retry therefore never creates a second revision, chunk or
  edge.
- `--no-checkpoints` disables the stage's checkpoint (reads and writes).

## Verification (PR 5)

`verify` reads the newest integration manifest, derives one query per indexed
entry (the title without its parenthetical aside) and asks the **actual**
retrieval services:

- lexical: `POST {lexical}/knowledge/search`
- vector: `POST {embedding}/embeddings` then
  `POST {vector}/collections/knowledge_chunks/query`
- hybrid: `POST {hybrid}/knowledge/search` (reported, not required)
- graph: `POST {graph}/traverse` from each of the three busiest categories of
  integrated entries, direction `in`, which must return the related datasets
  and their organizations, formats and licenses.

A read-only leg is retried once when a dependency is transiently unavailable
(`VERIFY_ATTEMPTS`, default 2); a genuine *miss* is never retried. A leg that
cannot run is reported `unavailable` with the service's own reason
(for example the hybrid service answering `{"ok": false, "error": {"code":
"service_unavailable", "message": "embedding: connection refused"}}`), never
counted as a pass. The vector leg is only *required* when the store is
functional (a non-empty `knowledge_chunks` collection); a functional store that
does not return the expected source fails the run. `--query TEXT` runs an
ad-hoc query with no known expected source; those results are reported as
`informational` and do not change the verdict.

Exit codes are 0 when every required leg verifies, 1 when there is nothing
ingested to verify, and 2 when a required leg fails.

## Resolution order (PR 2)

For each entry, candidates are tried in this order and no dataset body is
downloaded:

1. `sources[].access_url`, accepting `str`, `list[str]` and `list[dict]`.
2. The GitHub `blob` (or `raw`) -> `raw.githubusercontent.com` transform.
3. Documented host APIs:
   - GitHub repository / contents (`api.github.com`),
   - Zenodo records (`zenodo.org/api/records/{id}`, including Zenodo DOIs),
   - Hugging Face datasets (`huggingface.co/api/datasets/{repo}`),
   - DOI content negotiation (`application/vnd.citationstyles.csl+json`),
   - archive.org metadata (`archive.org/metadata/{id}`),
   - Dataverse `:persistentId` dataset/file APIs.
4. One bounded same-host landing-page fetch (HTML only, link count and bytes
   capped, `robots.txt` respected).

Direct-file confirmation uses `HEAD` with a `GET Range: bytes=0-0` fallback and
`store_body=False`; only bounded API JSON and one bounded landing page per
entry are read. Every attempt is recorded with a stable reason code, and when
several candidates fail the most informative reason wins (for example
`gated_authentication` or `not_found` rather than `landing_page_only`).

Resolution statuses are `resolved`, `unresolved`, `deferred` (retryable) and
`failed` (inherited catalog failure). Every non-`resolved` entry carries a
`reason_code`. Observed reasons include `invalid_url` (including the catalog's
`hhttps` typo), `unsupported_scheme`, `no_source_url`, `ssrf_blocked`,
`unsafe_redirect`, `too_many_redirects`, `gated_authentication`,
`robots_disallowed`, `requires_interaction`, `requires_javascript`, `gone`,
`not_found`, `rate_limited`, `blocked`, `timeout`, `dns_failure`,
`tls_failure`, `connection_failed`, `server_error`, `oversized_declared`,
`doi_unresolved`, `landing_page_only`, `resolver_not_implemented`,
`http_error`, `parse_error`, `resolver_error` and `run_budget_exhausted`.
`no_source_url` and `resolver_error` are explicit additions to the design's
taxonomy; everything else maps onto its table.

### Idempotency and resume

Resolution results are cached per entry at
`cache/resolutions/<sha256(entry_id, metadata_sha256, resolver_version)>.json`.
A re-run re-uses:

- a `resolved` result until `RECHECK_AFTER_DAYS`,
- a retryable failure until `next_retry_at`
  (`RESOLVE_RETRY_BACKOFF_SECONDS`, default 1 hour),
- a terminal failure indefinitely.

The key changes when the entry's `metadata_sha256` changes or the resolver
version is raised. Each result is written atomically as soon as it is produced,
so an interrupted run resumes where it stopped. Outcomes are appended with a
deterministic `event_id`, so a repeated run appends nothing.

## Artifacts

Under `POC_CLAUDE_ENSEMBLE_DATASET_DATA_ROOT` (default
`~/.FlossWare/poc-claude-ensemble/data/datasets`):

```
catalog/<repo>@<commit>/      # validated snapshot (remote mode only)
manifest/<commit>.jsonl       # inventory: one row per discovered entry, deterministic
manifest/<commit>.resolution.jsonl
                              # resolution: one row per discovered entry, deterministic
manifest/<commit>.acquisition.jsonl
                              # acquisition: one row per discovered entry
outcomes.jsonl                # append-only stage outcomes, deduplicated by event_id
cache/probes/<sha256>.json    # probe evidence only; no bodies or query strings
cache/resolutions/<sha256>.json
                              # resolution evidence only; no API or HTML bodies
cache/acquisitions/<sha256>.json
                              # per-entry acquisition evidence and parse summaries
cache/blobs/<sha[:2]>/<sha256>
                              # acquired content, content-addressed, mode 0600
cache/partial/<sha[:2]>/<sha>.part
cache/partial/<sha[:2]>/<sha>.json
                              # in-progress download plus its URL/validator sidecar
tmp/archive-*/                # 0700 extraction roots, removed after parsing
manifest/<commit>.integration.jsonl
                              # integration: one row per discovered entry
manifest/<commit>.graph.jsonl # graph: one row per discovered entry (PR 5)
cache/checkpoints/<stage>.jsonl
                              # append-only, fsynced per-entry completion (PR 5)
cache/integrations/<revision>.json
                              # downstream-indexing completion journal (PR 4)
runs/<run_id>.json            # latest run report
```

Blob names are SHA-256 digests; a filename supplied by a remote server is never
used as a path. Completed blobs are adopted with an atomic rename, so an
interrupted run never leaves a partially written blob under a content hash.

### Acquisition manifest row (PR 3)

One row per discovered entry, so `discovered == count(acquisition statuses)`.
Each row carries `entry_id`, `metadata_sha256`, `catalog{repo,commit,path,
meta_url}`, `category`, `title`, `disposition`, `license_policy`, `formats` and
an `acquisition` block:

```json
{
  "status": "acquired|unchanged|skipped|deferred|failed",
  "reason_code": null,
  "files_total": 2, "files_acquired": 2, "files_failed": 0,
  "bytes_total": 1234, "records_total": 42,
  "parse_status": "parsed|skipped|failed|null",
  "retryable": false, "next_retry_at": null, "attempts_count": 1,
  "checked_at": "2026-01-01T00:00:00Z",
  "resolution_method": "zenodo_api",
  "cached": false,
  "next_action": "normalize|resume|resolve|review|retry",
  "files": [
    {
      "uri": "https://…", "member": "table.csv", "kind": "direct_file",
      "status": "acquired", "media_type": "text/csv",
      "declared_bytes": 1234, "bytes": 1234, "sha256": "…",
      "blob": "cache/blobs/ab/ab…", "checksum_declared": "md5:…",
      "checksum_verified": true, "etag": "…", "last_modified": "…",
      "final_url": "https://…", "retrieved_at": "…",
      "resumed": false, "cached": false, "attempts": 1,
      "parse": {"adapter": "csv", "status": "parsed", "record_count": 40,
                "fields": ["name", "value"], "truncated": false,
                "sample": [{"index": 0, "kind": "row", "data": {"name": "a"}}]},
      "archive": null
    }
  ]
}
```

For an archive, `parse` is `null` and `archive` carries the inspection
(`format`, `member_count`, `total_uncompressed`, `ratio`, `nested`, `junk`,
`members[]`) plus one `parse` artifact per extracted data member.

### Acquisition statuses and reason codes (PR 3)

- `acquired`: at least one file was downloaded and stored this run.
- `unchanged`: the resolved content was already stored and reused (no network).
- `skipped`: refused by policy or scope — `license_unknown_metadata_only`,
  `license_restricted`, `license_review_required`, `no_source_url`,
  `not_resolved`, `repository_requires_clone` (a repository/collection source
  names a whole repository; cloning is out of scope, so the HTML page it points
  at is not downloaded).
- `deferred`: retryable — `run_budget_exhausted`, `disk_guard`,
  `oversized_declared`, `oversized_stream`, `checksum_mismatch`,
  `partial_download`, `unsupported_format`, `adapter_unavailable`,
  `encoding_error`, `limits_exceeded`, plus network reasons
  (`not_found`, `blocked`, `rate_limited`, `timeout`, `dns_failure`,
  `tls_failure`, `connection_failed`, `server_error`) inherited from the
  resolution stage.
- `failed`: terminal — `content_type_mismatch` (including an HTML page served
  as a data file, and binary bytes served as text), `ssrf_blocked`,
  `unsafe_redirect`, `too_many_redirects`, `gated_authentication`, and catalog
  failures inherited from PR 1.
- Archive outcomes: `archive_unsafe_path`, `archive_link_or_device`,
  `archive_bomb`, `archive_too_many_entries`, `archive_encrypted`,
  `archive_nested`, `parse_error`, `empty_content`.
- Parse outcomes: `parse_error`, `encoding_error`, `limits_exceeded`,
  `xml_unsafe`, `unsupported_format`, `adapter_unavailable`, `empty_content`.

A re-run on the same snapshot produces a byte-identical manifest and appends
no duplicate outcome or ledger events.

### Canonical record and integration (PR 4)

Every record carries the design's canonical envelope: `source_id`, `title`,
`content`, `content_sha256`, `origin`, `locator` and `revision_id` (the
Knowledge `revision` value). Two families exist:

| Family | `source_id` | Content |
|---|---|---|
| `metadata` | `<entry_id>` | deterministic text projection of the manifest row |
| `content` | `<entry_id>/src/<sha256(source_url)[:16]>/shard/<nnnn>` | one shard of one acquired file or archive member |

`metadata` records exist for every integrated entry, including
license-unknown ones. `content` records exist only when the license policy is
`index_content` and PR 3 stored the bytes. `origin` is
`apd-core@<commit>`; `locator` is the sanitized upstream URL (never a
query-string-bearing one). Long entry ids keep the 512-character service limit
by hashing the entry id in the shard prefix.

Text projection is deterministic and inert: tabular rows become `field: value`
lines in header order, JSON objects/scalars become canonical JSON (objects as
`field: value`), text lines stay verbatim, and every shard starts with a
`[title] filename` header. Shards are bounded by `MAX_SHARD_CHARS`
(3,500,000 characters, under Knowledge's 4,000,000 cap) and `MAX_SHARD_BYTES`
(3,500,000 UTF-8 bytes, under the 8 MiB request-body cap). A single oversized
record is split deterministically.

Each record is submitted with the index-derived sequence:

```
state = GET index /sources/{source_id}
state.revision_id == desired  -> unchanged (no writes)
otherwise                     -> POST knowledge /records with high_sequence + 1
accepted                      -> POST chunking /chunk (window-v1)
                              -> POST embedding /embeddings (64 texts per call)
                              -> POST lexical /knowledge/documents
                              -> POST vector /collections/knowledge_chunks/upsert
                              -> POST vector .../delete-superseded (only when this
                                 revision replaces an older one)
```

`revision_id` is a hash of the record's `source_id`, content hash, normalized
file hash, adapter version, license policy and profile versions, so any change
in content, metadata, adapter or profile produces a new revision, and an
unchanged record is a no-op. A re-run of an indexed entry makes **no** service
call.

Partial failure has an explicit recovery path. If Knowledge accepted a
revision but a downstream stage failed, the next run sees the same
`revision_id` and re-runs only chunk/embed/index (Knowledge would call a second
submission a duplicate). A small rebuildable completion journal at
`cache/integrations/<revision_id>.json` records which revisions finished, so a
healthy re-run is still a pure no-op.

#### Integration statuses and reason codes

Entry statuses are `indexed`, `metadata_indexed`, `unchanged`, `deferred`,
`skipped` and `failed`. Every status other than `indexed`/`metadata_indexed`
carries a `reason_code`. Record-level outcomes are `indexed`, `unchanged`,
`skipped` or `failed`; every non-indexed/unchanged record carries its own
reason (`parse_error`, `adapter_unavailable`, `license_unknown_metadata_only`,
`source_tombstoned`, `knowledge_rejected`, `index_conflict`,
`dependency_unavailable`, `payload_too_large`, ...). The report's
"Non-indexed records" table lists them explicitly.

A retryable downstream failure leaves the entry `deferred`, not `failed`, so
the next run heals it without a second Knowledge revision.

### Integration manifest row (PR 4)

`manifest/<commit>.integration.jsonl` has one row per discovered entry:

```json
{
  "schema_version": 1,
  "entry_id": "apd:Test/example.yml",
  "catalog": {"repo": "...", "commit": "...", "path": "...", "meta_url": "..."},
  "category": "Test", "title": "Example", "license_policy": "index_content",
  "metadata_sha256": "…",
  "integration": {
    "integrate_version": 1, "normalize_version": 1, "indexing_version": 1,
    "adapter_version": 2,
    "status": "indexed", "reason_code": null, "retryable": false,
    "records_total": 40, "shards": 1, "chunks": 3, "vectors": 3,
    "metadata": {"source_id": "...", "kind": "metadata", "status": "indexed",
                 "revision_id": "…", "revision_sequence": 1,
                 "knowledge_decision": "accepted", "chunks": 1, "vectors": 1},
    "content": {"shards": 1, "indexed": 1, "unchanged": 0, "skipped": 0,
                "failed": 0, "records": [{"source_id": ".../shard/0000",
                "kind": "content", "status": "indexed", "revision_id": "…",
                "content_sha256": "…", "revision_sequence": 1,
                "knowledge_decision": "accepted", "chunks": 2, "vectors": 2}]},
    "files": [{"uri": "https://…", "member": null, "status": "parsed",
               "adapter": "csv", "records": 40, "shards": 1}],
    "tombstoned": []
  }
}
```

`outcomes.jsonl` gains one `stage: "integrate"` row per record (plus one for a
record-less entry) with the design's provenance fields:
`knowledge_source_id`, `revision_id`, `revision_sequence`,
`knowledge_decision`, `chunks`, `vectors`. The `event_id` is deterministic, so
a repeated run appends nothing new.

### Resolution manifest row (PR 2)

Each row carries `entry_id`, `metadata_sha256`, `catalog{repo,commit,path,
meta_url}`, `category`, `title`, `disposition`, `license{family,policy}` and a
`resolution` block:

```json
{
  "status": "resolved|unresolved|deferred|failed",
  "reason_code": null,
  "method": "head_probe|github_blob_raw|github_repo_api|github_contents_api|zenodo_api|huggingface_api|doi_content_negotiation|archive_org_metadata_api|dataverse_api|landing_page",
  "kind": "direct_file|archive|repository|collection",
  "version": "upstream commit/branch, HF sha, Zenodo version or Dataverse version",
  "source_count": 1,
  "total_sources": 1,
  "retryable": false,
  "next_retry_at": null,
  "checked_at": "2026-01-01T00:00:00Z",
  "next_action": "acquire",
  "sources": [{"uri": "https://…", "kind": "direct_file", "method": "…",
               "media_type": "text/csv", "size_bytes": 123, "version": null,
               "member": null, "evidence_url": "https://…"}],
  "attempts": [{"url": "https://…", "method": "…", "status": "resolved",
                "reason_code": null, "http_status": 200, "detail": null}]
}
```

`total_sources` is the number of data-like members the resolver saw; the
`sources` list is capped by `MAX_RESOLVED_SOURCES` (default 20), so truncation
is numeric and explicit rather than silent. All URLs in artifacts have query
strings, fragments and userinfo stripped.

### Manifest row (PR 1)

Each row carries `entry_id`, `catalog{repo,commit,snapshot_sha256,mode,path,
file_sha256,bytes,meta_url,readme_status}`, the normalized catalog fields,
`extra` (all unknown keys preserved), `metadata_sha256`, `license{catalog,
metadata_repo,dataset{declared_raw,normalized,family,ambiguous,policy,
reason_code,review_required,evidence}}`, `classes`, `probe`, `disposition`,
`status`, `reason_code` and `next_action`.

`classes` is:

```json
{
  "access": "public|gated|blocked|inaccessible|unknown",
  "access_reason": null,
  "size": {"class": "unknown|small|medium|large", "bytes": null, "source": "metadata|probe"},
  "format": ["csv", "json"],
  "scheme": ["https"],
  "scheme_unsupported": []
}
```

PR 1 statuses are `deferred` (awaiting resolution/metadata indexing),
`skipped` (blocked license) and `failed` (catalog parse/validation error).
Every non-success row has a `reason_code`, and the report refuses to exit 0
unless `discovered == count(statuses)` and every non-success row has a reason.

## Configuration

Every knob is read from `POC_CLAUDE_ENSEMBLE_DATASET_<KNOB>`; CLI flags
override environment variables. The PR 1-relevant knobs are:

| Knob | Default |
|---|---|
| `DATA_ROOT` | `~/.FlossWare/poc-claude-ensemble/data/datasets` |
| `CATALOG_REPO` / `CATALOG_REF` | `awesomedata/apd-core` / `master` |
| `README_REPO` / `README_REF` | `awesomedata/awesome-public-datasets` / `master` |
| `CATALOG_DIR` / `README_PATH` | unset (local checkout / explicit README) |
| `CATALOG_MAX_FILES` / `CATALOG_MAX_FILE_BYTES` | 5,000 / 256 KiB |
| `YAML_MAX_DEPTH` / `YAML_MAX_NODES` / `YAML_MAX_ALIASES` | 32 / 10,000 / 0 |
| `CONCURRENCY` / `PER_HOST_RPS` | 4 / 1.0 |
| `LICENSE_ALLOW_FAMILIES` | public-domain, cc0, cc-by, cc-by-sa, mit, apache, odbl, ogl |
| `RECHECK_AFTER_DAYS` | 30 |
| `SAMPLE` / `LIMIT` | unset (resolve selection; CLI flags override) |
| `LANDING_PAGE_MAX_BYTES` / `LANDING_PAGE_MAX_LINKS` | 512 KiB / 200 |
| `API_MAX_BYTES` | 2 MiB |
| `MAX_RESOLVED_SOURCES` | 20 |
| `RESOLVE_RETRY_BACKOFF_SECONDS` | 3600 |
| `RESOLVE_CACHE_ENABLED` | true |
| `GITHUB_TOKEN` | unset (sent only to `api.github.com`) |
| `GRAPH_TIMEOUT` | 30 s |
| `GRAPH_CHECK_TOMBSTONES` | false (also asks Index; `--check-tombstones`) |
| `GRAPH_PRUNE_ENABLED` | true |
| `VERIFY_TOP_K` / `VERIFY_QUERIES` / `VERIFY_TIMEOUT` | 5 / 5 / 30 s |
| `RESUME_CHECKPOINTS_ENABLED` | true (`--no-checkpoints` disables) |
| `RESUME_MAX_ATTEMPTS` | 3 per run (`--max-attempts`) |
| `RESUME_MAX_TOTAL_ATTEMPTS` | 12 across runs |
| `RESUME_BASE_DELAY_SECONDS` / `RESUME_MAX_DELAY_SECONDS` | 0.25 s / 4 s |

Service URLs are never hard-coded: they come from the existing
`POC_CLAUDE_ENSEMBLE_{GRAPH,KNOWLEDGE,CHUNKING,EMBEDDING,LEXICAL,VECTOR,INDEX,HYBRID}_URL`
variables, defaulting to the loopback ports the services already use.

The full table (including limits reserved for earlier stages) lives in
`config.py`.

## Security posture

- YAML is loaded with `yaml.SafeLoader` only, after bounding file size, depth,
  node count and alias expansion. Aliases are rejected by default.
- Duplicate keys are detected and recorded as warnings; the pinned snapshot
  contains one benign duplicate-key entry, which is retained rather than
  silently dropped.
- Catalog tarballs are expanded member by member; `extractall` is never used.
  Absolute paths, `..`, backslashes, drive letters, NULs, links, devices,
  duplicate names, over-cap members and mismatched root directories are
  rejected.
- Local directories are walked without following symlinks; every resolved
  path must remain under the catalog root.
- The probe transport pins the connection to the validated IP, rejects
  non-global addresses (loopback, private, link-local, ULA, CGNAT), revalidates
  every redirect hop, refuses HTTPS→HTTP downgrades and never re-resolves DNS.
- Manifest, probe and ledger URLs are sanitized (no userinfo, query or
  fragment). Ledger rows contain only hashes, counts and enum projections.
- No downloaded content is executed. A static test checks the package for
  `eval`, `exec`, shell/subprocess use, `pickle`/`marshal`/`shelve` and unsafe
  YAML loaders.
- Every resolver request goes through `netguard`: only `http`/`https`, no
  userinfo, standard ports, every resolved address must be globally routable,
  the connection is pinned to the validated IP, every redirect hop is
  revalidated, and HTTPS→HTTP downgrades are refused. A redirect to
  `169.254.169.254` is a test case.
- Resolver API JSON, landing HTML and link counts are bounded. Only `href`
  targets and text are extracted; `javascript:`, `data:`, `file:`, `mailto:`
  and control-character URLs are ignored. API and HTML bodies are never written
  to the resolution cache.
- `robots.txt` is fetched (bounded) before a landing page and a broken or
  missing file is treated permissively rather than as a hard failure.

## Measured snapshot facts (PR 1)

At `awesomedata/apd-core` commit
`92d1b210bfba28055f672e6331eb5e5fa27227c5`:

- 1,142 metadata files: 1,126 `.yml`, 11 `.yaml`, 4 extensionless and 1 `.wml`.
- 1,142/1,142 README `Meta` links reconcile (1,010 OK, 132 FIXME).
- YAML aliases: **0** files. The default `YAML_MAX_ALIASES=0` is therefore
  safe; raise it only with measured cause.
- One file (`GIS/Crusader-Atlas.yml`) has duplicate keys; it is retained with a
  `duplicate_key:*` warning.

## Measured PR 2 sample (2026-11-19)

A real `resolve --sample 60` against the pinned local snapshot
(`92d1b210…`, 1,142 entries), with a 5 s connect / 10 s read budget:

- 60 entries processed: 28 `metadata_only_review`, 28 `index_content`,
  4 `blocked`; 1,082 entries marked `deferred` / `run_budget_exhausted`.
- 31 `resolved`, 23 `deferred`, 6 `unresolved`.
- Resolution methods used: `landing_page` 30, `source_access_url` 9,
  `head_probe` 8, `github_repo_api` 6, `zenodo_api` 4,
  `github_contents_api` 1, `huggingface_api` 1.
- Explicit non-resolution reasons observed: `landing_page_only` 13,
  `connection_failed` 4, `not_found` 3, `robots_disallowed` 2,
  `requires_interaction` 2, and one each of `invalid_url`, `timeout`,
  `unsafe_redirect`, `oversized_declared`, `oversized_stream`.
- `discovered == 1,142 == count(statuses)`; every non-`resolved` row has a
  reason. A second run made zero network calls (60 cache hits), finished in
  about 1.6 s, appended no outcomes and produced a byte-identical manifest.

These counts are a bounded sample, not a yield guarantee: a large share of the
catalog is homepage-only and many landing pages expose no data-like links.

## Measured PR 3 acquisition (2026-11-19)

Real runs over the network through `netguard`, against the pinned local
snapshot (`92d1b210…`, 1,142 entries), with `--no-ledger`:

```bash
PYTHONPATH=services python3 -m ingestion.dataset_ingest acquire \
    --catalog-dir /tmp/apd-design/apd-core \
    --sample 32 --max-files 2 --max-bytes 52428800 --no-ledger
```

- 32 of the 441 `index_content` entries were selected (round-robin by primary
  format). The remaining 1,110 rows carry an explicit status, so
  `discovered == 1,142 == count(statuses)`.
- 20 entries ended up with stored content: 23 files, 68,048,494 bytes and
  201,875 normalized records (adapter/record counts only; nothing is indexed).
- Adapters that produced records: csv 9 files, json 8, text 2, markdown 2.
  Two PDFs were acquired and refused with `adapter_unavailable`, which is the
  documented PDF policy.
- A targeted `--entry` run also stored a 3.7 MB `gzip` member
  (`entries.csv`, 28,032,537 uncompressed bytes); it was expanded
  member-by-member and parsed into 93,634 rows with 15 fields.
- Reason codes observed in the final run: `license_unknown_metadata_only` 644,
  `license_restricted` 57, `run_budget_exhausted` 409,
  `repository_requires_clone` 6, `not_found` 2, `connection_failed` 2,
  `landing_page_only` 1, `rate_limited` 1, `oversized_stream` 1.
  `repository_requires_clone` is the only addition to the design's taxonomy
  here: a GitHub `repository` source names a repo page, and PR 3 refuses it
  rather than downloading HTML.
- **Resume**: a second identical run made zero network calls
  (`budget.bytes_used = 0`, 20 acquisition-cache hits, 0 misses) and finished
  in 2.7 s. A partial entry (one file stored, one source 404) keeps its stored
  file and retries only the failing source on the next run.
- A bounded selection exits 3 (a `--sample` is a cap), matching `resolve`;
  `--entry <id>` runs exit 0.

Format coverage of the real sample was CSV (including CSV inside a gzip), JSON
(object, array and GeoJSON), `text/plain` and Markdown, plus PDF refusal. The
catalog declares **no** TSV entries at all, and every JSONL-declaring entry in
this sample either resolved to a repository or failed before download, so TSV
and JSONL parsing are covered by offline tests rather than by this run.

## Measured PR 5 graph, resume and verification (2026-11-19)

Catalog: `awesomedata/apd-core@92d1b210bfba28055f672e6331eb5e5fa27227c5`,
1,142 metadata files, 36 categories.

**Full projection.** `graph` over the whole inventory, against a throwaway
instance of the same Graph service (port 18766, empty store):

| Node | Count |
|---|---|
| `Dataset` | 1,142 |
| `DatasetVersion` | 1,142 |
| `Category` | 36 |
| `Organization` | 333 |
| `Repository` | 1,114 |
| `Format` | 15 |
| `License` | 17 |

- 3,799 unique nodes and 5,149 unique edges created; all 1,142 entries
  `indexed`; `discovered == 1,142 == status_total`; no failures.
- Wall clock ≈ 27 minutes. The cost is the Graph service's design: every
  `/graph/add-node` and `/graph/add-edge` re-serialises and fsyncs the whole
  store, so write time grows with the graph.

**Live shared graph.** The same command against the live service (port 8766,
an existing 11k-node, 19 MiB store) projects 40 entries:

- `SIGKILL` after 6 entries: the checkpoint held all 6 completed entries.
- Restart of the same command: `reused=6` (`unchanged` /
  `checkpoint_resume`), `attempted=34`, 147 nodes and 186 edges created, 68
  shared nodes already present, `graph_errors=0` (no conflict, no duplicate).
- Third run: 40 `unchanged`, zero writes, 1.8 s versus ~10 minutes for the
  incremental run.

**Traversal.** `verify --category ...` against the full projection, direction
`in`, edge `in_category`, depth 2:

| Category | Datasets | Organizations\* | Formats\* | Licenses\* |
|---|---|---|---|---|
| Government | 209 | 3 | 1 | 1 |
| SocialSciences | 66 | 3 | 2 | 6 |
| Healthcare | 50 | 7 | 6 | 6 |
| Biology | 50 | 1 | 0 | 3 |
| Finance | 18 | 4 | 8 | 5 |
| Education | 17 | 3 | 0 | 9 |

**Retrieval.** Five queries derived from the integrated entries, `top_k=5`,
against the live services:

| Query | Lexical hits | Vector hits | Top vector score |
|---|---|---|---|
| Furrow Farmland Price Index | 5 (match) | 5 (match) | 0.848 |
| India Agri Export Price Index | 5 (match) | 5 (match) | 0.822 |
| Indoor Light for Houseplants | 5 (match) | 5 (match) | 0.769 |
| Robot Lawn Mower Specs | 5 (match) | 5 (match) | 0.798 |
| Turkish Olive Production Data | 5 (match) | 5 (match) | 0.869 |

- Vector store functional: `chromadb`, `knowledge_chunks`, 627,554 vectors.
- Earlier runs recorded one or two vector queries as `unavailable`
  (`dependency_unavailable` from a transiently refused embedding call) rather
  than as misses; the leg retry now absorbs that, and a run with the
  embedding service healthy matches 5/5.
- The hybrid leg was `unavailable` for every query in every run because the
  deployed hybrid service reports `{"ok": false, "error": {"code":
  "service_unavailable", "message": "embedding: <urlopen error [Errno 111]
  Connection refused>"}}`. The report records that reason instead of counting
  it as a pass or a failure.

\* sampled over at most 10 datasets per category.

## Packaging note

`services/ingestion/__init__.py` is deliberately **not** added. Adding it was
tested and breaks the existing `pytest services` collection because the
existing tests in that directory use top-level imports (`import
ingestion_service`, `import acceptance`). `services/ingestion` remains a
PEP 420 namespace package, so the prescribed command
`python3 -m ingestion.dataset_ingest` still works with `PYTHONPATH=services`.
This is the packaging caveat recorded in the design; the fallback (a sibling
library directory) was not needed because the namespace package preserves both
the CLI path and existing test discovery.

For the same reason, `datasets/tests/__init__.py` defines a standard
``load_tests`` hook that returns an empty suite. The existing
`python -m unittest discover -s services/ingestion` CI job does not install
pytest; without the hook, discovery would import these pytest-based modules
and fail with `ModuleNotFoundError: pytest`. With the hook, unittest still
runs the 22 existing ingestion tests, while pytest collects this directory and
imports the shared helpers through `ingestion.datasets.tests.helpers`.
