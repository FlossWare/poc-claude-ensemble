# Dataset inventory, classification, resolution and acquisition (PR 1 + PR 2 + PR 3)

This package implements sections 5 (PR 1), the resolution stage (PR 2) and the
acquisition/archive/adapter stage (PR 3) of the arbitrated Awesome Public
Datasets ingestion design:

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

Out of scope for PR 3 (and deliberately not built here): normalization and
text projection, Knowledge/Index/Lexical/Vector writes, Graph writes, retry
hardening beyond the recorded `next_retry_at`, and nested archive expansion.

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
PYTHONPATH=services python3 -m ingestion.dataset_ingest report [--format md|json]
```

`report` renders the newest run (inventory, resolution or acquisition) from
`runs/`.

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

The full table (including limits reserved for PR 2+) lives in `config.py`.

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
