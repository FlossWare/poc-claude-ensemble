# Dataset inventory and classification (PR 1)

This package implements **only** section 5 of the arbitrated Awesome Public
Datasets ingestion design: full-catalog inventory, license/format/scheme
classification, an optional bounded accessibility probe, a manifest, a report
and one summary ledger event.

Out of scope for PR 1 (and deliberately not built here): dataset downloads,
resolvers and host APIs, archive handling, adapters, normalization/indexing,
Knowledge writes and Graph writes.

## Commands

```bash
PYTHONPATH=services python3 -m ingestion.dataset_ingest inventory \
    [--catalog-ref master | --catalog-dir PATH] [--probe] [--limit N] [--no-ledger]
PYTHONPATH=services python3 -m ingestion.dataset_ingest report [--format md|json]
```

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

## Artifacts

Under `POC_CLAUDE_ENSEMBLE_DATASET_DATA_ROOT` (default
`~/.FlossWare/poc-claude-ensemble/data/datasets`):

```
catalog/<repo>@<commit>/      # validated snapshot (remote mode only)
manifest/<commit>.jsonl       # one row per discovered entry, deterministic
outcomes.jsonl                # append-only stage outcomes, deduplicated by event_id
cache/probes/<sha256>.json    # probe evidence only; no bodies or query strings
runs/<run_id>.json            # latest run report
```

A re-run on the same snapshot produces a byte-identical manifest and appends
no duplicate outcome or ledger events.

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

## Measured snapshot facts (PR 1)

At `awesomedata/apd-core` commit
`92d1b210bfba28055f672e6331eb5e5fa27227c5`:

- 1,142 metadata files: 1,126 `.yml`, 11 `.yaml`, 4 extensionless and 1 `.wml`.
- 1,142/1,142 README `Meta` links reconcile (1,010 OK, 132 FIXME).
- YAML aliases: **0** files. The default `YAML_MAX_ALIASES=0` is therefore
  safe; raise it only with measured cause.
- One file (`GIS/Crusader-Atlas.yml`) has duplicate keys; it is retained with a
  `duplicate_key:*` warning.

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
