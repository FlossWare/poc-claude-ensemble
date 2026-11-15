# Awesome Public Datasets ingestion

This POC connector uses `awesomedata/apd-core` as the canonical catalog. The
generated `awesome-public-datasets` README is not parsed as a second catalog.
Every YAML record under `core/` is inventoried, including malformed records
(which are written to the manifest as errors).

## Initial inventory

From the repository root:

```sh
git clone --depth 1 https://github.com/awesomedata/apd-core.git var/awesome-public-datasets/apd-core
python -m pip install -r requirements.txt
PYTHONPATH=services/ingestion python -m awesome_datasets
```

The default manifest is append-only JSONL at
`var/awesome-public-datasets/manifest.jsonl`. The inventory records dataset
metadata, license evidence, candidate asset URLs, and explicit deferrals. A
homepage is not assumed to be a file download: the connector inspects small
HTML landing pages for links with recognizable data/archive extensions.

## Bounded acquisition

After reviewing the inventory and the license values, run:

```sh
PYTHONPATH=services/ingestion python -m awesome_datasets --acquire
```

Initial safety defaults are 64 MiB per file, 256 MiB per dataset, and 20 GiB
total. Override them with `--max-file-mib`, `--max-dataset-mib`, and
`--max-total-gib`. Downloads stream to temporary files, enforce byte limits,
and are atomically renamed only after completion. SHA-256 and local paths are
recorded for repeatable runs.

The acquisition allow-list is deliberately conservative (CC0, common CC-BY
and CC-BY-SA variants, ODbL/ODC-BY, MIT, Apache-2.0, and BSD variants). Missing,
unrecognized, or restrictive licenses are deferred for review. This is not a
legal determination: license strings and dataset terms must be checked before
reuse, especially for mixed-license collections.

## Current implementation boundary

This first connector increment implements catalog discovery, metadata
normalization, direct-asset/landing-page discovery, bounded downloads,
checksums, resumable local files, and explicit outcomes. It does **not yet**
claim that acquired binary files have been parsed, chunked, embedded, or
indexed. Archive expansion and format-specific ingestion into the existing
Knowledge/Vector/Graph services are the next implementation increment. The
manifest is the source of truth for the remaining work; unsupported formats,
gated sources, and failed downloads must remain visible there.

Downloaded content is untrusted. The connector never executes it. It rejects
non-HTTP(S), credential-bearing, non-standard-port, and private-network URLs,
including redirects to those destinations.
