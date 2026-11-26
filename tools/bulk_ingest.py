#!/usr/bin/env python3
"""Bulk ingestion client: reads local scraped JSON items (or PDFs) and pushes
them through the POC ingestion REST boundary.

The ingestion service never reads files; this client is the only reader.

Usage:
  python3 tools/bulk_ingest.py PATH [--source NAME] [--origin NAME]
    [--gateway URL] [--limit N] [--learn-batch] [--dry-run] [--resume-state FILE]

  python3 tools/bulk_ingest.py PATH --format pdf [--source books]
    [--origin books] [--limit N] [--max-pdf-bytes N] [--max-pdf-pages N]
    [--max-text-chars N] [--dry-run] [--resume-state FILE]

--learn-batch draws this run's item budget from the meta-learning batch_size
domain (25/50/100/250/500).  The client pushes items one at a time, so the
run's item budget is the only batching boundary it controls; the draw is used
when --limit is absent, and the run's throughput/failure outcome is recorded
afterwards.  Succeeded/failed ingest ledger rows also carry the chunking
profile identity from the preview response, which the meta-learning chunking
domain compares.

Formats:
  json (default)  Scraped JSON item files. Each object is normalized to
                  {title, text, url}; files may hold one object or a list.
  pdf             A directory tree (or single file) of PDFs. Text is
                  extracted locally with pypdf and pushed as inline text.
                  Title comes from the file name; locator is file://<path>.
                  PDFs are untrusted input: extraction is bounded by
                  --max-pdf-bytes, --max-pdf-pages, and --max-text-chars.

Gateway contract (via services/server/ensemble_server.py):
  POST /api/v1/ingestion/preview  -> {"preview_id": ..., "source": {...}}
  POST /api/v1/ingestion/ingest   -> {"status": "completed", ...}
  Underlying ingestion_service.preview/ingest requires exactly one of
  {uri, content_base64, text}. We use the "text" path (text+title).
  The service enforces MAX_BODY_BYTES=12MiB, MAX_TEXT_CHARS=4_000_000.

Ledger (default ~/.FlossWare/ledger/bulk-ingest/<source>.jsonl):
  One JSON line per item with content_sha256 + service response.
  Re-runs skip hashes already marked succeeded.
  Idempotency is client-side: ingestion_service passes
  revision_sequence=time.time_ns() on every push, so the Knowledge service
  accepts a repeat of identical content as a new revision instead of
  rejecting it (observed against the live gateway). Keep the ledger.
  For --format pdf the hash is sha256 of the EXTRACTED text (the bytes the
  service actually stores), not of the PDF file, so re-extraction of the
  same document dedups even if the PDF container changes.

Treat all file content as untrusted: cap sizes, validate JSON shape,
never execute. PDF parsing runs in-process on untrusted bytes; parser
errors are recorded as skip reasons, never allowed to abort the run.

Provenance note: the ingestion service's text path hard-codes
origin=inline-text and locator=text://sha256/<hash>. The requested
origin/locator/retrieved fields are recorded in the ledger and log but
cannot override the service's locator for text ingestion. This is
documented as a deviation in the report output (provenance_note).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Limits (mirrored from ingestion_service.py)
# ---------------------------------------------------------------------------
MAX_FILE_BYTES = 1 * 1024 * 1024        # per-JSON-file cap (client-side)
MAX_PDF_BYTES = 128 * 1024 * 1024       # per-PDF file cap (client-side)
MAX_PDF_PAGES = 2_000                   # per-PDF page cap (client-side)
MAX_TEXT_CHARS = 4_000_000              # ingestion_service.MAX_TEXT_CHARS
MAX_REQUEST_BYTES = 12 * 1024 * 1024     # ingestion_service.MAX_BODY_BYTES
MAX_TOTAL_ITEMS = 100_000                # max items per run (safety)
DEFAULT_LIMIT = None                     # no limit unless --limit
REQUEST_TIMEOUT = 180                   # ingestion can take minutes

INPUT_FORMATS = ("json", "pdf")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICES = os.path.join(REPO, "services")


def chunk_profile_of(preview_body):
    """The chunking profile identity (name@version) from a preview response."""
    profiles = preview_body.get("profiles") if isinstance(preview_body, dict) else None
    if isinstance(profiles, dict):
        value = profiles.get("chunking")
        if isinstance(value, str) and value:
            return value
    return None


def meta_learner():
    """The meta-learning learner, or None when the services tree is absent."""
    try:
        if SERVICES not in sys.path:
            sys.path.insert(0, SERVICES)
        from learning.meta import MetaLearner

        return MetaLearner()
    except Exception as exc:  # noqa: BLE001 - learning is additive, never fatal
        print(f"meta-learning unavailable: {exc}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# Field normalization: try known variants for each logical field.
# ---------------------------------------------------------------------------
TEXT_FIELDS = ("summary", "abstract", "extract", "text", "content", "body")
TITLE_FIELDS = ("title", "name")
URL_FIELDS = ("url", "link", "original_url", "locator", "source_url")
TIMESTAMP_FIELDS = ("stored_at", "fetched_at", "retrieved", "retrieved_at",
                    "published", "date", "timestamp")


def extract_item(raw: dict, file_path: str) -> tuple[dict | None, str | None]:
    """Normalize one JSON object to {title, text, url, timestamp} or return (None, reason).

    Returns (normalized_dict, None) on success or (None, reason_string) on skip.
    """
    if not isinstance(raw, dict):
        return None, "not_a_json_object"

    # Title
    title = None
    for field in TITLE_FIELDS:
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            title = val.strip()
            break
    if title is None:
        return None, "missing_title"

    # Text
    text = None
    for field in TEXT_FIELDS:
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            text = val.strip()
            break
    if text is None:
        return None, "missing_text"

    # URL (optional but tracked)
    url = None
    for field in URL_FIELDS:
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            url = val.strip()
            break

    # Timestamp (optional)
    timestamp = None
    for field in TIMESTAMP_FIELDS:
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            timestamp = val.strip()
            break
        if isinstance(val, (int, float)):
            timestamp = str(val)
            break

    return {
        "title": title,
        "text": text,
        "url": url,
        "timestamp": timestamp,
        "source_file": file_path,
    }, None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _require_pypdf():
    """Import pypdf lazily so JSON ingestion does not need the PDF dependency."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("error: --format pdf requires pypdf; install requirements.txt "
              "(pip install 'pypdf==6.1.1')", file=sys.stderr)
        sys.exit(2)
    return PdfReader


def pdf_title(file_path: Path) -> str:
    """Title from the file name: underscores become spaces, whitespace collapsed."""
    stem = " ".join(file_path.stem.replace("_", " ").split())
    return (stem or file_path.name)[:300]


def extract_pdf_text(file_path: Path, max_pdf_pages: int = MAX_PDF_PAGES) -> tuple[str | None, dict, str | None]:
    """Extract text from one PDF.

    Returns (text, meta, reason). On success reason is None and text is the
    per-page text joined by blank lines (empty string when no page has a text
    layer). On skip text is None and reason is one of:
      pdf_encrypted      password protected and not openable with ""
      too_many_pages     page count exceeds max_pdf_pages
      pdf_unreadable:<T> parser raised <T>
    A page that fails to extract is counted in meta["pdf_pages_failed"] and does
    not abort the document; total extraction failure still yields "".
    """
    PdfReader = _require_pypdf()
    meta: dict = {}
    try:
        reader = PdfReader(str(file_path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return None, meta, "pdf_encrypted"
            except Exception:
                return None, meta, "pdf_encrypted"
        pages = len(reader.pages)
        meta["pdf_pages"] = pages
        if pages > max_pdf_pages:
            return None, meta, "too_many_pages"
        chunks: list[str] = []
        failed = 0
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                failed += 1
                continue
            if page_text.strip():
                chunks.append(page_text.strip())
        meta["pdf_pages_failed"] = failed
        meta["pdf_pages_with_text"] = len(chunks)
    except Exception as exc:  # untrusted bytes: parser failures are data, not bugs
        return None, meta, f"pdf_unreadable:{type(exc).__name__}"
    return "\n\n".join(chunks), meta, None


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------

def load_ledger(path: Path) -> tuple[dict[str, dict], set[str]]:
    """Load existing ledger. Returns (all_entries_by_hash, succeeded_hashes)."""
    succeeded: set[str] = set()
    all_entries: dict[str, dict] = {}
    if not path.exists():
        return all_entries, succeeded
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            h = entry.get("content_sha256") or entry.get("content_hash") or ""
            if h:
                all_entries[h] = entry
                if entry.get("status") in ("succeeded", "completed", "duplicate", "accepted"):
                    succeeded.add(h)
    return all_entries, succeeded


def append_ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

def gateway_post(gateway: str, route: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[int, dict | str]:
    """POST JSON to gateway. Returns (status_code, parsed_body_or_error_string)."""
    url = gateway.rstrip("/") + route
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(body)
            except json.JSONDecodeError:
                return exc.code, body
        finally:
            exc.close()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def count_skip(report: dict, reason: str) -> None:
    report["skipped_reasons"][reason] = report["skipped_reasons"].get(reason, 0) + 1


def _ledger_entry(
    *,
    source: str,
    origin: str | None,
    locator: str | None,
    retrieved_at: str | None,
    source_file: str,
    title: str,
    digest: str,
    characters: int,
    extra_fields: dict | None,
) -> dict:
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "origin": origin,
        "locator": locator,
        "retrieved_at": retrieved_at,
        "file": source_file,
        "title": title,
        "content_sha256": digest,
        "characters": characters,
    }
    if extra_fields:
        entry.update(extra_fields)
    return entry


def push_text_item(
    *,
    text: str,
    title: str,
    locator: str | None,
    retrieved_at: str | None,
    source_file: str,
    source: str,
    origin: str | None,
    gateway: str,
    dry_run: bool,
    ledger_path: Path,
    succeeded_hashes: set[str],
    report: dict,
    max_text_chars: int = MAX_TEXT_CHARS,
    extra_fields: dict | None = None,
) -> str:
    """Validate, dedup and push one text item through preview+ingest.

    Returns "skipped", "dry_run", "succeeded" or "failed" so the caller can
    apply its own run-level bounds. Mutates report counters and the ledger.
    """
    def entry(fields: dict) -> dict:
        """Ledger entry for this item: common fields + per-outcome fields."""
        merged = dict(extra_fields or {})
        merged.update(fields)  # outcome fields win over document metadata
        return _ledger_entry(source=source, origin=origin, locator=locator,
                             retrieved_at=retrieved_at, source_file=source_file,
                             title=title, digest=h, characters=len(text),
                             extra_fields=merged)

    if not text.strip():
        report["skipped_malformed"] += 1
        count_skip(report, "empty_text")
        return "skipped"

    if len(text) > max_text_chars:
        report["skipped_malformed"] += 1
        count_skip(report, "text_too_large")
        return "skipped"

    # Request size guard. json.dumps(ensure_ascii=True) expands non-ASCII
    # to \uXXXX (6 bytes per BMP char, 12 for non-BMP), and quotes,
    # backslashes and newlines also double. Use the actual serialized
    # length, not a raw UTF-8 estimate.
    est_bytes = len(json.dumps({"text": text, "title": title}).encode("utf-8")) + 512
    if est_bytes > MAX_REQUEST_BYTES:
        report["skipped_malformed"] += 1
        count_skip(report, "request_too_large")
        return "skipped"

    h = content_hash(text)

    # Ledger dedup (client-side fast path)
    if h in succeeded_hashes:
        report["skipped_duplicate"] += 1
        count_skip(report, "ledger_duplicate")
        return "skipped"

    # Dry run: count but don't POST
    if dry_run:
        report["ingested"] += 1  # would have ingested
        append_ledger(ledger_path, entry({"status": "dry_run", "response": None}))
        succeeded_hashes.add(h)
        return "dry_run"

    # --- Gateway: preview then ingest ---
    preview_payload: dict = {"text": text, "title": title[:300]}

    status, preview_body = gateway_post(gateway, "/api/v1/ingestion/preview", preview_payload)

    if status != 200 or not isinstance(preview_body, dict) or not preview_body.get("ok"):
        report["failed"] += 1
        err_detail = preview_body if isinstance(preview_body, str) else json.dumps(preview_body)[:500]
        report["errors"].append({"file": source_file, "stage": "preview", "status": status, "detail": err_detail})
        append_ledger(ledger_path, entry({
            "status": "failed", "stage": "preview", "http_status": status,
            "response": preview_body if isinstance(preview_body, dict) else {"error": preview_body},
        }))
        return "failed"

    preview_id = preview_body.get("preview_id")
    if not isinstance(preview_id, str) or not preview_id:
        report["failed"] += 1
        report["errors"].append({"file": source_file, "stage": "preview", "status": status, "detail": "missing preview_id"})
        append_ledger(ledger_path, entry({"status": "failed", "stage": "preview", "response": preview_body}))
        return "failed"

    # The preview advertises the profile identity the ingest will use, so the
    # ledger can attribute chunks to a chunking profile.
    chunk_profile = chunk_profile_of(preview_body)

    # Ingest
    ingest_payload = {"preview_id": preview_id}
    status2, ingest_body = gateway_post(gateway, "/api/v1/ingestion/ingest", ingest_payload)

    if status2 == 200 and isinstance(ingest_body, dict) and ingest_body.get("ok"):
        ingest_status = ingest_body.get("status", "unknown")
        if ingest_status in ("completed", "rejected"):
            # Rejected means known but not stored (e.g. duplicate content);
            # treat as success for dedup purposes.
            report["ingested"] += 1
            append_ledger(ledger_path, entry({
                "ingestion_id": ingest_body.get("ingestion_id"),
                "ingest_status": ingest_status,
                "decision": ingest_body.get("decision"),
                "chunks": ingest_body.get("chunks"),
                "vectors": ingest_body.get("vectors"),
                "chunk_profile": chunk_profile,
                "status": "succeeded",
                "response": ingest_body,
            }))
            succeeded_hashes.add(h)
            return "succeeded"

        report["failed"] += 1
        append_ledger(ledger_path, entry({
            "status": "failed", "stage": "ingest", "http_status": status2,
            "chunk_profile": chunk_profile, "response": ingest_body,
        }))
        return "failed"

    report["failed"] += 1
    err_detail2 = ingest_body if isinstance(ingest_body, str) else json.dumps(ingest_body)[:500]
    report["errors"].append({"file": source_file, "stage": "ingest", "status": status2, "detail": err_detail2})
    append_ledger(ledger_path, entry({
        "status": "failed", "stage": "ingest", "http_status": status2,
        "chunk_profile": chunk_profile,
        "response": ingest_body if isinstance(ingest_body, dict) else {"error": ingest_body},
    }))
    return "failed"


def _pdf_candidates(input_path: Path) -> list[Path]:
    """Enumerate PDFs under PATH (case-insensitive suffix), sorted for determinism."""
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def push_batch_items(
    items: list[dict],
    *,
    source: str,
    origin: str | None,
    gateway: str,
    dry_run: bool,
    ledger_path: Path,
    succeeded_hashes: set[str],
    report: dict,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> int:
    """Validate, dedup and push many text items in ONE batch request.

    Items are dicts with keys text, title, locator, retrieved_at,
    source_file and optional extra (same shape push_text_item consumes).
    Validation and ledger dedup are per item (identical skip reasons); the
    accepted items are posted together to the ingestion service's batch
    endpoint, which embeds them in cross-item batches.  Returns the number
    of items pushed (0 when everything was skipped).
    """
    payload_items: list[dict] = []
    accepted: list[tuple[dict, str]] = []  # (item, digest)
    batch_hashes: set[str] = set()  # within-batch content dedupe
    for item in items:
        text = item["text"]
        title = item["title"]

        def _skip(reason: str) -> None:
            report["skipped_malformed"] += 1
            count_skip(report, reason)

        if not text.strip():
            _skip("empty_text")
            continue
        if len(text) > max_text_chars:
            _skip("text_too_large")
            continue
        est_bytes = len(json.dumps({"text": text, "title": title}).encode("utf-8")) + 512
        if est_bytes > MAX_REQUEST_BYTES:
            _skip("request_too_large")
            continue
        h = content_hash(text)
        if h in succeeded_hashes:
            report["skipped_duplicate"] += 1
            count_skip(report, "ledger_duplicate")
            continue
        if dry_run:
            report["ingested"] += 1
            append_ledger(ledger_path, _ledger_entry(
                source=source, origin=origin, locator=item.get("locator"),
                retrieved_at=item.get("retrieved_at"), source_file=item["source_file"],
                title=title, digest=h, characters=len(text),
                extra_fields=item.get("extra"),
            ) | {"status": "dry_run", "response": None})
            succeeded_hashes.add(h)
            continue
        if h in batch_hashes:
            # within-batch duplicate: identical texts share a source_id
            # (one index write); skip the redundant knowledge revision
            report["skipped_duplicate"] += 1
            count_skip(report, "duplicate_in_batch")
            continue
        batch_hashes.add(h)
        payload_items.append({"text": text, "title": title[:300]})
        accepted.append((item, h))

    if not payload_items:
        # nothing accepted (all skipped, or a dry run — which must never
        # touch the gateway): do not POST an empty items list
        return 0

    status, body = gateway_post(gateway, "/api/v1/ingestion/ingest/batch",
                                {"items": payload_items})
    chunk_profile = chunk_profile_of(body) if isinstance(body, dict) else None
    results = body.get("results") if isinstance(body, dict) else None

    if status != 200 or not isinstance(results, list) or len(results) != len(accepted):
        # Whole-batch transport failure: no item outcome is known.
        report["failed"] += len(accepted)
        err_detail = body if isinstance(body, str) else json.dumps(body)[:500]
        report["errors"].append({"stage": "batch", "status": status, "detail": err_detail})
        for item, h in accepted:
            append_ledger(ledger_path, _ledger_entry(
                source=source, origin=origin, locator=item.get("locator"),
                retrieved_at=item.get("retrieved_at"), source_file=item["source_file"],
                title=item["title"], digest=h, characters=len(item["text"]),
                extra_fields=item.get("extra"),
            ) | {"status": "failed", "stage": "batch", "http_status": status,
                 "response": body if isinstance(body, dict) else {"error": body}})
        return len(accepted)

    for (item, h), result in zip(accepted, results):
        entry = _ledger_entry(
            source=source, origin=origin, locator=item.get("locator"),
            retrieved_at=item.get("retrieved_at"), source_file=item["source_file"],
            title=item["title"], digest=h, characters=len(item["text"]),
            extra_fields=item.get("extra"),
        )
        if isinstance(result, dict) and result.get("ok") and result.get("status") in ("completed", "rejected"):
            report["ingested"] += 1
            append_ledger(ledger_path, entry | {
                "ingestion_id": result.get("ingestion_id"),
                "ingest_status": result.get("status"),
                "decision": result.get("decision"),
                "chunks": result.get("chunks"),
                "vectors": result.get("vectors"),
                "chunk_profile": chunk_profile,
                "status": "succeeded",
                "response": result,
            })
            succeeded_hashes.add(h)
        else:
            report["failed"] += 1
            append_ledger(ledger_path, entry | {
                "status": "failed", "stage": "batch_item",
                "response": result if isinstance(result, dict) else {"error": str(result)},
            })
    return len(accepted)


def run(
    input_path: Path,
    source: str,
    origin: str | None,
    gateway: str,
    limit: int | None,
    dry_run: bool,
    ledger_path: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_items: int = MAX_TOTAL_ITEMS,
    input_format: str = "json",
    max_text_chars: int = MAX_TEXT_CHARS,
    max_pdf_bytes: int = MAX_PDF_BYTES,
    max_pdf_pages: int = MAX_PDF_PAGES,
    batch_size: int = 1,
) -> dict:
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unsupported input_format: {input_format}")
    if input_format == "pdf":
        _require_pypdf()

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _all_entries, succeeded_hashes = load_ledger(ledger_path)

    # Enumerate files (sorted for determinism)
    if input_path.is_file():
        candidates = [input_path]
        if input_format == "pdf" and input_path.suffix.lower() != ".pdf":
            print(f"error: --format pdf expects a .pdf file: {input_path}", file=sys.stderr)
            sys.exit(2)
    elif input_path.is_dir():
        if input_format == "pdf":
            candidates = _pdf_candidates(input_path)
        else:
            # JSON items and JSONL batch files (one JSON object per line --
            # the fleet scrapes' dominant format) are both JSON input.
            candidates = sorted(
                p for p in input_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
            )
    else:
        print(f"error: PATH does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    report: dict = {
        "source": source,
        "origin": origin,
        "format": input_format,
        "gateway": gateway,
        "ledger": str(ledger_path),
        "input_path": str(input_path),
        "dry_run": dry_run,
        "ingested": 0,
        "skipped_duplicate": 0,
        "skipped_malformed": 0,
        "failed": 0,
        "skipped_reasons": {},  # reason -> count
        "errors": [],
    }
    provenance_note = (
        "The ingestion service text path hard-codes origin=inline-text and "
        "locator=text://sha256/<hash>; custom origin/locator are recorded in "
        "the ledger but do not override the service locator."
    )
    if origin:
        report["provenance_note"] = provenance_note

    processed = 0  # number of items we attempted to push (or would have)
    pending: list[dict] = []  # buffered items awaiting a batch push

    for file_path in candidates:
        # Global caps
        if limit is not None and processed >= limit:
            break
        if processed >= max_total_items:
            print(f"warning: reached max_total_items={max_total_items}, stopping", file=sys.stderr)
            break

        # File size guard (client-side, untrusted data)
        try:
            fsize = file_path.stat().st_size
        except OSError as exc:
            report["skipped_malformed"] += 1
            count_skip(report, f"stat_error:{type(exc).__name__}")
            continue

        if fsize == 0:
            report["skipped_malformed"] += 1
            count_skip(report, "empty_file")
            continue

        if input_format == "pdf":
            if fsize > max_pdf_bytes:
                report["skipped_malformed"] += 1
                count_skip(report, "pdf_too_large")
                continue

            text, meta, reason = extract_pdf_text(file_path, max_pdf_pages=max_pdf_pages)
            if text is None:
                report["skipped_malformed"] += 1
                count_skip(report, reason or "unknown")
                continue
            if not text.strip():
                # Scanned/image-only PDF or a PDF with no text layer.
                report["skipped_malformed"] += 1
                count_skip(report, "no_extractable_text")
                continue
            if len(text) > max_text_chars:
                report["skipped_malformed"] += 1
                count_skip(report, "text_too_large")
                continue

            try:
                modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(file_path.stat().st_mtime))
            except OSError:
                modified = None

            extra = {"format": "pdf", "pdf_bytes": fsize, "file_modified": modified}
            extra.update(meta)

            outcome = push_text_item(
                text=text,
                title=pdf_title(file_path),
                locator=file_path.as_uri(),  # file://<absolute path>
                retrieved_at=None,
                source_file=str(file_path),
                source=source,
                origin=origin,
                gateway=gateway,
                dry_run=dry_run,
                ledger_path=ledger_path,
                succeeded_hashes=succeeded_hashes,
                report=report,
                max_text_chars=max_text_chars,
                extra_fields=extra,
            )
            if outcome != "skipped":
                processed += 1
            continue

        # --- JSON path ---
        if fsize > max_file_bytes:
            report["skipped_malformed"] += 1
            count_skip(report, "file_too_large")
            continue

        # Read + parse (capped read)
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            report["skipped_malformed"] += 1
            count_skip(report, f"read_error:{type(exc).__name__}")
            continue

        if len(raw_bytes) > max_file_bytes:
            report["skipped_malformed"] += 1
            count_skip(report, "file_too_large")
            continue

        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try with replacement; if still not JSON, it will fail below
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        # Cap JSON text length to avoid huge parse
        if len(raw_text) > MAX_TEXT_CHARS * 2:
            report["skipped_malformed"] += 1
            count_skip(report, "json_too_large")
            continue

        try:
            if file_path.suffix.lower() == ".jsonl":
                # One JSON object per line; blank lines are skipped, bad
                # lines are counted individually (a single corrupt line
                # must not discard the rest of a batch file).
                items_to_extract: list[dict] = []
                for line in raw_text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        elem = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        report["skipped_malformed"] += 1
                        count_skip(report, "invalid_jsonl_line")
                        continue
                    if isinstance(elem, dict):
                        items_to_extract.append(elem)
                    else:
                        report["skipped_malformed"] += 1
                        count_skip(report, "jsonl_line_not_object")
                raw_item = None
            else:
                raw_item = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            report["skipped_malformed"] += 1
            count_skip(report, "invalid_json")
            continue

        if raw_item is not None:
            # Handle both single-object and array-of-objects files
            items_to_extract = []
            if isinstance(raw_item, list):
                for elem in raw_item:
                    if isinstance(elem, dict):
                        items_to_extract.append(elem)
                    else:
                        report["skipped_malformed"] += 1
                        count_skip(report, "array_element_not_object")
            elif isinstance(raw_item, dict):
                items_to_extract.append(raw_item)
            else:
                report["skipped_malformed"] += 1
                count_skip(report, "not_a_json_object")
                continue

        for raw_obj in items_to_extract:
            if limit is not None and processed >= limit:
                break
            if processed >= max_total_items:
                break

            normalized, reason = extract_item(raw_obj, str(file_path))
            if normalized is None:
                report["skipped_malformed"] += 1
                count_skip(report, reason or "unknown")
                continue

            if batch_size > 1:
                # Buffered batch mode: collect items and push them in one
                # cross-item batch request (see push_batch_items).
                pending.append({
                    "text": normalized["text"],
                    "title": normalized["title"],
                    "locator": normalized.get("url"),
                    "retrieved_at": normalized.get("timestamp"),
                    "source_file": str(file_path),
                })
                processed += 1
                if len(pending) >= batch_size:
                    push_batch_items(
                        pending, source=source, origin=origin, gateway=gateway,
                        dry_run=dry_run, ledger_path=ledger_path,
                        succeeded_hashes=succeeded_hashes, report=report,
                        max_text_chars=max_text_chars,
                    )
                    pending.clear()
                continue

            outcome = push_text_item(
                text=normalized["text"],
                title=normalized["title"],
                locator=normalized.get("url"),
                retrieved_at=normalized.get("timestamp"),
                source_file=str(file_path),
                source=source,
                origin=origin,
                gateway=gateway,
                dry_run=dry_run,
                ledger_path=ledger_path,
                succeeded_hashes=succeeded_hashes,
                report=report,
                max_text_chars=max_text_chars,
            )
            if outcome != "skipped":
                processed += 1

    if pending:
        push_batch_items(
            pending, source=source, origin=origin, gateway=gateway,
            dry_run=dry_run, ledger_path=ledger_path,
            succeeded_hashes=succeeded_hashes, report=report,
            max_text_chars=max_text_chars,
        )
        pending.clear()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ingestion client: push local JSON items or extracted PDF text "
                    "through the POC ingestion REST boundary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to a JSON file/directory (default) or a PDF file/directory (--format pdf)")
    parser.add_argument("--format", dest="input_format", choices=INPUT_FORMATS, default="json",
                        help="Input format: json (scraped items) or pdf (extract text with pypdf)")
    parser.add_argument("--source", default=None,
                        help="Source name (default: basename of PATH; 'books' for --format pdf)")
    parser.add_argument("--origin", default=None,
                        help="Origin tag recorded in ledger (default: none; 'books' for --format pdf)")
    parser.add_argument("--gateway", default=os.environ.get("POC_CLAUDE_ENSEMBLE_GATEWAY_URL", "http://127.0.0.1:8080"),
                        help="Gateway base URL (default: $POC_CLAUDE_ENSEMBLE_GATEWAY_URL or http://127.0.0.1:8080)")
    parser.add_argument("--limit", type=int, default=None, help="Max items to ingest this run")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Push N items per request through the ingestion batch "
                             "endpoint (cross-item embedding; 1 = classic per-item)")
    parser.add_argument("--learn-batch", action="store_true",
                        help="draw this run's item budget from the meta-learning batch_size domain "
                             "(used when --limit is not given; the outcome is recorded afterwards)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and ledger without POSTing to the gateway")
    parser.add_argument("--resume-state", default=None,
                        help="Ledger file path (default: ~/.FlossWare/ledger/bulk-ingest/<source>.jsonl)")
    parser.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES,
                        help=f"Max bytes per JSON file (default: {MAX_FILE_BYTES})")
    parser.add_argument("--max-pdf-bytes", type=int, default=MAX_PDF_BYTES,
                        help=f"Max bytes per PDF (default: {MAX_PDF_BYTES})")
    parser.add_argument("--max-pdf-pages", type=int, default=MAX_PDF_PAGES,
                        help=f"Max pages per PDF (default: {MAX_PDF_PAGES})")
    parser.add_argument("--max-text-chars", type=int, default=MAX_TEXT_CHARS,
                        help=f"Max extracted text characters per item, capped by the service "
                             f"limit {MAX_TEXT_CHARS} (default: {MAX_TEXT_CHARS})")
    args = parser.parse_args()

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        print(f"error: PATH does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    if args.input_format == "pdf":
        source = args.source or "books"
        origin = args.origin if args.origin is not None else "books"
    else:
        source = args.source or (input_path.name if input_path.name else "bulk-ingest")
        origin = args.origin

    # Sanitize source for filename
    safe_source = "".join(c if c.isalnum() or c in "-_." else "_" for c in source).strip("._") or "bulk-ingest"
    if args.resume_state:
        ledger_path = Path(args.resume_state).expanduser().resolve()
    else:
        ledger_path = Path.home() / ".FlossWare" / "ledger" / "bulk-ingest" / f"{safe_source}.jsonl"

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not 1 <= args.max_text_chars <= MAX_TEXT_CHARS:
        parser.error(f"--max-text-chars must be between 1 and {MAX_TEXT_CHARS} "
                     f"(ingestion service limit)")
    if args.max_pdf_bytes < 1:
        parser.error("--max-pdf-bytes must be >= 1")
    if args.max_pdf_pages < 1:
        parser.error("--max-pdf-pages must be >= 1")

    # Meta-learning batch size: the client processes items one at a time, so
    # the only batching boundary it controls is this run's item budget.  The
    # drawn arm is applied as the limit (unless the operator set one) and the
    # run's own throughput/failure outcome is recorded afterwards.
    learner = None
    batch_arm = None
    batch_applied = False
    limit = args.limit
    if args.learn_batch:
        learner = meta_learner()
        if learner is not None:
            try:
                detail = learner.decide("batch_size", {"op": "bulk_ingest"})
                batch_arm = int(detail["choice"])
                if limit is None:
                    limit = batch_arm
                    batch_applied = True
                print(f"batch size (meta-learning): {batch_arm} ({detail['source']}); "
                      f"items this run: {limit}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - fall back to normal limits
                print(f"batch-size draw failed: {exc}; using --limit as given", file=sys.stderr)

    started = time.perf_counter()
    report = run(
        input_path=input_path,
        source=source,
        origin=origin,
        gateway=args.gateway,
        limit=limit,
        dry_run=args.dry_run,
        ledger_path=ledger_path,
        max_file_bytes=args.max_file_bytes,
        input_format=args.input_format,
        max_text_chars=args.max_text_chars,
        max_pdf_bytes=args.max_pdf_bytes,
        max_pdf_pages=args.max_pdf_pages,
        batch_size=max(1, args.batch_size),
    )
    elapsed = time.perf_counter() - started

    # A dry run never touches the gateway, so it is not evidence about a real
    # batch's throughput or failure rate.
    if batch_arm is not None and learner is not None and batch_applied and not args.dry_run:
        total = (report["ingested"] + report["skipped_duplicate"]
                 + report["skipped_malformed"] + report["failed"])
        if total > 0:
            reward = round(max(0.0, 1.0 - report["failed"] / total), 6)
            try:
                learner.record(
                    "batch_size", batch_arm, reward, elapsed / max(1, total),
                    {"outcome_id": f"bulk-ingest:{source}:{time.time_ns()}:{batch_arm}",
                     "source": source, "processed": total, "failed": report["failed"],
                     "elapsed_seconds": round(elapsed, 3)},
                    context={"op": "bulk_ingest"},
                )
                print(f"batch-size outcome recorded (reward {reward}, "
                      f"{total} items, {report['failed']} failed)", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - the ingest already happened
                print(f"batch-size outcome record failed: {exc}", file=sys.stderr)

    # Final report to stdout
    print(json.dumps(report, indent=2, sort_keys=True))
    total = report["ingested"] + report["skipped_duplicate"] + report["skipped_malformed"] + report["failed"]
    print(f"\nDone: ingested={report['ingested']} skipped_duplicate={report['skipped_duplicate']} "
          f"skipped_malformed={report['skipped_malformed']} failed={report['failed']} total_seen={total}",
          file=sys.stderr)
    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
