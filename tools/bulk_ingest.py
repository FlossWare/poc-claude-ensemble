#!/usr/bin/env python3
"""Bulk ingestion client: reads scraped JSON items from a local directory
and pushes them through the POC ingestion REST boundary.

The ingestion service never reads files; this client is the only reader.

Usage:
  python3 tools/bulk_ingest.py PATH [--source NAME] [--origin NAME]
    [--gateway URL] [--limit N] [--dry-run] [--resume-state FILE]

Gateway contract (via services/server/ensemble_server.py):
  POST /api/v1/ingestion/preview  -> {"preview_id": ..., "source": {...}}
  POST /api/v1/ingestion/ingest   -> {"status": "completed", ...}
  Underlying ingestion_service.preview/ingest requires exactly one of
  {uri, content_base64, text}. We use the "text" path (text+title).
  The service enforces MAX_BODY_BYTES=12MiB, MAX_TEXT_CHARS=4_000_000.

Ledger (default ~/.FlossWare/ledger/bulk-ingest/<source>.jsonl):
  One JSON line per item with content_sha256 + service response.
  Re-runs skip hashes already marked succeeded.
  Service-side content_sha256 dedup is the final guard.

Treat all file content as untrusted: cap sizes, validate JSON shape,
never execute.

Provenance note: the ingestion service's text path hard-codes
origin=inline-text and locator=text://sha256/<hash>. The requested
origin/locator/retrieved fields are recorded in the ledger and log but
cannot override the service's locator for text ingestion. This is
documented as a deviation in the report output.
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
MAX_FILE_BYTES = 1 * 1024 * 1024        # per-file cap (client-side)
MAX_TEXT_CHARS = 4_000_000              # ingestion_service.MAX_TEXT_CHARS
MAX_REQUEST_BYTES = 12 * 1024 * 1024     # ingestion_service.MAX_BODY_BYTES
MAX_TOTAL_ITEMS = 100_000                # max items per run (safety)
DEFAULT_LIMIT = None                     # no limit unless --limit
REQUEST_TIMEOUT = 180                   # ingestion can take minutes

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
) -> dict:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _all_entries, succeeded_hashes = load_ledger(ledger_path)

    # Enumerate files (sorted for determinism)
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.rglob("*.json"))
    else:
        print(f"error: PATH does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    report: dict = {
        "source": source,
        "origin": origin,
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

    def count_skip(reason: str) -> None:
        report["skipped_reasons"][reason] = report["skipped_reasons"].get(reason, 0) + 1

    processed = 0  # number of items we attempted to push (or would have)

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
            count_skip(f"stat_error:{type(exc).__name__}")
            continue

        if fsize > max_file_bytes:
            report["skipped_malformed"] += 1
            count_skip("file_too_large")
            continue
        if fsize == 0:
            report["skipped_malformed"] += 1
            count_skip("empty_file")
            continue

        # Read + parse (capped read)
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            report["skipped_malformed"] += 1
            count_skip(f"read_error:{type(exc).__name__}")
            continue

        if len(raw_bytes) > max_file_bytes:
            report["skipped_malformed"] += 1
            count_skip("file_too_large")
            continue

        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try with replacement; if still not JSON, it will fail below
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        # Cap JSON text length to avoid huge parse
        if len(raw_text) > MAX_TEXT_CHARS * 2:
            report["skipped_malformed"] += 1
            count_skip("json_too_large")
            continue

        try:
            raw_item = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            report["skipped_malformed"] += 1
            count_skip("invalid_json")
            continue

        # Handle both single-object and array-of-objects files
        items_to_extract: list[dict] = []
        if isinstance(raw_item, list):
            for elem in raw_item:
                if isinstance(elem, dict):
                    items_to_extract.append(elem)
                else:
                    report["skipped_malformed"] += 1
                    count_skip("array_element_not_object")
        elif isinstance(raw_item, dict):
            items_to_extract.append(raw_item)
        else:
            report["skipped_malformed"] += 1
            count_skip("not_a_json_object")
            continue

        for raw_obj in items_to_extract:
            if limit is not None and processed >= limit:
                break
            if processed >= max_total_items:
                break

            normalized, reason = extract_item(raw_obj, str(file_path))
            if normalized is None:
                report["skipped_malformed"] += 1
                count_skip(reason or "unknown")
                continue

            text = normalized["text"]
            title = normalized["title"]

            # Text size guard
            if len(text) > MAX_TEXT_CHARS:
                report["skipped_malformed"] += 1
                count_skip("text_too_large")
                continue
            if not text.strip():
                report["skipped_malformed"] += 1
                count_skip("empty_text")
                continue

            # Request size guard (JSON overhead estimate)
            est_bytes = len(text.encode("utf-8")) + len(title.encode("utf-8")) + 1024
            if est_bytes > MAX_REQUEST_BYTES:
                report["skipped_malformed"] += 1
                count_skip("request_too_large")
                continue

            h = content_hash(text)

            # Ledger dedup (client-side fast path)
            if h in succeeded_hashes:
                report["skipped_duplicate"] += 1
                count_skip("ledger_duplicate")
                continue

            # Dry run: count but don't POST
            if dry_run:
                report["ingested"] += 1  # would have ingested
                processed += 1
                append_ledger(ledger_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": source,
                    "origin": origin,
                    "locator": normalized.get("url"),
                    "retrieved_at": normalized.get("timestamp"),
                    "file": str(file_path),
                    "title": title,
                    "content_sha256": h,
                    "characters": len(text),
                    "status": "dry_run",
                    "response": None,
                })
                succeeded_hashes.add(h)
                continue

            # --- Gateway: preview then ingest ---
            processed += 1

            # Build preview payload (text path)
            preview_payload: dict = {"text": text, "title": title[:300]}

            status, preview_body = gateway_post(gateway, "/api/v1/ingestion/preview", preview_payload)

            if status != 200 or not isinstance(preview_body, dict) or not preview_body.get("ok"):
                report["failed"] += 1
                err_detail = preview_body if isinstance(preview_body, str) else json.dumps(preview_body)[:500]
                report["errors"].append({"file": str(file_path), "stage": "preview", "status": status, "detail": err_detail})
                append_ledger(ledger_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": source,
                    "origin": origin,
                    "locator": normalized.get("url"),
                    "retrieved_at": normalized.get("timestamp"),
                    "file": str(file_path),
                    "title": title,
                    "content_sha256": h,
                    "characters": len(text),
                    "status": "failed",
                    "stage": "preview",
                    "http_status": status,
                    "response": preview_body if isinstance(preview_body, dict) else {"error": preview_body},
                })
                continue

            preview_id = preview_body.get("preview_id")
            if not isinstance(preview_id, str) or not preview_id:
                report["failed"] += 1
                report["errors"].append({"file": str(file_path), "stage": "preview", "status": status, "detail": "missing preview_id"})
                append_ledger(ledger_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": source,
                    "origin": origin,
                    "locator": normalized.get("url"),
                    "retrieved_at": normalized.get("timestamp"),
                    "file": str(file_path),
                    "title": title,
                    "content_sha256": h,
                    "characters": len(text),
                    "status": "failed",
                    "stage": "preview",
                    "response": preview_body,
                })
                continue

            # Ingest
            ingest_payload = {"preview_id": preview_id}
            status2, ingest_body = gateway_post(gateway, "/api/v1/ingestion/ingest", ingest_payload)

            if status2 == 200 and isinstance(ingest_body, dict) and ingest_body.get("ok"):
                ingest_status = ingest_body.get("status", "unknown")
                if ingest_status in ("completed", "rejected"):
                    # Rejected means known but not stored (e.g. duplicate content);
                    # treat as success for dedup purposes.
                    report["ingested"] += 1
                    append_ledger(ledger_path, {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "source": source,
                        "origin": origin,
                        "locator": normalized.get("url"),
                        "retrieved_at": normalized.get("timestamp"),
                        "file": str(file_path),
                        "title": title,
                        "content_sha256": h,
                        "ingestion_id": ingest_body.get("ingestion_id"),
                        "ingest_status": ingest_status,
                        "decision": ingest_body.get("decision"),
                        "chunks": ingest_body.get("chunks"),
                        "vectors": ingest_body.get("vectors"),
                        "characters": len(text),
                        "status": "succeeded",
                        "response": ingest_body,
                    })
                    succeeded_hashes.add(h)
                else:
                    report["failed"] += 1
                    append_ledger(ledger_path, {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "source": source,
                        "origin": origin,
                        "locator": normalized.get("url"),
                        "retrieved_at": normalized.get("timestamp"),
                        "file": str(file_path),
                        "title": title,
                        "content_sha256": h,
                        "status": "failed",
                        "stage": "ingest",
                        "http_status": status2,
                        "response": ingest_body,
                    })
            else:
                report["failed"] += 1
                err_detail2 = ingest_body if isinstance(ingest_body, str) else json.dumps(ingest_body)[:500]
                report["errors"].append({"file": str(file_path), "stage": "ingest", "status": status2, "detail": err_detail2})
                append_ledger(ledger_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": source,
                    "origin": origin,
                    "locator": normalized.get("url"),
                    "retrieved_at": normalized.get("timestamp"),
                    "file": str(file_path),
                    "title": title,
                    "content_sha256": h,
                    "status": "failed",
                    "stage": "ingest",
                    "http_status": status2,
                    "response": ingest_body if isinstance(ingest_body, dict) else {"error": ingest_body},
                })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ingestion client: push scraped JSON items through the POC ingestion REST boundary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to a JSON file or directory tree containing scraper item files")
    parser.add_argument("--source", default=None, help="Source name (default: basename of PATH)")
    parser.add_argument("--origin", default=None, help="Origin tag recorded in ledger (e.g. web-scrape)")
    parser.add_argument("--gateway", default=os.environ.get("POC_CLAUDE_ENSEMBLE_GATEWAY_URL", "http://127.0.0.1:8080"),
                        help="Gateway base URL (default: $POC_CLAUDE_ENSEMBLE_GATEWAY_URL or http://127.0.0.1:8080)")
    parser.add_argument("--limit", type=int, default=None, help="Max items to ingest this run")
    parser.add_argument("--dry-run", action="store_true", help="Parse and ledger without POSTing to the gateway")
    parser.add_argument("--resume-state", default=None,
                        help="Ledger file path (default: ~/.FlossWare/ledger/bulk-ingest/<source>.jsonl)")
    parser.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES,
                        help=f"Max bytes per JSON file (default: {MAX_FILE_BYTES})")
    args = parser.parse_args()

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        print(f"error: PATH does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    source = args.source or (input_path.name if input_path.name else "bulk-ingest")
    # Sanitize source for filename
    safe_source = "".join(c if c.isalnum() or c in "-_." else "_" for c in source).strip("._") or "bulk-ingest"
    if args.resume_state:
        ledger_path = Path(args.resume_state).expanduser().resolve()
    else:
        ledger_path = Path.home() / ".FlossWare" / "ledger" / "bulk-ingest" / f"{safe_source}.jsonl"

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    report = run(
        input_path=input_path,
        source=source,
        origin=args.origin,
        gateway=args.gateway,
        limit=args.limit,
        dry_run=args.dry_run,
        ledger_path=ledger_path,
        max_file_bytes=args.max_file_bytes,
    )

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
