#!/usr/bin/env python3
"""Best-effort operational telemetry for future tuning decisions.

Appends one JSONL row per recorded operation to
~/.FlossWare/ledger/ops/<name>.jsonl (override with
POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR). Recording NEVER raises into the caller:
a telemetry failure is logged once and dropped. Rows carry only numeric and
enum fields — never request bodies, headers, or credentials.

This exists so future optimization targets (gateway forwarding policy,
embedding batch parameters) have the evidence they need, because telemetry
cannot be reconstructed retroactively.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_MAX_LINE_BYTES = 16 * 1024
_max_file_bytes = 256 * 1024 * 1024
_warned = set()


def _ledger_dir() -> str:
    return os.environ.get(
        "POC_CLAUDE_ENSEMBLE_OPS_LEDGER_DIR",
        os.path.expanduser("~/.FlossWare/ledger/ops"),
    )


def record(name: str, **fields) -> bool:
    """Append one telemetry row. Returns True on success; never raises.

    Values must be JSON-serializable scalars (numbers, bools, short strings).
    Oversize or non-serializable rows are dropped, not truncated, so a
    malformed caller cannot smuggle large payloads into the ledger.
    """
    try:
        name = str(name)[:64]
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "op": name}
        row.update(fields)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
            return False
        directory = _ledger_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.jsonl")
        if os.path.exists(path) and os.path.getsize(path) > _max_file_bytes:
            if "size_cap" not in _warned:
                _warned.add("size_cap")
                logger.warning("ops telemetry ledger %s hit the size cap; dropping rows", path)
            return False
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the caller
        if name not in _warned:
            _warned.add(name)
            logger.warning("ops telemetry recording failed for %s: %s", name, exc)
        return False
