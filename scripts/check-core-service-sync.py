#!/usr/bin/env python3
"""Fail when a shared Python module drifts between core/ and services/.

services/ is authoritative. core/ retains a compatibility mirror for tooling
that still imports the historical package layout. Test configuration files
are intentionally independent and excluded.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
SERVICES = ROOT / "services"


def sources(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.name == "conftest.py":
            continue
        if any(part == "tests" or part == "test" or part.startswith("test_") for part in relative.parts):
            continue
        result[relative.as_posix()] = path
    return result


def main() -> int:
    core_sources = sources(CORE)
    service_sources = sources(SERVICES)
    shared = sorted(core_sources.keys() & service_sources.keys())
    drift = [
        relative
        for relative in shared
        if core_sources[relative].read_bytes() != service_sources[relative].read_bytes()
    ]
    if drift:
        print("core/ compatibility modules differ from canonical services/:", file=sys.stderr)
        for relative in drift:
            print(f"  {relative}", file=sys.stderr)
        print("Update services/ first, then copy the canonical file into core/.", file=sys.stderr)
        return 1
    print(f"OK: {len(shared)} shared Python modules match services/ (canonical).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
