"""Shared helpers for isolated, reproducible CE evaluations."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def host_metadata() -> dict:
    cpu = command_output(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"]) if sys.platform.startswith("linux") else platform.processor()
    ram = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {"hostname": platform.node(), "os": platform.platform(), "kernel": platform.release(), "python": platform.python_version(),
            "architecture": platform.machine(), "cpu": cpu, "ram_bytes": ram,
            "gpu_summary": command_output(["bash", "-lc", "lspci 2>/dev/null | grep -Ei 'vga|3d|display' | head -10"]) if sys.platform.startswith("linux") else None}


def git_metadata(root: Path) -> dict:
    def git(*args):
        return command_output(["git", "-C", str(root), *args])
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain"))}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_ms(samples: list[float]) -> dict:
    return {"samples": len(samples), "mean_ms": sum(samples) / len(samples) if samples else None,
            "p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
            "min_ms": min(samples) if samples else None, "max_ms": max(samples) if samples else None}


def write_result(track: str, payload: dict, output_dir: str) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    run_id = f"{track}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    document = {"schema_version": 1, "run_id": run_id, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": track, "host": host_metadata(), "git": git_metadata(Path(__file__).resolve().parents[1]), **payload}
    path = destination / f"{run_id}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
