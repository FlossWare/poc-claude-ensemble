"""Run the Sumo anomaly check and record detections to an append-only ledger.

The hourly systemd timer runs this wrapper instead of invoking
``sumo_anomaly check`` directly. The check command reports a detection as
**exit code 3**, which on its own would only mark the unit failed and vanish
into the journal. This wrapper makes a detection durable:

* exit 3  -> append one JSON line to ``~/.FlossWare/ledger/anomalies/detected.jsonl``
             (the future notification hook), then exit 0. A detection is a
             normal outcome, not an operational failure.
* exit 0  -> nothing to record.
* exit 1/2 -> propagate the failure so the timer run shows as failed. Missing
             credentials or a missing trained artifact must be visible.

The ledger is append-only; existing rows are never rewritten. Override the
destination with ``POC_CLAUDE_ENSEMBLE_ANOMALY_LEDGER``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

LEDGER_ENV = "POC_CLAUDE_ENSEMBLE_ANOMALY_LEDGER"
DEFAULT_LEDGER = "~/.FlossWare/ledger/anomalies/detected.jsonl"
REFRESH_ENV = "POC_CLAUDE_ENSEMBLE_SUMO_ANOMALY_REFRESH"
CHECK_MODULE = "learning.train.sumo_anomaly"

#: ``check --fail-on-anomaly`` uses 3 to mean "a recent anomaly exists".
ANOMALY_EXIT = 3

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def ledger_path() -> Path:
    """Return the detection ledger, honouring the test/override env var."""
    value = os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER
    return Path(os.path.expanduser(value))


def refresh_enabled() -> bool:
    """Whether the check should fetch fresh metrics (default yes).

    An hourly timer reading a stale cached series is useless, so refresh is on
    unless explicitly disabled.
    """
    raw = os.environ.get(REFRESH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def check_command(*, refresh: bool | None = None) -> list[str]:
    """Build the detector command this wrapper supervises."""
    command = [
        sys.executable,
        "-m",
        CHECK_MODULE,
        "check",
        "--json",
        "--fail-on-anomaly",
    ]
    if refresh is None:
        refresh = refresh_enabled()
    if refresh:
        command.append("--refresh")
    return command


def _iso(moment: datetime | None = None) -> str:
    value = moment or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_record(
    report: Mapping[str, Any],
    *,
    exit_code: int,
    command: Sequence[str],
    detected_at: datetime | None = None,
) -> dict[str, Any]:
    """Shape one ledger row from the ``--json`` report.

    Carries enough detail for a notifier to render a message without re-reading
    the series: window, model provenance, per-metric counts and the recent
    findings themselves.
    """
    return {
        "detected_at": _iso(detected_at),
        "source": "sumo_anomaly",
        "check_exit_code": exit_code,
        "command": list(command),
        "index": report.get("index"),
        "timeslice": report.get("timeslice"),
        "window_start": report.get("window_start"),
        "window_end": report.get("window_end"),
        "fetched_at": report.get("fetched_at"),
        "recent_minutes": report.get("recent_minutes"),
        "recent_anomalies": report.get("recent_anomalies"),
        "anomalies_by_metric": report.get("anomalies_by_metric") or {},
        "model": report.get("model") or {},
        "recent": report.get("recent") or [],
    }


def append_record(
    record: Mapping[str, Any], path: str | os.PathLike[str] | None = None
) -> Path:
    """Append ``record`` as one JSON line, creating the ledger if needed."""
    target = Path(path) if path is not None else ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    created = not target.exists()
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if created:
        # Metric series are not secrets, but keep operator ledgers owner-only.
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    return target


def parse_report(stdout: str) -> dict[str, Any] | None:
    try:
        document = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    return document if isinstance(document, dict) else None


def _summarize(report: Mapping[str, Any]) -> str:
    return (
        "sumo anomaly check: "
        f"index={report.get('index')} "
        f"window={report.get('window_start')}..{report.get('window_end')} "
        f"points={report.get('points')} "
        f"anomalies={report.get('anomalies')} "
        f"recent={report.get('recent_anomalies')}"
    )


def _describe(report: Mapping[str, Any]) -> list[str]:
    lines = []
    for finding in report.get("recent") or []:
        if not isinstance(finding, Mapping):
            continue
        metrics = sorted(
            deviation.get("metric")
            for deviation in finding.get("deviations") or []
            if isinstance(deviation, Mapping) and deviation.get("is_deviating")
        )
        lines.append(
            f"  - {finding.get('timestamp')} score={finding.get('anomaly_score')}"
            f" metrics={','.join(str(m) for m in metrics) or 'none'}"
        )
    return lines


def run(
    *,
    runner: Runner | None = None,
    ledger: str | os.PathLike[str] | None = None,
    now: datetime | None = None,
) -> int:
    """Execute the check and record a detection. Returns a process exit code."""
    command = check_command()
    invoke = runner or subprocess.run
    result = invoke(command, capture_output=True, text=True)
    report = parse_report(result.stdout)

    if result.returncode == ANOMALY_EXIT:
        if report is None:
            print(
                "sumo anomaly check exited 3 but produced no parseable JSON report",
                file=sys.stderr,
            )
            return 1
        record = build_record(
            report, exit_code=result.returncode, command=command, detected_at=now
        )
        destination = append_record(record, ledger)
        print(_summarize(report))
        for line in _describe(report):
            print(line)
        print(
            f"ALERT: {record.get('recent_anomalies')} recent Sumo anomal"
            f"{'y' if record.get('recent_anomalies') == 1 else 'ies'} "
            f"recorded to {destination}",
            file=sys.stderr,
        )
        # The detection was handled; do not mark the timer unit failed.
        return 0

    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        message = f"sumo anomaly check failed (exit {result.returncode})"
        if detail:
            message = f"{message}: {detail}"
        print(message, file=sys.stderr)
        return result.returncode

    if report is not None:
        print(_summarize(report))
    else:
        print("sumo anomaly check: no recent anomalies")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # No positional arguments today; kept for symmetry with other CLIs.
    return run()


if __name__ == "__main__":
    sys.exit(main())
