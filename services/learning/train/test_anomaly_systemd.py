"""Tests for the Sumo anomaly systemd units, installer, and alerting wrapper.

Covers the three failure modes that matter for an unattended timer:

* the unit files parse and carry the required scheduling/alerting directives;
* ``install.sh`` is idempotent and enables the timer;
* a detection (check exit 3) appends exactly one durable ledger row, while a
  clean run writes nothing and an operational failure is not swallowed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from learning.train import anomaly_alert

TRAIN_DIR = Path(__file__).resolve().parent
SYSTEMD_DIR = TRAIN_DIR / "systemd"
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_TEMPLATE = SYSTEMD_DIR / "poc-claude-ensemble-sumo-anomaly.service.template"
TIMER_TEMPLATE = SYSTEMD_DIR / "poc-claude-ensemble-sumo-anomaly.timer.template"
INSTALLER = SYSTEMD_DIR / "install.sh"

SERVICE_NAME = "poc-claude-ensemble-sumo-anomaly.service"
TIMER_NAME = "poc-claude-ensemble-sumo-anomaly.timer"


def _report(**overrides):
    report = {
        "index": "rh_cpav-001",
        "timeslice": "5m",
        "window_start": "2026-11-12T12:00:00+00:00",
        "window_end": "2026-11-19T12:00:00+00:00",
        "fetched_at": "2026-11-19T12:00:05+00:00",
        "points": 2016,
        "anomalies": 5,
        "recent_anomalies": 2,
        "recent_minutes": 60,
        "anomalies_by_metric": {"latency_p95": 2},
        "model": {"trained_at": "2026-11-19T10:00:00+00:00", "contamination": 0.02},
        "recent": [
            {
                "timestamp": "2026-11-19T11:45:00+00:00",
                "anomaly_score": -0.21,
                "is_anomaly": True,
                "recent": True,
                "deviations": [
                    {"metric": "latency_p95", "is_deviating": True, "direction": "above"}
                ],
            }
        ],
    }
    report.update(overrides)
    return report


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["check"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class FakeRunner:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        return _completed(self.returncode, self.stdout, self.stderr)


def _substitute(template: str, destination: Path) -> Path:
    text = template.replace("%REPO_PATH%", str(REPO_ROOT))
    destination.write_text(text, encoding="utf-8")
    return destination


class UnitTemplateTests(unittest.TestCase):
    def test_service_is_oneshot_and_reads_sumo_credentials(self):
        text = SERVICE_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("Type=oneshot", text)
        self.assertIn("ExecStart=/usr/bin/env python3 -m learning.train.anomaly_alert", text)
        self.assertIn("EnvironmentFile=-%h/.config/claude-ensemble/sumo.env", text)
        self.assertIn("PYTHONPATH=%REPO_PATH%/services", text)

    def test_timer_runs_hourly_with_jitter_and_catch_up(self):
        text = TIMER_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("OnCalendar=hourly", text)
        self.assertIn("RandomizedDelaySec=5m", text)
        self.assertIn("Persistent=true", text)
        self.assertIn(f"Unit={SERVICE_NAME}", text)
        self.assertIn("WantedBy=timers.target", text)

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_substituted_units_pass_systemd_analyze_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = _substitute(
                SERVICE_TEMPLATE.read_text(encoding="utf-8"), root / SERVICE_NAME
            )
            timer = _substitute(
                TIMER_TEMPLATE.read_text(encoding="utf-8"), root / TIMER_NAME
            )
            result = subprocess.run(
                ["systemd-analyze", "--user", "verify", str(service), str(timer)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


class InstallerTests(unittest.TestCase):
    def _run_installer(self, root: Path, linger: str) -> subprocess.CompletedProcess:
        fake_bin = root / "bin"
        fake_bin.mkdir(parents=True, exist_ok=True)
        log = root / "systemctl.log"
        systemctl = fake_bin / "systemctl"
        systemctl.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$SYSTEMCTL_TEST_LOG\"\n",
            encoding="utf-8",
        )
        systemctl.chmod(0o755)
        loginctl = fake_bin / "loginctl"
        loginctl.write_text(
            f"#!/usr/bin/env bash\necho '{linger}'\n",
            encoding="utf-8",
        )
        loginctl.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(root / "home"),
                "SYSTEMCTL_BIN": str(systemctl),
                "LOGINCTL_BIN": str(loginctl),
                "SYSTEMCTL_TEST_LOG": str(log),
                "POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR": str(root / "units"),
            }
        )
        env.pop("POC_CLAUDE_ENSEMBLE_ANOMALY_LEDGER", None)
        return subprocess.run(
            ["bash", str(INSTALLER)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def test_installer_is_idempotent_and_enables_the_timer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._run_installer(root, linger="no")
            self.assertEqual(first.returncode, 0, first.stderr)
            service = root / "units" / SERVICE_NAME
            timer = root / "units" / TIMER_NAME
            first_service = service.read_text(encoding="utf-8")
            first_timer = timer.read_text(encoding="utf-8")

            second = self._run_installer(root, linger="no")
            self.assertEqual(second.returncode, 0, second.stderr)

            # Idempotent: the second run leaves byte-identical units.
            self.assertEqual(service.read_text(encoding="utf-8"), first_service)
            self.assertEqual(timer.read_text(encoding="utf-8"), first_timer)

            self.assertNotIn("%REPO_PATH%", first_service + first_timer)
            self.assertIn(f"PYTHONPATH={REPO_ROOT}/services", first_service)
            self.assertIn(f"Unit={SERVICE_NAME}", first_timer)

            commands = (root / "systemctl.log").read_text(encoding="utf-8")
            self.assertEqual(commands.count("--user daemon-reload"), 2)
            self.assertEqual(commands.count(f"--user enable --now {TIMER_NAME}"), 2)

    def test_installer_reports_linger_caveat_when_not_lingering(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._run_installer(Path(temporary), linger="no")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CAVEAT", result.stdout)
            self.assertIn("does NOT fire while you are logged out", result.stdout)
            self.assertIn("loginctl enable-linger", result.stdout)

    def test_installer_omits_caveat_when_lingering(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._run_installer(Path(temporary), linger="yes")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Linger=yes", result.stdout)
            self.assertNotIn("CAVEAT", result.stdout)


class AlertingTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.ledger = Path(self._temporary.name) / "anomalies" / "detected.jsonl"

    def _rows(self):
        if not self.ledger.exists():
            return []
        return [
            json.loads(line)
            for line in self.ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_detection_appends_timestamped_ledger_row(self):
        runner = FakeRunner(3, stdout=json.dumps(_report()))
        code = anomaly_alert.run(
            runner=runner,
            ledger=self.ledger,
            now=datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(code, 0)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["detected_at"], "2026-11-19T12:00:00Z")
        self.assertEqual(row["source"], "sumo_anomaly")
        self.assertEqual(row["check_exit_code"], 3)
        self.assertEqual(row["index"], "rh_cpav-001")
        self.assertEqual(row["recent_anomalies"], 2)
        self.assertEqual(row["recent"][0]["timestamp"], "2026-11-19T11:45:00+00:00")
        self.assertEqual(row["anomalies_by_metric"], {"latency_p95": 2})

        # The wrapper invoked the documented check contract.
        command, _kwargs = runner.calls[0]
        self.assertIn("check", command)
        self.assertIn("--json", command)
        self.assertIn("--fail-on-anomaly", command)

    def test_ledger_is_append_only_across_detections(self):
        runner = FakeRunner(3, stdout=json.dumps(_report()))
        anomaly_alert.run(runner=runner, ledger=self.ledger)
        anomaly_alert.run(runner=runner, ledger=self.ledger)
        self.assertEqual(len(self._rows()), 2)

    def test_clean_run_writes_nothing(self):
        runner = FakeRunner(0, stdout=json.dumps(_report(recent_anomalies=0, recent=[])))
        code = anomaly_alert.run(runner=runner, ledger=self.ledger)
        self.assertEqual(code, 0)
        self.assertFalse(self.ledger.exists())

    def test_operational_failure_propagates_and_writes_nothing(self):
        runner = FakeRunner(2, stderr="Sumo credentials not found")
        code = anomaly_alert.run(runner=runner, ledger=self.ledger)
        self.assertEqual(code, 2)
        self.assertFalse(self.ledger.exists())

    def test_unparseable_report_on_anomaly_is_an_error(self):
        runner = FakeRunner(3, stdout="not json")
        code = anomaly_alert.run(runner=runner, ledger=self.ledger)
        self.assertEqual(code, 1)
        self.assertFalse(self.ledger.exists())

    def test_refresh_can_be_disabled_via_environment(self):
        with mock.patch.dict(
            os.environ, {anomaly_alert.REFRESH_ENV: "0"}, clear=False
        ):
            runner = FakeRunner(0, stdout=json.dumps(_report(recent_anomalies=0)))
            anomaly_alert.run(runner=runner, ledger=self.ledger)
        command, _kwargs = runner.calls[0]
        self.assertNotIn("--refresh", command)


if __name__ == "__main__":
    unittest.main()
