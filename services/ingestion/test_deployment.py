import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SERVICES_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = SERVICES_ROOT / "server" / "install.sh"
UNIT_TEMPLATE = Path(__file__).with_name("claude-ingestion.service.template")


class IngestionDeploymentTests(unittest.TestCase):
    def test_installer_installs_and_enables_gateway_and_ingestion_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            systemctl = fake_bin / "systemctl"
            log = root / "systemctl.log"
            systemctl.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$SYSTEMCTL_TEST_LOG\"\n",
                encoding="utf-8",
            )
            systemctl.chmod(0o755)
            unit_dir = root / "units"
            env = os.environ.copy()
            env.update({
                "HOME": str(root / "home"),
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "SYSTEMCTL_BIN": str(systemctl),
                "SYSTEMCTL_TEST_LOG": str(log),
                "POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR": str(unit_dir),
                "POC_CLAUDE_ENSEMBLE_DEFER_SERVICE_START": "1",
            })
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            gateway = (unit_dir / "claude-ensemble.service").read_text(encoding="utf-8")
            ingestion = (unit_dir / "claude-ingestion.service").read_text(encoding="utf-8")
            self.assertIn(f"ExecStart=/usr/bin/env python3 {SERVICES_ROOT}/server/ensemble_server.py", gateway)
            self.assertIn(f"ExecStart=/usr/bin/env python3 {SERVICES_ROOT}/ingestion/ingestion_service.py", ingestion)
            self.assertNotIn("%REPO_PATH%", gateway + ingestion)
            commands = log.read_text(encoding="utf-8")
            self.assertIn("--user daemon-reload", commands)
            self.assertIn("--user enable claude-ensemble.service claude-ingestion.service", commands)
            self.assertNotIn("--user start", commands)

    def test_ingestion_unit_is_loopback_only_and_sandboxed_without_hiding_checkout(self):
        unit = UNIT_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("POC_CLAUDE_ENSEMBLE_INGESTION_HOST=127.0.0.1", unit)
        self.assertIn("POC_CLAUDE_ENSEMBLE_INGESTION_PORT=8782", unit)
        self.assertIn("ProtectSystem=strict", unit)
        self.assertIn("ProtectHome=read-only", unit)
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertNotIn("POC_CLAUDE_ENSEMBLE_INGESTION_ENABLE_FTP=true", unit)


if __name__ == "__main__":
    unittest.main()
