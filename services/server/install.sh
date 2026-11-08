#!/usr/bin/env bash
# Install the canonical Claude Ensemble REST gateway as a systemd user service.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="claude-ensemble.service"
SERVICE_TEMPLATE="$REPO_ROOT/server/claude-ensemble.service.template"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-systemctl}"
SYSTEMD_DIR="${POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR:-${HOME}/.config/systemd/user}"
SERVICE_FILE="$SYSTEMD_DIR/$SERVICE_NAME"
INGESTION_SERVICE_NAME="claude-ingestion.service"
INGESTION_SERVICE_TEMPLATE="$REPO_ROOT/ingestion/claude-ingestion.service.template"
INGESTION_SERVICE_FILE="$SYSTEMD_DIR/$INGESTION_SERVICE_NAME"

if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
  echo "Error: service template not found: $SERVICE_TEMPLATE" >&2
  exit 1
fi
if [[ ! -f "$INGESTION_SERVICE_TEMPLATE" ]]; then
  echo "Error: ingestion service template not found: $INGESTION_SERVICE_TEMPLATE" >&2
  exit 1
fi

mkdir -p "$SYSTEMD_DIR"
sed "s|%REPO_PATH%|$REPO_ROOT|g" "$SERVICE_TEMPLATE" > "$SERVICE_FILE"
sed "s|%REPO_PATH%|$REPO_ROOT|g" "$INGESTION_SERVICE_TEMPLATE" > "$INGESTION_SERVICE_FILE"
chmod 0644 "$SERVICE_FILE" "$INGESTION_SERVICE_FILE"

"$SYSTEMCTL_BIN" --user daemon-reload
"$SYSTEMCTL_BIN" --user enable "$SERVICE_NAME" "$INGESTION_SERVICE_NAME"
if [[ "${POC_CLAUDE_ENSEMBLE_DEFER_SERVICE_START:-0}" == "1" ]]; then
  echo "REST gateway and ingestion service installed and enabled; start deferred to the top-level installer."
  exit 0
fi
"$SYSTEMCTL_BIN" --user start "$INGESTION_SERVICE_NAME"
"$SYSTEMCTL_BIN" --user start "$SERVICE_NAME"

for unit in "$INGESTION_SERVICE_NAME" "$SERVICE_NAME"; do
  if ! "$SYSTEMCTL_BIN" --user is-active --quiet "$unit"; then
    echo "Service failed to start: $unit. Check: journalctl --user -u $unit" >&2
    exit 1
  fi
done

# A running unit only establishes process liveness. Do not report a successful
# install until the gateway REST endpoint and required Graph/Memory dependencies
# answer their health probes. Keep this bounded so an unavailable dependency
# produces a useful failure instead of hanging installation indefinitely.
if ! python3 - <<'PY'
import time
import urllib.error
import urllib.request

checks = {
    "REST gateway": __import__("os").environ.get("POC_CLAUDE_ENSEMBLE_READINESS_URL", "http://127.0.0.1:8080/api/v1/ready"),
    "ingestion service": __import__("os").environ.get("POC_CLAUDE_ENSEMBLE_INGESTION_READINESS_URL", "http://127.0.0.1:8782/health"),
}
timeout = float(__import__("os").environ.get("POC_CLAUDE_ENSEMBLE_READINESS_TIMEOUT_SECONDS", "20"))
for name, url in checks.items():
    deadline = time.monotonic() + timeout
    last_error = "readiness endpoint did not become ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f"{name} is ready.")
                    break
                last_error = f"readiness endpoint returned HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    else:
        print(f"ERROR: {name} is active but not ready: {last_error}", file=__import__("sys").stderr)
        raise SystemExit(1)
PY
then
  echo "REST gateway is active but dependencies are not ready. Check Graph, Memory, and: journalctl --user -u $SERVICE_NAME" >&2
  exit 1
fi

echo "REST gateway and ingestion service installed and ready: $SERVICE_NAME, $INGESTION_SERVICE_NAME"
