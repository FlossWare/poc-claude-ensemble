#!/usr/bin/env bash
# Install and ENABLE the hourly Sumo anomaly check as systemd *user* units.
#
# Unlike the GA installer (report-only, installs disabled), this check is meant
# to run unattended: it enables and starts the timer. Re-running is safe and
# idempotent -- the unit files are rewritten with identical content and
# `systemctl enable` is a no-op once enabled.
#
# Overrides (used by the tests):
#   POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR  where to write the units
#   SYSTEMCTL_BIN                      systemctl binary to call
#   LOGINCTL_BIN                       loginctl binary to call (linger check)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
SYSTEMD_DIR="${POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR:-${HOME}/.config/systemd/user}"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-systemctl}"
LOGINCTL_BIN="${LOGINCTL_BIN:-loginctl}"

SERVICE_NAME="poc-claude-ensemble-sumo-anomaly.service"
TIMER_NAME="poc-claude-ensemble-sumo-anomaly.timer"
SERVICE_TEMPLATE="$HERE/${SERVICE_NAME}.template"
TIMER_TEMPLATE="$HERE/${TIMER_NAME}.template"

if [[ ! -f "$SERVICE_TEMPLATE" || ! -f "$TIMER_TEMPLATE" ]]; then
  echo "Error: Sumo anomaly unit templates not found under $HERE" >&2
  exit 1
fi

# Escape '&' so a repository path containing it is substituted literally.
REPO_SED="${REPO_ROOT//&/\\&}"

mkdir -p "$SYSTEMD_DIR"
sed "s|%REPO_PATH%|$REPO_SED|g" "$SERVICE_TEMPLATE" > "$SYSTEMD_DIR/$SERVICE_NAME"
sed "s|%REPO_PATH%|$REPO_SED|g" "$TIMER_TEMPLATE" > "$SYSTEMD_DIR/$TIMER_NAME"
chmod 0644 "$SYSTEMD_DIR/$SERVICE_NAME" "$SYSTEMD_DIR/$TIMER_NAME"

"$SYSTEMCTL_BIN" --user daemon-reload
# `enable --now` starts the timer immediately and is idempotent on re-run.
"$SYSTEMCTL_BIN" --user enable --now "$TIMER_NAME"

echo "Installed and started: $SERVICE_NAME and $TIMER_NAME"
echo "  units:   $SYSTEMD_DIR"
echo "  repo:    $REPO_ROOT"
echo "  schedule: hourly (RandomizedDelaySec=5m, Persistent=true)"
echo
echo "Check status with:"
echo "  systemctl --user list-timers $TIMER_NAME"
echo "  systemctl --user start $SERVICE_NAME   # run the check once now"
echo "  journalctl --user -u $SERVICE_NAME -n 50 --no-pager"

# A detection is written here, one JSON object per line:
echo "  ledger:  ${POC_CLAUDE_ENSEMBLE_ANOMALY_LEDGER:-$HOME/.FlossWare/ledger/anomalies/detected.jsonl}"

# A user timer only fires while the user's systemd instance is alive. Under
# Linger=no that instance dies at logout, so an unattended hourly check silently
# stops. Query with loginctl: `systemctl --user show-user` does not exist on
# systemd 259 and would report a bogus value.
LINGER="$("$LOGINCTL_BIN" show-user "$(id -un)" -p Linger --value 2>/dev/null || echo unknown)"
echo
if [[ "$LINGER" != "yes" ]]; then
  echo "CAVEAT: this account's systemd Linger setting is '${LINGER}'."
  echo "With Linger=no, the hourly timer does NOT fire while you are logged out."
  echo "To keep the anomaly check running after logout, decide deliberately whether to run:"
  echo "  loginctl enable-linger $(id -un)"
  echo "(Enabling linger keeps all of this account's user services running.)"
else
  echo "Linger=yes: the hourly timer will fire even while you are logged out."
fi
