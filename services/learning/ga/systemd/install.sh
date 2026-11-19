#!/usr/bin/env bash
# Install the GA tuning units as systemd *user* units, DISABLED.
#
# This installer never enables the timer and never enables linger. Report-only
# is the shipped mode; promotion still requires a human or an explicit
# POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY=1 override in the unit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
SYSTEMD_DIR="${POC_CLAUDE_ENSEMBLE_USER_UNIT_DIR:-${HOME}/.config/systemd/user}"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-systemctl}"

SERVICE_NAME="poc-claude-ensemble-ga-thompson.service"
TIMER_NAME="poc-claude-ensemble-ga-thompson.timer"
SERVICE_TEMPLATE="$HERE/${SERVICE_NAME}.template"
TIMER_TEMPLATE="$HERE/${TIMER_NAME}.template"

if [[ ! -f "$SERVICE_TEMPLATE" || ! -f "$TIMER_TEMPLATE" ]]; then
  echo "Error: GA unit templates not found under $HERE" >&2
  exit 1
fi

mkdir -p "$SYSTEMD_DIR"
sed "s|%REPO_PATH%|$REPO_ROOT|g" "$SERVICE_TEMPLATE" > "$SYSTEMD_DIR/$SERVICE_NAME"
sed "s|%REPO_PATH%|$REPO_ROOT|g" "$TIMER_TEMPLATE" > "$SYSTEMD_DIR/$TIMER_NAME"
chmod 0644 "$SYSTEMD_DIR/$SERVICE_NAME" "$SYSTEMD_DIR/$TIMER_NAME"

"$SYSTEMCTL_BIN" --user daemon-reload

# Install only. No `enable`, no `start`, no `loginctl enable-linger`.
echo "Installed (DISABLED): $SERVICE_NAME and $TIMER_NAME"
echo "Nothing will run until you explicitly enable the timer:"
echo "  systemctl --user enable --now $TIMER_NAME"

LINGER="$("$SYSTEMCTL_BIN" --user show-user "$(id -un)" -p Linger --value 2>/dev/null || echo unknown)"
echo
echo "CAVEAT: this account's systemd Linger setting is '${LINGER}'."
echo "With Linger=no, user timers do NOT fire while you are logged out."
echo "This installer will not change it. Decide deliberately whether to run:"
echo "  loginctl enable-linger $(id -un)"
echo "(Enabling linger keeps user services running after logout.)"
