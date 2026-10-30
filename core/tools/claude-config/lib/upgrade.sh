#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/lib/core.sh"

upgrade(){
  local installed="none"
  if [ -f "$CC_MANIFEST" ]; then
    installed="$(python3 - "$CC_MANIFEST" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1],encoding="utf-8")).get("version","unknown"))
except Exception: print("unknown")
PY
)"
  fi
  case "$installed" in
    none|0.1|unknown) : ;;
    0.2) ok "claude-config is already at 0.2"; return 0 ;;
    *) die "unsupported claude-config version: $installed" ;;
  esac
  install "$@"
}
