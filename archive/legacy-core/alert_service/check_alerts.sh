#!/bin/bash
# Simple cron wrapper - calls alert client every 30 minutes
# Add to crontab with: */30 * * * * /path/to/check_alerts.sh

set -e

# Find repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Log file
LOG_FILE="$HOME/.claude/check_alerts.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Python command to trigger check
python3 << 'EOF'
import sys
from pathlib import Path

# Add repo to path
repo_root = Path("$REPO_ROOT")
sys.path.insert(0, str(repo_root))

from tools.alert_client import AlertClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / '.claude' / 'check_alerts.log'),
    ]
)
logger = logging.getLogger(__name__)

try:
    client = AlertClient()
    alerts = client.trigger_check()
    if alerts:
        logger.info(f"Triggered check - {len(alerts)} alert(s) generated")
        for alert in alerts:
            logger.info(f"  {alert.get('alert_type')}: {alert.get('message')}")
    else:
        logger.debug("Triggered check - no alerts")
except Exception as e:
    logger.error(f"Error triggering check: {e}")
    sys.exit(1)
EOF

exit 0
