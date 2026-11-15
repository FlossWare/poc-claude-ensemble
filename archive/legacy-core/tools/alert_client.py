#!/usr/bin/env python3
"""
RH Alert Client

Session-side client for connecting to Alert Service daemon.
Used in tools and scripts to trigger checks and query alerts.

Usage:
    from tools.alert_client import AlertClient

    client = AlertClient()
    alerts = client.trigger_check()  # Runs checks and sends emails via daemon
    recent = client.get_recent_alerts(days=7)
    client.acknowledge('alert_id')
"""

import json
import socket
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

SOCKET_PATH = Path('/tmp/claude-alert.sock')


class AlertClient:
    """Client for RH Alert Service daemon"""

    def __init__(self, socket_path: Path = SOCKET_PATH, timeout: float = 5.0):
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self.connected = False

    def connect(self) -> bool:
        """Connect to alert service, fail gracefully if down"""
        try:
            if not self.socket_path.exists():
                logger.warning(f"Alert service not running (socket not found: {self.socket_path})")
                return False

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(str(self.socket_path))

            # Test connection with ping
            response = self._send_request({'op': 'ping'})
            if response.get('ok'):
                self.connected = True
                logger.info("Connected to alert service")
                return True
            else:
                logger.warning("Alert service not responding")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to alert service: {e}")
            return False

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to service, return parsed response"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(str(self.socket_path))

            # Send request as JSON
            request_str = json.dumps(request) + '\n'
            sock.sendall(request_str.encode('utf-8'))

            # Read response
            response = b''
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response += chunk
                if b'\n' in response:
                    break

            sock.close()

            return json.loads(response.decode('utf-8').strip())
        except Exception as e:
            logger.warning(f"Request error: {e}")
            return {'ok': False, 'error': str(e)}

    def trigger_check(self, request_id: str = None) -> List[Dict[str, Any]]:
        """Run all checks (cost spike, quality drop), return alerts sent

        Args:
            request_id: Request correlation ID for tracing

        Returns:
            List of alerts triggered (empty if none)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self._send_request({'op': 'trigger_check', 'request_id': request_id})
        if response.get('ok'):
            logger.info(f"[{request_id}] Alert check triggered, {len(response.get('alerts', []))} alerts")
            return response.get('alerts', [])
        logger.warning(f"[{request_id}] Failed to trigger check: {response.get('error')}")
        return []

    def get_recent_alerts(self, days: int = 7, request_id: str = None) -> List[Dict[str, Any]]:
        """List recent alerts from last N days

        Args:
            days: Number of days to look back (default 7)
            request_id: Request correlation ID for tracing

        Returns:
            List of recent alert objects
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self._send_request({'op': 'get_recent_alerts', 'days': days, 'request_id': request_id})
        if response.get('ok'):
            logger.info(f"[{request_id}] Retrieved {len(response.get('alerts', []))} recent alerts")
            return response.get('alerts', [])
        logger.warning(f"[{request_id}] Failed to get recent alerts: {response.get('error')}")
        return []

    def acknowledge(self, alert_id: str, request_id: str = None) -> bool:
        """Mark alert as acknowledged/reviewed

        Args:
            alert_id: Alert identifier
            request_id: Request correlation ID for tracing

        Returns:
            True if successful
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self._send_request({'op': 'acknowledge', 'alert_id': alert_id, 'request_id': request_id})
        if response.get('ok'):
            logger.info(f"[{request_id}] Acknowledged alert: {alert_id}")
        else:
            logger.warning(f"[{request_id}] Failed to acknowledge alert: {response.get('error')}")
        return response.get('ok', False)

    def get_config(self, request_id: str = None) -> Dict[str, Any]:
        """Get alert configuration and thresholds

        Args:
            request_id: Request correlation ID for tracing

        Returns:
            Config dict with thresholds
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self._send_request({'op': 'get_config', 'request_id': request_id})
        if response.get('ok'):
            logger.info(f"[{request_id}] Retrieved alert config")
            return response.get('config', {})
        logger.warning(f"[{request_id}] Failed to get config: {response.get('error')}")
        return {}



if __name__ == '__main__':
    # Test client
    logging.basicConfig(level=logging.INFO)
    client = AlertClient()
    client.connect()

    if client.connected:
        print("✓ Connected to alert service")
        config = client.get_config()
        print(f"  Config: {config}")
    else:
        print("✗ Alert service not available")
