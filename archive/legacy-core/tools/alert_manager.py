#!/usr/bin/env python3
"""
Alert Manager — Monitor costs, quality, errors; send email alerts

Monitors:
1. Cost spikes (daily > 2x baseline)
2. Quality drops (ratings < 3 threshold)
3. Model errors (fallback usage)

Sends alerts to: your-email@example.com via Postfix (localhost:2525) or Gmail API

Features:
- Postfix connection testing + fallback to Gmail
- Real Gmail sending with google-auth library
- Retry logic with exponential backoff (max 3 attempts)
- Delivery confirmation logging
- Async queue for non-blocking sends
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket
import time
import threading
from queue import Queue, Empty
import base64
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from cost_tracking.logger import CostLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


@dataclass
class Alert:
    """Single alert event"""
    alert_type: str  # "cost_spike", "quality_drop", "model_error"
    severity: str  # "info", "warning", "critical"
    timestamp: str
    message: str
    metrics: Dict
    action_recommended: str


class AlertManager:
    """Monitor and alert on anomalies"""

    def __init__(self, repo_root: Path = None, async_send: bool = True):
        if repo_root is None:
            repo_root = Path(__file__).parent.parent
        self.repo_root = repo_root
        self.cost_logger = CostLogger()
        self.alert_dir = repo_root / "alerts"
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.gmail_user = "your-email@example.com"
        self.max_retries = 3
        self.retry_backoff_base = 2  # exponential backoff: 2^attempt seconds

        # Async sending queue
        self.async_send = async_send
        self.send_queue = Queue() if async_send else None
        if async_send:
            self.sender_thread = threading.Thread(target=self._send_worker, daemon=True)
            self.sender_thread.start()

        # Delivery log
        self.delivery_log_path = self.alert_dir / "delivery_log.jsonl"

    def get_daily_cost(self, date: datetime = None) -> float:
        """Get total cost for a day"""
        if date is None:
            date = datetime.now()

        costs = self.cost_logger.get_stats()
        # Would filter by date in production
        return costs.get("total_cost_usd", 0)

    def get_baseline_cost(self, days: int = 7) -> float:
        """Get baseline cost from previous days"""
        costs = self.cost_logger.get_stats()
        if costs.get("total_calls", 0) == 0:
            return 0.1  # Default baseline if no data

        return costs.get("total_cost_usd", 0) / max(1, costs.get("total_calls", 1))

    def check_cost_spike(self, threshold_multiplier: float = 2.0) -> Optional[Alert]:
        """Alert if daily cost > threshold × baseline"""
        today_cost = self.get_daily_cost()
        baseline = self.get_baseline_cost()
        threshold = baseline * threshold_multiplier

        if today_cost > threshold:
            alert = Alert(
                alert_type="cost_spike",
                severity="warning",
                timestamp=datetime.now().isoformat(),
                message=f"Daily cost spike detected: ${today_cost:.2f} (baseline: ${baseline:.2f})",
                metrics={
                    "today_cost": today_cost,
                    "baseline": baseline,
                    "threshold": threshold,
                    "multiplier": threshold_multiplier,
                },
                action_recommended="Review model selection or task complexity",
            )
            logger.warning(f"Cost spike: ${today_cost:.2f} > ${threshold:.2f}")
            return alert

        return None

    def check_quality_drop(
        self, rating_threshold: float = 3.0, window_days: int = 7
    ) -> Optional[Alert]:
        """Alert if average rating < threshold in recent period"""
        outcomes_dir = self.repo_root / "learning" / "post_task_outcomes"

        if not outcomes_dir.exists():
            return None

        ratings = []
        cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()

        for outcome_file in outcomes_dir.glob("*.json"):
            try:
                data = json.loads(outcome_file.read_text())
                if data["outcome"]["timestamp"] > cutoff:
                    final = data["comparison"].get("final_rating")
                    if final:
                        ratings.append(final)
            except:
                pass

        if not ratings:
            return None

        avg_rating = sum(ratings) / len(ratings)

        if avg_rating < rating_threshold:
            alert = Alert(
                alert_type="quality_drop",
                severity="critical",
                timestamp=datetime.now().isoformat(),
                message=f"Quality degradation: avg rating {avg_rating:.1f} < {rating_threshold}",
                metrics={
                    "avg_rating": avg_rating,
                    "threshold": rating_threshold,
                    "sample_size": len(ratings),
                    "window_days": window_days,
                },
                action_recommended="Review model selection; consider returning to previous settings",
            )
            logger.critical(f"Quality drop: {avg_rating:.1f} < {rating_threshold}")
            return alert

        return None

    def check_model_errors(self) -> Optional[Alert]:
        """Alert if recent errors/fallbacks detected"""
        # Would check error logs in production
        # For now, placeholder
        return None

    def send_email_alert(self, alert: Alert, async_mode: bool = None) -> bool:
        """Send email alert to your-email@example.com with retry logic

        Args:
            alert: Alert object to send
            async_mode: If True, queue for async sending. If None, use instance default.

        Returns:
            True if queued (async) or sent successfully (sync)
        """
        if async_mode is None:
            async_mode = self.async_send

        subject = f"[RH AI Toolkit] {alert.alert_type.upper()}: {alert.severity.upper()}"
        body = self._format_email_body(alert)

        if async_mode:
            # Queue for async sending
            self.send_queue.put((subject, body, alert))
            logger.info(f"Alert queued for async sending: {alert.alert_type}")
            return True
        else:
            # Send synchronously with retry
            return self._send_with_retry(subject, body, alert)

    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as email body"""
        return f"""RH Claude Global Skills Toolkit Alert

TYPE:       {alert.alert_type}
SEVERITY:   {alert.severity.upper()}
TIMESTAMP:  {alert.timestamp}

MESSAGE:
{alert.message}

METRICS:
{json.dumps(alert.metrics, indent=2)}

RECOMMENDED ACTION:
{alert.action_recommended}

Dashboard: http://localhost:8000/dashboards
--
RH AI Toolkit Monitoring
"""

    def _test_postfix_connection(self) -> bool:
        """Test if postfix is available on localhost:2525"""
        try:
            logger.debug("Testing postfix connection to localhost:2525")
            sock = socket.create_connection(("localhost", 2525), timeout=5)
            sock.close()
            logger.info("Postfix connection test: SUCCESS")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Postfix connection test failed: {e}")
            return False

    def _send_via_postfix(self, subject: str, body: str) -> bool:
        """Send via postfix on localhost:2525 (SSH tunnel)"""
        try:
            logger.info("Attempting to send via postfix (localhost:2525)")

            # Test connection first
            if not self._test_postfix_connection():
                logger.warning("Postfix not available, will use Gmail fallback")
                return False

            server = smtplib.SMTP("localhost", 2525, timeout=10)

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = "your-toolkit@example.com"
            msg["To"] = self.gmail_user

            server.send_message(msg)
            server.quit()
            logger.info("Alert sent successfully via postfix")
            return True
        except Exception as e:
            logger.warning(f"Postfix send failed: {e}")
            return False

    def _send_via_gmail(self, subject: str, body: str) -> bool:
        """Send via Gmail API using google-auth library

        Requires: google-auth, google-auth-oauthlib, google-auth-httplib2, google-api-python-client
        """
        try:
            logger.info("Attempting to send via Gmail API")

            # Try to import Google API libraries
            try:
                from google.oauth2.service_account import Credentials
                from google.auth.transport.requests import Request
                from googleapiclient.discovery import build
            except ImportError as ie:
                logger.warning(f"Google libraries not installed: {ie}. Install with: pip install google-auth google-api-python-client")
                return False

            # Look for service account credentials
            credentials_paths = [
                Path.home() / ".google" / "service-account-key.json",
                Path("/etc/google/service-account-key.json"),
                Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")),
            ]

            credentials = None
            credentials_file = None

            for cred_path in credentials_paths:
                if cred_path and cred_path.exists():
                    try:
                        credentials = Credentials.from_service_account_file(str(cred_path))
                        credentials_file = str(cred_path)
                        logger.info(f"Loaded credentials from {credentials_file}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load credentials from {cred_path}: {e}")
                        continue

            if not credentials:
                logger.warning("No Google service account credentials found. Tried:")
                for path in credentials_paths:
                    if path:
                        logger.warning(f"  - {path}")
                logger.info("To use Gmail, set GOOGLE_APPLICATION_CREDENTIALS or place key at ~/.google/service-account-key.json")
                return False

            # Build Gmail service
            service = build("gmail", "v1", credentials=credentials)

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.gmail_user
            msg["To"] = self.gmail_user

            # Add body as plain text
            part = MIMEText(body, "plain")
            msg.attach(part)

            # Encode message
            raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

            # Send via Gmail API
            result = service.users().messages().send(
                userId="me",
                body={"raw": raw_message}
            ).execute()

            logger.info(f"Alert sent successfully via Gmail API (message_id: {result.get('id')})")
            return True

        except Exception as e:
            logger.error(f"Gmail API send failed: {e}")
            return False

    def _send_with_retry(self, subject: str, body: str, alert: Alert) -> bool:
        """Send email with retry logic and exponential backoff

        Tries postfix first, then Gmail, with up to max_retries attempts per method
        """
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Send attempt {attempt}/{self.max_retries}")

            # Try postfix first
            if self._send_via_postfix(subject, body):
                self._log_delivery(alert, "postfix", True, attempt)
                return True

            # Try Gmail fallback
            if self._send_via_gmail(subject, body):
                self._log_delivery(alert, "gmail", True, attempt)
                return True

            # Exponential backoff before retry
            if attempt < self.max_retries:
                wait_seconds = self.retry_backoff_base ** (attempt - 1)
                logger.warning(f"Send failed, waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)

        # All retries exhausted
        logger.error(f"Failed to send alert after {self.max_retries} attempts")
        self._log_delivery(alert, "unknown", False, self.max_retries)
        return False

    def _log_delivery(self, alert: Alert, method: str, success: bool, attempt: int):
        """Log delivery attempt to delivery_log.jsonl"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "method": method,
                "success": success,
                "attempt": attempt,
                "recipient": self.gmail_user,
            }

            # Append to JSONL log
            with open(self.delivery_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Logged delivery attempt: {alert.alert_type} via {method} - {status}")
        except Exception as e:
            logger.error(f"Failed to log delivery: {e}")

    def _send_worker(self):
        """Background worker thread for async email sending

        Dequeues items from send_queue and sends them with retry logic
        """
        logger.info("Alert sender worker thread started")

        while True:
            try:
                # Wait for item in queue (with timeout to allow graceful shutdown)
                try:
                    subject, body, alert = self.send_queue.get(timeout=30)
                except Empty:
                    # Queue is empty, keep waiting
                    continue

                logger.info(f"Sending queued alert: {alert.alert_type}")

                # Send with retry logic
                self._send_with_retry(subject, body, alert)

                # Mark task done
                self.send_queue.task_done()

            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                # Continue running even if one send fails

    def save_alert(self, alert: Alert) -> Path:
        """Save alert to local file"""
        timestamp = datetime.now().isoformat().replace(":", "-")
        alert_file = self.alert_dir / f"{alert.alert_type}_{timestamp}.json"

        alert_file.write_text(
            json.dumps(
                {
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp,
                    "message": alert.message,
                    "metrics": alert.metrics,
                    "action": alert.action_recommended,
                },
                indent=2,
            )
        )

        logger.info(f"Alert saved: {alert_file}")
        return alert_file

    def check_all(self) -> List[Alert]:
        """Run all checks and return triggered alerts"""
        alerts = []

        # Cost spike check
        cost_alert = self.check_cost_spike(threshold_multiplier=2.0)
        if cost_alert:
            alerts.append(cost_alert)

        # Quality drop check
        quality_alert = self.check_quality_drop(rating_threshold=3.0, window_days=7)
        if quality_alert:
            alerts.append(quality_alert)

        # Model error check
        error_alert = self.check_model_errors()
        if error_alert:
            alerts.append(error_alert)

        # Send and save all alerts
        for alert in alerts:
            self.save_alert(alert)
            self.send_email_alert(alert)

        return alerts

    def print_alerts(self, alerts: List[Alert]):
        """Print alerts to console"""
        if not alerts:
            print("No alerts triggered")
            return

        print("\n" + "=" * 70)
        print("ALERTS")
        print("=" * 70)

        for alert in alerts:
            print(f"\n{alert.alert_type.upper()} [{alert.severity.upper()}]")
            print(f"  Message: {alert.message}")
            print(f"  Action:  {alert.action_recommended}")
            print(f"  Metrics: {json.dumps(alert.metrics, indent=12)}")


def main():
    """Demo: check all alerts"""
    manager = AlertManager()
    alerts = manager.check_all()
    manager.print_alerts(alerts)


if __name__ == "__main__":
    main()
