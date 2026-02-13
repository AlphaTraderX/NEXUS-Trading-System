"""
Security Audit Logging

Tamper-evident audit trail for all trading activity and system access.
Required for regulatory compliance and security monitoring.
"""

import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditEventType(Enum):
    """Types of auditable events."""

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"

    # Authentication
    LICENSE_CHECK = "license_check"
    SECRET_ACCESS = "secret_access"

    # Trading events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk events
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    HEAT_LIMIT_REACHED = "heat_limit_reached"

    # Security events
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Single audit event."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user: str
    ip_address: Optional[str]
    component: str
    action: str
    details: Dict[str, Any]
    outcome: str  # success, failure, warning
    checksum: str  # Tamper detection

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["event_type"] = self.event_type.value
        return d


class AuditLogger:
    """
    Tamper-evident audit logging system.

    Features:
    - Chained checksums for tamper detection
    - Structured JSON-lines format
    - Daily log rotation
    - Query and integrity verification
    """

    DEFAULT_AUDIT_DIR = "audit_logs"

    def __init__(self, audit_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._event_count = 0
        self._last_checksum = "genesis"
        self._audit_dir = Path(audit_dir or self.DEFAULT_AUDIT_DIR)
        self._audit_dir.mkdir(exist_ok=True)

    def log(
        self,
        event_type: AuditEventType,
        component: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        user: Optional[str] = None,
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            component: System component (scanner, executor, etc.)
            action: Specific action taken
            details: Additional details
            outcome: success, failure, or warning
            user: User who triggered the event

        Returns:
            Event ID
        """
        with self._lock:
            self._event_count += 1
            now = datetime.now(timezone.utc)

            event_id = f"{now.strftime('%Y%m%d%H%M%S')}-{self._event_count:06d}"

            # Chain checksum for tamper detection
            checksum_data = (
                f"{self._last_checksum}|{event_id}|{event_type.value}|{action}"
            )
            checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

            event = AuditEvent(
                event_id=event_id,
                timestamp=now,
                event_type=event_type,
                user=user or os.environ.get("USERNAME", "system"),
                ip_address=self._get_ip(),
                component=component,
                action=action,
                details=details or {},
                outcome=outcome,
                checksum=checksum,
            )

            self._write_event(event)
            self._last_checksum = checksum

            # Alert on critical events
            if event_type in (
                AuditEventType.KILL_SWITCH_ACTIVATED,
                AuditEventType.UNAUTHORIZED_ACCESS,
                AuditEventType.SUSPICIOUS_ACTIVITY,
            ):
                self._alert(event)

            return event_id

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to daily log file."""
        today = event.timestamp.strftime("%Y-%m-%d")
        log_file = self._audit_dir / f"audit_{today}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def _get_ip(self) -> Optional[str]:
        """Get local IP address."""
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None

    def _alert(self, event: AuditEvent) -> None:
        """Log critical events to stderr. Integration point for Discord/Telegram."""
        import sys

        print(
            f"SECURITY ALERT: {event.event_type.value} - {event.action}",
            file=sys.stderr,
        )

    def query(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        component: Optional[str] = None,
    ) -> List[dict]:
        """
        Query audit logs.

        Args:
            start_date: Start of date range
            end_date: End of date range
            event_types: Filter by event types
            component: Filter by component

        Returns:
            List of matching event dicts
        """
        type_values = {e.value for e in event_types} if event_types else None
        events: List[dict] = []

        for log_file in sorted(self._audit_dir.glob("audit_*.jsonl")):
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_time = datetime.fromisoformat(data["timestamp"])

                    if start_date and event_time < start_date:
                        continue
                    if end_date and event_time > end_date:
                        continue
                    if type_values and data["event_type"] not in type_values:
                        continue
                    if component and data["component"] != component:
                        continue

                    events.append(data)

        return events

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify audit log integrity by replaying checksum chain.

        Returns:
            Dict with verified (bool), events_checked (int), issues (list)
        """
        issues: List[dict] = []
        last_checksum = "genesis"
        event_count = 0

        for log_file in sorted(self._audit_dir.glob("audit_*.jsonl")):
            with open(log_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        issues.append(
                            {
                                "file": str(log_file),
                                "line": line_num,
                                "issue": f"Parse error: {e}",
                            }
                        )
                        continue

                    event_count += 1
                    expected_data = f"{last_checksum}|{data['event_id']}|{data['event_type']}|{data['action']}"
                    expected = hashlib.sha256(expected_data.encode()).hexdigest()[:16]

                    if data.get("checksum") != expected:
                        issues.append(
                            {
                                "file": str(log_file),
                                "line": line_num,
                                "event_id": data.get("event_id"),
                                "issue": "Checksum mismatch - possible tampering",
                            }
                        )

                    last_checksum = data.get("checksum", last_checksum)

        return {
            "verified": len(issues) == 0,
            "events_checked": event_count,
            "issues": issues,
        }


# Global singleton
_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger(audit_dir: Optional[str] = None) -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    with _audit_lock:
        if _audit_logger is None:
            _audit_logger = AuditLogger(audit_dir=audit_dir)
        return _audit_logger


def audit(
    event_type: AuditEventType,
    component: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    outcome: str = "success",
) -> str:
    """Convenience function to log an audit event."""
    return get_audit_logger().log(event_type, component, action, details, outcome)
