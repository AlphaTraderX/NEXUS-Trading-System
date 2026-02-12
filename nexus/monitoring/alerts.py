"""
NEXUS Monitoring Alerts

Event-based alert system for threshold breaches, failures, and important events.
Separate from trading signal delivery (nexus/delivery/).
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    RISK = "risk"
    SYSTEM = "system"
    TRADING = "trading"
    PERFORMANCE = "performance"
    CONNECTION = "connection"


@dataclass
class MonitoringAlert:
    """A monitoring alert."""

    event_type: str
    severity: AlertSeverity
    category: AlertCategory
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
        }


class MonitoringAlertManager:
    """
    Manages monitoring alerts and event handlers.

    Events:
    - circuit_breaker_triggered
    - kill_switch_activated
    - drawdown_warning
    - connection_lost
    - data_feed_stale
    - edge_decay_detected
    - reconciliation_mismatch
    - daily_loss_limit
    - weekly_loss_limit
    """

    # Predefined event configurations
    EVENT_CONFIG = {
        "circuit_breaker_triggered": {
            "severity": AlertSeverity.WARNING,
            "category": AlertCategory.RISK,
        },
        "kill_switch_activated": {
            "severity": AlertSeverity.CRITICAL,
            "category": AlertCategory.RISK,
        },
        "drawdown_warning": {
            "severity": AlertSeverity.WARNING,
            "category": AlertCategory.RISK,
        },
        "connection_lost": {
            "severity": AlertSeverity.ERROR,
            "category": AlertCategory.CONNECTION,
        },
        "data_feed_stale": {
            "severity": AlertSeverity.WARNING,
            "category": AlertCategory.CONNECTION,
        },
        "edge_decay_detected": {
            "severity": AlertSeverity.WARNING,
            "category": AlertCategory.PERFORMANCE,
        },
        "reconciliation_mismatch": {
            "severity": AlertSeverity.ERROR,
            "category": AlertCategory.SYSTEM,
        },
        "daily_loss_limit": {
            "severity": AlertSeverity.ERROR,
            "category": AlertCategory.RISK,
        },
        "weekly_loss_limit": {
            "severity": AlertSeverity.CRITICAL,
            "category": AlertCategory.RISK,
        },
        "trade_executed": {
            "severity": AlertSeverity.INFO,
            "category": AlertCategory.TRADING,
        },
        "signal_generated": {
            "severity": AlertSeverity.INFO,
            "category": AlertCategory.TRADING,
        },
    }

    def __init__(self):
        self._handlers: Dict[str, List[Callable[[MonitoringAlert], None]]] = {}
        self._global_handlers: List[Callable[[MonitoringAlert], None]] = []
        self._history: List[MonitoringAlert] = []
        self._max_history = 1000
        self._alert_manager: Any = None  # Reference to delivery.AlertManager for forwarding

    def set_alert_manager(self, alert_manager: Any) -> None:
        """Set reference to delivery AlertManager for forwarding critical alerts."""
        self._alert_manager = alert_manager

    def register_handler(
        self,
        event: str,
        handler: Callable[[MonitoringAlert], None],
    ) -> None:
        """Register a handler for a specific event type."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
        logger.debug("Registered handler for event: %s", event)

    def register_global_handler(
        self,
        handler: Callable[[MonitoringAlert], None],
    ) -> None:
        """Register a handler for all events."""
        self._global_handlers.append(handler)

    def unregister_handler(
        self,
        event: str,
        handler: Callable[[MonitoringAlert], None],
    ) -> bool:
        """Unregister a handler."""
        if event in self._handlers and handler in self._handlers[event]:
            self._handlers[event].remove(handler)
            return True
        return False

    async def raise_alert(
        self,
        event: str,
        payload: Any = None,
        message: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
    ) -> MonitoringAlert:
        """
        Raise an alert to registered handlers.

        Args:
            event: Event type (e.g., "circuit_breaker_triggered")
            payload: Event-specific data
            message: Human-readable message (auto-generated if not provided)
            severity: Override default severity
            category: Override default category
        """
        # Get config or use defaults
        config = self.EVENT_CONFIG.get(event, {})

        alert = MonitoringAlert(
            event_type=event,
            severity=severity or config.get("severity", AlertSeverity.INFO),
            category=category or config.get("category", AlertCategory.SYSTEM),
            message=message or f"Event: {event}",
            details=payload if isinstance(payload, dict) else {"data": payload},
        )

        # Store in history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Call event-specific handlers
        handlers = self._handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Handler failed for %s: %s", event, e)

        # Call global handlers
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Global handler failed for %s: %s", event, e)

        # Forward critical alerts to delivery system
        if alert.severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL):
            await self._forward_to_delivery(alert)

        logger.info(
            "Alert raised: [%s] %s - %s",
            alert.severity.value,
            event,
            alert.message,
        )

        return alert

    async def _forward_to_delivery(self, alert: MonitoringAlert) -> None:
        """Forward critical alerts to Discord/Telegram."""
        if not self._alert_manager:
            return

        try:
            from nexus.core.enums import AlertPriority

            priority = (
                AlertPriority.CRITICAL
                if alert.severity == AlertSeverity.CRITICAL
                else AlertPriority.HIGH
            )
            await self._alert_manager.send_alert(
                message=f"[{alert.category.value.upper()}] {alert.message}",
                priority=priority,
            )
        except Exception as e:
            logger.error("Failed to forward alert to delivery: %s", e)

    def get_unacknowledged(self) -> List[MonitoringAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self._history if not a.acknowledged]

    def get_recent(self, count: int = 50) -> List[MonitoringAlert]:
        """Get recent alerts."""
        return self._history[-count:]

    def get_by_severity(self, severity: AlertSeverity) -> List[MonitoringAlert]:
        """Get alerts by severity."""
        return [a for a in self._history if a.severity == severity]

    def acknowledge(self, timestamp: datetime) -> bool:
        """Acknowledge an alert by timestamp."""
        for alert in self._history:
            if alert.timestamp == timestamp:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all unacknowledged alerts."""
        count = 0
        now = datetime.utcnow()
        for alert in self._history:
            if not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = now
                count += 1
        return count

    def clear_history(self) -> None:
        """Clear alert history."""
        self._history = []


# Module-level instance
_alert_manager = MonitoringAlertManager()


# Convenience functions matching original stub signatures
def register_alert_handler(
    event: str,
    handler: Callable[[Any], None],
) -> None:
    """Register a handler for an alert event."""
    _alert_manager.register_handler(event, handler)


async def raise_alert(event: str, payload: Any) -> None:
    """Raise an alert to registered handlers."""
    await _alert_manager.raise_alert(event, payload)


def get_alert_manager() -> MonitoringAlertManager:
    """Get the monitoring alert manager instance."""
    return _alert_manager
