"""
NEXUS Alert Manager - central delivery coordinator.

Routes signals and alerts to all configured channels (Discord, Telegram, etc.).
"""

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus.config.settings import settings
from nexus.core.enums import AlertPriority
from nexus.core.models import NexusSignal
from nexus.delivery.discord import DeliveryResult, create_discord_delivery
from nexus.delivery.telegram import create_telegram_delivery

logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Status of a multi-channel delivery."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DeliveryRecord:
    """Record of a delivery attempt across one or more channels."""

    id: str
    signal_id: Optional[str]
    channels: List[str]
    status: DeliveryStatus
    results: Dict[str, DeliveryResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


@dataclass
class QueuedAlert:
    """Alert queued for retry."""

    message: str
    priority: AlertPriority
    channels: List[str]
    attempts: int = 0
    last_attempt: Optional[datetime] = None


class AlertManager:
    """Central coordinator: route signals and alerts to all configured channels."""

    def __init__(self, max_history: int = 100):
        self._channels: Dict[str, Any] = {}
        self._delivery_history: deque = deque(maxlen=max_history)
        self._retry_queue: List[QueuedAlert] = []
        self._stats = {
            "signals_sent": 0,
            "alerts_sent": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
        }
        self._on_success: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None
        self._on_partial: Optional[Callable] = None

    def register_channel(self, name: str, delivery_instance: Any) -> None:
        """Register a delivery channel. Instance must have send_signal and send_alert."""
        if not hasattr(delivery_instance, "send_signal") or not callable(
            getattr(delivery_instance, "send_signal")
        ):
            raise ValueError(
                f"Channel '{name}' must have a send_signal method"
            )
        if not hasattr(delivery_instance, "send_alert") or not callable(
            getattr(delivery_instance, "send_alert")
        ):
            raise ValueError(
                f"Channel '{name}' must have a send_alert method"
            )
        self._channels[name] = delivery_instance
        logger.info("Registered delivery channel: %s", name)

    def unregister_channel(self, name: str) -> bool:
        """Remove a channel. Returns True if it existed."""
        if name in self._channels:
            del self._channels[name]
            logger.info("Unregistered delivery channel: %s", name)
            return True
        return False

    def _active_channels(
        self, channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Return registered channels that are configured, optionally filtered by name."""
        if channels is not None:
            subset = {
                k: v
                for k, v in self._channels.items()
                if k in channels and getattr(v, "is_configured", True)
            }
        else:
            subset = {
                k: v
                for k, v in self._channels.items()
                if getattr(v, "is_configured", True)
            }
        return subset

    async def send_signal(
        self,
        signal: NexusSignal,
        channels: Optional[List[str]] = None,
    ) -> DeliveryRecord:
        """Send signal to specified channels (or all configured)."""
        active = self._active_channels(channels)
        channel_list = list(active.keys())

        record = DeliveryRecord(
            id=uuid.uuid4().hex,
            signal_id=getattr(signal, "signal_id", None),
            channels=channel_list,
            status=DeliveryStatus.PENDING,
            results={},
        )

        if not active:
            record.status = DeliveryStatus.FAILED
            record.error = "No configured channels"
            logger.warning("send_signal: no configured channels")
            return record

        tasks = [
            self._deliver_signal(name, delivery, signal)
            for name, delivery in active.items()
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(active.items()):
            if i < len(results_list):
                r = results_list[i]
                if isinstance(r, Exception):
                    record.results[name] = DeliveryResult(
                        success=False,
                        error_message=str(r),
                    )
                    self._stats["failed_deliveries"] += 1
                else:
                    _, result = r
                    record.results[name] = result
                    if result.success:
                        self._stats["successful_deliveries"] += 1
                    else:
                        self._stats["failed_deliveries"] += 1
            else:
                record.results[name] = DeliveryResult(
                    success=False,
                    error_message="Missing result",
                )
                self._stats["failed_deliveries"] += 1

        successes = sum(1 for r in record.results.values() if r.success)
        total = len(record.results)
        if successes == total:
            record.status = DeliveryStatus.SENT
            self._stats["signals_sent"] += 1
            if self._on_success:
                self._on_success(record)
        elif successes == 0:
            record.status = DeliveryStatus.FAILED
            record.error = "All channels failed"
            if self._on_failure:
                self._on_failure(record)
        else:
            record.status = DeliveryStatus.RETRYING
            record.error = f"{total - successes}/{total} channels failed"
            if self._on_partial:
                self._on_partial(record)

        self._delivery_history.append(record)
        return record

    async def _deliver_signal(
        self,
        channel_name: str,
        delivery: Any,
        signal: NexusSignal,
    ) -> tuple:
        """Deliver signal to one channel. Returns (channel_name, DeliveryResult)."""
        try:
            result = await delivery.send_signal(signal)
            return (channel_name, result)
        except Exception as e:
            logger.exception("Delivery failed for channel %s: %s", channel_name, e)
            return (
                channel_name,
                DeliveryResult(success=False, error_message=str(e)),
            )

    async def send_alert(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
        channels: Optional[List[str]] = None,
    ) -> DeliveryRecord:
        """Send alert to specified channels (or all configured)."""
        active = self._active_channels(channels)
        channel_list = list(active.keys())

        record = DeliveryRecord(
            id=uuid.uuid4().hex,
            signal_id=None,
            channels=channel_list,
            status=DeliveryStatus.PENDING,
            results={},
        )

        if not active:
            record.status = DeliveryStatus.FAILED
            record.error = "No configured channels"
            logger.warning("send_alert: no configured channels")
            return record

        tasks = [
            self._deliver_alert(name, delivery, message, priority)
            for name, delivery in active.items()
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(active.items()):
            if i < len(results_list):
                r = results_list[i]
                if isinstance(r, Exception):
                    record.results[name] = DeliveryResult(
                        success=False,
                        error_message=str(r),
                    )
                    self._stats["failed_deliveries"] += 1
                else:
                    _, result = r
                    record.results[name] = result
                    if result.success:
                        self._stats["successful_deliveries"] += 1
                    else:
                        self._stats["failed_deliveries"] += 1
            else:
                record.results[name] = DeliveryResult(
                    success=False,
                    error_message="Missing result",
                )
                self._stats["failed_deliveries"] += 1

        successes = sum(1 for r in record.results.values() if r.success)
        total = len(record.results)
        if successes == total:
            record.status = DeliveryStatus.SENT
            self._stats["alerts_sent"] += 1
            if self._on_success:
                self._on_success(record)
        elif successes == 0:
            record.status = DeliveryStatus.FAILED
            record.error = "All channels failed"
            if self._on_failure:
                self._on_failure(record)
        else:
            record.status = DeliveryStatus.RETRYING
            record.error = f"{total - successes}/{total} channels failed"
            if self._on_partial:
                self._on_partial(record)

        self._delivery_history.append(record)
        return record

    async def _deliver_alert(
        self,
        channel_name: str,
        delivery: Any,
        message: str,
        priority: AlertPriority,
    ) -> tuple:
        """Deliver alert to one channel. Returns (channel_name, DeliveryResult)."""
        try:
            result = await delivery.send_alert(message, priority)
            return (channel_name, result)
        except Exception as e:
            logger.exception("Alert delivery failed for channel %s: %s", channel_name, e)
            return (
                channel_name,
                DeliveryResult(success=False, error_message=str(e)),
            )

    def queue_for_retry(
        self,
        message: str,
        priority: AlertPriority,
        channels: List[str],
    ) -> None:
        """Add an alert to the retry queue."""
        self._retry_queue.append(
            QueuedAlert(message=message, priority=priority, channels=channels)
        )
        logger.debug("Queued alert for retry (%d in queue)", len(self._retry_queue))

    async def process_retry_queue(self) -> int:
        """Process queued alerts. Remove successful, increment attempts on failures. Return count processed."""
        if not self._retry_queue:
            return 0

        processed = 0
        still_queued: List[QueuedAlert] = []

        for item in self._retry_queue:
            processed += 1
            record = await self.send_alert(
                item.message,
                item.priority,
                channels=item.channels,
            )
            successes = sum(1 for r in record.results.values() if r.success)
            total = len(record.results)

            if successes == total:
                logger.debug("Retry succeeded for queued alert, removing")
            else:
                item.attempts += 1
                item.last_attempt = datetime.now(timezone.utc)
                still_queued.append(item)
                logger.debug(
                    "Retry partial/failed for queued alert, attempts=%d",
                    item.attempts,
                )

        self._retry_queue = still_queued
        return processed

    def get_stats(self) -> dict:
        """Return a copy of delivery statistics."""
        return self._stats.copy()

    def get_recent_deliveries(self, limit: int = 10) -> List[DeliveryRecord]:
        """Return the most recent delivery records."""
        return list(self._delivery_history)[-limit:]

    async def test_all_channels(self) -> Dict[str, DeliveryResult]:
        """Test each registered channel. Returns channel name -> result."""
        results: Dict[str, DeliveryResult] = {}
        for name, delivery in self._channels.items():
            test_method = getattr(delivery, "test_connection", None)
            if test_method is None or not callable(test_method):
                results[name] = DeliveryResult(
                    success=False,
                    error_message="No test_connection method",
                )
                continue
            try:
                result = await test_method()
                results[name] = result
            except Exception as e:
                logger.exception("test_connection failed for %s: %s", name, e)
                results[name] = DeliveryResult(
                    success=False,
                    error_message=str(e),
                )
        return results

    async def close(self) -> None:
        """Close all channel connections."""
        for name, delivery in self._channels.items():
            close_method = getattr(delivery, "close", None)
            if close_method is not None and callable(close_method):
                try:
                    await close_method()
                    logger.debug("Closed channel: %s", name)
                except Exception as e:
                    logger.exception("Error closing channel %s: %s", name, e)

    def on_success(self, callback: Callable) -> None:
        """Register callback for full success (all channels)."""
        self._on_success = callback

    def on_failure(self, callback: Callable) -> None:
        """Register callback for full failure (no channels)."""
        self._on_failure = callback

    def on_partial(self, callback: Callable) -> None:
        """Register callback for partial delivery (some channels failed)."""
        self._on_partial = callback


def create_alert_manager() -> AlertManager:
    """Create AlertManager with auto-registered channels from settings."""
    manager = AlertManager()

    if settings.discord_webhook_url and settings.discord_enabled:
        discord = create_discord_delivery()
        manager.register_channel("discord", discord)

    if (
        settings.telegram_bot_token
        and settings.telegram_chat_id
        and settings.telegram_enabled
    ):
        telegram = create_telegram_delivery()
        manager.register_channel("telegram", telegram)

    return manager
