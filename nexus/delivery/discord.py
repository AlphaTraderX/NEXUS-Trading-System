"""
NEXUS Discord delivery - send signals and alerts via webhooks.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx

from nexus.config.settings import settings
from nexus.core.enums import AlertPriority
from nexus.core.models import NexusSignal
from nexus.delivery.formatter import formatter

logger = logging.getLogger(__name__)


@dataclass
class DeliveryResult:
    """Result of a delivery attempt."""

    success: bool
    response_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DiscordDelivery:
    """Send signals and alerts to Discord via webhooks."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 30.0,
    ):
        self.webhook_url = webhook_url or settings.discord_webhook_url
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.formatter = formatter
        self._on_success: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None

    @property
    def is_configured(self) -> bool:
        """Return True if webhook URL is configured."""
        return bool(self.webhook_url)

    async def send_signal(self, signal: NexusSignal) -> DeliveryResult:
        """Send a signal to Discord."""
        if not self.is_configured:
            logger.warning("Discord not configured: missing webhook URL")
            return DeliveryResult(
                success=False,
                error_message="Discord not configured (missing webhook URL)",
            )

        payload = self.formatter.format_for_discord(signal)
        result = await self._send_with_retry(payload)

        if result.success:
            logger.info("Signal sent to Discord successfully: %s", signal.symbol)
            if self._on_success:
                self._on_success(result)
        else:
            logger.error("Failed to send signal to Discord: %s", result.error_message)
            if self._on_failure:
                self._on_failure(result)

        return result

    async def send_alert(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
    ) -> DeliveryResult:
        """Send a general alert to Discord."""
        payload = self.formatter.format_alert(message, priority)
        return await self._send_with_retry(payload)

    async def send_custom(self, payload: dict) -> DeliveryResult:
        """Send an arbitrary Discord payload."""
        return await self._send_with_retry(payload)

    async def _send_with_retry(self, payload: dict) -> DeliveryResult:
        """Send payload with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(
                    "Sending to Discord (attempt %d/%d)",
                    attempt + 1,
                    self.retry_attempts,
                )
                response = await self.client.post(
                    self.webhook_url,
                    json=payload,
                )

                if response.status_code == 204:
                    logger.debug("Discord delivery succeeded (204)")
                    return DeliveryResult(
                        success=True,
                        response_code=204,
                        retry_count=attempt,
                    )

                if response.status_code == 429:
                    retry_after = response.json().get(
                        "retry_after", self.retry_delay
                    )
                    logger.warning(
                        "Discord rate limited (429), retrying after %.1fs",
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                logger.error(
                    "Discord request failed: %d %s",
                    response.status_code,
                    response.text[:200],
                )
                return DeliveryResult(
                    success=False,
                    response_code=response.status_code,
                    error_message=response.text,
                    retry_count=attempt,
                )

            except httpx.TimeoutException:
                logger.warning(
                    "Discord request timeout (attempt %d/%d)",
                    attempt + 1,
                    self.retry_attempts,
                )
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                return DeliveryResult(
                    success=False,
                    error_message="Timeout",
                    retry_count=attempt,
                )

            except httpx.ConnectError as e:
                logger.warning(
                    "Discord connection error (attempt %d/%d): %s",
                    attempt + 1,
                    self.retry_attempts,
                    e,
                )
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                return DeliveryResult(
                    success=False,
                    error_message=str(e),
                    retry_count=attempt,
                )

        return DeliveryResult(
            success=False,
            error_message="Max retries exceeded",
            retry_count=self.retry_attempts - 1,
        )

    def on_success(self, callback: Callable) -> None:
        """Register callback for successful deliveries."""
        self._on_success = callback

    def on_failure(self, callback: Callable) -> None:
        """Register callback for failed deliveries."""
        self._on_failure = callback

    async def test_connection(self) -> DeliveryResult:
        """Send a test message to verify Discord connection."""
        payload = {"content": "🔌 NEXUS Discord connection test"}
        return await self._send_with_retry(payload)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


def create_discord_delivery(
    webhook_url: Optional[str] = None,
) -> DiscordDelivery:
    """Create a DiscordDelivery instance with settings from config."""
    return DiscordDelivery(
        webhook_url=webhook_url,
        retry_attempts=settings.alert_retry_attempts,
        retry_delay=settings.alert_retry_delay_seconds,
        timeout=settings.alert_timeout_seconds,
    )
