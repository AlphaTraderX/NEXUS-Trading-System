"""
NEXUS Telegram delivery - send signals and alerts via Bot API.
"""

import asyncio
import logging
from typing import Callable, Optional

import httpx

from nexus.config.settings import settings
from nexus.core.enums import AlertPriority
from nexus.core.models import NexusSignal
from nexus.delivery.discord import DeliveryResult
from nexus.delivery.formatter import formatter

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramDelivery:
    """Send signals and alerts to Telegram via Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 30.0,
    ):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.api_url = (
            f"https://api.telegram.org/bot{self.bot_token}"
            if self.bot_token
            else None
        )
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.formatter = formatter
        self._on_success: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None

    @property
    def is_configured(self) -> bool:
        """Return True if bot token and chat_id are configured."""
        return bool(self.bot_token and self.chat_id)

    async def send_signal(self, signal: NexusSignal) -> DeliveryResult:
        """Send a signal to Telegram."""
        if not self.is_configured:
            logger.warning("Telegram not configured: missing bot token or chat_id")
            return DeliveryResult(
                success=False,
                error_message="Telegram not configured (missing bot token or chat_id)",
            )

        text = self.formatter.format_for_telegram(signal)
        result = await self._send_message(text, parse_mode="Markdown")

        if result.success:
            logger.info("Signal sent to Telegram successfully: %s", signal.symbol)
            if self._on_success:
                self._on_success(result)
        else:
            logger.error("Failed to send signal to Telegram: %s", result.error_message)
            if self._on_failure:
                self._on_failure(result)

        return result

    async def send_alert(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
    ) -> DeliveryResult:
        """Send a general alert to Telegram."""
        text = self.formatter.format_alert_telegram(message, priority)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_plain(self, text: str) -> DeliveryResult:
        """Send raw text without Markdown parsing."""
        return await self._send_message(text, parse_mode=None)

    async def _send_message(
        self,
        text: str,
        parse_mode: Optional[str] = "Markdown",
    ) -> DeliveryResult:
        """Send a message with retry logic and Telegram-specific handling."""
        if not self.is_configured:
            return DeliveryResult(
                success=False,
                error_message="Telegram not configured",
            )

        if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
            text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 3] + "..."
            logger.debug("Message truncated to %d chars", TELEGRAM_MAX_MESSAGE_LENGTH)

        for attempt in range(self.retry_attempts):
            try:
                payload: dict = {
                    "chat_id": self.chat_id,
                    "text": text,
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode

                logger.debug(
                    "Sending to Telegram (attempt %d/%d)",
                    attempt + 1,
                    self.retry_attempts,
                )
                response = await self.client.post(
                    f"{self.api_url}/sendMessage",
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        logger.debug("Telegram delivery succeeded")
                        return DeliveryResult(
                            success=True,
                            response_code=200,
                            retry_count=attempt,
                        )
                    # 200 but ok=false (shouldn't happen often)
                    return DeliveryResult(
                        success=False,
                        response_code=200,
                        error_message=str(data),
                        retry_count=attempt,
                    )

                if response.status_code == 429:
                    try:
                        data = response.json()
                        params = data.get("parameters", {})
                        retry_after = params.get(
                            "retry_after", self.retry_delay
                        )
                    except Exception:
                        retry_after = self.retry_delay
                    logger.warning(
                        "Telegram rate limited (429), retrying after %.1fs",
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code == 400:
                    body = response.text or ""
                    if (
                        "can't parse entities" in body
                        and parse_mode is not None
                    ):
                        logger.warning(
                            "Telegram parse error, retrying without Markdown"
                        )
                        payload_plain = {
                            "chat_id": self.chat_id,
                            "text": text,
                        }
                        response2 = await self.client.post(
                            f"{self.api_url}/sendMessage",
                            json=payload_plain,
                        )
                        if (
                            response2.status_code == 200
                            and response2.json().get("ok")
                        ):
                            return DeliveryResult(
                                success=True,
                                response_code=200,
                                retry_count=attempt,
                            )
                        return DeliveryResult(
                            success=False,
                            response_code=response2.status_code,
                            error_message=response2.text or "",
                            retry_count=attempt,
                        )
                    return DeliveryResult(
                        success=False,
                        response_code=400,
                        error_message=body,
                        retry_count=attempt,
                    )

                logger.error(
                    "Telegram request failed: %d %s",
                    response.status_code,
                    (response.text or "")[:200],
                )
                return DeliveryResult(
                    success=False,
                    response_code=response.status_code,
                    error_message=response.text,
                    retry_count=attempt,
                )

            except httpx.TimeoutException:
                logger.warning(
                    "Telegram request timeout (attempt %d/%d)",
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
                    "Telegram connection error (attempt %d/%d): %s",
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

    async def get_me(self) -> dict:
        """GET getMe - returns bot info for testing."""
        if not self.is_configured:
            return {}
        try:
            response = await self.client.get(f"{self.api_url}/getMe")
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return data.get("result", {})
            return {}
        except Exception as e:
            logger.exception("get_me failed: %s", e)
            return {}

    async def test_connection(self) -> DeliveryResult:
        """Send a test message to verify Telegram connection."""
        return await self._send_message(
            "🔌 NEXUS Telegram connection test",
            parse_mode=None,
        )

    def on_success(self, callback: Callable) -> None:
        """Register callback for successful deliveries."""
        self._on_success = callback

    def on_failure(self, callback: Callable) -> None:
        """Register callback for failed deliveries."""
        self._on_failure = callback

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


def create_telegram_delivery(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> TelegramDelivery:
    """Create a TelegramDelivery instance with settings from config."""
    return TelegramDelivery(
        bot_token=bot_token,
        chat_id=chat_id,
        retry_attempts=settings.alert_retry_attempts,
        retry_delay=settings.alert_retry_delay_seconds,
        timeout=settings.alert_timeout_seconds,
    )
