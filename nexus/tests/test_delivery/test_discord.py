"""Tests for nexus.delivery.discord (DiscordDelivery)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from nexus.core.enums import AlertPriority
from nexus.delivery.discord import (
    DiscordDelivery,
    DeliveryResult,
    create_discord_delivery,
)


@pytest.fixture
def webhook_url():
    return "https://discord.com/api/webhooks/123/abc"


@pytest.fixture
def discord_with_webhook(webhook_url):
    """DiscordDelivery with webhook set (no real HTTP)."""
    with patch("nexus.delivery.discord.settings") as mock_settings:
        mock_settings.discord_webhook_url = ""
        mock_settings.alert_retry_attempts = 2
        mock_settings.alert_retry_delay_seconds = 0.1
        mock_settings.alert_timeout_seconds = 10.0
        return DiscordDelivery(webhook_url=webhook_url, retry_attempts=2, retry_delay=0.01)


@pytest.fixture
def discord_unconfigured():
    """DiscordDelivery with no webhook."""
    with patch("nexus.delivery.discord.settings") as mock_settings:
        mock_settings.discord_webhook_url = ""
        return DiscordDelivery(webhook_url=None)


class TestConfigurationValidation:
    """Configuration validation with/without webhook."""

    def test_is_configured_true_when_webhook_set(self, discord_with_webhook):
        assert discord_with_webhook.is_configured is True

    def test_is_configured_false_when_no_webhook(self, discord_unconfigured):
        assert discord_unconfigured.is_configured is False

    @pytest.mark.asyncio
    async def test_send_signal_returns_failure_when_not_configured(
        self, discord_unconfigured, sample_signal
    ):
        result = await discord_unconfigured.send_signal(sample_signal)
        assert result.success is False
        assert "not configured" in (result.error_message or "").lower()


@pytest.mark.asyncio
class TestSuccessfulDelivery:
    """Successful delivery (mock HTTP 204)."""

    async def test_send_signal_returns_success_on_204(
        self, discord_with_webhook, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is True
        assert result.response_code == 204
        discord_with_webhook.client.post.assert_called_once()
        call_args = discord_with_webhook.client.post.call_args
        assert call_args[0][0] == discord_with_webhook.webhook_url
        assert "embeds" in call_args[1]["json"]

    async def test_send_alert_returns_success_on_204(
        self, discord_with_webhook
    ):
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        result = await discord_with_webhook.send_alert(
            "Test alert", priority=AlertPriority.NORMAL
        )

        assert result.success is True
        assert result.response_code == 204


@pytest.mark.asyncio
class TestFailureHandling:
    """Failure handling (mock HTTP 400/500)."""

    async def test_send_returns_failure_on_400(
        self, discord_with_webhook, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is False
        assert result.response_code == 400
        assert "Bad Request" in (result.error_message or "")

    async def test_send_returns_failure_on_500(
        self, discord_with_webhook, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is False
        assert result.response_code == 500


@pytest.mark.asyncio
class TestRateLimitHandling:
    """Rate limit handling (429 with retry_after)."""

    async def test_429_retries_then_succeeds(
        self, discord_with_webhook, sample_signal
    ):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.json.return_value = {"retry_after": 0.01}
        mock_204 = MagicMock()
        mock_204.status_code = 204
        discord_with_webhook.client.post = AsyncMock(
            side_effect=[mock_429, mock_204]
        )

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is True
        assert discord_with_webhook.client.post.call_count == 2


@pytest.mark.asyncio
class TestTimeoutAndConnectionError:
    """Timeout and connection error with exponential backoff."""

    async def test_timeout_retries_then_fails(
        self, discord_with_webhook, sample_signal
    ):
        discord_with_webhook.client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is False
        assert "Timeout" in (result.error_message or "")
        assert discord_with_webhook.client.post.call_count == 2

    async def test_connection_error_retries_then_fails(
        self, discord_with_webhook, sample_signal
    ):
        discord_with_webhook.client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection failed")
        )

        result = await discord_with_webhook.send_signal(sample_signal)

        assert result.success is False
        assert "connection" in (result.error_message or "").lower()


@pytest.mark.asyncio
class TestCallbacks:
    """Callbacks on_success, on_failure."""

    async def test_on_success_called_on_success(
        self, discord_with_webhook, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)
        callback = MagicMock()
        discord_with_webhook.on_success(callback)

        await discord_with_webhook.send_signal(sample_signal)

        callback.assert_called_once()
        assert callback.call_args[0][0].success is True

    async def test_on_failure_called_on_failure(
        self, discord_with_webhook, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Error"
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)
        callback = MagicMock()
        discord_with_webhook.on_failure(callback)

        await discord_with_webhook.send_signal(sample_signal)

        callback.assert_called_once()
        assert callback.call_args[0][0].success is False


@pytest.mark.asyncio
class TestCustomPayload:
    """Custom payload send."""

    async def test_send_custom_payload_success(self, discord_with_webhook):
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        payload = {"content": "Custom message"}
        result = await discord_with_webhook.send_custom(payload)

        assert result.success is True
        discord_with_webhook.client.post.assert_called_once_with(
            discord_with_webhook.webhook_url, json=payload
        )


@pytest.mark.asyncio
class TestTestConnection:
    """Test test_connection method."""

    async def test_test_connection_sends_test_message(
        self, discord_with_webhook
    ):
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord_with_webhook.client.post = AsyncMock(return_value=mock_response)

        result = await discord_with_webhook.test_connection()

        assert result.success is True
        call_args = discord_with_webhook.client.post.call_args
        assert "content" in call_args[1]["json"]
        assert "NEXUS" in call_args[1]["json"]["content"]
        assert "test" in call_args[1]["json"]["content"].lower()
