"""Tests for nexus.delivery.telegram (TelegramDelivery)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from nexus.core.enums import AlertPriority
from nexus.delivery.telegram import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TelegramDelivery,
    create_telegram_delivery,
)


@pytest.fixture
def bot_token():
    return "123456:ABC-DEF"


@pytest.fixture
def chat_id():
    return "987654321"


@pytest.fixture
def telegram_configured(bot_token, chat_id):
    """TelegramDelivery with token and chat_id (no real HTTP)."""
    with patch("nexus.delivery.telegram.settings") as mock_settings:
        mock_settings.telegram_bot_token = ""
        mock_settings.telegram_chat_id = ""
        mock_settings.alert_retry_attempts = 2
        mock_settings.alert_retry_delay_seconds = 0.1
        mock_settings.alert_timeout_seconds = 10.0
        return TelegramDelivery(
            bot_token=bot_token,
            chat_id=chat_id,
            retry_attempts=2,
            retry_delay=0.01,
        )


@pytest.fixture
def telegram_no_token(chat_id):
    with patch("nexus.delivery.telegram.settings") as mock_settings:
        mock_settings.telegram_bot_token = ""
        mock_settings.telegram_chat_id = chat_id
        return TelegramDelivery(bot_token=None, chat_id=chat_id)


@pytest.fixture
def telegram_no_chat_id(bot_token):
    with patch("nexus.delivery.telegram.settings") as mock_settings:
        mock_settings.telegram_bot_token = bot_token
        mock_settings.telegram_chat_id = ""
        return TelegramDelivery(bot_token=bot_token, chat_id=None)


class TestConfiguration:
    """Configuration: needs both token AND chat_id."""

    def test_is_configured_true_when_both_set(self, telegram_configured):
        assert telegram_configured.is_configured is True

    def test_is_configured_false_when_no_token(self, telegram_no_token):
        assert telegram_no_token.is_configured is False

    def test_is_configured_false_when_no_chat_id(self, telegram_no_chat_id):
        assert telegram_no_chat_id.is_configured is False


class TestAPIUrlConstruction:
    """API URL construction."""

    def test_api_url_contains_token_and_base(self, telegram_configured, bot_token):
        assert telegram_configured.api_url is not None
        assert "api.telegram.org" in telegram_configured.api_url
        assert f"bot{bot_token}" in telegram_configured.api_url

    def test_api_url_none_when_no_token(self, telegram_no_token):
        assert telegram_no_token.api_url is None


@pytest.mark.asyncio
class TestSuccessfulDelivery:
    """Successful delivery (mock 200 with message_id)."""

    async def test_send_signal_success(
        self, telegram_configured, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {"message_id": 123},
        }
        mock_response.text = ""
        telegram_configured.client.post = AsyncMock(return_value=mock_response)

        result = await telegram_configured.send_signal(sample_signal)

        assert result.success is True
        assert result.response_code == 200
        call_args = telegram_configured.client.post.call_args
        assert "sendMessage" in call_args[0][0]
        assert call_args[1]["json"]["chat_id"] == telegram_configured.chat_id
        assert "text" in call_args[1]["json"]
        assert call_args[1]["json"].get("parse_mode") == "Markdown"

    async def test_send_alert_success(self, telegram_configured):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {}}
        mock_response.text = ""
        telegram_configured.client.post = AsyncMock(return_value=mock_response)

        result = await telegram_configured.send_alert(
            "Alert text", priority=AlertPriority.HIGH
        )

        assert result.success is True
        assert result.response_code == 200


@pytest.mark.asyncio
class TestFailureHandling:
    """Failure handling."""

    async def test_send_returns_failure_on_500(
        self, telegram_configured, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        telegram_configured.client.post = AsyncMock(return_value=mock_response)

        result = await telegram_configured.send_signal(sample_signal)

        assert result.success is False
        assert result.response_code == 500


@pytest.mark.asyncio
class TestRateLimit:
    """Rate limit (429 with parameters.retry_after)."""

    async def test_429_retries_with_retry_after(
        self, telegram_configured, sample_signal
    ):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.json.return_value = {
            "ok": False,
            "parameters": {"retry_after": 0.01},
        }
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"ok": True, "result": {}}
        mock_200.text = ""
        telegram_configured.client.post = AsyncMock(
            side_effect=[mock_429, mock_200]
        )

        result = await telegram_configured.send_signal(sample_signal)

        assert result.success is True
        assert telegram_configured.client.post.call_count == 2


@pytest.mark.asyncio
class TestParseErrorFallback:
    """Parse error fallback: retries without Markdown."""

    async def test_400_parse_entities_retries_without_markdown(
        self, telegram_configured, sample_signal
    ):
        mock_400 = MagicMock()
        mock_400.status_code = 400
        mock_400.text = "can't parse entities: unsupported entity"
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"ok": True, "result": {}}
        mock_200.text = ""
        telegram_configured.client.post = AsyncMock(
            side_effect=[mock_400, mock_200]
        )

        result = await telegram_configured.send_signal(sample_signal)

        assert result.success is True
        assert telegram_configured.client.post.call_count == 2
        # Second call should be without parse_mode
        second_call_json = telegram_configured.client.post.call_args_list[1][1][
            "json"
        ]
        assert "parse_mode" not in second_call_json or second_call_json.get("parse_mode") is None


@pytest.mark.asyncio
class TestMessageTruncation:
    """Message truncation (4096 char limit)."""

    async def test_long_message_truncated(
        self, telegram_configured
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {}}
        mock_response.text = ""
        telegram_configured.client.post = AsyncMock(return_value=mock_response)

        long_text = "x" * (TELEGRAM_MAX_MESSAGE_LENGTH + 100)
        result = await telegram_configured.send_plain(long_text)

        assert result.success is True
        call_json = telegram_configured.client.post.call_args[1]["json"]
        assert len(call_json["text"]) <= TELEGRAM_MAX_MESSAGE_LENGTH
        assert call_json["text"].endswith("...")


@pytest.mark.asyncio
class TestTimeoutHandling:
    """Timeout handling."""

    async def test_timeout_retries_then_fails(
        self, telegram_configured, sample_signal
    ):
        telegram_configured.client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        result = await telegram_configured.send_signal(sample_signal)

        assert result.success is False
        assert "Timeout" in (result.error_message or "")


@pytest.mark.asyncio
class TestGetMe:
    """get_me method."""

    async def test_get_me_returns_bot_info(self, telegram_configured):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {"id": 123, "is_bot": True, "username": "nexus_bot"},
        }
        telegram_configured.client.get = AsyncMock(return_value=mock_response)

        info = await telegram_configured.get_me()

        assert info == {"id": 123, "is_bot": True, "username": "nexus_bot"}
        telegram_configured.client.get.assert_called_once()
        assert "getMe" in telegram_configured.client.get.call_args[0][0]

    async def test_get_me_returns_empty_when_not_configured(
        self, telegram_no_token
    ):
        info = await telegram_no_token.get_me()
        assert info == {}


@pytest.mark.asyncio
class TestCallbacks:
    """Callbacks on_success, on_failure."""

    async def test_on_success_called_on_success(
        self, telegram_configured, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {}}
        mock_response.text = ""
        telegram_configured.client.post = AsyncMock(return_value=mock_response)
        callback = MagicMock()
        telegram_configured.on_success(callback)

        await telegram_configured.send_signal(sample_signal)

        callback.assert_called_once()
        assert callback.call_args[0][0].success is True

    async def test_on_failure_called_on_failure(
        self, telegram_configured, sample_signal
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Error"
        telegram_configured.client.post = AsyncMock(return_value=mock_response)
        callback = MagicMock()
        telegram_configured.on_failure(callback)

        await telegram_configured.send_signal(sample_signal)

        callback.assert_called_once()
        assert callback.call_args[0][0].success is False


@pytest.mark.asyncio
class TestTestConnection:
    """Test connection method."""

    async def test_test_connection_sends_plain_message(
        self, telegram_configured
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {}}
        mock_response.text = ""
        telegram_configured.client.post = AsyncMock(return_value=mock_response)

        result = await telegram_configured.test_connection()

        assert result.success is True
        call_json = telegram_configured.client.post.call_args[1]["json"]
        assert "NEXUS" in call_json["text"]
        assert "test" in call_json["text"].lower()
        assert "parse_mode" not in call_json or call_json.get("parse_mode") is None
