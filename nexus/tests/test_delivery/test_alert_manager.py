"""Tests for nexus.delivery.alert_manager (AlertManager)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.core.enums import AlertPriority
from nexus.delivery.alert_manager import (
    AlertManager,
    DeliveryRecord,
    DeliveryStatus,
    QueuedAlert,
    create_alert_manager,
)
from nexus.delivery.discord import DeliveryResult


def _make_mock_channel(*, is_configured=True, send_signal_result=None, send_alert_result=None):
    """Create a mock delivery channel with send_signal and send_alert."""
    if send_signal_result is None:
        send_signal_result = DeliveryResult(success=True, response_code=204)
    if send_alert_result is None:
        send_alert_result = DeliveryResult(success=True, response_code=200)
    mock = MagicMock()
    mock.is_configured = is_configured
    mock.send_signal = AsyncMock(return_value=send_signal_result)
    mock.send_alert = AsyncMock(return_value=send_alert_result)
    mock.test_connection = AsyncMock(return_value=DeliveryResult(success=True))
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def manager():
    return AlertManager(max_history=20)


@pytest.fixture
def mock_discord():
    return _make_mock_channel()


@pytest.fixture
def mock_telegram():
    return _make_mock_channel()


@pytest.fixture
def manager_with_two_channels(manager, mock_discord, mock_telegram):
    manager.register_channel("discord", mock_discord)
    manager.register_channel("telegram", mock_telegram)
    return manager


class TestChannelRegistration:
    """Channel registration and unregistration."""

    def test_register_channel_stores_instance(self, manager, mock_discord):
        manager.register_channel("discord", mock_discord)
        assert "discord" in manager._channels
        assert manager._channels["discord"] is mock_discord

    def test_register_channel_without_send_signal_raises(self, manager):
        bad = MagicMock(spec=[])  # no send_signal
        with pytest.raises(ValueError, match="send_signal"):
            manager.register_channel("bad", bad)

    def test_register_channel_without_send_alert_raises(self, manager):
        bad = MagicMock()
        bad.send_signal = lambda x: None
        del bad.send_alert
        with pytest.raises(ValueError, match="send_alert"):
            manager.register_channel("bad", bad)

    def test_unregister_channel_returns_true_when_exists(
        self, manager, mock_discord
    ):
        manager.register_channel("discord", mock_discord)
        assert manager.unregister_channel("discord") is True
        assert "discord" not in manager._channels

    def test_unregister_channel_returns_false_when_missing(self, manager):
        assert manager.unregister_channel("nonexistent") is False


@pytest.mark.asyncio
class TestMultiChannelDelivery:
    """Multi-channel concurrent delivery."""

    async def test_send_signal_calls_all_channels(
        self, manager_with_two_channels, sample_signal, mock_discord, mock_telegram
    ):
        record = await manager_with_two_channels.send_signal(sample_signal)

        mock_discord.send_signal.assert_called_once_with(sample_signal)
        mock_telegram.send_signal.assert_called_once_with(sample_signal)
        assert record.status == DeliveryStatus.SENT
        assert len(record.results) == 2
        assert record.results["discord"].success is True
        assert record.results["telegram"].success is True

    async def test_send_signal_concurrent_both_called(self, manager, sample_signal):
        ch1 = _make_mock_channel()
        ch2 = _make_mock_channel()
        manager.register_channel("ch1", ch1)
        manager.register_channel("ch2", ch2)

        record = await manager.send_signal(sample_signal)

        assert record.status == DeliveryStatus.SENT
        assert len(record.results) == 2
        ch1.send_signal.assert_called_once()
        ch2.send_signal.assert_called_once()


@pytest.mark.asyncio
class TestPartialFailure:
    """Partial failure (one channel fails)."""

    async def test_partial_failure_sets_retrying_and_calls_on_partial(
        self, manager, sample_signal, mock_discord
    ):
        failing = _make_mock_channel(
            send_signal_result=DeliveryResult(
                success=False, error_message="Network error"
            )
        )
        manager.register_channel("discord", mock_discord)
        manager.register_channel("telegram", failing)
        on_partial = MagicMock()
        manager.on_partial(on_partial)

        record = await manager.send_signal(sample_signal)

        assert record.status == DeliveryStatus.RETRYING
        assert record.results["discord"].success is True
        assert record.results["telegram"].success is False
        on_partial.assert_called_once()
        assert on_partial.call_args[0][0].status == DeliveryStatus.RETRYING


@pytest.mark.asyncio
class TestTotalFailure:
    """Total failure (all channels fail)."""

    async def test_all_channels_fail_sets_failed_and_calls_on_failure(
        self, manager, sample_signal
    ):
        fail1 = _make_mock_channel(
            send_signal_result=DeliveryResult(success=False, error_message="E1")
        )
        fail2 = _make_mock_channel(
            send_signal_result=DeliveryResult(success=False, error_message="E2")
        )
        manager.register_channel("a", fail1)
        manager.register_channel("b", fail2)
        on_failure = MagicMock()
        manager.on_failure(on_failure)

        record = await manager.send_signal(sample_signal)

        assert record.status == DeliveryStatus.FAILED
        assert record.results["a"].success is False
        assert record.results["b"].success is False
        on_failure.assert_called_once()
        assert on_failure.call_args[0][0].status == DeliveryStatus.FAILED


@pytest.mark.asyncio
class TestSpecificChannelTargeting:
    """Specific channel targeting."""

    async def test_send_signal_to_specific_channels_only(
        self, manager_with_two_channels, sample_signal, mock_discord, mock_telegram
    ):
        record = await manager_with_two_channels.send_signal(
            sample_signal, channels=["discord"]
        )

        mock_discord.send_signal.assert_called_once_with(sample_signal)
        mock_telegram.send_signal.assert_not_called()
        assert list(record.results.keys()) == ["discord"]
        assert record.status == DeliveryStatus.SENT

    async def test_send_alert_to_specific_channels(
        self, manager_with_two_channels, mock_discord, mock_telegram
    ):
        record = await manager_with_two_channels.send_alert(
            "Hello", priority=AlertPriority.NORMAL, channels=["telegram"]
        )

        mock_telegram.send_alert.assert_called_once()
        mock_discord.send_alert.assert_not_called()
        assert list(record.results.keys()) == ["telegram"]


@pytest.mark.asyncio
class TestNoConfiguredChannels:
    """When no channels are configured or specified."""

    async def test_send_signal_no_channels_returns_failed_record(
        self, manager, sample_signal
    ):
        record = await manager.send_signal(sample_signal)

        assert record.status == DeliveryStatus.FAILED
        assert record.error == "No configured channels"
        assert len(record.results) == 0

    async def test_send_signal_specified_channels_none_configured(
        self, manager, sample_signal, mock_discord
    ):
        manager.register_channel("discord", mock_discord)
        record = await manager.send_signal(
            sample_signal, channels=["telegram"]
        )

        assert record.status == DeliveryStatus.FAILED
        assert len(record.results) == 0


class TestStatisticsTracking:
    """Statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_increment_on_signal_success(
        self, manager_with_two_channels, sample_signal
    ):
        stats_before = manager_with_two_channels.get_stats()
        await manager_with_two_channels.send_signal(sample_signal)
        stats_after = manager_with_two_channels.get_stats()

        assert stats_after["signals_sent"] == stats_before["signals_sent"] + 1
        assert stats_after["successful_deliveries"] == stats_before["successful_deliveries"] + 2

    @pytest.mark.asyncio
    async def test_stats_increment_on_alert(
        self, manager_with_two_channels
    ):
        stats_before = manager_with_two_channels.get_stats()
        await manager_with_two_channels.send_alert(
            "Test", priority=AlertPriority.NORMAL
        )
        stats_after = manager_with_two_channels.get_stats()

        assert stats_after["alerts_sent"] == stats_before["alerts_sent"] + 1

    def test_get_stats_returns_copy(self, manager):
        s1 = manager.get_stats()
        s2 = manager.get_stats()
        assert s1 is not s2
        s1["signals_sent"] = 999
        assert manager.get_stats()["signals_sent"] == 0


class TestDeliveryHistory:
    """Delivery history."""

    @pytest.mark.asyncio
    async def test_send_signal_appends_to_history(
        self, manager_with_two_channels, sample_signal
    ):
        await manager_with_two_channels.send_signal(sample_signal)
        recent = manager_with_two_channels.get_recent_deliveries(limit=5)
        assert len(recent) == 1
        assert recent[0].status == DeliveryStatus.SENT

    @pytest.mark.asyncio
    async def test_get_recent_deliveries_respects_limit(
        self, manager_with_two_channels, sample_signal
    ):
        for _ in range(5):
            await manager_with_two_channels.send_signal(sample_signal)
        recent = manager_with_two_channels.get_recent_deliveries(limit=2)
        assert len(recent) == 2


class TestRetryQueue:
    """Retry queue."""

    def test_queue_for_retry_appends_alert(self, manager):
        manager.queue_for_retry("Retry me", AlertPriority.HIGH, ["discord"])
        assert len(manager._retry_queue) == 1
        assert manager._retry_queue[0].message == "Retry me"
        assert manager._retry_queue[0].priority == AlertPriority.HIGH
        assert manager._retry_queue[0].channels == ["discord"]

    @pytest.mark.asyncio
    async def test_process_retry_queue_sends_and_removes_successful(
        self, manager_with_two_channels, mock_discord, mock_telegram
    ):
        manager_with_two_channels.queue_for_retry(
            "Alert", AlertPriority.NORMAL, ["discord", "telegram"]
        )
        processed = await manager_with_two_channels.process_retry_queue()

        assert processed == 1
        assert len(manager_with_two_channels._retry_queue) == 0
        mock_discord.send_alert.assert_called_once()
        mock_telegram.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_retry_queue_increments_attempts_on_failure(
        self, manager
    ):
        failing = _make_mock_channel(
            send_alert_result=DeliveryResult(success=False, error_message="Fail")
        )
        manager.register_channel("discord", failing)
        manager.queue_for_retry("Fail alert", AlertPriority.LOW, ["discord"])

        processed = await manager.process_retry_queue()

        assert processed == 1
        assert len(manager._retry_queue) == 1
        assert manager._retry_queue[0].attempts == 1
        assert manager._retry_queue[0].last_attempt is not None

    @pytest.mark.asyncio
    async def test_process_retry_queue_empty_returns_zero(self, manager):
        n = await manager.process_retry_queue()
        assert n == 0


@pytest.mark.asyncio
class TestCallbacks:
    """Callbacks success/failure/partial."""

    async def test_on_success_called_when_all_succeed(
        self, manager_with_two_channels, sample_signal
    ):
        on_success = MagicMock()
        manager_with_two_channels.on_success(on_success)
        await manager_with_two_channels.send_signal(sample_signal)
        on_success.assert_called_once()
        assert on_success.call_args[0][0].status == DeliveryStatus.SENT

    async def test_on_failure_called_when_all_fail(
        self, manager, sample_signal
    ):
        fail_ch = _make_mock_channel(
            send_signal_result=DeliveryResult(success=False)
        )
        manager.register_channel("only", fail_ch)
        on_failure = MagicMock()
        manager.on_failure(on_failure)
        await manager.send_signal(sample_signal)
        on_failure.assert_called_once()
        assert on_failure.call_args[0][0].status == DeliveryStatus.FAILED


class TestFactoryFunction:
    """Factory function auto-registration."""

    def test_create_alert_manager_returns_alert_manager(self):
        with patch("nexus.delivery.alert_manager.settings") as mock_settings:
            mock_settings.discord_webhook_url = ""
            mock_settings.discord_enabled = False
            mock_settings.telegram_bot_token = ""
            mock_settings.telegram_chat_id = ""
            mock_settings.telegram_enabled = False
            manager = create_alert_manager()
        assert isinstance(manager, AlertManager)
        assert len(manager._channels) == 0

    def test_create_alert_manager_registers_discord_when_configured(self):
        with patch("nexus.delivery.alert_manager.settings") as mock_settings, patch(
            "nexus.delivery.alert_manager.create_discord_delivery"
        ) as create_discord:
            mock_settings.discord_webhook_url = "https://discord.com/webhook/1"
            mock_settings.discord_enabled = True
            mock_settings.telegram_bot_token = ""
            mock_settings.telegram_chat_id = ""
            mock_settings.telegram_enabled = False
            mock_discord = MagicMock()
            mock_discord.is_configured = True
            create_discord.return_value = mock_discord
            manager = create_alert_manager()
        assert "discord" in manager._channels
        create_discord.assert_called_once()

    def test_create_alert_manager_registers_telegram_when_configured(self):
        with patch("nexus.delivery.alert_manager.settings") as mock_settings, patch(
            "nexus.delivery.alert_manager.create_telegram_delivery"
        ) as create_tg:
            mock_settings.discord_webhook_url = ""
            mock_settings.discord_enabled = False
            mock_settings.telegram_bot_token = "token"
            mock_settings.telegram_chat_id = "chat"
            mock_settings.telegram_enabled = True
            mock_telegram = MagicMock()
            mock_telegram.is_configured = True
            create_tg.return_value = mock_telegram
            manager = create_alert_manager()
        assert "telegram" in manager._channels
        create_tg.assert_called_once()


@pytest.mark.asyncio
class TestTestAllChannels:
    """test_all_channels."""

    async def test_test_all_channels_calls_test_connection(
        self, manager_with_two_channels, mock_discord, mock_telegram
    ):
        results = await manager_with_two_channels.test_all_channels()

        assert "discord" in results
        assert "telegram" in results
        assert results["discord"].success is True
        assert results["telegram"].success is True
        mock_discord.test_connection.assert_called_once()
        mock_telegram.test_connection.assert_called_once()


@pytest.mark.asyncio
class TestClose:
    """close()."""

    async def test_close_calls_channel_close(
        self, manager_with_two_channels, mock_discord, mock_telegram
    ):
        await manager_with_two_channels.close()

        mock_discord.close.assert_called_once()
        mock_telegram.close.assert_called_once()
