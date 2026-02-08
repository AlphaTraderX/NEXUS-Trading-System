"""
NEXUS Kill Switch tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.enums import KillSwitchAction, KillSwitchTrigger
from nexus.core.exceptions import KillSwitchError
from nexus.core.models import KillSwitchState, SystemHealth
from nexus.risk.kill_switch import KillSwitch

UTC = timezone.utc


def _healthy_health(
    seconds_since_heartbeat: float = 5.0,
    seconds_since_data: float = 2.0,
    drawdown_pct: float = -2.0,
    is_connected: bool = True,
    active_errors: list = None,
) -> SystemHealth:
    now = datetime.now(UTC)
    return SystemHealth(
        last_heartbeat=now - timedelta(seconds=seconds_since_heartbeat),
        last_data_update=now - timedelta(seconds=seconds_since_data),
        seconds_since_heartbeat=seconds_since_heartbeat,
        seconds_since_data=seconds_since_data,
        drawdown_pct=drawdown_pct,
        is_connected=is_connected,
        active_errors=active_errors or [],
    )


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.kill_switch_enabled = True
    s.connection_timeout_seconds = 300
    s.data_stale_threshold_seconds = 30
    s.kill_switch_cooldown_minutes = 60
    s.max_drawdown = -10.0
    return s


@pytest.fixture
def kill_switch(mock_settings):
    return KillSwitch(settings=mock_settings)


def test_initial_state(kill_switch):
    """Not triggered on creation."""
    assert kill_switch.is_triggered is False
    assert kill_switch.is_trading_allowed is True
    state = kill_switch.get_state()
    assert state.is_triggered is False
    assert state.trigger == KillSwitchTrigger.NONE
    assert state.action_taken == KillSwitchAction.NONE
    assert state.triggered_at is None
    assert state.can_reset is True
    assert state.cooldown_remaining_seconds == 0


def test_check_conditions_healthy(kill_switch):
    """No trigger when healthy."""
    health = _healthy_health()
    state = kill_switch.check_conditions(health)
    assert state.is_triggered is False
    assert state.trigger == KillSwitchTrigger.NONE
    assert kill_switch.is_trading_allowed is True


def test_max_drawdown_trigger(kill_switch):
    """Triggers on max drawdown."""
    health = _healthy_health(drawdown_pct=-12.0)
    state = kill_switch.check_conditions(health)
    assert state.is_triggered is True
    assert state.trigger == KillSwitchTrigger.MAX_DRAWDOWN
    assert state.action_taken == KillSwitchAction.FULL_SHUTDOWN
    assert "drawdown" in state.message.lower()
    assert "12" in state.message or "-12" in state.message
    assert kill_switch.is_trading_allowed is False


def test_connection_loss_trigger(kill_switch):
    """Triggers on connection timeout."""
    health = _healthy_health(seconds_since_heartbeat=400.0)
    state = kill_switch.check_conditions(health)
    assert state.is_triggered is True
    assert state.trigger == KillSwitchTrigger.CONNECTION_LOSS
    assert state.action_taken == KillSwitchAction.CANCEL_ALL_ORDERS
    assert "connection" in state.message.lower() or "heartbeat" in state.message.lower()


def test_stale_data_trigger(kill_switch):
    """Triggers on stale data."""
    health = _healthy_health(seconds_since_data=60.0)
    state = kill_switch.check_conditions(health)
    assert state.is_triggered is True
    assert state.trigger == KillSwitchTrigger.STALE_DATA
    assert state.action_taken == KillSwitchAction.DISABLE_NEW_TRADES
    assert "stale" in state.message.lower() or "data" in state.message.lower()


def test_manual_trigger(kill_switch):
    """Manual trigger works."""
    state = kill_switch.trigger(
        KillSwitchTrigger.MANUAL,
        "Manual shutdown requested by operator",
    )
    assert state.is_triggered is True
    assert state.trigger == KillSwitchTrigger.MANUAL
    assert state.action_taken == KillSwitchAction.FULL_SHUTDOWN
    assert "Manual" in state.message
    assert kill_switch.is_triggered is True
    assert kill_switch.is_trading_allowed is False


def test_trigger_actions_full_shutdown(kill_switch):
    """Returns all actions for full shutdown."""
    kill_switch.trigger(
        KillSwitchTrigger.MANUAL,
        "Full shutdown",
        action=KillSwitchAction.FULL_SHUTDOWN,
    )
    actions = kill_switch.get_actions_to_execute()
    assert KillSwitchAction.CANCEL_ALL_ORDERS in actions
    assert KillSwitchAction.CLOSE_ALL_POSITIONS in actions
    assert KillSwitchAction.DISABLE_NEW_TRADES in actions
    assert len(actions) == 3


def test_trigger_actions_partial(kill_switch):
    """Returns specific action for partial triggers."""
    health = _healthy_health(seconds_since_data=60.0)
    kill_switch.check_conditions(health)
    actions = kill_switch.get_actions_to_execute()
    assert actions == [KillSwitchAction.DISABLE_NEW_TRADES]

    kill_switch.reset(force=True)
    health = _healthy_health(seconds_since_heartbeat=400.0)
    kill_switch.check_conditions(health)
    actions = kill_switch.get_actions_to_execute()
    assert actions == [KillSwitchAction.CANCEL_ALL_ORDERS]


def test_reset_after_cooldown(kill_switch):
    """Reset works after cooldown."""
    kill_switch.trigger(KillSwitchTrigger.MANUAL, "Test")
    assert kill_switch.is_triggered is True
    # Cooldown end = triggered_at + 60 min; set "now" to 61 min after trigger
    after_cooldown = kill_switch._triggered_at + timedelta(minutes=61)

    with patch("nexus.risk.kill_switch.datetime") as mock_dt:
        mock_dt.now.return_value = after_cooldown
        state = kill_switch.reset()
    assert state.is_triggered is False
    assert kill_switch.is_trading_allowed is True


def test_reset_before_cooldown_fails(kill_switch):
    """Reset fails before cooldown."""
    kill_switch.trigger(KillSwitchTrigger.MANUAL, "Test")
    assert kill_switch.is_triggered is True
    # Still inside cooldown: 30 min after trigger (cooldown is 60 min)
    still_in_cooldown = kill_switch._triggered_at + timedelta(minutes=30)

    with patch("nexus.risk.kill_switch.datetime") as mock_dt:
        mock_dt.now.return_value = still_in_cooldown
        with pytest.raises(KillSwitchError) as exc_info:
            kill_switch.reset(force=False)
        assert "cooldown" in str(exc_info.value).lower()
    assert kill_switch.is_triggered is True


def test_reset_force(kill_switch):
    """Force reset ignores cooldown."""
    kill_switch.trigger(KillSwitchTrigger.MANUAL, "Test")
    state = kill_switch.reset(force=True)
    assert state.is_triggered is False
    assert kill_switch.is_trading_allowed is True


def test_update_heartbeat(kill_switch):
    """Heartbeat updates tracked."""
    kill_switch.update_heartbeat()
    state = kill_switch.get_state()
    assert state.system_status["last_heartbeat"] is not None


def test_update_data_timestamp(kill_switch):
    """Data timestamp updates tracked."""
    kill_switch.update_data_timestamp()
    state = kill_switch.get_state()
    assert state.system_status["last_data_update"] is not None


def test_is_trading_allowed(kill_switch):
    """Property reflects state correctly."""
    assert kill_switch.is_trading_allowed is True
    health = _healthy_health(drawdown_pct=-15.0)
    kill_switch.check_conditions(health)
    assert kill_switch.is_trading_allowed is False
    kill_switch.reset(force=True)
    assert kill_switch.is_trading_allowed is True


def test_idempotent_trigger(kill_switch):
    """Re-triggering doesn't reset timer."""
    kill_switch.trigger(KillSwitchTrigger.MANUAL, "First")
    first_at = kill_switch._triggered_at

    kill_switch.trigger(KillSwitchTrigger.SYSTEM_ERROR, "Second")
    assert kill_switch._triggered_at == first_at
    assert kill_switch._message == "Second"
    assert kill_switch._trigger == KillSwitchTrigger.MANUAL


def test_kill_switch_state_to_dict(kill_switch):
    """Serialization works."""
    kill_switch.trigger(KillSwitchTrigger.MANUAL, "Test message")
    state = kill_switch.get_state()
    d = state.to_dict()
    assert d["is_triggered"] is True
    assert d["trigger"] == "manual"
    assert d["action_taken"] == "full_shutdown"
    assert d["message"] == "Test message"
    assert "triggered_at" in d
    assert "can_reset" in d
    assert "cooldown_remaining_seconds" in d
    assert "system_status" in d
    assert isinstance(d["system_status"], dict)


def test_multiple_errors_trigger(kill_switch):
    """Triggers when >= 5 active errors."""
    health = _healthy_health(active_errors=["e1", "e2", "e3", "e4", "e5"])
    state = kill_switch.check_conditions(health)
    assert state.is_triggered is True
    assert state.trigger == KillSwitchTrigger.SYSTEM_ERROR
    assert state.action_taken == KillSwitchAction.DISABLE_NEW_TRADES
    assert "5" in state.message or "errors" in state.message.lower()


def test_disabled_kill_switch(kill_switch, mock_settings):
    """When disabled, conditions do not trigger."""
    mock_settings.kill_switch_enabled = False
    ks = KillSwitch(settings=mock_settings)
    health = _healthy_health(drawdown_pct=-20.0)
    state = ks.check_conditions(health)
    assert state.is_triggered is False


def test_reset_when_not_triggered(kill_switch):
    """Reset when not triggered returns current state."""
    state = kill_switch.reset()
    assert state.is_triggered is False
    assert state.can_reset is True


def test_get_actions_to_execute_none(kill_switch):
    """When not triggered, no actions."""
    actions = kill_switch.get_actions_to_execute()
    assert actions == []


def test_system_health_to_dict():
    """SystemHealth serialization."""
    now = datetime.now(UTC)
    health = SystemHealth(
        last_heartbeat=now,
        last_data_update=now,
        seconds_since_heartbeat=0.0,
        seconds_since_data=0.0,
        drawdown_pct=-3.0,
        is_connected=True,
        active_errors=["err1"],
    )
    d = health.to_dict()
    assert d["drawdown_pct"] == -3.0
    assert d["is_connected"] is True
    assert d["active_errors"] == ["err1"]
    assert "last_heartbeat" in d
    assert "last_data_update" in d
