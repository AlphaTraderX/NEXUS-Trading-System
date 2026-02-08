"""
NEXUS Smart Circuit Breaker tests.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from nexus.core.enums import CircuitBreakerStatus
from nexus.core.models import CircuitBreakerState
from nexus.risk.circuit_breaker import SmartCircuitBreaker

UTC = timezone.utc


@pytest.fixture
def mock_settings():
    """Settings with circuit breaker thresholds."""
    s = MagicMock()
    s.daily_loss_warning = -1.5
    s.daily_loss_reduce = -2.0
    s.daily_loss_stop = -3.0
    s.weekly_loss_stop = -6.0
    s.max_drawdown = -10.0
    return s


@pytest.fixture
def breaker(mock_settings):
    return SmartCircuitBreaker(settings=mock_settings)


def test_clear_status(breaker):
    """No triggers, full trading."""
    state = breaker.check_status(
        daily_pnl_pct=0.5,
        weekly_pnl_pct=2.0,
        drawdown_pct=-2.0,
    )
    assert state.status == CircuitBreakerStatus.CLEAR
    assert state.can_trade is True
    assert state.size_multiplier == 1.0
    assert "All clear" in state.message


def test_warning_level(breaker):
    """Daily loss at warning threshold."""
    state = breaker.check_status(
        daily_pnl_pct=-1.5,
        weekly_pnl_pct=-0.5,
        drawdown_pct=-3.0,
    )
    assert state.status == CircuitBreakerStatus.WARNING
    assert state.can_trade is True
    assert state.size_multiplier == 0.75
    assert "warning" in state.message.lower() or "caution" in state.message.lower()


def test_reduced_level(breaker):
    """Daily loss at reduce threshold."""
    state = breaker.check_status(
        daily_pnl_pct=-2.0,
        weekly_pnl_pct=-1.0,
        drawdown_pct=-4.0,
    )
    assert state.status == CircuitBreakerStatus.REDUCED
    assert state.can_trade is True
    assert state.size_multiplier == 0.5
    assert "50%" in state.message


def test_daily_stop(breaker):
    """Daily loss at stop threshold."""
    state = breaker.check_status(
        daily_pnl_pct=-3.0,
        weekly_pnl_pct=-2.0,
        drawdown_pct=-5.0,
    )
    assert state.status == CircuitBreakerStatus.DAILY_STOP
    assert state.can_trade is False
    assert state.size_multiplier == 0.0
    assert state.resume_at is not None
    # Resume should be tomorrow 00:00 UTC
    today = date.today()
    expected_resume = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=UTC) + timedelta(days=1)
    assert state.resume_at.date() == expected_resume.date()


def test_weekly_stop(breaker):
    """Weekly loss at stop threshold."""
    state = breaker.check_status(
        daily_pnl_pct=-1.0,
        weekly_pnl_pct=-6.0,
        drawdown_pct=-7.0,
    )
    assert state.status == CircuitBreakerStatus.WEEKLY_STOP
    assert state.can_trade is False
    assert state.size_multiplier == 0.0
    assert state.resume_at is not None
    # Resume should be next Monday 00:00 UTC
    assert state.resume_at.weekday() == 0  # Monday


def test_full_stop_drawdown(breaker):
    """Drawdown at max threshold."""
    state = breaker.check_status(
        daily_pnl_pct=-2.0,
        weekly_pnl_pct=-5.0,
        drawdown_pct=-10.0,
    )
    assert state.status == CircuitBreakerStatus.FULL_STOP
    assert state.can_trade is False
    assert state.size_multiplier == 0.0
    assert state.resume_at is None
    assert "manual review" in state.message.lower()


def test_severity_ordering(breaker):
    """Worse condition takes precedence."""
    # Daily -3% and weekly -6% and drawdown -10%: FULL_STOP wins
    state = breaker.check_status(
        daily_pnl_pct=-3.0,
        weekly_pnl_pct=-6.0,
        drawdown_pct=-10.0,
    )
    assert state.status == CircuitBreakerStatus.FULL_STOP

    # New breaker: weekly -6% and daily -3% but drawdown -5%: WEEKLY_STOP wins
    b2 = SmartCircuitBreaker(settings=breaker.settings)
    state2 = b2.check_status(
        daily_pnl_pct=-3.0,
        weekly_pnl_pct=-6.0,
        drawdown_pct=-5.0,
    )
    assert state2.status == CircuitBreakerStatus.WEEKLY_STOP

    # New breaker: daily -3%, weekly -2%, drawdown -4%: DAILY_STOP wins
    b3 = SmartCircuitBreaker(settings=breaker.settings)
    state3 = b3.check_status(
        daily_pnl_pct=-3.0,
        weekly_pnl_pct=-2.0,
        drawdown_pct=-4.0,
    )
    assert state3.status == CircuitBreakerStatus.DAILY_STOP


def test_reset_daily(breaker):
    """Daily stop clears after reset_daily."""
    breaker.check_status(daily_pnl_pct=-3.0, weekly_pnl_pct=0.0, drawdown_pct=-2.0)
    assert breaker._current_status == CircuitBreakerStatus.DAILY_STOP
    assert breaker._daily_stop_date is not None

    breaker.reset_daily()
    assert breaker._daily_stop_date is None
    assert breaker._current_status == CircuitBreakerStatus.CLEAR

    # After reset, can trade again with good PnL
    state = breaker.check_status(daily_pnl_pct=0.0, weekly_pnl_pct=0.0, drawdown_pct=-2.0)
    assert state.status == CircuitBreakerStatus.CLEAR
    assert state.can_trade is True


def test_reset_weekly(breaker):
    """Weekly stop clears after reset_weekly."""
    breaker.check_status(daily_pnl_pct=0.0, weekly_pnl_pct=-6.0, drawdown_pct=-5.0)
    assert breaker._current_status == CircuitBreakerStatus.WEEKLY_STOP
    assert breaker._weekly_stop_date is not None

    breaker.reset_weekly()
    assert breaker._weekly_stop_date is None
    assert breaker._current_status == CircuitBreakerStatus.CLEAR

    # After reset, with good weekly PnL we get CLEAR (no re-trigger)
    state = breaker.check_status(daily_pnl_pct=0.0, weekly_pnl_pct=-2.0, drawdown_pct=-3.0)
    assert state.status == CircuitBreakerStatus.CLEAR
    assert state.can_trade is True


def test_full_stop_requires_force_reset(breaker):
    """FULL_STOP doesn't clear with daily/weekly reset."""
    breaker.check_status(daily_pnl_pct=-1.0, weekly_pnl_pct=-2.0, drawdown_pct=-10.0)
    assert breaker._current_status == CircuitBreakerStatus.FULL_STOP

    breaker.reset_daily()
    assert breaker._current_status == CircuitBreakerStatus.FULL_STOP

    breaker.reset_weekly()
    assert breaker._current_status == CircuitBreakerStatus.FULL_STOP

    # Only force_reset clears it
    breaker.force_reset()
    assert breaker._current_status == CircuitBreakerStatus.CLEAR
    assert breaker._triggered_at is None


def test_size_multipliers(breaker):
    """Correct multipliers for each status."""
    cases = [
        (0.0, -1.0, -2.0, CircuitBreakerStatus.CLEAR, 1.0),
        (-1.5, -1.0, -2.0, CircuitBreakerStatus.WARNING, 0.75),
        (-2.0, -1.0, -3.0, CircuitBreakerStatus.REDUCED, 0.5),
        (-3.0, -2.0, -4.0, CircuitBreakerStatus.DAILY_STOP, 0.0),
        (-1.0, -6.0, -5.0, CircuitBreakerStatus.WEEKLY_STOP, 0.0),
        (-2.0, -5.0, -10.0, CircuitBreakerStatus.FULL_STOP, 0.0),
    ]
    for daily, weekly, drawdown, expected_status, expected_mult in cases:
        b = SmartCircuitBreaker(settings=breaker.settings)
        state = b.check_status(
            daily_pnl_pct=daily,
            weekly_pnl_pct=weekly,
            drawdown_pct=drawdown,
        )
        assert state.status == expected_status, f"daily={daily} weekly={weekly} dd={drawdown}"
        assert state.size_multiplier == expected_mult, f"status={state.status}"


def test_resume_at_calculation(breaker):
    """Correct resume times calculated."""
    # Daily stop -> resume tomorrow 00:00 UTC
    state_d = breaker.check_status(daily_pnl_pct=-3.0, weekly_pnl_pct=0.0, drawdown_pct=-1.0)
    assert state_d.resume_at is not None
    today = date.today()
    tomorrow = today + timedelta(days=1)
    assert state_d.resume_at.date() == tomorrow
    assert state_d.resume_at.hour == 0 and state_d.resume_at.minute == 0

    # Weekly stop -> resume next Monday 00:00 UTC
    b2 = SmartCircuitBreaker(settings=breaker.settings)
    state_w = b2.check_status(daily_pnl_pct=0.0, weekly_pnl_pct=-6.0, drawdown_pct=-3.0)
    assert state_w.resume_at is not None
    assert state_w.resume_at.weekday() == 0  # Monday
    assert state_w.resume_at.hour == 0 and state_w.resume_at.minute == 0

    # Full stop -> no resume_at
    b3 = SmartCircuitBreaker(settings=breaker.settings)
    state_f = b3.check_status(daily_pnl_pct=-1.0, weekly_pnl_pct=-2.0, drawdown_pct=-10.0)
    assert state_f.resume_at is None


def test_get_thresholds(breaker):
    """Returns all threshold values."""
    t = breaker.get_thresholds()
    assert t["daily_loss_warning"] == -1.5
    assert t["daily_loss_reduce"] == -2.0
    assert t["daily_loss_stop"] == -3.0
    assert t["weekly_loss_stop"] == -6.0
    assert t["max_drawdown"] == -10.0
    assert len(t) == 5


def test_sticky_daily_stop(breaker):
    """DAILY_STOP persists until reset_daily even if PnL improves in same day."""
    breaker.check_status(daily_pnl_pct=-3.0, weekly_pnl_pct=0.0, drawdown_pct=-1.0)
    # Same day, "recovered" daily PnL to 0% - still stopped
    state = breaker.check_status(daily_pnl_pct=0.0, weekly_pnl_pct=0.0, drawdown_pct=-1.0)
    assert state.status == CircuitBreakerStatus.DAILY_STOP
    assert state.can_trade is False


def test_circuit_breaker_state_to_dict(breaker):
    """CircuitBreakerState.to_dict serializes correctly."""
    state = breaker.check_status(daily_pnl_pct=-2.0, weekly_pnl_pct=-1.0, drawdown_pct=-4.0)
    d = state.to_dict()
    assert d["status"] == "reduced"
    assert d["can_trade"] is True
    assert d["size_multiplier"] == 0.5
    assert "daily_pnl_pct" in d
    assert "weekly_pnl_pct" in d
    assert "drawdown_pct" in d
    assert "message" in d

    # State with resume_at
    b2 = SmartCircuitBreaker(settings=breaker.settings)
    state2 = b2.check_status(daily_pnl_pct=-3.0, weekly_pnl_pct=0.0, drawdown_pct=-2.0)
    d2 = state2.to_dict()
    assert "resume_at" in d2
    assert d2["resume_at"] is not None
