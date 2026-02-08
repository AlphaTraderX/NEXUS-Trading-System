"""
NEXUS Smart Circuit Breaker

Loss-based circuit breaker that protects capital.
KEY PRINCIPLE: Only triggers on losses, NEVER caps profits.

Graduated response:
1. WARNING: Alert but continue trading (75% size)
2. REDUCED: Trade at 50% size
3. DAILY_STOP: No new trades today
4. WEEKLY_STOP: No new trades this week
5. FULL_STOP: Complete halt, manual review required
"""

import logging
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from nexus.config.settings import get_settings
from nexus.core.enums import CircuitBreakerStatus
from nexus.core.models import CircuitBreakerState
from nexus.risk.state_persistence import get_risk_persistence

logger = logging.getLogger(__name__)

UTC = timezone.utc


def _monday_of_week(d: date) -> date:
    """Return the Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


def _next_monday_utc(from_date: date) -> datetime:
    """Next Monday 00:00 UTC after from_date."""
    this_monday = _monday_of_week(from_date)
    next_monday = this_monday + timedelta(days=7)
    return datetime(next_monday.year, next_monday.month, next_monday.day, 0, 0, 0, tzinfo=UTC)


def _tomorrow_utc(from_date: date) -> datetime:
    """Tomorrow 00:00 UTC."""
    t = from_date + timedelta(days=1)
    return datetime(t.year, t.month, t.day, 0, 0, 0, tzinfo=UTC)


class SmartCircuitBreaker:
    """
    Loss-based circuit breaker that protects capital.

    KEY PRINCIPLE: Only triggers on losses, NEVER caps profits.

    Graduated response:
    1. WARNING: Alert but continue trading
    2. REDUCED: Trade at 50% size
    3. DAILY_STOP: No new trades today
    4. WEEKLY_STOP: No new trades this week
    5. FULL_STOP: Complete halt, manual review required
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        # Thresholds (all negative values)
        self.daily_loss_warning = getattr(self.settings, "daily_loss_warning", -1.5)
        self.daily_loss_reduce = getattr(self.settings, "daily_loss_reduce", -2.0)
        self.daily_loss_stop = getattr(self.settings, "daily_loss_stop", -3.0)
        self.weekly_loss_stop = getattr(self.settings, "weekly_loss_stop", -6.0)
        self.max_drawdown = getattr(self.settings, "max_drawdown", -10.0)

        # State tracking (sticky stops)
        self._current_status = CircuitBreakerStatus.CLEAR
        self._triggered_at: Optional[datetime] = None
        self._daily_stop_date: Optional[date] = None
        self._weekly_stop_date: Optional[date] = None  # Monday of week when we stopped
        self._lock = threading.RLock()

    def check_status(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        drawdown_pct: float,
    ) -> CircuitBreakerState:
        """
        Check circuit breaker status based on current P&L.

        Args:
            daily_pnl_pct: Today's P&L as percentage (negative for loss)
            weekly_pnl_pct: This week's P&L as percentage
            drawdown_pct: Current drawdown from peak (negative value)

        Returns:
            CircuitBreakerState with status, can_trade, size_multiplier, message
        """
        # FIRST: Check persisted state (survives restarts)
        persistence = get_risk_persistence()
        allowed, persist_reason = persistence.is_trading_allowed()
        if not allowed:
            name = persistence.get_circuit_breaker_status()
            try:
                status = CircuitBreakerStatus[name]
            except KeyError:
                status = CircuitBreakerStatus.FULL_STOP
            return CircuitBreakerState(
                status=status,
                can_trade=False,
                size_multiplier=0.0,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                message=persist_reason,
                triggered_at=None,
                resume_at=None,
            )

        now = datetime.now(UTC)
        today = now.date()

        with self._lock:
            # 1. FULL_STOP (sticky until force_reset)
            if self._current_status == CircuitBreakerStatus.FULL_STOP:
                logger.info(
                    "Circuit breaker status check: FULL_STOP (manual reset required)"
                )
                persistence = get_risk_persistence()
                persistence.set_circuit_breaker_status(
                    "FULL_STOP",
                    "Max drawdown limit hit. Full stop - manual review required.",
                )
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.FULL_STOP,
                    can_trade=False,
                    size_multiplier=0.0,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message="Max drawdown limit hit. Full stop - manual review required.",
                    triggered_at=self._triggered_at,
                    resume_at=None,
                )

            # 2. WEEKLY_STOP (sticky until reset_weekly)
            if self._weekly_stop_date is not None:
                if _monday_of_week(today) == self._weekly_stop_date:
                    resume = _next_monday_utc(today)
                    logger.info(
                        "Circuit breaker status check: WEEKLY_STOP (resume %s)",
                        resume.isoformat(),
                    )
                    persistence = get_risk_persistence()
                    persistence.set_circuit_breaker_status(
                        "WEEKLY_STOP",
                        f"Weekly loss {weekly_pnl_pct:.2f}% exceeds limit. Trading halted until next week.",
                    )
                    return CircuitBreakerState(
                        status=CircuitBreakerStatus.WEEKLY_STOP,
                        can_trade=False,
                        size_multiplier=0.0,
                        daily_pnl_pct=daily_pnl_pct,
                        weekly_pnl_pct=weekly_pnl_pct,
                        drawdown_pct=drawdown_pct,
                        message=f"Weekly loss {weekly_pnl_pct:.2f}% exceeds limit. Trading halted until next week.",
                        triggered_at=self._triggered_at,
                        resume_at=resume,
                    )

            # 3. DAILY_STOP (sticky until reset_daily)
            if self._daily_stop_date is not None and self._daily_stop_date == today:
                resume = _tomorrow_utc(today)
                logger.info(
                    "Circuit breaker status check: DAILY_STOP (resume %s)",
                    resume.isoformat(),
                )
                persistence = get_risk_persistence()
                persistence.set_circuit_breaker_status(
                    "DAILY_STOP",
                    f"Daily loss {daily_pnl_pct:.2f}% exceeds limit. Trading halted for today.",
                )
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.DAILY_STOP,
                    can_trade=False,
                    size_multiplier=0.0,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Daily loss {daily_pnl_pct:.2f}% exceeds limit. Trading halted for today.",
                    triggered_at=self._triggered_at,
                    resume_at=resume,
                )

            # Clear sticky daily/weekly if we're past the date (safety; normally reset_* is called)
            if self._daily_stop_date is not None and self._daily_stop_date < today:
                self._daily_stop_date = None
            if self._weekly_stop_date is not None and _monday_of_week(today) != self._weekly_stop_date:
                self._weekly_stop_date = None

            # 4. Evaluate from current PnL (severity order: worst first)

            # FULL_STOP: drawdown
            if drawdown_pct <= self.max_drawdown:
                prev = self._current_status
                self._current_status = CircuitBreakerStatus.FULL_STOP
                self._triggered_at = now
                if prev != CircuitBreakerStatus.FULL_STOP:
                    logger.warning(
                        "Circuit breaker: FULL_STOP - drawdown %.2f%% <= %.2f%%",
                        drawdown_pct,
                        self.max_drawdown,
                    )
                persistence = get_risk_persistence()
                persistence.set_circuit_breaker_status(
                    "FULL_STOP",
                    f"Max drawdown {drawdown_pct:.2f}% hit. Full stop - manual review required.",
                )
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.FULL_STOP,
                    can_trade=False,
                    size_multiplier=0.0,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Max drawdown {drawdown_pct:.2f}% hit. Full stop - manual review required.",
                    triggered_at=self._triggered_at,
                    resume_at=None,
                )

            # WEEKLY_STOP: weekly loss
            if weekly_pnl_pct <= self.weekly_loss_stop:
                prev = self._current_status
                self._current_status = CircuitBreakerStatus.WEEKLY_STOP
                self._triggered_at = now
                self._weekly_stop_date = _monday_of_week(today)
                if prev != CircuitBreakerStatus.WEEKLY_STOP:
                    logger.warning(
                        "Circuit breaker: WEEKLY_STOP - weekly loss %.2f%% <= %.2f%%",
                        weekly_pnl_pct,
                        self.weekly_loss_stop,
                    )
                persistence = get_risk_persistence()
                persistence.set_circuit_breaker_status(
                    "WEEKLY_STOP",
                    f"Weekly loss {weekly_pnl_pct:.2f}% exceeds limit. Trading halted until next week.",
                )
                resume = _next_monday_utc(today)
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.WEEKLY_STOP,
                    can_trade=False,
                    size_multiplier=0.0,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Weekly loss {weekly_pnl_pct:.2f}% exceeds limit. Trading halted until next week.",
                    triggered_at=self._triggered_at,
                    resume_at=resume,
                )

            # DAILY_STOP: daily loss
            if daily_pnl_pct <= self.daily_loss_stop:
                prev = self._current_status
                self._current_status = CircuitBreakerStatus.DAILY_STOP
                self._triggered_at = now
                self._daily_stop_date = today
                if prev != CircuitBreakerStatus.DAILY_STOP:
                    logger.warning(
                        "Circuit breaker: DAILY_STOP - daily loss %.2f%% <= %.2f%%",
                        daily_pnl_pct,
                        self.daily_loss_stop,
                    )
                persistence = get_risk_persistence()
                persistence.set_circuit_breaker_status(
                    "DAILY_STOP",
                    f"Daily loss {daily_pnl_pct:.2f}% exceeds limit. Trading halted for today.",
                )
                resume = _tomorrow_utc(today)
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.DAILY_STOP,
                    can_trade=False,
                    size_multiplier=0.0,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Daily loss {daily_pnl_pct:.2f}% exceeds limit. Trading halted for today.",
                    triggered_at=self._triggered_at,
                    resume_at=resume,
                )

            # REDUCED: daily loss reduce threshold
            if daily_pnl_pct <= self.daily_loss_reduce:
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.REDUCED,
                    can_trade=True,
                    size_multiplier=0.5,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Daily loss {daily_pnl_pct:.2f}% - trading at 50% size.",
                    triggered_at=None,
                    resume_at=None,
                )

            # WARNING: daily loss warning threshold
            if daily_pnl_pct <= self.daily_loss_warning:
                return CircuitBreakerState(
                    status=CircuitBreakerStatus.WARNING,
                    can_trade=True,
                    size_multiplier=0.75,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl_pct=weekly_pnl_pct,
                    drawdown_pct=drawdown_pct,
                    message=f"Daily loss {daily_pnl_pct:.2f}% - warning level. Trade with caution.",
                    triggered_at=None,
                    resume_at=None,
                )

            # CLEAR
            self._current_status = CircuitBreakerStatus.CLEAR
            return CircuitBreakerState(
                status=CircuitBreakerStatus.CLEAR,
                can_trade=True,
                size_multiplier=1.0,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                message="All clear. No loss limits triggered.",
                triggered_at=None,
                resume_at=None,
            )

    def reset_daily(self) -> None:
        """Reset daily stop at start of new trading day."""
        with self._lock:
            if self._current_status == CircuitBreakerStatus.DAILY_STOP:
                logger.warning("Circuit breaker: reset_daily() - clearing DAILY_STOP")
            self._daily_stop_date = None
            if self._current_status == CircuitBreakerStatus.DAILY_STOP:
                self._current_status = CircuitBreakerStatus.CLEAR
                self._triggered_at = None
                get_risk_persistence().set_circuit_breaker_status("CLEAR", "")

    def reset_weekly(self) -> None:
        """Reset weekly stop at start of new trading week."""
        with self._lock:
            if self._current_status == CircuitBreakerStatus.WEEKLY_STOP:
                logger.warning("Circuit breaker: reset_weekly() - clearing WEEKLY_STOP")
            self._weekly_stop_date = None
            if self._current_status == CircuitBreakerStatus.WEEKLY_STOP:
                self._current_status = CircuitBreakerStatus.CLEAR
                self._triggered_at = None
                get_risk_persistence().set_circuit_breaker_status("CLEAR", "")

    def force_reset(self) -> None:
        """Manual reset (requires explicit action). Clears FULL_STOP and sticky state."""
        with self._lock:
            if self._current_status == CircuitBreakerStatus.FULL_STOP:
                logger.warning("Circuit breaker: force_reset() - clearing FULL_STOP")
            self._current_status = CircuitBreakerStatus.CLEAR
            self._triggered_at = None
            self._daily_stop_date = None
            self._weekly_stop_date = None
            get_risk_persistence().set_circuit_breaker_status("CLEAR", "")

    def get_thresholds(self) -> dict:
        """Return all threshold values for display/logging."""
        return {
            "daily_loss_warning": self.daily_loss_warning,
            "daily_loss_reduce": self.daily_loss_reduce,
            "daily_loss_stop": self.daily_loss_stop,
            "weekly_loss_stop": self.weekly_loss_stop,
            "max_drawdown": self.max_drawdown,
        }
