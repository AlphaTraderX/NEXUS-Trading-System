"""
NEXUS Smart Circuit Breaker
Loss-based circuit breakers that protect capital WITHOUT capping profits.

KEY PRINCIPLE:
Circuit breakers trigger on LOSSES, not gains or trade count.
This protects downside while allowing unlimited upside.

LEVELS:
- Daily loss -1.5%: WARNING (alert, continue trading)
- Daily loss -2.0%: REDUCE (reduce position sizes 50%)
- Daily loss -3.0%: STOP (no new trades today)
- Weekly loss -6.0%: STOP (no new trades this week)
- Drawdown -10.0%: FULL STOP (manual review required)

THERE ARE NO PROFIT CAPS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BreakerStatus(Enum):
    """Circuit breaker status levels."""
    CLEAR = "clear"           # All systems go
    WARNING = "warning"       # Alert but continue
    REDUCED = "reduced"       # Trading with reduced size
    DAILY_STOP = "daily_stop" # No new trades today
    WEEKLY_STOP = "weekly_stop"  # No new trades this week
    FULL_STOP = "full_stop"   # Manual review required


class BreakerAction(Enum):
    """Actions to take when breaker triggers."""
    CONTINUE = "continue"           # Normal trading
    ALERT = "alert"                 # Send alert, continue
    REDUCE_SIZE = "reduce_size"     # Reduce position sizes
    STOP_NEW_TRADES = "stop_new"    # No new positions
    CLOSE_ALL = "close_all"         # Close all positions
    FULL_SHUTDOWN = "shutdown"      # Complete shutdown


@dataclass
class BreakerState:
    """Current state of circuit breakers."""
    status: BreakerStatus
    action: BreakerAction
    can_trade: bool
    size_multiplier: float
    reason: str

    # Current metrics
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float

    # Thresholds hit
    thresholds_hit: List[str] = field(default_factory=list)

    # Recovery info
    recovery_time: Optional[datetime] = None
    requires_manual_reset: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "action": self.action.value,
            "can_trade": self.can_trade,
            "size_multiplier": self.size_multiplier,
            "reason": self.reason,
            "metrics": {
                "daily_pnl_pct": round(self.daily_pnl_pct, 2),
                "weekly_pnl_pct": round(self.weekly_pnl_pct, 2),
                "drawdown_pct": round(self.drawdown_pct, 2),
            },
            "thresholds_hit": self.thresholds_hit,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "requires_manual_reset": self.requires_manual_reset,
        }


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_equity: float
    current_equity: float
    high_water_mark: float
    pnl: float
    pnl_pct: float
    trades_taken: int
    winners: int
    losers: int


class SmartCircuitBreaker:
    """
    Smart circuit breaker system.

    Protects capital with loss-based triggers.
    NO profit caps - only downside protection.
    """

    def __init__(
        self,
        # Daily limits
        daily_warning: float = -1.5,
        daily_reduce: float = -2.0,
        daily_stop: float = -3.0,

        # Weekly limits
        weekly_warning: float = -4.0,
        weekly_stop: float = -6.0,

        # Drawdown limits
        drawdown_warning: float = -7.5,
        drawdown_stop: float = -10.0,

        # Size reduction when triggered
        reduced_size_multiplier: float = 0.5,

        # Auto-reset behavior
        auto_reset_daily: bool = True,
        auto_reset_weekly: bool = True,
    ):
        """
        Initialize circuit breaker.

        Args:
            daily_warning: Daily loss % that triggers warning
            daily_reduce: Daily loss % that triggers size reduction
            daily_stop: Daily loss % that stops trading for day
            weekly_warning: Weekly loss % that triggers warning
            weekly_stop: Weekly loss % that stops trading for week
            drawdown_warning: Drawdown % that triggers warning
            drawdown_stop: Drawdown % that triggers full stop
            reduced_size_multiplier: Size multiplier when in reduced mode
            auto_reset_daily: Auto-reset daily breakers at start of day
            auto_reset_weekly: Auto-reset weekly breakers at start of week
        """
        # Daily thresholds
        self.daily_warning = daily_warning
        self.daily_reduce = daily_reduce
        self.daily_stop = daily_stop

        # Weekly thresholds
        self.weekly_warning = weekly_warning
        self.weekly_stop = weekly_stop

        # Drawdown thresholds
        self.drawdown_warning = drawdown_warning
        self.drawdown_stop = drawdown_stop

        # Behavior
        self.reduced_size_multiplier = reduced_size_multiplier
        self.auto_reset_daily = auto_reset_daily
        self.auto_reset_weekly = auto_reset_weekly

        # State tracking
        self.is_manually_stopped = False
        self.manual_stop_reason = ""
        self.last_check_time = None

        # History
        self.daily_stats: Dict[date, DailyStats] = {}
        self.breaker_history: List[Dict] = []

    def check(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        drawdown_pct: float,
        current_time: datetime = None
    ) -> BreakerState:
        """
        Check all circuit breakers and return current state.

        Args:
            daily_pnl_pct: Today's P&L as percentage (negative for loss)
            weekly_pnl_pct: This week's P&L as percentage
            drawdown_pct: Current drawdown from peak (negative)
            current_time: Current timestamp (for logging)

        Returns:
            BreakerState with status, action, and whether trading allowed
        """
        current_time = current_time or datetime.now()
        self.last_check_time = current_time

        thresholds_hit = []

        # Check for manual stop first
        if self.is_manually_stopped:
            return BreakerState(
                status=BreakerStatus.FULL_STOP,
                action=BreakerAction.FULL_SHUTDOWN,
                can_trade=False,
                size_multiplier=0,
                reason=f"Manual stop: {self.manual_stop_reason}",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=["manual_stop"],
                requires_manual_reset=True,
            )

        # Check drawdown (most severe)
        if drawdown_pct <= self.drawdown_stop:
            thresholds_hit.append(f"drawdown_stop ({drawdown_pct:.1f}% <= {self.drawdown_stop}%)")
            self._log_breaker_event("DRAWDOWN_STOP", drawdown_pct, current_time)
            return BreakerState(
                status=BreakerStatus.FULL_STOP,
                action=BreakerAction.CLOSE_ALL,
                can_trade=False,
                size_multiplier=0,
                reason=f"Max drawdown hit: {drawdown_pct:.1f}%",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=thresholds_hit,
                requires_manual_reset=True,
            )

        if drawdown_pct <= self.drawdown_warning:
            thresholds_hit.append(f"drawdown_warning ({drawdown_pct:.1f}%)")

        # Check weekly loss
        if weekly_pnl_pct <= self.weekly_stop:
            thresholds_hit.append(f"weekly_stop ({weekly_pnl_pct:.1f}% <= {self.weekly_stop}%)")
            self._log_breaker_event("WEEKLY_STOP", weekly_pnl_pct, current_time)

            # Calculate recovery time (next Monday)
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            recovery = current_time + timedelta(days=days_until_monday)
            recovery = recovery.replace(hour=0, minute=0, second=0, microsecond=0)

            return BreakerState(
                status=BreakerStatus.WEEKLY_STOP,
                action=BreakerAction.STOP_NEW_TRADES,
                can_trade=False,
                size_multiplier=0,
                reason=f"Weekly loss limit hit: {weekly_pnl_pct:.1f}%",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=thresholds_hit,
                recovery_time=recovery,
                requires_manual_reset=not self.auto_reset_weekly,
            )

        if weekly_pnl_pct <= self.weekly_warning:
            thresholds_hit.append(f"weekly_warning ({weekly_pnl_pct:.1f}%)")

        # Check daily loss
        if daily_pnl_pct <= self.daily_stop:
            thresholds_hit.append(f"daily_stop ({daily_pnl_pct:.1f}% <= {self.daily_stop}%)")
            self._log_breaker_event("DAILY_STOP", daily_pnl_pct, current_time)

            # Recovery tomorrow
            recovery = current_time + timedelta(days=1)
            recovery = recovery.replace(hour=0, minute=0, second=0, microsecond=0)

            return BreakerState(
                status=BreakerStatus.DAILY_STOP,
                action=BreakerAction.STOP_NEW_TRADES,
                can_trade=False,
                size_multiplier=0,
                reason=f"Daily loss limit hit: {daily_pnl_pct:.1f}%",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=thresholds_hit,
                recovery_time=recovery,
                requires_manual_reset=not self.auto_reset_daily,
            )

        if daily_pnl_pct <= self.daily_reduce:
            thresholds_hit.append(f"daily_reduce ({daily_pnl_pct:.1f}% <= {self.daily_reduce}%)")
            self._log_breaker_event("DAILY_REDUCE", daily_pnl_pct, current_time)

            return BreakerState(
                status=BreakerStatus.REDUCED,
                action=BreakerAction.REDUCE_SIZE,
                can_trade=True,
                size_multiplier=self.reduced_size_multiplier,
                reason=f"Daily loss warning: reducing size to {self.reduced_size_multiplier*100:.0f}%",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=thresholds_hit,
            )

        if daily_pnl_pct <= self.daily_warning:
            thresholds_hit.append(f"daily_warning ({daily_pnl_pct:.1f}%)")

            return BreakerState(
                status=BreakerStatus.WARNING,
                action=BreakerAction.ALERT,
                can_trade=True,
                size_multiplier=1.0,
                reason=f"Daily loss warning: {daily_pnl_pct:.1f}%",
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                drawdown_pct=drawdown_pct,
                thresholds_hit=thresholds_hit,
            )

        # All clear
        return BreakerState(
            status=BreakerStatus.CLEAR,
            action=BreakerAction.CONTINUE,
            can_trade=True,
            size_multiplier=1.0,
            reason="All systems clear",
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
            thresholds_hit=thresholds_hit,
        )

    def manual_stop(self, reason: str):
        """Manually trigger a full stop."""
        self.is_manually_stopped = True
        self.manual_stop_reason = reason
        self._log_breaker_event("MANUAL_STOP", 0, datetime.now())

    def manual_reset(self):
        """Reset manual stop (requires human intervention)."""
        self.is_manually_stopped = False
        self.manual_stop_reason = ""
        self._log_breaker_event("MANUAL_RESET", 0, datetime.now())

    def _log_breaker_event(self, event_type: str, value: float, timestamp: datetime):
        """Log a circuit breaker event."""
        self.breaker_history.append({
            "event": event_type,
            "value": value,
            "timestamp": timestamp.isoformat(),
        })
        # Keep last 100 events
        if len(self.breaker_history) > 100:
            self.breaker_history = self.breaker_history[-100:]

    def get_thresholds(self) -> Dict:
        """Get all threshold settings."""
        return {
            "daily": {
                "warning": self.daily_warning,
                "reduce": self.daily_reduce,
                "stop": self.daily_stop,
            },
            "weekly": {
                "warning": self.weekly_warning,
                "stop": self.weekly_stop,
            },
            "drawdown": {
                "warning": self.drawdown_warning,
                "stop": self.drawdown_stop,
            },
            "reduced_size_multiplier": self.reduced_size_multiplier,
        }

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent breaker events."""
        return self.breaker_history[-limit:]


# Test the circuit breaker
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS SMART CIRCUIT BREAKER TEST")
    print("=" * 60)

    breaker = SmartCircuitBreaker(
        daily_warning=-1.5,
        daily_reduce=-2.0,
        daily_stop=-3.0,
        weekly_warning=-4.0,
        weekly_stop=-6.0,
        drawdown_warning=-7.5,
        drawdown_stop=-10.0,
    )

    # Test 1: All clear
    print("\n--- Test 1: All Clear ---")
    state = breaker.check(
        daily_pnl_pct=0.5,
        weekly_pnl_pct=2.0,
        drawdown_pct=-3.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Size multiplier: {state.size_multiplier}x")
    print(f"Reason: {state.reason}")

    # Test 2: Daily warning
    print("\n--- Test 2: Daily Warning (-1.5%) ---")
    state = breaker.check(
        daily_pnl_pct=-1.5,
        weekly_pnl_pct=-1.0,
        drawdown_pct=-3.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Action: {state.action.value}")
    print(f"Thresholds hit: {state.thresholds_hit}")

    # Test 3: Daily reduce
    print("\n--- Test 3: Daily Reduce (-2.0%) ---")
    state = breaker.check(
        daily_pnl_pct=-2.0,
        weekly_pnl_pct=-2.0,
        drawdown_pct=-4.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Size multiplier: {state.size_multiplier}x")
    print(f"Reason: {state.reason}")

    # Test 4: Daily stop
    print("\n--- Test 4: Daily Stop (-3.0%) ---")
    state = breaker.check(
        daily_pnl_pct=-3.0,
        weekly_pnl_pct=-3.0,
        drawdown_pct=-5.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Action: {state.action.value}")
    print(f"Recovery time: {state.recovery_time}")

    # Test 5: Weekly stop
    print("\n--- Test 5: Weekly Stop (-6.0%) ---")
    state = breaker.check(
        daily_pnl_pct=-1.0,
        weekly_pnl_pct=-6.0,
        drawdown_pct=-7.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Recovery time: {state.recovery_time}")
    print(f"Requires manual reset: {state.requires_manual_reset}")

    # Test 6: Max drawdown (full stop)
    print("\n--- Test 6: Max Drawdown Stop (-10%) ---")
    state = breaker.check(
        daily_pnl_pct=-2.0,
        weekly_pnl_pct=-5.0,
        drawdown_pct=-10.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Action: {state.action.value}")
    print(f"Requires manual reset: {state.requires_manual_reset}")

    # Test 7: Manual stop
    print("\n--- Test 7: Manual Stop ---")
    breaker.manual_stop("Testing emergency shutdown")
    state = breaker.check(
        daily_pnl_pct=1.0,
        weekly_pnl_pct=5.0,
        drawdown_pct=-2.0
    )
    print(f"Status: {state.status.value}")
    print(f"Can trade: {state.can_trade}")
    print(f"Reason: {state.reason}")

    # Reset manual stop
    breaker.manual_reset()

    # Test 8: Display thresholds
    print("\n--- Test 8: Threshold Settings ---")
    thresholds = breaker.get_thresholds()
    print(f"Daily: warning={thresholds['daily']['warning']}%, reduce={thresholds['daily']['reduce']}%, stop={thresholds['daily']['stop']}%")
    print(f"Weekly: warning={thresholds['weekly']['warning']}%, stop={thresholds['weekly']['stop']}%")
    print(f"Drawdown: warning={thresholds['drawdown']['warning']}%, stop={thresholds['drawdown']['stop']}%")

    # Test 9: History
    print("\n--- Test 9: Breaker History ---")
    history = breaker.get_history(5)
    print(f"Recent events: {len(history)}")
    for event in history[-3:]:
        print(f"  {event['event']} at {event['timestamp']}")

    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER TEST COMPLETE [OK]")
    print("=" * 60)
