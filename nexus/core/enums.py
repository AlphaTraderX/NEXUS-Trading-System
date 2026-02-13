"""
NEXUS enumerations.
"""

from enum import Enum


class Direction(str, Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


class Market(str, Enum):
    """Markets that NEXUS trades."""

    US_STOCKS = "us_stocks"
    UK_STOCKS = "uk_stocks"
    EU_STOCKS = "eu_stocks"
    FOREX_MAJORS = "forex_majors"
    FOREX_CROSSES = "forex_crosses"
    US_FUTURES = "us_futures"
    COMMODITIES = "commodities"


class EdgeType(str, Enum):
    """Validated trading edges that NEXUS scans for."""

    TURN_OF_MONTH = "turn_of_month"
    MONTH_END = "month_end"
    GAP_FILL = "gap_fill"
    INSIDER_CLUSTER = "insider_cluster"
    EARNINGS_DRIFT = "earnings_drift"
    VWAP_DEVIATION = "vwap_deviation"
    RSI_EXTREME = "rsi_extreme"
    BOLLINGER_TOUCH = "bollinger_touch"
    ORB = "orb"
    LONDON_OPEN = "london_open"
    NY_OPEN = "ny_open"
    POWER_HOUR = "power_hour"
    ASIAN_RANGE = "asian_range"
    OVERNIGHT_PREMIUM = "overnight_premium"
    SENTIMENT_SPIKE = "sentiment_spike"


class NexusMode(str, Enum):
    """Trading mode."""

    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class MarketRegime(str, Enum):
    """Market regime for regime-aware scoring."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


class SignalTier(str, Enum):
    """Signal quality tier from scoring."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class SignalStatus(str, Enum):
    """Signal lifecycle status."""

    PENDING = "pending"
    SENT = "sent"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class AlertPriority(str, Enum):
    """Priority level for delivery alerts."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CircuitBreakerStatus(str, Enum):
    """Circuit breaker status for risk management."""

    CLEAR = "clear"  # All systems go
    WARNING = "warning"  # Caution, but can trade
    REDUCED = "reduced"  # Trading at reduced size
    DAILY_STOP = "daily_stop"  # No more trading today
    WEEKLY_STOP = "weekly_stop"  # No more trading this week
    FULL_STOP = "full_stop"  # Complete halt - manual review needed


class KillSwitchTrigger(str, Enum):
    """Reason the kill switch was triggered."""

    NONE = "none"
    MAX_DRAWDOWN = "max_drawdown"
    CONNECTION_LOSS = "connection_loss"
    STALE_DATA = "stale_data"
    MANUAL = "manual"
    SYSTEM_ERROR = "system_error"
    BROKER_ERROR = "broker_error"


class KillSwitchAction(str, Enum):
    """Action taken when kill switch triggers."""

    NONE = "none"
    CANCEL_ALL_ORDERS = "cancel_all_orders"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    DISABLE_NEW_TRADES = "disable_new_trades"
    FULL_SHUTDOWN = "full_shutdown"  # All of the above


class Timeframe(str, Enum):
    """Trading timeframes for multi-timeframe analysis."""

    M5 = "5m"    # 5 minutes - scalping
    M15 = "15m"  # 15 minutes - intraday
    M30 = "30m"  # 30 minutes - intraday
    H1 = "1h"    # 1 hour - swing
    H4 = "4h"    # 4 hours - swing
    D1 = "1d"    # Daily - position

    @property
    def minutes(self) -> int:
        """Get timeframe in minutes."""
        mapping = {
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping[self.value]

    @property
    def bars_per_day(self) -> int:
        """How many bars in a trading day."""
        return 1440 // self.minutes

    @classmethod
    def intraday(cls) -> list:
        """Get intraday timeframes only."""
        return [cls.M5, cls.M15, cls.M30, cls.H1]

    @classmethod
    def swing(cls) -> list:
        """Get swing trading timeframes."""
        return [cls.H4, cls.D1]


# Timeframe-specific thresholds for mean reversion scanners
TIMEFRAME_THRESHOLDS = {
    Timeframe.M5: {
        "vwap_deviation_std": 1.5,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
        "bollinger_std": 1.8,
    },
    Timeframe.M15: {
        "vwap_deviation_std": 1.7,
        "rsi_oversold": 22,
        "rsi_overbought": 78,
        "bollinger_std": 1.9,
    },
    Timeframe.M30: {
        "vwap_deviation_std": 1.8,
        "rsi_oversold": 23,
        "rsi_overbought": 77,
        "bollinger_std": 1.9,
    },
    Timeframe.H1: {
        "vwap_deviation_std": 2.0,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bollinger_std": 2.0,
    },
    Timeframe.H4: {
        "vwap_deviation_std": 2.0,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bollinger_std": 2.0,
    },
    Timeframe.D1: {
        "vwap_deviation_std": 2.0,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bollinger_std": 2.0,
    },
}
