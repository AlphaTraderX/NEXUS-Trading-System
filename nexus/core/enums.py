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


class NexusMode(str, Enum):
    """Trading mode."""

    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class SignalStatus(str, Enum):
    """Signal lifecycle status."""

    PENDING = "pending"
    SENT = "sent"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
