"""
Edge Configuration - Enable/disable edges and set weights based on backtest performance.

Backtest Results (2020-2024):
- VWAP Deviation: £36.4M, 66.9% WR  TOP PERFORMER
- Gap Fill: £33.0M, 71.5% WR  TOP PERFORMER
- Overnight Premium: £16.2M, 62.5% WR  STRONG
- Power Hour: £15.2M, 61.8% WR  STRONG
- RSI Extreme: £11.9M, 59.0% WR  SOLID
- Turn of Month: £6.4M, 64.7% WR  HIGH VALUE (low freq)
- ORB: £5.7M, 58.2% WR  SOLID
- London Open: £4.5M, 53.6% WR  MARGINAL
- Insider Cluster: £2.7M, 60.9% WR  HIGH VALUE (low freq)
- Asian Range: £2.6M, 67.6% WR  SOLID
- Bollinger Touch: £2.0M, 50.0% WR  MARGINAL
- Month End: -£285K, 50.0% WR  DISABLED
- NY Open: -£316K, 54.8% WR  DISABLED
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from nexus.core.enums import EdgeType


@dataclass
class EdgeConfig:
    """Configuration for a single edge."""

    edge_type: EdgeType
    enabled: bool
    weight: float  # 0.0-2.0, affects scoring
    min_score_override: Optional[int]  # Override global min score
    max_daily_signals: int  # Max signals per day from this edge
    cooldown_minutes: int  # Cooldown between signals on same instrument
    notes: str


# Master edge configuration based on backtest performance
EDGE_CONFIGS: Dict[EdgeType, EdgeConfig] = {
    # === TOP PERFORMERS (weight 1.2-1.5) ===
    EdgeType.VWAP_DEVIATION: EdgeConfig(
        edge_type=EdgeType.VWAP_DEVIATION,
        enabled=True,
        weight=1.5,
        min_score_override=None,
        max_daily_signals=8,
        cooldown_minutes=60,
        notes="£36.4M profit, 66.9% WR - TOP EDGE",
    ),
    EdgeType.GAP_FILL: EdgeConfig(
        edge_type=EdgeType.GAP_FILL,
        enabled=True,
        weight=1.4,
        min_score_override=None,
        max_daily_signals=6,
        cooldown_minutes=0,  # Gaps are time-sensitive
        notes="£33.0M profit, 71.5% WR - MOST RELIABLE",
    ),
    # === STRONG PERFORMERS (weight 1.0-1.2) ===
    EdgeType.OVERNIGHT_PREMIUM: EdgeConfig(
        edge_type=EdgeType.OVERNIGHT_PREMIUM,
        enabled=True,
        weight=1.2,
        min_score_override=None,
        max_daily_signals=10,
        cooldown_minutes=0,  # Once per day anyway
        notes="£16.2M profit, 62.5% WR - CONSISTENT",
    ),
    EdgeType.POWER_HOUR: EdgeConfig(
        edge_type=EdgeType.POWER_HOUR,
        enabled=True,
        weight=1.2,
        min_score_override=None,
        max_daily_signals=4,
        cooldown_minutes=0,  # Short window
        notes="£15.2M profit, 61.8% WR",
    ),
    EdgeType.RSI_EXTREME: EdgeConfig(
        edge_type=EdgeType.RSI_EXTREME,
        enabled=True,
        weight=1.1,
        min_score_override=None,
        max_daily_signals=6,
        cooldown_minutes=120,  # Avoid whipsaws
        notes="£11.9M profit, 59.0% WR",
    ),
    EdgeType.TURN_OF_MONTH: EdgeConfig(
        edge_type=EdgeType.TURN_OF_MONTH,
        enabled=True,
        weight=1.3,
        min_score_override=40,  # Lower threshold - always trade TOM
        max_daily_signals=5,
        cooldown_minutes=0,
        notes="£6.4M profit, 64.7% WR - CALENDAR EDGE",
    ),
    # === SOLID PERFORMERS (weight 1.0) ===
    EdgeType.ORB: EdgeConfig(
        edge_type=EdgeType.ORB,
        enabled=True,
        weight=1.0,
        min_score_override=None,
        max_daily_signals=4,
        cooldown_minutes=0,  # Morning only
        notes="£5.7M profit, 58.2% WR",
    ),
    EdgeType.INSIDER_CLUSTER: EdgeConfig(
        edge_type=EdgeType.INSIDER_CLUSTER,
        enabled=True,
        weight=1.3,
        min_score_override=35,  # Lower threshold - trust insider data
        max_daily_signals=3,
        cooldown_minutes=1440,  # Once per day per stock
        notes="£2.7M profit, 60.9% WR - SMART MONEY",
    ),
    EdgeType.ASIAN_RANGE: EdgeConfig(
        edge_type=EdgeType.ASIAN_RANGE,
        enabled=True,
        weight=1.0,
        min_score_override=None,
        max_daily_signals=3,
        cooldown_minutes=0,
        notes="£2.6M profit, 67.6% WR",
    ),
    # === MARGINAL PERFORMERS (weight 0.7-0.9) ===
    EdgeType.LONDON_OPEN: EdgeConfig(
        edge_type=EdgeType.LONDON_OPEN,
        enabled=True,
        weight=0.8,
        min_score_override=55,  # Higher threshold
        max_daily_signals=2,
        cooldown_minutes=30,
        notes="£4.5M profit, 53.6% WR - MARGINAL",
    ),
    EdgeType.BOLLINGER_TOUCH: EdgeConfig(
        edge_type=EdgeType.BOLLINGER_TOUCH,
        enabled=True,
        weight=0.7,
        min_score_override=60,  # Higher threshold
        max_daily_signals=3,
        cooldown_minutes=120,
        notes="£2.0M profit, 50.0% WR - USE WITH CAUTION",
    ),
    # === DISABLED (losing money in backtest) ===
    EdgeType.MONTH_END: EdgeConfig(
        edge_type=EdgeType.MONTH_END,
        enabled=False,
        weight=0.0,
        min_score_override=None,
        max_daily_signals=0,
        cooldown_minutes=0,
        notes="DISABLED - Lost £285K in backtest",
    ),
    EdgeType.NY_OPEN: EdgeConfig(
        edge_type=EdgeType.NY_OPEN,
        enabled=False,
        weight=0.0,
        min_score_override=None,
        max_daily_signals=0,
        cooldown_minutes=0,
        notes="DISABLED - Lost £316K in backtest",
    ),
    # === NOT BACKTESTED / DISABLED ===
    EdgeType.EARNINGS_DRIFT: EdgeConfig(
        edge_type=EdgeType.EARNINGS_DRIFT,
        enabled=False,
        weight=0.0,
        min_score_override=None,
        max_daily_signals=0,
        cooldown_minutes=0,
        notes="DISABLED - Not validated in backtest",
    ),
    EdgeType.SENTIMENT_SPIKE: EdgeConfig(
        edge_type=EdgeType.SENTIMENT_SPIKE,
        enabled=False,
        weight=0.0,
        min_score_override=None,
        max_daily_signals=0,
        cooldown_minutes=0,
        notes="DISABLED - Not validated in backtest",
    ),
}


def get_enabled_edges() -> List[EdgeType]:
    """Get list of enabled edges."""
    return [e for e, cfg in EDGE_CONFIGS.items() if cfg.enabled]


def get_edge_weight(edge: EdgeType) -> float:
    """Get weight for an edge."""
    cfg = EDGE_CONFIGS.get(edge)
    if cfg:
        return cfg.weight
    return 0.5  # Unknown edge default


def is_edge_enabled(edge: EdgeType) -> bool:
    """Check if edge is enabled."""
    cfg = EDGE_CONFIGS.get(edge)
    return cfg.enabled if cfg else False


def get_edge_cooldown(edge: EdgeType) -> int:
    """Get cooldown minutes for an edge."""
    cfg = EDGE_CONFIGS.get(edge)
    return cfg.cooldown_minutes if cfg else 60


def get_edge_max_daily(edge: EdgeType) -> int:
    """Get max daily signals for an edge."""
    cfg = EDGE_CONFIGS.get(edge)
    return cfg.max_daily_signals if cfg else 5
