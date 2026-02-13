"""
Timeframe Configuration - Weights based on backtest performance.

Backtest Results (2020-2024):
- 4H: £48.4M, 64.2% WR  BEST
- 1H: £40.7M, 62.9% WR  STRONG
- 1D: £34.1M, 66.4% WR  STRONG (highest WR)
- 15M: £13.0M, 54.3% WR  WEAK (10% lower WR)
"""

from dataclasses import dataclass
from typing import Dict, List

from nexus.core.enums import Timeframe


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe."""

    timeframe: Timeframe
    enabled: bool
    weight: float  # Affects signal scoring
    min_score_bonus: int  # Added to signal score
    notes: str


TIMEFRAME_CONFIGS: Dict[Timeframe, TimeframeConfig] = {
    Timeframe.M5: TimeframeConfig(
        timeframe=Timeframe.M5,
        enabled=False,
        weight=0.0,
        min_score_bonus=-10,
        notes="DISABLED - Too noisy, not backtested",
    ),
    Timeframe.M15: TimeframeConfig(
        timeframe=Timeframe.M15,
        enabled=True,
        weight=0.6,
        min_score_bonus=-5,
        notes="WEAK - 54.3% WR, 10% below others",
    ),
    Timeframe.M30: TimeframeConfig(
        timeframe=Timeframe.M30,
        enabled=True,
        weight=0.8,
        min_score_bonus=0,
        notes="Not separately tested - interpolated",
    ),
    Timeframe.H1: TimeframeConfig(
        timeframe=Timeframe.H1,
        enabled=True,
        weight=1.0,
        min_score_bonus=0,
        notes="STRONG - 62.9% WR, £40.7M profit",
    ),
    Timeframe.H4: TimeframeConfig(
        timeframe=Timeframe.H4,
        enabled=True,
        weight=1.2,
        min_score_bonus=+5,
        notes="BEST - 64.2% WR, £48.4M profit",
    ),
    Timeframe.D1: TimeframeConfig(
        timeframe=Timeframe.D1,
        enabled=True,
        weight=1.1,
        min_score_bonus=+3,
        notes="STRONG - 66.4% WR (highest), £34.1M profit",
    ),
}


def get_enabled_timeframes() -> List[Timeframe]:
    """Get list of enabled timeframes."""
    return [tf for tf, cfg in TIMEFRAME_CONFIGS.items() if cfg.enabled]


def get_timeframe_weight(tf: Timeframe) -> float:
    """Get weight for a timeframe."""
    cfg = TIMEFRAME_CONFIGS.get(tf)
    return cfg.weight if cfg else 0.8


def get_timeframe_score_bonus(tf: Timeframe) -> int:
    """Get score bonus/penalty for a timeframe."""
    cfg = TIMEFRAME_CONFIGS.get(tf)
    return cfg.min_score_bonus if cfg else 0
