"""
Market Regime Detection for GOD MODE Backtest

Detects current market regime (BULL, BEAR, SIDEWAYS, VOLATILE) and
adapts trading strategy accordingly.

Based on:
- 50/200 SMA relationship (trend direction)
- ADX (trend strength)
- VIX levels (volatility)
- Price vs 20 SMA (short-term momentum)

This module provides a 6-regime classification optimised for the GOD MODE
Monte-Carlo backtester.  It is intentionally separate from the simpler
4-regime ``nexus.intelligence.regime.RegimeDetector`` used in live scanning.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from nexus.core.enums import EdgeType, Direction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Enums & data-classes
# ═══════════════════════════════════════════════════════════════════════


class GodModeRegime(Enum):
    """Six-state market regime classification for GOD MODE."""

    STRONG_BULL = "strong_bull"   # Clear uptrend, low volatility
    BULL = "bull"                 # Uptrend
    SIDEWAYS = "sideways"         # Range-bound, no clear direction
    BEAR = "bear"                 # Downtrend
    STRONG_BEAR = "strong_bear"   # Crash / panic mode
    VOLATILE = "volatile"         # High volatility, unclear direction


@dataclass
class RegimeConfig:
    """Configuration for each regime."""

    regime: GodModeRegime
    position_size_multiplier: float   # Scale position sizes
    max_positions: int                # Max concurrent positions
    allowed_edges: List[EdgeType]     # Edges that work in this regime
    preferred_direction: Optional[Direction]  # Bias direction
    stop_multiplier: float            # Wider / tighter stops
    take_profit_multiplier: float     # Adjust targets
    max_heat: float                   # Portfolio heat limit
    description: str


# ═══════════════════════════════════════════════════════════════════════
# Regime-specific configurations
# ═══════════════════════════════════════════════════════════════════════

REGIME_CONFIGS: Dict[GodModeRegime, RegimeConfig] = {
    GodModeRegime.STRONG_BULL: RegimeConfig(
        regime=GodModeRegime.STRONG_BULL,
        position_size_multiplier=1.25,
        max_positions=12,
        allowed_edges=[
            EdgeType.TURN_OF_MONTH,
            EdgeType.OVERNIGHT_PREMIUM,
            EdgeType.GAP_FILL,
            EdgeType.ORB,
            EdgeType.POWER_HOUR,
            EdgeType.INSIDER_CLUSTER,
            EdgeType.MONTH_END,
            EdgeType.NY_OPEN,
            EdgeType.VWAP_DEVIATION,
        ],
        preferred_direction=Direction.LONG,
        stop_multiplier=1.0,
        take_profit_multiplier=1.5,
        max_heat=40.0,
        description="Strong uptrend - aggressive longs, trend following",
    ),
    GodModeRegime.BULL: RegimeConfig(
        regime=GodModeRegime.BULL,
        position_size_multiplier=1.0,
        max_positions=10,
        allowed_edges=[
            EdgeType.TURN_OF_MONTH,
            EdgeType.OVERNIGHT_PREMIUM,
            EdgeType.GAP_FILL,
            EdgeType.ORB,
            EdgeType.POWER_HOUR,
            EdgeType.INSIDER_CLUSTER,
            EdgeType.MONTH_END,
            EdgeType.NY_OPEN,
            EdgeType.LONDON_OPEN,
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
        ],
        preferred_direction=Direction.LONG,
        stop_multiplier=1.0,
        take_profit_multiplier=1.2,
        max_heat=35.0,
        description="Uptrend - favour longs, normal risk",
    ),
    GodModeRegime.SIDEWAYS: RegimeConfig(
        regime=GodModeRegime.SIDEWAYS,
        position_size_multiplier=0.9,
        max_positions=8,
        allowed_edges=[
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.GAP_FILL,
            EdgeType.ASIAN_RANGE,
            EdgeType.LONDON_OPEN,
            EdgeType.TURN_OF_MONTH,
            EdgeType.MONTH_END,
        ],
        preferred_direction=None,
        stop_multiplier=0.8,
        take_profit_multiplier=0.8,
        max_heat=30.0,
        description="Range-bound - mean reversion, quick profits",
    ),
    GodModeRegime.BEAR: RegimeConfig(
        regime=GodModeRegime.BEAR,
        position_size_multiplier=0.75,
        max_positions=6,
        allowed_edges=[
            EdgeType.RSI_EXTREME,
            EdgeType.VWAP_DEVIATION,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.GAP_FILL,
            EdgeType.TURN_OF_MONTH,
        ],
        preferred_direction=Direction.SHORT,
        stop_multiplier=1.2,
        take_profit_multiplier=1.0,
        max_heat=25.0,
        description="Downtrend - defensive, favour shorts, reduce exposure",
    ),
    GodModeRegime.STRONG_BEAR: RegimeConfig(
        regime=GodModeRegime.STRONG_BEAR,
        position_size_multiplier=0.5,
        max_positions=4,
        allowed_edges=[
            EdgeType.RSI_EXTREME,
            EdgeType.TURN_OF_MONTH,
            EdgeType.GAP_FILL,
        ],
        preferred_direction=Direction.SHORT,
        stop_multiplier=1.5,
        take_profit_multiplier=0.8,
        max_heat=15.0,
        description="Crash mode - capital preservation, minimal trading",
    ),
    GodModeRegime.VOLATILE: RegimeConfig(
        regime=GodModeRegime.VOLATILE,
        position_size_multiplier=0.6,
        max_positions=5,
        allowed_edges=[
            EdgeType.GAP_FILL,
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.ORB,
        ],
        preferred_direction=None,
        stop_multiplier=1.5,
        take_profit_multiplier=1.5,
        max_heat=20.0,
        description="High volatility - reduced size, wider stops/targets",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Detector
# ═══════════════════════════════════════════════════════════════════════


class GodModeRegimeDetector:
    """
    Detects market regime based on price action and volatility.

    Uses multiple indicators:
    - 50/200 SMA crossover (trend direction)
    - ADX proxy (trend strength)
    - VIX / ATR proxy (volatility)
    - 20-day rate of change (momentum)
    """

    # VIX / annualised-volatility thresholds
    VIX_LOW = 15
    VIX_NORMAL = 20
    VIX_HIGH = 25
    VIX_EXTREME = 35

    # ADX thresholds
    ADX_WEAK = 20
    ADX_STRONG = 30

    def __init__(self) -> None:
        self.current_regime = GodModeRegime.SIDEWAYS
        self.regime_history: List[Tuple[datetime, GodModeRegime]] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        prices: np.ndarray,
        vix: Optional[float] = None,
        date: Optional[datetime] = None,
    ) -> GodModeRegime:
        """Detect current market regime from a price array.

        Args:
            prices: Array of closing prices (most recent last).
            vix: Optional VIX value (if available).
            date: Current date for history tracking.

        Returns:
            GodModeRegime enum value.
        """
        if len(prices) < 200:
            return GodModeRegime.SIDEWAYS

        sma_20 = float(np.mean(prices[-20:]))
        sma_50 = float(np.mean(prices[-50:]))
        sma_200 = float(np.mean(prices[-200:]))
        current_price = float(prices[-1])

        # 20-day rate of change (%)
        roc_20 = (current_price - float(prices[-20])) / float(prices[-20]) * 100

        # Volatility: use VIX if provided, else annualised ATR proxy
        if vix is None:
            tr = np.abs(np.diff(prices[-15:]))
            atr = float(np.mean(tr))
            volatility = (atr / current_price) * 100 * 16  # ~annualised
        else:
            volatility = vix

        # ADX approximation
        up_moves = np.diff(prices[-15:])
        adx_proxy = abs(float(np.mean(up_moves))) / (float(np.std(up_moves)) + 1e-10) * 25

        regime = self._classify(
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            roc_20=roc_20,
            volatility=volatility,
            adx=adx_proxy,
        )

        self.current_regime = regime
        if date:
            self.regime_history.append((date, regime))
        return regime

    def get_config(self, regime: Optional[GodModeRegime] = None) -> RegimeConfig:
        """Get configuration for a regime."""
        if regime is None:
            regime = self.current_regime
        return REGIME_CONFIGS[regime]

    def is_edge_allowed(
        self,
        edge: EdgeType,
        regime: Optional[GodModeRegime] = None,
    ) -> bool:
        """Check if an edge is allowed in the given (or current) regime."""
        return edge in self.get_config(regime).allowed_edges

    def adjust_position_size(
        self,
        base_size: float,
        regime: Optional[GodModeRegime] = None,
    ) -> float:
        """Adjust position size based on regime."""
        return base_size * self.get_config(regime).position_size_multiplier

    def adjust_stops(
        self,
        base_stop_distance: float,
        regime: Optional[GodModeRegime] = None,
    ) -> float:
        """Adjust stop distance based on regime."""
        return base_stop_distance * self.get_config(regime).stop_multiplier

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _classify(
        self,
        current_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        roc_20: float,
        volatility: float,
        adx: float,
    ) -> GodModeRegime:
        # Extreme volatility first
        if volatility > self.VIX_EXTREME:
            return GodModeRegime.VOLATILE

        price_above_50 = current_price > sma_50
        price_above_200 = current_price > sma_200
        sma_50_above_200 = sma_50 > sma_200

        # Bull conditions
        if price_above_50 and price_above_200 and sma_50_above_200:
            if roc_20 > 5 and adx > self.ADX_STRONG and volatility < self.VIX_NORMAL:
                return GodModeRegime.STRONG_BULL
            return GodModeRegime.BULL

        # Bear conditions
        if not price_above_50 and not price_above_200 and not sma_50_above_200:
            if roc_20 < -5 and volatility > self.VIX_HIGH:
                return GodModeRegime.STRONG_BEAR
            return GodModeRegime.BEAR

        # High volatility with unclear direction
        if volatility > self.VIX_HIGH:
            return GodModeRegime.VOLATILE

        return GodModeRegime.SIDEWAYS


# ═══════════════════════════════════════════════════════════════════════
# Historical regime lookup (2020-2024)
# ═══════════════════════════════════════════════════════════════════════


def get_historical_regimes() -> Dict[str, GodModeRegime]:
    """Return historically accurate market regimes by month (YYYY-MM key).

    Based on actual S&P 500 performance.
    """
    return {
        # 2020 – COVID crash and recovery
        "2020-01": GodModeRegime.BULL,
        "2020-02": GodModeRegime.VOLATILE,
        "2020-03": GodModeRegime.STRONG_BEAR,
        "2020-04": GodModeRegime.VOLATILE,
        "2020-05": GodModeRegime.BULL,
        "2020-06": GodModeRegime.BULL,
        "2020-07": GodModeRegime.BULL,
        "2020-08": GodModeRegime.STRONG_BULL,
        "2020-09": GodModeRegime.VOLATILE,
        "2020-10": GodModeRegime.SIDEWAYS,
        "2020-11": GodModeRegime.STRONG_BULL,
        "2020-12": GodModeRegime.BULL,
        # 2021 – Bull market continues
        "2021-01": GodModeRegime.BULL,
        "2021-02": GodModeRegime.BULL,
        "2021-03": GodModeRegime.SIDEWAYS,
        "2021-04": GodModeRegime.BULL,
        "2021-05": GodModeRegime.SIDEWAYS,
        "2021-06": GodModeRegime.BULL,
        "2021-07": GodModeRegime.BULL,
        "2021-08": GodModeRegime.BULL,
        "2021-09": GodModeRegime.BEAR,
        "2021-10": GodModeRegime.STRONG_BULL,
        "2021-11": GodModeRegime.BULL,
        "2021-12": GodModeRegime.STRONG_BULL,
        # 2022 – Bear market
        "2022-01": GodModeRegime.BEAR,
        "2022-02": GodModeRegime.BEAR,
        "2022-03": GodModeRegime.VOLATILE,
        "2022-04": GodModeRegime.BEAR,
        "2022-05": GodModeRegime.BEAR,
        "2022-06": GodModeRegime.STRONG_BEAR,
        "2022-07": GodModeRegime.BULL,
        "2022-08": GodModeRegime.BEAR,
        "2022-09": GodModeRegime.STRONG_BEAR,
        "2022-10": GodModeRegime.VOLATILE,
        "2022-11": GodModeRegime.BULL,
        "2022-12": GodModeRegime.SIDEWAYS,
        # 2023 – Recovery and new bull
        "2023-01": GodModeRegime.BULL,
        "2023-02": GodModeRegime.SIDEWAYS,
        "2023-03": GodModeRegime.VOLATILE,
        "2023-04": GodModeRegime.BULL,
        "2023-05": GodModeRegime.BULL,
        "2023-06": GodModeRegime.STRONG_BULL,
        "2023-07": GodModeRegime.STRONG_BULL,
        "2023-08": GodModeRegime.SIDEWAYS,
        "2023-09": GodModeRegime.BEAR,
        "2023-10": GodModeRegime.VOLATILE,
        "2023-11": GodModeRegime.STRONG_BULL,
        "2023-12": GodModeRegime.STRONG_BULL,
        # 2024 – Continued bull
        "2024-01": GodModeRegime.BULL,
        "2024-02": GodModeRegime.BULL,
        "2024-03": GodModeRegime.BULL,
        "2024-04": GodModeRegime.SIDEWAYS,
        "2024-05": GodModeRegime.BULL,
        "2024-06": GodModeRegime.BULL,
        "2024-07": GodModeRegime.VOLATILE,
        "2024-08": GodModeRegime.VOLATILE,
        "2024-09": GodModeRegime.BULL,
        "2024-10": GodModeRegime.SIDEWAYS,
        "2024-11": GodModeRegime.STRONG_BULL,
        "2024-12": GodModeRegime.BULL,
    }
