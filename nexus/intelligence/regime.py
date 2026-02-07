"""
NEXUS Regime Detector
Detects market regime to adapt strategy selection.

REGIMES:
- TRENDING_UP: Momentum strategies work, mean reversion risky
- TRENDING_DOWN: Defensive, mean reversion shorts
- RANGING: Mean reversion works best (Bollinger, VWAP, RSI)
- VOLATILE: Reduce size, fewer trades, only highest conviction

KEY INDICATORS:
- ADX: Trend strength (>25 = trending, <20 = ranging)
- Bollinger Band Width: Volatility compression/expansion
- SMA Alignment: 20 SMA vs 50 SMA vs price position
- ATR: Absolute volatility level
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.core.enums import EdgeType


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeAnalysis:
    """Analysis results for regime detection."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in the regime call
    adx: float
    trend_direction: str  # "bullish", "bearish", "neutral"
    volatility_state: str  # "high", "normal", "low"
    indicators: Dict[str, float]
    allowed_edges: List[EdgeType]
    size_multiplier: float
    notes: List[str]


class RegimeDetector:
    """
    Detect market regime to adapt strategy selection.

    Uses multiple indicators:
    - ADX for trend strength
    - Bollinger Band width for volatility
    - SMA alignment for direction
    - ATR for volatility level
    """

    # Which edges work in each regime
    REGIME_EDGES = {
        MarketRegime.TRENDING_UP: [
            EdgeType.TURN_OF_MONTH,
            EdgeType.MONTH_END,
            EdgeType.INSIDER_CLUSTER,
            EdgeType.ORB,
            EdgeType.POWER_HOUR,
            EdgeType.NY_OPEN,
        ],
        MarketRegime.TRENDING_DOWN: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.GAP_FILL,
            EdgeType.INSIDER_CLUSTER,
        ],
        MarketRegime.RANGING: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.GAP_FILL,
            EdgeType.LONDON_OPEN,
        ],
        MarketRegime.VOLATILE: [
            # Very selective in volatile regime - only strongest edges
            EdgeType.INSIDER_CLUSTER,
            EdgeType.TURN_OF_MONTH,
        ],
    }

    # Position size multipliers by regime
    REGIME_SIZE_MULTIPLIERS = {
        MarketRegime.TRENDING_UP: 1.0,    # Full size
        MarketRegime.TRENDING_DOWN: 0.8,  # Slightly reduced
        MarketRegime.RANGING: 1.0,        # Full size
        MarketRegime.VOLATILE: 0.5,       # Half size - protect capital
    }

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        volatility_high_mult: float = 1.5,
        volatility_low_mult: float = 0.8
    ):
        """
        Initialize regime detector.

        Args:
            adx_trend_threshold: ADX above this = trending
            adx_range_threshold: ADX below this = ranging
            volatility_high_mult: ATR > avg * this = high volatility
            volatility_low_mult: ATR < avg * this = low volatility
        """
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.volatility_high_mult = volatility_high_mult
        self.volatility_low_mult = volatility_low_mult

    def detect_regime(
        self,
        prices: Any,  # DataFrame with OHLCV data
        symbol: str = "SPY"
    ) -> RegimeAnalysis:
        """
        Detect current market regime from price data.

        Args:
            prices: DataFrame with columns: open, high, low, close, volume
                   Should have at least 60 rows for reliable detection
            symbol: Symbol being analyzed (for logging)

        Returns:
            RegimeAnalysis with regime, confidence, and allowed edges
        """
        notes = []

        # Calculate indicators
        close = prices['close']
        high = prices['high']
        low = prices['low']

        # Simple Moving Averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        current_price = close.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]

        # ADX calculation (simplified)
        adx = self._calculate_adx(high, low, close, period=14)
        current_adx = adx.iloc[-1] if len(adx) > 0 else 20

        # ATR for volatility
        atr = self._calculate_atr(high, low, close, period=14)
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(50).mean().iloc[-1] if len(atr) >= 50 else current_atr

        # Bollinger Band width for volatility compression
        bb_width = self._calculate_bb_width(close, period=20, std_mult=2)
        current_bb_width = bb_width.iloc[-1]
        avg_bb_width = bb_width.rolling(50).mean().iloc[-1] if len(bb_width) >= 50 else current_bb_width

        # Store indicators
        indicators = {
            "price": current_price,
            "sma_20": current_sma_20,
            "sma_50": current_sma_50,
            "adx": current_adx,
            "atr": current_atr,
            "atr_avg": avg_atr,
            "bb_width": current_bb_width,
            "bb_width_avg": avg_bb_width,
        }

        # Determine trend direction from SMA alignment
        if current_price > current_sma_20 > current_sma_50:
            trend_direction = "bullish"
            notes.append("Price > SMA20 > SMA50: Bullish alignment")
        elif current_price < current_sma_20 < current_sma_50:
            trend_direction = "bearish"
            notes.append("Price < SMA20 < SMA50: Bearish alignment")
        else:
            trend_direction = "neutral"
            notes.append("SMAs not aligned: Neutral/transitioning")

        # Determine volatility state
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        if atr_ratio > self.volatility_high_mult:
            volatility_state = "high"
            notes.append(f"ATR {atr_ratio:.2f}x average: High volatility")
        elif atr_ratio < self.volatility_low_mult:
            volatility_state = "low"
            notes.append(f"ATR {atr_ratio:.2f}x average: Low volatility")
        else:
            volatility_state = "normal"

        # REGIME DETECTION LOGIC
        confidence = 0.5  # Start neutral

        # Check for high volatility FIRST (overrides other regimes)
        if volatility_state == "high":
            regime = MarketRegime.VOLATILE
            confidence = min(0.9, 0.5 + (atr_ratio - 1.5) * 0.3)
            notes.append("VOLATILE regime: High ATR detected")

        # Check for trending (ADX > threshold)
        elif current_adx > self.adx_trend_threshold:
            if trend_direction == "bullish":
                regime = MarketRegime.TRENDING_UP
                confidence = min(0.9, 0.5 + (current_adx - 25) * 0.02)
                notes.append(f"TRENDING UP: ADX {current_adx:.1f} with bullish alignment")
            elif trend_direction == "bearish":
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(0.9, 0.5 + (current_adx - 25) * 0.02)
                notes.append(f"TRENDING DOWN: ADX {current_adx:.1f} with bearish alignment")
            else:
                # Strong ADX but no clear direction - treat as ranging
                regime = MarketRegime.RANGING
                confidence = 0.5
                notes.append(f"ADX strong ({current_adx:.1f}) but direction unclear")

        # Check for ranging (ADX < threshold OR BB squeeze)
        elif current_adx < self.adx_range_threshold:
            regime = MarketRegime.RANGING
            confidence = min(0.9, 0.5 + (20 - current_adx) * 0.03)
            notes.append(f"RANGING: ADX {current_adx:.1f} below threshold")

        # Check for Bollinger squeeze (volatility compression)
        elif current_bb_width < avg_bb_width * 0.8:
            regime = MarketRegime.RANGING
            confidence = 0.7
            notes.append("RANGING: Bollinger Band squeeze detected")

        # Default to ranging if unclear
        else:
            regime = MarketRegime.RANGING
            confidence = 0.5
            notes.append("Defaulting to RANGING regime")

        # Get allowed edges and size multiplier
        allowed_edges = self.get_allowed_edges(regime)
        size_multiplier = self.get_size_multiplier(regime)

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            adx=current_adx,
            trend_direction=trend_direction,
            volatility_state=volatility_state,
            indicators=indicators,
            allowed_edges=allowed_edges,
            size_multiplier=size_multiplier,
            notes=notes
        )

    def _calculate_adx(self, high, low, close, period: int = 14):
        """Calculate Average Directional Index (simplified)."""
        import pandas as pd

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(period).mean()

        return adx.fillna(20)  # Default to 20 if not enough data

    def _calculate_atr(self, high, low, close, period: int = 14):
        """Calculate Average True Range."""
        import pandas as pd

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr.fillna(tr)

    def _calculate_bb_width(self, close, period: int = 20, std_mult: float = 2):
        """Calculate Bollinger Band width as percentage."""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)

        width = (upper - lower) / sma * 100  # Width as percentage

        return width.fillna(width.mean() if len(width) > 0 else 2.0)

    def get_allowed_edges(self, regime: MarketRegime) -> List[EdgeType]:
        """Get edges allowed in the given regime."""
        return self.REGIME_EDGES.get(regime, [])

    def get_size_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier for the regime."""
        return self.REGIME_SIZE_MULTIPLIERS.get(regime, 1.0)

    def is_edge_allowed(self, edge: EdgeType, regime: MarketRegime) -> bool:
        """Check if a specific edge is allowed in the regime."""
        return edge in self.get_allowed_edges(regime)


# Test the regime detector
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("NEXUS REGIME DETECTOR TEST")
    print("=" * 60)

    detector = RegimeDetector()

    # Helper to create test data
    def create_test_data(trend: str, volatility: str, days: int = 60):
        """Create synthetic price data for testing."""
        np.random.seed(42)

        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')

        if trend == "up":
            # Uptrend: price climbing steadily
            base = 100 + np.arange(days) * 0.5  # 0.5 per day uptrend
        elif trend == "down":
            # Downtrend: price falling
            base = 150 - np.arange(days) * 0.5
        else:
            # Ranging: oscillating around mean
            base = 100 + np.sin(np.arange(days) * 0.3) * 5

        if volatility == "high":
            noise_mult = 3.0
        elif volatility == "low":
            noise_mult = 0.5
        else:
            noise_mult = 1.0

        noise = np.random.randn(days) * noise_mult
        close = base + noise

        # Generate OHLC from close
        high = close + np.abs(np.random.randn(days)) * noise_mult
        low = close - np.abs(np.random.randn(days)) * noise_mult
        open_price = close.shift(1) if hasattr(close, 'shift') else np.roll(close, 1)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, days)
        }, index=dates)

        df.loc[df.index[0], 'open'] = df['close'].iloc[0]

        return df

    # Test 1: Trending Up
    print("\n--- Test 1: Trending Up Market ---")
    prices_up = create_test_data(trend="up", volatility="normal")
    result = detector.detect_regime(prices_up, "TEST")

    print(f"Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"ADX: {result.adx:.1f}")
    print(f"Trend Direction: {result.trend_direction}")
    print(f"Size Multiplier: {result.size_multiplier}x")
    print(f"Allowed Edges: {[e.value for e in result.allowed_edges]}")

    # Test 2: Trending Down
    print("\n--- Test 2: Trending Down Market ---")
    prices_down = create_test_data(trend="down", volatility="normal")
    result = detector.detect_regime(prices_down, "TEST")

    print(f"Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Trend Direction: {result.trend_direction}")
    print(f"Allowed Edges: {[e.value for e in result.allowed_edges]}")

    # Test 3: Ranging Market
    print("\n--- Test 3: Ranging Market ---")
    prices_range = create_test_data(trend="range", volatility="low")
    result = detector.detect_regime(prices_range, "TEST")

    print(f"Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"ADX: {result.adx:.1f}")
    print(f"Allowed Edges: {[e.value for e in result.allowed_edges]}")

    # Test 4: Volatile Market
    print("\n--- Test 4: Volatile Market ---")
    prices_volatile = create_test_data(trend="range", volatility="high")
    result = detector.detect_regime(prices_volatile, "TEST")

    print(f"Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Volatility State: {result.volatility_state}")
    print(f"Size Multiplier: {result.size_multiplier}x")
    print(f"Allowed Edges: {[e.value for e in result.allowed_edges]}")
    print("Notes:")
    for note in result.notes:
        print(f"  - {note}")

    # Test 5: Edge filtering
    print("\n--- Test 5: Edge Filtering by Regime ---")
    for regime in MarketRegime:
        edges = detector.get_allowed_edges(regime)
        mult = detector.get_size_multiplier(regime)
        print(f"{regime.value}: {len(edges)} edges allowed, {mult}x size")

    # Test 6: Check specific edge
    print("\n--- Test 6: Edge Compatibility Check ---")
    test_cases = [
        (EdgeType.BOLLINGER_TOUCH, MarketRegime.RANGING, True),
        (EdgeType.BOLLINGER_TOUCH, MarketRegime.TRENDING_UP, False),
        (EdgeType.INSIDER_CLUSTER, MarketRegime.VOLATILE, True),
        (EdgeType.ORB, MarketRegime.VOLATILE, False),
    ]

    for edge, regime, expected in test_cases:
        result = detector.is_edge_allowed(edge, regime)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {edge.value} in {regime.value}: {result} [{status}]")

    print("\n" + "=" * 60)
    print("REGIME DETECTOR TEST COMPLETE [OK]")
    print("=" * 60)
