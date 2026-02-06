"""
NEXUS Multi-Timeframe Trend Filter
Checks trend alignment across multiple timeframes.

KEY INSIGHT:
Trades with aligned signals across 2+ timeframes achieve 58% win rate
vs 39% for non-aligned trades. This is a SIGNIFICANT edge.

FRAMEWORK:
- Higher TF (Daily/Weekly): Trend direction
- Medium TF (4H/1H): Setup identification  
- Lower TF (15M/5M): Entry timing

RULE: Only trade when higher TF confirms direction.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import Direction


class TrendState(Enum):
    """Trend state for a single timeframe."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TrendAlignment(Enum):
    """Overall trend alignment across timeframes."""
    STRONG_BULLISH = "strong_bullish"   # All timeframes bullish
    STRONG_BEARISH = "strong_bearish"   # All timeframes bearish
    PARTIAL_BULLISH = "partial_bullish" # Higher TFs bullish, lower mixed
    PARTIAL_BEARISH = "partial_bearish" # Higher TFs bearish, lower mixed
    CONFLICTING = "conflicting"         # Timeframes disagree
    NEUTRAL = "neutral"                 # No clear trend


@dataclass
class TrendAnalysis:
    """Complete trend analysis across timeframes."""
    alignment: TrendAlignment
    direction: Optional[Direction]  # Suggested trade direction
    confidence: float  # 0-1
    can_trade: bool
    size_multiplier: float
    timeframes: Dict[str, TrendState]
    reason: str
    

class MultiTimeframeTrendFilter:
    """
    Check trend alignment across multiple timeframes.
    
    Uses SMA alignment to determine trend:
    - Price > SMA20 > SMA50 = Bullish
    - Price < SMA20 < SMA50 = Bearish
    - Otherwise = Neutral
    """
    
    # Timeframe hierarchy (higher = more important)
    TIMEFRAME_WEIGHTS = {
        "1W": 3.0,   # Weekly - strongest signal
        "1D": 2.5,   # Daily - very important
        "4H": 2.0,   # 4-hour
        "1H": 1.5,   # Hourly
        "30m": 1.0,  # 30-minute
        "15m": 0.75, # 15-minute
        "5m": 0.5,   # 5-minute - lowest weight
    }
    
    def __init__(
        self,
        timeframes: List[str] = None,
        require_higher_tf_alignment: bool = True
    ):
        """
        Initialize trend filter.
        
        Args:
            timeframes: List of timeframes to check (default: ["1D", "4H", "1H"])
            require_higher_tf_alignment: If True, only allow trades aligned with highest TF
        """
        self.timeframes = timeframes or ["1D", "4H", "1H"]
        self.require_higher_tf_alignment = require_higher_tf_alignment
    
    def analyze(
        self,
        price_data: Dict[str, Any],  # Dict of timeframe -> DataFrame
        symbol: str = "UNKNOWN"
    ) -> TrendAnalysis:
        """
        Analyze trend across multiple timeframes.
        
        Args:
            price_data: Dict mapping timeframe string to DataFrame with OHLCV
                       e.g., {"1D": daily_df, "4H": h4_df, "1H": h1_df}
            symbol: Symbol being analyzed
        
        Returns:
            TrendAnalysis with alignment, direction, and confidence
        """
        timeframe_trends = {}
        
        # Analyze each timeframe
        for tf in self.timeframes:
            if tf in price_data and price_data[tf] is not None:
                df = price_data[tf]
                trend = self._get_trend(df)
                timeframe_trends[tf] = trend
            else:
                # If data missing, mark as neutral
                timeframe_trends[tf] = TrendState.NEUTRAL
        
        # Determine overall alignment
        alignment, direction, confidence = self._calculate_alignment(timeframe_trends)
        
        # Determine if we can trade and size multiplier
        can_trade, size_mult, reason = self._get_trade_permission(alignment, direction, timeframe_trends)
        
        return TrendAnalysis(
            alignment=alignment,
            direction=direction,
            confidence=confidence,
            can_trade=can_trade,
            size_multiplier=size_mult,
            timeframes=timeframe_trends,
            reason=reason
        )
    
    def _get_trend(self, df: Any) -> TrendState:
        """
        Determine trend state from price data.
        
        Uses SMA alignment:
        - Price > SMA20 > SMA50 = Bullish
        - Price < SMA20 < SMA50 = Bearish
        - Otherwise = Neutral
        """
        if df is None or len(df) < 50:
            return TrendState.NEUTRAL
        
        close = df['close']
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return TrendState.BULLISH
        elif current_price < sma_20 < sma_50:
            return TrendState.BEARISH
        else:
            return TrendState.NEUTRAL
    
    def _calculate_alignment(
        self,
        timeframe_trends: Dict[str, TrendState]
    ) -> tuple:
        """
        Calculate overall alignment from individual timeframe trends.
        
        Returns:
            (alignment, direction, confidence)
        """
        bullish_count = sum(1 for t in timeframe_trends.values() if t == TrendState.BULLISH)
        bearish_count = sum(1 for t in timeframe_trends.values() if t == TrendState.BEARISH)
        neutral_count = sum(1 for t in timeframe_trends.values() if t == TrendState.NEUTRAL)
        total = len(timeframe_trends)
        
        if total == 0:
            return TrendAlignment.NEUTRAL, None, 0.0
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for tf, trend in timeframe_trends.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 1.0)
            total_weight += weight
            
            if trend == TrendState.BULLISH:
                weighted_score += weight
            elif trend == TrendState.BEARISH:
                weighted_score -= weight
        
        # Normalize to -1 to 1
        normalized_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine alignment
        if bullish_count == total:
            alignment = TrendAlignment.STRONG_BULLISH
            direction = Direction.LONG
            confidence = 0.9
        elif bearish_count == total:
            alignment = TrendAlignment.STRONG_BEARISH
            direction = Direction.SHORT
            confidence = 0.9
        elif normalized_score > 0.5:
            alignment = TrendAlignment.PARTIAL_BULLISH
            direction = Direction.LONG
            confidence = 0.5 + normalized_score * 0.3
        elif normalized_score < -0.5:
            alignment = TrendAlignment.PARTIAL_BEARISH
            direction = Direction.SHORT
            confidence = 0.5 + abs(normalized_score) * 0.3
        elif bullish_count > 0 and bearish_count > 0:
            alignment = TrendAlignment.CONFLICTING
            direction = None
            confidence = 0.3
        else:
            alignment = TrendAlignment.NEUTRAL
            direction = None
            confidence = 0.4
        
        return alignment, direction, confidence
    
    def _get_trade_permission(
        self,
        alignment: TrendAlignment,
        direction: Optional[Direction],
        timeframe_trends: Dict[str, TrendState]
    ) -> tuple:
        """
        Determine if trading is allowed and position size multiplier.
        
        Returns:
            (can_trade, size_multiplier, reason)
        """
        # Check highest timeframe alignment if required
        if self.require_higher_tf_alignment:
            # Get highest available timeframe
            highest_tf = None
            for tf in ["1W", "1D", "4H", "1H", "30m", "15m", "5m"]:
                if tf in timeframe_trends:
                    highest_tf = tf
                    break
            
            if highest_tf:
                highest_trend = timeframe_trends[highest_tf]
                
                if highest_trend == TrendState.NEUTRAL:
                    # Highest TF neutral - reduce size but allow
                    return True, 0.75, f"Highest TF ({highest_tf}) neutral - reduced size"
                
                if alignment == TrendAlignment.CONFLICTING:
                    return False, 0.0, "Timeframes conflicting - no trade"
        
        # Alignment-based permissions
        if alignment == TrendAlignment.STRONG_BULLISH:
            return True, 1.0, "Strong bullish alignment - full size"
        elif alignment == TrendAlignment.STRONG_BEARISH:
            return True, 1.0, "Strong bearish alignment - full size"
        elif alignment == TrendAlignment.PARTIAL_BULLISH:
            return True, 0.75, "Partial bullish alignment - reduced size"
        elif alignment == TrendAlignment.PARTIAL_BEARISH:
            return True, 0.75, "Partial bearish alignment - reduced size"
        elif alignment == TrendAlignment.CONFLICTING:
            return False, 0.0, "Conflicting timeframes - no trade"
        else:
            return True, 0.5, "Neutral/unclear trend - half size"
    
    def check_trade_alignment(
        self,
        price_data: Dict[str, Any],
        trade_direction: Direction
    ) -> Dict:
        """
        Check if a specific trade direction aligns with the trend.
        
        Args:
            price_data: Dict of timeframe -> DataFrame
            trade_direction: The intended trade direction (LONG or SHORT)
        
        Returns:
            Dict with alignment info and whether trade is aligned
        """
        analysis = self.analyze(price_data)
        
        is_aligned = False
        alignment_score = 0.0
        
        if analysis.direction == trade_direction:
            is_aligned = True
            alignment_score = analysis.confidence
        elif analysis.direction is None:
            # Neutral - partial alignment
            is_aligned = True
            alignment_score = 0.5
        else:
            # Trading against trend
            is_aligned = False
            alignment_score = 0.0
        
        return {
            "aligned": is_aligned,
            "alignment_score": alignment_score,
            "trend_direction": analysis.direction.value if analysis.direction else "neutral",
            "trade_direction": trade_direction.value,
            "alignment": analysis.alignment.value,
            "can_trade": analysis.can_trade and is_aligned,
            "size_multiplier": analysis.size_multiplier if is_aligned else 0.5,
            "reason": analysis.reason
        }
    
    def get_alignment_dict(self, analysis: TrendAnalysis) -> Dict:
        """
        Convert TrendAnalysis to dict format expected by scorer.
        
        Returns dict with 'alignment' key for scorer compatibility.
        """
        alignment_map = {
            TrendAlignment.STRONG_BULLISH: "STRONG_BULLISH",
            TrendAlignment.STRONG_BEARISH: "STRONG_BEARISH",
            TrendAlignment.PARTIAL_BULLISH: "PARTIAL",
            TrendAlignment.PARTIAL_BEARISH: "PARTIAL",
            TrendAlignment.CONFLICTING: "CONFLICTING",
            TrendAlignment.NEUTRAL: "NONE",
        }
        
        return {
            "alignment": alignment_map.get(analysis.alignment, "NONE"),
            "direction": analysis.direction.value if analysis.direction else None,
            "confidence": analysis.confidence,
            "can_trade": analysis.can_trade,
            "size_multiplier": analysis.size_multiplier
        }


class TrendFilter:
    """
    Convenience wrapper for the orchestrator.
    get_trend_alignment(daily_trend, h4_trend, h1_trend) returns TrendAnalysis
    without requiring price data.
    """

    def get_trend_alignment(
        self,
        daily_trend: str = "neutral",
        h4_trend: str = "neutral",
        h1_trend: str = "neutral",
    ) -> TrendAnalysis:
        """Map string trends to TrendAnalysis for use in signal generation."""
        mapping = {
            "bullish": TrendState.BULLISH,
            "bearish": TrendState.BEARISH,
            "neutral": TrendState.NEUTRAL,
        }
        d = mapping.get((daily_trend or "").lower(), TrendState.NEUTRAL)
        h4 = mapping.get((h4_trend or "").lower(), TrendState.NEUTRAL)
        h1 = mapping.get((h1_trend or "").lower(), TrendState.NEUTRAL)

        if d == h4 == h1 == TrendState.BULLISH:
            alignment = TrendAlignment.STRONG_BULLISH
            direction = Direction.LONG
            confidence = 0.9
        elif d == h4 == h1 == TrendState.BEARISH:
            alignment = TrendAlignment.STRONG_BEARISH
            direction = Direction.SHORT
            confidence = 0.9
        elif d == TrendState.BULLISH and (h4 == TrendState.BULLISH or h1 == TrendState.BULLISH):
            alignment = TrendAlignment.PARTIAL_BULLISH
            direction = Direction.LONG
            confidence = 0.6
        elif d == TrendState.BEARISH and (h4 == TrendState.BEARISH or h1 == TrendState.BEARISH):
            alignment = TrendAlignment.PARTIAL_BEARISH
            direction = Direction.SHORT
            confidence = 0.6
        elif (TrendState.BULLISH in (d, h4, h1)) and (TrendState.BEARISH in (d, h4, h1)):
            alignment = TrendAlignment.CONFLICTING
            direction = None
            confidence = 0.3
        else:
            alignment = TrendAlignment.NEUTRAL
            direction = None
            confidence = 0.4

        can_trade = alignment != TrendAlignment.CONFLICTING
        size_multiplier = 0.75 if alignment == TrendAlignment.NEUTRAL else 1.0

        return TrendAnalysis(
            alignment=alignment,
            direction=direction,
            confidence=confidence,
            can_trade=can_trade,
            size_multiplier=size_multiplier,
            timeframes={"1D": d, "4H": h4, "1H": h1},
            reason="From daily/h4/h1",
        )


# Test the trend filter
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    print("=" * 60)
    print("NEXUS MULTI-TIMEFRAME TREND FILTER TEST")
    print("=" * 60)
    
    filter = MultiTimeframeTrendFilter(timeframes=["1D", "4H", "1H"])
    
    # Helper to create test data
    def create_trend_data(trend: str, days: int = 60):
        """Create synthetic price data with specific trend."""
        np.random.seed(42)
        
        if trend == "bullish":
            # Price climbing: Price > SMA20 > SMA50
            base = 100 + np.arange(days) * 0.8
        elif trend == "bearish":
            # Price falling: Price < SMA20 < SMA50
            base = 200 - np.arange(days) * 0.8
        else:
            # Ranging/neutral
            base = 100 + np.sin(np.arange(days) * 0.2) * 3
        
        noise = np.random.randn(days) * 0.5
        close = base + noise
        
        df = pd.DataFrame({
            'open': close - np.random.rand(days) * 0.5,
            'high': close + np.abs(np.random.randn(days)) * 1,
            'low': close - np.abs(np.random.randn(days)) * 1,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, days)
        })
        
        return df
    
    # Test 1: All timeframes bullish
    print("\n--- Test 1: All Timeframes Bullish ---")
    price_data = {
        "1D": create_trend_data("bullish"),
        "4H": create_trend_data("bullish"),
        "1H": create_trend_data("bullish"),
    }
    
    result = filter.analyze(price_data, "TEST")
    print(f"Alignment: {result.alignment.value}")
    print(f"Direction: {result.direction.value if result.direction else 'None'}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Can Trade: {result.can_trade}")
    print(f"Size Multiplier: {result.size_multiplier}x")
    print(f"Timeframes: {[(tf, t.value) for tf, t in result.timeframes.items()]}")
    
    # Test 2: All timeframes bearish
    print("\n--- Test 2: All Timeframes Bearish ---")
    price_data = {
        "1D": create_trend_data("bearish"),
        "4H": create_trend_data("bearish"),
        "1H": create_trend_data("bearish"),
    }
    
    result = filter.analyze(price_data, "TEST")
    print(f"Alignment: {result.alignment.value}")
    print(f"Direction: {result.direction.value if result.direction else 'None'}")
    print(f"Can Trade: {result.can_trade}")
    
    # Test 3: Conflicting timeframes
    print("\n--- Test 3: Conflicting Timeframes ---")
    price_data = {
        "1D": create_trend_data("bullish"),
        "4H": create_trend_data("bearish"),
        "1H": create_trend_data("bullish"),
    }
    
    result = filter.analyze(price_data, "TEST")
    print(f"Alignment: {result.alignment.value}")
    print(f"Direction: {result.direction.value if result.direction else 'None'}")
    print(f"Can Trade: {result.can_trade}")
    print(f"Reason: {result.reason}")
    
    # Test 4: Check trade alignment
    print("\n--- Test 4: Check Trade Alignment ---")
    price_data = {
        "1D": create_trend_data("bullish"),
        "4H": create_trend_data("bullish"),
        "1H": create_trend_data("neutral"),
    }
    
    # Try LONG trade (should align)
    long_check = filter.check_trade_alignment(price_data, Direction.LONG)
    print(f"LONG trade aligned: {long_check['aligned']}")
    print(f"Alignment score: {long_check['alignment_score']:.2f}")
    
    # Try SHORT trade (should not align)
    short_check = filter.check_trade_alignment(price_data, Direction.SHORT)
    print(f"SHORT trade aligned: {short_check['aligned']}")
    print(f"Size multiplier: {short_check['size_multiplier']}x")
    
    # Test 5: Get alignment dict for scorer
    print("\n--- Test 5: Scorer-Compatible Format ---")
    result = filter.analyze(price_data, "TEST")
    scorer_dict = filter.get_alignment_dict(result)
    print(f"Scorer format: {scorer_dict}")
    
    print("\n" + "=" * 60)
    print("TREND FILTER TEST COMPLETE [OK]")
    print("=" * 60)
