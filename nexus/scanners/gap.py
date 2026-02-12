"""
Gap and Go Scanner - trade WITH gap momentum on high-volume breakaway gaps.

Ported from validated backtest logic (v13-v26): _signal_gap_fill in engine.py.
PF 1.41 (scored), +$5,991 on $10k (2022-2024), 152 trades, 51.3% WR.

Key insight: breakaway gaps with high volume DON'T fill — they continue.
The old fade strategy (SHORT on gap up) lost money. This trades WITH the gap.

Filters: gap 1-5%, volume 150%+, day confirms gap direction, 20 SMA trend.
Symbols: 10 validated high-beta stocks (mega caps and intl ETFs FAIL).
"""

import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class GapScanner(BaseScanner):
    """
    Gap and Go scanner - trade WITH gap momentum, not against it.

    VALIDATED: 152 trades, PF 1.41, +$5,991, Sharpe 2.49 (2022-2024)
    SIGNAL: Gap 1-5% + volume 150%+ → trade in gap direction
    DIRECTION: Gap UP → LONG, Gap DOWN → SHORT (momentum continuation)

    Ported from BacktestEngine._signal_gap_fill() to ensure live matches backtest.
    """

    # Validated symbol list — only high-beta stocks that sustain gap momentum.
    # Mega caps (GOOGL/META/MSFT/AMZN) FAIL: they fill gaps (PF 1.06).
    # International ETFs FAIL: gap-and-fill pattern (PF 0.91).
    # Leveraged ETFs FAIL: 3x amplifies noise (PF 0.80).
    INSTRUMENTS = {
        Market.US_STOCKS: [
            "SPY", "NVDA", "TSLA", "AAPL", "AMD",
            "COIN", "ROKU", "SHOP", "SQ", "MARA",
        ],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.GAP_FILL
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # Validated parameters (do not change without re-backtesting)
        self.min_gap_pct = 1.0    # Minimum gap size (%)
        self.max_gap_pct = 5.0    # Maximum gap size (%)
        self.min_volume_ratio = 1.5  # 150% of 20-day average volume
        self.notional_pct = 16    # Position size (% of capital)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Gap scanner runs on daily bars — active during US market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for Gap and Go opportunities.

        Logic (validated v13-v26):
        1. Gap must be 1-5% from prior close
        2. Volume must be 150%+ of 20-day average (CRITICAL)
        3. Day must close in gap direction (confirmation)
        4. Trade WITH gap direction (not fade)
        5. Dynamic scoring based on gap size + volume + trend
        """
        opportunities = []

        for market in self.markets:
            instruments = self._get_instruments(market)

            for symbol in instruments:
                try:
                    opp = await self._scan_symbol(symbol, market)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.warning(f"Gap scan failed for {symbol}: {e}")

        logger.info(f"Gap scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for Gap and Go opportunity.

        Ported from BacktestEngine._signal_gap_fill() — validated logic.
        """
        # Need 21 bars: 20 for volume average + current bar
        bars = await self.get_bars_safe(symbol, "1D", 25)

        if bars is None or len(bars) < 21:
            logger.debug(f"Gap insufficient data for {symbol}")
            return None

        cur = bars.iloc[-1]
        prev = bars.iloc[-2]

        # Calculate gap percentage
        if prev["close"] == 0:
            return None
        gap_pct = ((cur["open"] - prev["close"]) / prev["close"]) * 100

        # FILTER 1: Gap size must be 1-5%
        if abs(gap_pct) < self.min_gap_pct or abs(gap_pct) > self.max_gap_pct:
            return None

        # FILTER 2: Volume confirmation (CRITICAL — this is what makes it work)
        if "volume" not in bars.columns:
            return None

        avg_volume = bars["volume"].iloc[-21:-1].mean()  # 20-day avg excluding today
        cur_volume = cur["volume"]
        volume_ratio = cur_volume / avg_volume if avg_volume > 0 else 0

        if volume_ratio < self.min_volume_ratio:
            return None

        # FILTER 3: Day must close in gap direction (confirmation)
        day_move = cur["close"] - cur["open"]

        # Trend alignment: 20-day SMA direction matches gap direction
        sma_20 = bars["close"].rolling(20).mean().iloc[-1]
        trend_aligned = False
        if not pd.isna(sma_20):
            if gap_pct > 0 and cur["close"] > sma_20:
                trend_aligned = True
            elif gap_pct < 0 and cur["close"] < sma_20:
                trend_aligned = True

        # Dynamic score based on signal quality
        score = self._calculate_gap_score(gap_pct, volume_ratio, trend_aligned)

        # GAP UP + VOLUME → LONG (momentum continuation)
        if gap_pct >= self.min_gap_pct:
            if day_move < 0:  # Gap up but closed red = filling, skip
                return None

            entry = float(cur["close"])
            stop = float(prev["close"]) * 0.995  # 0.5% below prior close
            risk = entry - stop
            target = entry + (risk * 2.0)  # 2:1 R:R
            direction = Direction.LONG
            gap_direction = "up"

        # GAP DOWN + VOLUME → SHORT (momentum continuation)
        elif gap_pct <= -self.min_gap_pct:
            if day_move > 0:  # Gap down but closed green = filling, skip
                return None

            entry = float(cur["close"])
            stop = float(prev["close"]) * 1.005  # 0.5% above prior close
            risk = stop - entry
            target = entry - (risk * 2.0)  # 2:1 R:R
            direction = Direction.SHORT
            gap_direction = "down"

        else:
            return None

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "strategy": "gap_and_go",
                "gap_pct": round(gap_pct, 2),
                "volume_ratio": round(volume_ratio, 2),
                "day_move_pct": round((day_move / cur["open"]) * 100, 2),
                "gap_direction": gap_direction,
                "trend_aligned": trend_aligned,
                "notional_pct": self.notional_pct,
                "score": score,
            },
        )
        # Set raw_score so the orchestrator deduplicator picks highest-quality signal
        opp.raw_score = score

        logger.info(
            f"Gap and Go: {symbol} {direction.value} | "
            f"Gap {gap_pct:+.1f}% | Vol {volume_ratio:.1f}x | "
            f"Score {score} | Entry={entry:.2f} Stop={stop:.2f} Target={target:.2f}"
        )

        return opp

    @staticmethod
    def _calculate_gap_score(
        gap_pct: float,
        volume_ratio: float,
        trend_aligned: bool,
    ) -> int:
        """Dynamic scoring for gap signals based on signal quality.

        Ported from BacktestEngine._calculate_gap_score() — validated logic.

        Factors:
        - Gap size (sweet spot 2-3.5%)
        - Volume ratio (higher = stronger confirmation)
        - Trend alignment (20-day SMA direction matches gap)

        Returns score 0-100 mapping to tiers:
          A(80+)=1.5x, B(65-79)=1.25x, C(50-64)=1.0x, D(40-49)=0.5x, F(<40)=skip

        Base=35 ensures weakest valid signals (1% gap, 1.5x vol) land in C-tier
        (no amplification), preventing drawdown amplification on marginal signals.
        """
        base_score = 35

        # Gap size scoring (sweet spot is 2-3.5%)
        abs_gap = abs(gap_pct)
        if 2.0 <= abs_gap <= 3.5:
            base_score += 25
        elif 1.0 <= abs_gap < 2.0:
            base_score += 15
        elif 3.5 < abs_gap <= 5.0:
            base_score += 5
        else:
            base_score -= 10

        # Volume confirmation
        if volume_ratio >= 2.0:
            base_score += 15
        elif volume_ratio >= 1.5:
            base_score += 10
        elif volume_ratio >= 1.2:
            base_score += 5
        elif volume_ratio < 0.8:
            base_score -= 10

        # Trend alignment bonus
        if trend_aligned:
            base_score += 10

        return max(0, min(100, base_score))

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
