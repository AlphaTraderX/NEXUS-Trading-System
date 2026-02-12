import logging
from datetime import datetime, time, timezone
from typing import List, Optional

import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class ORBScanner(BaseScanner):
    """
    Opening Range Breakout with Volume Filter.

    Academic basis: Zarattini, Barbon & Aziz (2024) — SSRN #4729284
    7,000+ US stocks tested, 2016-2023.

    Key finding: Volume filter transforms marginal ORB into Sharpe 2.81.
    WITHOUT volume filter: marginal / unprofitable.
    WITH relative volume >= 100%: Sharpe 2.81, 36% annualised alpha.

    Parameters:
    - Opening range: First 5 minutes (single 5m bar)
    - Volume filter: Relative volume >= 100% of 20-day avg (CRITICAL)
    - Direction: Trade in direction of first candle's close bias
    - Stop: 10% of 14-day ATR (tight — moves fast or fails fast)
    - Target: Hold to EOD, no profit target (let winners run)
    - Max entry window: 30 min after open (bars 1-5)

    Expected: 30-40% win rate, 3:1+ R-multiples, Sharpe 2.0+
    """

    INSTRUMENTS = {
        Market.US_STOCKS: [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL",
        ],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.ORB
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # ORB parameters
        self.opening_range_minutes = 5        # First 5m bar
        self.volume_lookback_days = 20        # 20-day average volume
        self.min_relative_volume = 1.0        # 100% = 2x normal (CRITICAL)
        self.max_entry_bars = 6               # Bars 1-5 after opening bar
        self.atr_stop_pct = 0.25              # 25% of ATR
        self.notional_pct = 20                # 20% of capital per trade

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """ORB is active during first 30 min of US session."""
        if timestamp is None:
            return True
        # Active 14:30-15:00 UTC (09:30-10:00 ET)
        if timestamp.hour == 14 and timestamp.minute >= 30:
            return True
        if timestamp.hour == 15 and timestamp.minute < 5:
            return True
        return False

    async def scan(self) -> List[Opportunity]:
        """
        Scan for ORB opportunities with volume filter.

        Logic:
        1. Get 5m bars (need 100+ for 20-day volume history)
        2. Identify today's opening range (first 5m bar)
        3. Check relative volume >= 100% (CRITICAL)
        4. Determine direction bias from first candle
        5. Check for breakout with close confirmation
        6. Enter with tight ATR stop, hold to EOD
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
                    logger.warning(f"ORB scan failed for {symbol}: {e}")

        logger.info(f"ORB scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for ORB setup with volume filter."""

        # Need 100+ bars for 20-day first-bar volume average
        bars = await self.get_bars_safe(symbol, "5m", 100)
        if bars is None or len(bars) < 50:
            return None

        # Identify today's bars
        current_bar = bars.iloc[-1]
        today = (
            current_bar.name.date()
            if hasattr(current_bar.name, "date")
            else None
        )
        if today is None:
            return None

        today_mask = bars.index.date == today
        today_bars = bars[today_mask]
        if len(today_bars) < 2:
            return None

        first_bar = today_bars.iloc[0]
        bar_num = len(today_bars) - 1  # 0-indexed position in session

        # Must be in entry window (bars 1-5, i.e. 09:35-10:00)
        if bar_num < 1 or bar_num > 5:
            return None

        # ── Opening range (first 5m bar) ──────────────────────────
        or_high = first_bar["high"]
        or_low = first_bar["low"]
        is_bullish = first_bar["close"] > first_bar["open"]

        # ── Volume filter (CRITICAL — this IS the edge) ──────────
        first_bar_vol = first_bar["volume"]

        # 20-day average of first-bar volume from prior sessions
        prev_days = bars[~today_mask]
        if len(prev_days) < 50:
            return None

        prev_dates = sorted(set(prev_days.index.date))
        first_bar_vols = []
        for d in prev_dates:
            day_bars = prev_days[prev_days.index.date == d]
            if len(day_bars) > 0:
                first_bar_vols.append(day_bars.iloc[0]["volume"])

        if len(first_bar_vols) < 5:
            return None

        recent_vols = first_bar_vols[-20:]
        avg_vol = sum(recent_vols) / len(recent_vols)
        if avg_vol == 0:
            return None

        relative_volume = first_bar_vol / avg_vol

        if relative_volume < self.min_relative_volume:
            logger.debug(
                f"ORB {symbol}: Volume {relative_volume:.2f}x below threshold"
            )
            return None

        # ── Breakout check with direction bias ────────────────────
        cur_price = current_bar["close"]

        if is_bullish and cur_price > or_high:
            direction = Direction.LONG
        elif not is_bullish and cur_price < or_low:
            direction = Direction.SHORT
        else:
            return None

        # ── Tight ATR stop (10% of ATR) ──────────────────────────
        atr = self.calculate_atr(bars, 14)
        if atr is None or atr <= 0:
            atr = cur_price * 0.02
        stop_distance = atr * self.atr_stop_pct

        if direction == Direction.LONG:
            stop = cur_price - stop_distance
            target = cur_price * 1.10  # Placeholder — exit at EOD
        else:
            stop = cur_price + stop_distance
            target = cur_price * 0.90

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=cur_price,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "relative_volume": round(relative_volume, 2),
                "range_high": round(or_high, 4),
                "range_low": round(or_low, 4),
                "range_size": round(or_high - or_low, 4),
                "first_bar_bullish": is_bullish,
                "daily_atr": round(atr, 4),
                "stop_distance": round(stop_distance, 4),
                "exit_method": "end_of_day",
                "notional_pct": self.notional_pct,
            },
        )

        logger.info(
            f"ORB opportunity: {symbol} {direction.value} | "
            f"Volume={relative_volume:.1f}x | Range={or_high - or_low:.2f}"
        )

        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
