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
    Gap Fill scanner.

    EDGE: 60-92% of small gaps fill within the same trading day
    SIGNAL: Stock gaps up/down at open, trade toward previous close

    Rules:
    - Gap must be > 0.5% but < 3% (small gaps fill more reliably)
    - Trade in direction of gap fill (gap up = SHORT, gap down = LONG)
    - Target = previous close (the "fill")
    - Best in first 30-60 minutes of session
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"],
        Market.UK_STOCKS: ["BP", "SHEL", "HSBA", "AZN", "LLOY"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.GAP_FILL
        self.markets = [Market.US_STOCKS, Market.UK_STOCKS]
        self.instruments = []

        # Gap thresholds
        self.min_gap_pct = 0.5   # Minimum gap size (0.5%)
        self.max_gap_pct = 3.0   # Maximum gap size (3%) - larger gaps less likely to fill

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Gap scanner is most effective in first 30-60 minutes of session.
        For now, active during market hours.
        """
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for gap fill opportunities.

        Logic:
        1. Compare today's open to yesterday's close
        2. If gap > 0.5% and < 3%: potential opportunity
        3. Gap UP = SHORT (fill down toward prev close)
        4. Gap DOWN = LONG (fill up toward prev close)
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
        """Scan a single symbol for gap fill opportunity."""

        # Get recent daily bars (need at least 2 for gap calculation)
        bars = await self.get_bars_safe(symbol, "1D", 5)

        if bars is None or len(bars) < 2:
            logger.debug(f"Gap insufficient data for {symbol}")
            return None

        # Yesterday's close and today's open
        prev_close = bars['close'].iloc[-2]
        today_open = bars['open'].iloc[-1]
        current_price = bars['close'].iloc[-1]

        # Calculate gap percentage
        gap_pct = ((today_open - prev_close) / prev_close) * 100

        # Check if gap is in our sweet spot (0.5% to 3%)
        abs_gap = abs(gap_pct)
        if abs_gap < self.min_gap_pct or abs_gap > self.max_gap_pct:
            return None  # Gap too small or too large

        # Calculate ATR for stop placement
        atr = self.calculate_atr(bars, 14) if len(bars) >= 14 else None
        if atr is None or atr <= 0:
            atr = current_price * 0.015  # Fallback: 1.5% of price

        # Determine direction (FILL THE GAP)
        if gap_pct > 0:
            # Gap UP - price opened higher than prev close
            # Trade SHORT to fill the gap (back down to prev close)
            direction = Direction.SHORT
            entry = current_price
            target = prev_close  # Fill target is previous close
            stop = current_price + (atr * 1.5)  # Stop above entry

            # Check if gap already partially filled
            fill_progress = (today_open - current_price) / (today_open - prev_close) * 100 if today_open != prev_close else 0
        else:
            # Gap DOWN - price opened lower than prev close
            # Trade LONG to fill the gap (back up to prev close)
            direction = Direction.LONG
            entry = current_price
            target = prev_close  # Fill target is previous close
            stop = current_price - (atr * 1.5)  # Stop below entry

            # Check if gap already partially filled
            fill_progress = (current_price - today_open) / (prev_close - today_open) * 100 if prev_close != today_open else 0

        # Skip if gap already mostly filled (>70%)
        if fill_progress > 70:
            logger.debug(f"Gap already {fill_progress:.0f}% filled for {symbol}")
            return None

        # Calculate potential reward
        potential_reward_pct = abs(target - entry) / entry * 100

        # Create opportunity
        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "gap_pct": round(gap_pct, 2),
                "gap_direction": "UP" if gap_pct > 0 else "DOWN",
                "prev_close": round(prev_close, 4),
                "today_open": round(today_open, 4),
                "current_price": round(current_price, 4),
                "fill_progress_pct": round(fill_progress, 1),
                "potential_reward_pct": round(potential_reward_pct, 2),
                "atr": round(atr, 4),
            }
        )

        logger.info(
            f"Gap opportunity: {symbol} {direction.value} | "
            f"Gap {gap_pct:+.2f}% | Fill progress: {fill_progress:.0f}% | "
            f"Entry={entry:.2f} Target={target:.2f}"
        )

        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
