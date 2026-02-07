import logging
from datetime import datetime, time
from typing import List, Optional
import pandas as pd
import pytz

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class ORBScanner(BaseScanner):
    """
    Opening Range Breakout (ORB) scanner.

    EDGE: Breakout of first 15-30 min range with volume and VWAP confirmation

    REQUIREMENTS (all must be met):
    1. Clear breakout of opening range (high or low)
    2. Volume > 120% of average (conviction)
    3. VWAP alignment (price above VWAP for longs, below for shorts)

    Without these filters, ORB has too many false breakouts.
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.ORB
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # ORB parameters
        self.opening_range_minutes = 30  # First 30 minutes
        self.volume_threshold = 1.2  # 120% of average volume required
        self.breakout_buffer_pct = 0.1  # 0.1% buffer to confirm breakout

        # ORB window (UK time) - US market first 2 hours
        self.start_time = time(14, 30)
        self.end_time = time(17, 0)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Active during US market first 2.5 hours."""
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Europe/London'))

        current_time = timestamp.time()
        return self.start_time <= current_time <= self.end_time

    async def scan(self) -> List[Opportunity]:
        """
        Scan for ORB opportunities with volume and VWAP confirmation.
        """
        if not self.is_active():
            logger.debug("ORB scanner not active - outside window")
            return []

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

        logger.info(f"ORB scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan single symbol for ORB setup."""

        # Get 5-min bars (need ~40 bars for opening range + current + history)
        bars = await self.get_bars_safe(symbol, "5m", 50)

        if bars is None or len(bars) < 20:
            return None

        # Define opening range (first 6 bars = 30 minutes of 5-min bars)
        opening_bars = bars.iloc[:6]
        opening_high = opening_bars['high'].max()
        opening_low = opening_bars['low'].min()
        opening_range = opening_high - opening_low

        if opening_range <= 0:
            return None

        # Current price and recent bars
        current_price = bars['close'].iloc[-1]
        recent_bars = bars.iloc[-10:]  # Last 10 bars for volume comparison

        # FILTER 1: Check volume (must be > 120% of average)
        current_volume = recent_bars['volume'].iloc[-1]
        avg_volume = bars['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        if volume_ratio < self.volume_threshold:
            logger.debug(f"ORB {symbol}: Volume ratio {volume_ratio:.2f} below threshold")
            return None

        # FILTER 2: Calculate VWAP for confirmation
        vwap = self._calculate_vwap(bars)
        if vwap is None or vwap <= 0:
            return None

        # Check for breakout with buffer
        breakout_buffer = opening_range * self.breakout_buffer_pct

        # Determine breakout direction
        if current_price > opening_high + breakout_buffer:
            # Breakout UP
            # FILTER 3: VWAP confirmation (price should be above VWAP for long)
            if current_price < vwap:
                logger.debug(f"ORB {symbol}: Breakout UP but below VWAP - no confirmation")
                return None

            direction = Direction.LONG
            entry = current_price
            stop = opening_low - (opening_range * 0.2)  # Stop below range
            target = current_price + (opening_range * 1.5)  # 1.5x range extension

        elif current_price < opening_low - breakout_buffer:
            # Breakout DOWN
            # FILTER 3: VWAP confirmation (price should be below VWAP for short)
            if current_price > vwap:
                logger.debug(f"ORB {symbol}: Breakout DOWN but above VWAP - no confirmation")
                return None

            direction = Direction.SHORT
            entry = current_price
            stop = opening_high + (opening_range * 0.2)  # Stop above range
            target = current_price - (opening_range * 1.5)

        else:
            return None  # No breakout

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "opening_high": round(opening_high, 4),
                "opening_low": round(opening_low, 4),
                "opening_range": round(opening_range, 4),
                "vwap": round(vwap, 4),
                "vwap_confirmed": True,
                "volume_ratio": round(volume_ratio, 2),
                "volume_confirmed": True,
                "breakout_direction": "up" if direction == Direction.LONG else "down",
                "current_price": round(current_price, 4),
            }
        )

        logger.info(
            f"ORB: {symbol} {direction.value} | "
            f"Range: {opening_low:.2f}-{opening_high:.2f} | "
            f"Volume: {volume_ratio:.1f}x | VWAP: {vwap:.2f}"
        )

        return opp

    def _calculate_vwap(self, bars: pd.DataFrame) -> Optional[float]:
        """Calculate Volume Weighted Average Price."""
        try:
            typical_price = (bars['high'] + bars['low'] + bars['close']) / 3
            total_volume = bars['volume'].sum()

            if total_volume <= 0:
                return None

            vwap = (typical_price * bars['volume']).sum() / total_volume
            return float(vwap)
        except Exception as e:
            logger.warning(f"VWAP calculation failed: {e}")
            return None

    def _get_instruments(self, market: Market) -> List[str]:
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
