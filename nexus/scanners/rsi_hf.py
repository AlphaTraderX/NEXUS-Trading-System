"""
High-Frequency RSI Extreme Scanner

Multi-timeframe RSI scanner for intraday mean reversion.
Uses 2-period RSI with timeframe-adjusted extreme thresholds.

NOTE: The validated daily RSIScanner uses Connors RSI(2) with SMA200+ADX filters
on SPY/QQQ only. This HF scanner applies RSI extremes across more timeframes
with looser thresholds (20/80 vs 10/90) for broader opportunity detection.
"""

import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from nexus.core.enums import EdgeType, Direction, Market, Timeframe
from nexus.core.models import Opportunity
from nexus.data.instruments import get_instrument_registry, InstrumentType
from .base import BaseScanner

logger = logging.getLogger(__name__)


class RSIHighFrequencyScanner(BaseScanner):
    """
    High-frequency RSI extreme scanner.

    Uses 2-period RSI with timeframe-adjusted thresholds for mean reversion.
    Standard 14-period with 30/70 has NO edge -- we use extremes only.

    Complementary to the daily RSIScanner which uses strict Connors setup
    (RSI(2)<10 + SMA200 + ADX filter on SPY/QQQ only).
    """

    # Critical: Use short period RSI
    rsi_period = 2

    supported_timeframes = [
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.M30,
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1,
    ]

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.RSI_EXTREME
        self.markets = [Market.US_STOCKS]
        self.registry = get_instrument_registry()

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """RSI HF scanner is always active during market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """Default scan - use daily timeframe."""
        return await self.scan_timeframe(Timeframe.D1)

    async def scan_timeframe(
        self,
        timeframe: Timeframe,
        instruments: Optional[List[str]] = None,
    ) -> List[Opportunity]:
        """Scan specific timeframe for RSI extremes."""
        opportunities = []
        thresholds = self.get_thresholds(timeframe)
        oversold = thresholds["rsi_oversold"]
        overbought = thresholds["rsi_overbought"]

        if instruments is None:
            stocks = self.registry.get_by_type(InstrumentType.STOCK)
            instruments = [s.symbol for s in stocks if s.region.value == "us"][:50]

        for symbol in instruments:
            try:
                opp = await self._scan_symbol(symbol, timeframe, oversold, overbought)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error scanning {symbol} on {timeframe.value}: {e}")

        logger.info(
            f"RSI HF Scanner ({timeframe.value}): "
            f"Found {len(opportunities)} opportunities"
        )
        return opportunities

    async def _scan_symbol(
        self,
        symbol: str,
        timeframe: Timeframe,
        oversold: int,
        overbought: int,
    ) -> Optional[Opportunity]:
        """Scan single symbol for RSI extremes on given timeframe."""
        bars_needed = max(50, timeframe.bars_per_day)

        bars = await self.get_bars_safe(symbol, timeframe.value, bars_needed)
        if bars is None or len(bars) < 10:
            return None

        rsi = self._calculate_rsi_2(bars)
        current_rsi = float(rsi.iloc[-1])

        if pd.isna(current_rsi):
            return None

        # Only trade EXTREMES
        if oversold < current_rsi < overbought:
            return None

        current_price = float(bars["close"].iloc[-1])
        atr = self.calculate_atr(bars, 14)
        if atr == 0:
            return None

        if current_rsi <= oversold:
            direction = Direction.LONG
            entry = current_price
            stop = current_price - (atr * 1.5)
            target = current_price + (atr * 2.0)
            threshold_used = oversold
        else:
            direction = Direction.SHORT
            entry = current_price
            stop = current_price + (atr * 1.5)
            target = current_price - (atr * 2.0)
            threshold_used = overbought

        instrument = self.registry.get(symbol)
        market = Market.US_STOCKS
        if instrument:
            region_to_market = {
                "us": Market.US_STOCKS,
                "uk": Market.UK_STOCKS,
                "europe": Market.EU_STOCKS,
            }
            market = region_to_market.get(instrument.region.value, Market.US_STOCKS)

        return self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "strategy": "rsi_extreme_hf",
                "rsi": round(current_rsi, 2),
                "rsi_period": self.rsi_period,
                "threshold_used": threshold_used,
                "atr": round(atr, 4),
                "timeframe": timeframe.value,
            },
        )

    def _calculate_rsi_2(self, bars: pd.DataFrame) -> pd.Series:
        """Calculate RSI with period 2 using rolling mean (matches scan context)."""
        close = bars["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi
