"""
High-Frequency VWAP Deviation Scanner

Multi-timeframe VWAP mean reversion scanner for intraday trading.
Scans 5m, 15m, 30m, 1h, 4h, Daily timeframes for VWAP deviation opportunities.

NOTE: The validated daily VWAPScanner uses TREND-FOLLOWING (cross above = long).
This HF scanner uses MEAN REVERSION on shorter timeframes (extreme deviation = fade).
These are complementary strategies on different timeframes.
"""

import logging
from typing import List, Optional
from datetime import datetime

import pandas as pd

from nexus.core.enums import EdgeType, Direction, Market, Timeframe
from nexus.core.models import Opportunity
from nexus.data.instruments import get_instrument_registry, InstrumentType
from .base import BaseScanner

logger = logging.getLogger(__name__)


class VWAPHighFrequencyScanner(BaseScanner):
    """
    High-frequency VWAP deviation scanner.

    Scans multiple timeframes for mean reversion opportunities
    when price deviates significantly from VWAP.

    Complementary to the daily VWAPScanner (trend-following).
    This scanner fades extreme deviations on shorter timeframes.
    """

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
        self.edge_type = EdgeType.VWAP_DEVIATION
        self.markets = [Market.US_STOCKS]
        self.registry = get_instrument_registry()

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """VWAP HF scanner is always active during market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """Default scan - use daily timeframe."""
        return await self.scan_timeframe(Timeframe.D1)

    async def scan_timeframe(
        self,
        timeframe: Timeframe,
        instruments: Optional[List[str]] = None,
    ) -> List[Opportunity]:
        """
        Scan specific timeframe for VWAP deviation.

        Args:
            timeframe: Timeframe to scan
            instruments: Optional list of symbols (defaults to top 50 US stocks)

        Returns:
            List of opportunities where price deviates from VWAP
        """
        opportunities = []
        thresholds = self.get_thresholds(timeframe)
        deviation_threshold = thresholds["vwap_deviation_std"]

        if instruments is None:
            stocks = self.registry.get_by_type(InstrumentType.STOCK)
            instruments = [s.symbol for s in stocks if s.region.value == "us"][:50]

        for symbol in instruments:
            try:
                opp = await self._scan_symbol(symbol, timeframe, deviation_threshold)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error scanning {symbol} on {timeframe.value}: {e}")

        logger.info(
            f"VWAP HF Scanner ({timeframe.value}): "
            f"Found {len(opportunities)} opportunities"
        )
        return opportunities

    async def _scan_symbol(
        self,
        symbol: str,
        timeframe: Timeframe,
        deviation_threshold: float,
    ) -> Optional[Opportunity]:
        """Scan single symbol for VWAP deviation on given timeframe."""
        bars_needed = max(78, timeframe.bars_per_day)

        bars = await self.get_bars_safe(symbol, timeframe.value, bars_needed)
        if bars is None or len(bars) < 20:
            return None

        vwap = self._calculate_vwap(bars)
        vwap_std = self._calculate_vwap_std(bars, vwap)

        if vwap_std == 0:
            return None

        current_price = float(bars["close"].iloc[-1])
        deviation = (current_price - vwap) / vwap_std

        if abs(deviation) < deviation_threshold:
            return None

        atr = self.calculate_atr(bars, 14)
        if atr == 0:
            return None

        # Mean reversion: fade the deviation
        if deviation > deviation_threshold:
            direction = Direction.SHORT
            entry = current_price
            stop = current_price + (atr * 1.5)
            target = vwap
        else:
            direction = Direction.LONG
            entry = current_price
            stop = current_price - (atr * 1.5)
            target = vwap

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
                "strategy": "mean_reversion",
                "vwap": round(vwap, 4),
                "deviation_std": round(deviation, 2),
                "vwap_std": round(vwap_std, 4),
                "atr": round(atr, 4),
                "timeframe": timeframe.value,
                "threshold_used": deviation_threshold,
            },
        )

    @staticmethod
    def _calculate_vwap(bars: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price."""
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3

        if "volume" not in bars.columns or bars["volume"].sum() == 0:
            return float(typical_price.mean())

        vwap = (typical_price * bars["volume"]).sum() / bars["volume"].sum()
        return float(vwap)

    @staticmethod
    def _calculate_vwap_std(bars: pd.DataFrame, vwap: float) -> float:
        """Calculate standard deviation of price from VWAP."""
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3

        if "volume" not in bars.columns or bars["volume"].sum() == 0:
            return float(typical_price.std())

        variance = ((typical_price - vwap) ** 2 * bars["volume"]).sum() / bars["volume"].sum()
        return float(variance ** 0.5)
