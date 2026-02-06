"""
VWAP deviation scanner (mean reversion, strong academic backing).
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from core.enums import EdgeType, Market, Direction
from core.models import Opportunity
from scanners.base import BaseScanner

logger = logging.getLogger(__name__)


class VWAPScanner(BaseScanner):
    """
    VWAP Deviation Mean Reversion scanner.

    EDGE: Sharpe ratio 2.1 in academic study (Zarattini & Aziz)
    SIGNAL: Price deviates >2 standard deviations from VWAP
    DIRECTION: Mean reversion back to VWAP
    """

    # Instrument lists by market
    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"],
        Market.UK_STOCKS: ["BP", "SHEL", "HSBA", "AZN"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.VWAP_DEVIATION
        self.markets = [Market.US_STOCKS, Market.UK_STOCKS]
        self.instruments = []
        self.deviation_threshold = 2.0  # Standard deviations

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """VWAP scanner is active during market hours."""
        # For now, always active - can add session checks later
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for VWAP deviation opportunities.

        Logic:
        1. Get intraday 5-min bars for each instrument
        2. Calculate VWAP and standard deviation bands
        3. If price > VWAP + 2σ: SHORT (mean revert down)
        4. If price < VWAP - 2σ: LONG (mean revert up)
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
                    logger.warning(
                        "vwap_scan_symbol_failed",
                        extra={"symbol": symbol, "error": str(e)},
                    )

        logger.info(
            "vwap_scan_complete",
            extra={"opportunities_found": len(opportunities)},
        )
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for VWAP deviation."""

        # Get intraday bars (5-min, ~78 bars = full day)
        bars = await self.get_bars_safe(symbol, "5m", 78)

        if bars is None or len(bars) < 20:
            logger.debug(
                "vwap_insufficient_data",
                extra={
                    "symbol": symbol,
                    "bars": len(bars) if bars is not None else 0,
                },
            )
            return None

        # Calculate VWAP
        vwap = self._calculate_vwap(bars)
        if vwap is None or vwap <= 0:
            return None

        # Calculate VWAP standard deviation
        vwap_std = self._calculate_vwap_std(bars, vwap)
        if vwap_std is None or vwap_std <= 0:
            return None

        # Get current price
        current_price = float(bars["close"].iloc[-1])

        # Calculate deviation in standard deviations
        deviation = (current_price - vwap) / vwap_std

        # Check for extreme deviation
        if abs(deviation) < self.deviation_threshold:
            return None  # Not extreme enough

        # Determine direction (MEAN REVERSION)
        if deviation > self.deviation_threshold:
            # Price too HIGH - SHORT to mean revert down
            direction = Direction.SHORT
            entry = current_price
            target = vwap  # Target is VWAP
            stop = current_price + (vwap_std * 1.5)  # Stop above entry
        else:
            # Price too LOW - LONG to mean revert up
            direction = Direction.LONG
            entry = current_price
            target = vwap  # Target is VWAP
            stop = current_price - (vwap_std * 1.5)  # Stop below entry

        # Create opportunity
        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "vwap": round(vwap, 4),
                "deviation_std": round(deviation, 2),
                "vwap_std": round(vwap_std, 4),
                "current_price": round(current_price, 4),
                "threshold": self.deviation_threshold,
            },
        )

        logger.info(
            "vwap_opportunity_found",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "deviation": round(deviation, 2),
                "vwap": round(vwap, 4),
                "current_price": round(current_price, 4),
            },
        )

        return opp

    def _calculate_vwap(self, bars: pd.DataFrame) -> Optional[float]:
        """
        Calculate Volume Weighted Average Price.

        VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        Typical Price = (High + Low + Close) / 3
        """
        try:
            typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
            total_volume = bars["volume"].sum()

            if total_volume <= 0:
                return None

            vwap = (typical_price * bars["volume"]).sum() / total_volume
            return float(vwap)
        except Exception as e:
            logger.warning("vwap_calculation_failed", extra={"error": str(e)})
            return None

    def _calculate_vwap_std(self, bars: pd.DataFrame, vwap: float) -> Optional[float]:
        """
        Calculate volume-weighted standard deviation of price from VWAP.
        """
        try:
            typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
            total_volume = bars["volume"].sum()

            if total_volume <= 0:
                return None

            # Volume-weighted variance
            variance = (
                ((typical_price - vwap) ** 2 * bars["volume"]).sum() / total_volume
            )
            std = variance**0.5
            return float(std)
        except Exception as e:
            logger.warning("vwap_std_calculation_failed", extra={"error": str(e)})
            return None

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
