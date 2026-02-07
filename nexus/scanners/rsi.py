import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class RSIScanner(BaseScanner):
    """
    RSI Extreme Mean Reversion scanner.

    EDGE: Works with 2-5 period RSI and EXTREME thresholds (20/80)
    NOTE: Standard 14-period with 30/70 has NO EDGE - we use extremes only!

    Academic research shows short-period RSI with extreme thresholds
    produces consistent mean reversion opportunities.
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL"],
        Market.UK_STOCKS: ["BP", "SHEL", "HSBA", "AZN", "VOD"],
        Market.FOREX_MAJORS: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.RSI_EXTREME
        self.markets = [Market.US_STOCKS, Market.FOREX_MAJORS]
        self.instruments = []

        # KEY: Use short period and EXTREME thresholds
        self.rsi_period = 2  # NOT 14!
        self.oversold_threshold = 20  # NOT 30!
        self.overbought_threshold = 80  # NOT 70!

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """RSI scanner is active during market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for RSI extreme opportunities.

        Logic:
        1. Calculate RSI(2) for each instrument
        2. If RSI <= 20: LONG (oversold, expect bounce)
        3. If RSI >= 80: SHORT (overbought, expect pullback)
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
                    logger.warning(f"RSI scan failed for {symbol}: {e}")

        logger.info(f"RSI scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for RSI extremes."""

        # Get daily bars (need at least 30 for ATR and RSI calculation)
        bars = await self.get_bars_safe(symbol, "1D", 30)

        if bars is None or len(bars) < 10:
            logger.debug(f"RSI insufficient data for {symbol}: {len(bars) if bars is not None else 0} bars")
            return None

        # Calculate RSI(2)
        rsi_series = self._calculate_rsi(bars, self.rsi_period)
        if rsi_series is None or len(rsi_series) == 0:
            return None

        current_rsi = rsi_series.iloc[-1]

        # Only trade EXTREMES - this is where the edge exists
        if current_rsi > self.oversold_threshold and current_rsi < self.overbought_threshold:
            return None  # Not extreme enough

        current_price = bars['close'].iloc[-1]

        # Calculate ATR for stops
        atr = self.calculate_atr(bars, 14)
        if atr is None or atr <= 0:
            atr = current_price * 0.02  # Fallback: 2% of price

        # Determine direction based on RSI extreme
        if current_rsi <= self.oversold_threshold:
            # Oversold - LONG (expect bounce up)
            direction = Direction.LONG
            entry = current_price
            stop = current_price - (atr * 1.5)
            target = current_price + (atr * 2.0)
        else:
            # Overbought - SHORT (expect pullback down)
            direction = Direction.SHORT
            entry = current_price
            stop = current_price + (atr * 1.5)
            target = current_price - (atr * 2.0)

        # Create opportunity
        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "rsi": round(current_rsi, 2),
                "rsi_period": self.rsi_period,
                "threshold_used": self.oversold_threshold if direction == Direction.LONG else self.overbought_threshold,
                "atr": round(atr, 4),
                "current_price": round(current_price, 4),
            }
        )

        logger.info(
            f"RSI opportunity: {symbol} {direction.value} | RSI({self.rsi_period})={current_rsi:.1f} | "
            f"Entry={entry:.2f} Stop={stop:.2f} Target={target:.2f}"
        )

        return opp

    def _calculate_rsi(self, bars: pd.DataFrame, period: int) -> Optional[pd.Series]:
        """
        Calculate RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over period
        """
        try:
            # Calculate price changes
            delta = bars['close'].diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0.0)
            losses = (-delta).where(delta < 0, 0.0)

            # Calculate average gains and losses (using EMA for smoothing)
            avg_gain = gains.ewm(span=period, adjust=False).mean()
            avg_loss = losses.ewm(span=period, adjust=False).mean()

            # Avoid division by zero
            avg_loss = avg_loss.replace(0, 0.0001)

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return None

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
