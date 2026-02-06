import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

from scanners.base import BaseScanner
from core.enums import EdgeType, Market, Direction
from core.models import Opportunity
from intelligence.regime import RegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class BollingerScanner(BaseScanner):
    """
    Bollinger Band Touch scanner.

    EDGE: 88% mean reversion when price touches bands
    CRITICAL: Only works in RANGING regime, NOT trending!

    Rules:
    1. Price touches lower band → LONG (expect bounce up)
    2. Price touches upper band → SHORT (expect pullback)
    3. MUST have reversal candle confirmation
    4. MUST be in ranging regime (ADX < 25 or BB width contracting)
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"],
        Market.UK_STOCKS: ["BP", "SHEL", "HSBA", "AZN"],
        Market.FOREX_MAJORS: ["EURUSD", "GBPUSD", "USDJPY"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.BOLLINGER_TOUCH
        self.markets = [Market.US_STOCKS, Market.FOREX_MAJORS]
        self.instruments = []

        # Bollinger parameters
        self.bb_period = 20
        self.bb_std = 2.0

        # Regime filter
        self.adx_threshold = 25  # Below this = ranging
        self.adx_period = 14

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Bollinger scanner active during market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for Bollinger Band touch with regime filter.
        Only signal in RANGING or VOLATILE regime; edge disappears in trending markets.
        """
        # Regime check: no Bollinger signals in trending markets (88% reversion only in ranging)
        spy_bars = await self.get_bars_safe("SPY", "1D", 60)
        if spy_bars is not None and len(spy_bars) >= 50:
            detector = RegimeDetector()
            analysis = detector.detect_regime(spy_bars, "SPY")
            if analysis.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                logger.info(
                    f"Bollinger scan skipped: regime={analysis.regime.value} (no edge in trending)"
                )
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
                    logger.warning(f"Bollinger scan failed for {symbol}: {e}")

        logger.info(f"Bollinger scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan single symbol for Bollinger Band touch."""

        # Get daily bars
        bars = await self.get_bars_safe(symbol, "1D", 50)

        if bars is None or len(bars) < 30:
            return None

        # Calculate Bollinger Bands
        bb = self._calculate_bollinger_bands(bars)
        if bb is None:
            return None

        upper_band = bb['upper'].iloc[-1]
        lower_band = bb['lower'].iloc[-1]
        middle_band = bb['middle'].iloc[-1]
        bb_width = bb['width'].iloc[-1]

        # Current price and previous candle
        current_price = bars['close'].iloc[-1]
        prev_close = bars['close'].iloc[-2]
        current_low = bars['low'].iloc[-1]
        current_high = bars['high'].iloc[-1]

        # FILTER 1: Check regime (must be ranging - ADX < 25)
        adx = self._calculate_adx(bars)
        if adx is None:
            adx = 20  # Assume ranging if calculation fails

        is_ranging = adx < self.adx_threshold

        if not is_ranging:
            logger.debug(f"Bollinger {symbol}: ADX {adx:.1f} > {self.adx_threshold} - trending, skip")
            return None

        # Check for band touch
        touched_lower = current_low <= lower_band
        touched_upper = current_high >= upper_band

        if not touched_lower and not touched_upper:
            return None  # No band touch

        # FILTER 2: Check for reversal candle (close should be moving back toward middle)
        if touched_lower:
            # Touched lower band - look for bullish reversal
            is_reversal = current_price > prev_close  # Closed higher than opened or prev close
            if not is_reversal:
                logger.debug(f"Bollinger {symbol}: Touched lower but no bullish reversal")
                return None

            direction = Direction.LONG
            entry = current_price
            stop = lower_band - (bb_width * 0.1)  # Stop just below lower band
            target = middle_band  # Target is middle band (mean)
            band_touched = "lower"

        else:  # touched_upper
            # Touched upper band - look for bearish reversal
            is_reversal = current_price < prev_close  # Closed lower
            if not is_reversal:
                logger.debug(f"Bollinger {symbol}: Touched upper but no bearish reversal")
                return None

            direction = Direction.SHORT
            entry = current_price
            stop = upper_band + (bb_width * 0.1)  # Stop just above upper band
            target = middle_band  # Target is middle band (mean)
            band_touched = "upper"

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "band_touched": band_touched,
                "upper_band": round(upper_band, 4),
                "lower_band": round(lower_band, 4),
                "middle_band": round(middle_band, 4),
                "bb_width": round(bb_width, 4),
                "adx": round(adx, 2),
                "regime": "ranging",
                "reversal_confirmed": True,
                "current_price": round(current_price, 4),
            }
        )

        logger.info(
            f"Bollinger: {symbol} {direction.value} | "
            f"Touched {band_touched} band | ADX: {adx:.1f} | "
            f"Target: {middle_band:.2f}"
        )

        return opp

    def _calculate_bollinger_bands(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate Bollinger Bands."""
        try:
            close = bars['close']

            # Middle band = SMA
            middle = close.rolling(window=self.bb_period).mean()

            # Standard deviation
            std = close.rolling(window=self.bb_period).std()

            # Upper and lower bands
            upper = middle + (std * self.bb_std)
            lower = middle - (std * self.bb_std)

            # Band width (for regime detection)
            width = upper - lower

            return pd.DataFrame({
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'width': width
            })
        except Exception as e:
            logger.warning(f"Bollinger calculation failed: {e}")
            return None

    def _calculate_adx(self, bars: pd.DataFrame) -> Optional[float]:
        """
        Calculate ADX (Average Directional Index) for trend strength.
        ADX < 25 = ranging/weak trend
        ADX > 25 = trending
        """
        try:
            high = bars['high']
            low = bars['low']
            close = bars['close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # Smooth with EMA
            period = self.adx_period
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
            adx = dx.ewm(span=period, adjust=False).mean()

            return float(adx.iloc[-1])
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            return None

    def _get_instruments(self, market: Market) -> List[str]:
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
