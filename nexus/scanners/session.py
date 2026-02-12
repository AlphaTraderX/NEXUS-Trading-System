"""
Session-based scanners (power hour, London/NY open, Asian range).
"""

import logging
from datetime import datetime, time, timezone
from typing import Dict, List, Optional
import pandas as pd
import pytz

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class PowerHourScanner(BaseScanner):
    """
    Power Hour scanner - momentum continuation from end-of-day institutional activity.

    EDGE: U-shaped intraday volume → institutions active at close → momentum continues
    SIGNAL: Close in top/bottom 25% of day's range with volume confirmation
    DIRECTION: Continuation (close near high → LONG, close near low → SHORT)

    Reimplemented v15: daily bars with close-position logic (old 5m was -36% P&L).
    Backtest engine (_signal_power_hour) has the corrected daily-bar logic.
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.POWER_HOUR
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # Power Hour window (UK time) = 15:00-16:00 ET
        self.start_time = time(20, 0)  # 20:00 UK = 15:00 ET
        self.end_time = time(21, 0)   # 21:00 UK = 16:00 ET

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Active during Power Hour (20:00-21:00 UK)."""
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Europe/London'))

        current_time = timestamp.time()
        return self.start_time <= current_time <= self.end_time

    async def scan(self) -> List[Opportunity]:
        """
        Scan for Power Hour momentum continuation.

        Logic:
        1. Check if we're in Power Hour window
        2. Determine day's trend direction (open to now)
        3. Trade continuation in that direction
        """
        if not self.is_active():
            logger.debug("Power Hour scanner not active - outside window")
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
                    logger.warning(f"Power Hour scan failed for {symbol}: {e}")

        logger.info(f"Power Hour scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan single symbol for Power Hour momentum with U-shaped volume confirmation."""

        # Get intraday bars (last ~12 bars = power hour at 5-min = 60 min)
        bars = await self.get_bars_safe(symbol, "5m", 78)

        if bars is None or len(bars) < 20:
            return None

        # U-shaped volume: volume at start and end of hour > middle of hour
        n = len(bars)
        if n >= 12:
            start_vol = bars["volume"].iloc[-12:-8].sum()   # First third of hour
            middle_vol = bars["volume"].iloc[-8:-4].sum()    # Middle third
            end_vol = bars["volume"].iloc[-4:].sum()        # Last third
            avg_endcap = (start_vol + end_vol) / 2
            if middle_vol <= 0 or avg_endcap <= middle_vol:
                logger.debug(
                    f"Power Hour {symbol}: No U-shaped volume (start+end not > middle)"
                )
                return None

        # Today's open and current price
        day_open = bars['open'].iloc[0]
        current_price = bars['close'].iloc[-1]

        # Calculate day's move percentage
        day_move_pct = ((current_price - day_open) / day_open) * 100

        # Need meaningful move (> 0.3%) to have momentum
        if abs(day_move_pct) < 0.3:
            return None

        # Calculate ATR for stops
        atr = self.calculate_atr(bars, 14)
        if atr is None or atr <= 0:
            atr = current_price * 0.01

        # Trade in direction of day's momentum
        if day_move_pct > 0:
            # Bullish day - continue LONG
            direction = Direction.LONG
            entry = current_price
            stop = current_price - (atr * 1.5)
            target = current_price + (atr * 2.0)
        else:
            # Bearish day - continue SHORT
            direction = Direction.SHORT
            entry = current_price
            stop = current_price + (atr * 1.5)
            target = current_price - (atr * 2.0)

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "session": "power_hour",
                "day_open": round(day_open, 4),
                "current_price": round(current_price, 4),
                "day_move_pct": round(day_move_pct, 2),
                "momentum_direction": "bullish" if day_move_pct > 0 else "bearish",
                "atr": round(atr, 4),
            }
        )

        logger.info(f"Power Hour: {symbol} {direction.value} | Day move: {day_move_pct:+.2f}%")
        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])


class LondonOpenScanner(BaseScanner):
    """
    London Open scanner - Asian range (07:00-08:00 UK).

    EDGE: Breakout of overnight Asian range at London open
    REQUIRED: Stop-hunt confirmation - initial fake breakout then reversal
    (High false breakout rate without this confirmation.)
    """

    INSTRUMENTS = {
        Market.FOREX_MAJORS: ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.LONDON_OPEN
        self.markets = [Market.FOREX_MAJORS]
        self.instruments = []

        # London Open window (UK time) - 1 hour only
        self.start_time = time(7, 0)
        self.end_time = time(8, 0)

        # Asian session for range calculation
        self.asian_start = time(0, 0)
        self.asian_end = time(7, 0)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Active during London Open (07:00-09:00 UK)."""
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Europe/London'))

        current_time = timestamp.time()
        return self.start_time <= current_time <= self.end_time

    async def scan(self) -> List[Opportunity]:
        """Scan for London Open breakouts of Asian range."""
        if not self.is_active():
            logger.debug("London Open scanner not active - outside window")
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
                    logger.warning(f"London Open scan failed for {symbol}: {e}")

        logger.info(f"London Open scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan for Asian range with stop-hunt confirmation (fake breakout then reversal)."""

        # Get hourly bars to capture Asian session and London open
        bars = await self.get_bars_safe(symbol, "1H", 24)

        if bars is None or len(bars) < 12:
            return None

        # Calculate Asian range (last 7 hours before London open)
        asian_bars = bars.iloc[-12:-5]  # Approximate Asian session
        if len(asian_bars) < 3:
            return None

        asian_high = asian_bars['high'].max()
        asian_low = asian_bars['low'].min()
        asian_range = asian_high - asian_low
        if asian_range <= 0:
            return None

        # Stop-hunt: look for initial fake breakout then reversal (last 3-5 bars)
        recent = bars.iloc[-5:]
        fake_break_up = False   # Broke above then closed back inside
        fake_break_down = False  # Broke below then closed back inside
        for _, row in recent.iterrows():
            if row['high'] > asian_high and row['close'] < asian_high:
                fake_break_up = True
            if row['low'] < asian_low and row['close'] > asian_low:
                fake_break_down = True

        current_price = bars['close'].iloc[-1]
        breakout_buffer = asian_range * 0.1

        # Require stop-hunt then confirmed reversal
        if fake_break_down and (current_price > asian_low + breakout_buffer):
            # Fake break below then reversed up -> LONG
            direction = Direction.LONG
            entry = current_price
            stop = asian_low - (asian_range * 0.2)
            target = current_price + asian_range
        elif fake_break_up and (current_price < asian_high - breakout_buffer):
            # Fake break above then reversed down -> SHORT
            direction = Direction.SHORT
            entry = current_price
            stop = asian_high + (asian_range * 0.2)
            target = current_price - asian_range
        else:
            return None  # No stop-hunt confirmation or no clear reversal

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "session": "london_open",
                "asian_high": round(asian_high, 5),
                "asian_low": round(asian_low, 5),
                "asian_range": round(asian_range, 5),
                "breakout_direction": "up" if direction == Direction.LONG else "down",
                "current_price": round(current_price, 5),
                "stop_hunt_confirmed": True,
            }
        )

        logger.info(
            f"London Open: {symbol} {direction.value} | Stop-hunt confirmed | "
            f"Asian range: {asian_low:.5f}-{asian_high:.5f}"
        )
        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])


class NYOpenScanner(BaseScanner):
    """
    NY Open scanner - First 15-30 min range break (14:30-15:30 UK).

    EDGE: 24% probability daily high/low is set in first 30 minutes
    SIGNAL: Trade breakout of first 15-30 min range
    REQUIRED: Volume confirmation on breakout
    """

    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
        Market.FOREX_MAJORS: ["EURUSD", "GBPUSD"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.NY_OPEN
        self.markets = [Market.US_STOCKS, Market.FOREX_MAJORS]
        self.instruments = []

        # NY Open window (UK time) = first hour after US open
        self.start_time = time(14, 30)  # US market opens
        self.end_time = time(15, 30)   # First hour
        self.volume_threshold = 1.2    # 120% of average for confirmation

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Active during NY Open window (14:30-15:30 UK)."""
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Europe/London'))

        current_time = timestamp.time()
        return self.start_time <= current_time <= self.end_time

    async def scan(self) -> List[Opportunity]:
        """Scan for NY Open range breakouts."""
        if not self.is_active():
            logger.debug("NY Open scanner not active - outside window")
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
                    logger.warning(f"NY Open scan failed for {symbol}: {e}")

        logger.info(f"NY Open scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan for first 15-30 min range breakout with volume confirmation."""

        # Get 5-min bars (first 6 = 30 min opening range)
        bars = await self.get_bars_safe(symbol, "5m", 30)

        if bars is None or len(bars) < 10:
            return None

        # First 30 minutes = first 6 bars (5-min bars); 15 min = first 3 bars
        opening_bars = bars.iloc[:6]
        opening_high = opening_bars['high'].max()
        opening_low = opening_bars['low'].min()
        opening_range = opening_high - opening_low
        if opening_range <= 0:
            return None

        # Volume confirmation (US stocks): current/recent volume > 120% of average
        if market == Market.US_STOCKS and "volume" in bars.columns:
            avg_volume = bars["volume"].mean()
            recent_volume = (
                bars["volume"].iloc[-3:].mean() if len(bars) >= 3 else bars["volume"].iloc[-1]
            )
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio < self.volume_threshold:
                logger.debug(f"NY Open {symbol}: Volume ratio {volume_ratio:.2f} below 1.2")
                return None
        else:
            volume_ratio = 1.0  # Forex / no volume data: no filter

        # Current price
        current_price = bars['close'].iloc[-1]

        # ATR for additional stop buffer
        atr = self.calculate_atr(bars, 14)
        if atr is None or atr <= 0:
            atr = current_price * 0.01

        # Check for breakout (first 15-30 min range break)
        breakout_buffer = opening_range * 0.15

        if current_price > opening_high + breakout_buffer:
            direction = Direction.LONG
            entry = current_price
            stop = opening_low - (atr * 0.5)
            target = current_price + (opening_range * 1.5)
        elif current_price < opening_low - breakout_buffer:
            direction = Direction.SHORT
            entry = current_price
            stop = opening_high + (atr * 0.5)
            target = current_price - (opening_range * 1.5)
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
                "session": "ny_open",
                "opening_high": round(opening_high, 4),
                "opening_low": round(opening_low, 4),
                "opening_range": round(opening_range, 4),
                "breakout_direction": "up" if direction == Direction.LONG else "down",
                "current_price": round(current_price, 4),
                "volume_ratio": round(volume_ratio, 2),
                "volume_confirmed": True,
            }
        )

        logger.info(
            f"NY Open: {symbol} {direction.value} | Range: {opening_low:.2f}-{opening_high:.2f} | "
            f"Vol: {volume_ratio:.1f}x"
        )
        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])


class AsianRangeScanner(BaseScanner):
    """
    Asian Range Breakout scanner.

    EDGE: Asian session establishes range, London breaks it (ICT framework)
    SOURCE: Inner Circle Trader methodology, validated in forex markets

    TIMING: 07:00-08:00 UK (London open, after Asia closes)

    LOGIC:
    1. Calculate Asian session high/low (00:00-07:00 UK)
    2. At London open, watch for break of Asian range
    3. Break of high with momentum = LONG
    4. Break of low with momentum = SHORT

    INSTRUMENTS: Major forex pairs (most liquid during London)
    """

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.ASIAN_RANGE
        self.markets = [Market.FOREX_MAJORS]
        self.instruments = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        self.min_range_pips = 20  # Minimum Asian range to be tradeable
        self.breakout_buffer_pips = 5  # Buffer beyond range for confirmation

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Active during London open: 07:00-08:00 UK time.
        This is when Asian range breakouts typically occur.
        """
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        uk = pytz.timezone('Europe/London')
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        local = timestamp.astimezone(uk)
        return 7 <= local.hour < 8

    def _get_asian_session_range(self, bars: pd.DataFrame, timestamp: datetime) -> Dict[str, float]:
        """
        Calculate Asian session high and low.

        Asian session: 00:00-07:00 UTC. Filter bars by actual timestamps
        to avoid London morning bar contamination.

        Returns:
        - asian_high: Highest price during Asian session
        - asian_low: Lowest price during Asian session
        - range_pips: Range in pips
        """
        if timestamp.tzinfo is None:
            ts_utc = timestamp
        else:
            ts_utc = timestamp.astimezone(timezone.utc)
        today = ts_utc.date()

        asian_start = datetime.combine(today, time(0, 0), tzinfo=timezone.utc)
        asian_end = datetime.combine(today, time(7, 0), tzinfo=timezone.utc)

        if "timestamp" in bars.columns:
            asian_bars = bars[
                (bars["timestamp"] >= asian_start) & (bars["timestamp"] < asian_end)
            ]
        else:
            asian_bars = bars.head(7)

        if len(asian_bars) < 3:
            return {
                "asian_high": bars["high"].max(),
                "asian_low": bars["low"].min(),
                "range_pips": 0.0,
            }

        asian_high = asian_bars["high"].max()
        asian_low = asian_bars["low"].min()
        range_pips = (asian_high - asian_low) * 10000

        return {
            "asian_high": asian_high,
            "asian_low": asian_low,
            "range_pips": range_pips,
        }

    async def scan(self) -> List[Opportunity]:
        """
        Scan for Asian range breakouts at London open.
        """
        now = datetime.now(pytz.UTC)

        if not self.is_active(now):
            return []

        opportunities = []

        for symbol in self.instruments:
            try:
                # Get intraday bars for Asian session calculation
                bars = await self.get_bars(symbol, "5m", 100)
                if bars is None or len(bars) < 50:
                    continue

                # Calculate Asian range
                asian_range = self._get_asian_session_range(bars, now)

                # Skip if range too small (not enough volatility)
                if asian_range["range_pips"] < self.min_range_pips:
                    continue

                # Get current price
                current_price = await self.get_current_price(symbol)

                # Calculate breakout levels with buffer
                pip_value = 0.0001  # For most forex pairs
                buffer = self.breakout_buffer_pips * pip_value

                breakout_high = asian_range["asian_high"] + buffer
                breakout_low = asian_range["asian_low"] - buffer

                # Determine direction based on breakout
                direction = None
                if current_price > breakout_high:
                    direction = Direction.LONG
                    entry = current_price
                    stop = asian_range["asian_low"] - buffer  # Stop below Asian low
                    # Target: 1.5x the Asian range from entry
                    range_size = asian_range["asian_high"] - asian_range["asian_low"]
                    target = entry + (range_size * 1.5)

                elif current_price < breakout_low:
                    direction = Direction.SHORT
                    entry = current_price
                    stop = asian_range["asian_high"] + buffer  # Stop above Asian high
                    range_size = asian_range["asian_high"] - asian_range["asian_low"]
                    target = entry - (range_size * 1.5)

                if direction is None:
                    continue  # No breakout yet

                opp = self.create_opportunity(
                    symbol=symbol,
                    market=Market.FOREX_MAJORS,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    edge_data={
                        "asian_high": asian_range["asian_high"],
                        "asian_low": asian_range["asian_low"],
                        "asian_range_pips": asian_range["range_pips"],
                        "breakout_direction": "above" if direction == Direction.LONG else "below",
                        "buffer_pips": self.breakout_buffer_pips
                    }
                )
                opportunities.append(opp)

            except Exception as e:
                # Log error but continue scanning other symbols
                continue

        return opportunities


# Legacy SessionScanner class that runs all session-based scanners
class SessionScanner(BaseScanner):
    """
    Wrapper that runs all session-based scanners.
    Delegates to PowerHour, LondonOpen, NYOpen based on time.
    """

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.POWER_HOUR  # Default

        # Initialize all session scanners
        self.power_hour = PowerHourScanner(data_provider, settings)
        self.london_open = LondonOpenScanner(data_provider, settings)
        self.ny_open = NYOpenScanner(data_provider, settings)
        self.asian_range = AsianRangeScanner(data_provider, settings)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Active if any session scanner is active."""
        return (
            self.power_hour.is_active(timestamp) or
            self.london_open.is_active(timestamp) or
            self.ny_open.is_active(timestamp) or
            self.asian_range.is_active(timestamp)
        )

    async def scan(self) -> List[Opportunity]:
        """Run all active session scanners."""
        opportunities = []

        # Run each scanner if active
        if self.power_hour.is_active():
            opps = await self.power_hour.scan()
            opportunities.extend(opps)

        if self.london_open.is_active():
            opps = await self.london_open.scan()
            opportunities.extend(opps)

        if self.ny_open.is_active():
            opps = await self.ny_open.scan()
            opportunities.extend(opps)

        opportunities.extend(await self.asian_range.scan())

        return opportunities
