"""
Turn of Month Scanner - One of the strongest calendar edges.

Academic backing: McConnell & Xu (2008) found 100% of equity premium
earned in the 4-day window around month-end.

Window: Last trading day of month through Day 3 of new month
Direction: LONG only (pension fund buying)
Instruments: Index ETFs (SPY, QQQ, IWM) and futures (ES, NQ)
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

from zoneinfo import ZoneInfo

from nexus.core.enums import Direction, EdgeType, Market
from nexus.core.models import Opportunity
from nexus.scanners.base import BaseScanner

logger = logging.getLogger(__name__)

# US holidays that close markets (simplified - major ones)
US_MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS = US_MARKET_HOLIDAYS_2024 | US_MARKET_HOLIDAYS_2025


class TurnOfMonthScanner(BaseScanner):
    """
    Scan for Turn of Month opportunities.

    The TOM effect is one of the most robust calendar anomalies:
    - Last trading day of month: Day -1
    - First trading day of new month: Day 1
    - Second trading day: Day 2
    - Third trading day: Day 3

    We go LONG at start of window, exit at end of Day 3.
    """

    # Instruments to trade during TOM
    TOM_INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "IWM"],
        Market.US_FUTURES: ["ES", "NQ"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider=data_provider, settings=settings)
        self.edge_type = EdgeType.TURN_OF_MONTH
        self.markets = [Market.US_STOCKS, Market.US_FUTURES]
        self.tz = ZoneInfo("America/New_York")

    def is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day (not weekend or holiday)."""
        if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        if d in US_MARKET_HOLIDAYS:
            return False
        return True

    def get_last_trading_day_of_month(self, year: int, month: int) -> date:
        """Get the last trading day of a given month."""
        # Start from last day of month
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)

        last_day = next_month - timedelta(days=1)

        # Walk backward to find trading day
        while not self.is_trading_day(last_day):
            last_day -= timedelta(days=1)

        return last_day

    def get_trading_day_of_month(self, year: int, month: int, day_number: int) -> Optional[date]:
        """Get the Nth trading day of a month (1-indexed)."""
        current = date(year, month, 1)
        trading_days_found = 0

        while trading_days_found < day_number:
            if self.is_trading_day(current):
                trading_days_found += 1
                if trading_days_found == day_number:
                    return current
            current += timedelta(days=1)

            # Safety: don't go past month
            if current.month != month:
                return None

        return current

    def get_tom_window(self, reference_date: date) -> dict:
        """
        Get the TOM window that contains or is nearest to reference_date.

        Returns:
            {
                "in_window": bool,
                "window_start": date (last trading day of prev month),
                "window_end": date (3rd trading day of current month),
                "current_day": int (-1, 1, 2, or 3),
                "days_remaining": int
            }
        """
        year = reference_date.year
        month = reference_date.month

        # Get last trading day of PREVIOUS month
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        last_day_prev = self.get_last_trading_day_of_month(prev_year, prev_month)

        # Get first 3 trading days of CURRENT month
        day_1 = self.get_trading_day_of_month(year, month, 1)
        day_2 = self.get_trading_day_of_month(year, month, 2)
        day_3 = self.get_trading_day_of_month(year, month, 3)

        # Also check if we're at end of current month (entering next TOM)
        last_day_current = self.get_last_trading_day_of_month(year, month)

        # Determine which window we're in
        if reference_date == last_day_prev:
            return {
                "in_window": True,
                "window_start": last_day_prev,
                "window_end": day_3,
                "current_day": -1,
                "days_remaining": 4,
            }
        elif reference_date == day_1:
            return {
                "in_window": True,
                "window_start": last_day_prev,
                "window_end": day_3,
                "current_day": 1,
                "days_remaining": 3,
            }
        elif reference_date == day_2:
            return {
                "in_window": True,
                "window_start": last_day_prev,
                "window_end": day_3,
                "current_day": 2,
                "days_remaining": 2,
            }
        elif reference_date == day_3:
            return {
                "in_window": True,
                "window_start": last_day_prev,
                "window_end": day_3,
                "current_day": 3,
                "days_remaining": 1,
            }
        elif reference_date == last_day_current:
            # We're at end of THIS month - entering NEXT month's TOM
            if month == 12:
                next_year, next_month = year + 1, 1
            else:
                next_year, next_month = year, month + 1

            next_day_1 = self.get_trading_day_of_month(next_year, next_month, 1)
            next_day_3 = self.get_trading_day_of_month(next_year, next_month, 3)

            return {
                "in_window": True,
                "window_start": last_day_current,
                "window_end": next_day_3,
                "current_day": -1,
                "days_remaining": 4,
            }
        else:
            return {
                "in_window": False,
                "window_start": None,
                "window_end": None,
                "current_day": None,
                "days_remaining": 0,
            }

    def is_active(self, timestamp: datetime = None) -> bool:
        """Check if we're currently in a TOM window."""
        if timestamp is None:
            timestamp = datetime.now(self.tz)

        ref_date = timestamp.date()
        window = self.get_tom_window(ref_date)
        return window["in_window"]

    async def scan(self, timestamp: datetime = None) -> List[Opportunity]:
        """
        Scan for TOM opportunities.

        Only generates signals when:
        1. We're in the TOM window
        2. It's a trading day
        3. Market is open or about to open
        """
        if timestamp is None:
            timestamp = datetime.now(self.tz)

        ref_date = timestamp.date()
        window = self.get_tom_window(ref_date)

        if not window["in_window"]:
            logger.debug("TOM scanner: Not in TOM window")
            return []

        opportunities = []

        for market, symbols in self.TOM_INSTRUMENTS.items():
            for symbol in symbols:
                try:
                    current_price = await self.get_current_price(symbol)
                    bars = await self.get_bars(symbol, "1D", 20)

                    if bars is None or len(bars) < 14:
                        logger.warning("TOM: Insufficient data for %s", symbol)
                        continue

                    atr = self.calculate_atr(bars, 14)

                    # TOM is always LONG
                    entry = current_price
                    stop = entry - (atr * 1.5)  # 1.5 ATR stop
                    target = entry + (atr * 2.5)  # 2.5 ATR target

                    opp = self.create_opportunity(
                        symbol=symbol,
                        market=market,
                        direction=Direction.LONG,
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        edge_data={
                            "tom_day": window["current_day"],
                            "days_remaining": window["days_remaining"],
                            "window_start": str(window["window_start"]),
                            "window_end": str(window["window_end"]),
                            "atr": atr,
                        },
                        secondary_edges=[],
                    )

                    opportunities.append(opp)
                    logger.info(
                        "TOM signal: %s LONG @ %.2f, day %s, %s days left",
                        symbol, entry, window["current_day"], window["days_remaining"],
                    )

                except Exception as e:
                    logger.error("TOM scanner error for %s: %s", symbol, e)
                    continue

        return opportunities
