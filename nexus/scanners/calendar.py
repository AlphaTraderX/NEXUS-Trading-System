"""
Turn of Month (TOM) effect scanner.

Academic edge: 100% of S&P 500 equity premium in a 4-day window
(McConnell & Xu, 2008). Scanner detects when we're in the TOM window
and emits LONG opportunities on index ETFs.
"""

import calendar
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from .base import BaseScanner
from nexus.core.enums import Direction, EdgeType, Market
from nexus.core.models import Opportunity


class TurnOfMonthScanner(BaseScanner):
    """
    Turn of Month effect scanner.

    EDGE: 100% of S&P 500 equity premium earned in 4-day window
    SOURCE: McConnell & Xu (2008) academic study

    TIMING:
    - Day -1: Last trading day of previous month
    - Day 1-3: First three trading days of new month

    DIRECTION: Always LONG on indices/large-cap stocks

    INSTRUMENTS: SPY, QQQ, IWM (ETFs tracking major indices)
    """

    def __init__(self, data_provider=None, settings=None) -> None:
        """
        Initialize the Turn of Month scanner.

        Args:
            data_provider: Data source for prices/bars (optional; uses base placeholder if None).
            settings: Scanner/config object (stored for subclasses and future use).
        """
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.TURN_OF_MONTH
        self.markets = [Market.US_STOCKS, Market.US_FUTURES]
        self.instruments = ["SPY", "QQQ", "IWM"]

    def is_active(self, timestamp: datetime) -> bool:
        """
        Check if we're in the TOM window.

        Logic:
        1. Get the current date in US/Eastern timezone
        2. Determine if it's a trading day (Monday-Friday, not a holiday)
        3. Calculate if we're in the TOM window:
           - Last trading day of month = TOM Day -1
           - First three trading days of month = TOM Days 1, 2, 3

        For simplicity, use this approach:
        - If day of month <= 3: We're in TOM (days 1-3)
        - If it's last trading day of month: We're in TOM (day -1)

        To check last trading day:
        - Get last day of month
        - If it's weekend, move back to Friday
        - If today equals that day, we're in TOM day -1

        Returns True if in TOM window, False otherwise.
        """
        return self._get_tom_day(timestamp) != 0

    def _get_tom_day(self, timestamp: datetime) -> int:
        """
        Determine which TOM day we're in.

        Returns:
        - -1: Last trading day of previous month
        - 1, 2, 3: First three trading days of new month
        - 0: Not in TOM window
        """
        eastern = pytz.timezone("US/Eastern")
        if timestamp.tzinfo is None:
            ts_local = pytz.utc.localize(timestamp).astimezone(eastern)
        else:
            ts_local = timestamp.astimezone(eastern)
        d = ts_local.date()

        if not self._is_trading_day(d):
            return 0

        y, m, day = d.year, d.month, d.day
        last_trading = self._get_last_trading_day_of_month(y, m)

        # Day -1: last trading day of this month
        if day == last_trading:
            return -1

        # Days 1, 2, 3: first three trading days of this month
        for n in (1, 2, 3):
            nth_date = self._get_nth_trading_day_of_month(y, m, n)
            if nth_date is not None and d == nth_date:
                return n

        return 0

    def _is_trading_day(self, d: date) -> bool:
        """
        Check if date is a trading day.

        Simple check: Monday (0) through Friday (4).
        NOTE: For production, would add holiday calendar.
        """
        # weekday(): Monday=0, Sunday=6
        return d.weekday() < 5

    def _get_last_trading_day_of_month(self, year: int, month: int) -> int:
        """
        Get the last trading day of the given month.

        Uses calendar.monthrange to get last day, then adjusts for weekends.
        Returns the day number (1-31).
        """
        _, last_day = calendar.monthrange(year, month)
        w = calendar.weekday(year, month, last_day)  # Monday=0, Sunday=6
        if w == 5:  # Saturday -> Friday
            return last_day - 1
        if w == 6:  # Sunday -> Friday
            return last_day - 2
        return last_day

    def _get_nth_trading_day_of_month(self, year: int, month: int, n: int) -> Optional[date]:
        """
        Get the date of the n-th trading day of the month (n=1,2,3,...).

        Returns None if n is out of range (e.g. month has fewer than n trading days).
        """
        if n < 1:
            return None
        _, last_day = calendar.monthrange(year, month)
        count = 0
        for day in range(1, last_day + 1):
            w = calendar.weekday(year, month, day)
            if w < 5:  # Mon-Fri
                count += 1
                if count == n:
                    return date(year, month, day)
        return None

    def _tom_days_remaining(self, tom_day: int) -> int:
        """Number of TOM window days left (including current)."""
        if tom_day == 0:
            return 0
        if tom_day == -1:
            return 4  # -1, 1, 2, 3
        if tom_day == 1:
            return 3
        if tom_day == 2:
            return 2
        return 1  # tom_day == 3

    async def scan(self) -> List[Opportunity]:
        """
        Scan for TOM opportunities.

        Logic:
        1. Check if currently in TOM window using is_active()
        2. If not active, return empty list
        3. For each instrument in self.instruments:
           a. Get current price via base get_current_price()
           b. Get bars and calculate ATR via base get_bars() and calculate_atr()
           c. Direction = LONG
           d. Calculate entry, stop, target via calculate_entry_stop_target()
           e. Create Opportunity with edge_data (tom_day, window_remaining_days, atr)
           f. Append to opportunities list
        4. Return opportunities list
        """
        now = datetime.now(pytz.utc)

        if not self.is_active(now):
            return []

        opportunities: List[Opportunity] = []
        tom_day = self._get_tom_day(now)

        for symbol in self.instruments:
            # Use base class methods instead of hardcoded values
            current_price = await self.get_current_price(symbol)
            bars = await self.get_bars(symbol, "1D", 20)
            atr = self.calculate_atr(bars, 14)

            # TOM is always LONG
            entry, stop, target = self.calculate_entry_stop_target(
                current_price=current_price,
                direction=Direction.LONG,
                atr=atr,
                stop_multiplier=1.5,
                target_multiplier=2.5,
            )

            # TOM window: day -1 through day 3 (4 days total). Remaining = days left in window.
            if tom_day == -1:
                remaining = 4
            elif tom_day == 1:
                remaining = 3
            elif tom_day == 2:
                remaining = 2
            elif tom_day == 3:
                remaining = 1
            else:
                remaining = 0

            opp = self.create_opportunity(
                symbol=symbol,
                market=self._get_market_for_symbol(symbol),
                direction=Direction.LONG,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                edge_data={
                    "tom_day": tom_day,
                    "window_remaining_days": remaining,
                    "atr": atr,
                    "atr_percent": (atr / current_price) * 100,
                },
            )
            opportunities.append(opp)

        return opportunities

    def get_tom_status(self) -> Dict[str, Any]:
        """
        Return status for debugging/monitoring.

        Returns dict with:
        - is_active: bool
        - tom_day: int (-1, 0, 1, 2, 3)
        - days_remaining: int
        - next_window: datetime (when next TOM window starts, in UTC)
        """
        now = datetime.now(pytz.utc)
        eastern = pytz.timezone("US/Eastern")
        now_eastern = now.astimezone(eastern)
        tom_day = self._get_tom_day(now)
        days_remaining = self._tom_days_remaining(tom_day)

        # Next TOM window: next occurrence of "last trading day of month"
        y, m = now_eastern.year, now_eastern.month
        last_day_num = self._get_last_trading_day_of_month(y, m)
        last_trading_date = date(y, m, last_day_num)
        last_trading_dt = eastern.localize(
            datetime.combine(last_trading_date, time(9, 30))
        ).astimezone(pytz.utc)

        if now_eastern.date() <= last_trading_date and last_trading_dt > now:
            next_window = last_trading_dt
        else:
            # Next month
            if m == 12:
                next_y, next_m = y + 1, 1
            else:
                next_y, next_m = y, m + 1
            next_last = self._get_last_trading_day_of_month(next_y, next_m)
            next_date = date(next_y, next_m, next_last)
            next_window = eastern.localize(
                datetime.combine(next_date, time(9, 30))
            ).astimezone(pytz.utc)

        return {
            "is_active": tom_day != 0,
            "tom_day": tom_day,
            "days_remaining": days_remaining,
            "next_window": next_window,
        }


class MonthEndScanner(BaseScanner):
    """
    Month-End Rebalancing scanner.

    EDGE: $7.5 trillion in pension fund flows create predictable moves
    SOURCE: Institutional rebalancing patterns documented in academic research

    TIMING: Last 2 trading days of month

    DIRECTION:
    - If stocks outperformed bonds this month: Pension funds SELL stocks (SHORT bias)
    - If bonds outperformed stocks this month: Pension funds BUY stocks (LONG bias)
    - For v1, we default to LONG bias (mean reversion into month-end weakness)

    INSTRUMENTS: SPY, IWM (broad market ETFs most affected by rebalancing)
    """

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.MONTH_END
        self.markets = [Market.US_STOCKS]
        self.instruments = ["SPY", "IWM"]

    def is_active(self, timestamp: datetime) -> bool:
        """Check if we're in month-end window (last 2 trading days)."""
        eastern = pytz.timezone("US/Eastern")
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        local_time = timestamp.astimezone(eastern)

        if not self._is_trading_day(local_time.date()):
            return False

        return self._get_month_end_day(timestamp) != 0

    def _get_month_end_day(self, timestamp: datetime) -> int:
        """
        Determine which month-end day we're in.

        Returns:
        - 2: Second-to-last trading day of month
        - 1: Last trading day of month
        - 0: Not in month-end window
        """
        eastern = pytz.timezone("US/Eastern")
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        local_time = timestamp.astimezone(eastern)

        year = local_time.year
        month = local_time.month
        day = local_time.day

        last_trading_day = self._get_last_trading_day_of_month(year, month)
        second_to_last = self._get_second_to_last_trading_day(year, month)

        if day == last_trading_day:
            return 1
        elif day == second_to_last:
            return 2
        else:
            return 0

    def _is_trading_day(self, check_date) -> bool:
        """Check if date is a trading day (Mon-Fri)."""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return check_date.weekday() < 5

    def _get_last_trading_day_of_month(self, year: int, month: int) -> int:
        """Get the last trading day of the given month."""
        last_day = calendar.monthrange(year, month)[1]
        check_date = date(year, month, last_day)

        while check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            check_date -= timedelta(days=1)

        return check_date.day

    def _get_second_to_last_trading_day(self, year: int, month: int) -> int:
        """Get the second-to-last trading day of the month."""
        last_trading_day = self._get_last_trading_day_of_month(year, month)
        check_date = date(year, month, last_trading_day) - timedelta(days=1)

        while check_date.weekday() >= 5:
            check_date -= timedelta(days=1)

        return check_date.day

    async def scan(self) -> List[Opportunity]:
        """Scan for month-end rebalancing opportunities."""
        now = datetime.now(pytz.utc)

        if not self.is_active(now):
            return []

        opportunities = []
        month_end_day = self._get_month_end_day(now)

        for symbol in self.instruments:
            current_price = await self.get_current_price(symbol)
            bars = await self.get_bars(symbol, "1D", 20)
            atr = self.calculate_atr(bars, 14)

            # Default to LONG for v1 (buying into month-end weakness)
            # In v2, could check stock vs bond performance for direction
            direction = Direction.LONG

            entry, stop, target = self.calculate_entry_stop_target(
                current_price=current_price,
                direction=direction,
                atr=atr,
                stop_multiplier=1.5,
                target_multiplier=2.0,  # Slightly tighter target for shorter hold
            )

            opp = self.create_opportunity(
                symbol=symbol,
                market=self._get_market_for_symbol(symbol),
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                edge_data={
                    "month_end_day": month_end_day,
                    "days_remaining": month_end_day,  # 2 or 1 days left
                    "atr": atr,
                    "atr_percent": (atr / current_price) * 100,
                },
            )
            opportunities.append(opp)

        return opportunities

    def get_month_end_status(self) -> Dict[str, Any]:
        """Get current month-end window status."""
        now = datetime.now(pytz.utc)
        eastern = pytz.timezone("US/Eastern")
        local_now = now.astimezone(eastern)

        month_end_day = self._get_month_end_day(now)
        is_active = month_end_day != 0

        # Calculate next window start (second-to-last trading day of current/next month)
        year = local_now.year
        month = local_now.month

        if is_active or local_now.day > self._get_last_trading_day_of_month(year, month):
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

        next_second_to_last = self._get_second_to_last_trading_day(year, month)
        next_window = datetime(year, month, next_second_to_last, 9, 30)
        next_window = eastern.localize(next_window).astimezone(pytz.utc)

        return {
            "is_active": is_active,
            "month_end_day": month_end_day,
            "days_remaining": month_end_day if is_active else 0,
            "next_window": next_window,
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test TOM Scanner
        tom = TurnOfMonthScanner()
        print("=" * 50)
        print("TURN OF MONTH SCANNER")
        print("=" * 50)
        print(f"Status: {tom.get_tom_status()}")
        tom_opps = await tom.scan()
        print(f"Opportunities found: {len(tom_opps)}")
        for opp in tom_opps:
            print(f"  {opp.symbol}: Entry={opp.entry_price:.2f}, Stop={opp.stop_loss:.2f}, Target={opp.take_profit:.2f}")

        # Test Month End Scanner
        me = MonthEndScanner()
        print("\n" + "=" * 50)
        print("MONTH END SCANNER")
        print("=" * 50)
        print(f"Status: {me.get_month_end_status()}")
        me_opps = await me.scan()
        print(f"Opportunities found: {len(me_opps)}")
        for opp in me_opps:
            print(f"  {opp.symbol}: Entry={opp.entry_price:.2f}, Stop={opp.stop_loss:.2f}, Target={opp.take_profit:.2f}")

    asyncio.run(test())
