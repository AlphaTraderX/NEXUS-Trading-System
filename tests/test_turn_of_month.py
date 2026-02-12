"""
Tests for TurnOfMonthScanner.

Validates calendar logic, window detection, and signal generation
for the Turn of Month edge.
"""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, AsyncMock

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

from nexus.scanners.turn_of_month import (
    TurnOfMonthScanner,
    US_MARKET_HOLIDAYS,
)
from nexus.core.enums import Direction, EdgeType, Market


ET = ZoneInfo("America/New_York")


@pytest.fixture
def scanner():
    return TurnOfMonthScanner()


class TestIsTradingDay:
    """Test is_trading_day — weekends and holidays return False."""

    def test_normal_weekday_is_trading(self, scanner):
        """Wednesday 2024-06-05 is a normal trading day."""
        assert scanner.is_trading_day(date(2024, 6, 5)) is True

    def test_saturday_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2024, 6, 1)) is False

    def test_sunday_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2024, 6, 2)) is False

    def test_christmas_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2024, 12, 25)) is False

    def test_mlk_day_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2024, 1, 15)) is False

    def test_independence_day_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2024, 7, 4)) is False

    def test_good_friday_2025_not_trading(self, scanner):
        assert scanner.is_trading_day(date(2025, 4, 18)) is False

    def test_day_after_holiday_is_trading(self, scanner):
        """Day after MLK Day (2024-01-16, Tuesday) is trading."""
        assert scanner.is_trading_day(date(2024, 1, 16)) is True


class TestGetLastTradingDayOfMonth:
    """Test get_last_trading_day_of_month handles month boundaries."""

    def test_june_2024_ends_friday(self, scanner):
        """June 30 2024 is Sunday, so last trading day is Friday June 28."""
        assert scanner.get_last_trading_day_of_month(2024, 6) == date(2024, 6, 28)

    def test_march_2024_good_friday(self, scanner):
        """March 31 2024 is Sunday, March 30 is Saturday, March 29 is Good Friday.
        Last trading day should be March 28 (Thursday)."""
        assert scanner.get_last_trading_day_of_month(2024, 3) == date(2024, 3, 28)

    def test_january_2024(self, scanner):
        """January 31 2024 is Wednesday — that's a trading day."""
        assert scanner.get_last_trading_day_of_month(2024, 1) == date(2024, 1, 31)

    def test_december_2024(self, scanner):
        """December 31 2024 is Tuesday — trading day (Christmas is Dec 25)."""
        assert scanner.get_last_trading_day_of_month(2024, 12) == date(2024, 12, 31)

    def test_november_2024(self, scanner):
        """November 30 2024 is Saturday, so last trading day is Friday Nov 29."""
        assert scanner.get_last_trading_day_of_month(2024, 11) == date(2024, 11, 29)


class TestGetTomWindow:
    """Test get_tom_window correctly identifies days -1, 1, 2, 3."""

    def test_last_day_of_month_is_day_minus_1(self, scanner):
        """Last trading day of June 2024 (June 28) = Day -1."""
        window = scanner.get_tom_window(date(2024, 6, 28))
        assert window["in_window"] is True
        assert window["current_day"] == -1
        assert window["days_remaining"] == 4

    def test_first_trading_day_is_day_1(self, scanner):
        """First trading day of July 2024 (July 1, Monday) = Day 1."""
        window = scanner.get_tom_window(date(2024, 7, 1))
        assert window["in_window"] is True
        assert window["current_day"] == 1
        assert window["days_remaining"] == 3

    def test_second_trading_day_is_day_2(self, scanner):
        """Second trading day of July 2024 (July 2) = Day 2."""
        window = scanner.get_tom_window(date(2024, 7, 2))
        assert window["in_window"] is True
        assert window["current_day"] == 2
        assert window["days_remaining"] == 2

    def test_third_trading_day_is_day_3(self, scanner):
        """Third trading day of July 2024 (July 3) = Day 3."""
        window = scanner.get_tom_window(date(2024, 7, 3))
        assert window["in_window"] is True
        assert window["current_day"] == 3
        assert window["days_remaining"] == 1

    def test_mid_month_not_in_window(self, scanner):
        """June 15 2024 is not in any TOM window."""
        window = scanner.get_tom_window(date(2024, 6, 15))
        assert window["in_window"] is False
        assert window["current_day"] is None

    def test_end_of_current_month_starts_new_window(self, scanner):
        """Last trading day of current month starts the next TOM window."""
        # July 31, 2024 is Wednesday — last trading day of July
        window = scanner.get_tom_window(date(2024, 7, 31))
        assert window["in_window"] is True
        assert window["current_day"] == -1
        assert window["window_start"] == date(2024, 7, 31)

    def test_window_start_and_end_dates(self, scanner):
        """Window start = last trading day prev month, end = day 3 current month."""
        window = scanner.get_tom_window(date(2024, 7, 1))
        assert window["window_start"] == date(2024, 6, 28)
        # Day 3 of July 2024 = July 3 (Wed)
        assert window["window_end"] == date(2024, 7, 3)


class TestIsActive:
    """Test is_active returns True during window, False outside."""

    def test_active_on_last_trading_day(self, scanner):
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        assert scanner.is_active(ts) is True

    def test_active_on_first_trading_day(self, scanner):
        ts = datetime(2024, 7, 1, 10, 0, tzinfo=ET)
        assert scanner.is_active(ts) is True

    def test_not_active_mid_month(self, scanner):
        ts = datetime(2024, 6, 15, 10, 0, tzinfo=ET)
        assert scanner.is_active(ts) is False

    def test_active_on_day_3(self, scanner):
        ts = datetime(2024, 7, 3, 10, 0, tzinfo=ET)
        assert scanner.is_active(ts) is True


class TestScan:
    """Test scan generates LONG opportunities for SPY/QQQ/IWM during window."""

    @pytest.fixture
    def tom_scanner(self):
        """Scanner with no data provider (uses placeholders)."""
        return TurnOfMonthScanner()

    @pytest.mark.asyncio
    async def test_scan_in_window_generates_signals(self, tom_scanner):
        """During TOM window, scan should return LONG opportunities."""
        # June 28, 2024 is last trading day of June = Day -1
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        assert len(opps) > 0
        symbols = [o.symbol for o in opps]
        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "IWM" in symbols

    @pytest.mark.asyncio
    async def test_scan_outside_window_returns_empty(self, tom_scanner):
        """Outside TOM window, scan should return empty list."""
        ts = datetime(2024, 6, 15, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)
        assert opps == []

    @pytest.mark.asyncio
    async def test_all_signals_are_long(self, tom_scanner):
        """TOM is always LONG."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        for opp in opps:
            assert opp.direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_edge_data_has_tom_day(self, tom_scanner):
        """Edge data should include tom_day and days_remaining."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        for opp in opps:
            assert "tom_day" in opp.edge_data
            assert opp.edge_data["tom_day"] == -1
            assert "days_remaining" in opp.edge_data
            assert opp.edge_data["days_remaining"] == 4

    @pytest.mark.asyncio
    async def test_edge_type_is_turn_of_month(self, tom_scanner):
        """Primary edge should be TURN_OF_MONTH."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        for opp in opps:
            assert opp.primary_edge == EdgeType.TURN_OF_MONTH

    @pytest.mark.asyncio
    async def test_stop_loss_below_entry(self, tom_scanner):
        """For LONG, stop should be below entry."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        for opp in opps:
            assert opp.stop_loss < opp.entry_price

    @pytest.mark.asyncio
    async def test_take_profit_above_entry(self, tom_scanner):
        """For LONG, target should be above entry."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        for opp in opps:
            assert opp.take_profit > opp.entry_price

    @pytest.mark.asyncio
    async def test_futures_included(self, tom_scanner):
        """ES and NQ should be in the signals."""
        ts = datetime(2024, 6, 28, 10, 0, tzinfo=ET)
        opps = await tom_scanner.scan(timestamp=ts)

        symbols = [o.symbol for o in opps]
        assert "ES" in symbols
        assert "NQ" in symbols
