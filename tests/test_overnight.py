"""
Tests for OvernightPremiumScanner.

Validates that scanner matches backtest logic:
- Always LONG (overnight premium is long-only)
- 200 SMA bull market filter
- Skip Fridays (weekend gap risk)
- 10 validated symbols: SPY, QQQ + high-beta tech
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np

from nexus.scanners.overnight import OvernightPremiumScanner
from nexus.core.enums import Direction, EdgeType, Market


class TestOvernightPremiumConfig:
    """Test OvernightPremiumScanner static configuration."""

    @pytest.fixture
    def scanner(self):
        return OvernightPremiumScanner(data_provider=MagicMock())

    def test_edge_type(self, scanner):
        assert scanner.edge_type == EdgeType.OVERNIGHT_PREMIUM

    def test_symbols_match_backtest(self, scanner):
        expected = ["SPY", "QQQ", "TSLA", "NVDA", "AMD", "AAPL", "GOOGL", "META", "NFLX", "CRM"]
        actual = scanner.INSTRUMENTS[Market.US_STOCKS]
        assert actual == expected
        assert len(actual) == 10

    def test_score_is_60(self, scanner):
        assert scanner.score == 60

    def test_notional_is_20_percent(self, scanner):
        assert scanner.notional_pct == 20

    def test_sma_period(self, scanner):
        assert scanner.sma_period == 200


class TestOvernightFridaySkip:
    """Test Friday skip logic."""

    @pytest.fixture
    def scanner(self):
        return OvernightPremiumScanner(data_provider=MagicMock())

    def test_friday_inactive(self, scanner):
        """Scanner should not be active on Fridays."""
        friday = datetime(2024, 1, 5, 20, 0, tzinfo=timezone.utc)
        assert scanner.is_active(friday) is False

    def test_saturday_inactive(self, scanner):
        """Scanner should not be active on weekends."""
        saturday = datetime(2024, 1, 6, 20, 0, tzinfo=timezone.utc)
        assert scanner.is_active(saturday) is False

    def test_sunday_inactive(self, scanner):
        sunday = datetime(2024, 1, 7, 20, 0, tzinfo=timezone.utc)
        assert scanner.is_active(sunday) is False

    def test_thursday_active(self, scanner):
        """Scanner should be active Mon-Thu."""
        thursday = datetime(2024, 1, 4, 20, 0, tzinfo=timezone.utc)
        assert scanner.is_active(thursday) is True

    def test_monday_active(self, scanner):
        monday = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)
        assert scanner.is_active(monday) is True


class TestOvernightSignalLogic:
    """Test overnight signal generation."""

    @pytest.fixture
    def scanner(self):
        return OvernightPremiumScanner(data_provider=MagicMock())

    def _make_bars(self, price_above_sma: bool, n: int = 210, weekday: int = 2):
        """Create mock bars for overnight scanning."""
        dates = pd.bdate_range(end="2024-06-05", periods=n, freq="D")

        # Ensure last bar falls on the target weekday
        while dates[-1].weekday() != weekday:
            dates = pd.bdate_range(end=dates[-1] - pd.Timedelta(days=1), periods=n, freq="D")

        if price_above_sma:
            # Steady uptrend — price well above SMA200
            closes = np.linspace(80, 120, n)
        else:
            # Downtrend — price below SMA200
            closes = np.linspace(120, 80, n)

        return pd.DataFrame(
            {
                "open": closes * 0.999,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        )

    @pytest.mark.asyncio
    async def test_bull_market_generates_long(self, scanner):
        """Price > SMA200 on non-Friday → LONG signal."""
        bars = self._make_bars(price_above_sma=True, weekday=2)  # Wednesday
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("SPY", Market.US_STOCKS)

        assert opp is not None
        assert opp.direction == Direction.LONG
        assert opp.raw_score == 60
        assert opp.edge_data["notional_pct"] == 20

    @pytest.mark.asyncio
    async def test_bear_market_rejected(self, scanner):
        """Price < SMA200 → no signal (bear market filter)."""
        bars = self._make_bars(price_above_sma=False, weekday=2)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("SPY", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_friday_bar_rejected(self, scanner):
        """Last bar on Friday → no signal (weekend risk)."""
        bars = self._make_bars(price_above_sma=True, weekday=4)  # Friday
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("SPY", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_insufficient_data(self, scanner):
        """< 201 bars → no signal."""
        bars = self._make_bars(price_above_sma=True, n=100)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("SPY", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_edge_data_fields(self, scanner):
        """Edge data must include required keys."""
        bars = self._make_bars(price_above_sma=True, weekday=2)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("SPY", Market.US_STOCKS)

        assert opp is not None
        assert "regime_filter" in opp.edge_data
        assert "sma_200" in opp.edge_data
        assert "notional_pct" in opp.edge_data
        assert opp.edge_data["notional_pct"] == 20


class TestOvernightScan:
    """Test full scan across all symbols."""

    @pytest.fixture
    def scanner(self):
        return OvernightPremiumScanner(data_provider=MagicMock())

    @pytest.mark.asyncio
    async def test_scan_generates_multiple_signals(self, scanner):
        """Full scan should generate signals for multiple symbols."""
        dates = pd.bdate_range(end="2024-06-05", periods=210, freq="D")
        # Wed
        while dates[-1].weekday() != 2:
            dates = pd.bdate_range(end=dates[-1] - pd.Timedelta(days=1), periods=210, freq="D")

        closes = np.linspace(80, 120, 210)
        mock_bars = pd.DataFrame(
            {
                "open": closes * 0.999,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "volume": [1_000_000] * 210,
            },
            index=dates,
        )
        scanner.get_bars_safe = AsyncMock(return_value=mock_bars)

        signals = await scanner.scan()

        assert len(signals) == 10
        for sig in signals:
            assert sig.direction == Direction.LONG
            assert sig.raw_score == 60
