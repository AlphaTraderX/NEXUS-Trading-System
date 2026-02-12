"""
Tests for ExitManager.

Validates exit logic matches backtest engine:
- Overnight: MOO exit (always exits next cycle)
- RSI: Indicator exit (RSI>50 OR close>SMA5 for longs)
- Stop loss / take profit
- Max hold time expiry
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np

from nexus.execution.exit_manager import (
    ExitManager,
    ExitResult,
    MAX_HOLD_DAYS,
    AUCTION_SPREAD_PCT,
)


class TestExitManagerConfig:
    """Test ExitManager static configuration."""

    def test_max_hold_days(self):
        assert MAX_HOLD_DAYS["overnight_premium"] == 1
        assert MAX_HOLD_DAYS["gap_fill"] == 5
        assert MAX_HOLD_DAYS["vwap_deviation"] == 10
        assert MAX_HOLD_DAYS["rsi_extreme"] == 10

    def test_auction_spread(self):
        assert AUCTION_SPREAD_PCT == 0.005


class TestOvernightExit:
    """Test overnight premium MOO exit."""

    @pytest.fixture
    def exit_manager(self):
        return ExitManager(data_provider=MagicMock())

    @pytest.mark.asyncio
    async def test_overnight_always_exits(self, exit_manager):
        """Overnight positions always exit on next cycle."""
        # Mock bars with an open price
        bars = pd.DataFrame({
            "open": [502.0, 505.0],
            "high": [503.0, 506.0],
            "low": [501.0, 504.0],
            "close": [502.5, 505.5],
        })
        exit_manager._get_bars = AsyncMock(return_value=bars)

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 475.0,
            "take_profit": 550.0,
            "shares": 10,
            "edge": "overnight_premium",
            "opened_at": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(),
        }

        exits = await exit_manager.check_exits([pos])

        assert len(exits) == 1
        assert exits[0].exit_reason == "moo_exit"
        # Exit at last bar's open (505.0)
        assert exits[0].exit_price == 505.0
        # Profit: (505 - 500) * 10 = 50
        assert exits[0].gross_pnl == 50.0

    @pytest.mark.asyncio
    async def test_overnight_no_data_exits_flat(self, exit_manager):
        """Overnight with no data provider exits flat (no data)."""
        exit_manager_nodata = ExitManager(data_provider=None)

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 475.0,
            "take_profit": 550.0,
            "shares": 10,
            "edge": "overnight_premium",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        exits = await exit_manager_nodata.check_exits([pos])

        assert len(exits) == 1
        assert exits[0].exit_reason == "moo_exit_no_data"
        assert exits[0].exit_price == 500.0
        assert exits[0].net_pnl == 0.0


class TestRSIExit:
    """Test RSI indicator-based exit."""

    @pytest.fixture
    def exit_manager(self):
        return ExitManager(data_provider=MagicMock())

    @pytest.mark.asyncio
    async def test_rsi_exit_when_rsi_above_50(self, exit_manager):
        """RSI long position exits when RSI(2) > 50."""
        # Create bars with a recovery (RSI will be > 50)
        np.random.seed(42)
        n = 30
        # Strong recovery after pullback
        closes = list(np.linspace(100, 95, 15)) + list(np.linspace(95, 110, 15))
        bars = pd.DataFrame({
            "open": [c * 0.999 for c in closes],
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
        })
        exit_manager._get_bars = AsyncMock(return_value=bars)

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 95.0,
            "stop_loss": 85.0,
            "take_profit": 115.0,
            "shares": 10,
            "edge": "rsi_extreme",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        exits = await exit_manager.check_exits([pos])

        # Should exit because RSI recovered above 50 OR close > SMA(5)
        assert len(exits) == 1
        assert exits[0].exit_reason == "indicator_exit"

    @pytest.mark.asyncio
    async def test_rsi_exit_when_close_above_sma5(self, exit_manager):
        """RSI long position exits when close > SMA(5)."""
        # Strong uptrend — close always above SMA5
        closes = list(np.linspace(100, 120, 30))
        bars = pd.DataFrame({
            "open": [c * 0.999 for c in closes],
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
        })
        exit_manager._get_bars = AsyncMock(return_value=bars)

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 90.0,
            "take_profit": 130.0,
            "shares": 10,
            "edge": "rsi_extreme",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        exits = await exit_manager.check_exits([pos])

        assert len(exits) == 1
        assert exits[0].exit_reason == "indicator_exit"


class TestStopLoss:
    """Test stop-loss exit."""

    def test_long_stop_hit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "take_profit": 520.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        result = em._check_stop_target_time(pos, 489.0)

        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert result.exit_price == 490.0
        # Loss: (490 - 500) * 10 = -100
        assert result.gross_pnl == -100.0

    def test_short_stop_hit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "short",
            "entry_price": 500.0,
            "stop_loss": 510.0,
            "take_profit": 480.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        result = em._check_stop_target_time(pos, 511.0)

        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert result.exit_price == 510.0
        # Loss: (500 - 510) * 10 = -100
        assert result.gross_pnl == -100.0

    def test_long_price_above_stop_no_exit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "take_profit": 520.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        result = em._check_stop_target_time(pos, 495.0)

        assert result is None


class TestTakeProfit:
    """Test take-profit exit."""

    def test_long_target_hit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "take_profit": 520.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        result = em._check_stop_target_time(pos, 521.0)

        assert result is not None
        assert result.exit_reason == "take_profit"
        assert result.exit_price == 520.0
        # Profit: (520 - 500) * 10 = 200
        assert result.gross_pnl == 200.0

    def test_short_target_hit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "short",
            "entry_price": 500.0,
            "stop_loss": 510.0,
            "take_profit": 480.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        result = em._check_stop_target_time(pos, 479.0)

        assert result is not None
        assert result.exit_reason == "take_profit"
        assert result.exit_price == 480.0
        # Profit: (500 - 480) * 10 = 200
        assert result.gross_pnl == 200.0


class TestTimeExpiry:
    """Test max hold time expiry."""

    def test_gap_expires_after_5_days(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "take_profit": 520.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat(),
        }

        # Price between stop and target
        result = em._check_stop_target_time(pos, 505.0)

        assert result is not None
        assert result.exit_reason == "time_expiry"

    def test_gap_within_hold_period_no_exit(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 490.0,
            "take_profit": 520.0,
            "shares": 10,
            "edge": "gap_fill",
            "opened_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
        }

        # Price between stop and target, within hold period
        result = em._check_stop_target_time(pos, 505.0)

        assert result is None

    def test_rsi_expires_after_10_days(self):
        em = ExitManager()

        pos = {
            "symbol": "SPY",
            "direction": "long",
            "entry_price": 500.0,
            "stop_loss": 450.0,
            "take_profit": 600.0,
            "shares": 10,
            "edge": "rsi_extreme",
            "opened_at": (datetime.now(timezone.utc) - timedelta(days=11)).isoformat(),
        }

        result = em._check_stop_target_time(pos, 505.0)

        assert result is not None
        assert result.exit_reason == "time_expiry"


class TestPnLCalculation:
    """Test P&L calculation helper."""

    def test_long_profit(self):
        assert ExitManager._calc_pnl(100, 110, 10, True) == 100.0

    def test_long_loss(self):
        assert ExitManager._calc_pnl(100, 90, 10, True) == -100.0

    def test_short_profit(self):
        assert ExitManager._calc_pnl(100, 90, 10, False) == 100.0

    def test_short_loss(self):
        assert ExitManager._calc_pnl(100, 110, 10, False) == -100.0


class TestRSIExitIndicators:
    """Test RSI exit indicator calculation."""

    def test_computes_rsi_2_and_sma_5(self):
        """Should compute RSI(2) and SMA(5) matching backtest."""
        np.random.seed(42)
        closes = 500 + np.cumsum(np.random.randn(30))
        bars = pd.DataFrame({
            "close": closes,
            "high": closes + 1,
            "low": closes - 1,
        })

        result = ExitManager._calculate_rsi_exit_indicators(bars)

        assert "rsi_2" in result.columns
        assert "sma_5" in result.columns

        # SMA(5) should match rolling mean
        expected = result["close"].rolling(5).mean().iloc[-1]
        assert abs(result["sma_5"].iloc[-1] - expected) < 0.01
