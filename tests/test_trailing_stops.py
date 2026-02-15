"""Tests for trailing stop + breakeven logic in TradeSimulator."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from nexus.backtest.trade_simulator import (
    ExitReason,
    SimulatedTrade,
    TradeSimulator,
    TrailingStopConfig,
    score_to_tier,
    tier_multiplier,
)
from nexus.core.enums import Direction, EdgeType, Market
from nexus.core.models import Opportunity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opportunity(
    direction="long",
    entry_price=100.0,
    stop_loss=95.0,
    take_profit=115.0,
    score=70,
    edge=EdgeType.GAP_FILL,
    edge_data=None,
) -> Opportunity:
    return Opportunity(
        id="test-opp-001",
        detected_at=datetime.now(timezone.utc),
        scanner="test",
        symbol="SPY",
        market=Market.US_STOCKS,
        direction=Direction.LONG if direction == "long" else Direction.SHORT,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        primary_edge=edge,
        edge_data=edge_data or {},
        raw_score=score,
    )


def _make_trending_bars(n=30, start=100.0, end=120.0, seed=42) -> pd.DataFrame:
    """Create upward-trending bars for testing trailing stops."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="D")
    closes = np.linspace(start, end, n)
    return pd.DataFrame(
        {
            "open": closes * 0.998,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


def _make_reversal_bars(
    n=30, start=100.0, peak=112.0, final=96.0, peak_bar=15, seed=42,
) -> pd.DataFrame:
    """Create bars that go up to peak then reverse down (to test trailing stop trigger)."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="D")

    # Up phase
    up = np.linspace(start, peak, peak_bar)
    # Down phase
    down = np.linspace(peak, final, n - peak_bar)
    closes = np.concatenate([up, down])

    return pd.DataFrame(
        {
            "open": closes * 0.998,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


def _make_flat_bars(n=30, price=100.0) -> pd.DataFrame:
    """Create flat bars (no trailing should trigger)."""
    dates = pd.bdate_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price * 1.002] * n,
            "low": [price * 0.998] * n,
            "close": [price] * n,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrailingStopConfig:
    """Test TrailingStopConfig dataclass defaults."""

    def test_defaults(self):
        cfg = TrailingStopConfig()
        assert cfg.atr_trail_multiplier == 1.5
        assert cfg.breakeven_atr_mult == 1.0
        assert cfg.trailing_activation_atr == 1.5
        assert cfg.enabled is True

    def test_custom_values(self):
        cfg = TrailingStopConfig(
            atr_trail_multiplier=2.0,
            breakeven_atr_mult=0.5,
            trailing_activation_atr=1.0,
            enabled=False,
        )
        assert cfg.atr_trail_multiplier == 2.0
        assert cfg.breakeven_atr_mult == 0.5
        assert cfg.enabled is False


class TestTrailingStopNone:
    """Test that trailing_config=None preserves v1 behavior."""

    def test_no_trailing_same_as_v1(self):
        """With trailing_config=None, trade should exit via SL/TP/time as before."""
        sim = TradeSimulator(account_balance=10_000)
        bars = _make_trending_bars(n=30, start=100, end=110)
        opp = _make_opportunity(
            direction="long", entry_price=100.0, stop_loss=95.0, take_profit=115.0,
        )

        # No trailing config
        trade = sim.simulate_trade(opp, bars, signal_bar_idx=0)
        assert trade is not None
        # Should NOT be trailing/breakeven exit
        assert trade.exit_reason not in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)

    def test_disabled_trailing_same_as_none(self):
        """TrailingStopConfig with enabled=False should behave like None."""
        sim = TradeSimulator(account_balance=10_000)
        bars = _make_trending_bars(n=30)
        opp = _make_opportunity()

        cfg = TrailingStopConfig(enabled=False)
        trade = sim.simulate_trade(opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=2.0)
        assert trade is not None
        assert trade.exit_reason not in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)


class TestBreakevenStop:
    """Test breakeven stop-loss movement."""

    def test_breakeven_triggers_on_profit(self):
        """When profit >= breakeven_atr_mult * ATR, stop moves to entry."""
        sim = TradeSimulator(account_balance=10_000, max_hold_bars=25)

        # Bars go up then reverse back to entry
        bars = _make_reversal_bars(n=30, start=100, peak=108, final=99, peak_bar=12)
        opp = _make_opportunity(
            direction="long", entry_price=100.0, stop_loss=90.0, take_profit=120.0,
        )
        cfg = TrailingStopConfig(
            breakeven_atr_mult=1.0,
            atr_trail_multiplier=1.5,
            trailing_activation_atr=2.0,  # High activation so trailing doesn't engage
        )
        atr = 3.0  # breakeven triggers at 1.0 * 3.0 = 3.0 profit (price 103)

        trade = sim.simulate_trade(
            opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=atr,
        )
        assert trade is not None
        # Price goes to 108 (well above 103 breakeven threshold)
        # Then reverses to 99 — should hit breakeven stop at ~entry price
        assert trade.exit_reason in (ExitReason.BREAKEVEN_STOP, ExitReason.TRAILING_STOP)
        # Exit price should be near entry, not at original stop of 90
        assert trade.exit_price > 95.0  # Much better than original stop


class TestTrailingStopLong:
    """Test trailing stop for long positions."""

    def test_trailing_stop_locks_profit(self):
        """When price advances enough, trailing stop locks in profit."""
        sim = TradeSimulator(account_balance=10_000, max_hold_bars=25)

        # Price goes up to 115 then drops to 100
        bars = _make_reversal_bars(n=30, start=100, peak=115, final=100, peak_bar=12)
        opp = _make_opportunity(
            direction="long", entry_price=100.0, stop_loss=90.0, take_profit=130.0,  # High TP
        )
        cfg = TrailingStopConfig(
            atr_trail_multiplier=1.5,
            breakeven_atr_mult=1.0,
            trailing_activation_atr=1.5,
        )
        atr = 3.0  # Trailing activates at 1.5 * 3.0 = 4.5 profit (price 104.5)
        # Trail distance = 1.5 * 3.0 = 4.5 below peak

        trade = sim.simulate_trade(
            opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=atr,
        )
        assert trade is not None
        # Should exit via trailing or breakeven stop, not original stop at 90
        assert trade.exit_reason in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)
        # Exit should be much better than original stop of 90
        assert trade.exit_price > 95.0


class TestTrailingStopShort:
    """Test trailing stop for short positions."""

    def test_short_trailing_stop(self):
        """Short trailing stop should trail above the low."""
        sim = TradeSimulator(account_balance=10_000, max_hold_bars=25)

        # Price goes down to 85 then bounces to 100
        bars = _make_reversal_bars(n=30, start=100, peak=85, final=100, peak_bar=12)
        # For short: "peak" is actually the low point
        # Manually make bars that drop then rise
        n = 30
        dates = pd.bdate_range("2024-01-01", periods=n, freq="D")
        down = np.linspace(100, 85, 15)
        up = np.linspace(85, 100, 15)
        closes = np.concatenate([down, up])
        bars = pd.DataFrame(
            {
                "open": closes * 1.002,
                "high": closes * 1.005,
                "low": closes * 0.995,
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        )

        opp = _make_opportunity(
            direction="short", entry_price=100.0, stop_loss=110.0, take_profit=75.0,
        )
        cfg = TrailingStopConfig(
            atr_trail_multiplier=1.5,
            breakeven_atr_mult=1.0,
            trailing_activation_atr=1.5,
        )
        atr = 3.0

        trade = sim.simulate_trade(
            opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=atr,
        )
        assert trade is not None
        # Should exit via trailing stop as price bounces from 85 to 100
        assert trade.exit_reason in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)
        # Exit should be better than original stop at 110
        assert trade.exit_price < 110.0


class TestTrailingStopWithATR:
    """Test ATR requirement for trailing stops."""

    def test_no_atr_disables_trailing(self):
        """If atr is None, trailing config is ignored."""
        sim = TradeSimulator(account_balance=10_000)
        bars = _make_trending_bars(n=30)
        opp = _make_opportunity()

        cfg = TrailingStopConfig()
        trade = sim.simulate_trade(opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=None)
        assert trade is not None
        # No trailing without ATR
        assert trade.exit_reason not in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)

    def test_zero_atr_disables_trailing(self):
        """If atr is 0, trailing config is ignored."""
        sim = TradeSimulator(account_balance=10_000)
        bars = _make_trending_bars(n=30)
        opp = _make_opportunity()

        cfg = TrailingStopConfig()
        trade = sim.simulate_trade(opp, bars, signal_bar_idx=0, trailing_config=cfg, atr=0)
        assert trade is not None
        assert trade.exit_reason not in (ExitReason.TRAILING_STOP, ExitReason.BREAKEVEN_STOP)


class TestExitReasonEnum:
    """Test that new ExitReason values exist."""

    def test_trailing_stop_exists(self):
        assert ExitReason.TRAILING_STOP.value == "trailing_stop"

    def test_breakeven_stop_exists(self):
        assert ExitReason.BREAKEVEN_STOP.value == "breakeven_stop"

    def test_all_exit_reasons(self):
        expected = {
            "stop_loss", "take_profit", "trailing_stop", "breakeven_stop",
            "indicator_exit", "time_expiry", "end_of_data",
        }
        actual = {e.value for e in ExitReason}
        assert expected == actual


class TestStatisticsWithTrailing:
    """Test that statistics correctly count trailing/breakeven exits."""

    def test_statistics_count_trailing_exits(self):
        from nexus.backtest.statistics import StatisticsCalculator
        from datetime import timedelta

        trades = [
            SimulatedTrade(
                opportunity_id=f"t{i}",
                symbol="SPY",
                direction="long",
                entry_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                entry_price=100.0,
                exit_time=datetime(2024, 1, i + 2, tzinfo=timezone.utc),
                exit_price=105.0,
                exit_reason=ExitReason.TRAILING_STOP,
                position_size=100,
                gross_pnl=500,
                costs=2,
                net_pnl=498,
                net_pnl_pct=4.98,
                hold_duration=timedelta(days=1),
                primary_edge="gap_fill",
                score=70,
                score_tier="B",
            )
            for i in range(3)
        ]

        calc = StatisticsCalculator()
        stats = calc.calculate(
            trades=trades,
            edge_type="gap_fill",
            symbol="SPY",
            timeframe="1d",
            test_period="2024-01-01 to 2024-12-31",
            starting_balance=10_000,
        )
        assert stats.trailing_stop_exits == 3
        assert stats.breakeven_exits == 0

    def test_statistics_count_breakeven_exits(self):
        from nexus.backtest.statistics import StatisticsCalculator
        from datetime import timedelta

        trades = [
            SimulatedTrade(
                opportunity_id="t1",
                symbol="SPY",
                direction="long",
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                exit_price=100.0,
                exit_reason=ExitReason.BREAKEVEN_STOP,
                position_size=100,
                gross_pnl=0,
                costs=2,
                net_pnl=-2,
                net_pnl_pct=-0.02,
                hold_duration=timedelta(days=1),
                primary_edge="gap_fill",
                score=70,
                score_tier="B",
            ),
        ]

        calc = StatisticsCalculator()
        stats = calc.calculate(
            trades=trades,
            edge_type="gap_fill",
            symbol="SPY",
            timeframe="1d",
            test_period="2024-01-01 to 2024-12-31",
            starting_balance=10_000,
        )
        assert stats.breakeven_exits == 1
        assert stats.trailing_stop_exits == 0
