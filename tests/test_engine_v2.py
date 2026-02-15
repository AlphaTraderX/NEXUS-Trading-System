"""Tests for BacktestEngineV2 components."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.backtest.engine_v2 import (
    BacktestEngineV2,
    BacktestHeatTracker,
    EDGE_TRAILING_CONFIG,
)
from nexus.backtest.trade_simulator import TrailingStopConfig
from nexus.core.enums import EdgeType
from nexus.intelligence.regime_detector import GodModeRegime


# ---------------------------------------------------------------------------
# BacktestHeatTracker tests
# ---------------------------------------------------------------------------

class TestBacktestHeatTracker:
    """Test lightweight heat tracker for backtesting."""

    def test_initial_state(self):
        ht = BacktestHeatTracker()
        assert ht.current_heat == 0.0
        assert ht.can_add(5.0) is True

    def test_add_and_remove_position(self):
        ht = BacktestHeatTracker()
        ht.add_position("pos1", 5.0)
        assert ht.current_heat == 5.0

        ht.add_position("pos2", 3.0)
        assert ht.current_heat == 8.0

        ht.remove_position("pos1")
        assert ht.current_heat == 3.0

    def test_remove_nonexistent(self):
        ht = BacktestHeatTracker()
        ht.remove_position("nonexistent")  # Should not raise
        assert ht.current_heat == 0.0

    def test_reset(self):
        ht = BacktestHeatTracker()
        ht.add_position("pos1", 10.0)
        ht.reset()
        assert ht.current_heat == 0.0

    def test_heat_limit_profitable_day(self):
        ht = BacktestHeatTracker()
        # Very profitable day: limit expands to 35%
        assert ht.get_heat_limit(2.5) == 35.0
        # Moderately profitable: 30%
        assert ht.get_heat_limit(1.5) == 30.0
        # Slightly profitable: 25% (base)
        assert ht.get_heat_limit(0.5) == 25.0

    def test_heat_limit_losing_day(self):
        ht = BacktestHeatTracker()
        # Slight loss: 20%
        assert ht.get_heat_limit(-0.5) == 20.0
        # Big loss: 15% (min)
        assert ht.get_heat_limit(-2.0) == 15.0

    def test_can_add_respects_limit(self):
        ht = BacktestHeatTracker()
        # Base limit = 25% with 0 daily pnl
        ht.add_position("pos1", 20.0)
        assert ht.can_add(4.0, daily_pnl_pct=0.0) is True   # 24 <= 25
        assert ht.can_add(6.0, daily_pnl_pct=0.0) is False   # 26 > 25

    def test_can_add_expands_on_profit(self):
        ht = BacktestHeatTracker()
        ht.add_position("pos1", 30.0)
        # On a losing day (limit=15%), can't add
        assert ht.can_add(1.0, daily_pnl_pct=-2.0) is False
        # On a great day (limit=35%), can add
        assert ht.can_add(4.0, daily_pnl_pct=3.0) is True


# ---------------------------------------------------------------------------
# Edge trailing config tests
# ---------------------------------------------------------------------------

class TestEdgeTrailingConfig:
    """Test per-edge trailing stop configurations."""

    def test_gap_fill_has_trailing(self):
        cfg = EDGE_TRAILING_CONFIG[EdgeType.GAP_FILL]
        assert cfg is not None
        assert isinstance(cfg, TrailingStopConfig)
        assert cfg.atr_trail_multiplier == 1.5

    def test_insider_cluster_has_trailing(self):
        cfg = EDGE_TRAILING_CONFIG[EdgeType.INSIDER_CLUSTER]
        assert cfg is not None
        assert cfg.atr_trail_multiplier == 2.0

    def test_vwap_deviation_has_trailing(self):
        cfg = EDGE_TRAILING_CONFIG[EdgeType.VWAP_DEVIATION]
        assert cfg is not None
        assert cfg.atr_trail_multiplier == 1.0

    def test_rsi_extreme_no_trailing(self):
        assert EDGE_TRAILING_CONFIG[EdgeType.RSI_EXTREME] is None

    def test_overnight_no_trailing(self):
        assert EDGE_TRAILING_CONFIG[EdgeType.OVERNIGHT_PREMIUM] is None


# ---------------------------------------------------------------------------
# BacktestEngineV2 construction and feature toggles
# ---------------------------------------------------------------------------

class TestBacktestEngineV2Init:
    """Test engine initialization and feature toggles."""

    def test_default_features_enabled(self):
        engine = BacktestEngineV2(starting_balance=10_000)
        assert engine.use_regime_filter is True
        assert engine.use_trailing_stops is True
        assert engine.use_heat_management is True
        assert engine.use_momentum_scaling is True

    def test_features_can_be_disabled(self):
        engine = BacktestEngineV2(
            starting_balance=10_000,
            use_regime_filter=False,
            use_trailing_stops=False,
            use_heat_management=False,
            use_momentum_scaling=False,
        )
        assert engine.use_regime_filter is False
        assert engine.use_trailing_stops is False
        assert engine.use_heat_management is False
        assert engine.use_momentum_scaling is False

    def test_v1_engine_delegate(self):
        engine = BacktestEngineV2(starting_balance=20_000, risk_per_trade=2.0)
        assert engine.v1.starting_balance == 20_000

    def test_scanner_map_exposed(self):
        engine = BacktestEngineV2()
        assert len(engine.SCANNER_MAP) > 0

    def test_disabled_edges_exposed(self):
        engine = BacktestEngineV2()
        assert isinstance(engine.DISABLED_EDGES, set)


# ---------------------------------------------------------------------------
# Momentum multiplier
# ---------------------------------------------------------------------------

class TestMomentumMultiplier:
    """Test win-streak momentum scaling."""

    def test_no_streak_returns_1(self):
        engine = BacktestEngineV2()
        engine._win_streak = 0
        assert engine._get_momentum_multiplier() == 1.0

    def test_one_win_returns_1(self):
        engine = BacktestEngineV2()
        engine._win_streak = 1
        assert engine._get_momentum_multiplier() == 1.0

    def test_two_wins_returns_1_2(self):
        engine = BacktestEngineV2()
        engine._win_streak = 2
        assert engine._get_momentum_multiplier() == 1.2

    def test_three_wins_capped_at_1_3(self):
        engine = BacktestEngineV2()
        engine._win_streak = 3
        assert engine._get_momentum_multiplier() == 1.3

    def test_large_streak_capped(self):
        engine = BacktestEngineV2()
        engine._win_streak = 10
        assert engine._get_momentum_multiplier() == 1.3

    def test_disabled_returns_1(self):
        engine = BacktestEngineV2(use_momentum_scaling=False)
        engine._win_streak = 5
        assert engine._get_momentum_multiplier() == 1.0


# ---------------------------------------------------------------------------
# Streak and daily P&L tracking
# ---------------------------------------------------------------------------

class TestStreakTracking:
    """Test win/loss streak tracking."""

    def test_win_increments_streak(self):
        engine = BacktestEngineV2()
        engine._update_streak(100.0)
        assert engine._win_streak == 1
        engine._update_streak(50.0)
        assert engine._win_streak == 2

    def test_loss_resets_streak(self):
        engine = BacktestEngineV2()
        engine._win_streak = 5
        engine._update_streak(-10.0)
        assert engine._win_streak == 0

    def test_daily_pnl_accumulates(self):
        engine = BacktestEngineV2(starting_balance=10_000)
        engine._update_daily_pnl(100.0)
        engine._update_daily_pnl(50.0)
        assert engine._daily_pnl == 150.0
        assert engine._daily_pnl_pct == 1.5  # 150 / 10000 * 100


# ---------------------------------------------------------------------------
# Reset state
# ---------------------------------------------------------------------------

class TestResetForEdge:
    """Test state reset between edges."""

    def test_reset_clears_all_state(self):
        engine = BacktestEngineV2(starting_balance=10_000)
        engine._win_streak = 5
        engine._daily_pnl = 500.0
        engine.heat_tracker.add_position("test", 10.0)
        engine._regime_signal_counts = {"BULL": 10}
        engine._regime_filtered_counts = {"BEAR": 5}

        engine._reset_for_edge()

        assert engine._win_streak == 0
        assert engine._daily_pnl == 0.0
        assert engine.heat_tracker.current_heat == 0.0
        assert engine._regime_signal_counts == {}
        assert engine._regime_filtered_counts == {}


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

class TestRegimeDetection:
    """Test regime detection integration."""

    def test_short_bars_default_sideways(self):
        engine = BacktestEngineV2()
        bars = pd.DataFrame({"close": [100] * 50})
        regime = engine._detect_regime(bars)
        assert regime == GodModeRegime.SIDEWAYS

    def test_regime_config_lookup(self):
        engine = BacktestEngineV2()
        config = engine._get_regime_config(GodModeRegime.STRONG_BULL)
        assert config.position_size_multiplier > 0
        assert len(config.allowed_edges) > 0


# ---------------------------------------------------------------------------
# Compounding
# ---------------------------------------------------------------------------

class TestCompounding:
    """Test v2 compounding delegates to v1."""

    def test_compound_increases_balance(self):
        engine = BacktestEngineV2(starting_balance=10_000)
        engine._compound(500.0)
        assert engine.v1.simulator.account_balance == 10_500.0

    def test_compound_decreases_balance(self):
        engine = BacktestEngineV2(starting_balance=10_000)
        engine._compound(-300.0)
        assert engine.v1.simulator.account_balance == 9_700.0

    def test_compound_floor_at_zero(self):
        engine = BacktestEngineV2(starting_balance=100)
        engine._compound(-200.0)
        assert engine.v1.simulator.account_balance == 0.0

    def test_no_compound_without_score_sizing(self):
        engine = BacktestEngineV2(starting_balance=10_000, use_score_sizing=False)
        engine._compound(500.0)
        # Without score sizing, balance should NOT change
        assert engine.v1.simulator.account_balance == 10_000.0


# ---------------------------------------------------------------------------
# Compare backtests script
# ---------------------------------------------------------------------------

class TestCompareScript:
    """Test compare_backtests script can parse baselines."""

    def test_compare_identical_baselines(self, tmp_path):
        import json
        from nexus.scripts.compare_backtests import compare

        baseline = {
            "gap_fill": {
                "total_trades": 100,
                "winners": 55,
                "losers": 45,
                "win_rate": 55.0,
                "profit_factor": 1.5,
                "total_pnl": 5000.0,
                "total_pnl_pct": 50.0,
                "max_drawdown_pct": 15.0,
                "sharpe_ratio": 2.0,
                "verdict": "VALID",
            }
        }

        f1 = tmp_path / "baseline.json"
        f2 = tmp_path / "current.json"
        f1.write_text(json.dumps(baseline))
        f2.write_text(json.dumps(baseline))

        # Should not raise
        compare(str(f1), str(f2))
