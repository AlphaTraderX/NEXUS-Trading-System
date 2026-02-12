"""Tests for edge decay detection."""

import pytest
from datetime import datetime

from nexus.risk.edge_decay import (
    EdgeDecayMonitor,
    EdgeBaseline,
    EdgeHealth,
    TradeOutcome,
    get_decay_monitor,
)
from nexus.core.enums import EdgeType


class TestTradeOutcome:
    """Test TradeOutcome dataclass."""

    def test_creation(self):
        outcome = TradeOutcome(
            edge_type=EdgeType.VWAP_DEVIATION,
            symbol="SPY",
            pnl=150.0,
            pnl_pct=1.5,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            is_win=True,
        )
        assert outcome.is_win is True
        assert outcome.pnl_pct == 1.5


class TestEdgeBaseline:
    """Test EdgeBaseline defaults."""

    def test_defaults_exist(self):
        defaults = EdgeBaseline.get_defaults()
        assert EdgeType.VWAP_DEVIATION in defaults
        assert EdgeType.TURN_OF_MONTH in defaults
        assert EdgeType.GAP_FILL in defaults

    def test_baseline_values(self):
        defaults = EdgeBaseline.get_defaults()
        vwap = defaults[EdgeType.VWAP_DEVIATION]
        assert 0 < vwap.expected_win_rate < 1
        assert vwap.expected_avg_return > 0


class TestEdgeHealth:
    """Test EdgeHealth dataclass."""

    def test_is_healthy(self):
        health = EdgeHealth(
            edge_type=EdgeType.VWAP_DEVIATION,
            status="healthy",
            current_win_rate=0.55,
            current_avg_return=0.15,
            baseline_win_rate=0.54,
            baseline_avg_return=0.15,
            win_rate_z_score=0.1,
            return_z_score=0.0,
            trade_count=50,
            consecutive_failures=0,
            last_updated=datetime.now(),
        )
        assert health.is_healthy is True
        assert health.should_disable is False

    def test_should_disable(self):
        health = EdgeHealth(
            edge_type=EdgeType.VWAP_DEVIATION,
            status="disabled",
            current_win_rate=0.40,
            current_avg_return=-0.05,
            baseline_win_rate=0.54,
            baseline_avg_return=0.15,
            win_rate_z_score=-2.5,
            return_z_score=-2.0,
            trade_count=50,
            consecutive_failures=3,
            last_updated=datetime.now(),
        )
        assert health.is_healthy is False
        assert health.should_disable is True


class TestEdgeDecayMonitor:
    """Test EdgeDecayMonitor."""

    @pytest.fixture
    def monitor(self):
        return EdgeDecayMonitor(rolling_window=20)

    def _make_outcome(self, edge_type, is_win, pnl_pct):
        return TradeOutcome(
            edge_type=edge_type,
            symbol="SPY",
            pnl=100 if is_win else -100,
            pnl_pct=pnl_pct,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            is_win=is_win,
        )

    def test_record_trade(self, monitor):
        outcome = self._make_outcome(EdgeType.VWAP_DEVIATION, True, 1.0)
        monitor.record_trade(outcome)
        assert len(monitor._trade_history[EdgeType.VWAP_DEVIATION]) == 1

    def test_insufficient_data(self, monitor):
        for _ in range(5):
            monitor.record_trade(
                self._make_outcome(EdgeType.VWAP_DEVIATION, True, 1.0)
            )

        health = monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        assert health.status == "insufficient_data"

    def test_healthy_edge(self, monitor):
        monitor.baselines[EdgeType.VWAP_DEVIATION].min_trades_for_signal = 10

        for _ in range(12):
            monitor.record_trade(
                self._make_outcome(EdgeType.VWAP_DEVIATION, True, 0.20)
            )

        health = monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        assert health.status == "healthy"
        assert health.current_win_rate == 1.0

    def test_warning_edge(self, monitor):
        monitor.baselines[EdgeType.VWAP_DEVIATION].min_trades_for_signal = 10

        for i in range(12):
            is_win = i < 4  # 33% win rate
            monitor.record_trade(
                self._make_outcome(
                    EdgeType.VWAP_DEVIATION, is_win, 0.10 if is_win else -0.10
                )
            )

        health = monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        assert health.status in ("warning", "critical")

    def test_critical_edge(self, monitor):
        monitor.baselines[EdgeType.VWAP_DEVIATION].min_trades_for_signal = 10

        for i in range(12):
            is_win = i < 2  # 16% win rate
            monitor.record_trade(
                self._make_outcome(
                    EdgeType.VWAP_DEVIATION, is_win, 0.05 if is_win else -0.20
                )
            )

        health = monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        assert health.status == "critical"
        assert health.consecutive_failures >= 1

    def test_auto_disable(self, monitor):
        monitor.baselines[EdgeType.GAP_FILL].min_trades_for_signal = 10

        for period in range(3):
            for _ in range(12):
                monitor.record_trade(
                    self._make_outcome(EdgeType.GAP_FILL, False, -0.30)
                )
            health = monitor.check_edge_health(EdgeType.GAP_FILL)

        assert health.status == "disabled"
        assert EdgeType.GAP_FILL in monitor.get_disabled_edges()

    def test_is_edge_enabled(self, monitor):
        assert monitor.is_edge_enabled(EdgeType.VWAP_DEVIATION) is True

        monitor._disabled_edges.add(EdgeType.VWAP_DEVIATION)
        assert monitor.is_edge_enabled(EdgeType.VWAP_DEVIATION) is False

    def test_re_enable_edge(self, monitor):
        monitor._disabled_edges.add(EdgeType.VWAP_DEVIATION)

        result = monitor.re_enable_edge(EdgeType.VWAP_DEVIATION)

        assert result is True
        assert monitor.is_edge_enabled(EdgeType.VWAP_DEVIATION) is True

    def test_re_enable_not_disabled(self, monitor):
        result = monitor.re_enable_edge(EdgeType.VWAP_DEVIATION)
        assert result is False

    def test_get_decay_warnings(self, monitor):
        monitor.baselines[EdgeType.RSI_EXTREME].min_trades_for_signal = 10

        for _ in range(12):
            monitor.record_trade(
                self._make_outcome(EdgeType.RSI_EXTREME, False, -0.20)
            )

        warnings = monitor.get_decay_warnings()
        assert len(warnings) >= 1
        assert any(w.edge_type == EdgeType.RSI_EXTREME for w in warnings)

    def test_update_baseline(self, monitor):
        monitor.update_baseline(
            EdgeType.VWAP_DEVIATION,
            win_rate=0.60,
            avg_return=0.25,
        )

        baseline = monitor.baselines[EdgeType.VWAP_DEVIATION]
        assert baseline.expected_win_rate == 0.60
        assert baseline.expected_avg_return == 0.25

    def test_get_summary(self, monitor):
        summary = monitor.get_summary()

        assert "total_edges" in summary
        assert "healthy" in summary
        assert "warning" in summary
        assert "edges" in summary

    def test_rolling_window(self, monitor):
        """Test that rolling window limits trade history."""
        for _ in range(30):  # Window is 20
            monitor.record_trade(
                self._make_outcome(EdgeType.VWAP_DEVIATION, True, 1.0)
            )

        assert len(monitor._trade_history[EdgeType.VWAP_DEVIATION]) == 20

    def test_no_baseline(self, monitor):
        """Edge with no baseline returns unknown status."""
        # Create a monitor with empty baselines
        empty_monitor = EdgeDecayMonitor(baselines={}, rolling_window=20)
        empty_monitor.record_trade(
            self._make_outcome(EdgeType.VWAP_DEVIATION, True, 1.0)
        )
        health = empty_monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        assert health.status == "unknown"

    def test_check_all_edges(self, monitor):
        """check_all_edges returns results for all tracked edges."""
        monitor.baselines[EdgeType.VWAP_DEVIATION].min_trades_for_signal = 5
        monitor.baselines[EdgeType.GAP_FILL].min_trades_for_signal = 5

        for _ in range(6):
            monitor.record_trade(
                self._make_outcome(EdgeType.VWAP_DEVIATION, True, 0.20)
            )
            monitor.record_trade(
                self._make_outcome(EdgeType.GAP_FILL, True, 0.25)
            )

        results = monitor.check_all_edges()
        assert EdgeType.VWAP_DEVIATION in results
        assert EdgeType.GAP_FILL in results


class TestGetDecayMonitor:
    """Test singleton."""

    def test_singleton(self):
        import nexus.risk.edge_decay as mod

        # Reset singleton for test isolation
        mod._decay_monitor = None
        m1 = get_decay_monitor()
        m2 = get_decay_monitor()
        assert m1 is m2
        # Clean up
        mod._decay_monitor = None
