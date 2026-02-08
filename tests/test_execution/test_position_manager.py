"""
Tests for the Position Manager.

The Position Manager tracks all positions and their lifecycle,
syncing with Heat Manager and Correlation Monitor.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import uuid

from nexus.execution.position_manager import (
    PositionManager,
    Position,
    PositionStatus,
    PortfolioMetrics,
)
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence import CostBreakdown
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.correlation import CorrelationMonitor


# ============== Fixtures ==============

@pytest.fixture
def mock_heat_manager():
    """Mock heat manager."""
    manager = Mock(spec=DynamicHeatManager)
    manager.add_position = Mock()
    manager.remove_position = Mock()
    manager.get_heat_status = Mock(return_value=Mock(current_heat=5.0))
    return manager


@pytest.fixture
def mock_correlation_monitor():
    """Mock correlation monitor."""
    monitor = Mock(spec=CorrelationMonitor)
    monitor.add_position = Mock()
    monitor.remove_position = Mock()
    return monitor


@pytest.fixture
def position_manager(mock_heat_manager, mock_correlation_monitor):
    """Create a position manager with mocked dependencies."""
    return PositionManager(
        heat_manager=mock_heat_manager,
        correlation_monitor=mock_correlation_monitor,
        max_positions=8,
    )


@pytest.fixture
def position_manager_no_mocks():
    """Create a position manager without dependencies."""
    return PositionManager(max_positions=8)


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing."""
    return NexusSignal(
        signal_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        opportunity_id=str(uuid.uuid4()),
        symbol="AAPL",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=175.0,
        stop_loss=172.0,
        take_profit=181.0,
        position_size=10.0,
        position_value=1750.0,
        risk_amount=30.0,
        risk_percent=1.0,
        primary_edge=EdgeType.VWAP_DEVIATION,
        secondary_edges=[],
        edge_score=75,
        tier=SignalTier.B,
        gross_expected=0.15,
        costs=CostBreakdown(
            spread=0.02,
            commission=0.01,
            slippage=0.02,
            overnight=0.0,
            fx_conversion=0.0,
            other=0.0,
        ),
        net_expected=0.10,
        cost_ratio=33.3,
        ai_reasoning="Test signal",
        confluence_factors=["Factor 1"],
        risk_factors=["Risk 1"],
        market_context="Test context",
        session="regular",
        valid_until=datetime.utcnow() + timedelta(hours=4),
        status=SignalStatus.PENDING,
    )


@pytest.fixture
def sample_short_signal():
    """Create a sample short signal."""
    return NexusSignal(
        signal_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        opportunity_id=str(uuid.uuid4()),
        symbol="TSLA",
        market=Market.US_STOCKS,
        direction=Direction.SHORT,
        entry_price=250.0,
        stop_loss=258.0,
        take_profit=234.0,
        position_size=5.0,
        position_value=1250.0,
        risk_amount=40.0,
        risk_percent=1.2,
        primary_edge=EdgeType.RSI_EXTREME,
        secondary_edges=[],
        edge_score=68,
        tier=SignalTier.B,
        gross_expected=0.12,
        costs=CostBreakdown(
            spread=0.02,
            commission=0.01,
            slippage=0.02,
            overnight=0.0,
            fx_conversion=0.0,
            other=0.0,
        ),
        net_expected=0.07,
        cost_ratio=41.7,
        ai_reasoning="Test short signal",
        confluence_factors=["Factor 1"],
        risk_factors=["Risk 1"],
        market_context="Test context",
        session="regular",
        valid_until=datetime.utcnow() + timedelta(hours=4),
        status=SignalStatus.PENDING,
    )


# ============== Position Creation Tests ==============

class TestPositionCreation:
    """Test position creation from signals."""

    def test_create_position_from_signal(self, position_manager, sample_signal):
        """Test creating a position from a signal."""
        position = position_manager.create_position_from_signal(sample_signal)

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.direction == Direction.LONG
        assert position.entry_price == 175.0
        assert position.stop_loss == 172.0
        assert position.take_profit == 181.0
        assert position.size == 10.0
        assert position.status == PositionStatus.PENDING

    def test_position_has_correct_risk_info(self, position_manager, sample_signal):
        """Test position has correct risk information."""
        position = position_manager.create_position_from_signal(sample_signal)

        assert position.risk_amount == 30.0
        assert position.risk_percent == 1.0
        assert position.edge_score == 75

    def test_cannot_create_duplicate_position_for_signal(self, position_manager, sample_signal):
        """Test that duplicate positions cannot be created."""
        position_manager.create_position_from_signal(sample_signal)

        with pytest.raises(ValueError, match="already exists"):
            position_manager.create_position_from_signal(sample_signal)

    def test_max_positions_enforced(self, position_manager, sample_signal):
        """Test that max positions limit is enforced."""
        # Create max positions
        for i in range(8):
            signal = NexusSignal(
                signal_id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                opportunity_id=str(uuid.uuid4()),
                symbol=f"SYM{i}",
                market=Market.US_STOCKS,
                direction=Direction.LONG,
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                position_size=10.0,
                position_value=1000.0,
                risk_amount=20.0,
                risk_percent=1.0,
                primary_edge=EdgeType.VWAP_DEVIATION,
                secondary_edges=[],
                edge_score=70,
                tier=SignalTier.B,
                gross_expected=0.10,
                costs=CostBreakdown(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
                net_expected=0.07,
                cost_ratio=30.0,
                ai_reasoning="Test",
                confluence_factors=[],
                risk_factors=[],
                market_context="Test",
                session="regular",
                valid_until=datetime.utcnow() + timedelta(hours=4),
                status=SignalStatus.PENDING,
            )
            position_manager.create_position_from_signal(signal)

        # 9th should fail
        with pytest.raises(ValueError, match="Max positions"):
            position_manager.create_position_from_signal(sample_signal)


# ============== Position Lifecycle Tests ==============

class TestPositionLifecycle:
    """Test position lifecycle transitions."""

    def test_mark_submitted(self, position_manager, sample_signal):
        """Test marking position as submitted."""
        position = position_manager.create_position_from_signal(sample_signal)

        updated = position_manager.mark_submitted(position.position_id)

        assert updated.status == PositionStatus.SUBMITTED

    def test_mark_open(self, position_manager, sample_signal, mock_heat_manager, mock_correlation_monitor):
        """Test marking position as open."""
        position = position_manager.create_position_from_signal(sample_signal)
        fill_price = 175.50

        updated = position_manager.mark_open(
            position.position_id,
            fill_price=fill_price,
        )

        assert updated.status == PositionStatus.OPEN
        assert updated.entry_price == fill_price
        assert updated.opened_at is not None

        # Verify Heat Manager was called
        mock_heat_manager.add_position.assert_called_once()

        # Verify Correlation Monitor was called
        mock_correlation_monitor.add_position.assert_called_once()

    def test_close_position(self, position_manager, sample_signal, mock_heat_manager, mock_correlation_monitor):
        """Test closing a position."""
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_open(position.position_id, fill_price=175.0)

        closed = position_manager.close_position(
            position.position_id,
            exit_price=180.0,
            exit_reason="take_profit",
        )

        assert closed.status == PositionStatus.CLOSED
        assert closed.exit_price == 180.0
        assert closed.exit_reason == "take_profit"
        assert closed.closed_at is not None
        assert closed.realized_pnl > 0  # Profitable long

        # Verify removed from managers
        mock_heat_manager.remove_position.assert_called_once()
        mock_correlation_monitor.remove_position.assert_called_once()

    def test_cancel_pending_position(self, position_manager, sample_signal):
        """Test cancelling a pending position."""
        position = position_manager.create_position_from_signal(sample_signal)

        cancelled = position_manager.cancel_position(position.position_id, "order_rejected")

        assert cancelled.status == PositionStatus.CANCELLED
        assert cancelled.exit_reason == "order_rejected"


# ============== P&L Calculation Tests ==============

class TestPnLCalculations:
    """Test P&L calculations."""

    def test_unrealized_pnl_long_profit(self, position_manager_no_mocks, sample_signal):
        """Test unrealized P&L for profitable long."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Price goes up
        position_manager_no_mocks.update_price("AAPL", 180.0)

        updated_position = position_manager_no_mocks.get_position(position.position_id)
        assert updated_position.unrealized_pnl == 50.0  # (180-175) * 10 shares
        assert updated_position.unrealized_pnl_pct > 0
        assert updated_position.is_profitable

    def test_unrealized_pnl_long_loss(self, position_manager_no_mocks, sample_signal):
        """Test unrealized P&L for losing long."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Price goes down
        position_manager_no_mocks.update_price("AAPL", 170.0)

        updated_position = position_manager_no_mocks.get_position(position.position_id)
        assert updated_position.unrealized_pnl == -50.0  # (170-175) * 10 shares
        assert not updated_position.is_profitable

    def test_unrealized_pnl_short_profit(self, position_manager_no_mocks, sample_short_signal):
        """Test unrealized P&L for profitable short."""
        position = position_manager_no_mocks.create_position_from_signal(sample_short_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=250.0)

        # Price goes down (profit for short)
        position_manager_no_mocks.update_price("TSLA", 240.0)

        updated_position = position_manager_no_mocks.get_position(position.position_id)
        assert updated_position.unrealized_pnl == 50.0  # (250-240) * 5 shares
        assert updated_position.is_profitable

    def test_realized_pnl_calculation(self, position_manager_no_mocks, sample_signal):
        """Test realized P&L calculation on close."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        closed = position_manager_no_mocks.close_position(
            position.position_id,
            exit_price=180.0,
            exit_reason="take_profit",
            exit_costs=5.0,
        )

        # Gross: (180-175) * 10 = 50
        # Entry costs: 0.05 (from signal)
        # Exit costs: 5.0
        # Net should be close to 50 - 5 - entry_costs
        assert closed.realized_pnl > 0
        assert closed.unrealized_pnl == 0  # Cleared after close

    def test_r_multiple_calculation(self, position_manager_no_mocks, sample_signal):
        """Test R-multiple calculation."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Close at 2R profit (risk was 30, so 2R = 60 profit before costs)
        # Entry: 175, Risk: 30 for 10 shares = $3/share
        # 2R would be exit at 175 + 6 = 181
        closed = position_manager_no_mocks.close_position(
            position.position_id,
            exit_price=181.0,
            exit_reason="take_profit",
        )

        # R-multiple should be close to 2
        assert 1.5 <= closed.r_multiple <= 2.5


# ============== Stop/Target Detection Tests ==============

class TestStopTargetDetection:
    """Test stop loss and take profit detection."""

    def test_long_at_stop(self, position_manager_no_mocks, sample_signal):
        """Test detecting long position at stop loss."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Price hits stop (172)
        position_manager_no_mocks.update_price("AAPL", 171.0)

        at_stop = position_manager_no_mocks.get_positions_at_stop()
        assert len(at_stop) == 1
        assert at_stop[0].position_id == position.position_id

    def test_long_at_target(self, position_manager_no_mocks, sample_signal):
        """Test detecting long position at take profit."""
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Price hits target (181)
        position_manager_no_mocks.update_price("AAPL", 182.0)

        at_target = position_manager_no_mocks.get_positions_at_target()
        assert len(at_target) == 1

    def test_short_at_stop(self, position_manager_no_mocks, sample_short_signal):
        """Test detecting short position at stop loss."""
        position = position_manager_no_mocks.create_position_from_signal(sample_short_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=250.0)

        # Price goes up to stop (258)
        position_manager_no_mocks.update_price("TSLA", 260.0)

        at_stop = position_manager_no_mocks.get_positions_at_stop()
        assert len(at_stop) == 1

    def test_short_at_target(self, position_manager_no_mocks, sample_short_signal):
        """Test detecting short position at take profit."""
        position = position_manager_no_mocks.create_position_from_signal(sample_short_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=250.0)

        # Price drops to target (234)
        position_manager_no_mocks.update_price("TSLA", 230.0)

        at_target = position_manager_no_mocks.get_positions_at_target()
        assert len(at_target) == 1


# ============== Portfolio Metrics Tests ==============

class TestPortfolioMetrics:
    """Test portfolio metrics calculation."""

    def test_empty_portfolio_metrics(self, position_manager_no_mocks):
        """Test metrics for empty portfolio."""
        metrics = position_manager_no_mocks.get_portfolio_metrics()

        assert metrics.total_positions == 0
        assert metrics.open_positions == 0
        assert metrics.total_pnl == 0.0

    def test_portfolio_metrics_with_positions(self, position_manager_no_mocks, sample_signal, sample_short_signal):
        """Test metrics with multiple positions."""
        # Create and open two positions
        pos1 = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(pos1.position_id, fill_price=175.0)

        pos2 = position_manager_no_mocks.create_position_from_signal(sample_short_signal)
        position_manager_no_mocks.mark_open(pos2.position_id, fill_price=250.0)

        # Update prices - one profit, one loss
        position_manager_no_mocks.update_price("AAPL", 180.0)  # +50
        position_manager_no_mocks.update_price("TSLA", 255.0)  # -25 (short)

        metrics = position_manager_no_mocks.get_portfolio_metrics()

        assert metrics.open_positions == 2
        assert metrics.total_unrealized_pnl == 25.0  # 50 - 25
        assert metrics.long_exposure > 0
        assert metrics.short_exposure > 0

    def test_win_rate_calculation(self, position_manager_no_mocks, sample_signal):
        """Test win rate calculation after closing positions."""
        # Create, open, and close a winning position
        pos1 = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(pos1.position_id, fill_price=175.0)
        position_manager_no_mocks.close_position(pos1.position_id, exit_price=180.0, exit_reason="target")

        # Create another signal with different ID
        signal2 = NexusSignal(
            signal_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            opportunity_id=str(uuid.uuid4()),
            symbol="MSFT",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=350.0,
            stop_loss=345.0,
            take_profit=360.0,
            position_size=5.0,
            position_value=1750.0,
            risk_amount=25.0,
            risk_percent=0.8,
            primary_edge=EdgeType.VWAP_DEVIATION,
            secondary_edges=[],
            edge_score=72,
            tier=SignalTier.B,
            gross_expected=0.12,
            costs=CostBreakdown(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
            net_expected=0.09,
            cost_ratio=25.0,
            ai_reasoning="Test",
            confluence_factors=[],
            risk_factors=[],
            market_context="Test",
            session="regular",
            valid_until=datetime.utcnow() + timedelta(hours=4),
            status=SignalStatus.PENDING,
        )

        # Create, open, and close a losing position
        pos2 = position_manager_no_mocks.create_position_from_signal(signal2)
        position_manager_no_mocks.mark_open(pos2.position_id, fill_price=350.0)
        position_manager_no_mocks.close_position(pos2.position_id, exit_price=345.0, exit_reason="stop")

        metrics = position_manager_no_mocks.get_portfolio_metrics()

        # 1 win, 1 loss = 50% win rate
        assert metrics.win_rate == 50.0


# ============== Query Methods Tests ==============

class TestQueryMethods:
    """Test position query methods."""

    def test_get_position_by_signal(self, position_manager, sample_signal):
        """Test getting position by signal ID."""
        position = position_manager.create_position_from_signal(sample_signal)

        found = position_manager.get_position_by_signal(sample_signal.signal_id)

        assert found is not None
        assert found.position_id == position.position_id

    def test_get_positions_for_symbol(self, position_manager, sample_signal):
        """Test getting positions for a symbol."""
        position = position_manager.create_position_from_signal(sample_signal)

        positions = position_manager.get_positions_for_symbol("AAPL")

        assert len(positions) == 1
        assert positions[0].position_id == position.position_id

    def test_get_open_positions_for_symbol(self, position_manager, sample_signal):
        """Test getting only open positions for a symbol."""
        position = position_manager.create_position_from_signal(sample_signal)

        # Pending position should not be in open list
        open_positions = position_manager.get_open_positions_for_symbol("AAPL")
        assert len(open_positions) == 0

        # Mark as open
        position_manager.mark_open(position.position_id, fill_price=175.0)

        open_positions = position_manager.get_open_positions_for_symbol("AAPL")
        assert len(open_positions) == 1


# ============== Session Stats Tests ==============

class TestSessionStats:
    """Test session statistics tracking."""

    def test_session_stats_tracking(self, position_manager_no_mocks, sample_signal):
        """Test that session stats are tracked."""
        # Open and close a position
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)
        position_manager_no_mocks.close_position(position.position_id, exit_price=180.0, exit_reason="target")

        stats = position_manager_no_mocks.get_session_stats()

        assert stats["positions_opened"] == 1
        assert stats["positions_closed"] == 1
        assert stats["total_realized_pnl"] > 0

    def test_session_stats_reset(self, position_manager_no_mocks, sample_signal):
        """Test resetting session stats."""
        # Generate some stats
        position = position_manager_no_mocks.create_position_from_signal(sample_signal)
        position_manager_no_mocks.mark_open(position.position_id, fill_price=175.0)

        # Reset
        position_manager_no_mocks.reset_session_stats()

        stats = position_manager_no_mocks.get_session_stats()
        assert stats["positions_opened"] == 0
        assert stats["positions_closed"] == 0
        assert stats["total_realized_pnl"] == 0.0


# ============== Edge Cases ==============

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cannot_open_closed_position(self, position_manager, sample_signal):
        """Test that closed positions cannot be reopened."""
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_open(position.position_id, fill_price=175.0)
        position_manager.close_position(position.position_id, exit_price=180.0, exit_reason="target")

        with pytest.raises(ValueError, match="cannot be opened"):
            position_manager.mark_open(position.position_id, fill_price=175.0)

    def test_cannot_close_pending_position(self, position_manager, sample_signal):
        """Test that pending positions cannot be closed (must be cancelled)."""
        position = position_manager.create_position_from_signal(sample_signal)

        with pytest.raises(ValueError, match="not open"):
            position_manager.close_position(position.position_id, exit_price=180.0, exit_reason="target")

    def test_position_not_found(self, position_manager):
        """Test error when position not found."""
        with pytest.raises(ValueError, match="not found"):
            position_manager.mark_open("nonexistent-id", fill_price=100.0)

    def test_update_price_no_positions(self, position_manager):
        """Test updating price when no positions exist."""
        updated = position_manager.update_price("AAPL", 180.0)
        assert updated == []
