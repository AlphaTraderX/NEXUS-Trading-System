"""
Tests for the Order Manager.

The Order Manager handles order creation, submission, fills,
partial fills, slippage tracking, and timeouts.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import uuid

from nexus.execution.order_manager import (
    OrderManager,
    Order,
    OrderFill,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderPurpose,
    SlippageStats,
)
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown


# ============== Fixtures ==============

@pytest.fixture
def order_manager():
    """Create an order manager."""
    return OrderManager(
        default_order_type=OrderType.LIMIT,
        default_timeout_minutes=5,
        max_slippage_pct=0.5,
    )


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


# ============== Order Creation Tests ==============

class TestOrderCreation:
    """Test order creation from signals."""

    def test_create_entry_order_long(self, order_manager, sample_signal):
        """Test creating entry order for long signal."""
        order = order_manager.create_entry_order(sample_signal)

        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10.0
        assert order.limit_price == 175.0
        assert order.purpose == OrderPurpose.ENTRY
        assert order.status == OrderStatus.CREATED
        assert order.signal_id == sample_signal.signal_id

    def test_create_entry_order_short(self, order_manager, sample_short_signal):
        """Test creating entry order for short signal."""
        order = order_manager.create_entry_order(sample_short_signal)

        assert order.side == OrderSide.SELL
        assert order.quantity == 5.0
        assert order.limit_price == 250.0

    def test_create_market_order(self, order_manager, sample_signal):
        """Test creating market order."""
        order = order_manager.create_entry_order(
            sample_signal,
            order_type=OrderType.MARKET,
        )

        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None

    def test_order_has_expiry(self, order_manager, sample_signal):
        """Test that orders have expiry time."""
        order = order_manager.create_entry_order(sample_signal)

        assert order.expires_at is not None
        assert order.expires_at > datetime.utcnow()

    def test_create_stop_loss_order(self, order_manager):
        """Test creating stop loss order."""
        order = order_manager.create_stop_loss_order(
            position_id="pos-123",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            quantity=10.0,
            stop_price=172.0,
        )

        assert order.order_type == OrderType.STOP
        assert order.side == OrderSide.SELL  # Sell to close long
        assert order.stop_price == 172.0
        assert order.purpose == OrderPurpose.EXIT_STOP

    def test_create_take_profit_order(self, order_manager):
        """Test creating take profit order."""
        order = order_manager.create_take_profit_order(
            position_id="pos-123",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            quantity=10.0,
            limit_price=181.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL
        assert order.limit_price == 181.0
        assert order.purpose == OrderPurpose.EXIT_TARGET

    def test_create_exit_order_short(self, order_manager):
        """Test exit orders for short positions."""
        stop_order = order_manager.create_stop_loss_order(
            position_id="pos-456",
            symbol="TSLA",
            market=Market.US_STOCKS,
            direction=Direction.SHORT,
            quantity=5.0,
            stop_price=258.0,
        )

        # Stop loss for short = BUY (to cover)
        assert stop_order.side == OrderSide.BUY

        tp_order = order_manager.create_take_profit_order(
            position_id="pos-456",
            symbol="TSLA",
            market=Market.US_STOCKS,
            direction=Direction.SHORT,
            quantity=5.0,
            limit_price=234.0,
        )

        # Take profit for short = BUY (to cover)
        assert tp_order.side == OrderSide.BUY


# ============== Order Lifecycle Tests ==============

class TestOrderLifecycle:
    """Test order lifecycle transitions."""

    def test_mark_submitted(self, order_manager, sample_signal):
        """Test marking order as submitted."""
        order = order_manager.create_entry_order(sample_signal)

        updated = order_manager.mark_submitted(
            order.order_id,
            broker="ibkr",
            broker_order_id="IBKR-12345",
        )

        assert updated.status == OrderStatus.SUBMITTED
        assert updated.broker == "ibkr"
        assert updated.broker_order_id == "IBKR-12345"
        assert updated.submitted_at is not None

    def test_record_fill(self, order_manager, sample_signal):
        """Test recording a fill."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        updated = order_manager.record_fill(
            order.order_id,
            fill_price=175.10,
            fill_size=10.0,
            commission=1.50,
        )

        assert updated.status == OrderStatus.FILLED
        assert updated.filled_quantity == 10.0
        assert updated.remaining_quantity == 0.0
        assert updated.avg_fill_price == 175.10
        assert updated.total_commission == 1.50
        assert len(updated.fills) == 1

    def test_partial_fill(self, order_manager, sample_signal):
        """Test partial fill handling."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        # First partial fill
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=4.0)
        order = order_manager.get_order(order.order_id)

        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 4.0
        assert order.remaining_quantity == 6.0
        assert order.fill_rate == 40.0

        # Second partial fill
        order_manager.record_fill(order.order_id, fill_price=175.20, fill_size=6.0)
        order = order_manager.get_order(order.order_id)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10.0
        assert order.remaining_quantity == 0.0
        # Weighted avg: (4*175 + 6*175.20) / 10 = 175.12
        assert abs(order.avg_fill_price - 175.12) < 0.01

    def test_cancel_order(self, order_manager, sample_signal):
        """Test cancelling an order."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        cancelled = order_manager.cancel_order(order.order_id, "user_requested")

        assert cancelled.status == OrderStatus.CANCELLED
        assert cancelled.error_message == "user_requested"
        assert cancelled.cancelled_at is not None

    def test_reject_order(self, order_manager, sample_signal):
        """Test order rejection."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        rejected = order_manager.reject_order(order.order_id, "insufficient_funds")

        assert rejected.status == OrderStatus.REJECTED
        assert rejected.error_message == "insufficient_funds"


# ============== Slippage Tests ==============

class TestSlippageCalculation:
    """Test slippage calculation."""

    def test_positive_slippage_buy(self, order_manager, sample_signal):
        """Test slippage calculation for buy order (worse fill = positive slippage)."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        # Fill at higher price than expected (bad for buy)
        order_manager.record_fill(order.order_id, fill_price=175.50, fill_size=10.0)
        order = order_manager.get_order(order.order_id)

        # Slippage should be positive (unfavorable)
        assert order.slippage > 0
        assert order.slippage_pct > 0

    def test_negative_slippage_buy(self, order_manager, sample_signal):
        """Test favorable slippage for buy order."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        # Fill at lower price than expected (good for buy)
        order_manager.record_fill(order.order_id, fill_price=174.50, fill_size=10.0)
        order = order_manager.get_order(order.order_id)

        # Slippage should be negative (favorable)
        assert order.slippage < 0
        assert order.slippage_pct < 0

    def test_slippage_stats(self, order_manager, sample_signal):
        """Test slippage statistics calculation."""
        # Create and fill multiple orders
        for i in range(3):
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
            order = order_manager.create_entry_order(signal)
            order_manager.mark_submitted(order.order_id, broker="ibkr")
            # Fill with varying slippage
            fill_price = 100.0 + (i * 0.1)  # 100.0, 100.1, 100.2
            order_manager.record_fill(order.order_id, fill_price=fill_price, fill_size=10.0)

        stats = order_manager.get_slippage_stats()

        assert stats.filled_orders == 3
        assert stats.avg_slippage_pct > 0  # All had positive slippage


# ============== Order Expiry Tests ==============

class TestOrderExpiry:
    """Test order timeout/expiry handling."""

    def test_check_expired_orders(self, order_manager, sample_signal):
        """Test expiring timed-out orders."""
        # Create order with very short timeout
        order = order_manager.create_entry_order(
            sample_signal,
            timeout_minutes=0,  # Immediate expiry
        )
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        # Force expiry by setting expires_at in the past
        order.expires_at = datetime.utcnow() - timedelta(minutes=1)

        expired = order_manager.check_expired_orders()

        assert len(expired) == 1
        assert expired[0].status == OrderStatus.EXPIRED

    def test_active_orders_not_expired(self, order_manager, sample_signal):
        """Test that non-expired orders remain active."""
        order = order_manager.create_entry_order(
            sample_signal,
            timeout_minutes=60,  # Long timeout
        )
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        expired = order_manager.check_expired_orders()

        assert len(expired) == 0
        assert order.status == OrderStatus.SUBMITTED


# ============== Query Methods Tests ==============

class TestQueryMethods:
    """Test order query methods."""

    def test_get_orders_for_signal(self, order_manager, sample_signal):
        """Test getting orders by signal ID."""
        order = order_manager.create_entry_order(sample_signal)

        orders = order_manager.get_orders_for_signal(sample_signal.signal_id)

        assert len(orders) == 1
        assert orders[0].order_id == order.order_id

    def test_get_orders_for_position(self, order_manager):
        """Test getting orders by position ID."""
        position_id = "pos-123"

        order_manager.create_stop_loss_order(
            position_id=position_id,
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            quantity=10.0,
            stop_price=172.0,
        )

        order_manager.create_take_profit_order(
            position_id=position_id,
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            quantity=10.0,
            limit_price=181.0,
        )

        orders = order_manager.get_orders_for_position(position_id)

        assert len(orders) == 2

    def test_get_orders_for_symbol(self, order_manager, sample_signal):
        """Test getting orders by symbol."""
        order_manager.create_entry_order(sample_signal)

        orders = order_manager.get_orders_for_symbol("AAPL")

        assert len(orders) == 1

    def test_active_orders_property(self, order_manager, sample_signal):
        """Test active_orders property."""
        order = order_manager.create_entry_order(sample_signal)

        assert len(order_manager.active_orders) == 1

        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)

        assert len(order_manager.active_orders) == 0


# ============== Session Stats Tests ==============

class TestSessionStats:
    """Test session statistics tracking."""

    def test_session_stats_tracking(self, order_manager, sample_signal):
        """Test that session stats are tracked."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0, commission=1.50)

        stats = order_manager.get_session_stats()

        assert stats["orders_created"] == 1
        assert stats["orders_filled"] == 1
        assert stats["total_commission"] == 1.50

    def test_session_stats_reset(self, order_manager, sample_signal):
        """Test resetting session stats."""
        order = order_manager.create_entry_order(sample_signal)

        order_manager.reset_session_stats()

        stats = order_manager.get_session_stats()
        assert stats["orders_created"] == 0


# ============== Edge Cases ==============

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cannot_submit_filled_order(self, order_manager, sample_signal):
        """Test that filled orders cannot be submitted again."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)

        with pytest.raises(ValueError, match="cannot be submitted"):
            order_manager.mark_submitted(order.order_id, broker="other")

    def test_cannot_fill_cancelled_order(self, order_manager, sample_signal):
        """Test that cancelled orders cannot be filled."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.cancel_order(order.order_id, "cancelled")

        with pytest.raises(ValueError, match="not active"):
            order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)

    def test_cannot_overfill_order(self, order_manager, sample_signal):
        """Test that orders cannot be overfilled."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        with pytest.raises(ValueError, match="exceeds remaining"):
            order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=15.0)

    def test_order_not_found(self, order_manager):
        """Test error when order not found."""
        with pytest.raises(ValueError, match="not found"):
            order_manager.mark_submitted("nonexistent-id", broker="ibkr")

    def test_link_order_to_position(self, order_manager, sample_signal):
        """Test linking order to position after creation."""
        order = order_manager.create_entry_order(sample_signal)

        order_manager.link_order_to_position(order.order_id, "pos-123")

        assert order.position_id == "pos-123"
        orders = order_manager.get_orders_for_position("pos-123")
        assert len(orders) == 1


# ============== Order Properties Tests ==============

class TestOrderProperties:
    """Test Order dataclass properties."""

    def test_is_active(self, order_manager, sample_signal):
        """Test is_active property."""
        order = order_manager.create_entry_order(sample_signal)
        assert order.is_active

        order_manager.mark_submitted(order.order_id, broker="ibkr")
        assert order.is_active

        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)
        assert not order.is_active

    def test_is_complete(self, order_manager, sample_signal):
        """Test is_complete property."""
        order = order_manager.create_entry_order(sample_signal)
        assert not order.is_complete

        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)
        assert order.is_complete

    def test_fill_rate(self, order_manager, sample_signal):
        """Test fill_rate property."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=5.0)
        order = order_manager.get_order(order.order_id)

        assert order.fill_rate == 50.0

    def test_time_to_fill(self, order_manager, sample_signal):
        """Test time_to_fill_seconds property."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")

        # Simulate time passing
        order.submitted_at = datetime.utcnow() - timedelta(seconds=30)
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)

        assert order.time_to_fill_seconds is not None
        assert order.time_to_fill_seconds >= 30

    def test_to_dict(self, order_manager, sample_signal):
        """Test order serialization."""
        order = order_manager.create_entry_order(sample_signal)
        order_manager.mark_submitted(order.order_id, broker="ibkr")
        order_manager.record_fill(order.order_id, fill_price=175.0, fill_size=10.0)

        data = order.to_dict()

        assert data["order_id"] == order.order_id
        assert data["symbol"] == "AAPL"
        assert data["status"] == "filled"
        assert data["avg_fill_price"] == 175.0
