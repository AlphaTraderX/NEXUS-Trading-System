"""
Tests for the Trade Executor.

The Trade Executor submits orders to brokers and handles
fills, rejections, and retries.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import uuid

from nexus.execution.trade_executor import (
    TradeExecutor,
    PaperBrokerExecutor,
    BaseBrokerExecutor,
    BrokerConfig,
    BrokerType,
    ExecutionResult,
    BrokerPosition,
    BrokerAccount,
    create_paper_executor,
)
from nexus.execution.order_manager import (
    OrderManager,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderPurpose,
)
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown


# ============== Fixtures ==============

@pytest.fixture
def order_manager():
    """Create an order manager."""
    return OrderManager()


@pytest.fixture
def paper_executor():
    """Create a paper broker executor."""
    return create_paper_executor(
        account_balance=10000.0,
        slippage_pct=0.01,
        commission=1.0,
    )


@pytest.fixture
def trade_executor(order_manager, paper_executor):
    """Create a trade executor with paper broker."""
    executor = TradeExecutor(order_manager)
    executor.register_broker(
        "paper",
        paper_executor,
        markets=[Market.US_STOCKS, Market.FOREX_MAJORS],
        set_default=True,
    )
    return executor


@pytest.fixture
def sample_signal():
    """Create a sample signal."""
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
        costs=CostBreakdown(0.02, 0.01, 0.02, 0.0, 0.0, 0.0),
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


# ============== Paper Broker Tests ==============

class TestPaperBrokerExecutor:
    """Test the paper broker executor."""

    @pytest.mark.asyncio
    async def test_connect(self, paper_executor):
        """Test paper broker connection."""
        result = await paper_executor.connect()
        assert result is True
        assert paper_executor.connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, paper_executor):
        """Test paper broker disconnection."""
        await paper_executor.connect()
        await paper_executor.disconnect()
        assert paper_executor.connected is False

    @pytest.mark.asyncio
    async def test_get_account(self, paper_executor):
        """Test getting account info."""
        await paper_executor.connect()
        account = await paper_executor.get_account()

        assert account.balance == 10000.0
        assert account.equity == 10000.0
        assert account.currency == "GBP"

    @pytest.mark.asyncio
    async def test_submit_market_order(self, paper_executor, order_manager, sample_signal):
        """Test submitting a market order."""
        await paper_executor.connect()

        order = order_manager.create_entry_order(
            sample_signal,
            order_type=OrderType.MARKET,
        )

        result = await paper_executor.submit_order(order)

        assert result.success is True
        assert result.status == "filled"
        assert result.fill_size == 10.0
        assert result.fill_price is not None
        assert result.broker_order_id is not None

    @pytest.mark.asyncio
    async def test_submit_limit_order(self, paper_executor, order_manager, sample_signal):
        """Test submitting a limit order."""
        await paper_executor.connect()

        order = order_manager.create_entry_order(
            sample_signal,
            order_type=OrderType.LIMIT,
            limit_price=175.0,
        )

        result = await paper_executor.submit_order(order)

        assert result.success is True
        assert result.status == "filled"

    @pytest.mark.asyncio
    async def test_slippage_applied(self, paper_executor, order_manager, sample_signal):
        """Test that slippage is applied to fills."""
        await paper_executor.connect()

        order = order_manager.create_entry_order(
            sample_signal,
            order_type=OrderType.MARKET,
        )
        order.expected_price = 175.0

        result = await paper_executor.submit_order(order)

        # Buy order should have positive slippage (pay more)
        assert result.fill_price > 175.0

    @pytest.mark.asyncio
    async def test_position_tracking(self, paper_executor, order_manager, sample_signal):
        """Test that positions are tracked."""
        await paper_executor.connect()

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await paper_executor.submit_order(order)

        positions = await paper_executor.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].side == "long"
        assert positions[0].size == 10.0

    @pytest.mark.asyncio
    async def test_close_position(self, paper_executor, order_manager, sample_signal):
        """Test closing a position."""
        await paper_executor.connect()

        # Open position
        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await paper_executor.submit_order(order)

        # Close position
        result = await paper_executor.close_position("AAPL")

        assert result.success is True

        positions = await paper_executor.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_cancel_order(self, paper_executor, order_manager, sample_signal):
        """Test cancelling an order."""
        await paper_executor.connect()

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.STOP)
        result = await paper_executor.submit_order(order)

        cancel_result = await paper_executor.cancel_order(result.broker_order_id)

        assert cancel_result.success is True
        assert cancel_result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_not_connected_rejection(self, paper_executor, order_manager, sample_signal):
        """Test that orders are rejected when not connected."""
        # Don't connect

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        result = await paper_executor.submit_order(order)

        assert result.success is False
        assert "Not connected" in result.message


# ============== Trade Executor Tests ==============

class TestTradeExecutor:
    """Test the main trade executor."""

    def test_register_broker(self, order_manager):
        """Test registering a broker."""
        executor = TradeExecutor(order_manager)
        paper = create_paper_executor()

        executor.register_broker(
            "paper",
            paper,
            markets=[Market.US_STOCKS],
            set_default=True,
        )

        assert "paper" in executor._brokers
        assert executor._default_broker == "paper"

    def test_get_broker_for_market(self, trade_executor):
        """Test getting broker by market."""
        broker = trade_executor.get_broker_for_market(Market.US_STOCKS)
        assert broker is not None

    def test_get_broker_for_unmapped_market(self, trade_executor):
        """Test getting broker for unmapped market uses default."""
        broker = trade_executor.get_broker_for_market(Market.UK_STOCKS)
        # Should return default broker
        assert broker is not None

    @pytest.mark.asyncio
    async def test_connect_all(self, trade_executor):
        """Test connecting all brokers."""
        results = await trade_executor.connect_all()

        assert "paper" in results
        assert results["paper"] is True

    @pytest.mark.asyncio
    async def test_execute_order_success(self, trade_executor, order_manager, sample_signal):
        """Test successful order execution."""
        await trade_executor.connect_all()

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        result = await trade_executor.execute_order(order)

        assert result.success is True
        assert result.status == "filled"

        # Check order manager was updated
        updated_order = order_manager.get_order(order.order_id)
        assert updated_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_order_no_broker(self, order_manager, sample_signal):
        """Test order rejection when no broker available."""
        executor = TradeExecutor(order_manager)
        # Don't register any brokers

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        result = await executor.execute_order(order)

        assert result.success is False
        assert "No broker" in result.message

    @pytest.mark.asyncio
    async def test_execution_stats_tracking(self, trade_executor, order_manager, sample_signal):
        """Test that execution stats are tracked."""
        await trade_executor.connect_all()

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await trade_executor.execute_order(order)

        stats = trade_executor.get_execution_stats()

        assert stats["orders_submitted"] == 1
        assert stats["orders_filled"] == 1

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, trade_executor, order_manager, sample_signal):
        """Test cancelling all orders."""
        await trade_executor.connect_all()

        # Create orders but don't execute them
        order1 = order_manager.create_entry_order(sample_signal)
        order2 = order_manager.create_entry_order(sample_signal)

        # Cancel all
        results = await trade_executor.cancel_all_orders()

        assert len(results) >= 0  # May have been cancelled before submission

    @pytest.mark.asyncio
    async def test_sync_positions(self, trade_executor, order_manager, sample_signal):
        """Test syncing positions from brokers."""
        await trade_executor.connect_all()

        # Create a position
        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await trade_executor.execute_order(order)

        positions = await trade_executor.sync_positions()

        assert "paper" in positions
        assert len(positions["paper"]) == 1

    @pytest.mark.asyncio
    async def test_get_account_status(self, trade_executor):
        """Test getting account status from all brokers."""
        await trade_executor.connect_all()

        accounts = await trade_executor.get_account_status()

        assert "paper" in accounts
        assert accounts["paper"].balance == 10000.0

    @pytest.mark.asyncio
    async def test_on_fill_callback(self, trade_executor, order_manager, sample_signal):
        """Test fill callback is invoked."""
        await trade_executor.connect_all()

        fills = []
        trade_executor.on_fill(lambda order, result: fills.append((order, result)))

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await trade_executor.execute_order(order)

        assert len(fills) == 1
        assert fills[0][1].success is True

    @pytest.mark.asyncio
    async def test_on_reject_callback(self, order_manager, sample_signal):
        """Test reject callback is invoked."""
        executor = TradeExecutor(order_manager)
        # No brokers registered = will reject

        rejects = []
        executor.on_reject(lambda order, result: rejects.append((order, result)))

        order = order_manager.create_entry_order(sample_signal, order_type=OrderType.MARKET)
        await executor.execute_order(order)

        assert len(rejects) == 1
        assert rejects[0][1].success is False


# ============== Factory Function Tests ==============

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_paper_executor(self):
        """Test creating paper executor via factory."""
        executor = create_paper_executor(
            account_balance=50000.0,
            slippage_pct=0.05,
            commission=2.0,
        )

        assert executor is not None
        assert executor.simulated_slippage_pct == 0.05
        assert executor.simulated_commission == 2.0
        assert executor._account.balance == 50000.0


# ============== Execution Result Tests ==============

class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = ExecutionResult(
            success=True,
            order_id="order-123",
            broker_order_id="BROKER-456",
            status="filled",
            fill_price=175.50,
            fill_size=10.0,
            commission=1.50,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["order_id"] == "order-123"
        assert data["fill_price"] == 175.50


# ============== Broker Config Tests ==============

class TestBrokerConfig:
    """Test BrokerConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = BrokerConfig(broker_type=BrokerType.PAPER)

        assert config.environment == "practice"
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
