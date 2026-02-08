"""
Tests for the Reconciliation Engine.

The Site Inspector that compares internal state against broker state.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import uuid

from nexus.execution.reconciliation import (
    ReconciliationEngine,
    ReconciliationReport,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ReconciliationAction,
)
from nexus.execution.position_manager import PositionManager, Position, PositionStatus
from nexus.execution.order_manager import OrderManager, Order, OrderStatus
from nexus.execution.trade_executor import (
    TradeExecutor,
    PaperBrokerExecutor,
    BrokerPosition,
    BrokerAccount,
    BrokerConfig,
    BrokerType,
    create_paper_executor,
)
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown


# ============== Fixtures ==============

@pytest.fixture
def position_manager():
    """Create position manager."""
    return PositionManager(max_positions=10)


@pytest.fixture
def order_manager():
    """Create order manager."""
    return OrderManager()


@pytest.fixture
def paper_executor():
    """Create paper broker."""
    return create_paper_executor(account_balance=10000.0)


@pytest.fixture
def trade_executor(order_manager, paper_executor):
    """Create trade executor with paper broker."""
    executor = TradeExecutor(order_manager)
    executor.register_broker("paper", paper_executor, set_default=True)
    return executor


@pytest.fixture
def reconciliation_engine(position_manager, order_manager, trade_executor):
    """Create reconciliation engine."""
    return ReconciliationEngine(
        position_manager=position_manager,
        order_manager=order_manager,
        trade_executor=trade_executor,
        auto_sync_enabled=True,
    )


@pytest.fixture
def sample_signal():
    """Create sample signal."""
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
        ai_reasoning="Test",
        confluence_factors=[],
        risk_factors=[],
        market_context="Test",
        session="regular",
        valid_until=datetime.utcnow() + timedelta(hours=4),
        status=SignalStatus.PENDING,
    )


# ============== Discrepancy Tests ==============

class TestDiscrepancy:
    """Test Discrepancy dataclass."""

    def test_create_discrepancy(self):
        """Test creating a discrepancy."""
        d = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Size mismatch",
            internal_value=10,
            broker_value=8,
        )

        assert d.discrepancy_id == "disc-001"
        assert d.resolved is False

    def test_resolve_discrepancy(self):
        """Test resolving a discrepancy."""
        d = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Size mismatch",
            internal_value=10,
            broker_value=8,
        )

        d.resolve(ReconciliationAction.AUTO_SYNC, "Synced size")

        assert d.resolved is True
        assert d.action_taken == ReconciliationAction.AUTO_SYNC
        assert d.resolution_notes == "Synced size"
        assert d.resolved_at is not None

    def test_to_dict(self):
        """Test serialization."""
        d = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Size mismatch",
            internal_value=10,
            broker_value=8,
        )

        data = d.to_dict()

        assert data["discrepancy_id"] == "disc-001"
        assert data["type"] == "position_size_mismatch"
        assert data["severity"] == "warning"


# ============== Report Tests ==============

class TestReconciliationReport:
    """Test ReconciliationReport dataclass."""

    def test_empty_report_is_clean(self):
        """Test that empty report is clean."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )

        assert report.is_clean is True
        assert report.has_critical is False

    def test_report_with_discrepancy_not_clean(self):
        """Test report with discrepancy is not clean."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )

        report.add_discrepancy(Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Test",
            internal_value=10,
            broker_value=8,
        ))

        assert report.is_clean is False

    def test_has_critical(self):
        """Test critical detection."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )

        report.add_discrepancy(Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIDE_MISMATCH,
            severity=DiscrepancySeverity.CRITICAL,
            broker="paper",
            symbol="AAPL",
            description="Side mismatch",
            internal_value="long",
            broker_value="short",
        ))

        assert report.has_critical is True

    def test_unresolved_count(self):
        """Test unresolved count."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )

        d1 = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Test",
            internal_value=10,
            broker_value=8,
        )
        d2 = Discrepancy(
            discrepancy_id="disc-002",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="MSFT",
            description="Test",
            internal_value=5,
            broker_value=3,
        )

        report.add_discrepancy(d1)
        report.add_discrepancy(d2)

        assert report.unresolved_count == 2

        d1.resolve(ReconciliationAction.AUTO_SYNC)

        assert report.unresolved_count == 1

    def test_get_summary_clean(self):
        """Test clean summary."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
            positions_internal=5,
            positions_broker=5,
        )

        summary = report.get_summary()

        assert "✅" in summary
        assert "CLEAN" in summary

    def test_get_summary_with_issues(self):
        """Test summary with issues."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )

        report.add_discrepancy(Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Test",
            internal_value=10,
            broker_value=8,
        ))

        summary = report.get_summary()

        assert "⚠️" in summary
        assert "1 discrepancies" in summary

    def test_to_dict(self):
        """Test serialization."""
        report = ReconciliationReport(
            report_id="rpt-001",
            started_at=datetime.utcnow(),
        )
        report.completed_at = datetime.utcnow()

        data = report.to_dict()

        assert data["report_id"] == "rpt-001"
        assert data["is_clean"] is True


# ============== Reconciliation Engine Tests ==============

class TestReconciliationEngine:
    """Test the main reconciliation engine."""

    @pytest.mark.asyncio
    async def test_clean_reconciliation(self, reconciliation_engine, paper_executor):
        """Test reconciliation with no discrepancies."""
        await paper_executor.connect()

        report = await reconciliation_engine.reconcile()

        assert report.is_clean is True
        assert report.completed_at is not None

    @pytest.mark.asyncio
    async def test_detect_position_missing_internal(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test detecting broker position not tracked internally."""
        await paper_executor.connect()

        # Create position at broker only (manual trade simulation)
        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="long",
            size=10.0,
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=10.0,
        )

        report = await reconciliation_engine.reconcile()

        assert report.is_clean is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.POSITION_MISSING_INTERNAL

    @pytest.mark.asyncio
    async def test_detect_position_missing_broker(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test detecting internal position not at broker."""
        await paper_executor.connect()

        # Create internal position only
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 175.0, datetime.utcnow())

        report = await reconciliation_engine.reconcile()

        assert report.is_clean is False
        assert any(d.discrepancy_type == DiscrepancyType.POSITION_MISSING_BROKER
                   for d in report.discrepancies)

    @pytest.mark.asyncio
    async def test_detect_size_mismatch(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test detecting position size mismatch."""
        await paper_executor.connect()

        # Create matching positions with different sizes
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 175.0, datetime.utcnow())

        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="long",
            size=8.0,  # Different size!
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=8.0,
        )

        report = await reconciliation_engine.reconcile()

        assert any(d.discrepancy_type == DiscrepancyType.POSITION_SIZE_MISMATCH
                   for d in report.discrepancies)

    @pytest.mark.asyncio
    async def test_auto_sync_size(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test auto-sync updates internal size."""
        await paper_executor.connect()

        # Create position with size 10 internally
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 175.0, datetime.utcnow())

        assert position.size == 10.0

        # Broker has size 8
        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="long",
            size=8.0,
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=8.0,
        )

        report = await reconciliation_engine.reconcile()

        # Should have auto-synced
        assert report.auto_synced >= 1
        assert position.size == 8.0  # Updated!

    @pytest.mark.asyncio
    async def test_detect_side_mismatch_critical(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test side mismatch is critical."""
        await paper_executor.connect()

        # Create long position internally
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 175.0, datetime.utcnow())

        # Broker has short position!
        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="short",  # WRONG SIDE
            size=10.0,
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=-10.0,
        )

        report = await reconciliation_engine.reconcile()

        assert report.has_critical is True
        assert any(d.severity == DiscrepancySeverity.CRITICAL for d in report.discrepancies)

    @pytest.mark.asyncio
    async def test_discrepancy_callback(
        self, reconciliation_engine, paper_executor
    ):
        """Test discrepancy callback is fired."""
        await paper_executor.connect()

        discrepancies_received = []
        reconciliation_engine.on_discrepancy(lambda d: discrepancies_received.append(d))

        # Create broker position not tracked internally
        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="long",
            size=10.0,
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=10.0,
        )

        await reconciliation_engine.reconcile()

        assert len(discrepancies_received) == 1

    @pytest.mark.asyncio
    async def test_critical_callback(
        self, reconciliation_engine, paper_executor, position_manager, sample_signal
    ):
        """Test critical callback is fired for critical issues."""
        await paper_executor.connect()

        critical_received = []
        reconciliation_engine.on_critical(lambda d: critical_received.append(d))

        # Create side mismatch (critical)
        position = position_manager.create_position_from_signal(sample_signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 175.0, datetime.utcnow())

        paper_executor._positions["AAPL"] = BrokerPosition(
            symbol="AAPL",
            side="short",
            size=10.0,
            avg_entry_price=175.0,
            current_price=176.0,
            unrealized_pnl=-10.0,
        )

        await reconciliation_engine.reconcile()

        assert len(critical_received) >= 1

    def test_resolve_discrepancy(self, reconciliation_engine):
        """Test manually resolving a discrepancy."""
        # Add a discrepancy
        discrepancy = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Test",
            internal_value="None",
            broker_value="long 10",
        )
        reconciliation_engine._unresolved["disc-001"] = discrepancy

        result = reconciliation_engine.resolve_discrepancy(
            "disc-001",
            ReconciliationAction.MANUAL_REVIEW,
            "Manually checked, position is correct"
        )

        assert result is True
        assert "disc-001" not in reconciliation_engine._unresolved
        assert discrepancy.resolved is True

    def test_get_unresolved(self, reconciliation_engine):
        """Test getting unresolved discrepancies."""
        d1 = Discrepancy(
            discrepancy_id="disc-001",
            discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="AAPL",
            description="Test",
            internal_value="None",
            broker_value="long 10",
        )
        d2 = Discrepancy(
            discrepancy_id="disc-002",
            discrepancy_type=DiscrepancyType.POSITION_SIDE_MISMATCH,
            severity=DiscrepancySeverity.CRITICAL,
            broker="paper",
            symbol="MSFT",
            description="Test",
            internal_value="long",
            broker_value="short",
        )

        reconciliation_engine._unresolved["disc-001"] = d1
        reconciliation_engine._unresolved["disc-002"] = d2

        unresolved = reconciliation_engine.get_unresolved()
        assert len(unresolved) == 2

        critical = reconciliation_engine.get_unresolved_by_severity(DiscrepancySeverity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].symbol == "MSFT"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, reconciliation_engine, paper_executor):
        """Test stats are tracked."""
        await paper_executor.connect()

        await reconciliation_engine.reconcile()
        await reconciliation_engine.reconcile()

        stats = reconciliation_engine.get_stats()

        assert stats["total_reconciliations"] == 2
        assert stats["last_reconciliation"] is not None

    @pytest.mark.asyncio
    async def test_get_last_report(self, reconciliation_engine, paper_executor):
        """Test getting last report."""
        await paper_executor.connect()

        assert reconciliation_engine.get_last_report() is None

        await reconciliation_engine.reconcile()

        report = reconciliation_engine.get_last_report()
        assert report is not None


# ============== Integration Tests ==============

class TestReconciliationIntegration:
    """Integration tests for reconciliation."""

    @pytest.mark.asyncio
    async def test_full_reconciliation_flow(
        self,
        reconciliation_engine,
        paper_executor,
        position_manager,
        order_manager,
        trade_executor,
        sample_signal
    ):
        """Test complete reconciliation flow."""
        await trade_executor.connect_all()

        # Create position through proper flow
        position = position_manager.create_position_from_signal(sample_signal)
        order = order_manager.create_entry_order(sample_signal)

        # Execute order
        result = await trade_executor.execute_order(order)
        assert result.success is True

        # Mark position open
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(
            position.position_id,
            result.fill_price,
            datetime.utcnow()
        )

        # Now reconcile - should be clean since we're in sync
        report = await reconciliation_engine.reconcile()

        # May have small discrepancies due to slippage
        # but should not be critical
        assert not report.has_critical
