"""
Validation script for Reconciliation Engine.

Run with: python -m nexus.scripts.validate_reconciliation
"""

import asyncio
from datetime import datetime, timedelta
import uuid

from nexus.execution.reconciliation import (
    ReconciliationEngine,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ReconciliationAction,
)
from nexus.execution.position_manager import PositionManager
from nexus.execution.order_manager import OrderManager
from nexus.execution.trade_executor import (
    TradeExecutor,
    BrokerPosition,
    create_paper_executor,
)
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown


def create_test_signal(symbol: str = "AAPL", direction: Direction = Direction.LONG) -> NexusSignal:
    """Create a test signal."""
    return NexusSignal(
        signal_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        opportunity_id=str(uuid.uuid4()),
        symbol=symbol,
        market=Market.US_STOCKS,
        direction=direction,
        entry_price=175.0,
        stop_loss=172.0 if direction == Direction.LONG else 178.0,
        take_profit=181.0 if direction == Direction.LONG else 169.0,
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
        confluence_factors=[],
        risk_factors=[],
        market_context="Test",
        session="regular",
        valid_until=datetime.utcnow() + timedelta(hours=4),
        status=SignalStatus.PENDING,
    )


async def run_validation():
    """Run all validation scenarios."""

    print("=" * 60)
    print("RECONCILIATION ENGINE VALIDATION")
    print("=" * 60)

    passed = 0
    failed = 0

    # Setup
    position_manager = PositionManager(max_positions=10)
    order_manager = OrderManager()
    paper_executor = create_paper_executor(account_balance=10000.0)

    trade_executor = TradeExecutor(order_manager)
    trade_executor.register_broker("paper", paper_executor, set_default=True)

    engine = ReconciliationEngine(
        position_manager=position_manager,
        order_manager=order_manager,
        trade_executor=trade_executor,
        auto_sync_enabled=True,
    )

    await paper_executor.connect()

    # ============== Scenario 1: Clean reconciliation ==============
    print("\n[1] Clean Reconciliation (no positions)")
    try:
        report = await engine.reconcile()
        assert report.is_clean, "Expected clean report"
        assert report.completed_at is not None, "Should have completion time"
        print(f"    [PASS] - Reconciliation CLEAN (no positions)")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 2: Detect untracked broker position ==============
    print("\n[2] Detect Untracked Broker Position")
    try:
        # Add position at broker only
        paper_executor._positions["TSLA"] = BrokerPosition(
            symbol="TSLA",
            side="long",
            size=5.0,
            avg_entry_price=250.0,
            current_price=255.0,
            unrealized_pnl=25.0,
        )

        report = await engine.reconcile()

        assert not report.is_clean, "Should have discrepancies"
        assert any(d.discrepancy_type == DiscrepancyType.POSITION_MISSING_INTERNAL
                   for d in report.discrepancies), "Should detect missing internal"

        print(f"    [PASS] - Found untracked TSLA position")
        passed += 1

        # Cleanup
        del paper_executor._positions["TSLA"]
        engine._unresolved.clear()
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 3: Detect missing broker position ==============
    print("\n[3] Detect Position Missing at Broker")
    try:
        # Create internal position without broker position
        signal = create_test_signal("MSFT")
        position = position_manager.create_position_from_signal(signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 400.0, datetime.utcnow())

        report = await engine.reconcile()

        assert any(d.discrepancy_type == DiscrepancyType.POSITION_MISSING_BROKER
                   for d in report.discrepancies), "Should detect missing broker"

        print(f"    [PASS] - Detected MSFT missing at broker")
        passed += 1

        # Cleanup
        position_manager.close_position(position.position_id, 400.0, datetime.utcnow(), "test")
        engine._unresolved.clear()
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 4: Detect and auto-sync size mismatch ==============
    print("\n[4] Auto-Sync Size Mismatch")
    try:
        # Create matching positions with different sizes
        signal = create_test_signal("NVDA")
        position = position_manager.create_position_from_signal(signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 500.0, datetime.utcnow())

        original_size = position.size

        paper_executor._positions["NVDA"] = BrokerPosition(
            symbol="NVDA",
            side="long",
            size=7.0,  # Different from position.size (10)
            avg_entry_price=500.0,
            current_price=510.0,
            unrealized_pnl=70.0,
        )

        report = await engine.reconcile()

        assert report.auto_synced >= 1, "Should have auto-synced"
        assert position.size == 7.0, f"Size should be synced to 7.0, got {position.size}"

        print(f"    [PASS] - Auto-synced size from {original_size} to {position.size}")
        passed += 1

        # Cleanup
        position_manager.close_position(position.position_id, 510.0, datetime.utcnow(), "test")
        del paper_executor._positions["NVDA"]
        engine._unresolved.clear()
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 5: Side mismatch is CRITICAL ==============
    print("\n[5] Side Mismatch Triggers CRITICAL")
    try:
        signal = create_test_signal("GOOGL", Direction.LONG)
        position = position_manager.create_position_from_signal(signal)
        position_manager.mark_submitted(position.position_id)
        position_manager.mark_open(position.position_id, 150.0, datetime.utcnow())

        paper_executor._positions["GOOGL"] = BrokerPosition(
            symbol="GOOGL",
            side="short",  # WRONG SIDE!
            size=10.0,
            avg_entry_price=150.0,
            current_price=145.0,
            unrealized_pnl=50.0,
        )

        report = await engine.reconcile()

        assert report.has_critical, "Should have critical discrepancy"

        print(f"    [PASS] - Side mismatch flagged as CRITICAL")
        passed += 1

        # Cleanup
        position_manager.close_position(position.position_id, 150.0, datetime.utcnow(), "test")
        del paper_executor._positions["GOOGL"]
        engine._unresolved.clear()
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 6: Resolve discrepancy manually ==============
    print("\n[6] Manual Discrepancy Resolution")
    try:
        discrepancy = Discrepancy(
            discrepancy_id="test-disc-001",
            discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
            severity=DiscrepancySeverity.WARNING,
            broker="paper",
            symbol="TEST",
            description="Test discrepancy",
            internal_value="None",
            broker_value="long 10",
        )
        engine._unresolved["test-disc-001"] = discrepancy

        result = engine.resolve_discrepancy(
            "test-disc-001",
            ReconciliationAction.MANUAL_REVIEW,
            "Manually verified"
        )

        assert result is True, "Should resolve successfully"
        assert len(engine.get_unresolved()) == 0, "Should be no unresolved"

        print(f"    [PASS] - Discrepancy resolved manually")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 7: Callbacks fire correctly ==============
    print("\n[7] Discrepancy Callbacks")
    try:
        callbacks_received = []
        engine.on_discrepancy(lambda d: callbacks_received.append(d))

        # Create discrepancy-producing situation
        paper_executor._positions["AMZN"] = BrokerPosition(
            symbol="AMZN",
            side="long",
            size=3.0,
            avg_entry_price=180.0,
            current_price=185.0,
            unrealized_pnl=15.0,
        )

        await engine.reconcile()

        assert len(callbacks_received) > 0, "Callback should have fired"

        print(f"    [PASS] - Received {len(callbacks_received)} callback(s)")
        passed += 1

        # Cleanup
        del paper_executor._positions["AMZN"]
        engine._unresolved.clear()
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 8: Stats tracking ==============
    print("\n[8] Stats Tracking")
    try:
        stats = engine.get_stats()

        assert stats["total_reconciliations"] > 0, "Should have run reconciliations"
        assert stats["last_reconciliation"] is not None, "Should have timestamp"

        print(f"    [PASS] - {stats['total_reconciliations']} reconciliations tracked")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 9: Get last report ==============
    print("\n[9] Get Last Report")
    try:
        report = engine.get_last_report()

        assert report is not None, "Should have a report"
        assert report.report_id is not None, "Report should have ID"

        print(f"    [PASS] - Last report ID: {report.report_id[:8]}...")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # ============== Scenario 10: Report serialization ==============
    print("\n[10] Report Serialization")
    try:
        report = engine.get_last_report()
        data = report.to_dict()

        assert "report_id" in data, "Should have report_id"
        assert "is_clean" in data, "Should have is_clean"
        assert "discrepancies" in data, "Should have discrepancies list"

        print(f"    [PASS] - Report serializes correctly")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] - {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_validation())
    exit(0 if success else 1)
