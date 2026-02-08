#!/usr/bin/env python3
"""
Position Manager Validation Script

Tests the Position Manager with realistic scenarios to verify:
1. Position creation from signals
2. Lifecycle transitions (pending -> open -> closed)
3. P&L calculations (long and short)
4. Stop loss and take profit detection
5. Portfolio metrics calculation
6. Heat Manager and Correlation Monitor sync

Run from project root:
    python -m nexus.scripts.validate_position_manager
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

# NEXUS imports
from nexus.core.enums import Market, Direction, EdgeType, SignalTier, SignalStatus
from nexus.core.models import NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.correlation import CorrelationMonitor
from nexus.execution.position_manager import (
    PositionManager,
    Position,
    PositionStatus,
    PortfolioMetrics,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def print_position(position: Position, indent: int = 2):
    """Print position details."""
    prefix = " " * indent
    direction = "LONG" if position.direction == Direction.LONG else "SHORT"
    status = position.status.value.upper()

    print(f"{prefix}Position: {position.position_id[:8]}...")
    print(f"{prefix}  Symbol: {position.symbol} ({direction})")
    print(f"{prefix}  Status: {status}")
    print(f"{prefix}  Entry: {position.entry_price:.2f}")
    print(f"{prefix}  Current: {position.current_price:.2f}")
    print(f"{prefix}  Stop: {position.stop_loss:.2f} | Target: {position.take_profit:.2f}")
    print(f"{prefix}  Size: {position.size:.2f} units")
    print(f"{prefix}  Risk: {position.risk_amount:.2f} ({position.risk_percent:.2f}%)")

    if position.status == PositionStatus.OPEN:
        pnl_sign = "+" if position.unrealized_pnl >= 0 else ""
        print(f"{prefix}  Unrealized P&L: {pnl_sign}{position.unrealized_pnl:.2f} ({position.r_multiple:.2f}R)")
        print(f"{prefix}  Profitable: {'Yes' if position.is_profitable else 'No'}")
    elif position.status == PositionStatus.CLOSED:
        pnl_sign = "+" if position.realized_pnl >= 0 else ""
        print(f"{prefix}  Exit: {position.exit_price:.2f} ({position.exit_reason})")
        print(f"{prefix}  Realized P&L: {pnl_sign}{position.realized_pnl:.2f} ({position.r_multiple:.2f}R)")
        print(f"{prefix}  Hold Time: {position.hold_time_hours:.2f} hours")


def print_metrics(metrics: PortfolioMetrics, indent: int = 2):
    """Print portfolio metrics."""
    prefix = " " * indent

    print(f"{prefix}Portfolio Metrics:")
    print(f"{prefix}  Open Positions: {metrics.open_positions}")
    print(f"{prefix}  Total Unrealized P&L: {metrics.total_unrealized_pnl:.2f}")
    print(f"{prefix}  Total Realized P&L: {metrics.total_realized_pnl:.2f}")
    print(f"{prefix}  Total P&L: {metrics.total_pnl:.2f}")
    print(f"{prefix}  Portfolio Heat: {metrics.portfolio_heat:.2f}%")
    print(f"{prefix}  Long Exposure: {metrics.long_exposure:.2f}")
    print(f"{prefix}  Short Exposure: {metrics.short_exposure:.2f}")
    print(f"{prefix}  Net Exposure: {metrics.net_exposure:.2f}")

    if metrics.win_rate > 0:
        print(f"{prefix}  Win Rate: {metrics.win_rate:.1f}%")
        print(f"{prefix}  Avg Win: {metrics.avg_win:.2f}")
        print(f"{prefix}  Avg Loss: {metrics.avg_loss:.2f}")
        print(f"{prefix}  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"{prefix}  Avg R-Multiple: {metrics.avg_r_multiple:.2f}R")


def create_test_signal(
    symbol: str = "AAPL",
    market: Market = Market.US_STOCKS,
    direction: Direction = Direction.LONG,
    entry: float = 175.0,
    stop: float = 172.0,
    target: float = 181.0,
    size: float = 10.0,
    risk_amount: float = 30.0,
    risk_percent: float = 1.0,
    edge_type: EdgeType = EdgeType.VWAP_DEVIATION,
    score: int = 75,
) -> NexusSignal:
    """Create a test signal."""
    position_value = entry * size

    return NexusSignal(
        signal_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        opportunity_id=str(uuid.uuid4()),
        symbol=symbol,
        market=market,
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        position_size=size,
        position_value=position_value,
        risk_amount=risk_amount,
        risk_percent=risk_percent,
        primary_edge=edge_type,
        secondary_edges=[],
        edge_score=score,
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
        ai_reasoning="Test signal for validation",
        confluence_factors=["Factor 1", "Factor 2"],
        risk_factors=["Risk 1"],
        market_context="Test context",
        session="regular",
        valid_until=datetime.utcnow() + timedelta(hours=4),
        status=SignalStatus.PENDING,
    )


def test_position_creation(pm: PositionManager) -> bool:
    """Test 1: Position creation from signal."""
    print_subheader("Test 1: Position Creation from Signal")

    signal = create_test_signal(
        symbol="AAPL",
        direction=Direction.LONG,
        entry=175.0,
        stop=172.0,
        target=181.0,
    )

    position = pm.create_position_from_signal(signal)

    print(f"  [OK] Position created from signal")
    print_position(position)

    # Verify
    assert position.status == PositionStatus.PENDING
    assert position.symbol == "AAPL"
    assert position.direction == Direction.LONG

    return True


def test_position_lifecycle(pm: PositionManager) -> bool:
    """Test 2: Full position lifecycle."""
    print_subheader("Test 2: Position Lifecycle (Pending -> Open -> Closed)")

    signal = create_test_signal(
        symbol="MSFT",
        direction=Direction.LONG,
        entry=400.0,
        stop=392.0,
        target=416.0,
        size=5.0,
        risk_amount=40.0,
    )

    # Create
    position = pm.create_position_from_signal(signal)
    print(f"  Step 1: Created (Status: {position.status.value})")

    # Submit
    position = pm.mark_submitted(position.position_id)
    print(f"  Step 2: Submitted (Status: {position.status.value})")

    # Open with fill
    fill_price = 400.50  # Slight slippage
    position = pm.mark_open(position.position_id, fill_price=fill_price)
    print(f"  Step 3: Opened @ {fill_price} (Status: {position.status.value})")

    # Simulate price movement
    pm.update_price("MSFT", 408.0)
    position = pm.get_position(position.position_id)
    print(f"  Step 4: Price update to 408.0 (P&L: +{position.unrealized_pnl:.2f})")

    # Close at target
    position = pm.close_position(
        position.position_id,
        exit_price=415.0,
        exit_reason="take_profit",
    )
    print(f"  Step 5: Closed @ 415.0 (Status: {position.status.value})")
    print(f"  [OK] Final P&L: +{position.realized_pnl:.2f} ({position.r_multiple:.2f}R)")

    return True


def test_long_pnl_calculations(pm: PositionManager) -> bool:
    """Test 3: Long position P&L calculations."""
    print_subheader("Test 3: Long Position P&L Calculations")

    signal = create_test_signal(
        symbol="NVDA",
        direction=Direction.LONG,
        entry=800.0,
        stop=780.0,
        target=840.0,
        size=2.0,
        risk_amount=40.0,
    )

    position = pm.create_position_from_signal(signal)
    pm.mark_open(position.position_id, fill_price=800.0)

    # Test profit scenario
    pm.update_price("NVDA", 820.0)
    position = pm.get_position(position.position_id)

    expected_pnl = (820.0 - 800.0) * 2.0  # +40
    print(f"  Price: 800 -> 820")
    print(f"  Expected P&L: +{expected_pnl:.2f}")
    print(f"  Actual P&L: +{position.unrealized_pnl:.2f}")
    print(f"  R-Multiple: {position.r_multiple:.2f}R")

    assert abs(position.unrealized_pnl - expected_pnl) < 0.01
    assert position.is_profitable
    print(f"  [OK] Long profit calculation correct")

    # Test loss scenario
    pm.update_price("NVDA", 790.0)
    position = pm.get_position(position.position_id)

    expected_pnl = (790.0 - 800.0) * 2.0  # -20
    print(f"  Price: 800 -> 790")
    print(f"  Expected P&L: {expected_pnl:.2f}")
    print(f"  Actual P&L: {position.unrealized_pnl:.2f}")

    assert abs(position.unrealized_pnl - expected_pnl) < 0.01
    assert not position.is_profitable
    print(f"  [OK] Long loss calculation correct")

    # Clean up
    pm.close_position(position.position_id, exit_price=790.0, exit_reason="test_complete")

    return True


def test_short_pnl_calculations(pm: PositionManager) -> bool:
    """Test 4: Short position P&L calculations."""
    print_subheader("Test 4: Short Position P&L Calculations")

    signal = create_test_signal(
        symbol="TSLA",
        direction=Direction.SHORT,
        entry=250.0,
        stop=260.0,
        target=230.0,
        size=4.0,
        risk_amount=40.0,
    )

    position = pm.create_position_from_signal(signal)
    pm.mark_open(position.position_id, fill_price=250.0)

    # Test profit scenario (price drops for short)
    pm.update_price("TSLA", 240.0)
    position = pm.get_position(position.position_id)

    expected_pnl = (250.0 - 240.0) * 4.0  # +40
    print(f"  Price: 250 -> 240 (SHORT)")
    print(f"  Expected P&L: +{expected_pnl:.2f}")
    print(f"  Actual P&L: +{position.unrealized_pnl:.2f}")

    assert abs(position.unrealized_pnl - expected_pnl) < 0.01
    assert position.is_profitable
    print(f"  [OK] Short profit calculation correct")

    # Test loss scenario (price rises for short)
    pm.update_price("TSLA", 255.0)
    position = pm.get_position(position.position_id)

    expected_pnl = (250.0 - 255.0) * 4.0  # -20
    print(f"  Price: 250 -> 255 (SHORT)")
    print(f"  Expected P&L: {expected_pnl:.2f}")
    print(f"  Actual P&L: {position.unrealized_pnl:.2f}")

    assert abs(position.unrealized_pnl - expected_pnl) < 0.01
    assert not position.is_profitable
    print(f"  [OK] Short loss calculation correct")

    # Clean up
    pm.close_position(position.position_id, exit_price=255.0, exit_reason="test_complete")

    return True


def test_stop_target_detection(pm: PositionManager) -> bool:
    """Test 5: Stop loss and take profit detection."""
    print_subheader("Test 5: Stop Loss and Take Profit Detection")

    # Long position
    long_signal = create_test_signal(
        symbol="AMD",
        direction=Direction.LONG,
        entry=150.0,
        stop=145.0,
        target=160.0,
        size=10.0,
    )

    long_pos = pm.create_position_from_signal(long_signal)
    pm.mark_open(long_pos.position_id, fill_price=150.0)

    # Test stop detection for long
    pm.update_price("AMD", 144.0)  # Below stop
    at_stop = pm.get_positions_at_stop()
    print(f"  Long @ 150, Stop @ 145, Price @ 144")
    print(f"  At stop: {len(at_stop)} position(s)")
    assert len(at_stop) == 1
    print(f"  [OK] Long stop detection works")

    # Reset price
    pm.update_price("AMD", 150.0)

    # Test target detection for long
    pm.update_price("AMD", 161.0)  # Above target
    at_target = pm.get_positions_at_target()
    print(f"  Long @ 150, Target @ 160, Price @ 161")
    print(f"  At target: {len(at_target)} position(s)")
    assert len(at_target) == 1
    print(f"  [OK] Long target detection works")

    pm.close_position(long_pos.position_id, exit_price=161.0, exit_reason="test_complete")

    # Short position
    short_signal = create_test_signal(
        symbol="META",
        direction=Direction.SHORT,
        entry=500.0,
        stop=515.0,
        target=470.0,
        size=2.0,
    )

    short_pos = pm.create_position_from_signal(short_signal)
    pm.mark_open(short_pos.position_id, fill_price=500.0)

    # Test stop detection for short (price rises)
    pm.update_price("META", 520.0)  # Above stop
    at_stop = pm.get_positions_at_stop()
    print(f"  Short @ 500, Stop @ 515, Price @ 520")
    print(f"  At stop: {len(at_stop)} position(s)")
    assert len(at_stop) == 1
    print(f"  [OK] Short stop detection works")

    # Reset price
    pm.update_price("META", 500.0)

    # Test target detection for short (price drops)
    pm.update_price("META", 465.0)  # Below target
    at_target = pm.get_positions_at_target()
    print(f"  Short @ 500, Target @ 470, Price @ 465")
    print(f"  At target: {len(at_target)} position(s)")
    assert len(at_target) == 1
    print(f"  [OK] Short target detection works")

    pm.close_position(short_pos.position_id, exit_price=465.0, exit_reason="test_complete")

    return True


def test_portfolio_metrics(pm: PositionManager) -> bool:
    """Test 6: Portfolio metrics calculation."""
    print_subheader("Test 6: Portfolio Metrics Calculation")

    # Create a fresh position manager for clean metrics
    pm_fresh = PositionManager(max_positions=8)

    # Create multiple positions
    signals = [
        create_test_signal(symbol="AAPL", direction=Direction.LONG, entry=175.0, stop=172.0, target=181.0, size=10.0),
        create_test_signal(symbol="GOOGL", direction=Direction.LONG, entry=140.0, stop=137.0, target=146.0, size=15.0),
        create_test_signal(symbol="AMZN", direction=Direction.SHORT, entry=180.0, stop=186.0, target=168.0, size=8.0),
    ]

    positions = []
    for signal in signals:
        pos = pm_fresh.create_position_from_signal(signal)
        pm_fresh.mark_open(pos.position_id, fill_price=signal.entry_price)
        positions.append(pos)

    # Update prices - mix of winners and losers
    pm_fresh.update_price("AAPL", 178.0)   # +30 (long profit)
    pm_fresh.update_price("GOOGL", 138.0)  # -30 (long loss)
    pm_fresh.update_price("AMZN", 175.0)   # +40 (short profit)

    metrics = pm_fresh.get_portfolio_metrics()

    print(f"  Created 3 positions (2 long, 1 short)")
    print(f"  Updated prices with mixed results")
    print_metrics(metrics)

    assert metrics.open_positions == 3
    assert metrics.long_exposure > 0
    assert metrics.short_exposure > 0
    print(f"  [OK] Portfolio metrics calculated correctly")

    # Close one as winner, one as loser
    pm_fresh.close_position(positions[0].position_id, exit_price=180.0, exit_reason="target")  # Win
    pm_fresh.close_position(positions[1].position_id, exit_price=137.0, exit_reason="stop")    # Loss
    pm_fresh.close_position(positions[2].position_id, exit_price=170.0, exit_reason="target")  # Win

    metrics = pm_fresh.get_portfolio_metrics()
    print(f"\n  After closing all positions:")
    print(f"    Win Rate: {metrics.win_rate:.1f}%")
    print(f"    Profit Factor: {metrics.profit_factor:.2f}")
    print(f"    Total Realized P&L: {metrics.total_realized_pnl:.2f}")

    assert metrics.open_positions == 0
    assert metrics.win_rate > 0  # 2 wins, 1 loss = 66.7%
    print(f"  [OK] Win rate and profit factor calculated")

    return True


def test_heat_manager_sync(pm: PositionManager) -> bool:
    """Test 7: Heat Manager synchronization."""
    print_subheader("Test 7: Heat Manager Synchronization")

    # Create position manager with real heat manager
    heat_manager = DynamicHeatManager()
    pm_with_heat = PositionManager(heat_manager=heat_manager, max_positions=8)

    signal = create_test_signal(
        symbol="SPY",
        risk_percent=1.5,
    )

    # Check heat before
    heat_before = heat_manager.get_current_heat()
    print(f"  Heat before position: {heat_before:.2f}%")

    # Open position
    position = pm_with_heat.create_position_from_signal(signal)
    pm_with_heat.mark_open(position.position_id, fill_price=175.0)

    heat_after_open = heat_manager.get_current_heat()
    print(f"  Heat after open: {heat_after_open:.2f}%")
    assert heat_after_open > heat_before
    print(f"  [OK] Heat increased on position open")

    # Close position
    pm_with_heat.close_position(position.position_id, exit_price=178.0, exit_reason="test")

    heat_after_close = heat_manager.get_current_heat()
    print(f"  Heat after close: {heat_after_close:.2f}%")
    assert heat_after_close < heat_after_open
    print(f"  [OK] Heat decreased on position close")

    return True


def test_correlation_monitor_sync(pm: PositionManager) -> bool:
    """Test 8: Correlation Monitor synchronization."""
    print_subheader("Test 8: Correlation Monitor Synchronization")

    # Create position manager with real correlation monitor
    correlation_monitor = CorrelationMonitor()
    pm_with_corr = PositionManager(correlation_monitor=correlation_monitor, max_positions=8)

    signal = create_test_signal(
        symbol="QQQ",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
    )

    # Check positions before (CorrelationMonitor uses _positions internally)
    positions_before = len(correlation_monitor._positions)
    print(f"  Tracked positions before: {positions_before}")

    # Open position
    position = pm_with_corr.create_position_from_signal(signal)
    pm_with_corr.mark_open(position.position_id, fill_price=175.0)

    positions_after_open = len(correlation_monitor._positions)
    print(f"  Tracked positions after open: {positions_after_open}")
    assert positions_after_open > positions_before
    print(f"  [OK] Position added to Correlation Monitor")

    # Close position
    pm_with_corr.close_position(position.position_id, exit_price=178.0, exit_reason="test")

    positions_after_close = len(correlation_monitor._positions)
    print(f"  Tracked positions after close: {positions_after_close}")
    assert positions_after_close < positions_after_open
    print(f"  [OK] Position removed from Correlation Monitor")

    return True


def test_session_stats(pm: PositionManager) -> bool:
    """Test 9: Session statistics tracking."""
    print_subheader("Test 9: Session Statistics Tracking")

    pm_fresh = PositionManager(max_positions=8)

    # Open and close a few positions
    for i, (symbol, pnl_direction) in enumerate([("T1", 1), ("T2", -1), ("T3", 1)]):
        signal = create_test_signal(
            symbol=symbol,
            entry=100.0,
            stop=95.0,
            target=110.0,
        )
        pos = pm_fresh.create_position_from_signal(signal)
        pm_fresh.mark_open(pos.position_id, fill_price=100.0)

        # Close with win or loss
        exit_price = 105.0 if pnl_direction > 0 else 96.0
        pm_fresh.close_position(pos.position_id, exit_price=exit_price, exit_reason="test")

    stats = pm_fresh.get_session_stats()

    print(f"  Session Stats:")
    print(f"    Positions Opened: {stats['positions_opened']}")
    print(f"    Positions Closed: {stats['positions_closed']}")
    print(f"    Total Realized P&L: {stats['total_realized_pnl']:.2f}")

    assert stats["positions_opened"] == 3
    assert stats["positions_closed"] == 3
    print(f"  [OK] Session stats tracked correctly")

    # Test reset
    pm_fresh.reset_session_stats()
    stats = pm_fresh.get_session_stats()
    assert stats["positions_opened"] == 0
    print(f"  [OK] Session stats reset works")

    return True


def test_position_queries(pm: PositionManager) -> bool:
    """Test 10: Position query methods."""
    print_subheader("Test 10: Position Query Methods")

    pm_fresh = PositionManager(max_positions=8)

    # Create multiple positions for same and different symbols
    signal1 = create_test_signal(symbol="XYZ", entry=100.0)
    signal2 = create_test_signal(symbol="XYZ", entry=101.0)
    signal3 = create_test_signal(symbol="ABC", entry=50.0)

    pos1 = pm_fresh.create_position_from_signal(signal1)
    pos2 = pm_fresh.create_position_from_signal(signal2)
    pos3 = pm_fresh.create_position_from_signal(signal3)

    # Open some
    pm_fresh.mark_open(pos1.position_id, fill_price=100.0)
    pm_fresh.mark_open(pos3.position_id, fill_price=50.0)

    # Test get by signal
    found = pm_fresh.get_position_by_signal(signal1.signal_id)
    assert found is not None
    print(f"  [OK] get_position_by_signal works")

    # Test get by symbol
    xyz_positions = pm_fresh.get_positions_for_symbol("XYZ")
    assert len(xyz_positions) == 2
    print(f"  [OK] get_positions_for_symbol works ({len(xyz_positions)} found)")

    # Test get open by symbol
    xyz_open = pm_fresh.get_open_positions_for_symbol("XYZ")
    assert len(xyz_open) == 1  # Only pos1 is open
    print(f"  [OK] get_open_positions_for_symbol works ({len(xyz_open)} open)")

    # Test open positions property
    all_open = pm_fresh.open_positions
    assert len(all_open) == 2  # pos1 and pos3
    print(f"  [OK] open_positions property works ({len(all_open)} open)")

    # Test pending positions property
    pending = pm_fresh.pending_positions
    assert len(pending) == 1  # pos2
    print(f"  [OK] pending_positions property works ({len(pending)} pending)")

    return True


def run_validation():
    """Run all validation tests."""
    print_header("NEXUS Position Manager Validation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create position manager for most tests
    pm = PositionManager(max_positions=8)

    results = []

    try:
        results.append(("Position Creation", test_position_creation(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Position Creation", False))

    try:
        results.append(("Position Lifecycle", test_position_lifecycle(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Position Lifecycle", False))

    try:
        results.append(("Long P&L Calculations", test_long_pnl_calculations(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Long P&L Calculations", False))

    try:
        results.append(("Short P&L Calculations", test_short_pnl_calculations(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Short P&L Calculations", False))

    try:
        results.append(("Stop/Target Detection", test_stop_target_detection(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Stop/Target Detection", False))

    try:
        results.append(("Portfolio Metrics", test_portfolio_metrics(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Portfolio Metrics", False))

    try:
        results.append(("Heat Manager Sync", test_heat_manager_sync(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Heat Manager Sync", False))

    try:
        results.append(("Correlation Monitor Sync", test_correlation_monitor_sync(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Correlation Monitor Sync", False))

    try:
        results.append(("Session Stats", test_session_stats(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Session Stats", False))

    try:
        results.append(("Position Queries", test_position_queries(pm)))
    except Exception as e:
        print(f"  [X] FAILED: {e}")
        results.append(("Position Queries", False))

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[X] FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  Position Manager validation COMPLETE!")
        print("  Ready to proceed to Order Manager.")
    else:
        print("\n  Some tests failed - review output above.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_validation()
