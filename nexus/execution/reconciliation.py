"""
Reconciliation Engine - The Site Inspector

Compares internal state against broker state to detect and handle
discrepancies. Runs periodically to ensure system integrity.

Think of this as the site inspector who walks the job at shift end,
comparing the work orders against what's actually been built,
flagging anything that doesn't match.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from nexus.core.enums import Market, Direction
from nexus.execution.position_manager import PositionManager, Position, PositionStatus
from nexus.execution.order_manager import OrderManager, Order, OrderStatus
from nexus.execution.trade_executor import (
    TradeExecutor,
    BrokerPosition,
    BrokerAccount,
)


logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of discrepancies that can be detected."""
    # Position discrepancies
    POSITION_MISSING_INTERNAL = "position_missing_internal"  # Broker has, we don't
    POSITION_MISSING_BROKER = "position_missing_broker"      # We have, broker doesn't
    POSITION_SIZE_MISMATCH = "position_size_mismatch"        # Different sizes
    POSITION_SIDE_MISMATCH = "position_side_mismatch"        # Long vs short

    # Order discrepancies
    ORDER_FILLED_NOT_RECORDED = "order_filled_not_recorded"  # Broker filled, we show pending
    ORDER_CANCELLED_NOT_RECORDED = "order_cancelled_not_recorded"  # Broker cancelled
    ORDER_MISSING_BROKER = "order_missing_broker"            # We have pending, broker doesn't

    # Account discrepancies
    EQUITY_MISMATCH = "equity_mismatch"                      # Significant equity difference
    BALANCE_MISMATCH = "balance_mismatch"                    # Balance doesn't match


class DiscrepancySeverity(Enum):
    """Severity levels for discrepancies."""
    INFO = "info"           # Minor, auto-sync OK
    WARNING = "warning"     # Needs attention but not critical
    ERROR = "error"         # Manual review recommended
    CRITICAL = "critical"   # Trading should halt


class ReconciliationAction(Enum):
    """Actions that can be taken for discrepancies."""
    AUTO_SYNC = "auto_sync"     # Automatically sync internal to match broker
    ALERT_ONLY = "alert_only"   # Just alert, don't change anything
    HALT_TRADING = "halt"       # Stop trading until resolved
    LOG_ONLY = "log_only"       # Just log for analysis
    MANUAL_REVIEW = "manual"    # Requires human intervention


@dataclass
class Discrepancy:
    """A single discrepancy found during reconciliation."""
    discrepancy_id: str
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    broker: str
    symbol: Optional[str]
    description: str
    internal_value: Any
    broker_value: Any
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    action_taken: Optional[ReconciliationAction] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "discrepancy_id": self.discrepancy_id,
            "type": self.discrepancy_type.value,
            "severity": self.severity.value,
            "broker": self.broker,
            "symbol": self.symbol,
            "description": self.description,
            "internal_value": str(self.internal_value),
            "broker_value": str(self.broker_value),
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "action_taken": self.action_taken.value if self.action_taken else None,
            "resolution_notes": self.resolution_notes,
        }

    def resolve(self, action: ReconciliationAction, notes: str = ""):
        """Mark discrepancy as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        self.action_taken = action
        self.resolution_notes = notes


@dataclass
class ReconciliationReport:
    """Report from a reconciliation run."""
    report_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    brokers_checked: List[str] = field(default_factory=list)
    discrepancies: List[Discrepancy] = field(default_factory=list)
    positions_internal: int = 0
    positions_broker: int = 0
    orders_checked: int = 0
    auto_synced: int = 0
    alerts_raised: int = 0

    @property
    def is_clean(self) -> bool:
        """True if no discrepancies found."""
        return len(self.discrepancies) == 0

    @property
    def has_critical(self) -> bool:
        """True if any critical discrepancies."""
        return any(d.severity == DiscrepancySeverity.CRITICAL for d in self.discrepancies)

    @property
    def has_errors(self) -> bool:
        """True if any error-level discrepancies."""
        return any(d.severity in [DiscrepancySeverity.ERROR, DiscrepancySeverity.CRITICAL]
                   for d in self.discrepancies)

    @property
    def unresolved_count(self) -> int:
        """Count of unresolved discrepancies."""
        return sum(1 for d in self.discrepancies if not d.resolved)

    @property
    def duration_seconds(self) -> float:
        """How long reconciliation took."""
        if not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    def add_discrepancy(self, discrepancy: Discrepancy):
        """Add a discrepancy to the report."""
        self.discrepancies.append(discrepancy)
        if discrepancy.severity in [DiscrepancySeverity.WARNING, DiscrepancySeverity.ERROR, DiscrepancySeverity.CRITICAL]:
            self.alerts_raised += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "brokers_checked": self.brokers_checked,
            "is_clean": self.is_clean,
            "has_critical": self.has_critical,
            "positions_internal": self.positions_internal,
            "positions_broker": self.positions_broker,
            "orders_checked": self.orders_checked,
            "discrepancies_count": len(self.discrepancies),
            "unresolved_count": self.unresolved_count,
            "auto_synced": self.auto_synced,
            "alerts_raised": self.alerts_raised,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.is_clean:
            return (f"✅ Reconciliation CLEAN - "
                    f"{self.positions_internal} internal positions, "
                    f"{self.positions_broker} broker positions")

        lines = [
            f"⚠️ Reconciliation found {len(self.discrepancies)} discrepancies:",
            f"   Internal positions: {self.positions_internal}",
            f"   Broker positions: {self.positions_broker}",
            f"   Orders checked: {self.orders_checked}",
            f"   Auto-synced: {self.auto_synced}",
            f"   Alerts raised: {self.alerts_raised}",
            f"   Unresolved: {self.unresolved_count}",
        ]

        if self.has_critical:
            lines.append("   🚨 CRITICAL issues - trading should HALT!")

        return "\n".join(lines)


class ReconciliationEngine:
    """
    Main reconciliation engine.

    Compares internal state against broker state and handles discrepancies.
    Should be run periodically (e.g., every 5 minutes) or after each trade.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        order_manager: OrderManager,
        trade_executor: TradeExecutor,
        equity_tolerance_pct: float = 1.0,      # 1% equity mismatch tolerance
        size_tolerance: float = 0.01,           # Size mismatch tolerance (for rounding)
        auto_sync_enabled: bool = True,         # Auto-sync minor discrepancies
    ):
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.trade_executor = trade_executor

        # Tolerances
        self.equity_tolerance_pct = equity_tolerance_pct
        self.size_tolerance = size_tolerance
        self.auto_sync_enabled = auto_sync_enabled

        # History
        self._reports: List[ReconciliationReport] = []
        self._unresolved: Dict[str, Discrepancy] = {}

        # Callbacks
        self._on_discrepancy: List[Callable[[Discrepancy], None]] = []
        self._on_critical: List[Callable[[Discrepancy], None]] = []

        # Stats
        self._stats = {
            "total_reconciliations": 0,
            "total_discrepancies": 0,
            "total_auto_synced": 0,
            "last_reconciliation": None,
            "last_clean": None,
        }

    async def reconcile(self, broker_names: Optional[List[str]] = None) -> ReconciliationReport:
        """
        Run full reconciliation against broker(s).

        Args:
            broker_names: Specific brokers to check (None = all registered)

        Returns:
            ReconciliationReport with findings
        """
        report = ReconciliationReport(
            report_id=str(uuid.uuid4()),
            started_at=datetime.utcnow(),
        )

        # Determine which brokers to check
        if broker_names is None:
            broker_names = list(self.trade_executor._brokers.keys())

        report.brokers_checked = broker_names

        for broker_name in broker_names:
            broker = self.trade_executor._brokers.get(broker_name)
            if not broker:
                logger.warning(f"Broker {broker_name} not found")
                continue

            if not broker.connected:
                logger.warning(f"Broker {broker_name} not connected, skipping")
                continue

            # Reconcile positions
            await self._reconcile_positions(broker_name, broker, report)

            # Reconcile account
            await self._reconcile_account(broker_name, broker, report)

        # Reconcile orders (internal state, can check without broker)
        self._reconcile_orders(report)

        # Complete report
        report.completed_at = datetime.utcnow()

        # Update stats
        self._stats["total_reconciliations"] += 1
        self._stats["total_discrepancies"] += len(report.discrepancies)
        self._stats["total_auto_synced"] += report.auto_synced
        self._stats["last_reconciliation"] = report.completed_at

        if report.is_clean:
            self._stats["last_clean"] = report.completed_at

        # Store report
        self._reports.append(report)
        if len(self._reports) > 100:  # Keep last 100 reports
            self._reports = self._reports[-100:]

        # Fire callbacks for discrepancies
        for discrepancy in report.discrepancies:
            self._fire_discrepancy_callbacks(discrepancy)

        logger.info(report.get_summary())

        return report

    async def _reconcile_positions(
        self,
        broker_name: str,
        broker,
        report: ReconciliationReport
    ):
        """Reconcile positions with a broker."""

        # Get broker positions
        try:
            broker_positions = await broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions from {broker_name}: {e}")
            return

        broker_position_map = {p.symbol: p for p in broker_positions}
        report.positions_broker += len(broker_positions)

        # Get our internal open positions
        internal_positions = self.position_manager.open_positions
        report.positions_internal += len(internal_positions)

        internal_symbols = set()

        # Check each internal position against broker
        for position in internal_positions:
            internal_symbols.add(position.symbol)
            broker_pos = broker_position_map.get(position.symbol)

            if broker_pos is None:
                # We have position, broker doesn't
                discrepancy = Discrepancy(
                    discrepancy_id=str(uuid.uuid4()),
                    discrepancy_type=DiscrepancyType.POSITION_MISSING_BROKER,
                    severity=DiscrepancySeverity.ERROR,
                    broker=broker_name,
                    symbol=position.symbol,
                    description=f"Position {position.symbol} exists internally but not at broker",
                    internal_value=f"{position.direction.value} {position.size}",
                    broker_value="None",
                )
                report.add_discrepancy(discrepancy)
                self._unresolved[discrepancy.discrepancy_id] = discrepancy
                continue

            # Check size match
            internal_size = position.size
            broker_size = broker_pos.size

            if abs(internal_size - broker_size) > self.size_tolerance:
                discrepancy = Discrepancy(
                    discrepancy_id=str(uuid.uuid4()),
                    discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
                    severity=DiscrepancySeverity.WARNING,
                    broker=broker_name,
                    symbol=position.symbol,
                    description=f"Size mismatch for {position.symbol}",
                    internal_value=internal_size,
                    broker_value=broker_size,
                )

                # Auto-sync if enabled
                if self.auto_sync_enabled:
                    self._sync_position_size(position, broker_size)
                    discrepancy.resolve(ReconciliationAction.AUTO_SYNC,
                                       f"Updated internal size from {internal_size} to {broker_size}")
                    report.auto_synced += 1

                report.add_discrepancy(discrepancy)
                if not discrepancy.resolved:
                    self._unresolved[discrepancy.discrepancy_id] = discrepancy

            # Check side match
            internal_side = "long" if position.direction == Direction.LONG else "short"
            broker_side = broker_pos.side.lower()

            if internal_side != broker_side:
                discrepancy = Discrepancy(
                    discrepancy_id=str(uuid.uuid4()),
                    discrepancy_type=DiscrepancyType.POSITION_SIDE_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    broker=broker_name,
                    symbol=position.symbol,
                    description=f"Side mismatch for {position.symbol} - CRITICAL",
                    internal_value=internal_side,
                    broker_value=broker_side,
                )
                report.add_discrepancy(discrepancy)
                self._unresolved[discrepancy.discrepancy_id] = discrepancy

        # Check for broker positions we don't have internally
        for symbol, broker_pos in broker_position_map.items():
            if symbol not in internal_symbols:
                discrepancy = Discrepancy(
                    discrepancy_id=str(uuid.uuid4()),
                    discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
                    severity=DiscrepancySeverity.WARNING,
                    broker=broker_name,
                    symbol=symbol,
                    description=f"Broker has position {symbol} not tracked internally",
                    internal_value="None",
                    broker_value=f"{broker_pos.side} {broker_pos.size}",
                )
                report.add_discrepancy(discrepancy)
                self._unresolved[discrepancy.discrepancy_id] = discrepancy

    async def _reconcile_account(
        self,
        broker_name: str,
        broker,
        report: ReconciliationReport
    ):
        """Reconcile account equity."""

        try:
            broker_account = await broker.get_account()
        except Exception as e:
            logger.error(f"Failed to get account from {broker_name}: {e}")
            return

        # Get our calculated equity (PortfolioMetrics may have current_equity if set by caller)
        metrics = self.position_manager.get_portfolio_metrics()
        internal_equity = getattr(metrics, "current_equity", 0)
        broker_equity = broker_account.equity

        if internal_equity == 0:
            # No internal tracking yet, skip
            return

        # Check for significant mismatch
        diff_pct = abs(internal_equity - broker_equity) / internal_equity * 100

        if diff_pct > self.equity_tolerance_pct:
            severity = DiscrepancySeverity.WARNING
            if diff_pct > 5.0:
                severity = DiscrepancySeverity.ERROR
            if diff_pct > 10.0:
                severity = DiscrepancySeverity.CRITICAL

            discrepancy = Discrepancy(
                discrepancy_id=str(uuid.uuid4()),
                discrepancy_type=DiscrepancyType.EQUITY_MISMATCH,
                severity=severity,
                broker=broker_name,
                symbol=None,
                description=f"Equity mismatch: {diff_pct:.1f}% difference",
                internal_value=f"£{internal_equity:.2f}",
                broker_value=f"£{broker_equity:.2f}",
            )
            report.add_discrepancy(discrepancy)
            self._unresolved[discrepancy.discrepancy_id] = discrepancy

    def _reconcile_orders(self, report: ReconciliationReport):
        """Reconcile internal order state."""

        # Check for orders stuck in pending/submitted state too long
        stale_threshold = timedelta(minutes=30)
        now = datetime.utcnow()

        for order in self.order_manager.submitted_orders:
            report.orders_checked += 1

            if order.submitted_at and (now - order.submitted_at) > stale_threshold:
                discrepancy = Discrepancy(
                    discrepancy_id=str(uuid.uuid4()),
                    discrepancy_type=DiscrepancyType.ORDER_FILLED_NOT_RECORDED,
                    severity=DiscrepancySeverity.WARNING,
                    broker=order.broker or "unknown",
                    symbol=order.symbol,
                    description=f"Order {order.order_id[:8]} submitted >30min ago, status unknown",
                    internal_value=f"Status: {order.status.value}",
                    broker_value="Unknown - check broker",
                )
                report.add_discrepancy(discrepancy)
                self._unresolved[discrepancy.discrepancy_id] = discrepancy

        for order in self.order_manager.pending_orders:
            report.orders_checked += 1

    def _sync_position_size(self, position: Position, broker_size: float):
        """Sync internal position size to match broker."""
        logger.info(f"Auto-syncing {position.symbol} size: {position.size} -> {broker_size}")
        position.size = broker_size

    def _fire_discrepancy_callbacks(self, discrepancy: Discrepancy):
        """Fire callbacks for a discrepancy."""
        for callback in self._on_discrepancy:
            try:
                callback(discrepancy)
            except Exception as e:
                logger.error(f"Discrepancy callback error: {e}")

        if discrepancy.severity == DiscrepancySeverity.CRITICAL:
            for callback in self._on_critical:
                try:
                    callback(discrepancy)
                except Exception as e:
                    logger.error(f"Critical callback error: {e}")

    def on_discrepancy(self, callback: Callable[[Discrepancy], None]):
        """Register callback for any discrepancy."""
        self._on_discrepancy.append(callback)

    def on_critical(self, callback: Callable[[Discrepancy], None]):
        """Register callback for critical discrepancies."""
        self._on_critical.append(callback)

    def resolve_discrepancy(
        self,
        discrepancy_id: str,
        action: ReconciliationAction,
        notes: str = ""
    ) -> bool:
        """Manually resolve a discrepancy."""
        if discrepancy_id not in self._unresolved:
            return False

        discrepancy = self._unresolved[discrepancy_id]
        discrepancy.resolve(action, notes)
        del self._unresolved[discrepancy_id]

        logger.info(f"Resolved discrepancy {discrepancy_id}: {action.value} - {notes}")
        return True

    def get_unresolved(self) -> List[Discrepancy]:
        """Get all unresolved discrepancies."""
        return list(self._unresolved.values())

    def get_unresolved_by_severity(self, severity: DiscrepancySeverity) -> List[Discrepancy]:
        """Get unresolved discrepancies of a specific severity."""
        return [d for d in self._unresolved.values() if d.severity == severity]

    def get_stats(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        return {
            **self._stats,
            "unresolved_count": len(self._unresolved),
            "unresolved_critical": len(self.get_unresolved_by_severity(DiscrepancySeverity.CRITICAL)),
            "unresolved_errors": len(self.get_unresolved_by_severity(DiscrepancySeverity.ERROR)),
            "reports_stored": len(self._reports),
        }

    def get_last_report(self) -> Optional[ReconciliationReport]:
        """Get the most recent reconciliation report."""
        if not self._reports:
            return None
        return self._reports[-1]

    def clear_history(self):
        """Clear report history (but keep unresolved discrepancies)."""
        self._reports = []
