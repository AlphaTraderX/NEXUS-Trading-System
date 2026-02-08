"""
Position Manager - The Job Site Foreman

Tracks all open positions and their lifecycle:
- Add new positions (new jobs starting)
- Update prices and P&L (progress updates)
- Close positions (job completion)
- Portfolio metrics (overall site status)
- Sync with Heat Manager and Correlation Monitor

Think of this as the foreman who knows exactly what's happening
on every job site at any given moment.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from nexus.core.enums import Market, Direction, EdgeType, SignalStatus
from nexus.core.models import NexusSignal, TrackedPosition
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.correlation import CorrelationMonitor
from nexus.storage.service import get_storage_service


logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position lifecycle status."""
    PENDING = "pending"          # Signal generated, order not yet placed
    SUBMITTED = "submitted"      # Order submitted to broker
    OPEN = "open"               # Position is active
    CLOSING = "closing"         # Close order submitted
    CLOSED = "closed"           # Position closed
    CANCELLED = "cancelled"     # Order was cancelled before fill


@dataclass
class Position:
    """
    Represents an active or historical position.

    This is the internal tracking object - richer than what the broker sees.
    """
    # Identifiers
    position_id: str
    signal_id: str
    symbol: str
    market: Market

    # Direction and size
    direction: Direction
    size: float                  # Number of units (shares, lots, contracts)

    # Prices
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float

    # Risk tracking
    risk_amount: float           # Amount risked in base currency
    risk_percent: float         # Percentage of account risked

    # Edge info
    primary_edge: EdgeType
    edge_score: int

    # Status and timing
    status: PositionStatus = PositionStatus.PENDING
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # P&L tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0

    # Exit info
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Costs
    entry_costs: float = 0.0
    exit_costs: float = 0.0
    total_costs: float = 0.0

    # Metadata
    sector: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        """Initialize calculated fields."""
        self._update_pnl()

    def _update_pnl(self):
        """Update P&L based on current price."""
        if self.status == PositionStatus.OPEN and self.entry_price > 0:
            if self.direction == Direction.LONG:
                price_change = self.current_price - self.entry_price
            else:
                price_change = self.entry_price - self.current_price

            self.unrealized_pnl = price_change * self.size
            self.unrealized_pnl_pct = (price_change / self.entry_price) * 100

    def update_price(self, new_price: float):
        """Update current price and recalculate P&L."""
        self.current_price = new_price
        self._update_pnl()

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0

    @property
    def is_at_stop(self) -> bool:
        """Check if current price has hit stop loss."""
        if self.direction == Direction.LONG:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    @property
    def is_at_target(self) -> bool:
        """Check if current price has hit take profit."""
        if self.direction == Direction.LONG:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    @property
    def r_multiple(self) -> float:
        """Calculate R-multiple (profit in terms of initial risk)."""
        if self.risk_amount <= 0:
            return 0.0

        pnl = self.realized_pnl if self.status == PositionStatus.CLOSED else self.unrealized_pnl
        return pnl / self.risk_amount

    @property
    def position_value(self) -> float:
        """Current position value."""
        return self.size * self.current_price

    @property
    def hold_time_hours(self) -> float:
        """How long the position has been/was open."""
        if self.opened_at is None:
            return 0.0

        end_time = self.closed_at if self.closed_at else datetime.utcnow()
        delta = end_time - self.opened_at
        return delta.total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position_id": self.position_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "market": self.market.value if isinstance(self.market, Market) else self.market,
            "direction": self.direction.value if isinstance(self.direction, Direction) else self.direction,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "primary_edge": self.primary_edge.value if isinstance(self.primary_edge, EdgeType) else self.primary_edge,
            "edge_score": self.edge_score,
            "status": self.status.value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_pct": self.realized_pnl_pct,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "r_multiple": self.r_multiple,
            "hold_time_hours": self.hold_time_hours,
        }


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio metrics."""
    # Position counts
    total_positions: int = 0
    open_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0

    # P&L
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Risk
    total_risk_amount: float = 0.0
    portfolio_heat: float = 0.0

    # Performance
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_r_multiple: float = 0.0

    # Exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0

    # By market
    exposure_by_market: Dict[str, float] = field(default_factory=dict)
    positions_by_market: Dict[str, int] = field(default_factory=dict)


class PositionManager:
    """
    Manages all positions and their lifecycle.

    Responsibilities:
    - Track open and closed positions
    - Update prices and P&L
    - Sync with Heat Manager for risk tracking
    - Sync with Correlation Monitor for concentration limits
    - Provide portfolio metrics
    """

    def __init__(
        self,
        heat_manager: Optional[DynamicHeatManager] = None,
        correlation_monitor: Optional[CorrelationMonitor] = None,
        max_positions: int = 8,
    ):
        """
        Initialize the Position Manager.

        Args:
            heat_manager: For risk tracking sync
            correlation_monitor: For concentration tracking
            max_positions: Maximum allowed open positions
        """
        self.heat_manager = heat_manager
        self.correlation_monitor = correlation_monitor
        self.max_positions = max_positions

        # Position storage
        self._positions: Dict[str, Position] = {}  # position_id -> Position
        self._positions_by_signal: Dict[str, str] = {}  # signal_id -> position_id
        self._positions_by_symbol: Dict[str, List[str]] = {}  # symbol -> [position_ids]

        # Historical tracking
        self._closed_positions: List[Position] = []

        # Session tracking
        self._session_stats = {
            "positions_opened": 0,
            "positions_closed": 0,
            "total_realized_pnl": 0.0,
        }

    @property
    def open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]

    @property
    def pending_positions(self) -> List[Position]:
        """Get all pending positions (order not yet filled)."""
        return [p for p in self._positions.values() if p.status in (PositionStatus.PENDING, PositionStatus.SUBMITTED)]

    @property
    def all_active_positions(self) -> List[Position]:
        """Get all active positions (open + pending)."""
        return [p for p in self._positions.values() if p.status not in (PositionStatus.CLOSED, PositionStatus.CANCELLED)]

    def can_open_position(self) -> Dict[str, Any]:
        """Check if we can open a new position."""
        open_count = len(self.open_positions)
        pending_count = len(self.pending_positions)
        total_active = open_count + pending_count

        if total_active >= self.max_positions:
            return {
                "allowed": False,
                "reason": f"Max positions ({self.max_positions}) reached",
                "open": open_count,
                "pending": pending_count,
            }

        return {
            "allowed": True,
            "open": open_count,
            "pending": pending_count,
            "remaining": self.max_positions - total_active,
        }

    def create_position_from_signal(self, signal: NexusSignal) -> Position:
        """
        Create a new position from a signal.

        The position starts in PENDING status until the order is filled.
        """
        # Check if we can open
        can_open = self.can_open_position()
        if not can_open["allowed"]:
            raise ValueError(f"Cannot open position: {can_open['reason']}")

        # Check if position already exists for this signal
        if signal.signal_id in self._positions_by_signal:
            existing_id = self._positions_by_signal[signal.signal_id]
            raise ValueError(f"Position already exists for signal {signal.signal_id}: {existing_id}")

        # Normalize enums (handle both enum and string values)
        market = signal.market if isinstance(signal.market, Market) else Market(signal.market)
        direction = signal.direction if isinstance(signal.direction, Direction) else Direction(signal.direction)
        primary_edge = signal.primary_edge if isinstance(signal.primary_edge, EdgeType) else EdgeType(signal.primary_edge)

        entry_costs = 0.0
        if signal.costs is not None:
            entry_costs = getattr(signal.costs, "total", 0.0) if hasattr(signal.costs, "total") else (signal.costs.get("total", 0.0) if isinstance(signal.costs, dict) else 0.0)

        position = Position(
            position_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            market=market,
            direction=direction,
            size=signal.position_size,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=signal.risk_amount,
            risk_percent=signal.risk_percent,
            primary_edge=primary_edge,
            edge_score=signal.edge_score,
            status=PositionStatus.PENDING,
            entry_costs=entry_costs,
        )

        # Store position
        self._positions[position.position_id] = position
        self._positions_by_signal[signal.signal_id] = position.position_id

        if signal.symbol not in self._positions_by_symbol:
            self._positions_by_symbol[signal.symbol] = []
        self._positions_by_symbol[signal.symbol].append(position.position_id)

        logger.info(
            f"Position created: {position.position_id[:8]} for {signal.symbol} "
            f"({direction.value}) - Status: PENDING"
        )

        return position

    def mark_submitted(self, position_id: str) -> Position:
        """Mark a position as submitted (order sent to broker)."""
        position = self._get_position(position_id)

        if position.status != PositionStatus.PENDING:
            raise ValueError(f"Position {position_id} is not pending (status: {position.status})")

        position.status = PositionStatus.SUBMITTED
        logger.info(f"Position {position_id[:8]} marked as SUBMITTED")

        return position

    def mark_open(
        self,
        position_id: str,
        fill_price: float,
        fill_time: Optional[datetime] = None,
        actual_size: Optional[float] = None,
    ) -> Position:
        """
        Mark a position as open (order filled).

        This also syncs with Heat Manager and Correlation Monitor.
        """
        position = self._get_position(position_id)

        if position.status not in (PositionStatus.PENDING, PositionStatus.SUBMITTED):
            raise ValueError(f"Position {position_id} cannot be opened (status: {position.status})")

        # Update position
        position.status = PositionStatus.OPEN
        position.entry_price = fill_price
        position.current_price = fill_price
        position.opened_at = fill_time or datetime.utcnow()

        if actual_size is not None:
            position.size = actual_size

        # Sync with Heat Manager (TrackedPosition from core.models has full fields)
        if self.heat_manager:
            market = position.market if isinstance(position.market, Market) else Market(position.market)
            direction = position.direction if isinstance(position.direction, Direction) else Direction(position.direction)
            tracked = TrackedPosition(
                position_id=position.position_id,
                symbol=position.symbol,
                market=market,
                direction=direction,
                entry_price=position.entry_price,
                current_price=position.current_price,
                stop_loss=position.stop_loss,
                position_size=position.size,
                risk_amount=position.risk_amount,
                risk_pct=position.risk_percent,
                sector=position.sector,
                opened_at=position.opened_at,
            )
            self.heat_manager.add_position(tracked)

        # Sync with Correlation Monitor
        if self.correlation_monitor:
            market = position.market if isinstance(position.market, Market) else Market(position.market)
            direction = position.direction if isinstance(position.direction, Direction) else Direction(position.direction)
            self.correlation_monitor.add_position(
                position_id=position.position_id,
                symbol=position.symbol,
                market=market,
                direction=direction,
                risk_pct=position.risk_percent,
                sector=position.sector,
            )

        self._session_stats["positions_opened"] += 1

        logger.info(
            f"Position {position_id[:8]} OPENED: {position.symbol} @ {fill_price} "
            f"(size: {position.size})"
        )

        self._schedule_sync_to_database()
        return position

    def update_price(self, symbol: str, new_price: float) -> List[Position]:
        """
        Update price for all positions in a symbol.

        Returns list of updated positions.
        """
        updated = []

        position_ids = self._positions_by_symbol.get(symbol, [])
        for pos_id in position_ids:
            position = self._positions.get(pos_id)
            if position and position.status == PositionStatus.OPEN:
                position.update_price(new_price)
                updated.append(position)
        if updated:
            self._schedule_sync_to_database()
        return updated

    def update_all_prices(self, prices: Dict[str, float]) -> List[Position]:
        """
        Update prices for multiple symbols at once.

        Args:
            prices: Dict mapping symbol -> current price

        Returns:
            List of all updated positions
        """
        all_updated = []
        for symbol, price in prices.items():
            updated = self.update_price(symbol, price)
            all_updated.extend(updated)
        return all_updated

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[datetime] = None,
        exit_costs: float = 0.0,
    ) -> Position:
        """
        Close a position.

        This calculates final P&L and syncs with Heat Manager and Correlation Monitor.
        """
        position = self._get_position(position_id)

        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Position {position_id} is not open (status: {position.status})")

        # Calculate realized P&L
        if position.direction == Direction.LONG:
            price_change = exit_price - position.entry_price
        else:
            price_change = position.entry_price - exit_price

        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.closed_at = exit_time or datetime.utcnow()
        position.status = PositionStatus.CLOSED

        # Calculate realized P&L
        gross_pnl = price_change * position.size
        position.exit_costs = exit_costs
        position.total_costs = position.entry_costs + exit_costs
        position.realized_pnl = gross_pnl - position.total_costs
        position.realized_pnl_pct = (position.realized_pnl / (position.entry_price * position.size)) * 100

        # Clear unrealized
        position.unrealized_pnl = 0.0
        position.unrealized_pnl_pct = 0.0

        # Remove from Heat Manager
        if self.heat_manager:
            self.heat_manager.remove_position(position_id)

        # Remove from Correlation Monitor
        if self.correlation_monitor:
            self.correlation_monitor.remove_position(position_id)

        # Move to closed positions
        self._closed_positions.append(position)

        # Update session stats
        self._session_stats["positions_closed"] += 1
        self._session_stats["total_realized_pnl"] += position.realized_pnl

        logger.info(
            f"Position {position_id[:8]} CLOSED: {position.symbol} @ {exit_price} "
            f"(P&L: {position.realized_pnl:.2f}, R: {position.r_multiple:.2f}R, "
            f"Reason: {exit_reason})"
        )

        self._schedule_sync_to_database()
        return position

    def cancel_position(self, position_id: str, reason: str = "cancelled") -> Position:
        """Cancel a pending/submitted position that was never filled."""
        position = self._get_position(position_id)

        if position.status not in (PositionStatus.PENDING, PositionStatus.SUBMITTED):
            raise ValueError(f"Position {position_id} cannot be cancelled (status: {position.status})")

        position.status = PositionStatus.CANCELLED
        position.exit_reason = reason
        position.closed_at = datetime.utcnow()

        # Move to closed positions
        self._closed_positions.append(position)

        logger.info(f"Position {position_id[:8]} CANCELLED: {reason}")

        return position

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self._positions.get(position_id)

    def get_position_by_signal(self, signal_id: str) -> Optional[Position]:
        """Get a position by its signal ID."""
        position_id = self._positions_by_signal.get(signal_id)
        if position_id:
            return self._positions.get(position_id)
        return None

    def get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol."""
        position_ids = self._positions_by_symbol.get(symbol, [])
        return [self._positions[pid] for pid in position_ids if pid in self._positions]

    def get_open_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Get open positions for a symbol."""
        return [p for p in self.get_positions_for_symbol(symbol) if p.status == PositionStatus.OPEN]

    def get_positions_at_stop(self) -> List[Position]:
        """Get positions that have hit their stop loss."""
        return [p for p in self.open_positions if p.is_at_stop]

    def get_positions_at_target(self) -> List[Position]:
        """Get positions that have hit their take profit."""
        return [p for p in self.open_positions if p.is_at_target]

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio metrics."""
        metrics = PortfolioMetrics()

        open_positions = self.open_positions
        all_closed = self._closed_positions

        # Position counts
        metrics.total_positions = len(self._positions)
        metrics.open_positions = len(open_positions)

        # Calculate from open positions
        for pos in open_positions:
            metrics.total_unrealized_pnl += pos.unrealized_pnl
            metrics.total_risk_amount += pos.risk_amount

            if pos.is_profitable:
                metrics.winning_positions += 1
            else:
                metrics.losing_positions += 1

            # Exposure tracking
            exposure = pos.position_value
            if pos.direction == Direction.LONG:
                metrics.long_exposure += exposure
            else:
                metrics.short_exposure += exposure

            # By market
            market_key = pos.market.value if isinstance(pos.market, Market) else pos.market
            metrics.exposure_by_market[market_key] = metrics.exposure_by_market.get(market_key, 0) + exposure
            metrics.positions_by_market[market_key] = metrics.positions_by_market.get(market_key, 0) + 1

        # Calculate from closed positions
        wins = []
        losses = []
        r_multiples = []

        for pos in all_closed:
            if pos.status == PositionStatus.CLOSED:
                metrics.total_realized_pnl += pos.realized_pnl
                r_multiples.append(pos.r_multiple)

                if pos.realized_pnl > 0:
                    wins.append(pos.realized_pnl)
                elif pos.realized_pnl < 0:
                    losses.append(abs(pos.realized_pnl))

        # Total P&L
        metrics.total_pnl = metrics.total_unrealized_pnl + metrics.total_realized_pnl

        # Net exposure
        metrics.net_exposure = metrics.long_exposure - metrics.short_exposure

        # Performance metrics (from closed positions)
        total_closed = len([p for p in all_closed if p.status == PositionStatus.CLOSED])
        if total_closed > 0:
            metrics.win_rate = len(wins) / total_closed * 100

        if wins:
            metrics.avg_win = sum(wins) / len(wins)

        if losses:
            metrics.avg_loss = sum(losses) / len(losses)

        if losses and sum(losses) > 0:
            metrics.profit_factor = sum(wins) / sum(losses) if wins else 0

        if r_multiples:
            metrics.avg_r_multiple = sum(r_multiples) / len(r_multiples)

        # Portfolio heat from heat manager if available
        if self.heat_manager:
            heat_status = self.heat_manager.get_heat_status()
            metrics.portfolio_heat = heat_status.current_heat

        return metrics

    def get_session_stats(self) -> Dict[str, Any]:
        """Get stats for the current trading session."""
        return {
            **self._session_stats,
            "open_positions": len(self.open_positions),
            "pending_positions": len(self.pending_positions),
        }

    def reset_session_stats(self):
        """Reset session statistics (call at start of new trading day)."""
        self._session_stats = {
            "positions_opened": 0,
            "positions_closed": 0,
            "total_realized_pnl": 0.0,
        }

    def _get_position(self, position_id: str) -> Position:
        """Get a position or raise error."""
        position = self._positions.get(position_id)
        if not position:
            raise ValueError(f"Position not found: {position_id}")
        return position

    async def sync_to_database(self) -> None:
        """Sync current positions to database."""
        try:
            storage = get_storage_service()
            if not storage._initialized:
                return
            positions_dict = {
                pos_id: pos.to_dict() if hasattr(pos, "to_dict") else str(pos)
                for pos_id, pos in self._positions.items()
            }
            metrics = self.get_portfolio_metrics()
            heat = getattr(metrics, "portfolio_heat", 0.0)
            await storage.update_positions(positions_dict, heat)
        except Exception as e:
            logger.warning("Failed to sync positions to database: %s", e)

    def _schedule_sync_to_database(self) -> None:
        """Schedule async sync_to_database from sync context (fire-and-forget)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.sync_to_database())
        except RuntimeError:
            pass

    def get_positions_summary(self) -> str:
        """Get a human-readable summary of all positions."""
        open_pos = self.open_positions
        if not open_pos:
            return "No open positions"

        lines = [f"Open Positions ({len(open_pos)}):"]
        for pos in open_pos:
            direction = "LONG" if pos.direction == Direction.LONG else "SHORT"
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            lines.append(
                f"  {pos.symbol} {direction} @ {pos.entry_price:.2f} -> {pos.current_price:.2f} "
                f"({pnl_sign}{pos.unrealized_pnl:.2f} / {pos.r_multiple:.2f}R)"
            )

        return "\n".join(lines)
