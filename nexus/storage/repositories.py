"""
NEXUS Data Repositories

Data access layer for database operations.
Provides clean interface for storing and retrieving data.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, update, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    SignalRecord,
    TradeRecord,
    DailyPerformance,
    EdgePerformance,
    SystemState,
    AuditLog,
    AlertLog,
)
from nexus.core.models import NexusSignal, TradeResult

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNAL REPOSITORY
# =============================================================================

class SignalRepository:
    """Repository for signal operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, signal: NexusSignal) -> SignalRecord:
        """Save a new signal to database."""
        record = SignalRecord(
            signal_id=signal.signal_id,
            opportunity_id=signal.opportunity_id,
            symbol=signal.symbol,
            market=signal.market.value if hasattr(signal.market, 'value') else signal.market,
            direction=signal.direction.value if hasattr(signal.direction, 'value') else signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=signal.position_size,
            position_value=signal.position_value,
            risk_amount=signal.risk_amount,
            risk_percent=signal.risk_percent,
            primary_edge=(signal.primary_edge.value if hasattr(signal.primary_edge, "value") else str(signal.primary_edge))[:30],
            secondary_edges=[e.value if hasattr(e, 'value') else e for e in signal.secondary_edges],
            edge_score=signal.edge_score,
            tier=signal.tier.value if hasattr(signal.tier, 'value') else signal.tier,
            gross_expected=signal.gross_expected,
            net_expected=signal.net_expected,
            cost_ratio=signal.cost_ratio,
            costs=signal.costs.to_dict() if hasattr(signal.costs, 'to_dict') else signal.costs,
            ai_reasoning=signal.ai_reasoning,
            confluence_factors=signal.confluence_factors,
            risk_factors=signal.risk_factors,
            market_context=getattr(signal, "market_context", None) or None,
            session=getattr(signal, "session", None) or None,
            status=signal.status.value if hasattr(signal.status, "value") else signal.status,
            valid_until=signal.valid_until,
        )

        self.session.add(record)
        await self.session.flush()
        logger.info(f"Saved signal {signal.signal_id}")
        return record

    async def get_by_id(self, signal_id: str) -> Optional[SignalRecord]:
        """Get signal by ID."""
        result = await self.session.execute(
            select(SignalRecord).where(SignalRecord.signal_id == signal_id)
        )
        return result.scalar_one_or_none()

    async def get_recent(self, limit: int = 20) -> List[SignalRecord]:
        """Get most recent signals."""
        result = await self.session.execute(
            select(SignalRecord)
            .order_by(desc(SignalRecord.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_symbol(self, symbol: str, days: int = 7) -> List[SignalRecord]:
        """Get signals for a symbol within time range."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(SignalRecord)
            .where(
                and_(
                    SignalRecord.symbol == symbol,
                    SignalRecord.created_at >= cutoff
                )
            )
            .order_by(desc(SignalRecord.created_at))
        )
        return list(result.scalars().all())

    async def get_by_edge(self, edge_type: str, days: int = 30) -> List[SignalRecord]:
        """Get signals by edge type."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(SignalRecord)
            .where(
                and_(
                    SignalRecord.primary_edge == edge_type,
                    SignalRecord.created_at >= cutoff
                )
            )
            .order_by(desc(SignalRecord.created_at))
        )
        return list(result.scalars().all())

    async def update_status(self, signal_id: str, status: str) -> bool:
        """Update signal status."""
        result = await self.session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == signal_id)
            .values(status=status, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0

    async def get_pending(self) -> List[SignalRecord]:
        """Get all pending signals."""
        result = await self.session.execute(
            select(SignalRecord)
            .where(SignalRecord.status == "PENDING")
            .order_by(SignalRecord.created_at)
        )
        return list(result.scalars().all())


# =============================================================================
# TRADE REPOSITORY
# =============================================================================

class TradeRepository:
    """Repository for trade operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, trade: TradeResult, signal_id: str) -> TradeRecord:
        """Save a completed trade (full TradeResult from core.models)."""
        record = TradeRecord(
            signal_id=signal_id,
            symbol=trade.symbol,
            market=trade.market.value if hasattr(trade.market, "value") else trade.market,
            direction=trade.direction.value if hasattr(trade.direction, "value") else trade.direction,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            exit_price=getattr(trade, "exit_price", None),
            exit_time=getattr(trade, "exit_time", None),
            exit_reason=getattr(trade, "exit_reason", None),
            position_size=getattr(trade, "position_size", 0.0) or 0.0,
            pnl=getattr(trade, "pnl", None),
            pnl_percent=getattr(trade, "pnl_pct", None) or getattr(trade, "pnl_percent", None),
            slippage_entry=getattr(trade, "slippage_entry", None),
            slippage_exit=getattr(trade, "slippage_exit", None),
            costs_actual=(
                trade.actual_costs.to_dict()
                if hasattr(getattr(trade, "actual_costs", None), "to_dict")
                else getattr(trade, "actual_costs", None)
            ),
        )
        self.session.add(record)
        await self.session.flush()
        logger.info(f"Saved trade for signal {signal_id}")
        return record

    async def save_entry(self, data: dict, signal_id: str) -> TradeRecord:
        """Save trade entry only (e.g. when order is filled; exit data added when closed)."""
        record = TradeRecord(
            signal_id=signal_id,
            symbol=data.get("symbol", ""),
            market=data.get("market", ""),
            direction=data.get("direction", ""),
            entry_price=float(data.get("entry_price", 0)),
            entry_time=data.get("entry_time", datetime.utcnow()),
            exit_price=None,
            exit_time=None,
            exit_reason=None,
            position_size=float(data.get("position_size", 0)),
            pnl=None,
            pnl_percent=None,
            slippage_entry=data.get("slippage_entry"),
            slippage_exit=None,
            costs_actual=data.get("costs_actual"),
        )
        self.session.add(record)
        await self.session.flush()
        logger.info(f"Saved trade entry for signal {signal_id}")
        return record

    async def get_recent(self, limit: int = 50) -> List[TradeRecord]:
        """Get most recent trades."""
        result = await self.session.execute(
            select(TradeRecord)
            .order_by(desc(TradeRecord.entry_time))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_edge(self, edge_type: str, days: int = 30) -> List[TradeRecord]:
        """Get trades by edge type (via signal's primary_edge)."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(TradeRecord)
            .join(SignalRecord, TradeRecord.signal_id == SignalRecord.signal_id)
            .where(
                and_(
                    SignalRecord.primary_edge == edge_type,
                    TradeRecord.entry_time >= cutoff
                )
            )
            .order_by(desc(TradeRecord.entry_time))
        )
        return list(result.scalars().all())

    async def get_winners(self, days: int = 30) -> List[TradeRecord]:
        """Get winning trades."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(TradeRecord)
            .where(
                and_(
                    TradeRecord.pnl > 0,
                    TradeRecord.entry_time >= cutoff
                )
            )
            .order_by(desc(TradeRecord.pnl))
        )
        return list(result.scalars().all())


# =============================================================================
# SYSTEM STATE REPOSITORY
# =============================================================================

class SystemStateRepository:
    """Repository for system state operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self) -> Optional[SystemState]:
        """Get current system state (singleton)."""
        result = await self.session.execute(
            select(SystemState).where(SystemState.id == 1)
        )
        return result.scalar_one_or_none()

    async def initialize(self, starting_equity: float) -> SystemState:
        """Initialize system state (first run)."""
        state = SystemState(
            id=1,
            starting_equity=starting_equity,
            current_equity=starting_equity,
            peak_equity=starting_equity,
        )
        self.session.add(state)
        await self.session.flush()
        return state

    async def update_equity(
        self,
        current_equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
    ) -> None:
        """Update equity and P&L."""
        await self.session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                current_equity=current_equity,
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_pct,
                peak_equity=current_equity,
                updated_at=datetime.utcnow(),
            )
        )

    async def update_positions(
        self,
        positions: list,
        heat: float,
    ) -> None:
        """Update open positions and heat."""
        await self.session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                open_positions=positions,
                portfolio_heat=heat,
                portfolio_heat_percent=heat,
                updated_at=datetime.utcnow(),
            )
        )

    async def set_circuit_breaker(
        self,
        status: str,
        reason: str = "",
    ) -> None:
        """Set circuit breaker status."""
        await self.session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                circuit_breaker_status=status,
                updated_at=datetime.utcnow(),
            )
        )

    async def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch."""
        await self.session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                kill_switch_active=True,
                updated_at=datetime.utcnow(),
            )
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def save_signal(session: AsyncSession, signal: NexusSignal) -> SignalRecord:
    """Convenience function to save a signal."""
    repo = SignalRepository(session)
    return await repo.save(signal)


async def get_recent_signals(session: AsyncSession, limit: int = 20) -> List[SignalRecord]:
    """Convenience function to get recent signals."""
    repo = SignalRepository(session)
    return await repo.get_recent(limit)


async def save_trade(session: AsyncSession, trade: TradeResult, signal_id: str) -> TradeRecord:
    """Convenience function to save a trade."""
    repo = TradeRepository(session)
    return await repo.save(trade, signal_id)


async def get_system_state(session: AsyncSession) -> Optional[SystemState]:
    """Convenience function to get system state."""
    repo = SystemStateRepository(session)
    return await repo.get()


# =============================================================================
# DAILY PERFORMANCE REPOSITORY
# =============================================================================


class DailyPerformanceRepository:
    """Repository for daily performance metrics."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, date: datetime, metrics: dict) -> DailyPerformance:
        """Save or update daily performance record."""
        from sqlalchemy import select
        record = DailyPerformance(
            date=date.date() if hasattr(date, "date") else date,
            starting_equity=metrics.get("starting_equity", 0),
            ending_equity=metrics.get("ending_equity", 0),
            pnl=metrics.get("pnl", 0),
            pnl_percent=metrics.get("pnl_pct", 0) or metrics.get("pnl_percent", 0),
            trades_taken=metrics.get("trades_taken", 0),
            winners=metrics.get("winners", 0),
            losers=metrics.get("losers", 0),
            win_rate=metrics.get("win_rate", 0),
            largest_win=metrics.get("largest_win", 0),
            largest_loss=metrics.get("largest_loss", 0),
            average_win=metrics.get("average_win", 0),
            average_loss=metrics.get("average_loss", 0),
            profit_factor=metrics.get("profit_factor"),
            max_drawdown_day=metrics.get("max_drawdown_day", 0) or metrics.get("max_heat", 0),
            edges_used=metrics.get("edges_used", {}),
            notes=metrics.get("notes"),
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_date(self, date: datetime):
        """Get daily performance for a date."""
        from sqlalchemy import select
        d = date.date() if hasattr(date, "date") else date
        result = await self.session.execute(
            select(DailyPerformance).where(DailyPerformance.date == d)
        )
        return result.scalar_one_or_none()


# =============================================================================
# EDGE PERFORMANCE REPOSITORY
# =============================================================================


class EdgePerformanceRepository:
    """Repository for edge-specific performance."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def update(self, edge_type: str, period: str, period_start: datetime, metrics: dict) -> EdgePerformance:
        """Upsert edge performance record."""
        from sqlalchemy import select
        d = period_start.date() if hasattr(period_start, "date") else period_start
        result = await self.session.execute(
            select(EdgePerformance).where(
                and_(
                    EdgePerformance.edge_type == edge_type,
                    EdgePerformance.period == period,
                    EdgePerformance.period_start == d,
                )
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.trades = metrics.get("trades", existing.trades)
            existing.wins = metrics.get("wins", existing.wins)
            existing.losses = metrics.get("losses", existing.losses)
            existing.total_pnl = metrics.get("total_pnl", existing.total_pnl)
            existing.average_pnl = metrics.get("average_pnl", existing.average_pnl)
            existing.win_rate = metrics.get("win_rate", existing.win_rate)
            existing.expected_edge = metrics.get("expected_edge", existing.expected_edge)
            existing.actual_edge = metrics.get("actual_edge", existing.actual_edge)
            existing.is_healthy = metrics.get("is_healthy", existing.is_healthy)
            existing.decay_warnings = metrics.get("decay_warnings", existing.decay_warnings)
            await self.session.flush()
            return existing
        record = EdgePerformance(
            edge_type=edge_type,
            period=period,
            period_start=d,
            trades=metrics.get("trades", 0),
            wins=metrics.get("wins", 0),
            losses=metrics.get("losses", 0),
            total_pnl=metrics.get("total_pnl", 0),
            average_pnl=metrics.get("average_pnl", 0),
            win_rate=metrics.get("win_rate", 0),
            expected_edge=metrics.get("expected_edge", 0),
            actual_edge=metrics.get("actual_edge", 0),
            is_healthy=metrics.get("is_healthy", True),
            decay_warnings=metrics.get("decay_warnings", 0),
        )
        self.session.add(record)
        await self.session.flush()
        return record


# =============================================================================
# ALERT LOG REPOSITORY
# =============================================================================


class AlertLogRepository:
    """Repository for alert delivery logging."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(self, alert_type: str, message: str, channel: str, success: bool) -> AlertLog:
        """Log a sent alert."""
        record = AlertLog(
            alert_type=alert_type,
            message=message[:500] if message else "",
            channel=channel,
            success=success,
        )
        self.session.add(record)
        await self.session.flush()
        return record
