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

from storage.models import (
    SignalRecord,
    TradeRecord,
    DailyPerformance,
    EdgePerformance,
    SystemState,
    AlertLog,
)
from core.models import NexusSignal, TradeResult

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
            primary_edge=signal.primary_edge.value if hasattr(signal.primary_edge, 'value') else signal.primary_edge,
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
            market_context=signal.market_context,
            regime=signal.regime.value if signal.regime and hasattr(signal.regime, 'value') else signal.regime,
            session=signal.session,
            status=signal.status.value if hasattr(signal.status, 'value') else signal.status,
            valid_until=signal.valid_until,
            signal_data=signal.to_dict(),
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
            .where(SignalRecord.status == "pending")
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
        """Save a completed trade."""
        record = TradeRecord(
            signal_id=signal_id,
            symbol=trade.symbol,
            market=trade.market.value if hasattr(trade.market, 'value') else trade.market,
            direction=trade.direction.value if hasattr(trade.direction, 'value') else trade.direction,
            primary_edge=trade.primary_edge.value if hasattr(trade.primary_edge, 'value') else trade.primary_edge,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            planned_stop=trade.planned_stop,
            planned_target=trade.planned_target,
            exit_price=trade.exit_price,
            exit_time=trade.exit_time,
            exit_reason=trade.exit_reason,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            r_multiple=trade.r_multiple,
            position_size=0,  # TODO: Add to TradeResult
            planned_risk_pct=trade.planned_risk_pct,
            actual_risk_pct=trade.actual_risk_pct,
            slippage_entry=trade.slippage_entry,
            slippage_exit=trade.slippage_exit,
            actual_costs=trade.actual_costs.to_dict() if hasattr(trade.actual_costs, 'to_dict') else None,
            hold_duration_minutes=trade.hold_duration_minutes,
        )

        self.session.add(record)
        await self.session.flush()
        logger.info(f"Saved trade for signal {signal_id}")
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
        """Get trades by edge type."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(TradeRecord)
            .where(
                and_(
                    TradeRecord.primary_edge == edge_type,
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
            starting_equity_today=starting_equity,
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
                daily_pnl_pct=daily_pnl_pct,
                peak_equity=max(current_equity, current_equity),  # Will need actual peak
                last_updated=datetime.utcnow(),
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
                open_position_count=len(positions),
                portfolio_heat=heat,
                last_updated=datetime.utcnow(),
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
                circuit_breaker_reason=reason,
                last_updated=datetime.utcnow(),
            )
        )

    async def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch."""
        await self.session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                kill_switch_active=True,
                kill_switch_reason=reason,
                kill_switch_activated_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
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
