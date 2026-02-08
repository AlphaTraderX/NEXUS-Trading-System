"""
StorageService - Single interface to all database operations.
Used by execution layer components.
"""

from typing import Optional, List
from datetime import datetime

from nexus.storage.database import get_async_session, init_db_async
from nexus.storage.repositories import (
    SignalRepository,
    TradeRepository,
    SystemStateRepository,
    DailyPerformanceRepository,
    EdgePerformanceRepository,
    AlertLogRepository,
)
from nexus.core.models import NexusSignal
from nexus.core.enums import SignalStatus, EdgeType


class StorageService:
    """
    Unified storage interface for NEXUS.

    Usage:
        storage = StorageService()
        await storage.initialize()

        # Save signal
        await storage.save_signal(signal)

        # Get system state
        state = await storage.get_system_state()
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database connection and create tables if needed."""
        try:
            await init_db_async(url=self.database_url)
            self._initialized = True
            return True
        except Exception as e:
            print(f"Storage initialization failed: {e}")
            return False

    # === SIGNAL OPERATIONS ===

    async def save_signal(self, signal: NexusSignal) -> bool:
        """Save a generated signal to database."""
        async with get_async_session() as session:
            repo = SignalRepository(session)
            await repo.save(signal)
            return True

    async def get_signal(self, signal_id: str) -> Optional[dict]:
        """Get signal by ID."""
        async with get_async_session() as session:
            repo = SignalRepository(session)
            record = await repo.get_by_id(signal_id)
            return record.to_dict() if record else None

    async def get_pending_signals(self) -> List[dict]:
        """Get all pending signals."""
        async with get_async_session() as session:
            repo = SignalRepository(session)
            records = await repo.get_pending()
            return [r.to_dict() for r in records]

    async def update_signal_status(self, signal_id: str, status: SignalStatus) -> bool:
        """Update signal status."""
        async with get_async_session() as session:
            repo = SignalRepository(session)
            status_val = status.value if hasattr(status, "value") else str(status)
            await repo.update_status(signal_id, status_val)
            return True

    async def get_recent_signals(self, limit: int = 50) -> List[dict]:
        """Get recent signals."""
        async with get_async_session() as session:
            repo = SignalRepository(session)
            records = await repo.get_recent(limit)
            return [r.to_dict() for r in records]

    # === TRADE OPERATIONS ===

    async def save_trade(self, trade_data: dict, signal_id: str) -> bool:
        """Save executed trade (entry) to database."""
        async with get_async_session() as session:
            repo = TradeRepository(session)
            await repo.save_entry(trade_data, signal_id)
            return True

    async def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Get recent trades."""
        async with get_async_session() as session:
            repo = TradeRepository(session)
            records = await repo.get_recent(limit)
            return [r.to_dict() for r in records]

    async def get_trades_by_edge(self, edge_type: EdgeType, limit: int = 100) -> List[dict]:
        """Get trades filtered by edge type."""
        async with get_async_session() as session:
            repo = TradeRepository(session)
            edge_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
            records = await repo.get_by_edge(edge_val, days=30)
            return [r.to_dict() for r in records[:limit]]

    # === SYSTEM STATE OPERATIONS ===

    async def get_system_state(self) -> Optional[dict]:
        """Get current system state."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            state = await repo.get()
            if state:
                return {
                    "current_equity": state.current_equity,
                    "daily_pnl": state.daily_pnl,
                    "daily_pnl_pct": state.daily_pnl_percent,
                    "weekly_pnl": state.weekly_pnl,
                    "weekly_pnl_pct": state.weekly_pnl_percent,
                    "drawdown_pct": state.drawdown_percent,
                    "portfolio_heat": state.portfolio_heat,
                    "open_positions": state.open_positions,
                    "circuit_breaker_status": state.circuit_breaker_status,
                    "kill_switch_active": state.kill_switch_active,
                    "last_updated": state.updated_at.isoformat() if state.updated_at else None,
                }
            return None

    async def initialize_system_state(self, starting_equity: float) -> bool:
        """Initialize or reset system state."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            await repo.initialize(starting_equity)
            return True

    async def update_equity(self, current_equity: float, daily_pnl: float, daily_pnl_pct: float) -> bool:
        """Update equity and P&L."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            await repo.update_equity(current_equity, daily_pnl, daily_pnl_pct)
            return True

    async def update_positions(self, positions: dict, heat: float) -> bool:
        """Update open positions and heat."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            # SystemState.open_positions is JSON; accept dict or list
            positions_data = positions if isinstance(positions, list) else list(positions.values())
            await repo.update_positions(positions_data, heat)
            return True

    async def set_circuit_breaker(self, status: str, reason: str) -> bool:
        """Update circuit breaker status."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            await repo.set_circuit_breaker(status, reason)
            return True

    async def activate_kill_switch(self, reason: str) -> bool:
        """Activate kill switch."""
        async with get_async_session() as session:
            repo = SystemStateRepository(session)
            await repo.activate_kill_switch(reason)
            return True

    # === PERFORMANCE TRACKING ===

    async def record_daily_performance(self, date: datetime, metrics: dict) -> bool:
        """Record daily performance metrics."""
        async with get_async_session() as session:
            repo = DailyPerformanceRepository(session)
            await repo.save(date, metrics)
            return True

    async def record_edge_performance(self, edge_type: EdgeType, metrics: dict) -> bool:
        """Record edge-specific performance."""
        async with get_async_session() as session:
            repo = EdgePerformanceRepository(session)
            edge_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
            period = metrics.get("period", "daily")
            period_start = metrics.get("period_start", datetime.utcnow())
            if not isinstance(period_start, datetime):
                period_start = datetime(period_start.year, period_start.month, period_start.day)
            await repo.update(edge_val, period, period_start, metrics)
            return True

    # === ALERT LOGGING ===

    async def log_alert(self, alert_type: str, message: str, channel: str, success: bool) -> bool:
        """Log sent alert."""
        async with get_async_session() as session:
            repo = AlertLogRepository(session)
            await repo.log(alert_type, message, channel, success)
            return True


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
