"""
NEXUS Database Models

SQLAlchemy 2.0 models for persistent storage of signals, trades, and performance data.
PostgreSQL is the target database; models are compatible with SQLite for testing.
"""

from datetime import datetime, date
from typing import Optional, List
from sqlalchemy import (
    String, Float, Integer, Boolean, DateTime, Date, JSON,
    ForeignKey, Index, UniqueConstraint,
    TypeDecorator, CHAR,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid


class GUID(TypeDecorator):
    """
    Platform-independent GUID type.

    Uses PostgreSQL's UUID type when available, otherwise uses
    CHAR(36) storing as stringified UUIDs.

    This allows models to work with both PostgreSQL (production)
    and SQLite (testing).
    """
    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        else:
            return uuid.UUID(value)


class Base(DeclarativeBase):
    """Base class for all NEXUS database models."""
    pass


# =============================================================================
# SIGNAL RECORD
# =============================================================================


class SignalRecord(Base):
    """
    Stores every signal NEXUS generates.
    """
    __tablename__ = "signal_records"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.utcnow
    )
    signal_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    opportunity_id: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    market: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    position_value: Mapped[float] = mapped_column(Float, nullable=False)
    risk_amount: Mapped[float] = mapped_column(Float, nullable=False)
    risk_percent: Mapped[float] = mapped_column(Float, nullable=False)
    primary_edge: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    secondary_edges: Mapped[list] = mapped_column(JSON, default=lambda: [])
    edge_score: Mapped[int] = mapped_column(Integer, nullable=False)
    tier: Mapped[str] = mapped_column(String(1), index=True, nullable=False)
    gross_expected: Mapped[float] = mapped_column(Float, nullable=False)
    net_expected: Mapped[float] = mapped_column(Float, nullable=False)
    cost_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    costs: Mapped[dict] = mapped_column(JSON, nullable=False)
    ai_reasoning: Mapped[Optional[str]] = mapped_column(String(2000))
    confluence_factors: Mapped[list] = mapped_column(JSON, default=lambda: [])
    risk_factors: Mapped[list] = mapped_column(JSON, default=lambda: [])
    market_context: Mapped[Optional[str]] = mapped_column(String(500))
    session: Mapped[Optional[str]] = mapped_column(String(20))
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(20), index=True, nullable=False, default="PENDING"
    )
    outcome_entry_price: Mapped[Optional[float]] = mapped_column(Float)
    outcome_exit_price: Mapped[Optional[float]] = mapped_column(Float)
    outcome_pnl: Mapped[Optional[float]] = mapped_column(Float)
    outcome_pnl_percent: Mapped[Optional[float]] = mapped_column(Float)
    outcome_exit_reason: Mapped[Optional[str]] = mapped_column(String(50))
    outcome_closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    trades: Mapped[List["TradeRecord"]] = relationship(
        "TradeRecord", back_populates="signal"
    )

    __table_args__ = (
        Index("idx_signal_created", created_at.desc()),
        Index("idx_signal_symbol_created", symbol, created_at.desc()),
        Index("idx_signal_status_created", status, created_at.desc()),
        Index("idx_signal_edge_created", primary_edge, created_at.desc()),
        Index("idx_signal_tier_created", tier, created_at.desc()),
    )

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "signal_id": self.signal_id,
            "opportunity_id": self.opportunity_id,
            "symbol": self.symbol,
            "market": self.market,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "position_value": self.position_value,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "primary_edge": self.primary_edge,
            "secondary_edges": self.secondary_edges,
            "edge_score": self.edge_score,
            "tier": self.tier,
            "gross_expected": self.gross_expected,
            "net_expected": self.net_expected,
            "cost_ratio": self.cost_ratio,
            "costs": self.costs,
            "ai_reasoning": self.ai_reasoning,
            "confluence_factors": self.confluence_factors,
            "risk_factors": self.risk_factors,
            "market_context": self.market_context,
            "session": self.session,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "status": self.status,
            "outcome_entry_price": self.outcome_entry_price,
            "outcome_exit_price": self.outcome_exit_price,
            "outcome_pnl": self.outcome_pnl,
            "outcome_pnl_percent": self.outcome_pnl_percent,
            "outcome_exit_reason": self.outcome_exit_reason,
            "outcome_closed_at": (
                self.outcome_closed_at.isoformat() if self.outcome_closed_at else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"SignalRecord(signal_id={self.signal_id!r}, symbol={self.symbol!r}, "
            f"status={self.status!r})"
        )


# =============================================================================
# TRADE RECORD
# =============================================================================


class TradeRecord(Base):
    """
    Stores executed trades.
    """
    __tablename__ = "trade_records"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    signal_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("signal_records.signal_id"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    market: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[Optional[float]] = mapped_column(Float)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[Optional[float]] = mapped_column(Float)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(50))
    slippage_entry: Mapped[Optional[float]] = mapped_column(Float)
    slippage_exit: Mapped[Optional[float]] = mapped_column(Float)
    costs_actual: Mapped[Optional[dict]] = mapped_column(JSON)
    notes: Mapped[Optional[str]] = mapped_column(String(500))

    signal: Mapped[Optional["SignalRecord"]] = relationship(
        "SignalRecord", back_populates="trades"
    )

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "market": self.market,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "exit_reason": self.exit_reason,
            "slippage_entry": self.slippage_entry,
            "slippage_exit": self.slippage_exit,
            "costs_actual": self.costs_actual,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        return (
            f"TradeRecord(id={self.id!r}, symbol={self.symbol!r}, "
            f"pnl={self.pnl})"
        )


# =============================================================================
# DAILY PERFORMANCE
# =============================================================================


class DailyPerformance(Base):
    """
    Daily trading aggregates.
    """
    __tablename__ = "daily_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    date: Mapped[date] = mapped_column(Date, unique=True, index=True, nullable=False)
    starting_equity: Mapped[float] = mapped_column(Float, nullable=False)
    ending_equity: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_percent: Mapped[float] = mapped_column(Float, nullable=False)
    trades_taken: Mapped[int] = mapped_column(Integer, default=0)
    winners: Mapped[int] = mapped_column(Integer, default=0)
    losers: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    largest_win: Mapped[float] = mapped_column(Float, default=0.0)
    largest_loss: Mapped[float] = mapped_column(Float, default=0.0)
    average_win: Mapped[float] = mapped_column(Float, default=0.0)
    average_loss: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown_day: Mapped[float] = mapped_column(Float, default=0.0)
    edges_used: Mapped[dict] = mapped_column(JSON, default=lambda: {})
    notes: Mapped[Optional[str]] = mapped_column(String(500))

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": str(self.id),
            "date": self.date.isoformat() if self.date else None,
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "trades_taken": self.trades_taken,
            "winners": self.winners,
            "losers": self.losers,
            "win_rate": self.win_rate,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown_day": self.max_drawdown_day,
            "edges_used": self.edges_used,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        return (
            f"DailyPerformance(date={self.date!r}, pnl={self.pnl}, "
            f"pnl_percent={self.pnl_percent})"
        )


# =============================================================================
# EDGE PERFORMANCE
# =============================================================================


class EdgePerformance(Base):
    """
    Track performance by edge type.
    """
    __tablename__ = "edge_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    edge_type: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    period: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    period_start: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    average_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    expected_edge: Mapped[float] = mapped_column(Float, nullable=False)
    actual_edge: Mapped[float] = mapped_column(Float, default=0.0)
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    decay_warnings: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint(
            "edge_type", "period", "period_start",
            name="uq_edge_period"
        ),
    )

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": str(self.id),
            "edge_type": self.edge_type,
            "period": self.period,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "average_pnl": self.average_pnl,
            "win_rate": self.win_rate,
            "expected_edge": self.expected_edge,
            "actual_edge": self.actual_edge,
            "is_healthy": self.is_healthy,
            "decay_warnings": self.decay_warnings,
        }

    def __repr__(self) -> str:
        return (
            f"EdgePerformance(edge_type={self.edge_type!r}, period={self.period!r}, "
            f"win_rate={self.win_rate})"
        )


# =============================================================================
# SYSTEM STATE
# =============================================================================


class SystemState(Base):
    """
    Singleton for current system state (always id=1).
    """
    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    current_equity: Mapped[float] = mapped_column(Float, nullable=False)
    starting_equity: Mapped[float] = mapped_column(Float, nullable=False)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    daily_pnl_percent: Mapped[float] = mapped_column(Float, default=0.0)
    weekly_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    weekly_pnl_percent: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_from_peak: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_percent: Mapped[float] = mapped_column(Float, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    open_positions: Mapped[list] = mapped_column(JSON, default=lambda: [])
    portfolio_heat: Mapped[float] = mapped_column(Float, default=0.0)
    portfolio_heat_percent: Mapped[float] = mapped_column(Float, default=0.0)
    circuit_breaker_status: Mapped[str] = mapped_column(String(20), default="CLEAR")
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False)
    last_signal_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_trade_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    mode: Mapped[str] = mapped_column(String(20), default="conservative")

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": self.id,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "current_equity": self.current_equity,
            "starting_equity": self.starting_equity,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": self.daily_pnl_percent,
            "weekly_pnl": self.weekly_pnl,
            "weekly_pnl_percent": self.weekly_pnl_percent,
            "drawdown_from_peak": self.drawdown_from_peak,
            "drawdown_percent": self.drawdown_percent,
            "peak_equity": self.peak_equity,
            "open_positions": self.open_positions,
            "portfolio_heat": self.portfolio_heat,
            "portfolio_heat_percent": self.portfolio_heat_percent,
            "circuit_breaker_status": self.circuit_breaker_status,
            "kill_switch_active": self.kill_switch_active,
            "last_signal_at": (
                self.last_signal_at.isoformat() if self.last_signal_at else None
            ),
            "last_trade_at": (
                self.last_trade_at.isoformat() if self.last_trade_at else None
            ),
            "mode": self.mode,
        }

    def __repr__(self) -> str:
        return (
            f"SystemState(id={self.id}, current_equity={self.current_equity}, "
            f"portfolio_heat={self.portfolio_heat})"
        )


# =============================================================================
# AUDIT LOG
# =============================================================================


class AuditLog(Base):
    """
    System event logging.
    """
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True, nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    component: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    signal_id: Mapped[Optional[str]] = mapped_column(String(50))
    trade_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())

    __table_args__ = (
        Index("idx_audit_type_timestamp", event_type, timestamp.desc()),
        Index("idx_audit_severity_timestamp", severity, timestamp.desc()),
    )

    def to_dict(self) -> dict:
        """Return all fields as dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "severity": self.severity,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "signal_id": self.signal_id,
            "trade_id": str(self.trade_id) if self.trade_id else None,
        }

    def __repr__(self) -> str:
        return (
            f"AuditLog(id={self.id!r}, event_type={self.event_type!r}, "
            f"severity={self.severity!r})"
        )


# =============================================================================
# ALERT LOG
# =============================================================================


class AlertLog(Base):
    """
    Log of sent alerts (Discord, Telegram, etc.).
    """
    __tablename__ = "alert_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True, nullable=False
    )
    alert_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    channel: Mapped[str] = mapped_column(String(30), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "alert_type": self.alert_type,
            "message": self.message,
            "channel": self.channel,
            "success": self.success,
        }
