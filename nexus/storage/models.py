"""
NEXUS Database Models

SQLAlchemy models for persistent storage of signals, trades, and metrics.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from storage.database import Base


# =============================================================================
# SIGNAL RECORD
# =============================================================================

class SignalRecord(Base):
    """
    Stores all generated trading signals.

    This is the primary record of every signal NEXUS generates,
    whether traded or not.
    """
    __tablename__ = "signals"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Signal identification
    signal_id = Column(String(50), unique=True, nullable=False, index=True)
    opportunity_id = Column(String(50), nullable=False)

    # Instrument
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(30), nullable=False, index=True)
    direction = Column(String(10), nullable=False)

    # Prices
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)

    # Position sizing
    position_size = Column(Float, nullable=False)
    position_value = Column(Float, nullable=False)
    risk_amount = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)

    # Edge analysis
    primary_edge = Column(String(50), nullable=False, index=True)
    secondary_edges = Column(JSONB, default=list)
    edge_score = Column(Integer, nullable=False, index=True)
    tier = Column(String(1), nullable=False, index=True)

    # Cost analysis
    gross_expected = Column(Float, nullable=False)
    net_expected = Column(Float, nullable=False)
    cost_ratio = Column(Float, nullable=False)
    costs = Column(JSONB, nullable=False)  # Full CostBreakdown

    # AI reasoning
    ai_reasoning = Column(Text, default="")
    confluence_factors = Column(JSONB, default=list)
    risk_factors = Column(JSONB, default=list)

    # Context
    market_context = Column(Text, default="")
    regime = Column(String(20))
    session = Column(String(30), default="")

    # Status
    status = Column(String(20), nullable=False, default="pending", index=True)
    valid_until = Column(DateTime(timezone=True))

    # Full signal data (for reconstruction)
    signal_data = Column(JSONB, nullable=False)

    # Relationships
    trade = relationship("TradeRecord", back_populates="signal", uselist=False)

    # Indexes for common queries
    __table_args__ = (
        Index("ix_signals_created_at", "created_at"),
        Index("ix_signals_symbol_created", "symbol", "created_at"),
        Index("ix_signals_edge_created", "primary_edge", "created_at"),
        Index("ix_signals_status_created", "status", "created_at"),
    )

    def __repr__(self):
        return f"<Signal {self.signal_id}: {self.symbol} {self.direction} {self.status}>"


# =============================================================================
# TRADE RECORD
# =============================================================================

class TradeRecord(Base):
    """
    Stores executed trades with full outcome data.

    Linked to SignalRecord - every trade comes from a signal.
    """
    __tablename__ = "trades"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Link to signal
    signal_id = Column(String(50), ForeignKey("signals.signal_id"), nullable=False, index=True)
    signal = relationship("SignalRecord", back_populates="trade")

    # Instrument
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(30), nullable=False)
    direction = Column(String(10), nullable=False)
    primary_edge = Column(String(50), nullable=False, index=True)

    # Entry
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    planned_stop = Column(Float, nullable=False)
    planned_target = Column(Float, nullable=False)

    # Exit
    exit_price = Column(Float)
    exit_time = Column(DateTime(timezone=True))
    exit_reason = Column(String(30))  # stopped, target, manual, expired

    # P&L
    pnl = Column(Float)  # Absolute P&L in account currency
    pnl_pct = Column(Float)  # P&L as percentage
    r_multiple = Column(Float)  # Risk multiple achieved

    # Position
    position_size = Column(Float, nullable=False)
    planned_risk_pct = Column(Float, nullable=False)
    actual_risk_pct = Column(Float)

    # Execution quality
    slippage_entry = Column(Float, default=0.0)
    slippage_exit = Column(Float, default=0.0)

    # Costs
    actual_costs = Column(JSONB)  # Full CostBreakdown

    # Trade duration
    hold_duration_minutes = Column(Integer)

    # Indexes
    __table_args__ = (
        Index("ix_trades_entry_time", "entry_time"),
        Index("ix_trades_symbol_entry", "symbol", "entry_time"),
        Index("ix_trades_edge_entry", "primary_edge", "entry_time"),
        Index("ix_trades_pnl", "pnl"),
    )

    def __repr__(self):
        return f"<Trade {self.signal_id}: {self.symbol} PnL={self.pnl}>"


# =============================================================================
# DAILY PERFORMANCE
# =============================================================================

class DailyPerformance(Base):
    """
    Daily aggregated performance metrics.

    One row per trading day.
    """
    __tablename__ = "daily_performance"

    # Primary key is the date
    date = Column(DateTime(timezone=True), primary_key=True)

    # Equity tracking
    starting_equity = Column(Float, nullable=False)
    ending_equity = Column(Float, nullable=False)
    high_equity = Column(Float)  # Intraday high
    low_equity = Column(Float)   # Intraday low

    # P&L
    pnl = Column(Float, nullable=False, default=0.0)
    pnl_pct = Column(Float, nullable=False, default=0.0)

    # Trade stats
    trades_taken = Column(Integer, default=0)
    winners = Column(Integer, default=0)
    losers = Column(Integer, default=0)
    scratch = Column(Integer, default=0)  # Break-even trades

    # Win rate
    win_rate = Column(Float)

    # Best/worst
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)

    # Risk metrics
    max_heat_used = Column(Float)  # Peak portfolio heat
    max_drawdown_intraday = Column(Float)

    # Edge breakdown
    edges_used = Column(JSONB, default=dict)  # {edge_type: count}

    # Signals
    signals_generated = Column(Integer, default=0)
    signals_traded = Column(Integer, default=0)
    signals_skipped = Column(Integer, default=0)

    def __repr__(self):
        return f"<DailyPerformance {self.date.date()}: {self.pnl_pct:+.2f}%>"


# =============================================================================
# EDGE PERFORMANCE
# =============================================================================

class EdgePerformance(Base):
    """
    Performance tracking for each edge type.

    Used for edge decay detection and strategy optimization.
    """
    __tablename__ = "edge_performance"

    # Composite primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    edge_type = Column(String(50), nullable=False, index=True)
    period = Column(String(10), nullable=False)  # daily, weekly, monthly
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Trade stats
    trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)

    # P&L
    total_pnl = Column(Float, default=0.0)
    avg_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_pnl_pct = Column(Float, default=0.0)

    # Win rate
    win_rate = Column(Float)

    # R multiples
    avg_r_multiple = Column(Float)
    total_r = Column(Float)

    # Health flag (for edge decay detection)
    is_healthy = Column(Boolean, default=True)
    consecutive_losing_periods = Column(Integer, default=0)

    # Baseline comparison
    baseline_win_rate = Column(Float)
    baseline_avg_pnl = Column(Float)
    deviation_from_baseline = Column(Float)  # In standard deviations

    # Indexes
    __table_args__ = (
        Index("ix_edge_perf_type_period", "edge_type", "period", "period_start"),
    )

    def __repr__(self):
        return f"<EdgePerformance {self.edge_type} {self.period}: {self.win_rate:.1%}>"


# =============================================================================
# SYSTEM STATE
# =============================================================================

class SystemState(Base):
    """
    Current system state - singleton table.

    Tracks real-time system status for circuit breakers and monitoring.
    """
    __tablename__ = "system_state"

    # Always id=1 (singleton)
    id = Column(Integer, primary_key=True, default=1)

    # Last update
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Equity
    starting_equity_today = Column(Float, nullable=False)
    current_equity = Column(Float, nullable=False)
    peak_equity = Column(Float, nullable=False)

    # P&L
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    weekly_pnl = Column(Float, default=0.0)
    weekly_pnl_pct = Column(Float, default=0.0)
    monthly_pnl = Column(Float, default=0.0)
    monthly_pnl_pct = Column(Float, default=0.0)

    # Drawdown
    drawdown = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)

    # Positions
    open_positions = Column(JSONB, default=list)
    open_position_count = Column(Integer, default=0)
    portfolio_heat = Column(Float, default=0.0)

    # Streaks
    win_streak = Column(Integer, default=0)
    loss_streak = Column(Integer, default=0)

    # Circuit breaker status
    circuit_breaker_status = Column(String(20), default="clear")
    circuit_breaker_reason = Column(Text, default="")

    # Kill switch
    kill_switch_active = Column(Boolean, default=False)
    kill_switch_reason = Column(Text, default="")
    kill_switch_activated_at = Column(DateTime(timezone=True))

    # System health
    is_healthy = Column(Boolean, default=True)
    last_scan_at = Column(DateTime(timezone=True))
    last_signal_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<SystemState: equity={self.current_equity} heat={self.portfolio_heat}%>"


# =============================================================================
# ALERT LOG
# =============================================================================

class AlertLog(Base):
    """
    Log of all alerts sent.

    Tracks what was sent, when, and delivery status.
    """
    __tablename__ = "alert_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Alert details
    alert_type = Column(String(30), nullable=False)  # signal, warning, error, etc.
    priority = Column(String(10), nullable=False)  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)

    # Related entities
    signal_id = Column(String(50))

    # Delivery
    channels = Column(JSONB, default=list)  # ["discord", "telegram"]
    delivered_discord = Column(Boolean, default=False)
    delivered_telegram = Column(Boolean, default=False)
    delivery_error = Column(Text)

    # Indexes
    __table_args__ = (
        Index("ix_alerts_created", "created_at"),
        Index("ix_alerts_type", "alert_type", "created_at"),
    )

    def __repr__(self):
        return f"<Alert {self.alert_type}: {self.title[:30]}>"
