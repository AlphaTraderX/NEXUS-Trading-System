"""
NEXUS Core Data Models

Pydantic models for opportunities and signals.
Dataclasses for risk/position sizing results.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    CircuitBreakerStatus,
    Direction,
    EdgeType,
    KillSwitchAction,
    KillSwitchTrigger,
    Market,
    SignalStatus,
    SignalTier,
)

UTC = timezone.utc


@dataclass
class PositionSize:
    """Result of dynamic position size calculation."""

    risk_pct: float
    risk_amount: float
    position_size: float
    position_value: float
    stop_distance: float
    stop_distance_pct: float
    score_multiplier: float
    regime_multiplier: float
    momentum_multiplier: float
    kelly_multiplier: float = 1.0
    kelly_adjusted_risk: float = 0.0
    heat_after_trade: float = 0.0
    can_trade: bool = True
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackedPosition:
    """Position tracked for heat/risk management."""

    position_id: str
    symbol: str
    market: Market
    direction: Direction
    entry_price: float
    current_price: float
    stop_loss: float
    position_size: float
    risk_amount: float
    risk_pct: float
    sector: Optional[str] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def current_pnl(self) -> float:
        """Unrealized P&L in currency."""
        if self.direction == Direction.LONG:
            return (self.current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - self.current_price) * self.position_size

    @property
    def current_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of position value."""
        position_value = self.entry_price * self.position_size
        if position_value == 0:
            return 0.0
        return (self.current_pnl / position_value) * 100

    def to_dict(self) -> dict:
        """Serialize for storage/summary (enums as values)."""
        d = asdict(self)
        d["market"] = self.market.value if hasattr(self.market, "value") else str(self.market)
        d["direction"] = self.direction.value if hasattr(self.direction, "value") else str(self.direction)
        d["opened_at"] = self.opened_at.isoformat() if isinstance(self.opened_at, datetime) else self.opened_at
        return d


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker for risk management."""

    status: CircuitBreakerStatus
    can_trade: bool
    size_multiplier: float  # 1.0 = full, 0.5 = half, 0 = none
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float
    message: str
    triggered_at: Optional[datetime] = None
    resume_at: Optional[datetime] = None  # When can trading resume?

    def to_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        if self.triggered_at:
            result["triggered_at"] = self.triggered_at.isoformat()
        if self.resume_at:
            result["resume_at"] = self.resume_at.isoformat()
        return result


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""

    is_triggered: bool
    trigger: KillSwitchTrigger
    action_taken: KillSwitchAction
    triggered_at: Optional[datetime]
    message: str
    can_reset: bool  # True if cooldown period has passed
    cooldown_remaining_seconds: int
    system_status: Dict[str, Any]  # Current system health metrics

    def to_dict(self) -> dict:
        result = {
            "is_triggered": self.is_triggered,
            "trigger": self.trigger.value,
            "action_taken": self.action_taken.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "message": self.message,
            "can_reset": self.can_reset,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "system_status": self.system_status,
        }
        return result


@dataclass
class SystemHealth:
    """Current system health for kill switch monitoring."""

    last_heartbeat: Optional[datetime]
    last_data_update: Optional[datetime]
    seconds_since_heartbeat: float
    seconds_since_data: float
    drawdown_pct: float
    is_connected: bool
    active_errors: List[str]

    def to_dict(self) -> dict:
        return {
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "seconds_since_heartbeat": self.seconds_since_heartbeat,
            "seconds_since_data": self.seconds_since_data,
            "drawdown_pct": self.drawdown_pct,
            "is_connected": self.is_connected,
            "active_errors": self.active_errors,
        }


@dataclass
class HeatCheckResult:
    """Result of can_add_position check."""

    allowed: bool
    current_heat: float
    heat_after: float
    heat_limit: float
    market_heat: float
    market_heat_after: float
    market_limit: float
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeatSummary:
    """Complete portfolio heat summary."""

    total_heat: float
    heat_limit: float
    headroom: float
    position_count: int
    heat_by_market: Dict[str, float]
    heat_by_sector: Dict[str, float]
    positions: List[dict]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorrelationCheckResult:
    """Result of correlation/concentration check before adding a position."""

    allowed: bool
    nominal_risk_pct: float
    effective_risk_pct: float  # Correlation-adjusted
    risk_multiplier: float  # effective / nominal
    sector_count: Dict[str, int]  # Positions per sector
    direction_count: Dict[str, int]  # Positions per market+direction
    high_correlation_pairs: List[Tuple[str, str, float]]  # Correlated positions
    warnings: List[str]
    rejection_reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "nominal_risk_pct": self.nominal_risk_pct,
            "effective_risk_pct": self.effective_risk_pct,
            "risk_multiplier": self.risk_multiplier,
            "sector_count": self.sector_count,
            "direction_count": self.direction_count,
            "high_correlation_pairs": [
                {"symbol1": p[0], "symbol2": p[1], "correlation": p[2]}
                for p in self.high_correlation_pairs
            ],
            "warnings": self.warnings,
            "rejection_reasons": self.rejection_reasons,
        }


class Opportunity(BaseModel):
    """
    A detected trading opportunity from a scanner.

    Raw output from scanners before scoring and cost analysis.
    """

    id: str
    detected_at: datetime
    scanner: str = Field(description="Name of scanner that detected this")

    symbol: str
    market: Market

    direction: Direction
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)

    primary_edge: EdgeType
    secondary_edges: List[EdgeType] = Field(default_factory=list)
    edge_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scanner-specific data about the edge",
    )

    raw_score: int = Field(default=0, ge=0, le=100)
    adjusted_score: int = Field(default=0, ge=0, le=100)

    valid_until: Optional[datetime] = None

    # Confluence tracking
    confluence_count: int = Field(default=1, ge=1, description="How many edges fired on this symbol")
    confluence_edges: List[EdgeType] = Field(default_factory=list, description="All edges that fired")
    is_confluence: bool = Field(default=False, description="True if 2+ edges fired")

    model_config = ConfigDict(use_enum_values=True)

    @property
    def confluence_multiplier(self) -> float:
        """Position size multiplier based on confluence count."""
        if self.confluence_count >= 3:
            return 2.0
        if self.confluence_count == 2:
            return 1.5
        return 1.0

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio from entry, stop, and target."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk <= 0:
            return 0.0
        return reward / risk


@dataclass
class ScoredOpportunity:
    """An opportunity with its score and tier."""

    opportunity: Opportunity
    score: int
    tier: Union[SignalTier, str]
    factors: List[str]
    position_multiplier: float = 1.0


class NexusSignal(BaseModel):
    """
    Complete, actionable trading signal.

    Output of the signal generator - everything needed to execute.
    """

    signal_id: str
    created_at: datetime
    opportunity_id: str

    symbol: str
    market: Market
    direction: Direction
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)

    position_size: float = Field(gt=0)
    position_value: float = Field(gt=0)
    risk_amount: float = Field(ge=0)
    risk_percent: float = Field(ge=0)

    primary_edge: EdgeType
    secondary_edges: List[EdgeType] = Field(default_factory=list)
    edge_score: int = Field(ge=0, le=100)
    tier: Any = Field(description="SignalTier enum or string")

    gross_expected: float = 0.0
    costs: Any = Field(default_factory=dict, description="CostBreakdown or dict")
    net_expected: float = 0.0
    cost_ratio: float = 0.0

    ai_reasoning: str = ""
    confluence_factors: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    market_context: str = ""
    session: str = ""
    valid_until: Optional[datetime] = None
    regime: Optional[Any] = None

    status: SignalStatus = SignalStatus.PENDING

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio from entry, stop, and target."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk <= 0:
            return 0.0
        return reward / risk

    def to_dict(self) -> dict:
        """Serialize for storage."""
        data = self.model_dump(mode="json")
        if hasattr(self.costs, "to_dict"):
            data["costs"] = self.costs.to_dict()
        return data


@dataclass
class TradeResult:
    """
    Result of a completed (or partially completed) trade.
    Used when persisting trade records to storage.
    """
    symbol: str
    market: Union[Market, str]
    direction: Union[Direction, str]
    entry_price: float
    entry_time: datetime
    position_size: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    slippage_entry: Optional[float] = None
    slippage_exit: Optional[float] = None
    actual_costs: Optional[Any] = None
