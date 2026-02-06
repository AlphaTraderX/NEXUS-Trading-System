"""
NEXUS Core Data Models

Pydantic models for opportunities and signals.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core.enums import Direction, EdgeType, Market, SignalStatus


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

    class Config:
        use_enum_values = True

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio from entry, stop, and target."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk <= 0:
            return 0.0
        return reward / risk


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

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

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
