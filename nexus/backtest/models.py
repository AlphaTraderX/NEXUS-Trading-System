"""Backtest data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from nexus.core.enums import EdgeType, Direction, Market, Timeframe


class TradeOutcome(str, Enum):
    """How a trade ended."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STOPPED = "stopped"
    TARGET = "target"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"


@dataclass
class BacktestTrade:
    """A single backtest trade."""

    # Identification
    trade_id: str
    signal_id: str

    # Trade details
    symbol: str
    market: Market
    direction: Direction
    edge_type: EdgeType
    timeframe: Timeframe
    score: int
    tier: str

    # Prices
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float

    # Timing
    entry_time: datetime
    exit_time: datetime
    hold_duration_minutes: int

    # P&L
    gross_pnl: float
    gross_pnl_pct: float
    costs: float
    net_pnl: float
    net_pnl_pct: float

    # Risk metrics
    risk_amount: float
    risk_pct: float
    r_multiple: float  # Net P&L / Risk Amount

    # Outcome
    outcome: TradeOutcome
    exit_reason: str

    # Context
    edge_data: Dict[str, Any] = field(default_factory=dict)
    market_regime: Optional[str] = None

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.net_pnl < 0


@dataclass
class EdgePerformance:
    """Performance metrics for a single edge."""

    edge_type: EdgeType
    timeframe: Optional[Timeframe] = None
    market: Optional[Market] = None

    # Trade counts
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0

    # P&L
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0

    # Percentages
    total_gross_pct: float = 0.0
    total_net_pct: float = 0.0

    # Risk metrics
    total_risk: float = 0.0
    total_r_multiple: float = 0.0

    # Time
    avg_hold_minutes: float = 0.0
    total_hold_minutes: int = 0

    # Derived metrics (calculated)
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winners / self.total_trades * 100

    @property
    def avg_win(self) -> float:
        if self.winners == 0:
            return 0.0
        return self.net_pnl / self.winners if self.net_pnl > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        if self.losers == 0:
            return 0.0
        return 0.0

    @property
    def profit_factor(self) -> float:
        """Gross wins / Gross losses."""
        return 0.0

    @property
    def expectancy(self) -> float:
        """Average R-multiple per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_r_multiple / self.total_trades

    @property
    def avg_net_pct(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_net_pct / self.total_trades

    @property
    def cost_ratio(self) -> float:
        """Costs as % of gross P&L."""
        if self.gross_pnl == 0:
            return 0.0
        return abs(self.total_costs / self.gross_pnl) * 100

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0

    @property
    def statistical_significance(self) -> str:
        """
        Rough significance based on trade count.
        30+ trades = moderate, 100+ = good, 385+ = strong
        """
        if self.total_trades >= 385:
            return "STRONG"
        elif self.total_trades >= 100:
            return "GOOD"
        elif self.total_trades >= 30:
            return "MODERATE"
        else:
            return "INSUFFICIENT"


@dataclass
class MultiBacktestResult:
    """Complete backtest results for the multi-asset framework."""

    # Metadata
    start_date: datetime
    end_date: datetime
    starting_balance: float
    ending_balance: float

    # All trades
    trades: List[BacktestTrade] = field(default_factory=list)

    # Performance by edge
    edge_performance: Dict[str, EdgePerformance] = field(default_factory=dict)

    # Performance by timeframe
    timeframe_performance: Dict[str, EdgePerformance] = field(default_factory=dict)

    # Performance by market
    market_performance: Dict[str, EdgePerformance] = field(default_factory=dict)

    # Overall metrics
    total_trades: int = 0
    total_winners: int = 0
    total_losers: int = 0

    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Equity curve
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def net_return_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.ending_balance - self.starting_balance) / self.starting_balance * 100

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_winners / self.total_trades * 100

    @property
    def avg_trade_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.net_pnl / self.total_trades
