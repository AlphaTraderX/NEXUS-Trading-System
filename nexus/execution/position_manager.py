"""
NEXUS Position Manager
Tracks open/closed positions and portfolio state for the orchestrator.

Minimal implementation for main.py orchestrator; extend for full execution.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field


class ExitReason(Enum):
    """Reason a position was closed."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    KILL_SWITCH = "kill_switch"


@dataclass
class Position:
    """Minimal position representation for orchestrator."""
    symbol: str
    market: Any
    direction: Any
    current_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def risk_to_stop(self) -> float:
        """Current risk if stop is hit."""
        if self.current_size <= 0:
            return 0.0
        risk_per_unit = abs(self.entry_price - self.stop_loss)
        return risk_per_unit * abs(self.current_size)


class PositionManager:
    """
    Tracks positions and portfolio summary for the NEXUS orchestrator.
    """

    def __init__(self):
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.current_size != 0]

    def get_pending_positions(self) -> List[Position]:
        """Get pending positions (placeholder)."""
        return []

    def get_portfolio_summary(self) -> Dict:
        """Get summary of all positions."""
        open_positions = self.get_open_positions()
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_realized = sum(p.realized_pnl for p in self.closed_positions)
        total_risk = sum(p.risk_to_stop for p in open_positions)
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "open_count": len(open_positions),
            "pending_count": len(self.get_pending_positions()),
            "closed_count": len(self.closed_positions),
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "total_pnl": total_unrealized + total_realized,
            "total_risk_to_stop": total_risk,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
        }
