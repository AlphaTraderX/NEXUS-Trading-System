"""
NEXUS Metrics Collection

Collects and aggregates trading metrics from various components.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


def _get_summary_value(summary: Any, key: str, default: Any = 0) -> Any:
    """Get value from portfolio summary (dict or dataclass e.g. PortfolioMetrics)."""
    if summary is None:
        return default
    if isinstance(summary, dict):
        return summary.get(key, default)
    return getattr(summary, key, default)


@dataclass
class TradingMetrics:
    """Aggregated trading metrics."""
    # Equity
    current_equity: float = 0.0
    starting_equity: float = 0.0
    peak_equity: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl: float = 0.0
    monthly_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    # Drawdown
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Average trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_r_multiple: float = 0.0

    # Position info
    open_positions: int = 0
    portfolio_heat: float = 0.0
    portfolio_heat_pct: float = 0.0

    # Edge breakdown
    edge_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Timestamps
    last_signal_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "equity": {
                "current": self.current_equity,
                "starting": self.starting_equity,
                "peak": self.peak_equity,
            },
            "pnl": {
                "daily": self.daily_pnl,
                "daily_pct": self.daily_pnl_pct,
                "weekly": self.weekly_pnl,
                "weekly_pct": self.weekly_pnl_pct,
                "monthly": self.monthly_pnl,
                "monthly_pct": self.monthly_pnl_pct,
                "total": self.total_pnl,
                "total_pct": self.total_pnl_pct,
            },
            "drawdown": {
                "current": self.drawdown,
                "current_pct": self.drawdown_pct,
                "max": self.max_drawdown,
                "max_pct": self.max_drawdown_pct,
            },
            "trades": {
                "total": self.total_trades,
                "winners": self.winning_trades,
                "losers": self.losing_trades,
                "win_rate": self.win_rate,
            },
            "profit": {
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "profit_factor": self.profit_factor,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "avg_trade": self.avg_trade,
                "avg_r_multiple": self.avg_r_multiple,
            },
            "positions": {
                "open": self.open_positions,
                "heat": self.portfolio_heat,
                "heat_pct": self.portfolio_heat_pct,
            },
            "edges": self.edge_performance,
            "timestamps": {
                "last_signal": (
                    self.last_signal_at.isoformat() if self.last_signal_at else None
                ),
                "last_trade": (
                    self.last_trade_at.isoformat() if self.last_trade_at else None
                ),
                "updated": self.updated_at.isoformat(),
            },
        }


class MetricsCollector:
    """
    Collects metrics from NEXUS components.

    Sources:
    - PositionManager: Open positions, P&L, heat (via get_portfolio_metrics)
    - StorageService: Historical trades, daily performance
    - CircuitBreaker: Current status
    """

    def __init__(self):
        self._metrics = TradingMetrics()
        self._trade_history: List[dict] = []
        self._daily_history: List[dict] = []

    def update_from_position_manager(self, position_manager: Any) -> None:
        """Pull metrics from PositionManager (get_portfolio_metrics)."""
        if not position_manager:
            return

        try:
            if hasattr(position_manager, "get_portfolio_metrics"):
                summary = position_manager.get_portfolio_metrics()
            elif hasattr(position_manager, "get_portfolio_summary"):
                summary = position_manager.get_portfolio_summary()
                if hasattr(summary, "to_dict"):
                    summary = summary.to_dict()
            else:
                logger.warning(
                    "Position manager has no get_portfolio_metrics or get_portfolio_summary"
                )
                return

            self._metrics.daily_pnl = _get_summary_value(
                summary, "total_pnl", 0
            ) or (
                _get_summary_value(summary, "total_unrealized_pnl", 0)
                + _get_summary_value(summary, "total_realized_pnl", 0)
            )
            self._metrics.open_positions = _get_summary_value(
                summary, "open_positions", 0
            )
            self._metrics.portfolio_heat = _get_summary_value(
                summary, "total_risk_amount", 0
            )
            self._metrics.portfolio_heat_pct = _get_summary_value(
                summary, "portfolio_heat", 0
            )

            self._metrics.total_trades = _get_summary_value(
                summary, "total_trades", 0
            )
            self._metrics.winning_trades = _get_summary_value(
                summary, "winning_positions", 0
            )
            self._metrics.losing_trades = _get_summary_value(
                summary, "losing_positions", 0
            )
            self._metrics.win_rate = _get_summary_value(summary, "win_rate", 0)
            self._metrics.profit_factor = _get_summary_value(
                summary, "profit_factor", 0
            )
            self._metrics.avg_r_multiple = _get_summary_value(
                summary, "avg_r_multiple", 0
            )
            self._metrics.avg_win = _get_summary_value(summary, "avg_win", 0)
            self._metrics.avg_loss = _get_summary_value(summary, "avg_loss", 0)
            self._metrics.updated_at = datetime.utcnow()
        except Exception as e:
            logger.warning("Failed to update metrics from PositionManager: %s", e)

    def update_from_system_state(self, state: dict) -> None:
        """Pull metrics from system state."""
        if not state:
            return

        self._metrics.current_equity = state.get(
            "current_equity", self._metrics.current_equity
        )
        self._metrics.daily_pnl = state.get("daily_pnl", 0)
        self._metrics.daily_pnl_pct = state.get("daily_pnl_pct", 0)
        self._metrics.weekly_pnl = state.get("weekly_pnl", 0)
        self._metrics.weekly_pnl_pct = state.get("weekly_pnl_pct", 0)
        self._metrics.drawdown_pct = state.get("drawdown_pct", 0)
        self._metrics.portfolio_heat_pct = state.get("portfolio_heat", 0)

    def update_equity(
        self,
        current: float,
        starting: float,
        peak: float,
    ) -> None:
        """Update equity metrics."""
        self._metrics.current_equity = current
        self._metrics.starting_equity = starting
        self._metrics.peak_equity = max(peak, current)

        # Calculate drawdown
        if self._metrics.peak_equity > 0:
            self._metrics.drawdown = self._metrics.peak_equity - current
            self._metrics.drawdown_pct = (
                self._metrics.drawdown / self._metrics.peak_equity
            ) * 100

            if self._metrics.drawdown_pct > self._metrics.max_drawdown_pct:
                self._metrics.max_drawdown = self._metrics.drawdown
                self._metrics.max_drawdown_pct = self._metrics.drawdown_pct

        # Calculate total P&L
        self._metrics.total_pnl = current - starting
        if starting > 0:
            self._metrics.total_pnl_pct = (self._metrics.total_pnl / starting) * 100

        self._metrics.updated_at = datetime.utcnow()

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade."""
        self._trade_history.append(trade)

        pnl = trade.get("pnl", 0)

        self._metrics.total_trades += 1

        if pnl > 0:
            self._metrics.winning_trades += 1
            self._metrics.gross_profit += pnl
        elif pnl < 0:
            self._metrics.losing_trades += 1
            self._metrics.gross_loss += abs(pnl)

        # Recalculate averages
        if self._metrics.total_trades > 0:
            self._metrics.win_rate = (
                self._metrics.winning_trades / self._metrics.total_trades
            ) * 100

        if self._metrics.winning_trades > 0:
            self._metrics.avg_win = (
                self._metrics.gross_profit / self._metrics.winning_trades
            )

        if self._metrics.losing_trades > 0:
            self._metrics.avg_loss = (
                self._metrics.gross_loss / self._metrics.losing_trades
            )

        if self._metrics.gross_loss > 0:
            self._metrics.profit_factor = (
                self._metrics.gross_profit / self._metrics.gross_loss
            )

        self._metrics.avg_trade = (
            self._metrics.gross_profit - self._metrics.gross_loss
        ) / self._metrics.total_trades

        self._metrics.last_trade_at = datetime.utcnow()
        self._metrics.updated_at = datetime.utcnow()

    def update_edge_performance(self, edge_type: str, stats: dict) -> None:
        """Update performance for specific edge."""
        self._metrics.edge_performance[edge_type] = {
            "trades": stats.get("trades", 0),
            "win_rate": stats.get("win_rate", 0),
            "avg_pnl": stats.get("average_pnl", 0),
            "total_pnl": stats.get("total_pnl", 0),
            "is_healthy": stats.get("is_healthy", True),
        }

    def get_metrics(self) -> TradingMetrics:
        """Get current metrics."""
        return self._metrics

    def get_metrics_dict(self) -> dict:
        """Get metrics as dictionary."""
        return self._metrics.to_dict()

    def get_edge_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all edge performances."""
        return self._metrics.edge_performance


# Convenience function matching original stub signature
def get_portfolio_metrics() -> Dict[str, Any]:
    """Return current portfolio metrics."""
    collector = MetricsCollector()
    return collector.get_metrics_dict()
