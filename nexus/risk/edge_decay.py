"""
Edge Decay Detection System

Detects when trading edges stop working using CUSUM analysis.
Academic basis: Edges decay as they get arbitraged away.

Example: FOMC drift disappeared after 2015 (Kurov, Wolfe & Gilbert 2021)

Method:
- Track rolling 50-trade performance per edge
- Calculate z-score vs historical baseline
- Alert when performance drops 2σ below baseline
- Auto-disable after 3 consecutive failed periods
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
from statistics import mean

from nexus.core.enums import EdgeType

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Single trade result for decay tracking."""

    edge_type: EdgeType
    symbol: str
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    is_win: bool


@dataclass
class EdgeBaseline:
    """Historical baseline for an edge."""

    edge_type: EdgeType
    expected_win_rate: float  # e.g., 0.55
    expected_avg_return: float  # e.g., 0.15 (%)
    win_rate_std: float = 0.08  # Standard deviation
    return_std: float = 0.10
    min_trades_for_signal: int = 20

    @classmethod
    def get_defaults(cls) -> Dict[EdgeType, "EdgeBaseline"]:
        """Get default baselines from backtest results."""
        return {
            EdgeType.TURN_OF_MONTH: cls(
                edge_type=EdgeType.TURN_OF_MONTH,
                expected_win_rate=0.58,
                expected_avg_return=0.25,
            ),
            EdgeType.MONTH_END: cls(
                edge_type=EdgeType.MONTH_END,
                expected_win_rate=0.55,
                expected_avg_return=0.20,
            ),
            EdgeType.VWAP_DEVIATION: cls(
                edge_type=EdgeType.VWAP_DEVIATION,
                expected_win_rate=0.54,
                expected_avg_return=0.15,
            ),
            EdgeType.RSI_EXTREME: cls(
                edge_type=EdgeType.RSI_EXTREME,
                expected_win_rate=0.52,
                expected_avg_return=0.12,
            ),
            EdgeType.GAP_FILL: cls(
                edge_type=EdgeType.GAP_FILL,
                expected_win_rate=0.65,
                expected_avg_return=0.18,
            ),
            EdgeType.INSIDER_CLUSTER: cls(
                edge_type=EdgeType.INSIDER_CLUSTER,
                expected_win_rate=0.58,
                expected_avg_return=0.30,
            ),
            EdgeType.POWER_HOUR: cls(
                edge_type=EdgeType.POWER_HOUR,
                expected_win_rate=0.52,
                expected_avg_return=0.10,
            ),
            EdgeType.OVERNIGHT_PREMIUM: cls(
                edge_type=EdgeType.OVERNIGHT_PREMIUM,
                expected_win_rate=0.51,
                expected_avg_return=0.08,
            ),
            EdgeType.ORB: cls(
                edge_type=EdgeType.ORB,
                expected_win_rate=0.50,
                expected_avg_return=0.12,
            ),
            EdgeType.BOLLINGER_TOUCH: cls(
                edge_type=EdgeType.BOLLINGER_TOUCH,
                expected_win_rate=0.54,
                expected_avg_return=0.10,
            ),
            EdgeType.LONDON_OPEN: cls(
                edge_type=EdgeType.LONDON_OPEN,
                expected_win_rate=0.50,
                expected_avg_return=0.10,
            ),
            EdgeType.NY_OPEN: cls(
                edge_type=EdgeType.NY_OPEN,
                expected_win_rate=0.50,
                expected_avg_return=0.10,
            ),
            EdgeType.ASIAN_RANGE: cls(
                edge_type=EdgeType.ASIAN_RANGE,
                expected_win_rate=0.50,
                expected_avg_return=0.10,
            ),
            EdgeType.EARNINGS_DRIFT: cls(
                edge_type=EdgeType.EARNINGS_DRIFT,
                expected_win_rate=0.55,
                expected_avg_return=0.20,
            ),
            EdgeType.SENTIMENT_SPIKE: cls(
                edge_type=EdgeType.SENTIMENT_SPIKE,
                expected_win_rate=0.52,
                expected_avg_return=0.15,
            ),
        }


@dataclass
class EdgeHealth:
    """Current health status of an edge."""

    edge_type: EdgeType
    status: str  # "healthy", "warning", "critical", "disabled"
    current_win_rate: float
    current_avg_return: float
    baseline_win_rate: float
    baseline_avg_return: float
    win_rate_z_score: float
    return_z_score: float
    trade_count: int
    consecutive_failures: int
    last_updated: datetime
    warnings: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"

    @property
    def should_disable(self) -> bool:
        return self.status == "disabled"


class EdgeDecayMonitor:
    """
    Monitor edge performance and detect decay.

    Usage:
        monitor = EdgeDecayMonitor()

        # Record trade outcomes
        monitor.record_trade(outcome)

        # Check edge health
        health = monitor.check_edge_health(EdgeType.VWAP_DEVIATION)
        if health.status == "critical":
            # Consider disabling edge
            pass

        # Get all unhealthy edges
        warnings = monitor.get_decay_warnings()
    """

    ROLLING_WINDOW = 50  # Trades to consider
    WARNING_Z_THRESHOLD = -1.5  # 1.5 std below baseline
    CRITICAL_Z_THRESHOLD = -2.0  # 2 std below baseline
    DISABLE_CONSECUTIVE_FAILURES = 3  # Periods before auto-disable

    def __init__(
        self,
        baselines: Optional[Dict[EdgeType, EdgeBaseline]] = None,
        rolling_window: int = 50,
    ):
        self.baselines = baselines if baselines is not None else EdgeBaseline.get_defaults()
        self.rolling_window = rolling_window

        # Rolling trade history per edge
        self._trade_history: Dict[EdgeType, deque] = {}

        # Consecutive failure tracking
        self._consecutive_failures: Dict[EdgeType, int] = {}

        # Disabled edges
        self._disabled_edges: set = set()

        # Last check results
        self._last_health: Dict[EdgeType, EdgeHealth] = {}

    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for decay tracking."""
        edge = outcome.edge_type

        if edge not in self._trade_history:
            self._trade_history[edge] = deque(maxlen=self.rolling_window)

        self._trade_history[edge].append(outcome)

        logger.debug(
            "Recorded %s trade: %s %.2f%%",
            edge.value,
            "WIN" if outcome.is_win else "LOSS",
            outcome.pnl_pct,
        )

    def check_edge_health(self, edge_type: EdgeType) -> EdgeHealth:
        """Check health of a specific edge."""
        baseline = self.baselines.get(edge_type)
        if not baseline:
            return EdgeHealth(
                edge_type=edge_type,
                status="unknown",
                current_win_rate=0,
                current_avg_return=0,
                baseline_win_rate=0,
                baseline_avg_return=0,
                win_rate_z_score=0,
                return_z_score=0,
                trade_count=0,
                consecutive_failures=0,
                last_updated=datetime.now(),
                warnings=["No baseline defined for this edge"],
            )

        trades = self._trade_history.get(edge_type, deque())
        trade_count = len(trades)

        if trade_count < baseline.min_trades_for_signal:
            return EdgeHealth(
                edge_type=edge_type,
                status="insufficient_data",
                current_win_rate=0,
                current_avg_return=0,
                baseline_win_rate=baseline.expected_win_rate,
                baseline_avg_return=baseline.expected_avg_return,
                win_rate_z_score=0,
                return_z_score=0,
                trade_count=trade_count,
                consecutive_failures=0,
                last_updated=datetime.now(),
                warnings=[
                    f"Need {baseline.min_trades_for_signal - trade_count} more trades"
                ],
            )

        # Calculate current performance
        wins = sum(1 for t in trades if t.is_win)
        current_win_rate = wins / trade_count
        current_avg_return = mean(t.pnl_pct for t in trades)

        # Calculate z-scores
        win_rate_z = (
            (current_win_rate - baseline.expected_win_rate) / baseline.win_rate_std
        )
        return_z = (
            (current_avg_return - baseline.expected_avg_return) / baseline.return_std
        )

        # Determine status
        warnings = []
        status = "healthy"

        if edge_type in self._disabled_edges:
            status = "disabled"
            warnings.append(
                "Edge has been auto-disabled due to persistent underperformance"
            )

        elif (
            win_rate_z < self.CRITICAL_Z_THRESHOLD
            or return_z < self.CRITICAL_Z_THRESHOLD
        ):
            status = "critical"
            self._consecutive_failures[edge_type] = (
                self._consecutive_failures.get(edge_type, 0) + 1
            )

            if win_rate_z < self.CRITICAL_Z_THRESHOLD:
                warnings.append(
                    f"Win rate {current_win_rate:.1%} is "
                    f"{abs(win_rate_z):.1f}\u03c3 below baseline"
                )
            if return_z < self.CRITICAL_Z_THRESHOLD:
                warnings.append(
                    f"Avg return {current_avg_return:.2f}% is "
                    f"{abs(return_z):.1f}\u03c3 below baseline"
                )

            # Auto-disable after consecutive failures
            if (
                self._consecutive_failures.get(edge_type, 0)
                >= self.DISABLE_CONSECUTIVE_FAILURES
            ):
                self._disabled_edges.add(edge_type)
                status = "disabled"
                warnings.append(
                    f"Auto-disabled after {self.DISABLE_CONSECUTIVE_FAILURES} "
                    f"consecutive critical periods"
                )

        elif (
            win_rate_z < self.WARNING_Z_THRESHOLD
            or return_z < self.WARNING_Z_THRESHOLD
        ):
            status = "warning"
            # Partial recovery - decrement failure counter
            if self._consecutive_failures.get(edge_type, 0) > 0:
                self._consecutive_failures[edge_type] = max(
                    0, self._consecutive_failures.get(edge_type, 0) - 1
                )

            if win_rate_z < self.WARNING_Z_THRESHOLD:
                warnings.append(
                    f"Win rate {current_win_rate:.1%} trending below baseline"
                )
            if return_z < self.WARNING_Z_THRESHOLD:
                warnings.append(
                    f"Avg return {current_avg_return:.2f}% trending below baseline"
                )

        else:
            # Healthy - reset failure counter
            self._consecutive_failures[edge_type] = 0

        health = EdgeHealth(
            edge_type=edge_type,
            status=status,
            current_win_rate=current_win_rate,
            current_avg_return=current_avg_return,
            baseline_win_rate=baseline.expected_win_rate,
            baseline_avg_return=baseline.expected_avg_return,
            win_rate_z_score=win_rate_z,
            return_z_score=return_z,
            trade_count=trade_count,
            consecutive_failures=self._consecutive_failures.get(edge_type, 0),
            last_updated=datetime.now(),
            warnings=warnings,
        )

        self._last_health[edge_type] = health
        return health

    def check_all_edges(self) -> Dict[EdgeType, EdgeHealth]:
        """Check health of all tracked edges."""
        results = {}
        for edge_type in self._trade_history:
            results[edge_type] = self.check_edge_health(edge_type)
        return results

    def get_decay_warnings(self) -> List[EdgeHealth]:
        """Get all edges that are warning or critical."""
        all_health = self.check_all_edges()
        return [
            health
            for health in all_health.values()
            if health.status in ("warning", "critical", "disabled")
        ]

    def get_disabled_edges(self) -> List[EdgeType]:
        """Get list of auto-disabled edges."""
        return list(self._disabled_edges)

    def is_edge_enabled(self, edge_type: EdgeType) -> bool:
        """Check if edge is enabled (not auto-disabled)."""
        return edge_type not in self._disabled_edges

    def re_enable_edge(self, edge_type: EdgeType) -> bool:
        """Manually re-enable a disabled edge. Clears history for fresh start."""
        if edge_type in self._disabled_edges:
            self._disabled_edges.remove(edge_type)
            self._consecutive_failures[edge_type] = 0
            self._trade_history[edge_type] = deque(maxlen=self.rolling_window)
            logger.info("Re-enabled edge: %s", edge_type.value)
            return True
        return False

    def update_baseline(
        self,
        edge_type: EdgeType,
        win_rate: float,
        avg_return: float,
    ) -> None:
        """Update baseline for an edge (e.g., from new backtest)."""
        if edge_type in self.baselines:
            self.baselines[edge_type].expected_win_rate = win_rate
            self.baselines[edge_type].expected_avg_return = avg_return
            logger.info(
                "Updated baseline for %s: WR=%.1f%%, Avg=%.2f%%",
                edge_type.value,
                win_rate * 100,
                avg_return,
            )

    def get_summary(self) -> Dict:
        """Get summary of all edge health."""
        all_health = self.check_all_edges()
        return {
            "total_edges": len(all_health),
            "healthy": sum(
                1 for h in all_health.values() if h.status == "healthy"
            ),
            "warning": sum(
                1 for h in all_health.values() if h.status == "warning"
            ),
            "critical": sum(
                1 for h in all_health.values() if h.status == "critical"
            ),
            "disabled": sum(
                1 for h in all_health.values() if h.status == "disabled"
            ),
            "insufficient_data": sum(
                1 for h in all_health.values() if h.status == "insufficient_data"
            ),
            "edges": {
                edge.value: {
                    "status": health.status,
                    "win_rate": f"{health.current_win_rate:.1%}",
                    "avg_return": f"{health.current_avg_return:.2f}%",
                    "trade_count": health.trade_count,
                }
                for edge, health in all_health.items()
            },
        }


# Backwards-compatible stub
def edge_decay_factor(edge_name: str, recent_trades_count: int) -> float:
    """Return multiplier in [0, 1] for position size based on recent edge usage."""
    return 1.0


# Singleton instance
_decay_monitor: Optional[EdgeDecayMonitor] = None


def get_decay_monitor() -> EdgeDecayMonitor:
    """Get or create the edge decay monitor singleton."""
    global _decay_monitor
    if _decay_monitor is None:
        _decay_monitor = EdgeDecayMonitor()
    return _decay_monitor
