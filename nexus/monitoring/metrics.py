"""
Metrics collection (PnL, drawdown, win rate, etc.).
"""

from typing import Dict, Any


def get_portfolio_metrics() -> Dict[str, Any]:
    """Return current portfolio metrics."""
    return {
        "equity": 0.0,
        "drawdown_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
    }
