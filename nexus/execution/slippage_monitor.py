"""
Slippage monitoring and reporting.
"""

from typing import Optional


def record_fill(
    order_id: str,
    expected_price: float,
    fill_price: float,
    side: str,
) -> None:
    """Record a fill for slippage tracking."""
    pass


def get_avg_slippage_bps(symbol: Optional[str] = None) -> float:
    """Return average slippage in basis points."""
    return 0.0
