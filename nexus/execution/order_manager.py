"""
Order submission, tracking, and cancellation.
"""

from typing import Optional, List, Any


async def submit_order(signal: Any, paper: bool = True) -> Optional[str]:
    """Submit order; returns order id or None."""
    return None


async def cancel_order(order_id: str) -> bool:
    """Cancel order by id."""
    return False


def get_open_orders() -> List[Any]:
    """Return list of open orders."""
    return []
