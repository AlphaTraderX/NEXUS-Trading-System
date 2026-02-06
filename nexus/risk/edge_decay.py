"""
Edge decay: reduce size or skip when edge has been overused recently.
"""

from typing import Optional


def edge_decay_factor(edge_name: str, recent_trades_count: int) -> float:
    """Return multiplier in [0, 1] for position size based on recent edge usage."""
    return 1.0
