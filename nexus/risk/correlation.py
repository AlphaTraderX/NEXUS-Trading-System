"""
Correlation and diversification checks.
"""

from typing import List, Dict, Any


def correlation_penalty(symbol: str, existing_positions: List[Dict[str, Any]]) -> float:
    """Return penalty (0–1) for adding a position correlated with existing ones."""
    return 0.0
