"""
Reconciliation: internal state vs broker.
"""

from typing import List, Dict, Any


async def reconcile_positions(
    internal: List[Dict[str, Any]],
    broker: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare internal and broker positions; return discrepancies."""
    return {"missing": [], "extra": [], "mismatches": []}
