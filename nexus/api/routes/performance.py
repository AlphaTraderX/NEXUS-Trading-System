"""
API routes for performance metrics.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("")
async def get_performance():
    """Get portfolio performance metrics."""
    return {"equity": 0.0, "drawdown_pct": 0.0, "win_rate": 0.0}


@router.get("/history")
async def get_performance_history():
    """Get historical performance series."""
    return {"series": []}
