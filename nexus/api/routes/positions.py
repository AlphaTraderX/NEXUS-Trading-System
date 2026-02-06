"""
API routes for positions.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("")
async def list_positions():
    """List open positions."""
    return {"positions": []}


@router.get("/{symbol}")
async def get_position(symbol: str):
    """Get position by symbol."""
    return {"symbol": symbol, "quantity": 0, "side": ""}
