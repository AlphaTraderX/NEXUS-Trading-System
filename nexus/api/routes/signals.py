"""
API routes for signals.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
async def list_signals():
    """List recent signals."""
    return {"signals": []}


@router.get("/{signal_id}")
async def get_signal(signal_id: int):
    """Get signal by id."""
    return {"id": signal_id, "symbol": "", "direction": ""}
