"""
API routes for trading signals.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from nexus.storage.service import get_storage_service

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
async def list_signals(
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    edge: Optional[str] = Query(None),
):
    """List recent signals with optional filters."""
    storage = get_storage_service()
    signals = await storage.get_recent_signals(limit=limit)

    if status:
        signals = [s for s in signals if s.get("status") == status]
    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]
    if edge:
        signals = [s for s in signals if s.get("primary_edge") == edge]

    return {"signals": signals, "count": len(signals)}


@router.get("/pending")
async def list_pending_signals():
    """Get all pending signals."""
    storage = get_storage_service()
    signals = await storage.get_pending_signals()
    return {"signals": signals, "count": len(signals)}


@router.get("/stats")
async def get_signal_stats():
    """Get signal statistics."""
    storage = get_storage_service()
    signals = await storage.get_recent_signals(limit=100)

    by_status, by_edge, by_tier = {}, {}, {}
    for s in signals:
        by_status[s.get("status", "unknown")] = by_status.get(s.get("status", "unknown"), 0) + 1
        by_edge[s.get("primary_edge", "unknown")] = by_edge.get(s.get("primary_edge", "unknown"), 0) + 1
        by_tier[s.get("tier", "unknown")] = by_tier.get(s.get("tier", "unknown"), 0) + 1

    return {"total": len(signals), "by_status": by_status, "by_edge": by_edge, "by_tier": by_tier}


@router.get("/{signal_id}")
async def get_signal(signal_id: str):
    """Get signal by ID."""
    storage = get_storage_service()
    signal = await storage.get_signal(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    return signal
