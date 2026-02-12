"""
API routes for positions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException

from nexus.storage.service import get_storage_service

router = APIRouter(prefix="/positions", tags=["positions"])

_position_manager = None


def set_position_manager(pm):
    global _position_manager
    _position_manager = pm


def _position_to_dict(p):
    """Convert Position to API dict."""
    return {
        "id": p.position_id,
        "symbol": p.symbol,
        "direction": getattr(p.direction, "value", p.direction),
        "size": p.size,
        "entry_price": p.entry_price,
        "current_price": p.current_price,
        "unrealized_pnl": p.unrealized_pnl,
        "r_multiple": p.r_multiple,
        "status": getattr(p.status, "value", p.status),
    }


@router.get("")
async def list_positions(status: Optional[str] = None, symbol: Optional[str] = None):
    """List positions."""
    if not _position_manager:
        storage = get_storage_service()
        state = await storage.get_system_state()
        positions = state.get("open_positions", []) if state else []
        return {"positions": positions, "count": len(positions), "source": "database"}

    positions = []
    for p in _position_manager._positions.values():
        p_status = getattr(p.status, "value", p.status)
        if status == "open" and p_status != "OPEN":
            continue
        if status == "closed" and p_status == "OPEN":
            continue
        if symbol and p.symbol != symbol:
            continue
        positions.append(_position_to_dict(p))

    metrics = _position_manager.get_portfolio_metrics()
    return {
        "positions": positions,
        "count": len(positions),
        "portfolio_heat": getattr(metrics, "portfolio_heat", 0),
        "source": "live",
    }


@router.get("/summary")
async def get_positions_summary():
    """Get portfolio summary."""
    if not _position_manager:
        storage = get_storage_service()
        state = await storage.get_system_state()
        if not state:
            return {"error": "No system state available"}
        return {
            "open_positions": len(state.get("open_positions", [])),
            "portfolio_heat": state.get("portfolio_heat", 0),
            "current_equity": state.get("current_equity", 0),
            "source": "database",
        }
    metrics = _position_manager.get_portfolio_metrics()
    return {
        "portfolio_heat": getattr(metrics, "portfolio_heat", 0),
        "open_positions": getattr(metrics, "open_positions", 0),
        "total_unrealized_pnl": getattr(metrics, "total_unrealized_pnl", 0),
        "total_realized_pnl": getattr(metrics, "total_realized_pnl", 0),
        "source": "live",
    }


@router.get("/{symbol}")
async def get_position(symbol: str):
    """Get position by symbol."""
    if not _position_manager:
        storage = get_storage_service()
        state = await storage.get_system_state()
        positions = state.get("open_positions", []) if state else []
        for pos in positions:
            if isinstance(pos, dict) and pos.get("symbol") == symbol:
                return {"symbol": symbol, "size": pos.get("size", 0), "unrealized_pnl": pos.get("unrealized_pnl", 0)}
        raise HTTPException(status_code=503, detail="Position manager not available; use database state or start execution")
    open_positions = _position_manager.get_open_positions_for_symbol(symbol)
    if not open_positions:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    position = open_positions[0]
    return {"symbol": position.symbol, "size": position.size, "unrealized_pnl": position.unrealized_pnl}
