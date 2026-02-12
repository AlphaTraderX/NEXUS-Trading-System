"""
API routes for settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nexus.config.settings import settings

router = APIRouter(prefix="/settings", tags=["settings"])


class ModeUpdate(BaseModel):
    mode: str


@router.get("")
async def get_settings():
    """Get current settings (non-sensitive)."""
    return {
        "mode": settings.mode,
        "paper_trading": settings.paper_trading,
        "starting_balance": settings.starting_balance,
        "currency": settings.currency,
        "base_risk_pct": settings.base_risk_pct,
        "max_risk_pct": settings.max_risk_pct,
        "base_heat_limit": settings.base_heat_limit,
        "max_positions": settings.max_positions,
        "min_score_to_trade": settings.min_score_to_trade,
        "daily_loss_stop": settings.daily_loss_stop,
        "max_drawdown": settings.max_drawdown,
    }


@router.get("/mode")
async def get_mode():
    """Get current trading mode."""
    return {"mode": settings.mode, "multipliers": settings.get_mode_multipliers()}


@router.put("/mode")
async def update_mode(update: ModeUpdate):
    """Update trading mode."""
    valid = ["conservative", "standard", "aggressive", "maximum"]
    if update.mode not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be: {valid}")
    return {
        "mode": update.mode,
        "message": f"Mode would be updated to {update.mode}",
        "note": "Runtime changes not yet implemented",
    }


@router.get("/edges")
async def get_edge_settings():
    """Get edge configuration."""
    from nexus.core.enums import EdgeType
    return {
        "enabled_edges": [e.value for e in EdgeType],
        "edge_weights": {},  # EDGE_WEIGHTS not in codebase; expose if added
    }


@router.get("/brokers")
async def get_broker_status():
    """Get broker configuration status."""
    return {
        "ibkr": {"configured": bool(settings.ibkr_host), "host": settings.ibkr_host, "port": settings.ibkr_port},
        "ig": {"configured": bool(settings.ig_api_key)},
        "oanda": {"configured": bool(settings.oanda_api_key)},
    }
