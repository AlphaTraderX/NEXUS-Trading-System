"""
API routes for settings (read-only or safe overrides).
"""

from fastapi import APIRouter

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
async def get_settings():
    """Get current NEXUS settings (non-sensitive)."""
    return {"mode": "conservative", "paper_trading": True}
