"""
Health checks (DB, Redis, brokers, feeds).
"""

from typing import Dict, Any


async def check_health() -> Dict[str, Any]:
    """Run health checks; return dict of component -> status."""
    return {
        "database": "unknown",
        "redis": "unknown",
        "broker": "unknown",
    }
