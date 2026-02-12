"""
API routes for performance metrics.
"""

from typing import Optional
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter, Query

from nexus.storage.service import get_storage_service

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("")
async def get_performance():
    """Get current portfolio performance."""
    storage = get_storage_service()
    state = await storage.get_system_state()

    if not state:
        return {"equity": 0.0, "daily_pnl": 0.0, "drawdown_pct": 0.0, "win_rate": 0.0, "status": "no_data"}

    trades = await storage.get_recent_trades(limit=100)
    winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
    total = len([t for t in trades if t.get("pnl") is not None])

    return {
        "equity": state.get("current_equity", 0),
        "daily_pnl": state.get("daily_pnl", 0),
        "daily_pnl_pct": state.get("daily_pnl_pct", 0),
        "weekly_pnl": state.get("weekly_pnl", 0),
        "drawdown_pct": state.get("drawdown_pct", 0),
        "portfolio_heat": state.get("portfolio_heat", 0),
        "win_rate": (winners / total * 100) if total > 0 else 0,
        "total_trades": total,
        "circuit_breaker": state.get("circuit_breaker_status", "unknown"),
        "kill_switch": state.get("kill_switch_active", False),
    }


@router.get("/trades")
async def get_trades(limit: int = Query(50, ge=1, le=200), edge: Optional[str] = None):
    """Get recent trades."""
    storage = get_storage_service()
    if edge:
        from nexus.core.enums import EdgeType
        try:
            trades = await storage.get_trades_by_edge(EdgeType(edge), limit=limit)
        except ValueError:
            trades = []
    else:
        trades = await storage.get_recent_trades(limit=limit)
    return {"trades": trades, "count": len(trades)}


@router.get("/edges")
async def get_edge_performance():
    """Get performance by edge type."""
    storage = get_storage_service()
    trades = await storage.get_recent_trades(limit=500)
    signals = await storage.get_recent_signals(limit=500)
    signal_edges = {s["signal_id"]: s.get("primary_edge") for s in signals}

    edge_stats = {}
    for trade in trades:
        edge = signal_edges.get(trade.get("signal_id"), "unknown")
        pnl = trade.get("pnl", 0)
        if edge not in edge_stats:
            edge_stats[edge] = {"trades": 0, "winners": 0, "total_pnl": 0}
        edge_stats[edge]["trades"] += 1
        if pnl and pnl > 0:
            edge_stats[edge]["winners"] += 1
        if pnl:
            edge_stats[edge]["total_pnl"] += pnl

    for stats in edge_stats.values():
        stats["win_rate"] = (stats["winners"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        stats["avg_pnl"] = stats["total_pnl"] / stats["trades"] if stats["trades"] > 0 else 0

    return {"edges": edge_stats}


@router.get("/history")
async def get_performance_history(days: int = Query(30, ge=1, le=365)):
    """Get historical performance series."""
    storage = get_storage_service()
    trades = await storage.get_recent_trades(limit=1000)

    daily = defaultdict(lambda: {"pnl": 0, "trades": 0, "winners": 0})
    for trade in trades:
        if not trade.get("entry_time"):
            continue
        try:
            entry = trade["entry_time"]
            if hasattr(entry, "replace"):
                dt = datetime.fromisoformat(entry.replace("Z", "+00:00"))
            else:
                dt = entry
            date_key = dt.strftime("%Y-%m-%d")
            pnl = trade.get("pnl", 0) or 0
            daily[date_key]["pnl"] += pnl
            daily[date_key]["trades"] += 1
            if pnl > 0:
                daily[date_key]["winners"] += 1
        except Exception:
            continue

    series = [
        {
            "date": d,
            **daily[d],
            "win_rate": (daily[d]["winners"] / daily[d]["trades"] * 100) if daily[d]["trades"] else 0,
        }
        for d in sorted(daily.keys())[-days:]
    ]
    return {"series": series, "days": len(series)}
