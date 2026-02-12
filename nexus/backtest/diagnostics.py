"""
Edge diagnostics and A/B testing utilities.

Provides:
- diagnose_edge(): Analyze why an edge is underperforming
- should_revert(): Check if experimental results warrant a revert
- test_reversed_edge(): Test if flipping LONG↔SHORT makes a loser profitable
- save_baseline() / load_baseline(): Persist results for comparison
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nexus.backtest.trade_simulator import ExitReason, SimulatedTrade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge diagnosis
# ---------------------------------------------------------------------------

def diagnose_edge(trades: List[SimulatedTrade]) -> Dict[str, Any]:
    """Analyze why an edge is underperforming.

    Returns a diagnostic dict with breakdowns by time, direction, exit type,
    and score tier.
    """
    if not trades:
        return {"total_trades": 0, "error": "No trades to diagnose"}

    total = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    results: Dict[str, Any] = {
        "total_trades": total,
        "win_rate": len(winners) / total,
        "avg_winner": float(np.mean([t.net_pnl for t in winners])) if winners else 0.0,
        "avg_loser": float(np.mean([t.net_pnl for t in losers])) if losers else 0.0,
        "profit_factor": (
            abs(sum(t.net_pnl for t in winners) / sum(t.net_pnl for t in losers))
            if losers and sum(t.net_pnl for t in losers) != 0
            else float("inf")
        ),
    }

    # Exit reason distribution
    exit_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        reason = t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason)
        exit_counts[reason] += 1
    results["exit_distribution"] = dict(exit_counts)
    results["stop_loss_pct"] = exit_counts.get("stop_loss", 0) / total
    results["target_hit_pct"] = exit_counts.get("take_profit", 0) / total

    # Direction analysis
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    results["long_count"] = len(long_trades)
    results["short_count"] = len(short_trades)
    results["long_win_rate"] = (
        sum(1 for t in long_trades if t.net_pnl > 0) / len(long_trades)
        if long_trades else 0.0
    )
    results["short_win_rate"] = (
        sum(1 for t in short_trades if t.net_pnl > 0) / len(short_trades)
        if short_trades else 0.0
    )

    # By month
    by_month: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        month_key = t.entry_time.strftime("%Y-%m")
        by_month[month_key].append(t.net_pnl)
    results["by_month"] = {
        k: {"trades": len(v), "pnl": sum(v), "win_rate": sum(1 for x in v if x > 0) / len(v)}
        for k, v in sorted(by_month.items())
    }

    # By day of week (0=Mon, 4=Fri)
    by_dow: Dict[int, List[float]] = defaultdict(list)
    for t in trades:
        by_dow[t.entry_time.weekday()].append(t.net_pnl)
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    results["by_day_of_week"] = {
        dow_names.get(k, str(k)): {
            "trades": len(v),
            "pnl": sum(v),
            "win_rate": sum(1 for x in v if x > 0) / len(v),
        }
        for k, v in sorted(by_dow.items())
    }

    # By score tier
    by_tier: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        tier = getattr(t, "score_tier", "C")
        by_tier[tier].append(t.net_pnl)
    results["by_tier"] = {
        k: {"trades": len(v), "pnl": sum(v), "win_rate": sum(1 for x in v if x > 0) / len(v)}
        for k, v in sorted(by_tier.items())
    }

    # By symbol
    by_symbol: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t.net_pnl)
    results["by_symbol"] = {
        k: {"trades": len(v), "pnl": sum(v), "win_rate": sum(1 for x in v if x > 0) / len(v)}
        for k, v in sorted(by_symbol.items())
    }

    return results


# ---------------------------------------------------------------------------
# Revert checking
# ---------------------------------------------------------------------------

def should_revert(
    baseline: Dict[str, Any],
    experimental: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check if experimental results warrant a revert.

    Conditions that trigger revert:
    1. MaxDD on any edge exceeds 15%
    2. Profit factor drops below 1.2
    3. Total P&L drops more than 10% from baseline
    4. Win rate drops below 48%
    """
    reasons = []

    if experimental.get("max_dd_pct", 0) > 15:
        reasons.append(
            f"MaxDD {experimental['max_dd_pct']:.1f}% exceeds 15%"
        )

    if experimental.get("profit_factor", 0) < 1.2:
        reasons.append(
            f"PF {experimental['profit_factor']:.2f} below 1.2"
        )

    baseline_pnl = baseline.get("total_pnl", 0)
    exp_pnl = experimental.get("total_pnl", 0)
    if baseline_pnl > 0:
        pnl_change = (exp_pnl - baseline_pnl) / baseline_pnl
        if pnl_change < -0.10:
            reasons.append(f"P&L dropped {pnl_change * 100:.1f}%")

    if experimental.get("win_rate", 0) < 48.0:
        reasons.append(
            f"Win rate {experimental['win_rate']:.1f}% below 48%"
        )

    if reasons:
        return True, "REVERT RECOMMENDED: " + "; ".join(reasons)
    return False, "All checks passed"


# ---------------------------------------------------------------------------
# Signal reversal testing
# ---------------------------------------------------------------------------

async def test_reversed_edge(
    edge_type,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    starting_balance: float = 10_000.0,
) -> Dict[str, Any]:
    """Test if reversing a losing edge's signals makes it profitable.

    For edges with PF < 1.0 (wrong more than right), flipping LONG↔SHORT
    may produce a winner. E.g., NY_OPEN PF 0.37 means it's wrong 63% of
    the time — reversed = right 63%.

    This works by temporarily removing the edge from DISABLED_EDGES,
    running the backtest, then flipping all trade directions in the results
    to compute reversed statistics.
    """
    from nexus.backtest.engine import BacktestEngine
    from nexus.backtest.statistics import StatisticsCalculator

    engine = BacktestEngine(
        starting_balance=starting_balance,
        risk_per_trade=1.0,
        use_score_sizing=False,
    )

    # Temporarily enable the edge
    was_disabled = edge_type in engine.DISABLED_EDGES
    if was_disabled:
        engine.DISABLED_EDGES = engine.DISABLED_EDGES - {edge_type}

    try:
        result = await engine.run_edge_backtest(
            edge_type=edge_type,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        # Restore disabled state
        if was_disabled:
            engine.DISABLED_EDGES = engine.DISABLED_EDGES | {edge_type}

    original_trades = result.trades
    if not original_trades:
        return {
            "edge": edge_type.value,
            "original_trades": 0,
            "error": "No trades generated (check data availability)",
        }

    # Reverse all trades: flip direction and invert P&L
    reversed_trades: List[SimulatedTrade] = []
    for t in original_trades:
        reversed_dir = "short" if t.direction == "long" else "long"
        reversed_trades.append(SimulatedTrade(
            opportunity_id=t.opportunity_id,
            symbol=t.symbol,
            direction=reversed_dir,
            entry_time=t.entry_time,
            entry_price=t.entry_price,
            exit_time=t.exit_time,
            exit_price=t.exit_price,
            exit_reason=t.exit_reason,
            position_size=t.position_size,
            gross_pnl=-t.gross_pnl,
            costs=t.costs,
            net_pnl=-t.gross_pnl - t.costs,
            net_pnl_pct=-t.net_pnl_pct - (t.costs / starting_balance * 100 * 2),
            hold_duration=t.hold_duration,
            primary_edge=t.primary_edge,
            score=t.score,
            score_tier=t.score_tier,
        ))

    # Calculate stats for both original and reversed
    calc = StatisticsCalculator()
    original_stats = result.statistics

    # Recalculate reversed P&L percentages properly
    for rt in reversed_trades:
        rt.net_pnl_pct = (rt.net_pnl / starting_balance) * 100

    reversed_stats = calc.calculate(
        trades=reversed_trades,
        edge_type=f"{edge_type.value}_REVERSED",
        symbol=original_stats.symbol,
        timeframe=original_stats.timeframe,
        test_period=original_stats.test_period,
        starting_balance=starting_balance,
    )

    return {
        "edge": edge_type.value,
        "original": {
            "trades": original_stats.total_trades,
            "win_rate": original_stats.win_rate,
            "profit_factor": original_stats.profit_factor,
            "total_pnl_pct": original_stats.total_pnl_pct,
            "max_dd_pct": original_stats.max_drawdown_pct,
            "sharpe": original_stats.sharpe_ratio,
            "verdict": original_stats.verdict,
        },
        "reversed": {
            "trades": reversed_stats.total_trades,
            "win_rate": reversed_stats.win_rate,
            "profit_factor": reversed_stats.profit_factor,
            "total_pnl_pct": reversed_stats.total_pnl_pct,
            "max_dd_pct": reversed_stats.max_drawdown_pct,
            "sharpe": reversed_stats.sharpe_ratio,
            "verdict": reversed_stats.verdict,
        },
        "reversal_profitable": reversed_stats.profit_factor > 1.2,
        "recommendation": (
            f"REVERSE {edge_type.value}: PF {original_stats.profit_factor:.2f} → "
            f"{reversed_stats.profit_factor:.2f}"
            if reversed_stats.profit_factor > 1.2
            else f"KEEP DISABLED: reversed PF {reversed_stats.profit_factor:.2f} still weak"
        ),
    }


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def save_baseline(results: Dict[str, Any], filepath: str) -> None:
    """Save backtest results as a JSON baseline for comparison."""
    serializable = {}
    for edge_name, result in results.items():
        if result is None:
            serializable[edge_name] = None
            continue
        s = result.statistics
        serializable[edge_name] = {
            "total_trades": s.total_trades,
            "winners": s.winners,
            "losers": s.losers,
            "win_rate": s.win_rate,
            "profit_factor": s.profit_factor,
            "total_pnl": s.total_pnl,
            "total_pnl_pct": s.total_pnl_pct,
            "max_drawdown_pct": s.max_drawdown_pct,
            "sharpe_ratio": s.sharpe_ratio,
            "expected_value_pct": s.expected_value_pct,
            "p_value": s.p_value,
            "verdict": s.verdict,
            "symbol": s.symbol,
            "timeframe": s.timeframe,
        }

    Path(filepath).write_text(json.dumps(serializable, indent=2))
    logger.info("Baseline saved to %s", filepath)


def load_baseline(filepath: str) -> Dict[str, Any]:
    """Load a previously saved baseline."""
    return json.loads(Path(filepath).read_text())


def compare_to_baseline(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Tuple[bool, str]:
    """Compare current results to a saved baseline.

    Returns (all_passed, report_string).
    """
    lines = []
    any_revert = False

    lines.append(f"{'Edge':<22} {'Trades':>7} {'PF':>9} {'P&L%':>10} {'DD%':>9}  {'Status'}")
    lines.append("-" * 75)

    for edge_name in sorted(set(list(baseline.keys()) + list(current.keys()))):
        base = baseline.get(edge_name)
        curr = current.get(edge_name)

        if base is None or curr is None:
            lines.append(f"  {edge_name:<22} {'--':>7} {'--':>9} {'--':>10} {'--':>9}  SKIP")
            continue

        if base.get("total_trades", 0) == 0 and curr.get("total_trades", 0) == 0:
            continue

        trades_delta = curr.get("total_trades", 0) - base.get("total_trades", 0)
        pf_delta = curr.get("profit_factor", 0) - base.get("profit_factor", 0)
        pnl_delta = curr.get("total_pnl_pct", 0) - base.get("total_pnl_pct", 0)
        dd_delta = curr.get("max_drawdown_pct", 0) - base.get("max_drawdown_pct", 0)

        # Check revert conditions per edge
        revert, reason = should_revert(base, curr)
        status = "REVERT" if revert else "OK"
        if revert:
            any_revert = True

        lines.append(
            f"  {edge_name:<22} {trades_delta:>+7} {pf_delta:>+8.2f} "
            f"{pnl_delta:>+9.1f}% {dd_delta:>+8.1f}%  {status}"
        )

    report = "\n".join(lines)
    overall = not any_revert
    return overall, report
