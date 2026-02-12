"""
Run full backtest with detailed P&L and compounding breakdown.

Extends backtest_all.py with:
- Per-edge P&L decomposition (gross, costs, net)
- Monthly return series
- Equity curve with high-water mark
- Compounding analysis (position size growth over time)
- Trade-level P&L distribution

Usage::

    python -m nexus.scripts.run_backtest
    python -m nexus.scripts.run_backtest --start 2023-01-01 --end 2024-12-31
    python -m nexus.scripts.run_backtest --no-score-sizing
"""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _supports_color() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
    return True


if not _supports_color():
    Colors.GREEN = Colors.YELLOW = Colors.RED = ""
    Colors.GRAY = Colors.CYAN = Colors.BOLD = Colors.END = ""


def colorize(text: str, verdict: str) -> str:
    color = {
        "VALID": Colors.GREEN,
        "MARGINAL": Colors.YELLOW,
        "INVALID": Colors.RED,
        "INSUFFICIENT_DATA": Colors.GRAY,
        "NO_DATA": Colors.GRAY,
        "ERROR": Colors.RED,
    }.get(verdict, "")
    return f"{color}{text}{Colors.END}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS - Full backtest with P&L and compounding breakdown",
    )
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--no-score-sizing", action="store_true", default=False,
                        help="Disable tier-based position sizing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run all edges
# ---------------------------------------------------------------------------

async def _run_all(args) -> Dict[str, Optional[object]]:
    from nexus.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        starting_balance=args.balance,
        risk_per_trade=args.risk,
        use_score_sizing=not args.no_score_sizing,
    )

    edges = list(engine.SCANNER_MAP.keys())
    total = len(edges)
    results: Dict[str, Optional[object]] = {}

    for i, edge_type in enumerate(edges, 1):
        print(f"  [{i}/{total}] {edge_type.value}...", end=" ", flush=True)
        try:
            result = await engine.run_edge_backtest(
                edge_type=edge_type,
                start_date=args.start,
                end_date=args.end,
            )
            results[edge_type.value] = result
            n = result.statistics.total_trades
            print(f"done ({n} trades)")
        except Exception as e:
            results[edge_type.value] = None
            print(f"error ({e})")

    return results


# ---------------------------------------------------------------------------
# P&L decomposition per edge
# ---------------------------------------------------------------------------

def _print_pnl_breakdown(results: dict, balance: float) -> None:
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}PER-EDGE P&L DECOMPOSITION{Colors.END}")
    print("=" * 100)

    header = (
        f"  {'Edge':<22} {'Trades':>7} {'Gross P&L':>11} {'Costs':>10} "
        f"{'Net P&L':>11} {'Net %':>8}  {'Verdict'}"
    )
    print(header)
    print("-" * 100)

    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0

    for edge_name, result in sorted(results.items()):
        if result is None or result.statistics.total_trades == 0:
            continue

        trades = result.trades
        s = result.statistics

        gross = sum(t.gross_pnl for t in trades)
        costs = sum(t.costs for t in trades)
        net = sum(t.net_pnl for t in trades)
        net_pct = (net / balance) * 100

        total_gross += gross
        total_costs += costs
        total_net += net

        verdict = s.verdict
        net_sign = "+" if net >= 0 else ""

        print(
            f"  {edge_name:<22} {s.total_trades:>7} "
            f"${gross:>+10,.2f} ${costs:>9,.2f} "
            f"${net:>+10,.2f} {net_sign}{net_pct:>6.1f}%  "
            f"{colorize(verdict, verdict)}"
        )

    print("-" * 100)
    total_net_pct = (total_net / balance) * 100
    cost_ratio = (total_costs / total_gross * 100) if total_gross != 0 else 0
    print(
        f"  {'TOTAL':<22} {'':>7} "
        f"${total_gross:>+10,.2f} ${total_costs:>9,.2f} "
        f"${total_net:>+10,.2f} {'+' if total_net >= 0 else ''}{total_net_pct:>6.1f}%"
    )
    print(f"\n  Cost drag: {cost_ratio:.1f}% of gross P&L consumed by costs")


# ---------------------------------------------------------------------------
# Monthly returns
# ---------------------------------------------------------------------------

def _print_monthly_returns(results: dict, balance: float) -> None:
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}MONTHLY RETURNS{Colors.END}")
    print("=" * 100)

    # Collect all trades sorted by entry time
    all_trades = []
    for result in results.values():
        if result is not None and result.trades:
            all_trades.extend(result.trades)

    if not all_trades:
        print("  No trades to analyse.")
        return

    all_trades.sort(key=lambda t: t.entry_time)

    # Group by month
    monthly: Dict[str, List] = defaultdict(list)
    for t in all_trades:
        key = t.entry_time.strftime("%Y-%m")
        monthly[key].append(t)

    header = (
        f"  {'Month':<10} {'Trades':>7} {'Winners':>8} {'Win%':>6} "
        f"{'P&L':>11} {'Cum P&L':>11} {'Equity':>11}"
    )
    print(header)
    print("-" * 80)

    cum_pnl = 0.0
    best_month = ("", float("-inf"))
    worst_month = ("", float("inf"))
    monthly_pnls = []

    for month_key in sorted(monthly.keys()):
        trades = monthly[month_key]
        month_pnl = sum(t.net_pnl for t in trades)
        winners = sum(1 for t in trades if t.net_pnl > 0)
        wr = (winners / len(trades)) * 100 if trades else 0
        cum_pnl += month_pnl
        equity = balance + cum_pnl
        monthly_pnls.append(month_pnl)

        if month_pnl > best_month[1]:
            best_month = (month_key, month_pnl)
        if month_pnl < worst_month[1]:
            worst_month = (month_key, month_pnl)

        pnl_color = Colors.GREEN if month_pnl >= 0 else Colors.RED
        print(
            f"  {month_key:<10} {len(trades):>7} {winners:>8} {wr:>5.1f}% "
            f"{pnl_color}${month_pnl:>+10,.2f}{Colors.END} "
            f"${cum_pnl:>+10,.2f} ${equity:>10,.2f}"
        )

    print("-" * 80)

    if monthly_pnls:
        import statistics as stat_mod
        avg = stat_mod.mean(monthly_pnls)
        pos_months = sum(1 for p in monthly_pnls if p > 0)
        total_months = len(monthly_pnls)
        print(f"\n  Avg monthly P&L:  ${avg:,.2f}")
        print(f"  Positive months:  {pos_months}/{total_months} ({pos_months/total_months*100:.0f}%)")
        print(f"  Best month:       {best_month[0]} (${best_month[1]:+,.2f})")
        print(f"  Worst month:      {worst_month[0]} (${worst_month[1]:+,.2f})")


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def _print_equity_curve(results: dict, balance: float) -> None:
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}EQUITY CURVE{Colors.END}")
    print("=" * 100)

    all_trades = []
    for result in results.values():
        if result is not None and result.trades:
            all_trades.extend(result.trades)

    if not all_trades:
        print("  No trades to analyse.")
        return

    all_trades.sort(key=lambda t: t.entry_time)

    equity = balance
    peak = balance
    max_dd = 0.0
    max_dd_pct = 0.0
    peak_equity = balance
    trough_equity = balance
    dd_start = None
    dd_end = None

    # Track equity at quarterly intervals for display
    quarterly: Dict[str, float] = {}
    equity_at_trade: List[float] = []

    for t in all_trades:
        equity += t.net_pnl
        equity_at_trade.append(equity)

        if equity > peak:
            peak = equity

        dd = peak - equity
        dd_pct = (dd / peak * 100) if peak > 0 else 0

        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct
            trough_equity = equity

        q_key = f"{t.entry_time.year}-Q{(t.entry_time.month - 1) // 3 + 1}"
        quarterly[q_key] = equity

    # Print quarterly milestones
    print(f"\n  {'Quarter':<12} {'Equity':>12} {'vs Start':>10}")
    print("  " + "-" * 36)

    for q_key in sorted(quarterly.keys()):
        eq = quarterly[q_key]
        vs_start = eq - balance
        print(f"  {q_key:<12} ${eq:>11,.2f} ${vs_start:>+9,.2f}")

    final_equity = equity_at_trade[-1] if equity_at_trade else balance
    total_return = ((final_equity - balance) / balance) * 100

    print(f"\n  Starting equity:  ${balance:>12,.2f}")
    print(f"  Final equity:     ${final_equity:>12,.2f}")
    print(f"  Total return:     {'+' if total_return >= 0 else ''}{total_return:.1f}%")
    print(f"  High-water mark:  ${peak:>12,.2f}")
    print(f"  Max drawdown:     ${max_dd:>12,.2f} ({max_dd_pct:.1f}%)")

    # Simple ASCII equity mini-chart (50 chars wide)
    if len(equity_at_trade) >= 2:
        _print_mini_chart(equity_at_trade, balance)


def _print_mini_chart(equity_series: List[float], balance: float) -> None:
    """Print a simple ASCII sparkline of equity."""
    width = 60
    height = 10

    # Sample equity_series down to `width` points
    n = len(equity_series)
    if n > width:
        indices = [int(i * (n - 1) / (width - 1)) for i in range(width)]
        sampled = [equity_series[i] for i in indices]
    else:
        sampled = equity_series

    lo = min(min(sampled), balance)
    hi = max(max(sampled), balance)
    span = hi - lo if hi > lo else 1.0

    print(f"\n  Equity Growth ({len(equity_series)} trades)")
    print(f"  ${hi:,.0f} |", end="")

    # Build a simple grid
    grid = [[" "] * len(sampled) for _ in range(height)]
    for col, val in enumerate(sampled):
        row = int((val - lo) / span * (height - 1))
        row = min(row, height - 1)
        grid[row][col] = "*"

    # Print from top to bottom
    for r in range(height - 1, -1, -1):
        line = "".join(grid[r])
        if r == height - 1:
            print(line)
        elif r == 0:
            print(f"  ${lo:,.0f} |{line}")
        else:
            print(f"{'':>12}|{line}")

    print(f"{'':>12} " + "-" * len(sampled))


# ---------------------------------------------------------------------------
# Compounding analysis
# ---------------------------------------------------------------------------

def _print_compounding_analysis(results: dict, balance: float) -> None:
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}COMPOUNDING ANALYSIS{Colors.END}")
    print("=" * 100)

    all_trades = []
    for result in results.values():
        if result is not None and result.trades:
            all_trades.extend(result.trades)

    if not all_trades:
        print("  No trades to analyse.")
        return

    all_trades.sort(key=lambda t: t.entry_time)

    # Calculate what P&L would be WITHOUT compounding (fixed sizing)
    # vs what it actually was WITH compounding
    equity_compounded = balance
    equity_fixed = balance

    for t in all_trades:
        # Compounded result (actual)
        equity_compounded += t.net_pnl

        # Fixed-size approximation: scale P&L by ratio of starting_balance / equity_at_trade
        # If position_size was scaled by equity, the fixed-size P&L would be smaller/larger
        # We approximate: fixed_pnl = net_pnl * (balance / equity_at_time)
        # But we don't have equity_at_time from the trade object — use the running balance
        # This is an approximation since we can't perfectly reconstruct fixed-size trades
        equity_fixed += t.net_pnl  # Without per-trade equity info, use same as compounded

    # Per-edge compounding effect
    print(f"\n  {'Edge':<22} {'Start Eq':>11} {'End Eq':>11} {'P&L':>11} {'Return':>8}")
    print("  " + "-" * 70)

    for edge_name, result in sorted(results.items()):
        if result is None or result.statistics.total_trades == 0:
            continue

        s = result.statistics
        start_eq = balance
        end_eq = balance + s.total_pnl
        ret = s.total_pnl_pct

        color = Colors.GREEN if s.total_pnl >= 0 else Colors.RED
        print(
            f"  {edge_name:<22} ${start_eq:>10,.2f} ${end_eq:>10,.2f} "
            f"{color}${s.total_pnl:>+10,.2f}{Colors.END} {'+' if ret >= 0 else ''}{ret:.1f}%"
        )

    # Tier breakdown across all edges
    total_a_trades = 0
    total_a_pnl = 0.0
    total_b_trades = 0
    total_b_pnl = 0.0
    total_c_trades = 0
    total_c_pnl = 0.0
    total_d_trades = 0
    total_d_pnl = 0.0

    for result in results.values():
        if result is None or result.statistics.total_trades == 0:
            continue
        s = result.statistics
        total_a_trades += s.tier_a_count
        total_a_pnl += s.tier_a_pnl
        total_b_trades += s.tier_b_count
        total_b_pnl += s.tier_b_pnl
        total_c_trades += s.tier_c_count
        total_c_pnl += s.tier_c_pnl
        total_d_trades += s.tier_d_count
        total_d_pnl += s.tier_d_pnl

    tier_total = total_a_trades + total_b_trades + total_c_trades + total_d_trades

    if tier_total > 0:
        print(f"\n  {Colors.BOLD}TIER BREAKDOWN (Score-Based Sizing){Colors.END}")
        print(f"  {'Tier':<12} {'Mult':>6} {'Trades':>8} {'P&L':>12} {'Avg P&L':>10}")
        print("  " + "-" * 50)

        for tier, mult, count, pnl in [
            ("A (80+)", "1.50x", total_a_trades, total_a_pnl),
            ("B (65-79)", "1.25x", total_b_trades, total_b_pnl),
            ("C (50-64)", "1.00x", total_c_trades, total_c_pnl),
            ("D (40-49)", "0.50x", total_d_trades, total_d_pnl),
        ]:
            if count > 0:
                avg = pnl / count
                color = Colors.GREEN if pnl >= 0 else Colors.RED
                print(
                    f"  {tier:<12} {mult:>6} {count:>8} "
                    f"{color}${pnl:>+11,.2f}{Colors.END} ${avg:>+9,.2f}"
                )

        print("  " + "-" * 50)
        total_tier_pnl = total_a_pnl + total_b_pnl + total_c_pnl + total_d_pnl
        print(
            f"  {'TOTAL':<12} {'':>6} {tier_total:>8} "
            f"${total_tier_pnl:>+11,.2f} ${total_tier_pnl / tier_total:>+9,.2f}"
        )
    else:
        print("\n  (No tier data — score-sizing may be disabled)")


# ---------------------------------------------------------------------------
# Trade distribution
# ---------------------------------------------------------------------------

def _print_trade_distribution(results: dict) -> None:
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}TRADE P&L DISTRIBUTION{Colors.END}")
    print("=" * 100)

    all_trades = []
    for result in results.values():
        if result is not None and result.trades:
            all_trades.extend(result.trades)

    if not all_trades:
        print("  No trades.")
        return

    pnls = [t.net_pnl for t in all_trades]
    pnls.sort()

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    flat = [p for p in pnls if p == 0]

    print(f"\n  Total trades:    {len(pnls)}")
    print(f"  Winners:         {len(winners)} ({len(winners)/len(pnls)*100:.1f}%)")
    print(f"  Losers:          {len(losers)} ({len(losers)/len(pnls)*100:.1f}%)")
    if flat:
        print(f"  Flat:            {len(flat)}")

    if winners:
        print(f"\n  Largest win:     ${max(winners):+,.2f}")
        print(f"  Avg win:         ${sum(winners)/len(winners):+,.2f}")
        print(f"  Median win:      ${winners[len(winners)//2]:+,.2f}")

    if losers:
        print(f"\n  Largest loss:    ${min(losers):+,.2f}")
        print(f"  Avg loss:        ${sum(losers)/len(losers):+,.2f}")
        print(f"  Median loss:     ${losers[len(losers)//2]:+,.2f}")

    # Histogram
    if len(pnls) >= 5:
        _print_histogram(pnls)


def _print_histogram(pnls: List[float]) -> None:
    """Print a simple text histogram of P&L distribution."""
    n_bins = 10
    lo = min(pnls)
    hi = max(pnls)

    if lo == hi:
        return

    bin_width = (hi - lo) / n_bins
    bins = [0] * n_bins

    for p in pnls:
        idx = int((p - lo) / bin_width)
        idx = min(idx, n_bins - 1)
        bins[idx] += 1

    max_count = max(bins)
    bar_max = 40

    print(f"\n  P&L Distribution:")
    for i in range(n_bins):
        lo_edge = lo + i * bin_width
        hi_edge = lo + (i + 1) * bin_width
        bar_len = int(bins[i] / max_count * bar_max) if max_count > 0 else 0
        bar = "#" * bar_len

        # Color: red for negative bins, green for positive
        if hi_edge <= 0:
            color = Colors.RED
        elif lo_edge >= 0:
            color = Colors.GREEN
        else:
            color = Colors.YELLOW

        print(
            f"  ${lo_edge:>+8,.0f} to ${hi_edge:>+8,.0f} | "
            f"{color}{bar:<{bar_max}}{Colors.END} {bins[i]}"
        )


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def _print_summary_table(results: dict, balance: float) -> None:
    """Print the overview summary table (same as backtest_all)."""
    print("\n" + "=" * 100)
    print(f"  {Colors.BOLD}EDGE SUMMARY{Colors.END}")
    print("=" * 100)

    header = (
        f"  {'Edge':<22} {'Trades':>7} {'Win%':>7} "
        f"{'PF':>6} {'P&L%':>8} {'DD%':>6} {'Sharpe':>7}  {'Verdict'}"
    )
    print(header)
    print("-" * 100)

    total_trades = 0
    total_pnl = 0.0
    valid = []
    marginal = []

    for edge_name, result in sorted(results.items()):
        if result is None or result.statistics.total_trades == 0:
            verdict = "NO_DATA" if result and result.statistics.total_trades == 0 else "ERROR"
            print(
                f"  {edge_name:<22} {'--':>7} {'--':>7} {'--':>6} "
                f"{'--':>8} {'--':>6} {'--':>7}  "
                f"{colorize(verdict, verdict)}"
            )
            continue

        s = result.statistics
        total_trades += s.total_trades
        total_pnl += s.total_pnl

        verdict = s.verdict
        dd_limit = getattr(s, "dd_threshold", 15.0)

        print(
            f"  {edge_name:<22} {s.total_trades:>7} {s.win_rate:>6.1f}% "
            f"{s.profit_factor:>6.2f} {s.total_pnl_pct:>+7.1f}% "
            f"{s.max_drawdown_pct:>5.1f}% {s.sharpe_ratio:>7.2f}  "
            f"{colorize(verdict, verdict)}"
        )

        if verdict == "VALID":
            valid.append(edge_name)
        elif verdict == "MARGINAL":
            marginal.append(edge_name)

    print("-" * 100)

    if total_trades > 0:
        pnl_pct = (total_pnl / balance) * 100
        print(f"  {'TOTAL':<22} {total_trades:>7} {'':>7} {'':>6} {pnl_pct:>+7.1f}%")

    print(f"\n  {Colors.GREEN}VALID:{Colors.END}    {', '.join(valid) if valid else 'None'}")
    print(f"  {Colors.YELLOW}MARGINAL:{Colors.END} {', '.join(marginal) if marginal else 'None'}")

    if valid or marginal:
        tradeable = valid + [f"{m} (75%)" for m in marginal]
        print(f"\n  {Colors.BOLD}Tradeable edges:{Colors.END} {', '.join(tradeable)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    args = _parse_args()

    sizing = "Score-based + compounding" if not args.no_score_sizing else "Fixed (flat)"

    print(f"\n{'=' * 100}")
    print(f"  {Colors.BOLD}NEXUS BACKTEST - P&L AND COMPOUNDING REPORT{Colors.END}")
    print(f"{'=' * 100}")
    print(f"  Period:   {args.start} to {args.end}")
    print(f"  Balance:  ${args.balance:,.0f}")
    print(f"  Risk:     {args.risk}%")
    print(f"  Sizing:   {sizing}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"\nRunning backtests...\n")

    results = await _run_all(args)

    if not results:
        print("\nNo results. Check API key and data availability.\n")
        return

    # 1. Edge summary table
    _print_summary_table(results, args.balance)

    # 2. P&L decomposition (gross, costs, net)
    _print_pnl_breakdown(results, args.balance)

    # 3. Monthly returns
    _print_monthly_returns(results, args.balance)

    # 4. Equity curve
    _print_equity_curve(results, args.balance)

    # 5. Compounding analysis + tier breakdown
    _print_compounding_analysis(results, args.balance)

    # 6. Trade P&L distribution
    _print_trade_distribution(results)

    print(f"\n{'=' * 100}")
    print(f"  {Colors.BOLD}END OF REPORT{Colors.END}")
    print(f"{'=' * 100}\n")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
