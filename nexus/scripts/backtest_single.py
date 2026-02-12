"""
Backtest a single edge type over historical data.

Usage::

    python -m nexus.scripts.backtest_single --edge vwap_deviation
    python -m nexus.scripts.backtest_single --edge rsi_extreme --symbol SPY --start 2024-01-01 --end 2024-06-30
    python -m nexus.scripts.backtest_single --edge vwap_deviation --balance 25000 --risk 2.0

Outputs a detailed statistics report for the chosen edge.
"""

import argparse
import asyncio
import logging

from nexus.backtest.engine import BacktestEngine
from nexus.core.enums import EdgeType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# All edges that have backtest signal logic implemented
AVAILABLE_EDGES = {et.value for et in BacktestEngine.SCANNER_MAP}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS – Backtest a single edge",
    )
    parser.add_argument(
        "--edge",
        type=str,
        required=True,
        choices=sorted(AVAILABLE_EDGES),
        help="Edge type to backtest",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol to test (default: edge's primary instrument)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Bar size, e.g. 5m, 15m, 1h, 1d (default: edge's natural timeframe)",
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade %% (default: 1.0)")
    parser.add_argument("--score-sizing", action="store_true", default=False,
                        help="Enable score-based position sizing + compounding")
    parser.add_argument("--reverse", action="store_true", default=False,
                        help="Test reversed signals (LONG↔SHORT) for disabled edges")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save results as JSON baseline for future comparison")
    parser.add_argument("--experimental", action="store_true", default=False,
                        help="Run with experimental optimizations (flag for tracking)")
    return parser.parse_args()


def _print_report(result) -> None:
    """Print a detailed single-edge report."""
    s = result.statistics

    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULT: {s.edge_type}")
    print("=" * 70)
    print(f"  Symbol:     {s.symbol}")
    print(f"  Timeframe:  {s.timeframe}")
    print(f"  Period:     {s.test_period}")
    print(f"  Balance:    ${result.parameters['starting_balance']:,.0f}")
    print(f"  Risk/Trade: {result.parameters['risk_per_trade']:.1f}%")

    print("\n--- Trade Summary " + "-" * 52)
    print(f"  Total Trades:  {s.total_trades}")
    print(f"  Winners:       {s.winners}")
    print(f"  Losers:        {s.losers}")
    print(f"  Win Rate:      {s.win_rate:.1f}%")

    print("\n--- Profitability " + "-" * 52)
    print(f"  Total P&L:       ${s.total_pnl:,.2f} ({s.total_pnl_pct:+.1f}%)")
    print(f"  Profit Factor:   {s.profit_factor:.2f}")
    print(f"  Expected Value:  ${s.expected_value:,.2f} ({s.expected_value_pct:+.3f}%)")
    print(f"  Largest Win:     ${s.largest_win:,.2f}")
    print(f"  Largest Loss:    ${s.largest_loss:,.2f}")
    print(f"  Avg Win:         ${s.avg_win:,.2f} ({s.avg_win_pct:+.2f}%)")
    print(f"  Avg Loss:        ${s.avg_loss:,.2f} ({s.avg_loss_pct:+.2f}%)")

    print("\n--- Risk " + "-" * 62)
    print(f"  Max Drawdown:  ${s.max_drawdown:,.2f} ({s.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe Ratio:  {s.sharpe_ratio:.2f}")
    print(f"  Avg Hold:      {s.avg_hold_duration:.1f} hours")

    print("\n--- Exit Analysis " + "-" * 52)
    print(f"  Stop Loss:    {s.stop_loss_exits}")
    print(f"  Take Profit:  {s.take_profit_exits}")
    print(f"  Indicator:    {s.indicator_exits}")
    print(f"  Time Expiry:  {s.time_expiry_exits}")

    has_tiers = s.tier_a_count + s.tier_b_count + s.tier_c_count + s.tier_d_count
    if has_tiers:
        print("\n--- Score Tier Distribution " + "-" * 42)
        if s.tier_a_count:
            print(f"  A-tier (80+):   {s.tier_a_count:>4} trades  P&L: ${s.tier_a_pnl:>+10,.2f}  [1.5x]")
        if s.tier_b_count:
            print(f"  B-tier (65-79): {s.tier_b_count:>4} trades  P&L: ${s.tier_b_pnl:>+10,.2f}  [1.25x]")
        if s.tier_c_count:
            print(f"  C-tier (50-64): {s.tier_c_count:>4} trades  P&L: ${s.tier_c_pnl:>+10,.2f}  [1.0x]")
        if s.tier_d_count:
            print(f"  D-tier (40-49): {s.tier_d_count:>4} trades  P&L: ${s.tier_d_pnl:>+10,.2f}  [0.5x]")

    print("\n--- Statistical Significance " + "-" * 42)
    print(f"  T-Statistic:  {s.t_statistic:.3f}")
    print(f"  P-Value:      {s.p_value:.4f}")
    print(f"  Significant:  {s.is_significant}")

    print("\n" + "=" * 70)
    verdict_marker = {
        "VALID": "[PASS]",
        "MARGINAL": "[WARN]",
        "INVALID": "[FAIL]",
        "INSUFFICIENT_DATA": "[----]",
    }.get(s.verdict, "[????]")
    print(f"  {verdict_marker}  VERDICT: {s.verdict}")
    print(f"           {s.verdict_reason}")
    print("=" * 70 + "\n")


async def run() -> None:
    args = _parse_args()
    edge_type = EdgeType(args.edge)

    # Reverse mode: test flipped signals on disabled edges
    if args.reverse:
        from nexus.backtest.diagnostics import test_reversed_edge

        logger.info("Running REVERSED signal test for: %s", args.edge)
        result = await test_reversed_edge(
            edge_type=edge_type,
            start_date=args.start,
            end_date=args.end,
            starting_balance=args.balance,
        )

        print(f"\n{'=' * 70}")
        print(f"  SIGNAL REVERSAL TEST: {args.edge}")
        print(f"{'=' * 70}")

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            orig = result["original"]
            rev = result["reversed"]
            print(f"\n  {'Metric':<20} {'Original':>12} {'Reversed':>12} {'Delta':>12}")
            print(f"  {'-' * 56}")
            print(f"  {'Trades':<20} {orig['trades']:>12} {rev['trades']:>12}")
            print(f"  {'Win Rate':<20} {orig['win_rate']:>11.1f}% {rev['win_rate']:>11.1f}%")
            print(f"  {'Profit Factor':<20} {orig['profit_factor']:>12.2f} {rev['profit_factor']:>12.2f}")
            print(f"  {'P&L %':<20} {orig['total_pnl_pct']:>+11.1f}% {rev['total_pnl_pct']:>+11.1f}%")
            print(f"  {'Max DD %':<20} {orig['max_dd_pct']:>11.1f}% {rev['max_dd_pct']:>11.1f}%")
            print(f"  {'Sharpe':<20} {orig['sharpe']:>12.2f} {rev['sharpe']:>12.2f}")
            print(f"  {'Verdict':<20} {orig['verdict']:>12} {rev['verdict']:>12}")
            print(f"\n  Recommendation: {result['recommendation']}")

        print(f"{'=' * 70}\n")
        return

    engine = BacktestEngine(
        starting_balance=args.balance,
        risk_per_trade=args.risk,
        use_score_sizing=args.score_sizing,
    )

    logger.info(
        "Running backtest: %s  symbol=%s  tf=%s  %s to %s",
        args.edge,
        args.symbol or "(default)",
        args.timeframe or "(default)",
        args.start,
        args.end,
    )

    result = await engine.run_edge_backtest(
        edge_type=edge_type,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
    )

    _print_report(result)

    if args.save_baseline:
        from nexus.backtest.diagnostics import save_baseline
        save_baseline({args.edge: result}, args.save_baseline)
        print(f"\n  Baseline saved to: {args.save_baseline}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
