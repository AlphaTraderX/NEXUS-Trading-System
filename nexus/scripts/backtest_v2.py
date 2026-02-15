"""
Backtest ALL edges using Engine v2.1 (regime + heat + trailing).

Usage::

    python -m nexus.scripts.backtest_v2
    python -m nexus.scripts.backtest_v2 --start 2022-01-01 --end 2024-12-31
    python -m nexus.scripts.backtest_v2 --save-baseline data/backtest_v2_baseline.json
    python -m nexus.scripts.backtest_v2 --compare-to data/backtest_v1_baseline.json
    python -m nexus.scripts.backtest_v2 --no-regime-filter   # A/B test: disable regime
    python -m nexus.scripts.backtest_v2 --no-trailing-stops  # A/B test: disable trailing
    python -m nexus.scripts.backtest_v2 --no-heat-management # A/B test: disable heat

"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal colours (ANSI escape codes)
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
        description="NEXUS - Backtest v2.1 (regime + heat + trailing)",
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade %%")
    parser.add_argument("--no-score-sizing", action="store_true", default=False,
                        help="Disable tier-based position sizing")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save results as JSON baseline")
    parser.add_argument("--compare-to", type=str, default=None,
                        help="Compare results to a saved JSON baseline")

    # v2.1 A/B testing toggles
    parser.add_argument("--no-regime-filter", action="store_true", default=False,
                        help="Disable regime filtering (A/B test)")
    parser.add_argument("--no-trailing-stops", action="store_true", default=False,
                        help="Disable trailing stops (A/B test)")
    parser.add_argument("--no-heat-management", action="store_true", default=False,
                        help="Disable heat management (A/B test)")
    parser.add_argument("--no-momentum-scaling", action="store_true", default=False,
                        help="Disable momentum scaling (A/B test)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run backtests
# ---------------------------------------------------------------------------

async def _run_all_backtests(args) -> Dict[str, Optional[object]]:
    from nexus.backtest.engine_v2 import BacktestEngineV2

    engine = BacktestEngineV2(
        starting_balance=args.balance,
        risk_per_trade=args.risk,
        use_score_sizing=not args.no_score_sizing,
        use_regime_filter=not args.no_regime_filter,
        use_trailing_stops=not args.no_trailing_stops,
        use_heat_management=not args.no_heat_management,
        use_momentum_scaling=not args.no_momentum_scaling,
    )

    edges = list(engine.SCANNER_MAP.keys())
    total = len(edges)
    results: Dict[str, Optional[object]] = {}

    for i, edge_type in enumerate(edges, 1):
        print(f"  Testing {edge_type.value}... [{i}/{total}]", end=" ", flush=True)

        try:
            result = await engine.run_edge_backtest(
                edge_type=edge_type,
                start_date=args.start,
                end_date=args.end,
            )
            results[edge_type.value] = result
            trades = result.statistics.total_trades
            print(f"done  ({trades} trades)")
        except Exception as e:
            logger.debug("Failed %s: %s", edge_type.value, e)
            results[edge_type.value] = None
            print(f"error ({e})")

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _print_report(results: dict, start: str, end: str, balance: float,
                  features: dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "=" * 115)
    print(f"  {Colors.BOLD}NEXUS ENGINE v2.1 BACKTEST REPORT{Colors.END}")
    print("=" * 115)
    print(f"  Generated:  {now}")
    print(f"  Period:     {start} to {end}")
    print(f"  Balance:    ${balance:,.0f}")
    print(f"  Edges:      {len(results)} tested")

    # Feature toggles
    feat_parts = []
    for name, enabled in features.items():
        tag = f"{Colors.GREEN}ON{Colors.END}" if enabled else f"{Colors.RED}OFF{Colors.END}"
        feat_parts.append(f"{name}={tag}")
    print(f"  Features:   {', '.join(feat_parts)}")

    # Summary table
    print("\n" + "-" * 115)
    header = (
        f"  {'Edge':<22} {'Trades':>7} {'Win%':>7} "
        f"{'PF':>6} {'P&L%':>8} {'DD%':>6} {'Sharpe':>7} {'Trail':>6} {'BkEv':>5}  {'Verdict'}"
    )
    print(header)
    print("-" * 115)

    valid_edges = []
    marginal_edges = []
    invalid_edges = []
    no_data_edges = []

    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    total_gross_wins = 0.0
    total_gross_losses = 0.0

    for edge_name, result in sorted(results.items()):
        if result is None:
            row = (
                f"  {edge_name:<22} {'--':>7} {'--':>7} {'--':>6} "
                f"{'--':>8} {'--':>6} {'--':>7} {'--':>6} {'--':>5}  "
                f"{colorize('ERROR', 'ERROR')}"
            )
            invalid_edges.append(edge_name)
        else:
            s = result.statistics
            if s.total_trades == 0:
                row = (
                    f"  {edge_name:<22} {0:>7} {'--':>7} {'--':>6} "
                    f"{'--':>8} {'--':>6} {'--':>7} {'--':>6} {'--':>5}  "
                    f"{colorize('NO DATA', 'NO_DATA')}"
                )
                no_data_edges.append(edge_name)
            else:
                total_trades += s.total_trades
                total_winners += s.winners
                total_pnl += s.total_pnl

                if s.profit_factor == float("inf"):
                    total_gross_wins += s.total_pnl
                else:
                    gross_w = s.total_pnl + abs(s.avg_loss) * s.losers if s.losers else s.total_pnl
                    gross_l = abs(s.avg_loss) * s.losers if s.losers else 0.0
                    total_gross_wins += gross_w
                    total_gross_losses += gross_l

                trail = getattr(s, "trailing_stop_exits", 0)
                bkev = getattr(s, "breakeven_exits", 0)

                verdict = s.verdict
                row = (
                    f"  {edge_name:<22} {s.total_trades:>7} {s.win_rate:>6.1f}% "
                    f"{s.profit_factor:>6.2f} {s.total_pnl_pct:>+7.1f}% "
                    f"{s.max_drawdown_pct:>5.1f}% {s.sharpe_ratio:>7.2f} "
                    f"{trail:>6} {bkev:>5}  "
                    f"{colorize(verdict, verdict)}"
                )

                if verdict == "VALID":
                    valid_edges.append(edge_name)
                elif verdict == "MARGINAL":
                    marginal_edges.append(edge_name)
                else:
                    invalid_edges.append(edge_name)

        print(row)

    # Regime distribution
    _print_regime_stats(results)

    # Detailed results
    print("\n" + "=" * 115)
    print(f"  {Colors.BOLD}DETAILED RESULTS{Colors.END}")
    print("=" * 115)

    for edge_name, result in sorted(results.items()):
        if result is None:
            print(f"\n  {colorize('[ERR]', 'ERROR')} {edge_name}")
            print("    Backtest raised an error")
            continue

        s = result.statistics
        if s.total_trades == 0:
            print(f"\n  {colorize('[---]', 'NO_DATA')} {edge_name}")
            print("    No trades generated")
            continue

        verdict_tag = colorize(f"[{s.verdict[:4]}]", s.verdict)
        print(f"\n  {verdict_tag} {edge_name}")
        print(f"    Symbol: {s.symbol}  |  Timeframe: {s.timeframe}")
        print(
            f"    Trades: {s.total_trades}  |  "
            f"Win Rate: {s.win_rate:.1f}%  |  "
            f"PF: {s.profit_factor:.2f}"
        )
        print(
            f"    P&L: ${s.total_pnl:,.2f} ({s.total_pnl_pct:+.1f}%)  |  "
            f"EV: ${s.expected_value:,.2f} ({s.expected_value_pct:+.3f}%)"
        )
        print(
            f"    Max DD: {s.max_drawdown_pct:.1f}%  |  Sharpe: {s.sharpe_ratio:.2f}"
        )
        trail = getattr(s, "trailing_stop_exits", 0)
        bkev = getattr(s, "breakeven_exits", 0)
        print(
            f"    Exits: SL={s.stop_loss_exits}  TP={s.take_profit_exits}  "
            f"Trail={trail}  BkEv={bkev}  "
            f"Ind={s.indicator_exits}  Time={s.time_expiry_exits}"
        )
        print(f"    Significance: p={s.p_value:.4f}")

        # Regime info from parameters
        params = getattr(result, "parameters", {})
        regime_signals = params.get("regime_signals", {})
        regime_filtered = params.get("regime_filtered", {})
        if regime_signals:
            regime_parts = [f"{k}={v}" for k, v in sorted(regime_signals.items())]
            print(f"    Regime signals: {', '.join(regime_parts)}")
        if regime_filtered:
            filtered_parts = [f"{k}={v}" for k, v in sorted(regime_filtered.items())]
            print(f"    Regime filtered: {', '.join(filtered_parts)}")

    # Summary
    print("\n" + "=" * 115)
    print(f"  {Colors.BOLD}SUMMARY{Colors.END}")
    print("=" * 115)

    print(
        f"  {Colors.GREEN}VALID ({len(valid_edges)}):{Colors.END}  "
        f"{', '.join(valid_edges) if valid_edges else 'None'}"
    )
    print(
        f"  {Colors.YELLOW}MARGINAL ({len(marginal_edges)}):{Colors.END}  "
        f"{', '.join(marginal_edges) if marginal_edges else 'None'}"
    )
    print(
        f"  {Colors.RED}INVALID ({len(invalid_edges)}):{Colors.END}  "
        f"{', '.join(invalid_edges) if invalid_edges else 'None'}"
    )
    if no_data_edges:
        print(
            f"  {Colors.GRAY}NO DATA ({len(no_data_edges)}):{Colors.END}  "
            f"{', '.join(no_data_edges)}"
        )

    # Combined statistics
    print(f"\n  {Colors.BOLD}COMBINED STATISTICS:{Colors.END}")
    print(f"    Total trades:       {total_trades}")

    if total_trades > 0:
        combined_wr = (total_winners / total_trades) * 100
        combined_pf = (
            total_gross_wins / total_gross_losses
            if total_gross_losses > 0
            else float("inf")
        )
        pnl_pct = (total_pnl / balance) * 100

        print(f"    Combined win rate:  {combined_wr:.1f}%")
        print(f"    Combined PF:        {combined_pf:.2f}")
        print(f"    Total P&L:          ${total_pnl:,.2f} ({pnl_pct:+.1f}%)")

        # Portfolio MaxDD
        all_trades = []
        for edge_name, result in results.items():
            if result is not None and result.trades:
                all_trades.extend(result.trades)

        if all_trades:
            all_trades.sort(key=lambda t: t.entry_time)
            equity = balance
            peak = balance
            portfolio_max_dd_pct = 0.0
            for t in all_trades:
                equity += t.net_pnl
                if equity > peak:
                    peak = equity
                dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0
                if dd_pct > portfolio_max_dd_pct:
                    portfolio_max_dd_pct = dd_pct

            dd_color = Colors.GREEN if portfolio_max_dd_pct <= 30.0 else Colors.RED
            print(
                f"    Portfolio MaxDD:     {dd_color}{portfolio_max_dd_pct:.1f}%{Colors.END}"
                f" (limit: 30%)"
            )
    else:
        print("    (no trades to summarise)")

    print("=" * 115 + "\n")


def _print_regime_stats(results: dict) -> None:
    """Print regime distribution across all edges."""
    all_signals = {}
    all_filtered = {}

    for edge_name, result in results.items():
        if result is None:
            continue
        params = getattr(result, "parameters", {})
        for regime, count in params.get("regime_signals", {}).items():
            all_signals[regime] = all_signals.get(regime, 0) + count
        for regime, count in params.get("regime_filtered", {}).items():
            all_filtered[regime] = all_filtered.get(regime, 0) + count

    if not all_signals:
        return

    total_signals = sum(all_signals.values())
    total_filtered = sum(all_filtered.values())

    print(f"\n  {Colors.BOLD}REGIME DISTRIBUTION:{Colors.END}")
    print(f"    {'Regime':<18} {'Signals':>8} {'%':>6} {'Filtered':>9} {'%':>6}")
    print("    " + "-" * 50)
    for regime in sorted(all_signals.keys()):
        signals = all_signals[regime]
        filtered = all_filtered.get(regime, 0)
        sig_pct = (signals / total_signals * 100) if total_signals else 0
        filt_pct = (filtered / signals * 100) if signals else 0
        print(f"    {regime:<18} {signals:>8} {sig_pct:>5.1f}% {filtered:>9} {filt_pct:>5.1f}%")
    print(f"    {'TOTAL':<18} {total_signals:>8}        {total_filtered:>9}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    args = _parse_args()

    features = {
        "regime": not args.no_regime_filter,
        "trailing": not args.no_trailing_stops,
        "heat": not args.no_heat_management,
        "momentum": not args.no_momentum_scaling,
    }

    print(f"\n{Colors.BOLD}NEXUS Backtester v2.1 - Enhanced Engine{Colors.END}")
    print(f"  Period:  {args.start} to {args.end}")
    sizing_mode = "Fixed (flat)" if args.no_score_sizing else "Score-based + compounding"
    feat_str = ", ".join(f"{k}={'ON' if v else 'OFF'}" for k, v in features.items())
    print(f"  Balance: ${args.balance:,.0f}  |  Risk: {args.risk}%  |  Sizing: {sizing_mode}")
    print(f"  Features: {feat_str}\n")

    results = await _run_all_backtests(args)

    if not results:
        print("\nNo backtest results generated.\n")
        return

    _print_report(results, args.start, args.end, args.balance, features)

    # Baseline save/compare
    if args.save_baseline:
        from nexus.backtest.diagnostics import save_baseline
        save_baseline(results, args.save_baseline)
        print(f"\n  Baseline saved to: {args.save_baseline}")

    if args.compare_to:
        from nexus.backtest.diagnostics import compare_to_baseline, load_baseline

        baseline = load_baseline(args.compare_to)

        current: Dict[str, Optional[dict]] = {}
        for edge_name, result in results.items():
            if result is None or result.statistics.total_trades == 0:
                current[edge_name] = None
            else:
                s = result.statistics
                current[edge_name] = {
                    "total_trades": s.total_trades,
                    "win_rate": s.win_rate,
                    "profit_factor": s.profit_factor,
                    "total_pnl": s.total_pnl,
                    "total_pnl_pct": s.total_pnl_pct,
                    "max_drawdown_pct": s.max_drawdown_pct,
                    "max_dd_pct": s.max_drawdown_pct,
                    "sharpe_ratio": s.sharpe_ratio,
                    "verdict": s.verdict,
                }

        all_ok, report = compare_to_baseline(baseline, current)

        print(f"\n{'=' * 75}")
        print(f"  {Colors.BOLD}COMPARISON vs BASELINE: {args.compare_to}{Colors.END}")
        print(f"{'=' * 75}")
        print(report)
        print(f"{'=' * 75}")
        if all_ok:
            print(f"  {Colors.GREEN}ALL CHECKS PASSED{Colors.END}")
        else:
            print(f"  {Colors.RED}REGRESSIONS DETECTED{Colors.END}")
        print(f"{'=' * 75}\n")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
