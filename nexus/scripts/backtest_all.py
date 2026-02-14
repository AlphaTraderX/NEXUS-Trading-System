"""
Backtest ALL configured edges and produce a validation report.

Usage::

    python -m nexus.scripts.backtest_all
    python -m nexus.scripts.backtest_all --start 2024-01-01 --end 2024-12-31
    python -m nexus.scripts.backtest_all --save validation_results.csv

"""

import argparse
import asyncio
import csv
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
    BOLD = "\033[1m"
    END = "\033[0m"


def _supports_color() -> bool:
    """Best-effort check for ANSI colour support."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Modern Windows Terminal / PowerShell support ANSI, cmd may not.
        # Enable VT processing just in case.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
        return True
    return True


# Disable colours when piped or on dumb terminals
if not _supports_color():
    Colors.GREEN = Colors.YELLOW = Colors.RED = ""
    Colors.GRAY = Colors.BOLD = Colors.END = ""


def colorize(text: str, verdict: str) -> str:
    """Add colour based on verdict."""
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
        description="NEXUS - Backtest all edges and generate validation report",
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade %% (default: 1.0)")
    parser.add_argument("--save", type=str, default=None, help="Save results to CSV file")
    parser.add_argument("--no-score-sizing", action="store_true", default=False,
                        help="Disable tier-based position sizing (use flat sizing)")
    # DEPRECATED alias kept for backwards compat — score-sizing is now ON by default
    parser.add_argument("--score-sizing", action="store_true", default=False,
                        help="(deprecated) Score-sizing is now default; use --no-score-sizing to disable")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save results as JSON baseline for future comparison")
    parser.add_argument("--compare-to", type=str, default=None,
                        help="Compare results to a saved JSON baseline")
    parser.add_argument("--experimental", action="store_true", default=False,
                        help="Run with experimental optimizations (flag for tracking)")
    parser.add_argument("--registry", action="store_true", default=False,
                        help="Use InstrumentRegistry + cache instead of hardcoded symbol lists")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run backtests with progress
# ---------------------------------------------------------------------------

async def _run_all_backtests(args) -> Dict[str, Optional[object]]:
    from nexus.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        starting_balance=args.balance,
        risk_per_trade=args.risk,
        use_score_sizing=not args.no_score_sizing,
        use_registry=args.registry,
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

def _print_report(results: dict, start: str, end: str, balance: float) -> None:
    """Print the full edge validation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "=" * 110)
    print(f"  {Colors.BOLD}NEXUS EDGE VALIDATION REPORT{Colors.END}")
    print("=" * 110)
    print(f"  Generated:  {now}")
    print(f"  Period:     {start} to {end}")
    print(f"  Balance:    ${balance:,.0f}")
    print(f"  Edges:      {len(results)} tested")

    # ---- summary table -------------------------------------------------
    print("\n" + "-" * 110)
    header = (
        f"  {'Edge':<22} {'Trades':>7} {'Win%':>7} "
        f"{'PF':>6} {'P&L%':>8} {'DD%':>6} {'Limit':>6} {'Sharpe':>7}  {'Verdict'}"
    )
    print(header)
    print("-" * 110)

    valid_edges: list[str] = []
    marginal_edges: list[str] = []
    invalid_edges: list[str] = []
    no_data_edges: list[str] = []

    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    total_gross_wins = 0.0
    total_gross_losses = 0.0

    for edge_name, result in sorted(results.items()):
        if result is None:
            # Engine raised an error for this edge
            row = (
                f"  {edge_name:<22} {'--':>7} {'--':>7} {'--':>6} "
                f"{'--':>8} {'--':>6} {'--':>6} {'--':>7}  "
                f"{colorize('ERROR', 'ERROR')}"
            )
            invalid_edges.append(edge_name)
        else:
            s = result.statistics
            if s.total_trades == 0:
                row = (
                    f"  {edge_name:<22} {0:>7} {'--':>7} {'--':>6} "
                    f"{'--':>8} {'--':>6} {'--':>6} {'--':>7}  "
                    f"{colorize('NO DATA', 'NO_DATA')}"
                )
                no_data_edges.append(edge_name)
            else:
                total_trades += s.total_trades
                total_winners += s.winners
                total_pnl += s.total_pnl

                # Accumulate gross wins/losses for combined profit factor
                if s.profit_factor == float("inf"):
                    total_gross_wins += s.total_pnl
                else:
                    gross_w = s.total_pnl + abs(s.avg_loss) * s.losers if s.losers else s.total_pnl
                    gross_l = abs(s.avg_loss) * s.losers if s.losers else 0.0
                    total_gross_wins += gross_w
                    total_gross_losses += gross_l

                verdict = s.verdict
                dd_limit = getattr(s, "dd_threshold", 15.0)
                row = (
                    f"  {edge_name:<22} {s.total_trades:>7} {s.win_rate:>6.1f}% "
                    f"{s.profit_factor:>6.2f} {s.total_pnl_pct:>+7.1f}% "
                    f"{s.max_drawdown_pct:>5.1f}% {dd_limit:>5.0f}% {s.sharpe_ratio:>7.2f}  "
                    f"{colorize(verdict, verdict)}"
                )

                if verdict == "VALID":
                    valid_edges.append(edge_name)
                elif verdict == "MARGINAL":
                    marginal_edges.append(edge_name)
                else:
                    invalid_edges.append(edge_name)

        print(row)

    # ---- detailed results per edge -------------------------------------
    print("\n" + "=" * 110)
    print(f"  {Colors.BOLD}DETAILED RESULTS{Colors.END}")
    print("=" * 110)

    for edge_name, result in sorted(results.items()):
        if result is None:
            print(f"\n  {colorize('[ERR]', 'ERROR')} {edge_name}")
            print("    Backtest raised an error — check data / API key")
            continue

        s = result.statistics
        if s.total_trades == 0:
            print(f"\n  {colorize('[---]', 'NO_DATA')} {edge_name}")
            print("    No trades generated (edge may require external data)")
            continue

        verdict_tag = colorize(f"[{s.verdict[:4]}]", s.verdict)
        risk_label = getattr(s, "risk_category", "medium")
        print(f"\n  {verdict_tag} {edge_name}  [{risk_label}-risk]")
        print(f"    Symbol: {s.symbol}  |  Timeframe: {s.timeframe}")
        print(
            f"    Trades: {s.total_trades}  |  "
            f"Win Rate: {s.win_rate:.1f}%  |  "
            f"Profit Factor: {s.profit_factor:.2f} (min: {getattr(s, 'pf_threshold', 1.2):.1f})"
        )
        print(
            f"    P&L: ${s.total_pnl:,.2f} ({s.total_pnl_pct:+.1f}%)  |  "
            f"EV: ${s.expected_value:,.2f} ({s.expected_value_pct:+.3f}%)"
        )
        dd_thresh = getattr(s, "dd_threshold", 15.0)
        dd_ok = s.max_drawdown_pct <= dd_thresh
        dd_status = "" if dd_ok else f"  {Colors.RED}OVER{Colors.END}"
        print(
            f"    Max DD: ${s.max_drawdown:,.2f} ({s.max_drawdown_pct:.1f}% / {dd_thresh:.0f}% limit)"
            f"{dd_status}  |  Sharpe: {s.sharpe_ratio:.2f}"
        )
        print(
            f"    Exits: SL={s.stop_loss_exits}  TP={s.take_profit_exits}  "
            f"Ind={s.indicator_exits}  Time={s.time_expiry_exits}"
        )
        # Show edge-specific p-value threshold from risk profile
        from nexus.backtest.engine import BacktestEngine
        _rp = BacktestEngine.EDGE_RISK_PROFILES.get(
            None, BacktestEngine.DEFAULT_RISK_PROFILE
        )
        # Try to look up by edge name
        try:
            from nexus.core.enums import EdgeType
            _edge_enum = EdgeType(edge_name)
            _rp = BacktestEngine.EDGE_RISK_PROFILES.get(
                _edge_enum, BacktestEngine.DEFAULT_RISK_PROFILE
            )
        except (ValueError, KeyError):
            pass
        _p_thresh = _rp.get("p_value_threshold", 0.05)
        _p_ok = s.p_value < _p_thresh
        print(
            f"    Significance: t={s.t_statistic:.2f}  p={s.p_value:.4f}  "
            f"{'YES' if _p_ok else 'NO'} (threshold: {_p_thresh})"
        )
        tier_total = s.tier_a_count + s.tier_b_count + s.tier_c_count + s.tier_d_count
        if tier_total:
            parts = []
            if s.tier_a_count:
                parts.append(f"A={s.tier_a_count}(${s.tier_a_pnl:+,.0f})")
            if s.tier_b_count:
                parts.append(f"B={s.tier_b_count}(${s.tier_b_pnl:+,.0f})")
            if s.tier_c_count:
                parts.append(f"C={s.tier_c_count}(${s.tier_c_pnl:+,.0f})")
            if s.tier_d_count:
                parts.append(f"D={s.tier_d_count}(${s.tier_d_pnl:+,.0f})")
            print(f"    Tiers: {', '.join(parts)}")
        print(f"    Reason: {s.verdict_reason}")

    # ---- verdict summary -----------------------------------------------
    print("\n" + "=" * 110)
    print(f"  {Colors.BOLD}SUMMARY{Colors.END}")
    print("=" * 110)

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

    # ---- combined statistics -------------------------------------------
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
        monthly_pnl = total_pnl / 12
        monthly_pct = pnl_pct / 12

        print(f"    Combined win rate:  {combined_wr:.1f}%")
        print(f"    Combined PF:        {combined_pf:.2f}")
        print(f"    Total P&L:          ${total_pnl:,.2f} ({pnl_pct:+.1f}%)")
        print(f"    Est. monthly:       ${monthly_pnl:,.2f} ({monthly_pct:+.2f}%/month)")

        # Portfolio-level MaxDD: combine all trades chronologically
        from nexus.backtest.engine import BacktestEngine
        portfolio_max_dd_limit = BacktestEngine.PORTFOLIO_MAX_DD_PCT

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

            dd_ok = portfolio_max_dd_pct <= portfolio_max_dd_limit
            dd_color = Colors.GREEN if dd_ok else Colors.RED
            print(
                f"    Portfolio MaxDD:     {dd_color}{portfolio_max_dd_pct:.1f}%{Colors.END}"
                f" (limit: {portfolio_max_dd_limit:.0f}%)"
            )
            if not dd_ok:
                print(
                    f"    {Colors.RED}WARNING: Portfolio MaxDD exceeds "
                    f"{portfolio_max_dd_limit:.0f}% limit!{Colors.END}"
                )
    else:
        print("    (no trades to summarise)")

    # ---- recommendation ------------------------------------------------
    print(f"\n  {Colors.BOLD}RECOMMENDATION:{Colors.END}")
    if valid_edges:
        print(f"    Trade VALID edges: {', '.join(valid_edges)}")
    if marginal_edges:
        print(f"    Trade MARGINAL edges at 75% size: {', '.join(marginal_edges)}")
    if not valid_edges and not marginal_edges:
        print("    No edges passed validation. Review parameters or date range.")
    print("=" * 110 + "\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_to_csv(results: dict, filepath: str) -> None:
    """Save results to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Edge", "Symbol", "Timeframe", "Trades", "Winners", "Losers",
            "Win%", "PF", "P&L", "P&L%", "MaxDD%", "Sharpe",
            "T-Stat", "P-Value", "Significant", "Verdict", "Reason",
            "Tier_A", "Tier_A_PnL", "Tier_B", "Tier_B_PnL",
            "Tier_C", "Tier_C_PnL", "Tier_D", "Tier_D_PnL",
        ])

        for edge_name, result in sorted(results.items()):
            if result is None or result.statistics.total_trades == 0:
                writer.writerow([edge_name, "", "", 0, "", "", "", "",
                                 "", "", "", "", "", "", "", "NO_DATA", "",
                                 "", "", "", "", "", "", "", ""])
            else:
                s = result.statistics
                writer.writerow([
                    edge_name,
                    s.symbol,
                    s.timeframe,
                    s.total_trades,
                    s.winners,
                    s.losers,
                    f"{s.win_rate:.1f}",
                    f"{s.profit_factor:.2f}",
                    f"{s.total_pnl:.2f}",
                    f"{s.total_pnl_pct:.1f}",
                    f"{s.max_drawdown_pct:.1f}",
                    f"{s.sharpe_ratio:.2f}",
                    f"{s.t_statistic:.2f}",
                    f"{s.p_value:.4f}",
                    s.is_significant,
                    s.verdict,
                    s.verdict_reason,
                    s.tier_a_count, f"{s.tier_a_pnl:.2f}",
                    s.tier_b_count, f"{s.tier_b_pnl:.2f}",
                    s.tier_c_count, f"{s.tier_c_pnl:.2f}",
                    s.tier_d_count, f"{s.tier_d_pnl:.2f}",
                ])

    print(f"\nResults saved to: {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    args = _parse_args()

    print(f"\n{Colors.BOLD}NEXUS Backtester - Full Edge Validation{Colors.END}")
    print(f"  Period:  {args.start} to {args.end}")
    sizing_mode = "Fixed (flat)" if args.no_score_sizing else "Score-based + compounding"
    mode_tag = " [EXPERIMENTAL]" if args.experimental else ""
    registry_tag = " [REGISTRY]" if args.registry else ""
    print(f"  Balance: ${args.balance:,.0f}  |  Risk: {args.risk}%  |  Sizing: {sizing_mode}{mode_tag}{registry_tag}\n")

    results = await _run_all_backtests(args)

    if not results:
        print("\nNo backtest results generated. Check API key and data availability.\n")
        return

    _print_report(results, args.start, args.end, args.balance)

    if args.save:
        _save_to_csv(results, args.save)

    # Baseline save/compare
    if args.save_baseline:
        from nexus.backtest.diagnostics import save_baseline
        save_baseline(results, args.save_baseline)
        print(f"\n  Baseline saved to: {args.save_baseline}")

    if args.compare_to:
        from nexus.backtest.diagnostics import compare_to_baseline, load_baseline, save_baseline as _sb

        baseline = load_baseline(args.compare_to)

        # Build current results dict in same format as baseline
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
            print(f"  {Colors.RED}REVERT RECOMMENDED — one or more edges failed thresholds{Colors.END}")
        print(f"{'=' * 75}\n")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
