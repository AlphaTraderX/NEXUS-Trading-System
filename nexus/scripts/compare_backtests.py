"""
Compare two backtest baselines side-by-side.

Usage::

    python -m nexus.scripts.compare_backtests data/backtest_v1_baseline.json data/backtest_v2_baseline.json

"""

import argparse
import json
import sys
from pathlib import Path


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
    return True


if not _supports_color():
    Colors.GREEN = Colors.YELLOW = Colors.RED = ""
    Colors.GRAY = Colors.CYAN = Colors.BOLD = Colors.END = ""


def _delta_color(val: float, higher_is_better: bool = True) -> str:
    """Colorize a delta value (green=good, red=bad)."""
    if val == 0:
        return f"{val:+.2f}"
    if (val > 0) == higher_is_better:
        return f"{Colors.GREEN}{val:+.2f}{Colors.END}"
    return f"{Colors.RED}{val:+.2f}{Colors.END}"


def _delta_color_pct(val: float, higher_is_better: bool = True) -> str:
    if val == 0:
        return f"{val:+.1f}%"
    if (val > 0) == higher_is_better:
        return f"{Colors.GREEN}{val:+.1f}%{Colors.END}"
    return f"{Colors.RED}{val:+.1f}%{Colors.END}"


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare(baseline_path: str, current_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text())
    current = json.loads(Path(current_path).read_text())

    print(f"\n{'=' * 100}")
    print(f"  {Colors.BOLD}BACKTEST COMPARISON{Colors.END}")
    print(f"{'=' * 100}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Current:  {current_path}")

    # Per-edge comparison
    print(f"\n  {Colors.BOLD}PER-EDGE COMPARISON:{Colors.END}")
    print(f"  {'Edge':<22} {'Trades':>10} {'WR%':>10} {'PF':>10} {'P&L%':>12} {'DD%':>10} {'Verdict'}")
    print("  " + "-" * 90)

    all_edges = sorted(set(list(baseline.keys()) + list(current.keys())))

    base_total_pnl = 0.0
    curr_total_pnl = 0.0
    base_total_trades = 0
    curr_total_trades = 0
    improvements = 0
    regressions = 0

    for edge in all_edges:
        base = baseline.get(edge, {})
        curr = current.get(edge, {})

        base_trades = base.get("total_trades", 0) if base else 0
        curr_trades = curr.get("total_trades", 0) if curr else 0

        # Skip if both have no trades
        if base_trades == 0 and curr_trades == 0:
            continue

        base_total_trades += base_trades
        curr_total_trades += curr_trades
        base_total_pnl += base.get("total_pnl", 0) if base else 0
        curr_total_pnl += curr.get("total_pnl", 0) if curr else 0

        # Calculate deltas
        d_trades = curr_trades - base_trades
        d_wr = (curr.get("win_rate", 0) if curr else 0) - (base.get("win_rate", 0) if base else 0)
        d_pf = (curr.get("profit_factor", 0) if curr else 0) - (base.get("profit_factor", 0) if base else 0)
        d_pnl = (curr.get("total_pnl_pct", 0) if curr else 0) - (base.get("total_pnl_pct", 0) if base else 0)
        d_dd = (curr.get("max_drawdown_pct", 0) if curr else 0) - (base.get("max_drawdown_pct", 0) if base else 0)

        base_verdict = base.get("verdict", "N/A") if base else "N/A"
        curr_verdict = curr.get("verdict", "N/A") if curr else "N/A"

        if d_pnl > 0:
            verdict_change = f"{Colors.GREEN}{base_verdict} -> {curr_verdict}{Colors.END}"
            improvements += 1
        elif d_pnl < 0:
            verdict_change = f"{Colors.RED}{base_verdict} -> {curr_verdict}{Colors.END}"
            regressions += 1
        else:
            verdict_change = curr_verdict

        trades_str = f"{d_trades:+d}" if d_trades != 0 else "="
        wr_str = _delta_color_pct(d_wr)
        pf_str = _delta_color(d_pf)
        pnl_str = _delta_color_pct(d_pnl)
        dd_str = _delta_color_pct(d_dd, higher_is_better=False)

        print(f"  {edge:<22} {trades_str:>10} {wr_str:>20} {pf_str:>20} {pnl_str:>22} {dd_str:>20}  {verdict_change}")

    # Overall summary
    print(f"\n  {'=' * 90}")
    print(f"  {Colors.BOLD}OVERALL SUMMARY:{Colors.END}")

    d_total_pnl = curr_total_pnl - base_total_pnl
    d_total_pnl_pct = (curr_total_pnl / 100) - (base_total_pnl / 100)  # as % of $10k

    print(f"    Baseline total P&L: ${base_total_pnl:,.2f} ({base_total_pnl / 100:+.1f}%)")
    print(f"    Current total P&L:  ${curr_total_pnl:,.2f} ({curr_total_pnl / 100:+.1f}%)")

    delta_color = Colors.GREEN if d_total_pnl > 0 else Colors.RED
    print(f"    Delta:              {delta_color}${d_total_pnl:+,.2f}{Colors.END}")

    print(f"    Baseline trades:    {base_total_trades}")
    print(f"    Current trades:     {curr_total_trades} ({curr_total_trades - base_total_trades:+d})")
    print(f"    Improvements:       {Colors.GREEN}{improvements}{Colors.END}")
    print(f"    Regressions:        {Colors.RED}{regressions}{Colors.END}")

    if d_total_pnl > 0:
        print(f"\n  {Colors.GREEN}RESULT: v2.1 OUTPERFORMS baseline by ${d_total_pnl:+,.2f}{Colors.END}")
    elif d_total_pnl < 0:
        print(f"\n  {Colors.RED}RESULT: v2.1 UNDERPERFORMS baseline by ${d_total_pnl:+,.2f}{Colors.END}")
    else:
        print(f"\n  {Colors.YELLOW}RESULT: No change from baseline{Colors.END}")

    print(f"{'=' * 100}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two backtest baselines")
    parser.add_argument("baseline", help="Path to baseline JSON file")
    parser.add_argument("current", help="Path to current/new JSON file")
    args = parser.parse_args()

    compare(args.baseline, args.current)


if __name__ == "__main__":
    main()
