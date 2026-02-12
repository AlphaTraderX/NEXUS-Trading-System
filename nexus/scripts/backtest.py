"""
Backtest entry point – delegates to backtest_single or backtest_all.

Usage::

    python -m nexus.scripts.backtest --edge vwap_deviation
    python -m nexus.scripts.backtest --all
    python -m nexus.scripts.backtest --all --start 2024-01-01 --end 2024-06-30

See also:
    nexus/scripts/backtest_single.py  – single-edge backtest
    nexus/scripts/backtest_all.py     – all-edge validation report
"""

import sys


def main() -> None:
    if "--all" in sys.argv:
        sys.argv.remove("--all")
        from nexus.scripts.backtest_all import main as run_all

        run_all()
    elif "--edge" in sys.argv:
        from nexus.scripts.backtest_single import main as run_single

        run_single()
    else:
        print("NEXUS Backtester")
        print()
        print("Single edge:")
        print("  python -m nexus.scripts.backtest --edge vwap_deviation")
        print("  python -m nexus.scripts.backtest --edge rsi_extreme --symbol SPY")
        print()
        print("All edges:")
        print("  python -m nexus.scripts.backtest --all")
        print("  python -m nexus.scripts.backtest --all --start 2024-01-01 --end 2024-06-30")
        print()
        print("Options: --start, --end, --symbol, --timeframe, --balance, --risk")


if __name__ == "__main__":
    main()
