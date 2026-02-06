"""
Backtest script: run strategies over historical data.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def main() -> None:
    """Run backtest."""
    print("NEXUS backtest: not implemented yet.")
    print("Usage: python scripts/backtest.py --start YYYY-MM-DD --end YYYY-MM-DD")


if __name__ == "__main__":
    main()
