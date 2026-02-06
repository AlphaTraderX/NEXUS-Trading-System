"""
Paper trading script: run NEXUS in paper mode.
"""

import asyncio
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


async def run_paper() -> None:
    """Run paper trading loop."""
    print("NEXUS paper trading: not implemented yet.")
    print("Will run scanners -> intelligence -> risk -> execution (paper) -> delivery.")


def main() -> None:
    """Entry point."""
    asyncio.run(run_paper())


if __name__ == "__main__":
    main()
