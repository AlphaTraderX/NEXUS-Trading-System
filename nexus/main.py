"""
NEXUS Trading System - Main Entry Point

Usage:
    python -m nexus.main                    # Run scheduler (default)
    python -m nexus.main --test             # Run scanner test
    python -m nexus.main --status           # Show market status
    python -m nexus.main --interval 30      # Scan every 30 seconds
"""

import argparse
import asyncio
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from nexus.scheduler.market_hours import MarketHours
from nexus.scheduler.main_loop import NexusScheduler

UTC = ZoneInfo("UTC")


def show_market_status():
    """Display current market status."""
    status = MarketHours.get_market_status()
    next_open = MarketHours.get_next_market_open()
    
    print("\n" + "=" * 60)
    print("              NEXUS MARKET STATUS")
    print("=" * 60)
    print(f"  Time: {status['timestamp']}")
    print(f"  Day: {status['weekday']}")
    print()
    print("  MARKETS:")
    for market, is_open in status["markets"].items():
        state = "OPEN" if is_open else "CLOSED"
        emoji = "[OK]" if is_open else "[--]"
        print(f"    {emoji} {market}: {state}")
    print()
    print(f"  ACTIVE SESSIONS: {', '.join(status['active_sessions']) or 'None'}")
    print()
    print(f"  NEXT: {next_open['message']}")
    print("=" * 60 + "\n")


async def run_scanner_test():
    """Run a quick scanner test."""
    # Import here to avoid circular imports
    from nexus.scripts.test_scanners_live import main as test_main
    await test_main()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NEXUS Automated Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nexus.main                 Start the scheduler
  python -m nexus.main --status        Show market status
  python -m nexus.main --interval 30   Scan every 30 seconds
  python -m nexus.main --test          Run scanner test
        """
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current market status and exit"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run scanner test and exit"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Scan interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        show_market_status()
        return
    
    # Handle test command
    if args.test:
        await run_scanner_test()
        return
    
    # Run scheduler
    scheduler = NexusScheduler(
        scan_interval=args.interval,
        verbose=not args.quiet,
    )
    
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if scheduler.running:
            await scheduler.stop()


if __name__ == "__main__":
    # Set up Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    asyncio.run(main())
