"""
NEXUS Scanner Integration Test - LIVE DATA

Runs scanners against real market data from Massive and OANDA.
Tests that the full pipeline works: Provider → Scanner → Opportunity.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# API Credentials - MOVE TO .env IN PRODUCTION
MASSIVE_API_KEY = "***REDACTED***"
OANDA_API_KEY = "***REDACTED***"
OANDA_ACCOUNT_ID = "101-004-38468446-001"


async def test_data_fetch():
    """Test that we can fetch real data even when scanners aren't active."""
    from nexus.data.massive import MassiveProvider
    from nexus.scanners.calendar import TurnOfMonthScanner

    print("\n" + "=" * 60)
    print("📊 TESTING REAL DATA FETCH (Always runs)")
    print("=" * 60)

    provider = MassiveProvider(api_key=MASSIVE_API_KEY)
    connected = await provider.connect()

    if not connected:
        print("❌ Failed to connect")
        return False

    print("✅ Connected to Massive API\n")

    try:
        scanner = TurnOfMonthScanner(data_provider=provider)

        for symbol in ["SPY", "QQQ", "IWM"]:
            price = await scanner.get_current_price(symbol)
            bars = await scanner.get_bars(symbol, "1D", 20)
            atr = scanner.calculate_atr(bars, 14)

            print(f"   {symbol}:")
            print(f"      Price: ${price:.2f}")
            print(f"      ATR(14): ${atr:.2f}")
            print(f"      ATR %: {(atr/price)*100:.2f}%")
            print()

        return True

    finally:
        await provider.disconnect()
        print("✅ Disconnected")


async def test_single_scanner_with_massive():
    """Test a single scanner with Massive (US stocks) data."""
    from nexus.data.massive import MassiveProvider
    from nexus.scanners.calendar import TurnOfMonthScanner, MonthEndScanner

    print("\n" + "=" * 60)
    print("🔵 TESTING SCANNERS WITH MASSIVE (US STOCKS)")
    print("=" * 60)

    # Create and connect provider
    provider = MassiveProvider(api_key=MASSIVE_API_KEY)
    connected = await provider.connect()

    if not connected:
        print("❌ Failed to connect to Massive")
        return False

    print("✅ Connected to Massive API\n")

    try:
        # Test Turn of Month Scanner
        print("-" * 40)
        print("📅 Turn of Month Scanner")
        print("-" * 40)

        tom_scanner = TurnOfMonthScanner(data_provider=provider)
        status = tom_scanner.get_tom_status()
        print(f"   Is Active: {status['is_active']}")
        print(f"   TOM Day: {status['tom_day']}")
        print(f"   Days Remaining: {status['days_remaining']}")
        print(f"   Next Window: {status['next_window']}")

        opportunities = await tom_scanner.scan()
        print(f"\n   Opportunities Found: {len(opportunities)}")

        for opp in opportunities:
            print(f"\n   📊 {opp.symbol}:")
            print(f"      Direction: {opp.direction}")
            print(f"      Entry: ${opp.entry_price:.2f}")
            print(f"      Stop: ${opp.stop_loss:.2f}")
            print(f"      Target: ${opp.take_profit:.2f}")
            print(f"      ATR: ${opp.edge_data.get('atr', 0):.2f}")

        # Test Month End Scanner
        print("\n" + "-" * 40)
        print("📅 Month End Scanner")
        print("-" * 40)

        me_scanner = MonthEndScanner(data_provider=provider)
        status = me_scanner.get_month_end_status()
        print(f"   Is Active: {status['is_active']}")
        print(f"   Month End Day: {status['month_end_day']}")

        opportunities = await me_scanner.scan()
        print(f"\n   Opportunities Found: {len(opportunities)}")

        for opp in opportunities:
            print(f"\n   📊 {opp.symbol}:")
            print(f"      Direction: {opp.direction}")
            print(f"      Entry: ${opp.entry_price:.2f}")

        return True

    finally:
        await provider.disconnect()
        print("\n✅ Disconnected from Massive")


async def test_forex_scanners_with_oanda():
    """Test forex scanners with OANDA data."""
    from nexus.data.oanda import OANDAProvider
    from nexus.scanners.session import LondonOpenScanner, NYOpenScanner, AsianRangeScanner, PowerHourScanner

    print("\n" + "=" * 60)
    print("🟢 TESTING SCANNERS WITH OANDA (FOREX)")
    print("=" * 60)

    # Create and connect provider
    provider = OANDAProvider(
        api_key=OANDA_API_KEY,
        account_id=OANDA_ACCOUNT_ID,
        practice=True
    )
    connected = await provider.connect()

    if not connected:
        print("❌ Failed to connect to OANDA")
        return False

    print("✅ Connected to OANDA Practice API\n")

    try:
        # Test each forex scanner
        scanners = [
            ("🌅 London Open", LondonOpenScanner(data_provider=provider)),
            ("🗽 NY Open", NYOpenScanner(data_provider=provider)),
            ("🌏 Asian Range", AsianRangeScanner(data_provider=provider)),
            ("⚡ Power Hour", PowerHourScanner(data_provider=provider)),
        ]

        now = datetime.now(timezone.utc)

        for name, scanner in scanners:
            print("-" * 40)
            print(f"{name} Scanner")
            print("-" * 40)

            is_active = scanner.is_active(now)
            print(f"   Is Active: {is_active}")

            if is_active:
                opportunities = await scanner.scan()
                print(f"   Opportunities: {len(opportunities)}")
                for opp in opportunities[:3]:  # Limit output
                    print(f"      {opp.symbol}: {opp.direction} @ {opp.entry_price:.5f}")
            else:
                print("   (Not active at current time)")
            print()

        return True

    finally:
        await provider.disconnect()
        print("✅ Disconnected from OANDA")


async def test_orchestrator():
    """Test full orchestrator with real data."""
    from nexus.data.massive import MassiveProvider
    from nexus.scanners.orchestrator import ScannerOrchestrator

    print("\n" + "=" * 60)
    print("🎯 TESTING FULL ORCHESTRATOR")
    print("=" * 60)

    # Create and connect provider
    provider = MassiveProvider(api_key=MASSIVE_API_KEY)
    connected = await provider.connect()

    if not connected:
        print("❌ Failed to connect to Massive")
        return False

    print("✅ Connected to Massive API\n")

    try:
        # Create orchestrator
        orchestrator = ScannerOrchestrator(data_provider=provider)

        # Show scanner status
        status = orchestrator.get_scanner_status()
        print(f"Total Scanners: {status['total_scanners']}")
        print(f"\nActive Scanners:")

        for name, info in status['scanners'].items():
            active_str = "✅" if info['active'] else "⏸️"
            print(f"   {active_str} {name}: {info['edge_type']}")

        # Run all scanners
        print("\n" + "-" * 40)
        print("Running scan cycle...")
        print("-" * 40)

        opportunities = await orchestrator.run_all_scanners()

        print(f"\n🎯 Total Opportunities: {len(opportunities)}")

        for opp in opportunities:
            print(f"\n   {opp.symbol} ({opp.primary_edge}):")
            print(f"      Direction: {opp.direction}")
            print(f"      Entry: {opp.entry_price:.4f}")
            print(f"      Stop: {opp.stop_loss:.4f}")
            print(f"      Target: {opp.take_profit:.4f}")

        return True

    finally:
        await provider.disconnect()
        print("\n✅ Disconnected from Massive")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 NEXUS SCANNER LIVE DATA TEST")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    results = {}

    # Test data fetch (always runs)
    results['data_fetch'] = await test_data_fetch()

    # Test 1: Single scanner with Massive
    results['massive_scanners'] = await test_single_scanner_with_massive()

    # Test 2: Forex scanners with OANDA
    results['oanda_scanners'] = await test_forex_scanners_with_oanda()

    # Test 3: Full orchestrator (optional - may take longer)
    # results['orchestrator'] = await test_orchestrator()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - SCANNERS WIRED TO REAL DATA!")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
