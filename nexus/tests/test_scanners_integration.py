"""
Integration test for all 13 scanners.
Verifies each scanner can be instantiated and called.
"""

import asyncio
import os
import sys
from datetime import datetime
import pytz

# Add nexus package to path so scanners can be imported
_nexus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _nexus_root not in sys.path:
    sys.path.insert(0, _nexus_root)


async def test_all_scanners():
    """Test all 13 scanners can run without errors."""

    print("=" * 60)
    print("NEXUS SCANNER INTEGRATION TEST")
    print("=" * 60)

    results = []

    # Import all scanners
    from nexus.scanners.calendar import TurnOfMonthScanner, MonthEndScanner
    from nexus.scanners.vwap import VWAPScanner
    from nexus.scanners.rsi import RSIScanner
    from nexus.scanners.gap import GapScanner
    from nexus.scanners.orb import ORBScanner
    from nexus.scanners.bollinger import BollingerScanner
    from nexus.scanners.session import (
        PowerHourScanner,
        LondonOpenScanner,
        NYOpenScanner,
        AsianRangeScanner,
    )
    from nexus.scanners.insider import InsiderScanner
    from nexus.scanners.earnings import EarningsDriftScanner

    scanners = [
        ("Turn of Month", TurnOfMonthScanner()),
        ("Month End", MonthEndScanner()),
        ("VWAP Deviation", VWAPScanner()),
        ("RSI Extreme", RSIScanner()),
        ("Gap Fill", GapScanner()),
        ("ORB", ORBScanner()),
        ("Bollinger Touch", BollingerScanner()),
        ("Power Hour", PowerHourScanner()),
        ("London Open", LondonOpenScanner()),
        ("NY Open", NYOpenScanner()),
        ("Asian Range", AsianRangeScanner()),
        ("Insider Cluster", InsiderScanner()),
        ("Earnings Drift", EarningsDriftScanner()),
    ]

    now = datetime.now(pytz.UTC)

    for name, scanner in scanners:
        try:
            # Test is_active
            is_active = scanner.is_active(now)

            # Test scan
            opportunities = await scanner.scan()

            status = "✅ PASS"
            details = f"Active: {is_active}, Opportunities: {len(opportunities)}"

        except Exception as e:
            status = "❌ FAIL"
            details = str(e)[:50]

        results.append((name, status, details))
        print(f"{status} {name:20} | {details}")

    print("\n" + "=" * 60)
    passed = sum(1 for _, status, _ in results if "PASS" in status)
    print(f"RESULTS: {passed}/{len(results)} scanners passed")
    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(test_all_scanners())
    sys.exit(0 if success else 1)
