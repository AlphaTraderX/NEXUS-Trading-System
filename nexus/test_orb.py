import asyncio
import logging
import pandas as pd
from scanners.orb import ORBScanner
from core.enums import Market

logging.basicConfig(level=logging.INFO)

class MockDataProvider:
    """Mock with confirmed ORB breakout setup."""

    def __init__(self, breakout_type="up_confirmed"):
        self.breakout_type = breakout_type

    async def get_bars(self, symbol, timeframe, limit):
        if self.breakout_type == "up_confirmed":
            # Opening range: 100-101, then breakout to 102 with high volume (>120% avg), above VWAP
            opening = [100.2, 100.5, 100.3, 100.8, 100.6, 101.0]  # First 6 bars (opening range)
            continuation = ([101.2, 101.5, 101.8, 102.0, 102.2] * 10)[:limit - 6]
            closes = (opening + continuation)[:limit]
            n = len(closes)
            # Volume: many bars at 500k, last 10 at 1M so last-bar/avg > 1.2
            vol = [500000] * (n - 10) + [1000000] * 10
            data = {
                'open': [c - 0.1 for c in closes],
                'high': [c + 0.2 for c in closes],
                'low': [c - 0.2 for c in closes],
                'close': closes,
                'volume': vol,
            }
        elif self.breakout_type == "up_no_volume":
            # Breakout but LOW volume (should be rejected)
            opening = [100.2, 100.5, 100.3, 100.8, 100.6, 101.0]
            continuation = ([101.2, 101.5, 101.8, 102.0, 102.2] * 10)[:limit - 6]
            closes = (opening + continuation)[:limit]
            n = len(closes)
            data = {
                'open': [c - 0.1 for c in closes],
                'high': [c + 0.2 for c in closes],
                'low': [c - 0.2 for c in closes],
                'close': closes,
                'volume': [500000] * n,
            }
        elif self.breakout_type == "up_wrong_vwap":
            # Breakout UP but price below VWAP (should be rejected)
            # VWAP will be high because of high prices early in the day
            base = [105.0, 104.5, 104.0, 103.5, 103.0, 102.5] + [101.0, 101.2, 101.5, 101.8] * 12
            closes = base[:limit]
            n = len(closes)
            data = {
                'open': [c + 0.1 for c in closes],
                'high': [c + 0.3 for c in closes],
                'low': [c - 0.2 for c in closes],
                'close': closes,
                'volume': [500000] * 6 + [800000] * (n - 6),
            }
        else:
            # No breakout
            n = limit
            closes = [100.5] * n
            data = {
                'open': [100.4] * n,
                'high': [100.8] * n,
                'low': [100.2] * n,
                'close': closes,
                'volume': [500000] * n,
            }

        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None

async def test():
    print("=" * 60)
    print("ORB SCANNER TEST")
    print("=" * 60)

    # Test 1: Confirmed breakout UP (volume + VWAP aligned)
    print("\n--- TEST 1: Confirmed Breakout UP ---")
    provider = MockDataProvider(breakout_type="up_confirmed")
    scanner = ORBScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True

    opps = await scanner.scan()
    print(f"Expected: Opportunities (all filters pass)")
    print(f"Result: {len(opps)} opportunities")
    if opps:
        opp = opps[0]
        print(f"  Direction: {opp.direction}")
        print(f"  Volume ratio: {opp.edge_data.get('volume_ratio')}")
        print(f"  VWAP confirmed: {opp.edge_data.get('vwap_confirmed')}")

    # Test 2: Breakout but LOW volume (should reject)
    print("\n--- TEST 2: Breakout But Low Volume ---")
    provider = MockDataProvider(breakout_type="up_no_volume")
    scanner = ORBScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (volume filter)")
    print(f"Result: {len(opps)} opportunities")

    # Test 3: Breakout but wrong VWAP (should reject)
    print("\n--- TEST 3: Breakout But Wrong VWAP ---")
    provider = MockDataProvider(breakout_type="up_wrong_vwap")
    scanner = ORBScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (VWAP filter)")
    print(f"Result: {len(opps)} opportunities")

    # Test 4: No breakout
    print("\n--- TEST 4: No Breakout ---")
    provider = MockDataProvider(breakout_type="none")
    scanner = ORBScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (no breakout)")
    print(f"Result: {len(opps)} opportunities")

if __name__ == "__main__":
    asyncio.run(test())
