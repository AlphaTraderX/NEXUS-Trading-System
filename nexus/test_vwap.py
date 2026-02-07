# test_vwap.py
"""Verification script for VWAP scanner calculations."""
import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add project root so imports work when run from nexus/
sys.path.insert(0, str(Path(__file__).parent))

from nexus.scanners.vwap import VWAPScanner


# Create mock data provider
class MockDataProvider:
    async def get_bars(self, symbol, timeframe, limit):
        # Build enough bars so scan passes the 20-bar minimum
        # Last bar is extreme so deviation > 2σ
        n = max(limit, 25)
        base_high = [100.5, 101.0, 101.5, 102.0] * (n // 4 + 1)
        base_low = [99.5, 100.0, 100.5, 101.0] * (n // 4 + 1)
        base_close = [100.0, 100.5, 101.0, 101.5] * (n // 4 + 1)
        high = base_high[:n]
        low = base_low[:n]
        close = base_close[:n]
        # Make last bar extreme (price well above VWAP)
        high[-1] = 105.0
        low[-1] = 104.0
        close[-1] = 104.5
        data = {
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000] * n,
        }
        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None


async def test():
    provider = MockDataProvider()
    scanner = VWAPScanner(provider)

    # Test VWAP calculation
    bars = await provider.get_bars("TEST", "5m", 5)
    vwap = scanner._calculate_vwap(bars)
    print(f"VWAP: {vwap}")

    vwap_std = scanner._calculate_vwap_std(bars, vwap)
    print(f"VWAP Std: {vwap_std}")

    # Current price deviation
    current = bars["close"].iloc[-1]
    deviation = (current - vwap) / vwap_std
    print(f"Current: {current}, Deviation: {deviation:.2f} std devs")

    # Run scan (mock returns 25 bars per symbol so >= 20)
    opps = await scanner.scan()
    print(f"Opportunities found: {len(opps)}")
    for opp in opps:
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price}")


if __name__ == "__main__":
    asyncio.run(test())
