import asyncio
import logging
import pandas as pd
from datetime import datetime, time
import pytz
from scanners.session import PowerHourScanner, LondonOpenScanner, NYOpenScanner, SessionScanner
from core.enums import Market

logging.basicConfig(level=logging.INFO)

class MockDataProvider:
    """Mock with bullish intraday data."""

    async def get_bars(self, symbol, timeframe, limit):
        if timeframe == "5m":
            # Bullish intraday - open 100, trending up to 102
            prices = [100 + (i * 0.1) for i in range(limit)]
            data = {
                'open': prices,
                'high': [p + 0.3 for p in prices],
                'low': [p - 0.2 for p in prices],
                'close': [p + 0.2 for p in prices],
                'volume': [1000000] * limit,
            }
        else:  # 1H bars for London Open
            # Asian range: 1.1000-1.1050, then breakout to 1.1080
            data = {
                'open':  [1.1010, 1.1020, 1.1015, 1.1030, 1.1025, 1.1040, 1.1050, 1.1055, 1.1060, 1.1070, 1.1075, 1.1080],
                'high':  [1.1030, 1.1035, 1.1025, 1.1045, 1.1040, 1.1055, 1.1060, 1.1065, 1.1075, 1.1085, 1.1090, 1.1095],
                'low':   [1.1000, 1.1010, 1.1005, 1.1020, 1.1015, 1.1030, 1.1045, 1.1050, 1.1055, 1.1065, 1.1070, 1.1075],
                'close': [1.1020, 1.1015, 1.1025, 1.1025, 1.1035, 1.1050, 1.1055, 1.1060, 1.1070, 1.1075, 1.1080, 1.1090],
                'volume': [100000] * 12,
            }
            # Extend to requested limit
            while len(data['open']) < limit:
                for key in data:
                    data[key].append(data[key][-1])

        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None

async def test():
    print("=" * 60)
    print("SESSION SCANNERS TEST")
    print("=" * 60)

    provider = MockDataProvider()

    # Test Power Hour (force active by mocking time check)
    print("\n--- POWER HOUR SCANNER ---")
    scanner = PowerHourScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True  # Force active

    opps = await scanner.scan()
    print(f"Opportunities: {len(opps)}")
    for opp in opps[:3]:
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")
        print(f"    Day move: {opp.edge_data.get('day_move_pct'):+.2f}%")

    # Test London Open
    print("\n--- LONDON OPEN SCANNER ---")
    scanner = LondonOpenScanner(provider, markets=[Market.FOREX_MAJORS])
    scanner.is_active = lambda ts=None: True  # Force active

    opps = await scanner.scan()
    print(f"Opportunities: {len(opps)}")
    for opp in opps[:3]:
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.5f}")
        print(f"    Asian range: {opp.edge_data.get('asian_low'):.5f}-{opp.edge_data.get('asian_high'):.5f}")

    # Test NY Open
    print("\n--- NY OPEN SCANNER ---")
    scanner = NYOpenScanner(provider, markets=[Market.US_STOCKS])
    scanner.is_active = lambda ts=None: True  # Force active

    opps = await scanner.scan()
    print(f"Opportunities: {len(opps)}")
    for opp in opps[:3]:
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")
        print(f"    Opening range: {opp.edge_data.get('opening_low'):.2f}-{opp.edge_data.get('opening_high'):.2f}")

    # Test combined SessionScanner
    print("\n--- COMBINED SESSION SCANNER ---")
    scanner = SessionScanner(provider)
    # Force all sub-scanners active
    scanner.power_hour.is_active = lambda ts=None: True
    scanner.london_open.is_active = lambda ts=None: True
    scanner.ny_open.is_active = lambda ts=None: True

    opps = await scanner.scan()
    print(f"Total opportunities from all sessions: {len(opps)}")

if __name__ == "__main__":
    asyncio.run(test())
