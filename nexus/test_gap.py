import asyncio
import logging
import pandas as pd
from nexus.scanners.gap import GapScanner
from nexus.core.enums import Market

logging.basicConfig(level=logging.INFO)


class MockDataProvider:
    """Mock that returns data with gap scenarios."""

    def __init__(self, gap_type="up"):
        self.gap_type = gap_type

    async def get_bars(self, symbol, timeframe, limit):
        if self.gap_type == "up":
            # Gap UP scenario: yesterday closed at 100, today opened at 102 (2% gap up)
            data = {
                'open':  [99.0, 100.5, 101.0, 100.0, 102.0],   # Today opened at 102
                'high':  [100.5, 101.5, 102.0, 101.0, 102.5],
                'low':   [98.5, 100.0, 100.5, 99.5, 101.0],
                'close': [100.0, 101.0, 101.5, 100.0, 101.5],  # Yesterday closed at 100, today at 101.5
                'volume': [1000000] * 5,
            }
        else:
            # Gap DOWN scenario: yesterday closed at 100, today opened at 98 (2% gap down)
            data = {
                'open':  [99.0, 100.5, 101.0, 100.0, 98.0],    # Today opened at 98
                'high':  [100.5, 101.5, 102.0, 101.0, 99.0],
                'low':   [98.5, 100.0, 100.5, 99.5, 97.5],
                'close': [100.0, 101.0, 101.5, 100.0, 98.5],   # Yesterday closed at 100, today at 98.5
                'volume': [1000000] * 5,
            }
        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None


async def test():
    print("=" * 60)
    print("GAP FILL SCANNER TEST")
    print("=" * 60)

    # Test GAP UP scenario
    print("\n--- GAP UP SCENARIO ---")
    provider = MockDataProvider(gap_type="up")
    scanner = GapScanner(provider, markets=[Market.US_STOCKS])

    bars = await provider.get_bars("TEST", "1D", 5)
    prev_close = bars['close'].iloc[-2]
    today_open = bars['open'].iloc[-1]
    gap_pct = ((today_open - prev_close) / prev_close) * 100

    print(f"Yesterday close: {prev_close}")
    print(f"Today open: {today_open}")
    print(f"Gap: {gap_pct:+.2f}%")
    print(f"Expected direction: SHORT (fill the gap down)")

    opps = await scanner.scan()
    print(f"\nOpportunities found: {len(opps)}")
    for opp in opps[:3]:  # Show first 3
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")
        print(f"    Gap: {opp.edge_data.get('gap_pct'):+.2f}% | Target: {opp.take_profit:.2f}")

    # Test GAP DOWN scenario
    print("\n--- GAP DOWN SCENARIO ---")
    provider = MockDataProvider(gap_type="down")
    scanner = GapScanner(provider, markets=[Market.US_STOCKS])

    bars = await provider.get_bars("TEST", "1D", 5)
    prev_close = bars['close'].iloc[-2]
    today_open = bars['open'].iloc[-1]
    gap_pct = ((today_open - prev_close) / prev_close) * 100

    print(f"Yesterday close: {prev_close}")
    print(f"Today open: {today_open}")
    print(f"Gap: {gap_pct:+.2f}%")
    print(f"Expected direction: LONG (fill the gap up)")

    opps = await scanner.scan()
    print(f"\nOpportunities found: {len(opps)}")
    for opp in opps[:3]:  # Show first 3
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")
        print(f"    Gap: {opp.edge_data.get('gap_pct'):+.2f}% | Target: {opp.take_profit:.2f}")


if __name__ == "__main__":
    asyncio.run(test())
