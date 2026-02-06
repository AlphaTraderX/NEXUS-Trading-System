import asyncio
import logging
import pandas as pd
from scanners.rsi import RSIScanner
from core.enums import Market

logging.basicConfig(level=logging.INFO)

class MockDataProvider:
    """Mock that returns data with extreme RSI conditions."""

    async def get_bars(self, symbol, timeframe, limit):
        # Create data that will produce OVERSOLD RSI (consecutive down days)
        # Start at 100, drop each day
        prices = [100 - i*2 for i in range(25)]  # 100, 98, 96, 94... down trend

        data = {
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 25,
        }
        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None

async def test():
    print("=" * 60)
    print("RSI SCANNER TEST")
    print("=" * 60)

    provider = MockDataProvider()
    scanner = RSIScanner(provider, markets=[Market.US_STOCKS])

    # Test RSI calculation
    bars = await provider.get_bars("TEST", "1D", 25)
    rsi_series = scanner._calculate_rsi(bars, scanner.rsi_period)

    print(f"\nPrice trend: {bars['close'].iloc[0]:.0f} -> {bars['close'].iloc[-1]:.0f} (downtrend)")
    print(f"RSI Period: {scanner.rsi_period}")
    print(f"Current RSI: {rsi_series.iloc[-1]:.2f}")
    print(f"Oversold threshold: {scanner.oversold_threshold}")
    print(f"Overbought threshold: {scanner.overbought_threshold}")

    # Run scan
    print("\n" + "-" * 40)
    print("Running scan...")
    opps = await scanner.scan()

    print(f"\nOpportunities found: {len(opps)}")
    for opp in opps:
        print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")
        print(f"    RSI: {opp.edge_data.get('rsi')}")
        print(f"    Stop: {opp.stop_loss:.2f}, Target: {opp.take_profit:.2f}")

if __name__ == "__main__":
    asyncio.run(test())
