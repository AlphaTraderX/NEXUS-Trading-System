import asyncio
import logging
import pandas as pd
from scanners.bollinger import BollingerScanner
from core.enums import Market

logging.basicConfig(level=logging.INFO)

class MockDataProvider:
    """Mock with different Bollinger scenarios."""

    def __init__(self, scenario="lower_touch_reversal"):
        self.scenario = scenario

    async def get_bars(self, symbol, timeframe, limit):
        if self.scenario == "lower_touch_reversal":
            # Ranging market (low volatility), touch lower band, bullish reversal
            # Price oscillates 98-102, then dips to touch lower band, closes higher
            base_prices = [100 + (i % 3) - 1 for i in range(limit - 1)]  # 99, 100, 101, 99...
            base_prices.append(99.5)  # Last bar: touched low but closed higher

            data = {
                'open': [p - 0.2 for p in base_prices],
                'high': [p + 0.5 for p in base_prices[:-1]] + [100.0],  # Last high = 100
                'low': [p - 0.5 for p in base_prices[:-1]] + [97.0],   # Last low = 97 (touches band)
                'close': base_prices[:-1] + [99.5],  # Reversal: closed above open
                'volume': [1000000] * limit,
            }
        elif self.scenario == "upper_touch_reversal":
            # Touch upper band with bearish reversal (close < prev_close)
            base_prices = [100 + (i % 3) - 1 for i in range(limit - 1)]  # last of these is 99
            prev_close = base_prices[-1]  # 99

            data = {
                'open': [p + 0.2 for p in base_prices] + [102.0],
                'high': [p + 0.5 for p in base_prices] + [103.0],   # Last high touches upper band
                'low': [p - 0.5 for p in base_prices] + [101.0],
                'close': base_prices + [98.5],  # Bearish reversal: closed below prev_close (99)
                'volume': [1000000] * limit,
            }
        elif self.scenario == "touch_no_reversal":
            # Touch lower band but NO reversal (continued down)
            base_prices = [100 - (i * 0.1) for i in range(limit)]  # Downtrend

            data = {
                'open': [p + 0.1 for p in base_prices],
                'high': [p + 0.2 for p in base_prices],
                'low': [p - 0.3 for p in base_prices],
                'close': base_prices,  # No reversal - continued lower
                'volume': [1000000] * limit,
            }
        elif self.scenario == "trending":
            # Strong uptrend (ADX > 25) - should reject
            base_prices = [100 + (i * 0.5) for i in range(limit)]  # Strong uptrend

            data = {
                'open': [p - 0.3 for p in base_prices],
                'high': [p + 0.5 for p in base_prices],
                'low': [p - 0.2 for p in base_prices],
                'close': base_prices,
                'volume': [1000000] * limit,
            }
        else:
            # No touch
            data = {
                'open': [100.0] * limit,
                'high': [100.5] * limit,
                'low': [99.5] * limit,
                'close': [100.0] * limit,
                'volume': [1000000] * limit,
            }

        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None

async def test():
    print("=" * 60)
    print("BOLLINGER SCANNER TEST")
    print("=" * 60)

    # Test 1: Lower band touch with reversal (should signal LONG)
    print("\n--- TEST 1: Lower Band Touch + Reversal ---")
    provider = MockDataProvider(scenario="lower_touch_reversal")
    scanner = BollingerScanner(provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: LONG opportunities (touch lower, reversal confirmed)")
    print(f"Result: {len(opps)} opportunities")
    if opps:
        opp = opps[0]
        print(f"  Direction: {opp.direction}")
        print(f"  Band touched: {opp.edge_data.get('band_touched')}")
        print(f"  ADX: {opp.edge_data.get('adx')}")

    # Test 2: Upper band touch with reversal (should signal SHORT)
    print("\n--- TEST 2: Upper Band Touch + Reversal ---")
    provider = MockDataProvider(scenario="upper_touch_reversal")
    scanner = BollingerScanner(provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: SHORT opportunities (touch upper, reversal confirmed)")
    print(f"Result: {len(opps)} opportunities")
    if opps:
        opp = opps[0]
        print(f"  Direction: {opp.direction}")
        print(f"  Band touched: {opp.edge_data.get('band_touched')}")

    # Test 3: Touch but no reversal (should reject)
    print("\n--- TEST 3: Touch But No Reversal ---")
    provider = MockDataProvider(scenario="touch_no_reversal")
    scanner = BollingerScanner(provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (no reversal candle)")
    print(f"Result: {len(opps)} opportunities")

    # Test 4: Trending market (should reject even with touch)
    print("\n--- TEST 4: Trending Market ---")
    provider = MockDataProvider(scenario="trending")
    scanner = BollingerScanner(provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (ADX > 25, trending)")
    print(f"Result: {len(opps)} opportunities")

if __name__ == "__main__":
    asyncio.run(test())
