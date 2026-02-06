import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from scanners.insider import InsiderScanner, InsiderCluster, InsiderTransaction, MockInsiderProvider
from core.enums import Market

logging.basicConfig(level=logging.INFO)


class MockDataProvider:
    """Mock price data provider."""

    async def get_bars(self, symbol, timeframe, limit):
        # Price around $50
        prices = [50 + (i * 0.1) for i in range(limit)]
        data = {
            'open': [p - 0.2 for p in prices],
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [1000000] * limit,
        }
        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None


def create_test_cluster(
    symbol: str,
    insider_count: int,
    total_value: float,
    avg_price: float,
    days_span: int = 10,
    titles: List[str] = None
) -> InsiderCluster:
    """Helper to create test clusters."""

    if titles is None:
        titles = ["CEO", "CFO", "VP Sales", "Director", "Director"][:insider_count]

    transactions = []
    for i in range(insider_count):
        txn = InsiderTransaction(
            filing_date=datetime.now() - timedelta(days=i),
            symbol=symbol,
            company=f"{symbol} Inc.",
            insider_name=f"Insider {i+1}",
            insider_title=titles[i] if i < len(titles) else "Director",
            transaction_type="P",
            shares=int(total_value / insider_count / avg_price),
            price=avg_price,
            value=total_value / insider_count
        )
        transactions.append(txn)

    return InsiderCluster(
        symbol=symbol,
        company=f"{symbol} Inc.",
        insider_count=insider_count,
        total_shares=int(total_value / avg_price),
        total_value=total_value,
        avg_price=avg_price,
        transactions=transactions,
        days_span=days_span,
        score=0  # Will be calculated
    )


async def test():
    print("=" * 60)
    print("INSIDER CLUSTER SCANNER TEST")
    print("=" * 60)

    price_provider = MockDataProvider()

    # Test 1: Strong cluster (5 insiders, $500k, CEO+CFO)
    print("\n--- TEST 1: Strong Insider Cluster ---")
    cluster1 = create_test_cluster(
        symbol="ACME",
        insider_count=5,
        total_value=500000,
        avg_price=48.0,  # Close to current price of ~52
        titles=["CEO", "CFO", "VP Operations", "Director", "Director"]
    )

    insider_provider = MockInsiderProvider(clusters=[cluster1])
    scanner = InsiderScanner(price_provider, insider_provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 1 LONG opportunity (strong cluster)")
    print(f"Result: {len(opps)} opportunities")
    if opps:
        opp = opps[0]
        print(f"  Symbol: {opp.symbol}")
        print(f"  Direction: {opp.direction}")
        print(f"  Insider count: {opp.edge_data.get('insider_count')}")
        print(f"  Total value: ${opp.edge_data.get('total_value'):,.0f}")
        print(f"  Cluster score: {opp.edge_data.get('cluster_score')}")
        print(f"  Price vs insider: {opp.edge_data.get('price_vs_insider_pct'):+.1f}%")

    # Test 2: Weak cluster (only 2 insiders - below threshold)
    print("\n--- TEST 2: Weak Cluster (Below Threshold) ---")
    cluster2 = create_test_cluster(
        symbol="WEAK",
        insider_count=2,  # Below minimum of 3
        total_value=200000,
        avg_price=50.0
    )

    insider_provider = MockInsiderProvider(clusters=[cluster2])
    scanner = InsiderScanner(price_provider, insider_provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (only 2 insiders)")
    print(f"Result: {len(opps)} opportunities")

    # Test 3: Low value cluster
    print("\n--- TEST 3: Low Value Cluster ---")
    cluster3 = create_test_cluster(
        symbol="SMALL",
        insider_count=3,
        total_value=50000,  # Below $100k minimum
        avg_price=50.0
    )

    insider_provider = MockInsiderProvider(clusters=[cluster3])
    scanner = InsiderScanner(price_provider, insider_provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (value below minimum)")
    print(f"Result: {len(opps)} opportunities")

    # Test 4: Multiple clusters
    print("\n--- TEST 4: Multiple Clusters ---")
    clusters = [
        create_test_cluster("AAPL", 4, 1000000, 49.0, titles=["CEO", "CFO", "COO", "VP"]),
        create_test_cluster("MSFT", 3, 300000, 50.0, titles=["CFO", "Director", "Director"]),
        create_test_cluster("NVDA", 5, 750000, 47.0, titles=["CEO", "CFO", "VP", "VP", "Director"]),
    ]

    insider_provider = MockInsiderProvider(clusters=clusters)
    scanner = InsiderScanner(price_provider, insider_provider, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 3 opportunities (all valid clusters)")
    print(f"Result: {len(opps)} opportunities")
    for opp in opps:
        print(f"  {opp.symbol}: Score {opp.edge_data.get('cluster_score')}, "
              f"{opp.edge_data.get('insider_count')} insiders, "
              f"${opp.edge_data.get('total_value'):,.0f}")

    # Test 5: No insider provider (production would need real API)
    print("\n--- TEST 5: No Insider Provider ---")
    scanner = InsiderScanner(price_provider, insider_data_provider=None, markets=[Market.US_STOCKS])

    opps = await scanner.scan()
    print(f"Expected: 0 opportunities (no data provider)")
    print(f"Result: {len(opps)} opportunities")


if __name__ == "__main__":
    asyncio.run(test())
