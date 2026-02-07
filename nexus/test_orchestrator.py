import asyncio
import logging
import pandas as pd
from datetime import datetime
from nexus.scanners.orchestrator import ScannerOrchestrator
from nexus.scanners.insider import MockInsiderProvider, InsiderCluster, InsiderTransaction

logging.basicConfig(level=logging.INFO)

class MockDataProvider:
    """
    Mock data provider that returns different data based on symbol.
    Designed to trigger various scanners.
    """

    async def get_bars(self, symbol, timeframe, limit):
        # Return bullish trending data (will trigger some scanners)
        prices = [100 + (i * 0.2) for i in range(limit)]

        data = {
            'open': [p - 0.1 for p in prices],
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.3 for p in prices],
            'close': prices,
            'volume': [1000000] * limit,
        }
        return pd.DataFrame(data)

    async def get_quote(self, symbol):
        return None

def create_mock_cluster():
    """Create a mock insider cluster."""
    from datetime import timedelta

    transactions = [
        InsiderTransaction(
            filing_date=datetime.now() - timedelta(days=i),
            symbol="TEST",
            company="Test Inc.",
            insider_name=f"Insider {i}",
            insider_title=["CEO", "CFO", "VP", "Director"][i % 4],
            transaction_type="P",
            shares=1000,
            price=100.0,
            value=100000.0
        )
        for i in range(4)
    ]

    return InsiderCluster(
        symbol="TEST",
        company="Test Inc.",
        insider_count=4,
        total_shares=4000,
        total_value=400000.0,
        avg_price=100.0,
        transactions=transactions,
        days_span=4,
        score=0
    )

async def test():
    print("=" * 60)
    print("SCANNER ORCHESTRATOR TEST")
    print("=" * 60)

    # Create providers
    data_provider = MockDataProvider()
    insider_provider = MockInsiderProvider(clusters=[create_mock_cluster()])

    # Create orchestrator
    orchestrator = ScannerOrchestrator(
        data_provider=data_provider,
        insider_provider=insider_provider
    )

    # Check scanner initialization
    print(f"\nInitialized scanners: {len(orchestrator.scanners)}")
    for scanner in orchestrator.scanners:
        print(f"  - {scanner.__class__.__name__} ({scanner.edge_type.value if scanner.edge_type else 'N/A'})")

    # Get active scanners
    print(f"\nActive scanners:")
    active = orchestrator.get_active_scanners()
    for name in active:
        print(f"  - {name}")

    # Run all scanners
    print("\n" + "-" * 40)
    print("Running all scanners...")
    print("-" * 40)

    opportunities = await orchestrator.run_all_scanners()

    print(f"\n{'=' * 40}")
    print(f"RESULTS: {len(opportunities)} opportunities found")
    print(f"{'=' * 40}")

    # Group by edge type (handle both enum and string values)
    by_edge = {}
    for opp in opportunities:
        if opp.primary_edge is None:
            edge = "unknown"
        elif isinstance(opp.primary_edge, str):
            edge = opp.primary_edge
        else:
            edge = opp.primary_edge.value

        if edge not in by_edge:
            by_edge[edge] = []
        by_edge[edge].append(opp)

    for edge, opps in sorted(by_edge.items()):
        print(f"\n{edge}: {len(opps)} opportunities")
        for opp in opps[:3]:  # Show first 3
            print(f"  {opp.symbol} {opp.direction} @ {opp.entry_price:.2f}")

    # Get status
    print(f"\n{'=' * 40}")
    print("SCANNER STATUS")
    print(f"{'=' * 40}")

    status = orchestrator.get_scanner_status()
    print(f"Total scans: {status['total_scans']}")
    print(f"Last scan: {status['last_scan']}")

    for name, info in status['scanners'].items():
        stats = info.get('stats', {})
        print(f"  {name}: {stats.get('opportunities', 0)} opps, {stats.get('errors', 0)} errors")

if __name__ == "__main__":
    asyncio.run(test())
