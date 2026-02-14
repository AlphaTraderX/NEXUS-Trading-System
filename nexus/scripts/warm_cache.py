"""
Cache Warming Script

Downloads historical data for instruments from the InstrumentRegistry,
preparing the data/cache/ directory for registry-backed backtests.

Supports all four providers:
- Polygon (US stocks) — via BacktestDataLoader
- OANDA (forex) — via BacktestDataLoader
- Binance (crypto) — direct via BinanceProvider
- IG Markets (UK/EU/Asia stocks, indices, commodities) — direct via IGProvider

Usage::

    # Warm all Polygon US stocks (top 30) for gap_fill edge
    python -m nexus.scripts.warm_cache --edge gap_fill

    # Warm by provider (all instruments for that provider)
    python -m nexus.scripts.warm_cache --provider polygon --max 50
    python -m nexus.scripts.warm_cache --provider oanda
    python -m nexus.scripts.warm_cache --provider binance
    python -m nexus.scripts.warm_cache --provider ig --delay 1.0

    # Warm specific symbols
    python -m nexus.scripts.warm_cache --symbols SPY,QQQ,AAPL

    # Custom date range (--days converts to start/end relative to today)
    python -m nexus.scripts.warm_cache --provider polygon --days 730 --delay 0.5

    # Print cache stats
    python -m nexus.scripts.warm_cache --stats
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from nexus.backtest.data_loader import BacktestDataLoader
from nexus.data.instruments import (
    DataProvider,
    InstrumentType,
    get_instrument_registry,
)
from nexus.data.cache import get_cache_manager
from nexus.core.enums import EdgeType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

# Edge -> (provider, instrument_types, default_max_instruments)
EDGE_DATA_MAP: Dict[EdgeType, tuple] = {
    # Stock edges — Polygon
    EdgeType.GAP_FILL: (DataProvider.POLYGON, [InstrumentType.STOCK], 30),
    EdgeType.OVERNIGHT_PREMIUM: (DataProvider.POLYGON, [InstrumentType.STOCK], 30),
    EdgeType.INSIDER_CLUSTER: (DataProvider.POLYGON, [InstrumentType.STOCK], 20),
    EdgeType.VWAP_DEVIATION: (DataProvider.POLYGON, [InstrumentType.STOCK], 20),
    EdgeType.RSI_EXTREME: (DataProvider.POLYGON, [InstrumentType.STOCK], 10),
    EdgeType.TURN_OF_MONTH: (DataProvider.POLYGON, [InstrumentType.STOCK], 10),
    EdgeType.MONTH_END: (DataProvider.POLYGON, [InstrumentType.STOCK], 10),
    EdgeType.POWER_HOUR: (DataProvider.POLYGON, [InstrumentType.STOCK], 20),
    EdgeType.ORB: (DataProvider.POLYGON, [InstrumentType.STOCK], 10),
    EdgeType.BOLLINGER_TOUCH: (DataProvider.POLYGON, [InstrumentType.STOCK], 10),
    # Forex edges — OANDA
    EdgeType.LONDON_OPEN: (DataProvider.OANDA, [InstrumentType.FOREX], 10),
    EdgeType.NY_OPEN: (DataProvider.OANDA, [InstrumentType.FOREX], 10),
    EdgeType.ASIAN_RANGE: (DataProvider.OANDA, [InstrumentType.FOREX], 10),
}

# Registry forex symbols use underscores (EUR_USD) but BacktestDataLoader
# routes via "/" in symbol (EUR/USD).  Map registry -> loader format.
_FOREX_REGISTRY_TO_LOADER = {
    "EUR_USD": "EUR/USD",
    "GBP_USD": "GBP/USD",
    "USD_JPY": "USD/JPY",
    "AUD_USD": "AUD/USD",
    "USD_CAD": "USD/CAD",
    "EUR_GBP": "EUR/GBP",
    "NZD_USD": "NZD/USD",
    "USD_CHF": "USD/CHF",
    "EUR_JPY": "EUR/JPY",
    "GBP_JPY": "GBP/JPY",
    "AUD_NZD": "AUD/NZD",
    "EUR_CHF": "EUR/CHF",
    "GBP_CHF": "GBP/CHF",
    "CAD_JPY": "CAD/JPY",
    "AUD_JPY": "AUD/JPY",
    "NZD_JPY": "NZD/JPY",
    "GBP_AUD": "GBP/AUD",
    "EUR_AUD": "EUR/AUD",
    "EUR_CAD": "EUR/CAD",
    "AUD_CAD": "AUD/CAD",
    "GBP_CAD": "GBP/CAD",
    "GBP_NZD": "GBP/NZD",
    "EUR_NZD": "EUR/NZD",
    "CHF_JPY": "CHF/JPY",
    "NZD_CAD": "NZD/CAD",
    "NZD_CHF": "NZD/CHF",
    "AUD_CHF": "AUD/CHF",
    "USD_SGD": "USD/SGD",
}


def _to_loader_symbol(registry_symbol: str) -> str:
    """Convert registry symbol to BacktestDataLoader-compatible format."""
    return _FOREX_REGISTRY_TO_LOADER.get(registry_symbol, registry_symbol)


def get_symbols_for_edge(
    edge_name: str, max_instruments: Optional[int] = None
) -> List[str]:
    """Get symbols from registry for a given edge type."""
    try:
        edge_type = EdgeType(edge_name)
    except ValueError:
        logger.error(f"Unknown edge: {edge_name}")
        return []

    if edge_type not in EDGE_DATA_MAP:
        logger.error(f"No data mapping for edge: {edge_name}")
        return []

    provider, inst_types, default_max = EDGE_DATA_MAP[edge_type]
    max_inst = max_instruments or default_max

    registry = get_instrument_registry()
    symbols = []
    for inst in registry.get_by_provider(provider):
        if inst.instrument_type in inst_types:
            symbols.append(inst.symbol)

    return symbols[:max_inst]


def get_symbols_for_provider(
    provider_name: str, max_instruments: Optional[int] = None
) -> List[str]:
    """Get symbols from registry for a given provider."""
    try:
        provider = DataProvider(provider_name)
    except ValueError:
        logger.error(f"Unknown provider: {provider_name}")
        return []

    registry = get_instrument_registry()
    symbols = [inst.symbol for inst in registry.get_by_provider(provider)]

    if max_instruments:
        symbols = symbols[:max_instruments]

    return symbols


# -----------------------------------------------------------------------
# Crypto fetching (Binance) — not in BacktestDataLoader, so we fetch
# directly via BinanceProvider.get_bars() and cache as CSV ourselves.
# -----------------------------------------------------------------------

async def _fetch_binance_bars(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch crypto bars from Binance and return DataFrame.

    Handles pagination (1000-bar limit per request).
    """
    from nexus.data.crypto import BinanceProvider

    provider = BinanceProvider()
    connected = await provider.connect()
    if not connected:
        logger.error("Cannot connect to Binance")
        return pd.DataFrame()

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc,
        )

        all_dfs: list = []
        current_end = end_dt

        while current_end > start_dt:
            df = await provider.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                end_date=current_end,
            )
            if df is None or df.empty:
                break

            # BinanceProvider returns 'timestamp' as a column, not the index.
            # Normalize: set timestamp as index if it's a column.
            if "timestamp" in df.columns and df.index.name != "timestamp":
                df = df.set_index("timestamp")
                df = df.sort_index()

            all_dfs.append(df)

            # Move window back
            earliest = df.index.min()
            if earliest <= start_dt:
                break
            current_end = earliest - timedelta(seconds=1)
            await asyncio.sleep(delay)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        # Trim to requested range
        combined = combined[combined.index >= start_dt]
        combined = combined[combined.index <= end_dt]

        return combined

    finally:
        await provider.disconnect()


# -----------------------------------------------------------------------
# IG Markets fetching — not in BacktestDataLoader, so we fetch directly
# via IGProvider.get_bars() and cache as CSV ourselves.
# -----------------------------------------------------------------------

async def _fetch_ig_bars(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    delay: float = 1.0,
    ig_provider=None,
) -> pd.DataFrame:
    """Fetch historical bars from IG Markets and return DataFrame.

    IG's prices endpoint returns max ~500 bars per request.
    We paginate backwards from end_date to start_date.

    The ig_provider parameter allows reusing an authenticated session
    across multiple symbols (avoids re-auth per symbol).
    """
    from nexus.data.ig import IGProvider

    own_provider = ig_provider is None
    if own_provider:
        ig_provider = IGProvider()
        connected = await ig_provider.connect()
        if not connected:
            logger.error("Cannot connect to IG Markets")
            return pd.DataFrame()

    try:
        # Use the provider's get_bars (returns DataFrame with OHLCV)
        # IG uses epic codes; the provider converts symbol -> epic
        df = await ig_provider.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            limit=500,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize: set timestamp as index if it's a column
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            df = df.sort_index()

        return df

    except Exception as e:
        logger.error(f"IG fetch error for {symbol}: {e}")
        return pd.DataFrame()

    finally:
        if own_provider:
            await ig_provider.disconnect()


def _cache_path_for(symbol: str, timeframe: str, start: str, end: str) -> Path:
    """Generate cache file path matching BacktestDataLoader convention."""
    clean = symbol.replace("/", "_").replace(":", "_")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{clean}_{timeframe}_{start}_{end}.csv"


# -----------------------------------------------------------------------
# Main warm logic
# -----------------------------------------------------------------------

async def warm_symbols(
    symbols: List[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    delay: float = 0.5,
) -> dict:
    """Download and cache data for given symbols.

    Routes to the correct provider based on registry lookup:
    - Crypto (Binance) → BinanceProvider
    - IG (UK/EU/Asia stocks, indices, commodities) → IGProvider
    - Forex (EUR_USD / EUR/USD) → BacktestDataLoader (OANDA)
    - US stocks → BacktestDataLoader (Polygon)

    Returns dict with 'cached', 'skipped', 'failed' lists.
    """
    loader = BacktestDataLoader()
    cache_mgr = get_cache_manager()

    # Identify provider-specific symbols from registry
    registry = get_instrument_registry()
    crypto_symbols = {
        inst.symbol
        for inst in registry.get_by_provider(DataProvider.BINANCE)
    }
    ig_symbols = {
        inst.symbol
        for inst in registry.get_by_provider(DataProvider.IG)
    }

    # For IG symbols, authenticate once and reuse the session
    ig_provider = None
    ig_needed = any(s in ig_symbols for s in symbols)
    if ig_needed:
        from nexus.data.ig import IGProvider
        ig_provider = IGProvider()
        connected = await ig_provider.connect()
        if not connected:
            logger.warning("IG Markets auth failed — IG symbols will be skipped")
            ig_provider = None

    results = {"cached": [], "skipped": [], "failed": []}
    total = len(symbols)

    try:
        for i, symbol in enumerate(symbols, 1):
            # Check if already cached (use registry symbol for cache lookup)
            if cache_mgr.has_data(symbol, timeframe, start_date, end_date):
                print(f"  [{i}/{total}] {symbol:12} CACHED (skip)")
                results["skipped"].append(symbol)
                continue

            print(f"  [{i}/{total}] {symbol:12} fetching...", end=" ", flush=True)

            try:
                if symbol in crypto_symbols:
                    # Crypto: fetch via Binance directly
                    df = await _fetch_binance_bars(
                        symbol, timeframe, start_date, end_date, delay,
                    )
                    # Save to cache ourselves
                    if not df.empty:
                        path = _cache_path_for(symbol, timeframe, start_date, end_date)
                        df.to_csv(path)

                elif symbol in ig_symbols:
                    # IG: fetch via IGProvider (UK/EU/Asia stocks, indices, commodities)
                    if ig_provider is None:
                        print("SKIP (IG not connected)")
                        results["failed"].append(symbol)
                        continue
                    df = await _fetch_ig_bars(
                        symbol, timeframe, start_date, end_date, delay,
                        ig_provider=ig_provider,
                    )
                    if not df.empty:
                        path = _cache_path_for(symbol, timeframe, start_date, end_date)
                        df.to_csv(path)

                else:
                    # Forex or US stocks: BacktestDataLoader (OANDA / Polygon)
                    loader_sym = _to_loader_symbol(symbol)
                    df = await loader.load_bars(
                        symbol=loader_sym,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )

                if df.empty:
                    print("NO DATA")
                    results["failed"].append(symbol)
                else:
                    print(f"OK ({len(df)} bars)")
                    results["cached"].append(symbol)
            except Exception as e:
                print(f"ERROR: {e}")
                results["failed"].append(symbol)

            # Courtesy delay between requests
            if delay > 0 and i < total:
                await asyncio.sleep(delay)

    finally:
        if ig_provider is not None:
            await ig_provider.disconnect()

    return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Warm data cache for registry instruments"
    )
    parser.add_argument(
        "--edge", type=str, default=None,
        help="Edge name to warm (e.g., gap_fill, overnight_premium)",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Provider to warm (polygon, oanda, binance, ig)",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (e.g., SPY,QQQ,AAPL)",
    )
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (default: 1d)")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--days", type=int, default=None,
        help="Lookback days from today (overrides --start/--end, e.g. 730 for 2 years)",
    )
    parser.add_argument("--max", type=int, default=None, help="Max instruments to warm")
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API requests (default: 0.5)",
    )
    parser.add_argument("--stats", action="store_true", help="Print cache stats and exit")
    return parser.parse_args()


async def run() -> None:
    args = _parse_args()

    if args.stats:
        cache_mgr = get_cache_manager()
        cache_mgr.print_stats()
        return

    # --days overrides --start/--end
    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days)
        args.start = str(start)
        args.end = str(end)

    # Determine symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.edge:
        symbols = get_symbols_for_edge(args.edge, args.max)
    elif args.provider:
        symbols = get_symbols_for_provider(args.provider, args.max)
    else:
        print("Specify --edge, --provider, or --symbols")
        sys.exit(1)

    if not symbols:
        print("No symbols to warm")
        sys.exit(1)

    print(f"\nWarming cache: {len(symbols)} symbols")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Period:    {args.start} to {args.end}")
    print(f"  Delay:     {args.delay}s between requests\n")

    results = await warm_symbols(
        symbols, args.timeframe, args.start, args.end, args.delay,
    )

    print(f"\nResults:")
    print(f"  Cached:  {len(results['cached'])}")
    print(f"  Skipped: {len(results['skipped'])} (already cached)")
    print(f"  Failed:  {len(results['failed'])}")

    if results["failed"]:
        print(f"  Failed symbols: {', '.join(results['failed'])}")

    # Print cache stats
    cache_mgr = get_cache_manager()
    cache_mgr.print_stats()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
