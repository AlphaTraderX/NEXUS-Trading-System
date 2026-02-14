"""
Historical Data Cache Manager

Wraps the existing data/cache/ CSV storage used by BacktestDataLoader.
Provides cache inspection, stats, and integration with InstrumentRegistry
so the backtest engine can discover which instruments have cached data.

Cache structure (matches BacktestDataLoader):
    data/cache/
        SPY_1d_2022-01-01_2024-12-31.csv
        EUR_USD_1h_2022-01-01_2024-12-31.csv
        ...
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from nexus.data.instruments import (
    DataProvider,
    Instrument,
    InstrumentRegistry,
    InstrumentType,
    get_instrument_registry,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


class DataCacheManager:
    """
    Read-only inspector for the existing CSV cache produced by BacktestDataLoader.

    Does NOT fetch data — use BacktestDataLoader or warm_cache.py for that.
    This class only answers: "what do we already have?"
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._registry = get_instrument_registry()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_cached_files(self) -> List[Path]:
        """Return all cached CSV files."""
        return sorted(self.cache_dir.glob("*.csv"))

    def get_cached_symbols(
        self,
        timeframe: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Set[str]:
        """
        Return set of symbols that have cached data.

        Cache filename pattern: {symbol}_{timeframe}_{start}_{end}.csv
        If timeframe/dates are given, only match files for that exact config.
        """
        symbols: Set[str] = set()
        for f in self.list_cached_files():
            parts = self._parse_filename(f.stem)
            if parts is None:
                continue
            sym, tf, sd, ed = parts

            if timeframe and tf != timeframe:
                continue
            if start_date and sd != start_date:
                continue
            if end_date and ed != end_date:
                continue

            symbols.add(sym)
        return symbols

    def has_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Check if a specific symbol/timeframe/date-range is cached."""
        clean = symbol.replace("/", "_").replace(":", "_")
        path = self.cache_dir / f"{clean}_{timeframe}_{start_date}_{end_date}.csv"
        return path.exists()

    def get_cached_instruments(
        self,
        timeframe: str = "1d",
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
    ) -> List[Instrument]:
        """
        Return Instrument objects from the registry that have cached data.
        """
        cached_syms = self.get_cached_symbols(timeframe, start_date, end_date)
        result: List[Instrument] = []

        # Build reverse lookup: clean_symbol -> registry symbol
        # BacktestDataLoader replaces "/" with "_" in filenames
        for inst in self._registry.get_all():
            clean = inst.symbol.replace("/", "_").replace(":", "_")
            if clean in cached_syms or inst.symbol in cached_syms:
                result.append(inst)

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Cache statistics."""
        files = self.list_cached_files()
        total_size = sum(f.stat().st_size for f in files)

        symbols: Set[str] = set()
        timeframes: Set[str] = set()
        by_provider: Dict[str, int] = {}

        for f in files:
            parts = self._parse_filename(f.stem)
            if parts is None:
                continue
            sym, tf, _, _ = parts
            symbols.add(sym)
            timeframes.add(tf)

            # Try to map to provider via registry
            inst = self._registry.get(sym)
            if inst is None:
                # Try un-cleaning: EUR_USD -> EUR/USD
                inst = self._registry.get(sym.replace("_", "/"))
            if inst:
                prov = inst.provider.value
            else:
                prov = "unknown"
            by_provider[prov] = by_provider.get(prov, 0) + 1

        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "unique_symbols": len(symbols),
            "timeframes": sorted(timeframes),
            "by_provider": by_provider,
            "cache_dir": str(self.cache_dir),
        }

    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.get_stats()
        print(f"\nData Cache: {stats['cache_dir']}")
        print(f"  Files:     {stats['total_files']}")
        print(f"  Size:      {stats['total_size_mb']:.1f} MB")
        print(f"  Symbols:   {stats['unique_symbols']}")
        print(f"  Timeframes: {', '.join(stats['timeframes']) or 'none'}")
        if stats["by_provider"]:
            print(f"  By provider:")
            for prov, count in sorted(stats["by_provider"].items()):
                print(f"    {prov}: {count} files")
        print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_filename(stem: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse cache filename stem into (symbol, timeframe, start, end).

        Pattern: {symbol}_{timeframe}_{YYYY-MM-DD}_{YYYY-MM-DD}
        Symbol may contain underscores (e.g. EUR_USD), so we parse
        from the right where the date pattern is unambiguous.
        """
        # Dates are always YYYY-MM-DD (10 chars). Work backwards.
        parts = stem.split("_")
        if len(parts) < 4:
            return None

        # Last 6 parts should be: ..., tf, YYYY, MM, DD, YYYY, MM, DD
        # But dates use hyphens inside: 2022-01-01 stays as one token
        # Actually, the filename is: SPY_1d_2022-01-01_2024-12-31
        # So splitting on "_" gives: [SPY, 1d, 2022-01-01, 2024-12-31]
        # For EUR_USD_1d_2022-01-01_2024-12-31: [EUR, USD, 1d, 2022-01-01, 2024-12-31]

        # End date is always last, start date second to last
        end_date = parts[-1]
        start_date = parts[-2]

        # Validate date patterns
        if len(end_date) != 10 or len(start_date) != 10:
            return None
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return None

        # Timeframe is the part before start_date
        timeframe = parts[-3]

        # Symbol is everything before the timeframe
        symbol = "_".join(parts[:-3])

        return symbol, timeframe, start_date, end_date


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_cache_mgr: Optional[DataCacheManager] = None


def get_cache_manager() -> DataCacheManager:
    """Get singleton cache manager."""
    global _cache_mgr
    if _cache_mgr is None:
        _cache_mgr = DataCacheManager()
    return _cache_mgr
