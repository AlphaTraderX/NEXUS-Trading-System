"""
Targeted IG Markets Cache Warming

Smart approach: 186 IG instruments map to only 27 unique epics.
Fetches one representative per epic, then copies for all instruments
sharing the same epic.

Usage:
    python -m nexus.scripts.warm_ig_cache
    python -m nexus.scripts.warm_ig_cache --dry-run   # Show plan without fetching
    python -m nexus.scripts.warm_ig_cache --delay 15   # 15s between requests
"""

import argparse
import asyncio
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from nexus.data.instruments import DataProvider, get_instrument_registry
from nexus.data.cache import get_cache_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
TIMEFRAME = "1d"
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# Spread betting epics (.DAILY.IP) → CFD epics (.IFD.IP / .IFM.IP)
# The registry uses spread betting epics but the demo CFD account (Z68G02)
# needs different epic codes for price data access.
CFD_EPIC_FALLBACK = {
    "IX.D.NASDAQ.DAILY.IP": "IX.D.NASDAQ.IFD.IP",
    "IX.D.ASX.DAILY.IP": "IX.D.ASX.IFD.IP",
    "IX.D.STXE.DAILY.IP": "IX.D.STXE.FN2A.IP",
    "IX.D.VIX.DAILY.IP": "IN.D.VIX.FWS1.IP",
    "IX.D.VDAX.DAILY.IP": None,  # No CFD equivalent found
    "IX.D.CHINAA.DAILY.IP": "IX.D.XINHUA.IFM.IP",
    "IX.D.NIFTY.DAILY.IP": None,  # No direct index (ETF only)
    "IX.D.HSI.DAILY.IP": None,  # Need to find correct epic
    "CS.D.USCPT.TODAY.IP": "MT.D.PL.FWM1.IP",
    "CS.D.USCPD.TODAY.IP": "MT.D.PA.FWS1.IP",
}


def _cache_path(symbol: str) -> Path:
    clean = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{clean}_{TIMEFRAME}_{START_DATE}_{END_DATE}.csv"


def build_epic_groups() -> dict:
    """Group all IG instruments by their epic code.

    Returns: {epic: [symbol1, symbol2, ...]}
    """
    registry = get_instrument_registry()
    groups = defaultdict(list)
    for inst in registry.get_by_provider(DataProvider.IG):
        epic = inst.provider_symbol or inst.symbol
        groups[epic].append(inst.symbol)
    return dict(groups)


def find_cached_epic(epic: str, symbols: list) -> str | None:
    """Check if any symbol for this epic is already cached. Return the path if so."""
    for sym in symbols:
        path = _cache_path(sym)
        if path.exists() and path.stat().st_size > 100:
            return str(path)
    return None


async def fetch_epic(epic: str, representative_symbol: str, ig_provider, delay: float) -> bool:
    """Fetch data for one epic using the representative symbol.

    If the spread betting epic returns 404, tries the CFD fallback epic.
    """
    try:
        df = await ig_provider.get_bars(
            symbol=representative_symbol,
            timeframe=TIMEFRAME,
            limit=500,
        )

        # If empty and we have a CFD fallback, try that directly
        if (df is None or df.empty) and epic in CFD_EPIC_FALLBACK:
            cfd_epic = CFD_EPIC_FALLBACK[epic]
            if cfd_epic is None:
                logger.warning(f"  No CFD equivalent for {epic} — skipped")
                return False
            logger.info(f"  Retrying with CFD epic: {cfd_epic}")
            # Call get_bars with the epic directly (bypass symbol conversion)
            import pandas as pd
            response = await ig_provider._client.get(
                f"{ig_provider._base_url}/prices/{cfd_epic}",
                headers={**ig_provider._auth_headers(), "Version": "3"},
                params={"resolution": "DAY", "max": 500, "pageSize": 500},
            )
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                if prices:
                    df = pd.DataFrame([{
                        'timestamp': p.get("snapshotTime"),
                        'open': float(p.get("openPrice", {}).get("bid", 0)),
                        'high': float(p.get("highPrice", {}).get("bid", 0)),
                        'low': float(p.get("lowPrice", {}).get("bid", 0)),
                        'close': float(p.get("closePrice", {}).get("bid", 0)),
                        'volume': int(p.get("lastTradedVolume", 0)),
                    } for p in prices])

        if df is None or df.empty:
            logger.warning(f"  No data for {epic} (via {representative_symbol})")
            return False

        # Normalize timestamp
        import pandas as pd
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            df = df.sort_index()

        # Save as representative symbol
        path = _cache_path(representative_symbol)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info(f"  Fetched {epic} -> {path.name} ({len(df)} bars)")

        if delay > 0:
            await asyncio.sleep(delay)

        return True

    except Exception as e:
        logger.error(f"  Error fetching {epic}: {e}")
        return False


def copy_for_group(source_path: str, symbols: list, source_symbol: str) -> int:
    """Copy the cached source file for all other symbols in the group."""
    copied = 0
    for sym in symbols:
        if sym == source_symbol:
            continue
        dest = _cache_path(sym)
        if dest.exists() and dest.stat().st_size > 100:
            continue  # Already cached
        shutil.copy2(source_path, dest)
        copied += 1
    return copied


async def run(delay: float = 10.0, dry_run: bool = False):
    epic_groups = build_epic_groups()

    print(f"\n{'='*60}")
    print(f"IG Cache Warming Plan")
    print(f"{'='*60}")
    print(f"Total IG instruments: {sum(len(v) for v in epic_groups.values())}")
    print(f"Unique epics: {len(epic_groups)}")
    print(f"Cache period: {START_DATE} to {END_DATE}")
    print(f"Delay between requests: {delay}s")
    print()

    # Categorize epics
    already_cached = {}   # epic -> source_path
    needs_fetch = {}      # epic -> (representative_symbol, all_symbols)

    for epic, symbols in sorted(epic_groups.items()):
        cached_path = find_cached_epic(epic, symbols)
        if cached_path:
            already_cached[epic] = (cached_path, symbols)
            # Find which symbol is cached
            for sym in symbols:
                if str(_cache_path(sym)) == cached_path:
                    source_sym = sym
                    break
            else:
                source_sym = symbols[0]
            uncached = [s for s in symbols if not _cache_path(s).exists()]
            if uncached:
                print(f"  CACHED  {epic:35} ({len(symbols)} symbols, {len(uncached)} need copy)")
            else:
                print(f"  CACHED  {epic:35} ({len(symbols)} symbols, all copied)")
        else:
            needs_fetch[epic] = (symbols[0], symbols)
            print(f"  FETCH   {epic:35} ({len(symbols)} symbols, via {symbols[0]})")

    print(f"\nSummary: {len(already_cached)} cached, {len(needs_fetch)} to fetch")

    if dry_run:
        print("\n[DRY RUN] Would fetch these epics:")
        for epic, (rep, syms) in needs_fetch.items():
            print(f"  {epic} via {rep} ({len(syms)} symbols)")
        return

    # Step 1: Copy data for already-cached epics that have uncached symbols
    total_copies = 0
    for epic, (source_path, symbols) in already_cached.items():
        # Find which symbol is the source
        source_sym = None
        for sym in symbols:
            if str(_cache_path(sym)) == source_path:
                source_sym = sym
                break
        if source_sym:
            copies = copy_for_group(source_path, symbols, source_sym)
            if copies:
                total_copies += copies
                print(f"  Copied {copies} files for {epic}")

    if total_copies:
        print(f"\nCopied {total_copies} files from existing cache")

    # Step 2: Fetch uncached epics via IG API
    if needs_fetch:
        print(f"\nFetching {len(needs_fetch)} epics from IG API...")

        from nexus.data.ig import IGProvider
        ig = IGProvider()
        # Skip account switch — use default CFD account for data access.
        # The spread betting (Z68G03) demo has limited market access.
        saved_desired = ig._desired_account_id
        ig._desired_account_id = None
        connected = await ig.connect()
        ig._desired_account_id = saved_desired
        if not connected:
            logger.error("Cannot connect to IG Markets")
            return

        try:
            fetched = 0
            failed = 0
            for i, (epic, (rep_symbol, symbols)) in enumerate(needs_fetch.items(), 1):
                print(f"  [{i}/{len(needs_fetch)}] {epic} via {rep_symbol}...", end=" ", flush=True)

                ok = await fetch_epic(epic, rep_symbol, ig, delay)
                if ok:
                    fetched += 1
                    # Copy for all other symbols in the group
                    source_path = str(_cache_path(rep_symbol))
                    copies = copy_for_group(source_path, symbols, rep_symbol)
                    print(f"  + copied to {copies} other symbols")
                else:
                    failed += 1
                    print()

            print(f"\nFetch results: {fetched} OK, {failed} failed")

        finally:
            await ig.disconnect()

    # Final stats
    cache_mgr = get_cache_manager()
    ig_total = sum(len(v) for v in epic_groups.values())
    ig_cached = sum(1 for syms in epic_groups.values() for s in syms if _cache_path(s).exists())
    print(f"\nIG cache coverage: {ig_cached}/{ig_total} instruments")


def main():
    parser = argparse.ArgumentParser(description="Smart IG cache warming")
    parser.add_argument("--delay", type=float, default=10.0, help="Seconds between API requests")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without fetching")
    args = parser.parse_args()
    asyncio.run(run(delay=args.delay, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
