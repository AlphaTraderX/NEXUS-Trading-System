"""
Historical data loader with caching.

- Fetches from Polygon API
- Caches to local CSV to avoid API limits
- Provides consistent DataFrame format for all scanners
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

from nexus.config.settings import settings

logger = logging.getLogger(__name__)

# Forex symbols that need C: prefix on Polygon
_FOREX_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDNZD", "USDCHF",
}

# OANDA forex data configuration
_OANDA_INSTRUMENTS = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "USD/JPY": "USD_JPY",
    "AUD/USD": "AUD_USD",
    "USD/CAD": "USD_CAD",
    "EUR/GBP": "EUR_GBP",
    "NZD/USD": "NZD_USD",
    "USD/CHF": "USD_CHF",
}

_OANDA_GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"


class BacktestDataLoader:
    """
    Load historical data for backtesting.

    Caches data locally to avoid hitting API limits.
    Polygon free tier = 5 calls/minute, so caching is essential.
    """

    CACHE_DIR = Path("data/cache")

    # Polygon timeframe mapping: nexus_tf -> (timespan, multiplier)
    _TF_MAP = {
        "1m": ("minute", 1),
        "5m": ("minute", 5),
        "15m": ("minute", 15),
        "30m": ("minute", 30),
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.polygon_key = settings.polygon_api_key
        self.oanda_api_key = settings.oanda_api_key

    def get_cache_path(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> Path:
        """Generate cache file path."""
        clean_symbol = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{clean_symbol}_{timeframe}_{start}_{end}.csv"

    async def load_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV bars for backtesting.

        Args:
            symbol: Ticker symbol (e.g. "SPY", "EUR/USD")
            timeframe: Bar size – "1m", "5m", "15m", "30m", "1h", "4h", "1d"
            start_date: ISO date string, inclusive
            end_date: ISO date string, inclusive
            use_cache: If True, return cached CSV when available

        Returns:
            DataFrame indexed by ``timestamp`` with columns:
            open, high, low, close, volume
        """
        # Route forex symbols to OANDA
        if "/" in symbol and symbol in _OANDA_INSTRUMENTS:
            return await self.load_forex_bars(
                symbol, timeframe, start_date, end_date, use_cache
            )

        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)

        # Try cache first
        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {cache_path.name}")
            df = pd.read_csv(
                cache_path, parse_dates=["timestamp"], index_col="timestamp"
            )
            return df

        # Fetch from Polygon
        if not self.polygon_key:
            logger.warning("No Polygon API key configured – cannot fetch data")
            return pd.DataFrame()

        # Intraday timeframes: fetch in 3-month chunks (Polygon free tier limit)
        if timeframe in ("1m", "5m", "15m", "30m"):
            df = await self._fetch_polygon_chunked(
                symbol, timeframe, start_date, end_date
            )
        else:
            df = await self._fetch_polygon(symbol, timeframe, start_date, end_date)

        # Cache for next time
        if not df.empty:
            df.to_csv(cache_path)
            logger.info(
                f"Cached {len(df)} bars for {symbol} {timeframe} -> {cache_path.name}"
            )

        return df

    async def _fetch_polygon(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch from Polygon.io API.

        Handles pagination for large date ranges and respects the
        free-tier rate limit (5 req/min) with a 12-second back-off on 429.
        """
        if timeframe not in self._TF_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Choose from: {', '.join(self._TF_MAP)}"
            )

        timespan, multiplier = self._TF_MAP[timeframe]
        ticker = self._polygon_ticker(symbol)

        base_url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
            f"/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        )

        all_results: list = []
        url: Optional[str] = base_url
        is_first_request = True

        async with httpx.AsyncClient(timeout=30.0) as client:
            while url:
                # First request uses base_url + full params;
                # paginated requests use next_url which already contains params.
                if is_first_request:
                    params = {
                        "apiKey": self.polygon_key,
                        "limit": 50000,
                        "sort": "asc",
                    }
                    resp = await client.get(url, params=params)
                    is_first_request = False
                else:
                    # Paginated: next_url may not include apiKey, so add it.
                    sep = "&" if "?" in url else "?"
                    paginated_url = f"{url}{sep}apiKey={self.polygon_key}"
                    resp = await client.get(paginated_url)

                if resp.status_code == 429:
                    logger.info("Polygon rate limit hit – waiting 12s")
                    await asyncio.sleep(12)
                    is_first_request = url == base_url
                    continue

                if resp.status_code != 200:
                    logger.error(
                        f"Polygon API error {resp.status_code}: {resp.text[:200]}"
                    )
                    break

                data = resp.json()

                results = data.get("results")
                if results:
                    all_results.extend(results)

                # Polygon returns next_url with apiKey already embedded; use as-is.
                url = data.get("next_url")

                # Rate-limit courtesy pause between pages
                if url:
                    await asyncio.sleep(0.5)

        if not all_results:
            logger.warning(f"No data returned for {symbol} ({start_date} – {end_date})")
            return pd.DataFrame()

        # Convert to standard DataFrame
        df = pd.DataFrame(all_results)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.set_index("timestamp")

        return df

    async def _fetch_polygon_chunked(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch intraday data in 3-month chunks to work around Polygon free tier limits."""
        from datetime import date
        from dateutil.relativedelta import relativedelta

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        chunks: list[pd.DataFrame] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + relativedelta(months=3) - timedelta(days=1), end)
            logger.info(
                f"Fetching {symbol} {timeframe} chunk: {chunk_start} to {chunk_end}"
            )
            chunk = await self._fetch_polygon(
                symbol, timeframe, str(chunk_start), str(chunk_end)
            )
            if not chunk.empty:
                chunks.append(chunk)
            chunk_start = chunk_end + timedelta(days=1)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df

    @staticmethod
    def _polygon_ticker(symbol: str) -> str:
        """Convert NEXUS symbol to Polygon ticker format."""
        clean = symbol.replace("/", "")
        if "/" in symbol or clean.upper() in _FOREX_SYMBOLS:
            return f"C:{clean.upper()}"
        return symbol.upper()

    def load_bars_sync(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Synchronous wrapper for load_bars."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. Jupyter) – run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.load_bars(symbol, timeframe, start_date, end_date, use_cache),
                ).result()

        return asyncio.run(
            self.load_bars(symbol, timeframe, start_date, end_date, use_cache)
        )

    def get_available_cache(self) -> list:
        """List all cached data files."""
        return sorted(self.cache_dir.glob("*.csv"))

    def clear_cache(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        files = list(self.cache_dir.glob("*.csv"))
        for f in files:
            f.unlink()
        return len(files)

    # ------------------------------------------------------------------
    # OANDA forex data loading
    # ------------------------------------------------------------------

    async def load_forex_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load forex OHLCV data from OANDA.

        Args:
            symbol: Forex pair (e.g., "EUR/USD")
            timeframe: Bar size ("1m","5m","15m","30m","1h","4h","1d")
            start_date: ISO date string, inclusive
            end_date: ISO date string, inclusive
            use_cache: Return cached CSV when available

        Returns:
            DataFrame with open, high, low, close, volume columns;
            DatetimeIndex named ``timestamp``.
        """
        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)

        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit (OANDA): {cache_path.name}")
            return pd.read_csv(
                cache_path, parse_dates=["timestamp"], index_col="timestamp"
            )

        if not self.oanda_api_key:
            logger.warning("No OANDA API key — cannot fetch forex data")
            return pd.DataFrame()

        instrument = _OANDA_INSTRUMENTS.get(symbol)
        if not instrument:
            logger.warning(f"Unknown forex pair: {symbol}")
            return pd.DataFrame()

        granularity = _OANDA_GRANULARITY.get(timeframe)
        if not granularity:
            logger.warning(f"Unsupported OANDA timeframe: {timeframe}")
            return pd.DataFrame()

        # Paginate: OANDA limits to 5000 candles per request.
        # Use from + count (not from + to) to avoid 400 errors on large ranges.
        all_candles: list = []
        current_from = f"{start_date}T00:00:00Z"
        end_dt = pd.to_datetime(f"{end_date}T23:59:59Z", utc=True)

        while True:
            candles = await self._fetch_oanda_candles(
                instrument, granularity, current_from, count=5000
            )
            if not candles:
                break

            all_candles.extend(candles)

            # If fewer than 5000 returned, we have all the data
            if len(candles) < 5000:
                break

            # Advance past last candle
            last_time = candles[-1]["time"]
            last_dt = pd.to_datetime(last_time, utc=True)

            # Stop if we've passed the end date
            if last_dt >= end_dt:
                break

            current_from = (
                (last_dt + timedelta(seconds=1))
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            # Rate limit courtesy
            await asyncio.sleep(0.5)

        if not all_candles:
            logger.warning(f"No OANDA data for {symbol} {timeframe}")
            return pd.DataFrame()

        df = self._oanda_candles_to_dataframe(all_candles)

        # Trim to requested date range
        if not df.empty:
            df = df[df.index <= end_dt]

        if not df.empty:
            df.to_csv(cache_path)
            logger.info(
                f"Cached {len(df)} OANDA bars for {symbol} {timeframe} "
                f"-> {cache_path.name}"
            )

        return df

    async def _fetch_oanda_candles(
        self,
        instrument: str,
        granularity: str,
        from_time: str,
        count: int = 5000,
    ) -> list:
        """Fetch candles from OANDA v3 API.

        Uses from + count (not from + to) to avoid 400 errors
        when the date range contains more than 5000 candles.
        """
        url = f"{OANDA_API_URL}/instruments/{instrument}/candles"
        headers = {
            "Authorization": f"Bearer {self.oanda_api_key}",
            "Content-Type": "application/json",
        }
        params = {
            "granularity": granularity,
            "from": from_time,
            "count": count,
            "price": "M",  # Mid prices
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers, params=params)

                if resp.status_code == 429:
                    logger.info("OANDA rate limit hit — waiting 5s")
                    await asyncio.sleep(5)
                    resp = await client.get(url, headers=headers, params=params)

                resp.raise_for_status()
                data = resp.json()

            return data.get("candles", [])
        except Exception as e:
            logger.error(f"OANDA fetch error for {instrument}: {e}")
            return []

    @staticmethod
    def _oanda_candles_to_dataframe(candles: list) -> pd.DataFrame:
        """Convert OANDA candles to standard DataFrame format."""
        records = []
        for candle in candles:
            if not candle.get("complete", True):
                continue
            mid = candle.get("mid", {})
            records.append({
                "timestamp": pd.to_datetime(candle["time"], utc=True),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        return df
