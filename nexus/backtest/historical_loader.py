"""
Historical Data Loader for Multi-Asset Backtesting

Loads historical OHLCV data from Polygon.io for backtesting.
Caches data locally to avoid repeated API calls.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from nexus.core.enums import Timeframe
from nexus.data.instruments import get_instrument_registry, InstrumentType

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    Load and cache historical market data for backtesting.

    Uses Polygon.io for US stocks, generates synthetic data for others
    (in production, would use appropriate providers for each market).
    """

    CACHE_DIR = Path("data/historical")

    # Timeframe to Polygon multiplier/timespan
    POLYGON_TIMEFRAMES = {
        Timeframe.M5: (5, "minute"),
        Timeframe.M15: (15, "minute"),
        Timeframe.M30: (30, "minute"),
        Timeframe.H1: (1, "hour"),
        Timeframe.H4: (4, "hour"),
        Timeframe.D1: (1, "day"),
    }

    def __init__(self, polygon_api_key: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            polygon_api_key: Polygon.io API key (optional, uses env var if not provided)
        """
        self.api_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "")
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache: Dict[str, pd.DataFrame] = {}
        self.registry = get_instrument_registry()

    def get_cache_path(self, symbol: str, timeframe: Timeframe) -> Path:
        """Get cache file path for symbol/timeframe."""
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe.value}.parquet"

    async def load_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol.

        Args:
            symbol: Instrument symbol
            timeframe: Data timeframe
            start_date: Start of data range
            end_date: End of data range
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{timeframe.value}"

        # Check memory cache
        if use_cache and cache_key in self._data_cache:
            df = self._data_cache[cache_key]
            return self._filter_date_range(df, start_date, end_date)

        # Check file cache
        cache_path = self.get_cache_path(symbol, timeframe)
        if use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self._data_cache[cache_key] = df
                return self._filter_date_range(df, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")

        # Fetch from API or generate synthetic
        instrument = self.registry.get(symbol)

        if instrument and instrument.provider.value == "polygon":
            df = await self._fetch_polygon(symbol, timeframe, start_date, end_date)
        else:
            # Generate synthetic data for non-Polygon instruments
            df = self._generate_synthetic_data(symbol, timeframe, start_date, end_date)

        if df is not None and not df.empty:
            # Cache the data
            self._data_cache[cache_key] = df
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache data for {symbol}: {e}")

        return df

    async def _fetch_polygon(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io API."""
        if not self.api_key:
            logger.warning("No Polygon API key, generating synthetic data")
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)

        import httpx

        multiplier, timespan = self.POLYGON_TIMEFRAMES[timeframe]

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/{start_str}/{end_str}"
        )

        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)

                if response.status_code != 200:
                    logger.error(f"Polygon API error: {response.status_code}")
                    return None

                data = response.json()

                if data.get("resultsCount", 0) == 0:
                    logger.warning(f"No data from Polygon for {symbol}")
                    return None

                results = data.get("results", [])

                df = pd.DataFrame(results)
                df = df.rename(columns={
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                })

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Failed to fetch Polygon data for {symbol}: {e}")
            return None

    def _generate_synthetic_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for backtesting.

        Uses random walk with realistic characteristics.
        """
        # Calculate number of bars
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        num_bars = total_minutes // timeframe.minutes

        if num_bars <= 0:
            return pd.DataFrame()

        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            periods=num_bars,
            freq=f"{timeframe.minutes}min",
            tz=timezone.utc,
        )

        # Set seed based on symbol for reproducibility
        seed = sum(ord(c) for c in symbol) % 10000
        rng = np.random.RandomState(seed)

        # Determine base price based on instrument type
        instrument = self.registry.get(symbol)
        if instrument:
            if instrument.instrument_type == InstrumentType.CRYPTO:
                base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            elif instrument.instrument_type == InstrumentType.FOREX:
                base_price = 1.1 if "EUR" in symbol else 1.3 if "GBP" in symbol else 110
            else:
                base_price = 150  # Stocks
        else:
            base_price = 100

        # Generate random walk
        returns = rng.normal(0.0001, 0.02, num_bars)  # Small drift, 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        volatility = 0.005  # Intrabar volatility
        opens = prices * (1 + rng.uniform(-volatility, volatility, num_bars))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, volatility, num_bars)))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, volatility, num_bars)))
        closes = prices

        # Generate volume
        base_volume = 1_000_000
        volumes = rng.lognormal(np.log(base_volume), 0.5, num_bars).astype(int)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

        return df

    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Filter dataframe to date range."""
        if df.empty:
            return df

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        return df[mask].copy()

    async def load_multiple(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        data = {}

        for symbol in symbols:
            df = await self.load_data(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                data[symbol] = df

        return data

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            # Clear specific symbol
            for key in list(self._data_cache.keys()):
                if key.startswith(symbol):
                    del self._data_cache[key]

            for path in self.cache_dir.glob(f"{symbol.replace('/', '_')}*.parquet"):
                path.unlink()
        else:
            # Clear all
            self._data_cache.clear()
            for path in self.cache_dir.glob("*.parquet"):
                path.unlink()
